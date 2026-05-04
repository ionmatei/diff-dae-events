"""
Adam optimization on the Cauer DAE WITH events — discrete-adjoint
gradients.

Mirrors `src/run/optimization_cauer_events.py` (which uses diffrax + AD)
but routes the gradient via:
  * `src/discrete_adjoint/dae_solver.py`'s `DAESolver.solve_augmented`
    (Sundials/IDA-based DAE solve with native event handling), and
  * `src/discrete_adjoint/dae_padded_gradient.py`'s `DAEPaddedGradient`
    (segmented adjoint with sigmoid-blended losses, JIT-compiled).

Same config file as the diffrax runner: `config/config_cauer_events.yaml`.
The discrete-adjoint runner reads the discrete-adjoint-specific fields
(`max_blocks`, `max_points_per_segment`, `max_targets`,
`downsample_segments`, `blend_sharpness`); the diffrax runner ignores
them and reads its own keys (`diffrax_solver`, `max_segments`, ...).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

import jax
import jax.numpy as jnp
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
if not os.environ.get("JAX_PLATFORMS"):
    jax_config.update("jax_platform_name", "cpu")

# Persistent JIT cache (mirrors `optimization_jax_bouncing_balls_N.py`).
_jax_cache_dir = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "jax_dae_optim"),
)
os.makedirs(_jax_cache_dir, exist_ok=True)
jax_config.update("jax_compilation_cache_dir", _jax_cache_dir)
jax_config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax_config.update("jax_persistent_cache_min_compile_time_secs", 0)

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from src.discrete_adjoint.dae_solver import DAESolver  # noqa: E402
from src.discrete_adjoint.dae_padded_gradient import DAEPaddedGradient  # noqa: E402


# ---------------------------------------------------------------------- #
# Config / spec loading
# ---------------------------------------------------------------------- #

def load_config(config_path: str, spec_override: str = None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    solver_cfg = cfg["dae_solver"]
    opt_cfg = cfg["optimizer"]

    spec_path = spec_override or solver_cfg["dae_specification_file"]
    if not os.path.isabs(spec_path):
        spec_path = os.path.join(root_dir, spec_path)
    with open(spec_path, "r") as f:
        if spec_path.endswith((".yaml", ".yml")):
            dae_data = yaml.safe_load(f)
        else:
            dae_data = json.load(f)
    return dae_data, solver_cfg, opt_cfg, spec_path


def bias_parameters(p_nominal, opt_indices, factor: float, seed: int = 0):
    """Log-uniform multiplicative bias: stays strictly positive for any
    factor. Mirrors `optimization_cauer_events.py:bias_parameters`."""
    rng = np.random.default_rng(seed)
    p_init = p_nominal.copy()
    log_perturb = factor * rng.uniform(-1.0, 1.0, size=len(opt_indices))
    for k, idx in enumerate(opt_indices):
        p_init[idx] = p_nominal[idx] * float(np.exp(log_perturb[k]))
    return p_init


def prepare_loss_targets(sol):
    """Stack interior target times / data from each segment, dropping
    the last point of each segment so targets sit strictly inside the
    segment interior. Mirrors the bouncing-balls discrete-adjoint
    runner.
    """
    all_t, all_x = [], []
    for seg in sol.segments:
        if len(seg.t) > 0:
            all_t.append(seg.t)
            all_x.append(seg.x)
    if not all_t:
        return jnp.array([]), jnp.array([])
    target_times = jnp.concatenate([jnp.array(t[:-1]) for t in all_t])
    target_data = jnp.concatenate([jnp.array(x[:-1]) for x in all_x])
    return target_times, target_data


# ---------------------------------------------------------------------- #
# Plot helpers
# ---------------------------------------------------------------------- #

def plot_results(history, t_array, y_target, y_init, y_opt,
                 param_names_opt, p_true, p_init_opt, p_opt,
                 out_dir: str, prefix: str,
                 state_names: list | None = None):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1, len(history["loss"]) + 1), history["loss"],
                "-o", markersize=3, color="tab:purple")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Cauer DAE (events, discrete adjoint) — Adam loss")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    p_loss = os.path.join(out_dir, f"{prefix}_loss.png")
    plt.savefig(p_loss, dpi=150); plt.close(fig)
    print(f"  Saved: {p_loss}")

    n = len(param_names_opt)
    x = np.arange(n); width = 0.27
    fig, ax = plt.subplots(figsize=(max(7, 1.0 * n), 4))
    ax.bar(x - width, p_true, width, label="true", color="tab:green")
    ax.bar(x, p_init_opt, width, label="initial", color="tab:orange")
    ax.bar(x + width, p_opt, width, label="optimized", color="tab:purple")
    ax.set_xticks(x); ax.set_xticklabels(param_names_opt, rotation=30, ha="right")
    ax.set_ylabel("Parameter value")
    ax.set_title("Cauer DAE (events, discrete adjoint) — parameter recovery")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p_par = os.path.join(out_dir, f"{prefix}_params.png")
    plt.savefig(p_par, dpi=150); plt.close(fig)
    print(f"  Saved: {p_par}")

    n_out = y_target.shape[1]
    fig, axes = plt.subplots(n_out, 1, figsize=(8, 1.6 * n_out + 0.6),
                             sharex=True)
    if n_out == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t_array, y_target[:, i], "-", color="tab:green",
                linewidth=1.4, label="true" if i == 0 else None)
        ax.plot(t_array, y_init[:, i], "--", color="tab:orange",
                linewidth=1.0, label="initial" if i == 0 else None)
        ax.plot(t_array, y_opt[:, i], ":", color="tab:purple",
                linewidth=1.4, label="optimized" if i == 0 else None)
        ylab = (state_names[i] if state_names is not None
                and i < len(state_names) else f"x[{i}]")
        ax.set_ylabel(ylab); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time")
    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    p_traj = os.path.join(out_dir, f"{prefix}_outputs.png")
    plt.savefig(p_traj, dpi=150); plt.close(fig)
    print(f"  Saved: {p_traj}")


# ---------------------------------------------------------------------- #
# Main pipeline
# ---------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Adam on event-aware Cauer DAE via the discrete-"
                    "adjoint DAEPaddedGradient.",
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(root_dir, "config", "config_cauer_events.yaml"),
        help="Path to the events YAML config (shared with the diffrax runner).",
    )
    parser.add_argument(
        "--spec", type=str, default=None,
        help="Path to the DAE spec (overrides the value in --config).",
    )
    parser.add_argument(
        "--max-blocks", type=int, default=None,
        help="Override `optimizer.max_blocks` in the YAML.",
    )
    parser.add_argument(
        "--max-points-per-segment", type=int, default=None,
        help="Override `optimizer.max_points_per_segment` in the YAML.",
    )
    parser.add_argument(
        "--max-targets", type=int, default=None,
        help="Override `optimizer.max_targets` in the YAML.",
    )
    parser.add_argument(
        "--blend-sharpness", type=float, default=None,
        help="Override `optimizer.blend_sharpness` in the YAML.",
    )
    parser.add_argument(
        "--bias-factor", type=float, default=0.1,
        help="Half-width of the log-uniform multiplicative perturbation "
             "(see optimization_cauer_events.py).",
    )
    parser.add_argument("--bias-seed", type=int, default=0)
    parser.add_argument(
        "--max-iter-override", type=int, default=None,
        help="Override `optimizer.max_iterations` in the YAML.",
    )
    parser.add_argument(
        "--results-dir", type=str,
        default=os.path.join(root_dir, "results"),
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--best-every", type=int, default=5,
        help="If > 0, scan the recorded `loss_history`/`p_history` every N "
             "iterations and return the best-loss snapshot at the end "
             "instead of the last iterate. Useful when Adam oscillates near "
             "the basin floor. 0 (default) disables and returns the last "
             "iterate as before. Set to 1 to compare every iter.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Cauer DAE WITH EVENTS — discrete-adjoint Adam optimization")
    print("=" * 70)

    dae_data, solver_cfg, opt_cfg, spec_path = load_config(args.config, args.spec)
    print(f"  spec: {spec_path}")
    print(f"  when clauses: {len(dae_data.get('when') or [])}")

    # --- 1. Solver / time grid ---
    t_start = float(solver_cfg["start_time"])
    t_stop = float(solver_cfg["stop_time"])
    ncp = int(solver_cfg["ncp"])
    t_span = (t_start, t_stop)

    # --- 2. Optimizer hyperparameters ---
    opt_param_names = list(opt_cfg["opt_params"])
    max_iter = int(args.max_iter_override or opt_cfg["max_iterations"])
    tol = float(opt_cfg["tol"])
    print_every = int(opt_cfg.get("print_every", 1))

    alg = opt_cfg["algorithm"]
    params = alg.get("params", {})
    step_size = float(params.get("step_size", 1.0e-3))
    beta1 = float(params.get("beta1", 0.9))
    beta2 = float(params.get("beta2", 0.999))
    epsilon = float(params.get("epsilon", 1e-8))

    # Discrete-adjoint padding budgets
    max_blocks = int(args.max_blocks or opt_cfg.get("max_blocks", 20))
    max_pts = int(args.max_points_per_segment or
                  opt_cfg.get("max_points_per_segment", 150))
    max_targets = int(args.max_targets or opt_cfg.get("max_targets", 1200))
    downsample_segments = bool(opt_cfg.get("downsample_segments", True))
    blend_sharpness = float(args.blend_sharpness or
                             opt_cfg.get("blend_sharpness", 300.0))

    # --- 3. Parameter indices ---
    param_names_all = [p["name"] for p in dae_data["parameters"]]
    opt_param_indices = [param_names_all.index(n) for n in opt_param_names]
    print(f"  opt_params: {opt_param_names} -> indices {opt_param_indices}")

    # --- 4. Ground truth via IDA-based DAESolver ---
    p_nominal = np.asarray(
        [p["value"] for p in dae_data["parameters"]], dtype=float
    )
    p_true_opt = np.asarray([p_nominal[i] for i in opt_param_indices])

    print("\nSimulating ground truth at nominal parameters...")
    t0 = time.time()
    solver = DAESolver(dae_data, verbose=False, use_compiled_residual=True)
    solver.update_parameters(p_nominal)
    sol_true = solver.solve_augmented(t_span, ncp=ncp)
    print(f"  done ({time.time() - t0:.2f}s)")
    print(f"  segments: {len(sol_true.segments)}, "
          f"events: {len(sol_true.events)}")

    target_times, target_data = prepare_loss_targets(sol_true)
    if target_times.size == 0:
        raise RuntimeError("Empty target trajectory — check the config.")
    delta_t = target_times[1:] - target_times[:-1]
    print(f"  target points: {len(target_times)}, "
          f"min Δt={float(jnp.min(delta_t)):.4e}")

    # --- 5. Biased initial parameters ---
    p_init_full = bias_parameters(
        p_nominal, opt_param_indices,
        factor=args.bias_factor, seed=args.bias_seed,
    )
    p_init_opt = np.asarray([p_init_full[i] for i in opt_param_indices])
    print("\nInitial vs. true:")
    for n, t_, i_ in zip(opt_param_names, p_true_opt, p_init_opt):
        rel = (i_ - t_) / t_ if t_ != 0 else float("nan")
        print(f"  {n:12s}  true={t_: .5f}   init={i_: .5f}   "
              f"rel_bias={rel:+.2%}")

    # Cache initial trajectory for the output plot.
    print("\nSimulating initial-guess trajectory...")
    solver.update_parameters(p_init_full)
    sol_init = solver.solve_augmented(t_span, ncp=ncp)
    t_init_full, x_init_full = _stack_segments(sol_init)
    solver.update_parameters(p_nominal)  # reset for safety
    t_true_full, x_true_full = _stack_segments(sol_true)

    # --- 6. Build padded-gradient kernel ---
    # `DAEPaddedGradient` does `len(dae_data['h'])` without a None check;
    # the Cauer spec has `"h": null` (no output map => identity), so
    # normalize to an empty list before constructing the kernel.
    if dae_data.get("h") is None:
        dae_data = {**dae_data, "h": []}
    grad_computer = DAEPaddedGradient(
        dae_data,
        max_blocks=max_blocks,
        max_pts=max_pts,
        max_targets=max_targets,
        downsample_segments=downsample_segments,
    )

    # --- 7. Run Adam ---
    print(f"\nRunning Adam (discrete adjoint): "
          f"step_size={step_size}, max_iter={max_iter}, tol={tol}")
    p_init_jax = jnp.asarray(p_init_full, dtype=jnp.float64)
    result = grad_computer.optimize_adam(
        solver=solver,
        p_init=p_init_jax,
        opt_param_indices=opt_param_indices,
        target_times=target_times,
        target_data=target_data,
        t_span=t_span,
        ncp=ncp,
        max_iter=max_iter,
        tol=tol,
        step_size=step_size,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        blend_sharpness=blend_sharpness,
        print_every=print_every,
        adaptive_horizon=False,
    )

    p_opt_full = np.asarray(result["p_opt"])
    p_opt = np.asarray([p_opt_full[i] for i in opt_param_indices])

    # --- 7b. Best-snapshot post-processing ---
    # `optimize_adam` records `loss_history[i]` paired with `p_history[i]`
    # (both captured *before* the iter-i Adam update), so we can scan the
    # histories to recover the best-loss iterate without re-instrumenting
    # the inner loop. When `--best-every N > 0` and an earlier iterate
    # beats the last one, override `p_opt`/`p_opt_full` with that snapshot.
    best_every = max(0, int(args.best_every))
    if best_every > 0:
        print(f"\nBest-snapshot tracking enabled: every {best_every} iter.")
    best_iter_da = -1
    best_loss_da = float("inf")
    best_grad_da = float("nan")
    last_loss_da = float("nan")
    if best_every > 0:
        loss_h = list(result.get("loss_history", []))
        grad_h = list(result.get("grad_norm_history", []))
        p_h = list(result.get("p_history", []))
        last_loss_da = float(loss_h[-1]) if loss_h else float("nan")
        for i, lv in enumerate(loss_h):
            it = i + 1
            if (it == 1 or it % best_every == 0):
                lvf = float(lv)
                if np.isfinite(lvf) and lvf < best_loss_da and i < len(p_h):
                    best_loss_da = lvf
                    best_iter_da = it
                    best_grad_da = (float(grad_h[i])
                                    if i < len(grad_h) else float("nan"))
                    best_p_opt = np.asarray(p_h[i])
                    full = np.asarray(p_init_full, dtype=float).copy()
                    for k, idx in enumerate(opt_param_indices):
                        full[idx] = float(best_p_opt[k])
                    best_p_full = full
        if (best_iter_da > 0 and np.isfinite(last_loss_da)
                and best_loss_da < last_loss_da):
            print(f"  Returning best snapshot: loss={best_loss_da:.6e} "
                  f"at iter {best_iter_da} "
                  f"(last iter loss={last_loss_da:.6e}).")
            p_opt_full = best_p_full
            p_opt = np.asarray([p_opt_full[i] for i in opt_param_indices])

    # --- 8. Optimized trajectory ---
    solver.update_parameters(p_opt_full)
    sol_opt = solver.solve_augmented(t_span, ncp=ncp)
    t_opt_full, x_opt_full = _stack_segments(sol_opt)

    # --- 9. Report ---
    print("\n" + "=" * 70)
    print("Result")
    print("=" * 70)
    history = result.get("history", {})
    loss_hist = result.get("loss_history", history.get("loss", []))
    grad_hist = result.get("grad_norm_history", history.get("grad_norm", []))
    n_iter = result.get("n_iter", len(loss_hist))
    converged = result.get("converged", False)
    print(f"  iterations:    {n_iter}")
    print(f"  converged:     {converged}")
    if loss_hist:
        print(f"  final loss:    {float(loss_hist[-1]):.6e}")
    if grad_hist:
        print(f"  final |grad|:  {float(grad_hist[-1]):.3e}")
    if best_every > 0 and best_iter_da > 0:
        print(f"  best snapshot: loss={best_loss_da:.6e}  "
              f"@ iter {best_iter_da}  "
              f"(last iter loss={last_loss_da:.6e})")
    print(f"\n  parameter recovery:")
    print(f"  {'name':12s}  {'true':>12s}  {'init':>12s}  "
          f"{'optim':>12s}  {'err_init':>10s}  {'err_opt':>10s}")
    for n, t_, i_, o_ in zip(opt_param_names, p_true_opt, p_init_opt, p_opt):
        ei = abs(i_ - t_) / abs(t_) if t_ != 0 else float("nan")
        eo = abs(o_ - t_) / abs(t_) if t_ != 0 else float("nan")
        print(f"  {n:12s}  {t_:12.6f}  {i_:12.6f}  {o_:12.6f}  "
              f"{ei:10.2%}  {eo:10.2%}")

    # --- 10. Persist + plot ---
    os.makedirs(args.results_dir, exist_ok=True)
    record = {
        "config_path": os.path.abspath(args.config),
        "spec_path": os.path.abspath(spec_path),
        "n_segments_true": len(sol_true.segments),
        "n_events_true": len(sol_true.events),
        "n_targets": int(len(target_times)),
        "max_blocks": max_blocks,
        "max_points_per_segment": max_pts,
        "max_targets": max_targets,
        "downsample_segments": downsample_segments,
        "blend_sharpness": blend_sharpness,
        "step_size": step_size, "beta1": beta1, "beta2": beta2,
        "epsilon": epsilon, "max_iter": max_iter, "tol": tol,
        "bias_factor": args.bias_factor, "bias_seed": args.bias_seed,
        "opt_param_names": opt_param_names,
        "p_true_opt": p_true_opt.tolist(),
        "p_init_opt": p_init_opt.tolist(),
        "p_opt": p_opt.tolist(),
        "loss_history": [float(v) for v in loss_hist],
        "grad_norm_history": [float(v) for v in grad_hist],
        "n_iter": int(n_iter),
        "converged": bool(converged),
        "best_every": best_every,
        "best_iter": int(best_iter_da),
        "best_loss": float(best_loss_da) if np.isfinite(best_loss_da)
                     else None,
        "loss_last": float(last_loss_da) if np.isfinite(last_loss_da)
                     else None,
    }
    prefix = "optimization_cauer_events_da"
    record_path = os.path.join(args.results_dir, f"{prefix}.json")
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n  Saved: {record_path}")

    if not args.no_plot:
        # Resample the three trajectories onto a common uniform grid for
        # the output plot (segments have different sample counts).
        t_grid = np.linspace(t_start, t_stop, ncp + 1)
        y_true_g = _resample_trajectory(t_true_full, x_true_full, t_grid)
        y_init_g = _resample_trajectory(t_init_full, x_init_full, t_grid)
        y_opt_g = _resample_trajectory(t_opt_full, x_opt_full, t_grid)

        plot_history = {
            "loss": [float(v) for v in loss_hist],
            "grad_norm": [float(v) for v in grad_hist],
        }
        # The DA path plots differential states from `_stack_segments`
        # (one column per state in spec order); use those names directly.
        state_names_plot = [s["name"] for s in (dae_data.get("states") or [])]
        if len(state_names_plot) != y_true_g.shape[1]:
            state_names_plot = None
        plot_results(
            plot_history, t_grid, y_true_g, y_init_g, y_opt_g,
            opt_param_names, p_true_opt, p_init_opt, p_opt,
            args.results_dir, prefix=prefix,
            state_names=state_names_plot,
        )


# ---------------------------------------------------------------------- #
# Trajectory utilities
# ---------------------------------------------------------------------- #

def _stack_segments(sol):
    """Concatenate (t, x) arrays from `AugmentedSolution.segments`."""
    ts, xs = [], []
    for seg in sol.segments:
        if len(seg.t) > 0:
            ts.append(np.asarray(seg.t))
            xs.append(np.asarray(seg.x))
    if not ts:
        return np.zeros(0), np.zeros((0, 0))
    return np.concatenate(ts), np.concatenate(xs, axis=0)


def _resample_trajectory(t_src, x_src, t_grid):
    """Linearly resample a per-state trajectory onto `t_grid`. Returns
    shape (n_grid, n_states). Right-continuous at events: a sample at
    an event boundary picks the post-event value (consistent with the
    diffrax runner's plot)."""
    if t_src.size == 0:
        return np.zeros((len(t_grid), 0))
    n_states = x_src.shape[1]
    out = np.empty((len(t_grid), n_states), dtype=float)
    for j in range(n_states):
        out[:, j] = np.interp(t_grid, t_src, x_src[:, j])
    return out


if __name__ == "__main__":
    main()
