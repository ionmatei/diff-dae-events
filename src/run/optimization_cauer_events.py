"""
Adam optimization on the Cauer DAE WITH state-dependent events,
using the diffrax + composite-event + reinit path
(`src/dae/dae_optimizer_events.py`).

Pipeline:
  1. Load `config/config_cauer_events.yaml` (spec overridable via --spec).
  2. Simulate at nominal parameters to build the ground-truth trajectory.
  3. Bias the parameters to optimize.
  4. Run Adam against the squared output error.
  5. Save loss / parameter / output plots and a JSON record.
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

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from src.dae.dae_optimizer_events import DAEOptimizerJaxADEvents  # noqa: E402
from src.run._cauer_events_ida_loss import evaluate_ida_mse  # noqa: E402


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
    """Log-uniform multiplicative bias: `p_init = p_true * exp(factor * U(-1,1))`.

    Always strictly positive (the optimizer is positive-definite —
    capacitances, inductances, resistances). `factor` is the half-width
    of the log-uniform range, so the multiplier lies in
    `[exp(-factor), exp(+factor)]`. `factor=0.3` ≈ ±35%, `factor=0.5` ≈
    ±65%, `factor=1.0` ≈ a 2.7x band on either side.
    """
    rng = np.random.default_rng(seed)
    p_init = p_nominal.copy()
    log_perturb = factor * rng.uniform(-1.0, 1.0, size=len(opt_indices))
    for k, idx in enumerate(opt_indices):
        p_init[idx] = p_nominal[idx] * float(np.exp(log_perturb[k]))
    return p_init


def lbfgs_optimize(
    optimizer: DAEOptimizerJaxADEvents,
    target_times: np.ndarray,
    y_target: np.ndarray,
    p_opt0: np.ndarray,
    *,
    max_iter: int, tol: float, print_every: int = 1,
    bounds_eps: float = 1.0e-8,
    best_check_every: int = 0,
):
    """L-BFGS-B via scipy.optimize.minimize, using the JIT'd
    `_loss_and_grad` for the function and Jacobian.

    L-BFGS approximates the Hessian from past gradient deltas, so it
    follows the curvature of the loss surface — unlike Adam, which only
    adapts per-coordinate magnitudes. On a poorly-conditioned, ridgy
    loss like the events-DAE one, this typically recovers parameters
    that Adam can't even with the correct gradient.
    """
    from scipy.optimize import minimize

    p0_np = np.asarray(p_opt0, dtype=np.float64)
    t_jax = jnp.asarray(target_times, dtype=jnp.float64)
    y_jax = jnp.asarray(y_target, dtype=jnp.float64)
    if y_jax.shape[0] != t_jax.shape[0]:
        y_jax = y_jax.T

    # Loose positivity bounds — keep parameters strictly > 0 so the DAE
    # stays well-defined even if L-BFGS' line-search probes a wide step.
    bounds = [(bounds_eps, None)] * len(p0_np)

    history = {"loss": [], "grad_norm": [], "params": [], "iter_time_s": []}
    state = {"n_eval": 0, "t_last_iter": time.time(), "fail": False, "exc": None}
    best = {"loss": float("inf"), "iter": -1, "params": None,
            "grad_norm": float("nan")}

    def fun_and_jac(p_np):
        if state["fail"]:
            # Once we've recorded a failure, return a finite-but-huge
            # surrogate so scipy can short-circuit gracefully.
            return 1e30, np.zeros_like(p_np)
        try:
            p_jax = jnp.asarray(p_np, dtype=jnp.float64)
            loss, g = optimizer._loss_and_grad(
                p_jax, t_jax, y_jax, optimizer.x0
            )
            loss.block_until_ready(); g.block_until_ready()
            state["n_eval"] += 1
            return float(loss), np.asarray(g, dtype=np.float64)
        except Exception as exc:
            state["fail"] = True
            state["exc"] = exc
            return 1e30, np.zeros_like(p_np)

    def callback(p_np):
        # Called at the end of each successful L-BFGS iteration. We
        # re-eval to capture the post-step loss/grad for history; the
        # extra cost is one fwd-bwd per iter (line-search inside
        # L-BFGS already does several), so it's a few % overhead.
        loss, g = optimizer._loss_and_grad(
            jnp.asarray(p_np, dtype=jnp.float64), t_jax, y_jax, optimizer.x0
        )
        gnorm = float(jnp.linalg.norm(g))
        loss_f = float(loss)
        dt = time.time() - state["t_last_iter"]
        history["loss"].append(loss_f)
        history["grad_norm"].append(gnorm)
        history["params"].append(np.asarray(p_np))
        history["iter_time_s"].append(dt)
        n_iter = len(history["loss"])
        if (best_check_every > 0
                and (n_iter == 1 or n_iter % best_check_every == 0)
                and np.isfinite(loss_f) and loss_f < best["loss"]):
            best["loss"] = loss_f
            best["iter"] = n_iter
            best["params"] = np.asarray(p_np)
            best["grad_norm"] = gnorm
        if n_iter == 1 or n_iter % print_every == 0:
            print(f"  iter {n_iter:4d}  loss={loss_f:.6e}  |grad|={gnorm:.3e}"
                  f"  ({dt:.2f}s, n_eval={state['n_eval']})")
        state["t_last_iter"] = time.time()

    result = minimize(
        fun_and_jac, p0_np, jac=True, method="L-BFGS-B",
        bounds=bounds, callback=callback,
        options={"maxiter": max_iter, "gtol": tol, "disp": False},
    )

    if state["fail"] and state["exc"] is not None:
        print(f"  L-BFGS aborted: integration failed — "
              f"{type(state['exc']).__name__}: "
              f"{str(state['exc']).splitlines()[0][:150]}")

    p_last = np.asarray(result.x)
    loss_last = (
        float(result.fun) if np.isfinite(result.fun)
        else (history["loss"][-1] if history["loss"] else float("nan"))
    )
    grad_last = (history["grad_norm"][-1] if history["grad_norm"]
                 else float("nan"))

    use_best = (best_check_every > 0 and best["params"] is not None
                and best["loss"] < loss_last)
    if use_best:
        print(f"  Returning best snapshot: loss={best['loss']:.6e} "
              f"at iter {best['iter']} (last iter loss={loss_last:.6e}).")
        p_final = best["params"]
        loss_final = best["loss"]
        grad_final = best["grad_norm"]
    else:
        p_final = p_last
        loss_final = loss_last
        grad_final = grad_last

    return {
        "p_opt": p_final,
        "loss_final": loss_final,
        "grad_norm_final": grad_final,
        "converged": bool(result.success),
        "n_iter": int(result.nit),
        "history": history,
        "p_last": p_last,
        "loss_last": loss_last,
        "best_iter": best["iter"],
        "best_loss": best["loss"] if best["params"] is not None else float("nan"),
    }


def adam_optimize(
    optimizer: DAEOptimizerJaxADEvents,
    target_times: np.ndarray,
    y_target: np.ndarray,
    p_opt0: np.ndarray,
    *,
    step_size: float, beta1: float, beta2: float, epsilon: float,
    max_iter: int, tol: float, print_every: int = 1,
    best_check_every: int = 0,
):
    p = jnp.asarray(p_opt0, dtype=jnp.float64)
    m = jnp.zeros_like(p)
    v = jnp.zeros_like(p)

    t_jax = jnp.asarray(target_times, dtype=jnp.float64)
    y_jax = jnp.asarray(y_target, dtype=jnp.float64)
    if y_jax.shape[0] != t_jax.shape[0]:
        y_jax = y_jax.T

    history = {"loss": [], "grad_norm": [], "params": [], "iter_time_s": []}
    converged = False
    best = {"loss": float("inf"), "iter": -1, "params": None,
            "grad_norm": float("nan")}

    for it in range(1, max_iter + 1):
        t0 = time.time()
        # Bracket the loss/grad call: when Adam wanders into a parameter
        # region where the dynamics produce non-finite state (e.g., an
        # algebraic Jacobian goes singular, or `1/L_L` blows up),
        # diffrax's event refinement passes NaN into lineax and an
        # `_EquinoxRuntimeError` ("linear solver received non-finite
        # input") propagates out of JIT. Without this catch, the entire
        # run dies and the partial optimization history is lost.
        try:
            loss, g = optimizer._loss_and_grad(p, t_jax, y_jax, optimizer.x0)
        except Exception as exc:  # diffrax / lineax / equinox runtime errors
            print(f"  iter {it}: integration failed — {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:150]}")
            print(f"  Aborting Adam loop with last known good params at iter {it - 1}.")
            break
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1 ** it)
        v_hat = v / (1.0 - beta2 ** it)
        p_new = p - step_size * m_hat / (jnp.sqrt(v_hat) + epsilon)
        p_new.block_until_ready()
        dt = time.time() - t0

        gnorm = float(jnp.linalg.norm(g))
        loss_f = float(loss)

        # If the loss/grad already went NaN (e.g., scan saturation hit),
        # bail before the update poisons all subsequent iters.
        if not np.isfinite(loss_f) or not np.isfinite(gnorm):
            print(f"  iter {it}: loss or |grad| is non-finite "
                  f"(loss={loss_f}, |grad|={gnorm}). Aborting; the most "
                  f"likely causes are `max_segments` saturation or Adam "
                  f"having driven the parameters into a singular regime.")
            break

        # `loss_f` was computed at the current (pre-update) `p` — capture
        # the snapshot before mutating `p` so (params, loss) are paired.
        if (best_check_every > 0
                and (it == 1 or it % best_check_every == 0)
                and loss_f < best["loss"]):
            best["loss"] = loss_f
            best["iter"] = it
            best["params"] = np.asarray(p)
            best["grad_norm"] = gnorm

        p = p_new
        history["loss"].append(loss_f)
        history["grad_norm"].append(gnorm)
        history["params"].append(np.asarray(p))
        history["iter_time_s"].append(dt)

        if it == 1 or it == max_iter or it % print_every == 0:
            print(f"  iter {it:4d}  loss={loss_f:.6e}  |grad|={gnorm:.3e}  "
                  f"({dt:.2f}s)")

        if gnorm < tol:
            converged = True
            print(f"  Converged at iter {it} (|grad|={gnorm:.3e}).")
            break

    p_last = np.asarray(p)
    loss_last = history["loss"][-1] if history["loss"] else float("nan")
    grad_last = (history["grad_norm"][-1] if history["grad_norm"]
                 else float("nan"))

    use_best = (best_check_every > 0 and best["params"] is not None
                and best["loss"] < loss_last)
    if use_best:
        print(f"  Returning best snapshot: loss={best['loss']:.6e} "
              f"at iter {best['iter']} (last iter loss={loss_last:.6e}).")
        p_final = best["params"]
        loss_final = best["loss"]
        grad_final = best["grad_norm"]
    else:
        p_final = p_last
        loss_final = loss_last
        grad_final = grad_last

    return {
        "p_opt": p_final,
        "loss_final": loss_final,
        "grad_norm_final": grad_final,
        "converged": converged,
        "n_iter": len(history["loss"]),
        "history": history,
        "p_last": p_last,
        "loss_last": loss_last,
        "best_iter": best["iter"],
        "best_loss": best["loss"] if best["params"] is not None else float("nan"),
    }


def plot_results(history, t_array, y_target, y_init, y_opt,
                 param_names_opt, p_true, p_init_opt, p_opt,
                 out_dir: str, prefix: str,
                 state_names: list | None = None):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1, len(history["loss"]) + 1), history["loss"],
                "-o", markersize=3, color="tab:blue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Cauer DAE (events) — Adam loss")
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
    ax.bar(x + width, p_opt, width, label="optimized", color="tab:blue")
    ax.set_xticks(x); ax.set_xticklabels(param_names_opt, rotation=30, ha="right")
    ax.set_ylabel("Parameter value")
    ax.set_title("Cauer DAE (events) — parameter recovery")
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
        ax.plot(t_array, y_opt[:, i], ":", color="tab:blue",
                linewidth=1.4, label="optimized" if i == 0 else None)
        ylab = (state_names[i] if state_names is not None
                and i < len(state_names) else f"y[{i}]")
        ax.set_ylabel(ylab); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time")
    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    p_traj = os.path.join(out_dir, f"{prefix}_outputs.png")
    plt.savefig(p_traj, dpi=150); plt.close(fig)
    print(f"  Saved: {p_traj}")


def main():
    parser = argparse.ArgumentParser(
        description="Adam on event-aware Cauer DAE via "
                    "DAEOptimizerJaxADEvents."
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(root_dir, "config", "config_cauer_events.yaml"),
        help="Path to the events YAML config.",
    )
    parser.add_argument(
        "--spec", type=str, default=None,
        help="Path to the DAE spec (overrides the value in --config).",
    )
    parser.add_argument(
        "--diffrax-solver", default=None,
        choices=("Tsit5", "Dopri5", "Dopri8", "Heun"),
        help="Override `optimizer.diffrax_solver` in the YAML.",
    )
    parser.add_argument(
        "--rtol", type=float, default=None,
        help="Override `dae_solver.rtol` in the YAML.",
    )
    parser.add_argument(
        "--atol", type=float, default=None,
        help="Override `dae_solver.atol` in the YAML.",
    )
    parser.add_argument(
        "--max-segments", type=int, default=None,
        help="Override `optimizer.max_segments` in the YAML.",
    )
    parser.add_argument(
        "--blend-sharpness", type=float, default=None,
        help="If set (and > 0), use sigmoid-blended segment selection in "
             "stage 3 of the segmented integration instead of hard argmax. "
             "Larger β = sharper transition (β→∞ recovers argmax). "
             "Smaller β widens the loss basin around the truth — useful "
             "for wider initial bias. Off by default for the AD runner: "
             "the YAML's `optimizer.blend_sharpness` is a DA-only knob and "
             "is NOT read here. To enable for AD, either pass this flag or "
             "add `optimizer.ad_blend_sharpness: <value>` to the YAML. "
             "Typical value: 100-300.",
    )
    parser.add_argument(
        "--bias-factor", type=float, default=0.1,
        help="Half-width of the log-uniform multiplicative perturbation: "
             "p_init = p_true * exp(factor * U(-1, 1)). 0.3 ≈ ±35%%; "
             "0.5 ≈ ±65%%; 1.0 ≈ ±170%% on the multiplier.",
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
        help="If > 0, snapshot the (params, loss) pair every N iterations "
             "and return the best-loss snapshot at the end instead of the "
             "last iterate. Useful when the optimizer oscillates near the "
             "basin floor (Adam especially), so the final iterate may be "
             "worse than an earlier one. 0 (default) disables and returns "
             "the last iterate as before. Set to 1 to compare every iter.",
    )
    parser.add_argument(
        "--optimizer", choices=("adam", "lbfgs"), default="adam",
        help="Outer optimizer. 'adam' = the existing manual Adam loop "
             "with this runner's hyperparameters from the YAML. "
             "'lbfgs' = scipy L-BFGS-B (Hessian-aware, follows curvature, "
             "ignores `step_size`/`beta*`/`epsilon`). For the events-DAE "
             "loss surface — which is poorly conditioned around the "
             "minimum — L-BFGS typically converges in <50 iters where "
             "Adam can't recover at all.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Cauer DAE WITH EVENTS — Adam optimization")
    print("=" * 70)

    dae_data, solver_cfg, opt_cfg, spec_path = load_config(args.config, args.spec)
    print(f"  spec: {spec_path}")
    print(f"  when clauses: {len(dae_data.get('when') or [])}")

    t_start = float(solver_cfg["start_time"])
    t_stop = float(solver_cfg["stop_time"])
    ncp = int(solver_cfg["ncp"])
    n_steps = ncp + 1
    t_array = np.linspace(t_start, t_stop, n_steps)
    print(f"  time grid: [{t_start}, {t_stop}], n_steps={n_steps}")

    opt_param_names = list(opt_cfg["opt_params"])
    loss_type = opt_cfg.get("loss_type", "sum")
    max_iter = int(args.max_iter_override or opt_cfg["max_iterations"])
    tol = float(opt_cfg["tol"])
    print_every = int(opt_cfg.get("print_every", 1))

    alg = opt_cfg["algorithm"]
    params = alg.get("params", {})
    step_size = float(params.get("step_size", 1.0e-4))
    beta1 = float(params.get("beta1", 0.9))
    beta2 = float(params.get("beta2", 0.999))
    epsilon = float(params.get("epsilon", 1e-8))

    # Event-optimizer knobs come from the YAML; `--*` flags override.
    diffrax_solver = args.diffrax_solver or opt_cfg.get("diffrax_solver", "Tsit5")
    rtol = args.rtol if args.rtol is not None else float(solver_cfg.get("rtol", 1e-6))
    atol = args.atol if args.atol is not None else float(solver_cfg.get("atol", 1e-6))
    max_segments = (args.max_segments if args.max_segments is not None
                    else int(opt_cfg.get("max_segments", 16)))
    newton_max_iter = int(opt_cfg.get("newton_max_iter", 10))
    diffrax_max_steps = int(opt_cfg.get("diffrax_max_steps", 4096))

    # `blend_sharpness` is opt-in for the AD runner. The YAML's
    # `optimizer.blend_sharpness` is documented for DA only and is NOT
    # read here — using it would silently turn blending on for AD.
    # AD-only override key: `optimizer.ad_blend_sharpness` (absent by
    # default ⇒ blending disabled). CLI flag still wins. Treat 0 / None
    # as "disabled".
    if args.blend_sharpness is not None:
        bs_raw = args.blend_sharpness
    else:
        bs_raw = opt_cfg.get("ad_blend_sharpness", None)
    blend_sharpness = (
        None if (bs_raw is None or float(bs_raw) <= 0.0)
        else float(bs_raw)
    )

    optimizer = DAEOptimizerJaxADEvents(
        dae_data, optimize_params=opt_param_names, loss_type=loss_type,
        diffrax_solver=diffrax_solver,
        rtol=rtol, atol=atol,
        max_segments=max_segments,
        newton_max_iter=newton_max_iter,
        diffrax_max_steps=diffrax_max_steps,
        blend_sharpness=blend_sharpness,
    )

    p_nominal = np.asarray(optimizer.p_all)
    param_names_all = optimizer.param_names
    p_true_opt = np.asarray([
        p_nominal[param_names_all.index(n)] for n in opt_param_names
    ])

    # Ground truth
    print("\nSimulating ground truth at nominal parameters...")
    t0 = time.time()
    sim_true = optimizer.simulate(t_array, p_nominal)
    print(f"  done ({time.time() - t0:.2f}s)  y shape = {sim_true['y'].shape}")
    y_target = sim_true["y"].T

    # Biased init
    p_init_full = bias_parameters(
        p_nominal, optimizer.optimize_indices,
        factor=args.bias_factor, seed=args.bias_seed,
    )
    p_init_opt = np.asarray([
        p_init_full[param_names_all.index(n)] for n in opt_param_names
    ])
    print("\nInitial vs. true:")
    for n, t_, i_ in zip(opt_param_names, p_true_opt, p_init_opt):
        rel = (i_ - t_) / t_ if t_ != 0 else float("nan")
        print(f"  {n:12s}  true={t_: .5f}   init={i_: .5f}   "
              f"rel_bias={rel:+.2%}")
    sim_init = optimizer.simulate(t_array, p_init_full)
    y_init = sim_init["y"].T

    best_every = max(0, int(args.best_every))
    if best_every > 0:
        print(f"\nBest-snapshot tracking enabled: every {best_every} iter.")

    # Outer optimizer: Adam or L-BFGS-B.
    if args.optimizer == "adam":
        print(f"\nRunning Adam: step_size={step_size}, beta1={beta1}, "
              f"beta2={beta2}, max_iter={max_iter}, tol={tol}")
        result = adam_optimize(
            optimizer, t_array, y_target, p_init_opt,
            step_size=step_size, beta1=beta1, beta2=beta2, epsilon=epsilon,
            max_iter=max_iter, tol=tol, print_every=print_every,
            best_check_every=best_every,
        )
    else:  # lbfgs
        print(f"\nRunning L-BFGS-B: max_iter={max_iter}, gtol={tol}")
        print(f"  (Adam hyperparameters from YAML are ignored under L-BFGS.)")
        result = lbfgs_optimize(
            optimizer, t_array, y_target, p_init_opt,
            max_iter=max_iter, tol=tol, print_every=print_every,
            best_check_every=best_every,
        )

    p_opt = result["p_opt"]
    p_opt_full = p_nominal.copy()
    for k, idx in enumerate(optimizer.optimize_indices):
        p_opt_full[idx] = p_opt[k]
    sim_opt = optimizer.simulate(t_array, p_opt_full)
    y_opt = sim_opt["y"].T

    # Report
    print("\n" + "=" * 70)
    print("Result")
    print("=" * 70)
    print(f"  iterations:    {result['n_iter']}")
    print(f"  converged:     {result['converged']}")
    print(f"  final loss:    {result['loss_final']:.6e}")
    print(f"  final |grad|:  {result['grad_norm_final']:.3e}")
    if best_every > 0 and "best_iter" in result and result["best_iter"] >= 0:
        print(f"  best snapshot: loss={result['best_loss']:.6e}  "
              f"@ iter {result['best_iter']}  "
              f"(last iter loss={result['loss_last']:.6e})")
    print(f"\n  parameter recovery:")
    print(f"  {'name':12s}  {'true':>12s}  {'init':>12s}  "
          f"{'optim':>12s}  {'err_init':>10s}  {'err_opt':>10s}")
    for n, t_, i_, o_ in zip(opt_param_names, p_true_opt, p_init_opt, p_opt):
        ei = abs(i_ - t_) / abs(t_) if t_ != 0 else float("nan")
        eo = abs(o_ - t_) / abs(t_) if t_ != 0 else float("nan")
        print(f"  {n:12s}  {t_:12.6f}  {i_:12.6f}  {o_:12.6f}  "
              f"{ei:10.2%}  {eo:10.2%}")

    # IDA-based MSE for apples-to-apples comparison with the DA runner.
    print("\nIDA-based MSE evaluation (trusted forward solver):")
    try:
        ida_eval = evaluate_ida_mse(
            dae_data, (t_start, t_stop), ncp,
            p_nominal=p_nominal, p_init_full=p_init_full,
            p_opt_full=p_opt_full, rtol=rtol, atol=atol,
        )
        print(f"  mse_init (IDA): {ida_eval['mse_init_ida']:.6e}")
        print(f"  mse_opt  (IDA): {ida_eval['mse_opt_ida']:.6e}")
    except Exception as exc:
        print(f"  [warn] IDA-based evaluation failed: "
              f"{type(exc).__name__}: {exc}")
        ida_eval = {"mse_init_ida": None, "mse_opt_ida": None,
                    "ncp_ida_eval": int(ncp),
                    "rtol_ida_eval": float(rtol),
                    "atol_ida_eval": float(atol)}

    os.makedirs(args.results_dir, exist_ok=True)
    record = {
        "config_path": os.path.abspath(args.config),
        "spec_path": os.path.abspath(spec_path),
        "n_steps": int(n_steps),
        "diffrax_solver": diffrax_solver,
        "rtol": rtol, "atol": atol,
        "max_segments": max_segments,
        "newton_max_iter": newton_max_iter,
        "diffrax_max_steps": diffrax_max_steps,
        "blend_sharpness": blend_sharpness,
        "loss_type": loss_type,
        "step_size": step_size, "beta1": beta1, "beta2": beta2,
        "epsilon": epsilon, "max_iter": max_iter, "tol": tol,
        "bias_factor": args.bias_factor, "bias_seed": args.bias_seed,
        "opt_param_names": opt_param_names,
        "p_true_opt": p_true_opt.tolist(),
        "p_init_opt": p_init_opt.tolist(),
        "p_opt": p_opt.tolist(),
        "loss_history": [float(v) for v in result["history"]["loss"]],
        "grad_norm_history": [float(v) for v in result["history"]["grad_norm"]],
        "iter_time_s_history": [float(v) for v in result["history"]["iter_time_s"]],
        "n_iter": int(result["n_iter"]),
        "converged": bool(result["converged"]),
        "loss_final": float(result["loss_final"]),
        "grad_norm_final": float(result["grad_norm_final"]),
        "best_every": best_every,
        "best_iter": int(result.get("best_iter", -1)),
        "best_loss": float(result.get("best_loss", float("nan"))),
        "p_last": (np.asarray(result["p_last"]).tolist()
                   if "p_last" in result else None),
        "loss_last": (float(result["loss_last"])
                      if "loss_last" in result else None),
        **ida_eval,
    }
    prefix = f"optimization_cauer_events_{args.optimizer}"
    record_path = os.path.join(args.results_dir, f"{prefix}.json")
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n  Saved: {record_path}")

    if not args.no_plot:
        # `h` is None ⇒ identity output map ⇒ y has the same names as
        # the differential states. Fall back to generic indices if a
        # different output map ever shrinks/expands the y dimension.
        state_names_plot = [s["name"] for s in (dae_data.get("states") or [])]
        if len(state_names_plot) != y_target.shape[1]:
            state_names_plot = None
        plot_results(
            result["history"], t_array, y_target, y_init, y_opt,
            opt_param_names, p_true_opt, p_init_opt, p_opt,
            args.results_dir, prefix=prefix,
            state_names=state_names_plot,
        )


if __name__ == "__main__":
    main()
