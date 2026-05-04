"""
Adam optimization on the Cauer DAE using DAEOptimizerJaxAD.

Pipeline:
  1. Load `config/config_cauer.yaml` and the DAE specification it
     points to.
  2. Simulate the DAE at nominal parameter values (RK4 + per-stage
     Newton on the algebraic constraint) to build ground-truth output
     trajectories on a uniform time grid.
  3. Perturb the parameters listed in `optimizer.opt_params` to create a
     biased initial guess.
  4. Run Adam (driven from this script) on the squared output error,
     using `DAEOptimizerJaxAD._loss_and_grad` for each step.
  5. Save a loss curve, a parameter-recovery bar plot, and a JSON dump
     of the run into `results/`.
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

from src.dae.dae_optimizer import DAEOptimizerJaxAD  # noqa: E402
from src.dae.dae_optimizer_fast import DAEOptimizerJaxADFast  # noqa: E402


def build_optimizer(variant: str, dae_data, opt_param_names, loss_type,
                    diffrax_solver: str = "Tsit5",
                    rtol: float = 1.0e-6, atol: float = 1.0e-6):
    """Construct the optimizer requested by `--variant`.

    `ad`      → DAEOptimizerJaxAD with RK4 + per-stage Newton on `g`.
    `diffrax` → DAEOptimizerJaxAD with adaptive diffrax integrator
                (Tsit5 by default; pass `--diffrax-solver Kvaerno5` for
                stiff systems). Uses the same custom_vjp algebraic
                solve, so gradients flow through IFT.
    `fast`    → DAEOptimizerJaxADFast with implicit Euler + DEER chord
                iteration on the augmented state `[x, z]`.
    """
    if variant == "ad":
        return DAEOptimizerJaxAD(
            dae_data,
            optimize_params=opt_param_names,
            loss_type=loss_type,
            solver_method="rk4",
        )
    if variant == "diffrax":
        return DAEOptimizerJaxAD(
            dae_data,
            optimize_params=opt_param_names,
            loss_type=loss_type,
            solver_method="diffrax",
            diffrax_solver=diffrax_solver,
            rtol=rtol,
            atol=atol,
        )
    if variant == "fast":
        return DAEOptimizerJaxADFast(
            dae_data,
            optimize_params=opt_param_names,
            loss_type=loss_type,
            solver_method="implicit_euler",
            use_deer_iteration=True,
        )
    raise ValueError(
        f"Unknown variant {variant!r}; expected 'ad', 'diffrax', or 'fast'."
    )


def load_config(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    solver_cfg = cfg["dae_solver"]
    opt_cfg = cfg["optimizer"]

    spec_path = solver_cfg["dae_specification_file"]
    if not os.path.isabs(spec_path):
        spec_path = os.path.join(root_dir, spec_path)
    with open(spec_path, "r") as f:
        if spec_path.endswith((".yaml", ".yml")):
            dae_data = yaml.safe_load(f)
        else:
            dae_data = json.load(f)
    return dae_data, solver_cfg, opt_cfg


def bias_parameters(p_nominal: np.ndarray, opt_indices, factor: float,
                    seed: int = 0) -> np.ndarray:
    """Multiplicative perturbation of the parameters to optimize.

    Each parameter is multiplied by `1 + factor * u` where `u ~ U(-1, 1)`,
    so the bias has roughly `factor * 100 %` magnitude in either direction.
    Non-optimized parameters are left untouched.
    """
    rng = np.random.default_rng(seed)
    p_init = p_nominal.copy()
    perturb = 1.0 + factor * rng.uniform(-1.0, 1.0, size=len(opt_indices))
    for k, idx in enumerate(opt_indices):
        p_init[idx] = p_nominal[idx] * perturb[k]
    return p_init


def adam_optimize(
    optimizer: DAEOptimizerJaxAD,
    t_array: np.ndarray,
    y_target: np.ndarray,
    p_opt0: np.ndarray,
    *,
    step_size: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    max_iter: int,
    tol: float,
    print_every: int = 10,
):
    """Adam loop using `optimizer._loss_and_grad` (already JIT-compiled)."""
    p = jnp.asarray(p_opt0, dtype=jnp.float64)
    m = jnp.zeros_like(p)
    v = jnp.zeros_like(p)

    t_jax = jnp.asarray(t_array, dtype=jnp.float64)
    y_jax = jnp.asarray(y_target, dtype=jnp.float64)
    if y_jax.shape[0] != t_jax.shape[0]:
        y_jax = y_jax.T

    history = {"loss": [], "grad_norm": [], "params": [], "iter_time_s": []}
    converged = False

    for it in range(1, max_iter + 1):
        t0 = time.time()
        loss, g = optimizer._loss_and_grad(
            p, optimizer.x0, optimizer.z0, t_jax, y_jax
        )
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1 ** it)
        v_hat = v / (1.0 - beta2 ** it)
        p = p - step_size * m_hat / (jnp.sqrt(v_hat) + epsilon)
        # Block once so iter timing is real.
        p.block_until_ready()
        dt = time.time() - t0

        gnorm = float(jnp.linalg.norm(g))
        loss_f = float(loss)
        history["loss"].append(loss_f)
        history["grad_norm"].append(gnorm)
        history["params"].append(np.asarray(p))
        history["iter_time_s"].append(dt)

        if it == 1 or it == max_iter or it % print_every == 0:
            print(f"  iter {it:4d}  loss={loss_f:.6e}  |grad|={gnorm:.3e}  "
                  f"({dt*1000:.1f} ms)")

        if gnorm < tol:
            converged = True
            print(f"  Converged at iter {it} (|grad|={gnorm:.3e} < tol={tol}).")
            break

    return {
        "p_opt": np.asarray(p),
        "loss_final": history["loss"][-1],
        "grad_norm_final": history["grad_norm"][-1],
        "converged": converged,
        "n_iter": len(history["loss"]),
        "history": history,
    }


def plot_results(history, t_array, y_target, y_init, y_opt,
                 param_names_opt, p_true, p_init_opt, p_opt,
                 out_dir: str, prefix: str = "cauer"):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # --- loss curve ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1, len(history["loss"]) + 1), history["loss"],
                "-o", markersize=3, color="tab:blue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (sum/mean of squared output error)")
    ax.set_title("Cauer DAE — Adam optimization loss")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    p_loss = os.path.join(out_dir, f"{prefix}_loss.png")
    plt.savefig(p_loss, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p_loss}")

    # --- parameter recovery ---
    n = len(param_names_opt)
    x = np.arange(n)
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(7, 1.0 * n), 4))
    ax.bar(x - width, p_true, width, label="true", color="tab:green")
    ax.bar(x, p_init_opt, width, label="initial (biased)", color="tab:orange")
    ax.bar(x + width, p_opt, width, label="optimized", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(param_names_opt, rotation=30, ha="right")
    ax.set_ylabel("Parameter value")
    ax.set_title("Cauer DAE — parameter recovery")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p_par = os.path.join(out_dir, f"{prefix}_params.png")
    plt.savefig(p_par, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p_par}")

    # --- output trajectories ---
    n_out = y_target.shape[1]
    fig, axes = plt.subplots(n_out, 1, figsize=(8, 1.6 * n_out + 0.6),
                             sharex=True)
    if n_out == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t_array, y_target[:, i], "-", color="tab:green",
                linewidth=1.6, label="true" if i == 0 else None)
        ax.plot(t_array, y_init[:, i], "--", color="tab:orange",
                linewidth=1.0, label="initial" if i == 0 else None)
        ax.plot(t_array, y_opt[:, i], ":", color="tab:blue",
                linewidth=1.4, label="optimized" if i == 0 else None)
        ax.set_ylabel(f"y[{i}]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time")
    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    p_traj = os.path.join(out_dir, f"{prefix}_outputs.png")
    plt.savefig(p_traj, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p_traj}")


def main():
    parser = argparse.ArgumentParser(
        description="Adam optimization on Cauer DAE via DAEOptimizerJaxADFast."
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(root_dir, "config", "config_cauer.yaml"),
        help="Path to the Cauer YAML config.",
    )
    parser.add_argument(
        "--bias-factor", type=float, default=0.3,
        help="Multiplicative perturbation magnitude for the initial guess "
             "(p_init = p_true * (1 + factor * U(-1,1))).",
    )
    parser.add_argument(
        "--bias-seed", type=int, default=0,
        help="RNG seed for the biased initial guess.",
    )
    parser.add_argument(
        "--results-dir", type=str,
        default=os.path.join(root_dir, "results"),
        help="Directory for plots and the JSON run record.",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip figure generation (still writes the JSON record).",
    )
    parser.add_argument(
        "--variant", choices=("ad", "diffrax", "fast"), default="ad",
        help="Which DAE optimizer to use: 'ad' = RK4 + per-stage Newton "
             "(src/dae/dae_optimizer.py), 'diffrax' = adaptive diffrax "
             "integrator on the same file (configure with "
             "--diffrax-solver), 'fast' = implicit Euler + DEER "
             "(src/dae/dae_optimizer_fast.py). Default: ad.",
    )
    parser.add_argument(
        "--diffrax-solver", default="Tsit5",
        choices=("Tsit5", "Dopri5", "Dopri8", "Heun",
                 "Kvaerno3", "Kvaerno5"),
        help="Diffrax solver name (only used when --variant diffrax). "
             "Use Kvaerno3/Kvaerno5 for stiff systems.",
    )
    parser.add_argument(
        "--rtol", type=float, default=1.0e-6,
        help="Relative tolerance for the diffrax PID controller.",
    )
    parser.add_argument(
        "--atol", type=float, default=1.0e-6,
        help="Absolute tolerance for the diffrax PID controller.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Cauer DAE — Adam optimization (DAEOptimizerJaxADFast)")
    print("=" * 70)

    # --- 1. Config ---
    dae_data, solver_cfg, opt_cfg = load_config(args.config)
    t_start = float(solver_cfg["start_time"])
    t_stop = float(solver_cfg["stop_time"])
    ncp = int(solver_cfg["ncp"])
    n_steps = ncp + 1
    t_array = np.linspace(t_start, t_stop, n_steps)
    print(f"  time grid: [{t_start}, {t_stop}], n_steps={n_steps} "
          f"(dt={t_array[1] - t_array[0]:.4g})")

    opt_param_names = list(opt_cfg["opt_params"])
    loss_type = opt_cfg.get("loss_type", "sum")
    max_iter = int(opt_cfg["max_iterations"])
    tol = float(opt_cfg["tol"])
    print_every = int(opt_cfg.get("print_every", 10))

    alg = opt_cfg["algorithm"]
    params = alg.get("params", {})
    step_size = float(params.get("step_size", 1e-3))
    beta1 = float(params.get("beta1", 0.9))
    beta2 = float(params.get("beta2", 0.999))
    epsilon = float(params.get("epsilon", 1e-8))
    if str(alg.get("type", "ADAM")).upper() != "ADAM":
        print(f"  [info] config algorithm.type = {alg.get('type')!r}, "
              f"but this script always runs Adam.")

    # --- 2. Build optimizer (selected by --variant) ---
    print(f"  variant: {args.variant}")
    if args.variant == "diffrax":
        print(f"  diffrax solver: {args.diffrax_solver}  "
              f"rtol={args.rtol:g} atol={args.atol:g}")
    optimizer = build_optimizer(
        args.variant, dae_data, opt_param_names, loss_type,
        diffrax_solver=args.diffrax_solver,
        rtol=args.rtol, atol=args.atol,
    )

    param_names_all = optimizer.param_names
    p_nominal = np.asarray(optimizer.p_all)
    p_true_opt = np.asarray([p_nominal[param_names_all.index(n)]
                             for n in opt_param_names])

    # --- 3. Ground truth ---
    print("\nSimulating ground truth at nominal parameters...")
    t0 = time.time()
    sim_true = optimizer.simulate(t_array, p_nominal)
    print(f"  done ({time.time() - t0:.2f} s)  "
          f"y shape = {sim_true['y'].shape}")
    # `simulate` returns y with shape (n_outputs, n_steps); we want
    # (n_steps, n_outputs) for the loss/plotting.
    y_target = sim_true["y"].T

    # --- 4. Biased initial guess ---
    p_init_full = bias_parameters(
        p_nominal, optimizer.optimize_indices,
        factor=args.bias_factor, seed=args.bias_seed,
    )
    p_init_opt = np.asarray([p_init_full[param_names_all.index(n)]
                             for n in opt_param_names])
    print("\nInitial vs. true (parameters being optimized):")
    for n, t_, i_ in zip(opt_param_names, p_true_opt, p_init_opt):
        rel = (i_ - t_) / t_ if t_ != 0 else float("nan")
        print(f"  {n:12s}  true={t_: .5f}   init={i_: .5f}   "
              f"rel_bias={rel:+.2%}")

    sim_init = optimizer.simulate(t_array, p_init_full)
    y_init = sim_init["y"].T

    # --- 5. Adam ---
    print(f"\nRunning Adam: step_size={step_size}, beta1={beta1}, "
          f"beta2={beta2}, eps={epsilon}, max_iter={max_iter}, tol={tol}")
    result = adam_optimize(
        optimizer, t_array, y_target, p_init_opt,
        step_size=step_size, beta1=beta1, beta2=beta2, epsilon=epsilon,
        max_iter=max_iter, tol=tol, print_every=print_every,
    )

    p_opt = result["p_opt"]
    p_opt_full = p_nominal.copy()
    for k, idx in enumerate(optimizer.optimize_indices):
        p_opt_full[idx] = p_opt[k]
    sim_opt = optimizer.simulate(t_array, p_opt_full)
    y_opt = sim_opt["y"].T

    # --- 6. Report ---
    print("\n" + "=" * 70)
    print("Result")
    print("=" * 70)
    print(f"  iterations:    {result['n_iter']}")
    print(f"  converged:     {result['converged']}")
    print(f"  final loss:    {result['loss_final']:.6e}")
    print(f"  final |grad|:  {result['grad_norm_final']:.3e}")
    print("\n  parameter recovery:")
    print(f"  {'name':12s}  {'true':>12s}  {'init':>12s}  "
          f"{'optim':>12s}  {'err_init':>10s}  {'err_opt':>10s}")
    for n, t_, i_, o_ in zip(opt_param_names, p_true_opt, p_init_opt, p_opt):
        ei = abs(i_ - t_) / abs(t_) if t_ != 0 else float("nan")
        eo = abs(o_ - t_) / abs(t_) if t_ != 0 else float("nan")
        print(f"  {n:12s}  {t_:12.6f}  {i_:12.6f}  {o_:12.6f}  "
              f"{ei:10.2%}  {eo:10.2%}")

    # --- 7. Persist record + plots ---
    os.makedirs(args.results_dir, exist_ok=True)
    record = {
        "config_path": os.path.abspath(args.config),
        "variant": args.variant,
        "n_steps": int(n_steps),
        "t_start": t_start,
        "t_stop": t_stop,
        "loss_type": loss_type,
        "step_size": step_size,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "max_iter": max_iter,
        "tol": tol,
        "bias_factor": args.bias_factor,
        "bias_seed": args.bias_seed,
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
    }
    prefix = f"optimization_cauer_{args.variant}"
    record_path = os.path.join(args.results_dir, f"{prefix}.json")
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n  Saved: {record_path}")

    if not args.no_plot:
        plot_results(
            result["history"], t_array, y_target, y_init, y_opt,
            opt_param_names, p_true_opt, p_init_opt, p_opt,
            args.results_dir, prefix=prefix,
        )


if __name__ == "__main__":
    main()
