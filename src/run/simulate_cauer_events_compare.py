"""
Side-by-side simulation of the Cauer-events DAE under two integrators:

  * IDA via `src/discrete_adjoint/dae_solver.py` (Sundials, native event
    handling).
  * diffrax via `src/dae/dae_optimizer_events.py` (Bisection-rooted
    composite event + segmented integration).

Generates one figure per differential state with both trajectories
overlaid, plus a residual plot. Useful for verifying that the JAX-AD
event detection lands on the same event times and produces matching
state trajectories as the IDA reference.
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
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
if not os.environ.get("JAX_PLATFORMS"):
    jax_config.update("jax_platform_name", "cpu")

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from src.discrete_adjoint.dae_solver import DAESolver  # noqa: E402
from src.dae.dae_optimizer_events import DAEOptimizerJaxADEvents  # noqa: E402


def load_spec(spec_path: str) -> dict:
    with open(spec_path, "r") as f:
        if spec_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


def stack_segments(sol):
    """Concatenate (t, x) from `AugmentedSolution.segments`."""
    ts, xs = [], []
    for seg in sol.segments:
        if len(seg.t) > 0:
            ts.append(np.asarray(seg.t))
            xs.append(np.asarray(seg.x))
    if not ts:
        return np.zeros(0), np.zeros((0, 0))
    return np.concatenate(ts), np.concatenate(xs, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Compare IDA and diffrax simulation of Cauer-events.",
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(root_dir, "config", "config_cauer_events.yaml"),
    )
    parser.add_argument(
        "--spec", type=str, default=None,
        help="Override the DAE spec from --config.",
    )
    parser.add_argument(
        "--diffrax-solver", default="Tsit5",
        choices=("Tsit5", "Dopri5", "Dopri8", "Heun"),
    )
    parser.add_argument(
        "--rtol", type=float, default=1.0e-6,
        help="Tolerance used by both integrators.",
    )
    parser.add_argument(
        "--atol", type=float, default=1.0e-6,
        help="Tolerance used by both integrators.",
    )
    parser.add_argument(
        "--max-segments", type=int, default=80,
        help="diffrax events: static upper bound on segments.",
    )
    parser.add_argument(
        "--ncp-override", type=int, default=None,
        help="Override the number of communication points (default from YAML).",
    )
    parser.add_argument(
        "--stop-time-override", type=float, default=None,
        help="Override stop_time (default from YAML).",
    )
    parser.add_argument(
        "--results-dir", type=str,
        default=os.path.join(root_dir, "results"),
    )
    parser.add_argument(
        "--prefix", type=str, default="cauer_events_compare",
        help="Output file prefix (under --results-dir).",
    )
    args = parser.parse_args()

    # --- 1. Load config + spec ---
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    spec_path = args.spec or cfg["dae_solver"]["dae_specification_file"]
    if not os.path.isabs(spec_path):
        spec_path = os.path.join(root_dir, spec_path)
    dae_data = load_spec(spec_path)

    t_start = float(cfg["dae_solver"]["start_time"])
    t_stop = float(args.stop_time_override or cfg["dae_solver"]["stop_time"])
    ncp = int(args.ncp_override or cfg["dae_solver"]["ncp"])
    t_array = np.linspace(t_start, t_stop, ncp + 1)
    state_names = [s["name"] for s in dae_data["states"]]

    print("=" * 70)
    print("Cauer-events: IDA vs diffrax simulation comparison")
    print("=" * 70)
    print(f"  spec: {spec_path}")
    print(f"  t_span: [{t_start}, {t_stop}], ncp = {ncp}")
    print(f"  rtol={args.rtol:g}  atol={args.atol:g}")
    print(f"  diffrax solver: {args.diffrax_solver}  "
          f"max_segments: {args.max_segments}")
    print(f"  states: {state_names}")
    when_clauses = dae_data.get("when") or []
    print(f"  when clauses ({len(when_clauses)}):")
    for c in when_clauses:
        print(f"    cond={c.get('condition')!r}  reinit={c.get('reinit')!r}")

    # --- 2. IDA reference ---
    print("\n[IDA] solve_augmented...")
    t0 = time.time()
    solver = DAESolver(dae_data, verbose=False, use_compiled_residual=True)
    sol_ida = solver.solve_augmented((t_start, t_stop), ncp=ncp)
    t_ida = time.time() - t0
    t_seg, x_seg = stack_segments(sol_ida)
    n_events_ida = len(sol_ida.events)
    n_segs_ida = len(sol_ida.segments)
    print(f"  done ({t_ida:.2f}s)  segments: {n_segs_ida}  events: {n_events_ida}")
    if sol_ida.events:
        ev_t_ida = np.asarray([e.t_event for e in sol_ida.events])
        print(f"  IDA event times (first 5): "
              f"{[f'{t:.4f}' for t in ev_t_ida[:5]]}")

    # Resample IDA onto the uniform target grid so we can subtract.
    n_states = x_seg.shape[1]
    x_ida = np.empty((len(t_array), n_states), dtype=float)
    for j in range(n_states):
        x_ida[:, j] = np.interp(t_array, t_seg, x_seg[:, j])

    # --- 3. diffrax events ---
    print("\n[diffrax events] simulate...")
    optimizer = DAEOptimizerJaxADEvents(
        dae_data,
        optimize_params=[dae_data["parameters"][0]["name"]],  # any single param
        loss_type="mean",
        diffrax_solver=args.diffrax_solver,
        rtol=args.rtol, atol=args.atol,
        max_segments=args.max_segments,
    )
    t0 = time.time()
    sim_dx = optimizer.simulate(t_array)
    t_dx = time.time() - t0
    print(f"  done ({t_dx:.2f}s)  y shape: {sim_dx['y'].shape}")
    x_dx = np.asarray(sim_dx["x"]).T  # (n_steps, n_states)

    # --- 4. Compare ---
    diff = x_dx - x_ida
    abs_diff = np.abs(diff)
    print("\nPer-state diff (IDA vs diffrax):")
    print(f"  {'state':12s}  {'max |diff|':>11s}  {'rms diff':>11s}  "
          f"{'IDA range':>20s}  {'dx range':>20s}")
    for j, name in enumerate(state_names):
        print(f"  {name:12s}  {abs_diff[:, j].max():11.4e}  "
              f"{np.sqrt(np.mean(diff[:, j] ** 2)):11.4e}  "
              f"[{x_ida[:, j].min():+.3f}, {x_ida[:, j].max():+.3f}]"
              f"   [{x_dx[:, j].min():+.3f}, {x_dx[:, j].max():+.3f}]")
    print(f"  total max |diff|: {abs_diff.max():.4e}")
    print(f"  total rms diff:   {np.sqrt(np.mean(diff ** 2)):.4e}")

    # --- 5. Plots ---
    os.makedirs(args.results_dir, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        n_states + 1, 1, figsize=(10, 1.6 * (n_states + 1) + 0.6),
        sharex=True,
    )
    if n_states == 0:
        axes = [axes]

    # Per-state overlays
    for j, name in enumerate(state_names):
        ax = axes[j]
        ax.plot(t_array, x_ida[:, j], "-", color="tab:blue",
                linewidth=1.4, label="IDA" if j == 0 else None)
        ax.plot(t_array, x_dx[:, j], "--", color="tab:orange",
                linewidth=1.0, label="diffrax events" if j == 0 else None)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(
                f"Cauer-events: IDA vs diffrax  "
                f"(IDA events: {n_events_ida}, t∈[{t_start},{t_stop}])"
            )

    # Residual on a log scale
    ax = axes[-1]
    res = np.linalg.norm(diff, axis=1)
    ax.semilogy(t_array, np.maximum(res, 1e-16), "-", color="tab:red",
                linewidth=1.0, label="‖x_diffrax − x_IDA‖₂")
    ax.set_ylabel("|residual|")
    ax.set_xlabel("time")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out = os.path.join(args.results_dir, f"{args.prefix}.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Saved plot: {out}")


if __name__ == "__main__":
    main()
