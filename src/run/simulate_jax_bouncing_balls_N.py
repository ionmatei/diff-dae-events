"""
Simulate the 25-ball bouncing-balls example using the IDA-based DAESolver
(same simulation pipeline as src/run/optimization_jax_bouncing_balls.py),
then plot x(t) and y(t) for every ball over the simulation interval.
"""

import os
import sys

import numpy as np
import yaml

import jax
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_platform_name", "cpu")

# Path setup -- match the layout used by optimization_jax_bouncing_balls.py
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "src"))

from src.discrete_adjoint.dae_solver import DAESolver  # noqa: E402


def load_config(config_path: str):
    """Load the run config and the DAE spec it references.

    Uses yaml.safe_load for both files so that JSON specs (subset of YAML)
    and YAML specs are both accepted.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    solver_cfg = cfg["dae_solver"]
    dae_spec_path = solver_cfg["dae_specification_file"]
    if not os.path.isabs(dae_spec_path):
        dae_spec_path = os.path.join(root_dir, dae_spec_path)
    with open(dae_spec_path, "r") as f:
        dae_data = yaml.safe_load(f)
    return cfg, dae_data, solver_cfg


def simulate(config_path: str):
    """Run the IDA-based simulation defined by `config_path` and return
    (times, trajectory, dae_data, solver_cfg, events).

    `times`      : (T,)   numpy array of stacked segment times
    `trajectory` : (T, S) numpy array of stacked segment states (S = #states)
    `events`     : list of solver `EventInfo` (each with t_event, event_idx,
                   x_pre/x_post, z_pre/z_post)
    """
    cfg, dae_data, solver_cfg = load_config(config_path)

    t_start = float(solver_cfg["start_time"])
    t_stop = float(solver_cfg["stop_time"])
    ncp = int(solver_cfg["ncp"])

    solver = DAESolver(dae_data, verbose=False, use_compiled_residual=True)
    true_p = [p["value"] for p in dae_data["parameters"]]
    solver.update_parameters(true_p)

    sol = solver.solve_augmented((t_start, t_stop), ncp=ncp)

    seg_t, seg_x = [], []
    for seg in sol.segments:
        if len(seg.t) > 0:
            seg_t.append(np.asarray(seg.t))
            seg_x.append(np.asarray(seg.x))

    times = np.concatenate(seg_t) if seg_t else np.zeros((0,))
    traj = np.concatenate(seg_x) if seg_x else np.zeros((0, len(dae_data["states"])))
    events = list(sol.events) if getattr(sol, "events", None) else []

    print(f"Simulated [{t_start}, {t_stop}] with {len(sol.segments)} segments, "
          f"{len(times)} stacked points, {traj.shape[1]} states, {len(events)} events.")

    print_event_diagnostics(sol, dae_data, last_k=20)

    return times, traj, dae_data, solver_cfg, events


def print_event_diagnostics(sol, dae_data: dict, last_k: int = 20) -> None:
    """Print the last `last_k` events recorded by the solver, with their
    when-clause comment, the absolute time, and the Δt to the previous event.

    Useful for diagnosing Zeno chattering: when the solver terminates early,
    inspecting the event tail tells you which condition(s) re-fired with no
    flow between them (very small Δt next to a long Δt).
    """
    when = dae_data.get("when") or []
    events = list(sol.events) if getattr(sol, "events", None) else []
    if not events:
        print("No events were processed.")
        return

    print(f"\nEvent diagnostics: {len(events)} events total.")
    print("Showing last "
          f"{min(last_k, len(events))} (time | dt | idx | comment):")

    tail = events[-last_k:]
    prev_t = None if len(events) == len(tail) else events[-len(tail) - 1].t_event
    for ev in tail:
        idx = int(ev.event_idx)
        comment = when[idx].get("comment", "?") if 0 <= idx < len(when) else "?"
        dt = (ev.t_event - prev_t) if prev_t is not None else float("nan")
        print(f"  t={ev.t_event:.6f}  dt={dt:.3e}  [{idx:3d}]  {comment}")
        prev_t = ev.t_event


def plot_xy_trajectories(times: np.ndarray,
                         traj: np.ndarray,
                         dae_data: dict,
                         output_path: str) -> None:
    """Plot 2D x-y trajectories for every ball over the simulation interval."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    state_names = [s["name"] for s in dae_data["states"]]
    n_balls = sum(1 for n in state_names if n.startswith("x") and n[1:].isdigit())

    p_dict = {p["name"]: p["value"] for p in dae_data["parameters"]}
    x_min = p_dict.get("x_min", None)
    x_max = p_dict.get("x_max", None)
    y_min = p_dict.get("y_min", None)
    y_max = p_dict.get("y_max", None)

    cmap = colormaps.get_cmap("hsv").resampled(n_balls)

    fig, ax = plt.subplots(figsize=(9, 9))

    for i in range(1, n_balls + 1):
        xi = state_names.index(f"x{i}")
        yi = state_names.index(f"y{i}")
        c = cmap(i - 1)
        # Trajectory
        ax.plot(traj[:, xi], traj[:, yi], "-", color=c, alpha=0.7, linewidth=0.9)
        # Start position
        ax.plot(traj[0, xi], traj[0, yi], "o", color=c, markersize=6,
                markeredgecolor="k", markeredgewidth=0.5)
        # End position
        ax.plot(traj[-1, xi], traj[-1, yi], "s", color=c, markersize=6,
                markeredgecolor="k", markeredgewidth=0.5)
        # Ball number near the start
        ax.annotate(str(i), (traj[0, xi], traj[0, yi]),
                    fontsize=7, ha="center", va="center")

    # Draw the rectangular box if known
    if None not in (x_min, x_max, y_min, y_max):
        ax.plot([x_min, x_max, x_max, x_min, x_min],
                [y_min, y_min, y_max, y_max, y_min],
                "k--", alpha=0.5, linewidth=1)
        pad = 0.05 * max(x_max - x_min, y_max - y_min)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

    t0 = float(times[0]) if len(times) else 0.0
    t1 = float(times[-1]) if len(times) else 0.0

    # Legend entries for start/end markers (one each, not one per ball)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([], [], marker="o", linestyle="", color="0.4",
               markeredgecolor="k", markersize=6, label="start"),
        Line2D([], [], marker="s", linestyle="", color="0.4",
               markeredgecolor="k", markersize=6, label="end"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8)

    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_title(f"{n_balls}-ball trajectories  (IDA / DAESolver,  t in [{t0:.2f}, {t1:.2f}] s)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def plot_xyt_trajectories(times: np.ndarray,
                          traj: np.ndarray,
                          dae_data: dict,
                          output_path: str) -> None:
    """Plot 3D (x, y, t) trajectories for every ball."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    state_names = [s["name"] for s in dae_data["states"]]
    n_balls = sum(1 for n in state_names if n.startswith("x") and n[1:].isdigit())

    p_dict = {p["name"]: p["value"] for p in dae_data["parameters"]}
    x_min = p_dict.get("x_min", None)
    x_max = p_dict.get("x_max", None)
    y_min = p_dict.get("y_min", None)
    y_max = p_dict.get("y_max", None)

    cmap = colormaps.get_cmap("hsv").resampled(n_balls)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    for i in range(1, n_balls + 1):
        xi = state_names.index(f"x{i}")
        yi = state_names.index(f"y{i}")
        c = cmap(i - 1)
        ax.plot(traj[:, xi], traj[:, yi], times,
                color=c, alpha=0.8, linewidth=0.8)
        # Start marker (at t = times[0])
        ax.scatter(traj[0, xi], traj[0, yi], times[0],
                   color=c, marker="o", s=18, edgecolors="k", linewidths=0.4)
        # End marker (at t = times[-1])
        ax.scatter(traj[-1, xi], traj[-1, yi], times[-1],
                   color=c, marker="s", s=18, edgecolors="k", linewidths=0.4)

    # Draw the rectangular box at t=times[0] and t=times[-1] for spatial reference
    if None not in (x_min, x_max, y_min, y_max) and len(times) > 0:
        for t_ref in (times[0], times[-1]):
            ax.plot([x_min, x_max, x_max, x_min, x_min],
                    [y_min, y_min, y_max, y_max, y_min],
                    [t_ref] * 5,
                    color="k", linestyle="--", alpha=0.35, linewidth=1)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    t0 = float(times[0]) if len(times) else 0.0
    t1 = float(times[-1]) if len(times) else 0.0

    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_zlabel("time [s]")
    ax.set_title(f"{n_balls}-ball trajectories in (x, y, t)  "
                 f"(IDA / DAESolver,  t in [{t0:.2f}, {t1:.2f}] s)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def plot_xy_timeseries(times: np.ndarray,
                       traj: np.ndarray,
                       dae_data: dict,
                       output_path: str) -> None:
    """Plot x_i(t) and y_i(t) for every ball as two stacked subplots."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    state_names = [s["name"] for s in dae_data["states"]]
    n_balls = sum(1 for n in state_names if n.startswith("x") and n[1:].isdigit())

    p_dict = {p["name"]: p["value"] for p in dae_data["parameters"]}
    x_min = p_dict.get("x_min", None)
    x_max = p_dict.get("x_max", None)
    y_min = p_dict.get("y_min", None)
    y_max = p_dict.get("y_max", None)

    cmap = colormaps.get_cmap("hsv").resampled(n_balls)

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for i in range(1, n_balls + 1):
        xi = state_names.index(f"x{i}")
        yi = state_names.index(f"y{i}")
        c = cmap(i - 1)
        ax_x.plot(times, traj[:, xi], "-", color=c, alpha=0.8, linewidth=0.9,
                  label=f"Ball {i}")
        ax_y.plot(times, traj[:, yi], "-", color=c, alpha=0.8, linewidth=0.9)

    if x_min is not None and x_max is not None:
        ax_x.axhline(x_min, color="k", linestyle="--", alpha=0.4)
        ax_x.axhline(x_max, color="k", linestyle="--", alpha=0.4)
    if y_min is not None and y_max is not None:
        ax_y.axhline(y_min, color="k", linestyle="--", alpha=0.4)
        ax_y.axhline(y_max, color="k", linestyle="--", alpha=0.4)

    t0 = float(times[0]) if len(times) else 0.0
    t1 = float(times[-1]) if len(times) else 0.0

    ax_x.set_ylabel("x position")
    ax_x.set_title(f"x(t) and y(t) for {n_balls} balls "
                   f"(IDA / DAESolver,  t in [{t0:.2f}, {t1:.2f}] s)")
    ax_x.grid(True, alpha=0.3)

    ax_y.set_xlabel("time [s]")
    ax_y.set_ylabel("y position")
    ax_y.grid(True, alpha=0.3)

    handles, labels = ax_x.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7, ncol=1,
               bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def main():
    config_path = os.path.join(root_dir, "config", "config_bouncing_balls_N3.yaml")
    times, traj, dae_data, _, _ = simulate(config_path)

    out_dir = os.path.join(root_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    out_path_2d = os.path.join(out_dir, "simulation_jax_bouncing_balls_N25.png")
    plot_xy_trajectories(times, traj, dae_data, out_path_2d)

    out_path_3d = os.path.join(out_dir, "simulation_jax_bouncing_balls_N25_3d.png")
    plot_xyt_trajectories(times, traj, dae_data, out_path_3d)

    out_path_ts = os.path.join(out_dir, "simulation_jax_bouncing_balls_N25_timeseries.png")
    plot_xy_timeseries(times, traj, dae_data, out_path_ts)


if __name__ == "__main__":
    main()
