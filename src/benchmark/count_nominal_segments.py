"""
Run a nominal-parameter forward simulation with each of the three
methods (IDA truth, PyTorch AD baseline, JAX AD baseline) for one or
more bouncing-balls configs, and report the resulting segment / event
counts.

Useful for sanity-checking the JAX baseline's `max_segments` budget: if
the truth has K segments, both the JAX-AD baseline and the
discrete-adjoint runner need `max_segments >= K + a small safety
margin`. Otherwise events get silently truncated and the forward
trajectory drifts.

Usage:
    .venv/bin/python -m src.benchmark.count_nominal_segments
    .venv/bin/python -m src.benchmark.count_nominal_segments --config config/config_bouncing_balls_N7.yaml
    .venv/bin/python -m src.benchmark.count_nominal_segments --N 3,7,15
    .venv/bin/python -m src.benchmark.count_nominal_segments --method ida,jax
    .venv/bin/python -m src.benchmark.count_nominal_segments --N 7 --show-events 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

import numpy as np
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from src.discrete_adjoint.dae_solver import DAESolver  # noqa: E402


DEFAULT_CONFIGS = {
    '3':  'config/config_bouncing_balls_N3.yaml',
    '7':  'config/config_bouncing_balls_N7.yaml',
    '15': 'config/config_bouncing_balls_N15.yaml',
}

VALID_METHODS = ('ida', 'pytorch', 'jax')


# --------------------------------------------------------------------- #
# Spec / config loading
# --------------------------------------------------------------------- #
def _load(cfg_path: str):
    cfg_path_abs = cfg_path if os.path.isabs(cfg_path) else os.path.join(ROOT, cfg_path)
    if not os.path.exists(cfg_path_abs):
        raise FileNotFoundError(f"Config not found: {cfg_path_abs}")
    with open(cfg_path_abs, 'r') as f:
        cfg = yaml.safe_load(f)
    solver_cfg = cfg['dae_solver']
    spec_path = solver_cfg['dae_specification_file']
    if not os.path.isabs(spec_path):
        spec_path = os.path.join(ROOT, spec_path)
    with open(spec_path, 'r') as f:
        dae_data = yaml.safe_load(f)
    return cfg_path_abs, cfg, solver_cfg, dae_data


def _extract_common(dae_data, solver_cfg):
    n_balls = sum(
        1 for s in dae_data['states']
        if s['name'].startswith('x') and s['name'][1:].isdigit()
    )
    p_true = {p['name']: p['value'] for p in dae_data['parameters']}
    initial_state = [float(s['start']) for s in dae_data['states']]
    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    ncp = int(solver_cfg.get('ncp', 500))
    return n_balls, p_true, initial_state, t_span, ncp


# --------------------------------------------------------------------- #
# Per-method counters
# --------------------------------------------------------------------- #
def _count_ida(dae_data, t_span, ncp, show_events=0):
    solver = DAESolver(dae_data, verbose=False, use_compiled_residual=True)
    solver.update_parameters([p['value'] for p in dae_data['parameters']])
    t0 = time.perf_counter()
    sol = solver.solve_augmented(t_span, ncp=ncp)
    elapsed = time.perf_counter() - t0
    seg_lens = [len(s.t) for s in sol.segments]
    if show_events > 0 and sol.events:
        print(f'    first {min(show_events, len(sol.events))} IDA events:')
        for ev in sol.events[:show_events]:
            print(f'      t={ev.t_event:.6f}  idx={ev.event_idx}')

    if sol.segments:
        all_t = [np.asarray(s.t, dtype=float) for s in sol.segments]
        all_x = [np.asarray(s.x, dtype=float) for s in sol.segments]
        times = np.concatenate(all_t)
        states = np.concatenate(all_x, axis=0)
    else:
        times = np.array([], dtype=float)
        states = np.zeros((0, 0), dtype=float)

    return {
        'segments': len(sol.segments),
        'events': len(sol.events),
        'total_pts': sum(seg_lens),
        'pts_per_seg_min': min(seg_lens) if seg_lens else 0,
        'pts_per_seg_med': sorted(seg_lens)[len(seg_lens)//2] if seg_lens else 0,
        'pts_per_seg_max': max(seg_lens) if seg_lens else 0,
        'elapsed_s': elapsed,
        'times': times,
        'states': states,
    }


def _count_pytorch(n_balls, p_true, initial_state, t_span, ncp, show_events=0):
    """Count events the PyTorch baseline detects at nominal params."""
    import torch
    torch.set_default_dtype(torch.float64)
    from src.pytorch.bouncing_balls_n import BouncingBallsNModel  # noqa: E402

    model = BouncingBallsNModel(
        N=n_balls,
        g=float(p_true['g']),
        e_g=float(p_true['e_g']),
        e_b=float(p_true['e_b']),
        d_sq=float(p_true['d_sq']),
        x_min=float(p_true['x_min']), x_max=float(p_true['x_max']),
        y_min=float(p_true['y_min']), y_max=float(p_true['y_max']),
        initial_state=initial_state,
        adjoint=False,
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        times, trajectory, event_list = model.simulate_fixed_grid(
            t_end=float(t_span[1]),
            n_points=ncp,
            max_events=400,
        )
    elapsed = time.perf_counter() - t0
    if show_events > 0 and event_list:
        print(f'    first {min(show_events, len(event_list))} PyTorch events:')
        for ev_t, ev_idx in event_list[:show_events]:
            print(f'      t={float(ev_t):.6f}  idx={int(ev_idx)}')
    return {
        # n_segments = events + 1 trailing-to-t_end segment.
        'segments': len(event_list) + 1,
        'events': len(event_list),
        'total_pts': int(times.shape[0]) if hasattr(times, 'shape') else len(times),
        'elapsed_s': elapsed,
        'times': np.asarray(times.detach().cpu().numpy(), dtype=float),
        'states': np.asarray(trajectory.detach().cpu().numpy(), dtype=float),
    }


def _count_jax(n_balls, p_true, initial_state, t_span, ncp, show_events=0):
    """Count events the JAX AD baseline detects at nominal params.

    Uses a non-JIT'd Python loop that mirrors `_simulate_at_targets_jit`'s
    `detect_step` scan: same diffrax solver, controller, composite event,
    state_update — just unrolled so we can observe each event without
    going through a fixed-size lax.scan.
    """
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import diffrax
    from src.jax_baseline.bouncing_balls_n_jax import (  # noqa: E402
        BouncingBallsNModelJAX, BouncingBallsParams,
    )

    model = BouncingBallsNModelJAX(
        N=n_balls,
        g=float(p_true['g']),
        e_g=float(p_true['e_g']),
        e_b=float(p_true['e_b']),
        d_sq=float(p_true['d_sq']),
        x_min=float(p_true['x_min']), x_max=float(p_true['x_max']),
        y_min=float(p_true['y_min']), y_max=float(p_true['y_max']),
        initial_state=initial_state,
        max_segments=256,           # generous; eager loop ignores anyway
        max_pts_per_seg=64,
    )
    params = BouncingBallsParams(
        g=jnp.asarray(float(p_true['g']), dtype=jnp.float64),
        e_g=jnp.asarray(float(p_true['e_g']), dtype=jnp.float64),
        e_b=jnp.asarray(float(p_true['e_b']), dtype=jnp.float64),
    )

    EPS_ACTIVE = 1e-9
    LARGE = jnp.asarray(1e30, dtype=jnp.float64)
    t_end = jnp.asarray(float(t_span[1]), dtype=jnp.float64)
    current_t = jnp.asarray(float(t_span[0]), dtype=jnp.float64)
    current_state = model.initial_state
    events = []
    seg_starts = [(float(current_t), np.asarray(current_state))]
    HARD_LIMIT = 500  # safety ceiling; truth never exceeds this for our cases

    t0 = time.perf_counter()
    for _ in range(HARD_LIMIT):
        if float(current_t) >= float(t_end) - 1e-12:
            break
        events_at_start = model.event_fn(current_t, current_state, params)
        active_mask = events_at_start > EPS_ACTIVE
        dt0 = max((float(t_end) - float(current_t)) * 1e-3, 1e-12)
        sol = diffrax.diffeqsolve(
            model._term, model._solver,
            t0=current_t, t1=t_end, dt0=jnp.asarray(dt0, dtype=jnp.float64),
            y0=current_state,
            args=(params, active_mask),
            event=model._composite_event,
            stepsize_controller=model._controller(),
            max_steps=4096,
        )
        if not bool(sol.event_mask):
            break
        et = float(sol.ts[-1])
        if et >= float(t_end) - 1e-12:
            break
        state_at_event = sol.ys[-1]
        ev_vals = model.event_fn(jnp.asarray(et, dtype=jnp.float64),
                                 state_at_event, params)
        ev_vals_masked = jnp.where(active_mask, ev_vals, LARGE)
        ev_idx = int(jnp.argmin(ev_vals_masked))
        events.append((et, ev_idx))
        current_state = model.state_update(state_at_event, ev_idx, params)
        current_t = jnp.asarray(et, dtype=jnp.float64)
        seg_starts.append((float(current_t), np.asarray(current_state)))

    # Densely re-integrate each segment using the same controller, so we
    # can plot a smooth trajectory comparable to IDA / PyTorch.
    seg_bounds = [(seg_starts[i][0],
                   events[i][0] if i < len(events) else float(t_end),
                   seg_starts[i][1])
                  for i in range(len(seg_starts))]
    total_duration = float(t_end) - float(t_span[0])
    all_t_chunks, all_x_chunks = [], []
    for t_s, t_e, y0 in seg_bounds:
        dur = max(t_e - t_s, 0.0)
        n_seg = max(2, int(ncp * dur / total_duration)) if total_duration > 0 else 2
        ts = jnp.linspace(jnp.asarray(t_s, dtype=jnp.float64),
                          jnp.asarray(t_e, dtype=jnp.float64), n_seg)
        sol = diffrax.diffeqsolve(
            model._term, model._solver,
            t0=jnp.asarray(t_s, dtype=jnp.float64),
            t1=jnp.asarray(t_e, dtype=jnp.float64),
            dt0=jnp.asarray(max(dur * 1e-3, 1e-12), dtype=jnp.float64),
            y0=jnp.asarray(y0, dtype=jnp.float64),
            args=(params, model._all_active_mask),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=model._controller(),
            max_steps=4096,
        )
        all_t_chunks.append(np.asarray(sol.ts, dtype=float))
        all_x_chunks.append(np.asarray(sol.ys, dtype=float))
    times = (np.concatenate(all_t_chunks) if all_t_chunks
             else np.array([], dtype=float))
    states = (np.concatenate(all_x_chunks, axis=0) if all_x_chunks
              else np.zeros((0, 0), dtype=float))
    elapsed = time.perf_counter() - t0

    if show_events > 0 and events:
        print(f'    first {min(show_events, len(events))} JAX events:')
        for et, ei in events[:show_events]:
            print(f'      t={et:.6f}  idx={ei}')
    return {
        'segments': len(events) + 1,
        'events': len(events),
        'elapsed_s': elapsed,
        'times': times,
        'states': states,
    }


# --------------------------------------------------------------------- #
# Per-config report
# --------------------------------------------------------------------- #
def report(cfg_path: str, methods, show_events: int = 0) -> dict:
    cfg_path_abs, cfg, solver_cfg, dae_data = _load(cfg_path)
    n_balls, p_true, initial_state, t_span, ncp = _extract_common(
        dae_data, solver_cfg
    )

    print(f'{os.path.basename(cfg_path_abs)}')
    print(f'  N balls   : {n_balls}')
    print(f'  t_span    : {t_span}')
    print(f'  ncp       : {ncp}')

    row = {
        'config': os.path.basename(cfg_path_abs),
        'n_balls': n_balls,
        't_span': t_span,
        'ncp': ncp,
        'box': (float(p_true['x_min']), float(p_true['x_max']),
                float(p_true['y_min']), float(p_true['y_max'])),
    }

    if 'ida' in methods:
        print(f'  -- IDA (truth) --')
        r = _count_ida(dae_data, t_span, ncp, show_events=show_events)
        print(f'    segments: {r["segments"]}   events: {r["events"]}   '
              f'total pts: {r["total_pts"]}   '
              f'pts/seg min/med/max: '
              f'{r["pts_per_seg_min"]}/{r["pts_per_seg_med"]}/{r["pts_per_seg_max"]}'
              f'   ({r["elapsed_s"]:.2f}s)')
        row['ida'] = r

    if 'pytorch' in methods:
        print(f'  -- PyTorch AD --')
        r = _count_pytorch(n_balls, p_true, initial_state, t_span, ncp,
                           show_events=show_events)
        print(f'    segments: {r["segments"]}   events: {r["events"]}   '
              f'total pts: {r["total_pts"]}   ({r["elapsed_s"]:.2f}s)')
        row['pytorch'] = r

    if 'jax' in methods:
        print(f'  -- JAX AD (diffrax) --')
        r = _count_jax(n_balls, p_true, initial_state, t_span, ncp,
                       show_events=show_events)
        print(f'    segments: {r["segments"]}   events: {r["events"]}'
              f'   ({r["elapsed_s"]:.2f}s)')
        row['jax'] = r

    print()
    return row


# --------------------------------------------------------------------- #
# Trajectory plot (rows = N, cols = method)
# --------------------------------------------------------------------- #
_METHOD_TITLE = {'ida': 'IDA (truth)', 'pytorch': 'PyTorch AD',
                 'jax': 'JAX AD (diffrax)'}


def _plot_trajectories(rows, methods, out_path, p_true_box):
    """Render xy-trajectory grid: rows = N case, cols = method."""
    import matplotlib.pyplot as plt

    n_rows = len(rows)
    n_cols = len(methods)
    if n_rows == 0 or n_cols == 0:
        return None

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )

    for i, row in enumerate(rows):
        n_balls = row['n_balls']
        cmap = plt.get_cmap('tab20' if n_balls > 10 else 'tab10')
        colors = [cmap(k % cmap.N) for k in range(n_balls)]
        x_min, x_max, y_min, y_max = p_true_box[i]
        for j, method in enumerate(methods):
            ax = axes[i][j]
            sub = row.get(method, {})
            states = sub.get('states')
            if states is not None and states.size:
                # state layout: [x_i, y_i, vx_i, vy_i] per ball.
                for k in range(n_balls):
                    ax.plot(
                        states[:, 4 * k + 0],
                        states[:, 4 * k + 1],
                        '-', color=colors[k], linewidth=0.8,
                        alpha=0.85,
                    )
                    # Mark start and end of each ball's path.
                    ax.plot(states[0, 4 * k + 0], states[0, 4 * k + 1],
                            'o', color=colors[k], markersize=3)
                    ax.plot(states[-1, 4 * k + 0], states[-1, 4 * k + 1],
                            's', color=colors[k], markersize=3)
            # Box outline.
            ax.plot([x_min, x_max, x_max, x_min, x_min],
                    [y_min, y_min, y_max, y_max, y_min],
                    '-', color='0.6', linewidth=0.7)
            ax.set_xlim(x_min - 0.05 * (x_max - x_min),
                        x_max + 0.05 * (x_max - x_min))
            ax.set_ylim(y_min - 0.05 * (y_max - y_min),
                        y_max + 0.05 * (y_max - y_min))
            ax.set_aspect('equal', adjustable='box')
            seg_str = (f"  segs={sub.get('segments', '?')}"
                       f"  events={sub.get('events', '?')}")
            if i == 0:
                ax.set_title(f"{_METHOD_TITLE.get(method, method)}\n"
                             f"N={n_balls}{seg_str}", fontsize=10)
            else:
                ax.set_title(f"N={n_balls}{seg_str}", fontsize=10)
            if j == 0:
                ax.set_ylabel('y')
            if i == n_rows - 1:
                ax.set_xlabel('x')

    fig.suptitle("Nominal-parameter forward trajectories (xy)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------- #
def _print_summary(rows: list, methods: list) -> None:
    if not rows:
        return
    print('=' * 90)
    print(f"{'Nominal forward-simulation segment counts':^90}")
    print('=' * 90)
    header = f"{'config':<38}  {'N':>3}"
    for m in methods:
        header += f"  {m + ' segs':>11}  {m + ' events':>13}"
    print(header)
    print('-' * len(header))
    for r in rows:
        line = f"{r['config']:<38}  {r['n_balls']:>3}"
        for m in methods:
            sub = r.get(m, {})
            line += f"  {sub.get('segments', '-'):>11}  {sub.get('events', '-'):>13}"
        print(line)


# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Count IDA / PyTorch-AD / JAX-AD segments at nominal "
                    "params for bouncing-balls configs."
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help="Path to a single YAML config (relative to repo root or absolute).",
    )
    parser.add_argument(
        '--N', type=str, default=None,
        help="Comma-separated subset of {3,7,15}. Default: all three.",
    )
    parser.add_argument(
        '--method', type=str, default=None,
        help=f"Comma-separated subset of {VALID_METHODS}. Default: all.",
    )
    parser.add_argument(
        '--show-events', type=int, default=0,
        help="If > 0, list the first K events of each method's trajectory.",
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help="Skip writing the trajectory comparison figure.",
    )
    parser.add_argument(
        '--results-dir', type=str,
        default=os.path.join(ROOT, 'results'),
        help="Directory to save the trajectory plot into.",
    )
    args = parser.parse_args()

    configs: List[str] = []
    if args.config is not None:
        configs.append(args.config)
    else:
        keys = (sorted(DEFAULT_CONFIGS.keys(), key=int) if args.N is None
                else [s.strip() for s in args.N.split(',') if s.strip()])
        for k in keys:
            if k not in DEFAULT_CONFIGS:
                parser.error(f"Unknown N={k}. Valid: {sorted(DEFAULT_CONFIGS.keys(), key=int)}")
            configs.append(DEFAULT_CONFIGS[k])

    if args.method is None:
        methods = list(VALID_METHODS)
    else:
        methods = [s.strip() for s in args.method.split(',') if s.strip()]
        bad = set(methods) - set(VALID_METHODS)
        if bad:
            parser.error(f"Unknown method(s): {sorted(bad)}. Valid: {VALID_METHODS}")
    if not methods:
        parser.error("--method must select at least one method.")

    rows = [report(c, methods, args.show_events) for c in configs]

    if len(rows) > 1:
        _print_summary(rows, methods)

    if not args.no_plot and rows and methods:
        os.makedirs(args.results_dir, exist_ok=True)
        plot_path = os.path.join(
            args.results_dir, 'nominal_segments_trajectories.png'
        )
        boxes = [r['box'] for r in rows]
        out = _plot_trajectories(rows, methods, plot_path, boxes)
        if out:
            print(f"\nTrajectory plot saved: {out}")


if __name__ == '__main__':
    main()
