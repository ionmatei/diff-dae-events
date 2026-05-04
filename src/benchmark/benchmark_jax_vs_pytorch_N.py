"""
Benchmark JAX vs PyTorch optimization across N=3, N=7, N=15 bouncing-balls
configurations. Captures wall time, average iteration time, iteration
count, convergence flag, and the final validation loss (PyTorch simulator
evaluated for both methods, so the comparison is apples-to-apples).

Per-N results are written to:
    results/benchmark_N{N}.json

A summary table across all three N values is printed at the end.

Animations are suppressed during the benchmark (overridden in a tmp config
copy) so wall-time measurements reflect optimizer cost, not video encoding.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile

import numpy as np
import yaml

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from src.run.optimization_jax_bouncing_balls_N import (  # noqa: E402
    run_optimization_test as run_jax_N,
)
from src.run.optimization_pytorch_bouncing_balls_N import (  # noqa: E402
    run_bouncing_balls_test as run_pt_N,
    load_config as load_pt_config,
)


CONFIGS = [
    ('N3',  os.path.join('config', 'config_bouncing_balls_N3.yaml')),
    ('N7',  os.path.join('config', 'config_bouncing_balls_N7.yaml')),
    ('N15', os.path.join('config', 'config_bouncing_balls_N15.yaml')),
]


def _write_no_anim_config(orig_path: str) -> str:
    """Copy the config to a tmp file with `generate_animation: false`.

    Returns the tmp file path. The caller is responsible for deleting it
    when done (or just letting /tmp clean it up).
    """
    with open(orig_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['generate_animation'] = False
    fd, tmp_path = tempfile.mkstemp(prefix='bench_', suffix='.yaml')
    os.close(fd)
    with open(tmp_path, 'w') as f:
        yaml.safe_dump(cfg, f)
    return tmp_path


def _serializable(obj):
    """Coerce numpy / tensor / dict-of-numpy values to JSON-friendly types."""
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def run_one_case(label: str, config_relpath: str) -> dict:
    """Run JAX then PyTorch optimization on the given config."""
    print('\n' + '#' * 80)
    print(f'#  Benchmark case: {label}   ({config_relpath})')
    print('#' * 80)

    config_path = os.path.join(root_dir, config_relpath)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    tmp_config = _write_no_anim_config(config_path)

    try:
        # ------------------------------------------------------------------ #
        # JAX optimization (run script handles its own setup + plotting)
        # ------------------------------------------------------------------ #
        print(f"\n>>> [{label}] JAX optimization")
        t0 = time.perf_counter()
        jax_result = run_jax_N(tmp_config)
        jax_total_s = time.perf_counter() - t0

        # ------------------------------------------------------------------ #
        # PyTorch optimization
        # ------------------------------------------------------------------ #
        print(f"\n>>> [{label}] PyTorch optimization")
        cfg = load_pt_config(tmp_config)
        t0 = time.perf_counter()
        pt_result = run_pt_N(cfg, os.path.dirname(tmp_config))
        pt_total_s = time.perf_counter() - t0
    finally:
        try:
            os.unlink(tmp_config)
        except OSError:
            pass

    case_result = {
        'label': label,
        'config_path': config_relpath,
        'n_balls': int(jax_result.get('n_balls', pt_result.get('n_balls', -1))),
        'ncp': int(jax_result['ncp']),
        'jax': {
            'total_time_s': jax_total_s,
            'avg_iter_time_ms': float(jax_result['avg_iter_time']),
            'iterations': int(jax_result['iterations']),
            'converged': bool(jax_result['converged']),
            'final_validation_loss': float(jax_result['final_validation_loss']),
            'p_opt': jax_result.get('p_opt', {}),
            'p_true': jax_result.get('p_true', {}),
            'opt_param_names': jax_result.get('opt_param_names', []),
            'prediction_error_history': jax_result.get('prediction_error_history', []),
        },
        'pytorch': {
            'total_time_s': pt_total_s,
            'avg_iter_time_ms': float(pt_result['avg_iter_time']),
            'iterations': int(pt_result['iterations']),
            'converged': bool(pt_result['converged']),
            'final_validation_loss': float(pt_result['final_validation_loss']),
            'p_opt': pt_result.get('p_opt', {}),
            'p_true': pt_result.get('p_true', {}),
            'opt_param_names': pt_result.get('opt_param_names', []),
            'prediction_error_history': pt_result.get('prediction_error_history', []),
        },
    }
    return case_result


def save_case(case: dict, results_dir: str) -> str:
    label = case['label']
    out_path = os.path.join(results_dir, f'benchmark_{label}.json')
    with open(out_path, 'w') as f:
        json.dump(_serializable(case), f, indent=2)
    print(f"  Saved: {out_path}")
    return out_path


def save_prediction_error_plot(case: dict, results_dir: str) -> str:
    """Plot ‖p_iter − p_true‖ over iterations, JAX vs PyTorch."""
    import matplotlib.pyplot as plt

    label = case['label']
    n_balls = case['n_balls']
    jax_err = case['jax'].get('prediction_error_history', []) or []
    pt_err = case['pytorch'].get('prediction_error_history', []) or []

    fig, ax = plt.subplots(figsize=(9, 5))
    if jax_err:
        ax.plot(range(1, len(jax_err) + 1), jax_err, '-',
                label='JAX', color='tab:blue', linewidth=1.5)
    if pt_err:
        ax.plot(range(1, len(pt_err) + 1), pt_err, '-',
                label='PyTorch', color='tab:orange', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Prediction error  $\|p - p_{\mathrm{true}}\|_2$')
    ax.set_title(f'Prediction error over iterations  ({n_balls} balls)')
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    if jax_err or pt_err:
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(results_dir, f'benchmark_{label}_prediction_error.png')
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")
    return out_path


def print_summary(cases: list) -> None:
    print('\n' + '=' * 95)
    print(f"{'Benchmark summary':^95}")
    print('=' * 95)
    header = (f"{'N':>4} {'method':<8} {'iter':>5} {'avg/iter [ms]':>14} "
              f"{'total [s]':>11} {'final loss':>14} {'pred err':>12} {'conv':>6}")
    print(header)
    print('-' * len(header))

    for case in cases:
        n_balls = case['n_balls']
        for method in ('jax', 'pytorch'):
            d = case[method]
            peh = d.get('prediction_error_history', [])
            pred_err = peh[-1] if peh else float('nan')
            print(f"{n_balls:>4} {method:<8} {d['iterations']:>5} "
                  f"{d['avg_iter_time_ms']:>14.2f} "
                  f"{d['total_time_s']:>11.2f} "
                  f"{d['final_validation_loss']:>14.6e} "
                  f"{pred_err:>12.6e} "
                  f"{('yes' if d['converged'] else 'no'):>6}")

    print('=' * 95)

    print("\nRelative speed (PyTorch / JAX):")
    print(f"  {'N':>4} {'iter ratio':>14} {'total ratio':>14}")
    for case in cases:
        n_balls = case['n_balls']
        jax_d = case['jax']; pt_d = case['pytorch']
        ir = pt_d['avg_iter_time_ms'] / jax_d['avg_iter_time_ms'] if jax_d['avg_iter_time_ms'] > 0 else float('nan')
        tr = pt_d['total_time_s'] / jax_d['total_time_s'] if jax_d['total_time_s'] > 0 else float('nan')
        print(f"  {n_balls:>4} {ir:>14.2f}x {tr:>14.2f}x")


def main():
    parser = argparse.ArgumentParser(description="JAX vs PyTorch optimization benchmark over N=3/7/15")
    parser.add_argument(
        '--results-dir', type=str,
        default=os.path.join(root_dir, 'results'),
        help='Directory to write per-N JSON results into.',
    )
    parser.add_argument(
        '--only', type=str, default=None,
        help="Comma-separated subset of {N3,N7,N15} to run (default: all).",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    selected = None if args.only is None else set(s.strip() for s in args.only.split(','))

    print("=" * 80)
    print(f"{'Benchmark: JAX vs PyTorch optimization (N=3, 7, 15)':^80}")
    print("=" * 80)

    bench_t0 = time.perf_counter()
    cases = []
    for label, cfg_path in CONFIGS:
        if selected is not None and label not in selected:
            continue
        case = run_one_case(label, cfg_path)
        save_case(case, args.results_dir)
        save_prediction_error_plot(case, args.results_dir)
        cases.append(case)
    bench_total = time.perf_counter() - bench_t0

    print_summary(cases)
    print(f"\nTotal benchmark wall time: {bench_total:.2f} s")


if __name__ == "__main__":
    main()
