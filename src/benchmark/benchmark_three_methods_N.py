"""
Benchmark THREE optimizer approaches across N=3, N=7, N=15 bouncing-balls
configurations:

    jax_da     - JAX discrete-adjoint (the user's contribution; uses the
                 padded-Jacobian unified gradient kernel in
                 src/discrete_adjoint/dae_padded_gradient.py).
    pytorch_ad - PyTorch AD-through-simulation baseline using torchdiffeq
                 (src/run/optimization_pytorch_bouncing_balls_N.py).
    jax_ad     - JAX AD-through-simulation baseline using diffrax with
                 fully JIT'd lax.scan event detection + per-segment
                 integration (src/run/optimization_jax_baseline_bouncing_balls_N.py).

Per-N results are written to:
    results/benchmark_three_N{N}.json

A summary table across all selected N values is printed at the end.
Plots are NOT generated here — run `src/benchmark/plot_three_methods_N.py`
afterwards to render figures from the saved JSON.

Animations are suppressed during the benchmark (overridden in a tmp
config copy) so wall-time measurements reflect optimizer cost, not video
encoding.
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

# Runner imports are deferred to `_load_runners()` so that `--device gpu`
# can set `JAX_PLATFORMS` before any JAX-touching module is imported.
# Set as module-level placeholders; bound in main() before run_one_case.
run_jax_da_N = None
run_pt_N = None
load_pt_config = None
run_jax_ad_N = None
load_jax_ad_config = None


def _load_runners(methods):
    """Import only the runners we actually need, after env vars are set."""
    global run_jax_da_N, run_pt_N, load_pt_config
    global run_jax_ad_N, load_jax_ad_config

    if 'jax_da' in methods:
        from src.run.optimization_jax_bouncing_balls_N import (
            run_optimization_test as _run_jax_da_N,
        )
        run_jax_da_N = _run_jax_da_N
    if 'pytorch_ad' in methods:
        from src.run.optimization_pytorch_bouncing_balls_N import (
            run_bouncing_balls_test as _run_pt_N,
            load_config as _load_pt_config,
        )
        run_pt_N = _run_pt_N
        load_pt_config = _load_pt_config
    if 'jax_ad' in methods:
        from src.run.optimization_jax_baseline_bouncing_balls_N import (
            run_bouncing_balls_test as _run_jax_ad_N,
            load_config as _load_jax_ad_config,
        )
        run_jax_ad_N = _run_jax_ad_N
        load_jax_ad_config = _load_jax_ad_config


CONFIGS = [
    ('N3',  os.path.join('config', 'config_bouncing_balls_N3.yaml')),
    ('N7',  os.path.join('config', 'config_bouncing_balls_N7.yaml')),
    ('N15', os.path.join('config', 'config_bouncing_balls_N15.yaml')),
]

# Display info per method (key in case dict, label, color).
METHOD_INFO = [
    ('jax_da',     'JAX discrete adjoint', 'tab:blue'),
    ('pytorch_ad', 'PyTorch AD',           'tab:orange'),
    ('jax_ad',     'JAX AD (diffrax)',     'tab:green'),
]


def _write_no_anim_config(orig_path: str) -> str:
    """Copy the config to a tmp file with `generate_animation: false`."""
    with open(orig_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['generate_animation'] = False
    fd, tmp_path = tempfile.mkstemp(prefix='bench3_', suffix='.yaml')
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


def _result_to_record(result: dict, total_s: float) -> dict:
    """Pack a runner's return-dict + outer wall time into a uniform record."""
    return {
        'total_time_s': float(total_s),
        'avg_iter_time_ms': float(result.get('avg_iter_time', 0.0)),
        'iterations': int(result.get('iterations', 0)),
        'converged': bool(result.get('converged', False)),
        'final_validation_loss': float(result.get('final_validation_loss', float('nan'))),
        'p_opt': result.get('p_opt', {}),
        'p_true': result.get('p_true', {}),
        'opt_param_names': result.get('opt_param_names', []),
        'prediction_error_history': result.get('prediction_error_history', []),
    }


def run_one_case(label: str, config_relpath: str,
                 selected_methods: set) -> dict:
    """Run the selected optimizers on the given config.

    `selected_methods` is a set drawn from {'jax_da','pytorch_ad','jax_ad'};
    only those entries are populated in the returned case dict.
    """
    print('\n' + '#' * 80)
    print(f'#  Benchmark case: {label}   ({config_relpath})')
    print(f'#  Methods: {", ".join(sorted(selected_methods))}')
    print('#' * 80)

    config_path = os.path.join(root_dir, config_relpath)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    tmp_config = _write_no_anim_config(config_path)

    results = {}
    try:
        if 'jax_da' in selected_methods:
            print(f"\n>>> [{label}] JAX discrete-adjoint")
            t0 = time.perf_counter()
            res = run_jax_da_N(tmp_config)
            results['jax_da'] = (res, time.perf_counter() - t0)

        if 'pytorch_ad' in selected_methods:
            print(f"\n>>> [{label}] PyTorch AD")
            cfg_pt = load_pt_config(tmp_config)
            t0 = time.perf_counter()
            res = run_pt_N(cfg_pt, os.path.dirname(tmp_config))
            results['pytorch_ad'] = (res, time.perf_counter() - t0)

        if 'jax_ad' in selected_methods:
            print(f"\n>>> [{label}] JAX AD (diffrax)")
            cfg_jax_ad = load_jax_ad_config(tmp_config)
            t0 = time.perf_counter()
            res = run_jax_ad_N(cfg_jax_ad, os.path.dirname(tmp_config))
            results['jax_ad'] = (res, time.perf_counter() - t0)

    finally:
        try:
            os.unlink(tmp_config)
        except OSError:
            pass

    # n_balls / ncp can come from any runner (they all read the same spec).
    n_balls = -1
    ncp = -1
    for res, _ in results.values():
        if n_balls < 0 and res.get('n_balls') is not None:
            n_balls = int(res['n_balls'])
        if ncp < 0 and res.get('ncp') is not None:
            ncp = int(res['ncp'])

    case_result = {
        'label': label,
        'config_path': config_relpath,
        'n_balls': n_balls,
        'ncp': ncp,
        'methods_run': sorted(results.keys()),
    }
    for key, (res, total_s) in results.items():
        case_result[key] = _result_to_record(res, total_s)
    return case_result


def save_case(case: dict, results_dir: str) -> str:
    label = case['label']
    out_path = os.path.join(results_dir, f'benchmark_three_{label}.json')
    with open(out_path, 'w') as f:
        json.dump(_serializable(case), f, indent=2)
    print(f"  Saved: {out_path}")
    return out_path


# --------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------- #
def print_summary(cases: list) -> None:
    print('\n' + '=' * 110)
    print(f"{'Benchmark summary':^110}")
    print('=' * 110)
    header = (f"{'N':>4}  {'method':<22}  {'iter':>5}  {'avg/iter [ms]':>14}  "
              f"{'total [s]':>11}  {'final loss':>14}  {'pred err':>12}  {'conv':>6}")
    print(header)
    print('-' * len(header))

    for case in cases:
        n_balls = case['n_balls']
        any_row = False
        for key, lab, _ in METHOD_INFO:
            if key not in case:
                continue
            d = case[key]
            peh = d.get('prediction_error_history', [])
            pred_err = peh[-1] if peh else float('nan')
            print(f"{n_balls:>4}  {lab:<22}  {d.get('iterations', 0):>5}  "
                  f"{d.get('avg_iter_time_ms', 0.0):>14.2f}  "
                  f"{d.get('total_time_s', 0.0):>11.2f}  "
                  f"{d.get('final_validation_loss', float('nan')):>14.6e}  "
                  f"{pred_err:>12.6e}  "
                  f"{('yes' if d.get('converged') else 'no'):>6}")
            any_row = True
        if any_row:
            print('-' * len(header))

    print('=' * 110)
    # Relative speed only makes sense when jax_da is present alongside ≥1 other.
    has_da = any('jax_da' in c for c in cases)
    if has_da:
        print("\nRelative speed (avg/iter ratios, baseline = JAX discrete adjoint):")
        print(f"  {'N':>4}  {'PT/jax_da':>12}  {'jax_ad/jax_da':>16}")
        for case in cases:
            if 'jax_da' not in case:
                continue
            n_balls = case['n_balls']
            a = float(case['jax_da'].get('avg_iter_time_ms', 0.0))
            b = float(case.get('pytorch_ad', {}).get('avg_iter_time_ms', 0.0)) if 'pytorch_ad' in case else float('nan')
            c = float(case.get('jax_ad', {}).get('avg_iter_time_ms', 0.0)) if 'jax_ad' in case else float('nan')
            r1 = (b / a) if (a > 0 and np.isfinite(b)) else float('nan')
            r2 = (c / a) if (a > 0 and np.isfinite(c)) else float('nan')
            print(f"  {n_balls:>4}  {r1:>12.2f}x  {r2:>16.2f}x")


# --------------------------------------------------------------------- #
def main():
    valid_methods = {k for k, _, _ in METHOD_INFO}
    valid_labels = {lab for lab, _ in CONFIGS}

    parser = argparse.ArgumentParser(
        description="Optimization benchmark over N (3/7/15) and method "
                    "(jax_da, pytorch_ad, jax_ad)."
    )
    parser.add_argument(
        '--results-dir', type=str,
        default=os.path.join(root_dir, 'results'),
        help='Directory to write per-N JSON results into.',
    )
    parser.add_argument(
        '--only', '--N', dest='only', type=str, default=None,
        help="Comma-separated subset of {N3,N7,N15} to run (default: all).",
    )
    parser.add_argument(
        '--methods', type=str, default=None,
        help="Comma-separated subset of {jax_da, pytorch_ad, jax_ad} "
             "to run (default: all).",
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        choices=('cpu', 'gpu', 'cuda'),
        help="JAX device for jax_da and jax_ad (default: cpu). "
             "PyTorch is unaffected.",
    )
    args = parser.parse_args()

    selected_N = (None if args.only is None
                  else set(s.strip() for s in args.only.split(',') if s.strip()))
    if selected_N is not None:
        bad = selected_N - valid_labels
        if bad:
            parser.error(f"Unknown N label(s): {sorted(bad)}. "
                         f"Valid: {sorted(valid_labels)}")

    if args.methods is None:
        selected_methods = set(valid_methods)
    else:
        selected_methods = {s.strip() for s in args.methods.split(',') if s.strip()}
        bad = selected_methods - valid_methods
        if bad:
            parser.error(f"Unknown method(s): {sorted(bad)}. "
                         f"Valid: {sorted(valid_methods)}")
    if not selected_methods:
        parser.error("--methods must select at least one method.")

    # Set JAX device BEFORE importing the runners. The runners default to
    # CPU if JAX_PLATFORMS is unset; setting it here lets us route jax_da
    # and jax_ad to GPU. Normalize "gpu" -> "cuda" since on this stack
    # JAX maps both to the CUDA backend.
    if args.device in ('gpu', 'cuda'):
        os.environ['JAX_PLATFORMS'] = 'cuda'
    # else: leave unset → runners pin CPU (existing behavior).

    _load_runners(selected_methods)

    os.makedirs(args.results_dir, exist_ok=True)

    print('=' * 80)
    title = (f"Benchmark  N={'/'.join(sorted(selected_N or valid_labels))}  "
             f"methods={','.join(sorted(selected_methods))}  "
             f"device={args.device}")
    print(f"{title:^80}")
    print('=' * 80)

    # Confirm the JAX backend that the runners actually got — useful for
    # catching cases where CUDA was requested but JAX silently fell back
    # to CPU (driver mismatch, missing CUDA libs, etc.).
    if {'jax_da', 'jax_ad'} & selected_methods:
        try:
            import jax  # already imported transitively, this is cheap
            print(f"  JAX backend: {jax.default_backend()}  "
                  f"devices: {jax.devices()}")
            if args.device in ('gpu', 'cuda') and jax.default_backend() == 'cpu':
                print("  WARNING: --device gpu requested but JAX is on CPU. "
                      "Check CUDA install / drivers.")
        except Exception as e:
            print(f"  [could not query jax backend: {e}]")

    bench_t0 = time.perf_counter()
    cases = []
    for label, cfg_path in CONFIGS:
        if selected_N is not None and label not in selected_N:
            continue
        case = run_one_case(label, cfg_path, selected_methods)
        save_case(case, args.results_dir)
        cases.append(case)
    bench_total = time.perf_counter() - bench_t0

    print_summary(cases)
    print(f"\nTotal benchmark wall time: {bench_total:.2f} s")


if __name__ == "__main__":
    main()
