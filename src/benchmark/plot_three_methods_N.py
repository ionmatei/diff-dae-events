"""
Generate figures from `benchmark_three_N{N}.json` files produced by
`src/benchmark/benchmark_three_methods_N.py`.

For each selected N label this script writes (into `--results-dir`):
    benchmark_three_N{N}_prediction_error.png            (vs. iteration)
    benchmark_three_N{N}_prediction_error_vs_time.png    (vs. training time)
    benchmark_three_N{N}_validation_loss.png             (final validation
                                                          loss bar chart)

The benchmark JSON may contain only a subset of methods (when the
benchmark was launched with `--methods ...`); plots gracefully skip any
method that has no recorded history. The same `--methods` flag here
restricts which methods are drawn even when the JSON contains more.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))


# Mirror the benchmark file's METHOD_INFO so colors/labels stay aligned.
METHOD_INFO = [
    ('jax_da',     'JAX discrete adjoint', 'tab:blue'),
    ('pytorch_ad', 'PyTorch AD',           'tab:orange'),
    ('jax_ad',     'JAX AD (diffrax)',     'tab:green'),
]

LABELS_ALL = ['N3', 'N7', 'N15']


def _build_time_axis(method_data: dict) -> np.ndarray:
    """Cumulative training time per iter, derived from avg_iter_time_ms."""
    err = method_data.get('prediction_error_history', []) or []
    if not err:
        return np.zeros(0, dtype=float)
    avg_ms = float(method_data.get('avg_iter_time_ms', 0.0))
    n = len(err)
    return (np.arange(1, n + 1, dtype=float) * avg_ms) / 1000.0


def _filtered_method_info(selected: set):
    return [(k, lab, c) for k, lab, c in METHOD_INFO if k in selected]


def save_prediction_error_plot(case: dict, results_dir: str,
                               method_info: list) -> str:
    import matplotlib.pyplot as plt

    label = case['label']
    n_balls = case['n_balls']

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted_any = False
    for key, lab, color in method_info:
        err = np.asarray(case.get(key, {}).get('prediction_error_history', []) or [],
                         dtype=float)
        if err.size:
            ax.plot(range(1, err.size + 1), err, '-',
                    label=lab, color=color, linewidth=1.5)
            plotted_any = True

    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Prediction error  $\|p - p_{\mathrm{true}}\|_2$')
    ax.set_title(f'Prediction error over iterations  ({n_balls} balls)')
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    if plotted_any:
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(results_dir, f'benchmark_three_{label}_prediction_error.png')
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    if not plotted_any:
        print(f"  [warn] no method history in {label}; pred-error plot is empty")
    print(f"  Plot saved: {out_path}")
    return out_path


def save_prediction_error_vs_time_plot(case: dict, results_dir: str,
                                       method_info: list) -> str:
    import matplotlib.pyplot as plt

    label = case['label']
    n_balls = case['n_balls']

    fig, ax = plt.subplots(figsize=(9, 5))
    info_lines = []
    plotted_any = False
    for key, lab, color in method_info:
        d = case.get(key, {})
        err = np.asarray(d.get('prediction_error_history', []) or [], dtype=float)
        if err.size == 0:
            continue
        t_axis = _build_time_axis(d)
        ax.plot(t_axis, err, '-', label=lab, color=color, linewidth=1.5)
        plotted_any = True
        info_lines.append(
            f"{lab:24s}: {d.get('iterations', '?'):>4} iters,"
            f" {d.get('avg_iter_time_ms', 0.0):>9.1f} ms/iter,"
            f" total {d.get('total_time_s', 0.0):>7.1f} s"
        )

    ax.set_xlabel('Training time [s]   (= iteration $\\times$ avg iter time)')
    ax.set_ylabel(r'Prediction error  $\|p - p_{\mathrm{true}}\|_2$')
    ax.set_title(f'Prediction error vs. training time  ({n_balls} balls, {label})')
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    if plotted_any:
        ax.legend()

    if info_lines:
        ax.text(0.99, 0.02, "\n".join(info_lines),
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, family='monospace',
                bbox=dict(facecolor='white', edgecolor='0.7', alpha=0.85))

    plt.tight_layout()
    out_path = os.path.join(
        results_dir, f'benchmark_three_{label}_prediction_error_vs_time.png'
    )
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    if not plotted_any:
        print(f"  [warn] no method history in {label}; pred-vs-time plot is empty")
    print(f"  Plot saved: {out_path}")
    return out_path


def save_validation_loss_plot(case: dict, results_dir: str,
                              method_info: list) -> str:
    import matplotlib.pyplot as plt

    label = case['label']
    n_balls = case['n_balls']

    keys = [k for k, _, _ in method_info if k in case]
    if not keys:
        print(f"  [skip validation-loss plot]: no recorded methods in {label}")
        return ''
    labels = [lab for k, lab, _ in method_info if k in case]
    colors = [c for k, _, c in method_info if k in case]
    losses = [float(case[k].get('final_validation_loss', float('nan'))) for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, losses, color=colors)
    ax.set_yscale('log')
    ax.set_ylabel('Final validation loss (positions only)')
    ax.set_title(f'Final validation loss  ({n_balls} balls, {label})')
    ax.grid(True, which='both', axis='y', alpha=0.3)
    for bar, v in zip(bars, losses):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2.0, v, f'{v:.2e}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(results_dir, f'benchmark_three_{label}_validation_loss.png')
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")
    return out_path


def load_case(results_dir: str, label: str) -> dict:
    path = os.path.join(results_dir, f'benchmark_three_{label}.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Benchmark JSON not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)


def main():
    valid_methods = {k for k, _, _ in METHOD_INFO}

    parser = argparse.ArgumentParser(
        description="Render figures from benchmark_three_N{N}.json files."
    )
    parser.add_argument(
        '--results-dir', type=str,
        default=os.path.join(root_dir, 'results'),
        help='Directory containing the benchmark_three_N{N}.json files '
             '(also where PNGs are written).',
    )
    parser.add_argument(
        '--only', '--N', dest='only', type=str, default=None,
        help="Comma-separated subset of {N3,N7,N15} (default: all that exist).",
    )
    parser.add_argument(
        '--methods', type=str, default=None,
        help="Comma-separated subset of {jax_da, pytorch_ad, jax_ad} "
             "to draw (default: all available in each JSON).",
    )
    args = parser.parse_args()

    if args.only is None:
        labels_req = LABELS_ALL
    else:
        labels_req = [s.strip() for s in args.only.split(',') if s.strip()]
        bad = set(labels_req) - set(LABELS_ALL)
        if bad:
            parser.error(f"Unknown N label(s): {sorted(bad)}. "
                         f"Valid: {LABELS_ALL}")

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

    method_info = _filtered_method_info(selected_methods)

    if not os.path.isdir(args.results_dir):
        parser.error(f"--results-dir does not exist: {args.results_dir}")

    rendered = 0
    for label in labels_req:
        path = os.path.join(args.results_dir, f'benchmark_three_{label}.json')
        if not os.path.exists(path):
            print(f"  [skip {label}]: no JSON at {path}")
            continue
        with open(path, 'r') as f:
            case = json.load(f)
        print(f"\nRendering plots for {label} (n_balls={case.get('n_balls', '?')})")
        save_prediction_error_plot(case, args.results_dir, method_info)
        save_prediction_error_vs_time_plot(case, args.results_dir, method_info)
        save_validation_loss_plot(case, args.results_dir, method_info)
        rendered += 1

    if rendered == 0:
        print("\nNo benchmark JSON files matched — nothing to plot.")
        sys.exit(1)


if __name__ == '__main__':
    main()
