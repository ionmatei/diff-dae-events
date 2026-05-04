"""
Plot prediction error vs. training time for a single benchmark JSON
(e.g. `results/benchmark_N3.json`).

Time axis is derived from the per-method average iteration time:

    elapsed[i] = (i + 1) * avg_iter_time

where i runs over the recorded `prediction_error_history` entries. This
treats each iteration as costing the average — fine for a comparison
across methods, since both methods see the same definition.

Usage:
    python src/benchmark/plot_prediction_error_vs_time.py results/benchmark_N3.json
    python src/benchmark/plot_prediction_error_vs_time.py results/benchmark_N15.json --out custom.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _load_case(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def _build_time_axis(method_data: dict) -> np.ndarray:
    """Return cumulative training time (in seconds) for each iteration in
    `prediction_error_history`, using `avg_iter_time_ms` as the per-iter
    cost. Returns an empty array if no error history is recorded."""
    err = method_data.get('prediction_error_history', []) or []
    if not err:
        return np.zeros(0, dtype=float)
    avg_ms = float(method_data.get('avg_iter_time_ms', 0.0))
    n = len(err)
    return (np.arange(1, n + 1, dtype=float) * avg_ms) / 1000.0  # ms -> s


def plot_case(case: dict, out_path: str) -> str:
    import matplotlib.pyplot as plt

    label = case.get('label', '')
    n_balls = case.get('n_balls', '?')

    jax_d = case.get('jax', {}) or {}
    pt_d = case.get('pytorch', {}) or {}

    jax_t = _build_time_axis(jax_d)
    pt_t = _build_time_axis(pt_d)
    jax_err = np.asarray(jax_d.get('prediction_error_history', []) or [], dtype=float)
    pt_err = np.asarray(pt_d.get('prediction_error_history', []) or [], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))

    if jax_err.size:
        ax.plot(jax_t, jax_err, '-', label='JAX', color='tab:blue', linewidth=1.5)
    if pt_err.size:
        ax.plot(pt_t, pt_err, '-', label='PyTorch', color='tab:orange', linewidth=1.5)

    ax.set_xlabel('Training time [s]   (= iteration $\\times$ avg iter time)')
    ax.set_ylabel(r'Prediction error  $\|p - p_{\mathrm{true}}\|_2$')
    ax.set_title(f'Prediction error vs. training time  ({n_balls} balls, {label})')
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    if jax_err.size or pt_err.size:
        ax.legend()

    # Annotate per-method totals so the plot is self-documenting.
    info_lines = []
    if jax_err.size:
        info_lines.append(
            f"JAX:     {jax_d.get('iterations', '?')} iters,"
            f" {jax_d.get('avg_iter_time_ms', 0.0):.1f} ms/iter,"
            f" total {jax_d.get('total_time_s', 0.0):.1f} s"
        )
    if pt_err.size:
        info_lines.append(
            f"PyTorch: {pt_d.get('iterations', '?')} iters,"
            f" {pt_d.get('avg_iter_time_ms', 0.0):.1f} ms/iter,"
            f" total {pt_d.get('total_time_s', 0.0):.1f} s"
        )
    if info_lines:
        ax.text(0.99, 0.02, "\n".join(info_lines),
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, family='monospace',
                bbox=dict(facecolor='white', edgecolor='0.7', alpha=0.85))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot prediction error vs. training time from a benchmark JSON."
    )
    parser.add_argument('input', type=str, help="Path to a benchmark_N*.json file.")
    parser.add_argument('--out', type=str, default=None,
                        help="Output PNG path. Defaults to the input directory with "
                             "suffix '_prediction_error_vs_time.png'.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    case = _load_case(args.input)

    if args.out is None:
        in_dir = os.path.dirname(os.path.abspath(args.input))
        in_base = os.path.splitext(os.path.basename(args.input))[0]
        out_path = os.path.join(in_dir, f"{in_base}_prediction_error_vs_time.png")
    else:
        out_path = args.out

    saved = plot_case(case, out_path)
    print(f"Plot saved: {saved}")


if __name__ == "__main__":
    main()
