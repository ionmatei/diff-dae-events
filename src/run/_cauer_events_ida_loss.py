"""
Internal helper used by both Cauer-events optimization runners
(`src.run.optimization_cauer_events`, `src.run.optimization_cauer_events_da`)
to re-score initial / optimized parameters on a single trusted forward
solver (IDA) and a shared uniform time grid.

Sidesteps the gap where AD natively reports a loss with a diffrax-
generated `y_pred` while DA reports it with an IDA-segment-blended
`y_pred`: here both methods are scored under the same forward solver,
the same time grid, and the same MSE formula
(`mean((x_pred - x_true) ** 2)` over all time-state entries), so the
two values are directly comparable.
"""

from __future__ import annotations

import numpy as np


def _stack_and_interp(sol, t_grid: np.ndarray) -> np.ndarray:
    ts, xs = [], []
    for seg in sol.segments:
        if len(seg.t) > 0:
            ts.append(np.asarray(seg.t))
            xs.append(np.asarray(seg.x))
    if not ts:
        return np.zeros((len(t_grid), 0))
    t_seg = np.concatenate(ts)
    x_seg = np.concatenate(xs, axis=0)
    out = np.empty((len(t_grid), x_seg.shape[1]), dtype=float)
    for j in range(x_seg.shape[1]):
        out[:, j] = np.interp(t_grid, t_seg, x_seg[:, j])
    return out


def evaluate_ida_mse(
    dae_data: dict,
    t_span: tuple[float, float],
    ncp: int,
    p_nominal: np.ndarray,
    p_init_full: np.ndarray,
    p_opt_full: np.ndarray,
    *,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-6,
) -> dict:
    """Run IDA at p_nominal, p_init_full, p_opt_full and return a dict of
    MSEs (against the IDA trajectory at p_nominal) on the uniform grid
    `np.linspace(t_span[0], t_span[1], ncp + 1)`.
    """
    # Imported lazily so callers that never invoke this don't pull SUNDIALS.
    from src.discrete_adjoint.dae_solver import DAESolver

    t_start, t_stop = t_span
    t_grid = np.linspace(t_start, t_stop, ncp + 1)
    solver = DAESolver(dae_data, verbose=False, use_compiled_residual=True)
    solver.atol = atol
    solver.rtol = rtol

    def _sim(p_full):
        solver.update_parameters(np.asarray(p_full, dtype=float))
        sol = solver.solve_augmented(t_span, ncp=ncp)
        return _stack_and_interp(sol, t_grid)

    def _mse(x_pred, x_true):
        return float(np.mean((x_pred - x_true) ** 2))

    x_true = _sim(p_nominal)
    x_init = _sim(p_init_full)
    x_opt = _sim(p_opt_full)
    return {
        "mse_init_ida": _mse(x_init, x_true),
        "mse_opt_ida": _mse(x_opt, x_true),
        "ncp_ida_eval": int(ncp),
        "rtol_ida_eval": float(rtol),
        "atol_ida_eval": float(atol),
    }
