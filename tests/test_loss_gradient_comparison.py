"""
Compare loss functions and loss gradients (dL/dW_flat) between the
Matrix and Padded gradient approaches.

For each parameter set:
  1. Solve the DAE to obtain a solution (W_flat).
  2. Compute loss and dL/dW_flat using both approaches.
  3. Check that loss values match.
  4. Check that dL/dW_flat vectors match.
  5. Verify that gradients at event-time positions are non-zero.
"""

import numpy as np
import yaml
import json
import os
import sys

import jax
import jax.numpy as jnp
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_platform_name", "cpu")

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from src.discrete_adjoint.dae_solver import DAESolver
from debug.dae_padded_gradient import DAEPaddedGradient
from debug.dae_matrix_gradient import DAEMatrixGradient


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    solver_cfg = cfg['dae_solver']
    opt_cfg = cfg['optimizer']
    dae_spec_path = solver_cfg['dae_specification_file']
    with open(dae_spec_path, 'r') as f:
        dae_data = json.load(f)
    return dae_data, solver_cfg, opt_cfg


def prepare_loss_targets(sol):
    """Extract target times/data from solution (all interior points)."""
    all_t = []
    all_x = []
    for seg in sol.segments:
        if len(seg.t) > 0:
            all_t.append(seg.t[:-1])
            all_x.append(seg.x[:-1])
    if not all_t:
        return jnp.array([]), jnp.array([])
    return jnp.concatenate([jnp.array(t) for t in all_t]), \
           jnp.concatenate([jnp.array(x) for x in all_x])


def find_event_positions(structure):
    """Return indices into W_flat that correspond to event times."""
    positions = []
    idx = 0
    for kind, count, *extra in structure:
        length = extra[0] if kind == 'segment' else count
        if kind == 'event_time':
            positions.append(idx)
        idx += length
    return positions


def compute_padded_loss_and_grad(grad_padded, sol, p_val, target_times, target_data, blend_sharpness):
    """Compute (loss_val, dL_dW_flat) using the padded approach."""
    # dL_dW via existing API
    dL_dW, dL_dp, structure = grad_padded.compute_loss_gradients(
        sol, p_val, target_times, target_data, blend_sharpness=blend_sharpness
    )

    # loss value: pad data and call _loss_fn_padded directly
    ts_all = [np.asarray(s.t) for s in sol.segments]
    ys_all = [np.concatenate([s.x, s.z], axis=1) for s in sol.segments]
    event_infos = [e.t_event for e in sol.events]

    if grad_padded.downsample_segments:
        for i in range(len(ts_all)):
            should_downsample = grad_padded.all_segments or (ts_all[i].shape[0] > grad_padded.max_pts)
            if should_downsample:
                ts_all[i], ys_all[i] = grad_padded._downsample_segment(
                    ts_all[i], ys_all[i], grad_padded.max_pts
                )

    W_p, TS_p, b_types, b_indices, b_param, _ = grad_padded._pad_problem_data(
        ts_all, ys_all, event_infos
    )
    tt_padded, td_padded, n_tgt = grad_padded._pad_targets(target_times, target_data)
    W_p, TS_p, b_types, b_indices, b_param, tt_padded, td_padded = jax.device_put(
        (W_p, TS_p, b_types, b_indices, b_param, tt_padded, td_padded)
    )
    n_x = grad_padded.dims[0]
    actual_t_final = sol.segments[-1].t[-1]

    loss_val = DAEPaddedGradient._loss_fn_padded(
        W_p, p_val, TS_p, b_types, b_indices, b_param,
        tt_padded, td_padded, jnp.int32(n_tgt), actual_t_final, blend_sharpness, n_x
    )
    return float(loss_val), dL_dW, structure


def compare_at_params(label, solver, p_values, target_times, target_data,
                      grad_matrix, grad_padded, t_span, ncp, blend_sharpness):
    """Run comparison at one parameter set. Returns True if all checks pass."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # 1. Solve
    solver.update_parameters(np.asarray(p_values))
    sol = solver.solve_augmented(t_span, ncp=ncp)
    n_seg = len(sol.segments)
    n_ev = len(sol.events)
    print(f"  Segments: {n_seg}   Events: {n_ev}")
    for i, ev in enumerate(sol.events):
        print(f"    event {i}: t = {ev.t_event:.6f}")

    # 2. Matrix approach
    loss_mat, dL_dW_mat, W_flat_mat, structure_mat = grad_matrix.compute_loss_and_loss_grad_W(
        sol, p_values, target_times, target_data, blend_sharpness
    )
    loss_mat = float(loss_mat)

    # 3. Padded approach
    loss_pad, dL_dW_pad, structure_pad = compute_padded_loss_and_grad(
        grad_padded, sol, p_values, target_times, target_data, blend_sharpness
    )

    # 4. Compare losses
    loss_diff = abs(loss_mat - loss_pad)
    loss_rel = loss_diff / (abs(loss_mat) + 1e-15)
    print(f"\n  Loss (matrix):  {loss_mat:.10e}")
    print(f"  Loss (padded):  {loss_pad:.10e}")
    print(f"  Abs diff:       {loss_diff:.4e}")
    print(f"  Rel diff:       {loss_rel:.4e}")
    loss_ok = loss_rel < 1e-4
    print(f"  Loss match:     {'PASS' if loss_ok else 'FAIL'}")

    # 5. Compare dL/dW vectors
    dL_mat = np.asarray(dL_dW_mat)
    dL_pad = np.asarray(dL_dW_pad)

    min_len = min(len(dL_mat), len(dL_pad))
    if len(dL_mat) != len(dL_pad):
        print(f"\n  WARNING: dL_dW lengths differ: matrix={len(dL_mat)}, padded={len(dL_pad)}")
        print(f"  Comparing first {min_len} elements.")

    abs_diff = np.abs(dL_mat[:min_len] - dL_pad[:min_len])
    max_abs_diff = np.max(abs_diff)
    scale = np.maximum(np.abs(dL_mat[:min_len]), np.abs(dL_pad[:min_len]))
    rel_diff = abs_diff / (scale + 1e-15)
    max_rel_diff = np.max(rel_diff)

    print(f"\n  dL/dW comparison (len={min_len}):")
    print(f"    Max abs diff: {max_abs_diff:.4e}")
    print(f"    Max rel diff: {max_rel_diff:.4e}")
    grad_ok = max_rel_diff < 1e-2
    print(f"    Gradient match: {'PASS' if grad_ok else 'FAIL'}")

    # 6. Event-time gradient check
    event_positions = find_event_positions(structure_mat)
    print(f"\n  Event-time positions in W_flat: {event_positions}")

    events_ok = True
    for pos in event_positions:
        g_mat = float(dL_dW_mat[pos])
        g_pad = float(dL_dW_pad[pos]) if pos < len(dL_dW_pad) else float('nan')
        mat_nonzero = abs(g_mat) > 1e-12
        pad_nonzero = abs(g_pad) > 1e-12

        status_mat = "PASS" if mat_nonzero else "FAIL (zero)"
        status_pad = "PASS" if pad_nonzero else "FAIL (zero)"

        print(f"    pos {pos}: matrix grad = {g_mat:+.6e} [{status_mat}]"
              f"   padded grad = {g_pad:+.6e} [{status_pad}]")

        if not mat_nonzero or not pad_nonzero:
            events_ok = False

    # 7. Print a few surrounding gradient values for context
    if event_positions:
        print(f"\n  Gradient neighborhood around first event (pos {event_positions[0]}):")
        ep = event_positions[0]
        lo = max(0, ep - 3)
        hi = min(min_len, ep + 4)
        for j in range(lo, hi):
            marker = " <-- EVENT" if j == ep else ""
            print(f"    [{j:4d}]  matrix={dL_mat[j]:+.6e}  padded={dL_pad[j]:+.6e}{marker}")

    all_pass = loss_ok and grad_ok and events_ok
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


def run_test():
    print("=" * 70)
    print(" Loss & Gradient Comparison: Matrix vs Padded")
    print("=" * 70)

    # Load config
    config_path = os.path.join(root_dir, 'config/config_bouncing_ball.yaml')
    dae_data, solver_cfg, opt_cfg = load_config(config_path)

    t_start = solver_cfg['start_time']
    t_stop = solver_cfg['stop_time']
    ncp = solver_cfg['ncp']
    t_span = (t_start, t_stop)
    blend_sharpness = opt_cfg['blend_sharpness']
    max_blocks = opt_cfg['max_blocks']
    max_pts = opt_cfg['max_points_per_segment']
    max_targets = opt_cfg['max_targets']
    downsample_segments = opt_cfg.get('downsample_segments', False)
    all_segments = opt_cfg.get('all_segments', False)

    param_names = [p['name'] for p in dae_data['parameters']]
    true_p = [p['value'] for p in dae_data['parameters']]
    print(f"Parameters: {param_names}")
    print(f"True values: {true_p}")

    # Solver
    solver = DAESolver(dae_data, verbose=False)

    # Generate target data at true parameters
    solver.update_parameters(true_p)
    sol_true = solver.solve_augmented(t_span, ncp=ncp)
    target_times, target_data = prepare_loss_targets(sol_true)
    print(f"Target points: {len(target_times)}")

    # Build gradient computers
    grad_padded = DAEPaddedGradient(
        dae_data, max_blocks=max_blocks, max_pts=max_pts, max_targets=max_targets,
        downsample_segments=downsample_segments,
        all_segments=all_segments
    )
    grad_matrix = DAEMatrixGradient(
        dae_data, max_pts=max_pts, downsample_segments=downsample_segments,
        all_segments=all_segments
    )

    # --- Test 1: Small bias ---
    bias1 = {'g': -1.0, 'e': 0.05}
    p_test1 = list(true_p)
    for name, b in bias1.items():
        idx = param_names.index(name)
        p_test1[idx] += b
    p_test1 = jnp.array(p_test1)
    print(f"\nTest 1 params: {dict(zip(param_names, np.asarray(p_test1)))}")

    ok1 = compare_at_params(
        "Test 1: Small bias (g-1.0, e+0.05)",
        solver, p_test1, target_times, target_data,
        grad_matrix, grad_padded, t_span, ncp, blend_sharpness
    )

    # --- Test 2: Larger bias ---
    bias2 = {'g': -3.0, 'e': 0.15}
    p_test2 = list(true_p)
    for name, b in bias2.items():
        idx = param_names.index(name)
        p_test2[idx] += b
    p_test2 = jnp.array(p_test2)
    print(f"\nTest 2 params: {dict(zip(param_names, np.asarray(p_test2)))}")

    ok2 = compare_at_params(
        "Test 2: Larger bias (g-3.0, e+0.15)",
        solver, p_test2, target_times, target_data,
        grad_matrix, grad_padded, t_span, ncp, blend_sharpness
    )

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Test 1 (small bias):  {'PASS' if ok1 else 'FAIL'}")
    print(f"  Test 2 (larger bias): {'PASS' if ok2 else 'FAIL'}")
    all_ok = ok1 and ok2
    print(f"  Overall:              {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return all_ok


if __name__ == "__main__":
    run_test()
