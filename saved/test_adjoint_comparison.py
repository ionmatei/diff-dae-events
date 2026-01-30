"""
Compare adjoint solutions (multipliers) between Matrix and Padded approaches.

For each parameter set:
  1. Solve the DAE.
  2. Matrix approach: compute (loss, dL_dW, dL_dp, lam, total_grad) via
     direct Jacobian build + linear solve.
  3. Padded approach: compute (loss, total_grad) via reverse adjoint sweep.
  4. Compare total_grad (which implicitly validates the adjoint).
  5. Show the adjoint multiplier vector from the matrix approach,
     highlighting entries at event-time positions.
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
    positions = []
    idx = 0
    for kind, count, *extra in structure:
        length = extra[0] if kind == 'segment' else count
        if kind == 'event_time':
            positions.append(idx)
        idx += length
    return positions


def compare_at_params(label, solver, p_values, param_names, target_times, target_data,
                      grad_matrix, grad_padded, t_span, ncp, blend_sharpness):
    """Run adjoint comparison at one parameter set. Returns True if all checks pass."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Parameters: {dict(zip(param_names, np.asarray(p_values)))}")

    # 1. Solve
    solver.update_parameters(np.asarray(p_values))
    sol = solver.solve_augmented(t_span, ncp=ncp)
    n_seg = len(sol.segments)
    n_ev = len(sol.events)
    print(f"  Segments: {n_seg}   Events: {n_ev}")
    for i, ev in enumerate(sol.events):
        print(f"    event {i}: t = {ev.t_event:.6f}")

    # 2. Matrix: full adjoint decomposition
    loss_mat, dL_dW_mat, dL_dp_mat, lam_mat, total_grad_mat, W_flat, structure = \
        grad_matrix.compute_full_adjoint(
            sol, p_values, target_times, target_data, blend_sharpness
        )
    loss_mat = float(loss_mat)

    # 3. Padded: total gradient (loss + adjoint sweep)
    loss_pad, total_grad_pad = grad_padded.compute_total_gradient(
        sol, p_values, target_times, target_data,
        blend_sharpness=blend_sharpness
    )
    loss_pad = float(loss_pad)

    # 4. Compare losses
    loss_diff = abs(loss_mat - loss_pad)
    loss_rel = loss_diff / (abs(loss_mat) + 1e-15)
    print(f"\n  Loss (matrix): {loss_mat:.10e}")
    print(f"  Loss (padded): {loss_pad:.10e}")
    print(f"  Rel diff:      {loss_rel:.4e}")
    loss_ok = loss_rel < 1e-4
    print(f"  Loss match:    {'PASS' if loss_ok else 'FAIL'}")

    # 5. Compare total_grad (parameter gradients)
    tg_mat = np.asarray(total_grad_mat)
    tg_pad = np.asarray(total_grad_pad)
    print(f"\n  Total gradient (dp):")
    print(f"    Matrix: {tg_mat}")
    print(f"    Padded: {tg_pad}")

    abs_diff_tg = np.abs(tg_mat - tg_pad)
    scale_tg = np.maximum(np.abs(tg_mat), np.abs(tg_pad))
    rel_diff_tg = abs_diff_tg / (scale_tg + 1e-15)
    max_rel_tg = np.max(rel_diff_tg)
    print(f"    Max rel diff: {max_rel_tg:.4e}")
    grad_ok = max_rel_tg < 1e-2
    print(f"    Match: {'PASS' if grad_ok else 'FAIL'}")

    for i, name in enumerate(param_names):
        print(f"      d/d{name}: matrix={tg_mat[i]:+.8e}  padded={tg_pad[i]:+.8e}  "
              f"rel_diff={rel_diff_tg[i]:.4e}")

    # 6. Show dL_dp (direct loss gradient w.r.t. parameters)
    print(f"\n  dL/dp (direct, from matrix): {np.asarray(dL_dp_mat)}")

    # 7. Adjoint multiplier vector (matrix approach)
    lam = np.asarray(lam_mat)
    event_positions = find_event_positions(structure)
    print(f"\n  Adjoint multiplier lam (len={len(lam)}):")
    print(f"    ||lam|| = {np.linalg.norm(lam):.6e}")
    print(f"    max|lam| = {np.max(np.abs(lam)):.6e}")
    print(f"    min|lam| = {np.min(np.abs(lam)):.6e}")

    print(f"\n  Adjoint at event-time positions:")
    events_ok = True
    for pos in event_positions:
        lam_val = lam[pos]
        nonzero = abs(lam_val) > 1e-12
        status = "PASS" if nonzero else "FAIL (zero)"
        print(f"    pos {pos}: lam = {lam_val:+.8e}  [{status}]")
        if not nonzero:
            events_ok = False

    # 8. Show neighborhood around each event
    for pos in event_positions:
        print(f"\n  Adjoint neighborhood around event at pos {pos}:")
        lo = max(0, pos - 3)
        hi = min(len(lam), pos + 4)
        for j in range(lo, hi):
            marker = " <-- EVENT" if j == pos else ""
            print(f"    [{j:4d}] lam={lam[j]:+.10e}{marker}")

    all_pass = loss_ok and grad_ok and events_ok
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


def run_test():
    print("=" * 70)
    print(" Adjoint Comparison: Matrix vs Padded")
    print("=" * 70)

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

    param_names = [p['name'] for p in dae_data['parameters']]
    true_p = [p['value'] for p in dae_data['parameters']]
    print(f"Parameters: {param_names}")
    print(f"True values: {true_p}")

    solver = DAESolver(dae_data, verbose=False)

    # Ground-truth targets
    solver.update_parameters(true_p)
    sol_true = solver.solve_augmented(t_span, ncp=ncp)
    target_times, target_data = prepare_loss_targets(sol_true)
    print(f"Target points: {len(target_times)}")

    # Build gradient computers
    grad_padded = DAEPaddedGradient(
        dae_data, max_blocks=max_blocks, max_pts=max_pts, max_targets=max_targets
    )
    grad_matrix = DAEMatrixGradient(dae_data)

    # --- Test 1: Small bias ---
    bias1 = {'g': -1.0, 'e': 0.05}
    p_test1 = list(true_p)
    for name, b in bias1.items():
        idx = param_names.index(name)
        p_test1[idx] += b
    p_test1 = jnp.array(p_test1)

    ok1 = compare_at_params(
        "Test 1: Small bias (g-1.0, e+0.05)",
        solver, p_test1, param_names, target_times, target_data,
        grad_matrix, grad_padded, t_span, ncp, blend_sharpness
    )

    # --- Test 2: Larger bias (matches user's test files) ---
    bias2 = {'g': -2.0, 'e': -0.15}
    p_test2 = list(true_p)
    for name, b in bias2.items():
        idx = param_names.index(name)
        p_test2[idx] += b
    p_test2 = jnp.array(p_test2)

    ok2 = compare_at_params(
        "Test 2: Larger bias (g-2.0, e-0.15)",
        solver, p_test2, param_names, target_times, target_data,
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
