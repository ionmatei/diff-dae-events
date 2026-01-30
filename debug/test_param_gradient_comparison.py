"""
Compare the total gradient of the loss w.r.t. parameters (dp) between the
Matrix and Padded approaches at two biased parameter sets.

For each bias:
  1. Solve the DAE at the biased parameters.
  2. Matrix: compute total dp via direct Jacobian + adjoint linear solve.
  3. Padded: compute total dp via reverse adjoint sweep.
  4. Compare per-parameter gradients.
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
    all_t, all_x = [], []
    for seg in sol.segments:
        if len(seg.t) > 0:
            all_t.append(seg.t[:-1])
            all_x.append(seg.x[:-1])
    if not all_t:
        return jnp.array([]), jnp.array([])
    return jnp.concatenate([jnp.array(t[:-1]) for t in all_t]), \
           jnp.concatenate([jnp.array(x[:-1]) for x in all_x])


def compare_at_params(label, solver, p_values, param_names,
                      target_times, target_data,
                      grad_matrix, grad_padded,
                      t_span, ncp, blend_sharpness):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Parameters: {dict(zip(param_names, np.asarray(p_values)))}")

    # Solve
    solver.update_parameters(np.asarray(p_values))
    sol = solver.solve_augmented(t_span, ncp=ncp)
    print(f"  Segments: {len(sol.segments)}   Events: {len(sol.events)}")

    # Matrix: full adjoint gives (loss, dL_dW, dL_dp, lam, total_grad, ...)
    loss_mat, _, dL_dp_mat, _, tg_mat, _, _ = grad_matrix.compute_full_adjoint(
        sol, p_values, target_times, target_data, blend_sharpness
    )

    # Padded: compute_total_gradient gives (loss, total_grad)
    loss_pad, tg_pad = grad_padded.compute_total_gradient(
        sol, p_values, target_times, target_data, blend_sharpness=blend_sharpness
    )

    loss_mat = float(loss_mat)
    loss_pad = float(loss_pad)
    tg_mat = np.asarray(tg_mat)
    tg_pad = np.asarray(tg_pad)
    dL_dp_mat = np.asarray(dL_dp_mat)

    # Loss
    loss_rel = abs(loss_mat - loss_pad) / (abs(loss_mat) + 1e-15)
    print(f"\n  Loss:  matrix={loss_mat:.10e}  padded={loss_pad:.10e}  rel_diff={loss_rel:.2e}")

    # Direct dL/dp (should be zero — parameters don't appear in the loss directly)
    print(f"  dL/dp direct (matrix): {dL_dp_mat}")

    # --- Finite Difference Approximation ---
    print("\n  Computing Finite Difference Gradient...")
    fd_grad = compute_finite_difference_gradient(
        solver, p_values, target_times, target_data, grad_padded,
        t_span, ncp, blend_sharpness
    )
    print(f"  FD Gradient: {fd_grad}")

    # Per-parameter total gradient
    print(f"\n  {'Parameter':<10} {'Matrix dp':>16} {'Padded dp':>16} {'FD dp':>16} {'Abs diff':>12} {'Rel diff':>12}")
    print(f"  {'-'*10} {'-'*16} {'-'*16} {'-'*16} {'-'*12} {'-'*12}")

    all_ok = True
    for i, name in enumerate(param_names):
        m = tg_mat[i]
        p = tg_pad[i]
        fd = fd_grad[i]
        
        # Check agreement between Matrix and Padded (primary test)
        ad = abs(m - p)
        rd = ad / (max(abs(m), abs(p)) + 1e-15)
        ok = rd < 1e-4
        flag = "" if ok else " <-- MISMATCH"
        
        # Also show agreement with FD
        # (FD is approximate, so we don't strictly fail on it, but it's good context)
        fd_diff = abs(p - fd)
        
        print(f"  {name:<10} {m:>+16.8e} {p:>+16.8e} {fd:>+16.8e} {ad:>12.4e} {rd:>12.4e}{flag}")
        if not ok:
            all_ok = False

    max_rd = np.max(np.abs(tg_mat - tg_pad) / (np.maximum(np.abs(tg_mat), np.abs(tg_pad)) + 1e-15))
    print(f"\n  Max rel diff across parameters: {max_rd:.4e}")
    print(f"  Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def compute_finite_difference_gradient(solver, base_params, target_times, target_data, 
                                     grad_computer, t_span, ncp, blend_sharpness, 
                                     epsilon=1e-5):
    """
    Compute gradient via finite differences.
    uses grad_computer to calculate loss (which computes gradients too but we ignore them).
    """
    n_p = len(base_params)
    fd_grad = np.zeros(n_p)
    
    # Base loss
    solver.update_parameters(base_params)
    sol0 = solver.solve_augmented(t_span, ncp=ncp)
    loss0, _ = grad_computer.compute_total_gradient(
        sol0, base_params, target_times, target_data, blend_sharpness=blend_sharpness
    )
    loss0 = float(loss0)
    
    # Convert to numpy for mutable perturbation
    base_params_np = np.asarray(base_params)
    
    for i in range(n_p):
        p_perturbed = base_params_np.copy()
        p_perturbed[i] += epsilon
        
        solver.update_parameters(p_perturbed)
        sol_p = solver.solve_augmented(t_span, ncp=ncp)
        
        loss_p, _ = grad_computer.compute_total_gradient(
            sol_p, p_perturbed, target_times, target_data, blend_sharpness=blend_sharpness
        )
        loss_p = float(loss_p)
        
        fd_grad[i] = (loss_p - loss0) / epsilon
        
    return fd_grad


def run_test():
    print("=" * 70)
    print(" Parameter Gradient Comparison: Matrix vs Padded")
    print("=" * 70)

    config_path = os.path.join(root_dir, 'config/config_bouncing_ball.yaml')
    dae_data, solver_cfg, opt_cfg = load_config(config_path)

    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    ncp = solver_cfg['ncp']
    blend_sharpness = opt_cfg['blend_sharpness']
    max_blocks = opt_cfg['max_blocks']
    max_pts = opt_cfg['max_points_per_segment']
    max_targets = opt_cfg['max_targets']
    max_targets = opt_cfg['max_targets']
    downsample_segments = opt_cfg.get('downsample_segments', False)
    all_segments = opt_cfg.get('all_segments', False)

    param_names = [p['name'] for p in dae_data['parameters']]
    true_p = [p['value'] for p in dae_data['parameters']]
    print(f"Parameters: {param_names},  True: {true_p}")

    solver = DAESolver(dae_data, verbose=False)

    # Ground-truth targets
    solver.update_parameters(true_p)
    sol_true = solver.solve_augmented(t_span, ncp=ncp)
    target_times, target_data = prepare_loss_targets(sol_true)
    print(f"Target points: {len(target_times)}")

    grad_padded = DAEPaddedGradient(
        dae_data, max_blocks=max_blocks, max_pts=max_pts, max_targets=max_targets,
        downsample_segments=downsample_segments,
        all_segments=all_segments
    )
    grad_matrix = DAEMatrixGradient(
        dae_data, max_pts=max_pts, downsample_segments=downsample_segments,
        all_segments=all_segments
    )

    # --- Test 1: bias g-2.0, e-0.15 (matches user test files) ---
    p1 = list(true_p)
    for name, b in {'g': -2.0, 'e': -0.15}.items():
        p1[param_names.index(name)] += b
    ok1 = compare_at_params(
        "Test 1: g-2.0, e-0.15",
        solver, jnp.array(p1), param_names,
        target_times, target_data,
        grad_matrix, grad_padded, t_span, ncp, blend_sharpness
    )

    # --- Test 2: bias g-1.0, e+0.05 ---
    p2 = list(true_p)
    for name, b in {'g': -1.0, 'e': 0.05}.items():
        p2[param_names.index(name)] += b
    ok2 = compare_at_params(
        "Test 2: g-1.0, e+0.05",
        solver, jnp.array(p2), param_names,
        target_times, target_data,
        grad_matrix, grad_padded, t_span, ncp, blend_sharpness
    )

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Test 1 (g-2.0, e-0.15): {'PASS' if ok1 else 'FAIL'}")
    print(f"  Test 2 (g-1.0, e+0.05): {'PASS' if ok2 else 'FAIL'}")
    all_ok = ok1 and ok2
    print(f"  Overall:                {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return all_ok


if __name__ == "__main__":
    run_test()
