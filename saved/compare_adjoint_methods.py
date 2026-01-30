import numpy as np
import yaml
import json
import os
import sys
import time
import jax
import jax.numpy as jnp
from jax import config
from functools import partial

config.update("jax_enable_x64", True)

# Add src to path
sys.path.append(os.getcwd())

from src.discrete_adjoint.dae_solver import DAESolver

# Import both versions of residual logic
import debug.verify_residual as dense
import debug.verify_residual_gmres as gmres_impl

def compare_methods():
    print("="*80)
    print("COMPARISON: Dense vs. GMRES-based Adjoint")
    print("="*80)

    # 1. Setup system
    dae_data, solver_cfg = gmres_impl.load_system('config/config_bouncing_ball.yaml')
    solver = DAESolver(dae_data, verbose=False)

    t_span = (0.0, 2.0)
    ncp = 15

    print(f"Generating solution (ncp={ncp})...")
    sol = solver.solve_augmented(t_span, ncp=ncp)

    # 2. Pack solution
    W_flat, structure, grid_taus = gmres_impl.pack_solution(sol, dae_data)
    funcs = gmres_impl.create_jax_functions(dae_data)
    f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_p = dims
    n_w = n_x + n_z

    p_opt = jnp.array([p['value'] for p in dae_data['parameters']])
    param_names = [p['name'] for p in dae_data['parameters']]
    param_mapping = (p_opt, [0, 1])

    # 3. Preparation for Loss
    target_times, target_data = gmres_impl.prepare_loss_targets(sol, dae_data['states'], *t_span)

    def loss_function(W, p):
        segs_t, segs_x, segs_z, ev_tau = gmres_impl.unpack_solution_structure(W, structure, (n_x, n_z, n_w), grid_taus)
        y_pred = gmres_impl.predict_trajectory_sigmoid(segs_t, segs_x, segs_z, ev_tau, target_times, blend_sharpness=150.0)
        return jnp.mean((y_pred - target_data)**2)

    def R_global(W, p):
        return gmres_impl.unpack_and_compute_residual(W, p, dae_data, structure, funcs, param_mapping, grid_taus)

    # Gradients of Loss
    dL_dp = jax.grad(loss_function, 1)(W_flat, p_opt)
    dL_dW = jax.grad(loss_function, 0)(W_flat, p_opt)

    # =========================================================================
    # METHOD A: Dense Matrix Solve
    # =========================================================================
    print("\nMethod A: Dense Matrix Solve...")
    dR_dW_dense = jax.jacfwd(R_global, 0)(W_flat, p_opt)
    dR_dp_dense = jax.jacfwd(R_global, 1)(W_flat, p_opt)

    start_dense = time.time()
    lambda_dense = jnp.linalg.solve(dR_dW_dense.T, -dL_dW)
    grad_A = dL_dp + jnp.dot(lambda_dense, dR_dp_dense)
    time_dense = time.time() - start_dense
    print(f"  Time: {time_dense:.4f}s")
    print(f"  Result: {grad_A}")

    # =========================================================================
    # METHOD B: GMRES-based Adjoint Solve
    # =========================================================================
    print("\nMethod B: GMRES-based Adjoint Solve...")
    from jax.scipy.sparse.linalg import gmres

    start_gmres = time.time()

    # Define A^T operator using VJP (matrix-free)
    def At_operator(v):
        _, vjp_fun = jax.vjp(lambda w: R_global(w, p_opt), W_flat)
        return vjp_fun(v)[0]

    # Solve A^T @ lambda = -dL/dW
    lambda_gmres, info = gmres(At_operator, -dL_dW, tol=1e-10, maxiter=2000)

    # Compute dR/dp^T @ lambda using VJP
    _, vjp_p = jax.vjp(lambda p: R_global(W_flat, p), p_opt)
    term_2 = vjp_p(lambda_gmres)[0]

    grad_B = dL_dp + term_2
    time_gmres = time.time() - start_gmres
    print(f"  Time: {time_gmres:.4f}s")
    print(f"  Result: {grad_B}")

    # =========================================================================
    # METHOD C: Direct Adjoint Sweep
    # =========================================================================
    print("\nMethod C: Direct Adjoint Sweep (Recursive/Scan)...")
    import debug.verify_residual_direct_v2 as direct_impl
    
    start_direct = time.time()
    grad_C = direct_impl.compute_adjoint_sweep_direct(W_flat, p_opt, dae_data, structure, funcs, grid_taus, dL_dW, dL_dp)
    time_direct = time.time() - start_direct
    print(f"  Time: {time_direct:.4f}s")
    print(f"  Result: {grad_C}")

    # =========================================================================
    # COMPARISON
    # =========================================================================
    # =========================================================================
    # COMPARISON
    # =========================================================================
    diff_total_B = jnp.linalg.norm(grad_A - grad_B)
    diff_total_C = jnp.linalg.norm(grad_A - grad_C)
    
    print("\n" + "="*80)
    print("FINAL COMPARISON RESULTS")
    print("="*80)
    print(f"{'Parameter':<10} | {'Dense (A)':<18} | {'GMRES (B)':<18} | {'Direct (C)':<18} | {'Diff(A,C)':<10}")
    print("-" * 100)
    for i, name in enumerate(param_names):
        print(f"{name:<10} | {grad_A[i]:<18.6e} | {grad_B[i]:<18.6e} | {grad_C[i]:<18.6e} | {abs(grad_A[i]-grad_C[i]):<10.2e}")
    
    print("\n" + "-"*40)
    print(f"Total Gradient Difference (A vs B): {diff_total_B:.4e}")
    print(f"Total Gradient Difference (A vs C): {diff_total_C:.4e}")

    if diff_total_C < 1e-8:
        print("SUCCESS: Direct method matches Dense reference.")
    else:
        print("WARNING: Direct method divergence.")
    print("-"*40)

    dist_R = jnp.linalg.norm(R_global(W_flat, p_opt))
    print(f"Residual Norm at W0: {dist_R:.2e}")

    # Plot Comparison Result
    print("\nGenerating Gradient Comparison Plot (grad_comparison.png)...")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    x_indices = jnp.arange(len(param_names))
    width = 0.25
    plt.bar(x_indices - width, grad_A, width=width, label='Dense (A)', alpha=0.7)
    plt.bar(x_indices, grad_B, width=width, label='GMRES (B)', alpha=0.7)
    plt.bar(x_indices + width, grad_C, width=width, label='Direct (C)', alpha=0.7)
    plt.xticks(x_indices, param_names)
    plt.title('Parameter Gradient Comparison')
    plt.ylabel('dJ/dp')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('debug/grad_comparison.png')
    print("  Saved to debug/grad_comparison.png")

if __name__ == "__main__":
    compare_methods()
