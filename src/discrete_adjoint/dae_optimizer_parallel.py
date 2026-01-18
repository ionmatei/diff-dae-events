"""
Parallel DAE Optimizer using DEER methods for adjoint solving.

This module provides DAEOptimizerParallel, a subclass of DAEOptimizer that uses
parallel associative scan (via deer.maths.matmul_recursive) to solve the
adjoint system, providing O(log N) depth instead of O(N).
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Tuple, Dict, List
import numpy as np
import time

from .dae_jacobian import DAEOptimizer
from src.deer.maths import matmul_recursive


class DAEOptimizerParallel(DAEOptimizer):
    """
    DAE optimizer that uses parallel associative scan for the adjoint solve.
    
    This replaces the sequential backward substitution in DAEOptimizer with
    a parallel algorithm, which is significantly faster on GPUs/TPUs for long
    trajectories.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # JIT-compile the combined gradient function (which now uses the parallel solver)
        self._compute_gradient_combined_jit = jit(self._compute_gradient_combined)

    def _compute_gradient_combined(
        self,
        t_sol: jnp.ndarray,
        y_array: jnp.ndarray,
        y_target_use: jnp.ndarray,
        p_opt_vals_jax: jnp.ndarray,
        step_size: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Combined optimization step using parallel adjoint solver.
        
        Args:
            t_sol: time points, shape (N+1,)
            y_array: states [x, z] at time points, shape (n_total, N+1)
            y_target_use: target outputs, shape (N+1, n_outputs)
            p_opt_vals_jax: optimized parameter values, shape (n_params_opt,)
            step_size: gradient descent step size

        Returns:
            p_opt_new: updated optimized parameters, shape (n_params_opt,)
            grad_p_opt: gradient w.r.t. optimized parameters, shape (n_params_opt,)
        """
        # Step 2: Compute loss gradient dL/dy
        # Ensure y_array is (N+1, n_total) so that dL_dy comes back as (N+1, n_total)
        # y_array from optimization_step_combined is typically (n_total, N+1)
        if y_array.shape[0] == self.jac.n_total and y_array.shape[1] == t_sol.shape[0]:
            y_array_loss = y_array.T
        else:
            y_array_loss = y_array

        dL_dy = self.jac.trajectory_loss_gradient_analytical(t_sol, y_array_loss, y_target_use, p_opt_vals_jax)

        # Scale by 1/N if using mean loss
        if self.loss_type == 'mean':
            n_outputs = y_target_use.shape[1] if y_target_use.shape[0] == t_sol.shape[0] else y_target_use.shape[0]
            n_time = t_sol.shape[0]
            N_total = n_outputs * n_time
            dL_dy = dL_dy / N_total

        # Exclude initial condition which is fixed
        dL_dy_adjoint = dL_dy[1:, :]  # shape: (N, n_total)

        # Step 3: Compute Jacobian blocks (pure JAX version)
        # Detect and transpose if needed
        if y_array.shape[0] == self.jac.n_total and y_array.shape[1] == t_sol.shape[0]:
            y_array_for_jac = y_array.T
        else:
            y_array_for_jac = y_array

        N = t_sol.shape[0] - 1  # Number of intervals

        # Prepare arrays for vectorized computation
        t_k = t_sol[:-1]      # t_0, t_1, ..., t_{N-1}
        t_kp1 = t_sol[1:]     # t_1, t_2, ..., t_N
        y_k = y_array_for_jac[:-1]      # y_0, y_1, ..., y_{N-1}
        y_kp1 = y_array_for_jac[1:]     # y_1, y_2, ..., y_N

        # Compute Jacobians in parallel using vmapped functions
        J_prev = self.jac._jac_y_k_vmapped(t_k, t_kp1, y_k, y_kp1, p_opt_vals_jax)      # shape: (N, n_total, n_total)
        J_curr = self.jac._jac_y_kp1_vmapped(t_k, t_kp1, y_k, y_kp1, p_opt_vals_jax)    # shape: (N, n_total, n_total)

        # Step 4: Solve adjoint system (Parallel Version)
        # Backward recurrence: lambda[k] = (J_curr[k]^T)^-1 * (b[k] - J_prev[k+1]^T * lambda[k+1])
        # Can be written as: lambda[k] = M[k] * lambda[k+1] + v[k]
        # where M[k] = -(J_curr[k]^T)^-1 * J_prev[k+1]^T
        # and v[k] = (J_curr[k]^T)^-1 * b[k]
        
        # Parallel inversions / system solves
        # Transpose matrices for the linear systems
        J_curr_T = jnp.transpose(J_curr, (0, 2, 1))  # (N, n, n)
        J_prev_T = jnp.transpose(J_prev, (0, 2, 1))  # (N, n, n)
        
        # Compute v[k] for all k: Solve J_curr[k]^T * v[k] = b[k]
        # b[k] here is dL_dy_adjoint[k]
        v_all = vmap(jnp.linalg.solve)(J_curr_T, dL_dy_adjoint)  # (N, n_total)
        
        # Compute M[k] for k = 0...N-2
        # Need J_curr_T[0...N-2] and J_prev_T[1...N-1]
        J_curr_T_m = J_curr_T[:-1]
        J_prev_T_shift = J_prev_T[1:]
        
        # Solve J_curr[k]^T * M[k] = -J_prev[k+1]^T
        # Equivalently solve J_curr[k]^T * X = J_prev[k+1]^T and negate
        M_blocks = -vmap(jnp.linalg.solve)(J_curr_T_m, J_prev_T_shift)  # (N-1, n, n)
        
        # Setup for matmul_recursive (solving reverse recurrence)
        # The recurrence is backward: y[i] = M * y[i+1] + v
        # We transform to forward scan by reversing indices.
        # Let y_rev[j] = y[N-1-j]
        # Then y_rev[j] = M[N-1-j] * y_rev[j-1] + v[N-1-j]
        
        # Input construction for matmul_recursive
        y0 = v_all[-1]            # Last element (start of reverse scan)
        vecs = v_all[:-1][::-1]   # Remaining elements, reversed
        mats = M_blocks[::-1]     # Matrices, reversed
        
        if N == 1:
            # Special case for single interval, matmul_recursive might expect arrays
            lambda_adjoint = v_all
        else:
            # Apply parallel scan
            y_rev = matmul_recursive(mats, vecs, y0)
            
            # Reverse back to original order
            lambda_adjoint = y_rev[::-1]
            
        # Step 5: Compute parameter Jacobian dR/dp_opt
        J_param = self.jac._jac_p_vmapped(t_k, t_kp1, y_k, y_kp1, p_opt_vals_jax)
        dR_dp_opt = J_param.reshape(N * self.jac.n_total, -1)

        # Step 6: Compute parameter gradient
        lambda_flat = lambda_adjoint.flatten()
        grad_p_opt = -dR_dp_opt.T @ lambda_flat

        # Step 7: Update
        p_opt_new = p_opt_vals_jax - step_size * grad_p_opt

        return p_opt_new, grad_p_opt
