"""
Fast DAE Optimizer using Parallel Adjoint Solver.

This module provides DAEOptimizerFast, which inherits from DAEOptimizer but uses
the fast parallel adjoint solver from adjoint_solver_fast.py.

The fast adjoint solver batch-precomputes all matrix operations before the scan,
making it more efficient than the sequential version.

Key difference from dae_jacobian.DAEOptimizer:
- Uses solve_adjoint_system_scan from adjoint_solver_fast.py in the adjoint solve step
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from jax import jit

from .dae_jacobian import DAEOptimizer
from .adjoint_solver_fast import solve_adjoint_system_scan


class DAEOptimizerFast(DAEOptimizer):
    """
    Fast DAE optimizer using optimized adjoint solver.

    This is a subclass of DAEOptimizer that uses the fast parallel
    adjoint solver for improved performance.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the fast DAE optimizer."""
        super().__init__(*args, **kwargs)

        # Override the JIT-compiled gradient function with our fast version
        self._compute_gradient_combined_jit = jit(self._compute_gradient_combined)

        print("(Using Fast Parallel Adjoint Solver)")

    def _compute_gradient_combined(
        self,
        t_sol: jnp.ndarray,
        y_array: jnp.ndarray,
        y_target_use: jnp.ndarray,
        p_opt_vals_jax: jnp.ndarray,
        step_size: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Combined JIT-compiled function for steps 2-7 of optimization.

        This overrides the parent method to use solve_adjoint_system_scan
        instead of the sequential backward scan.

        Args:
            t_sol: time points, shape (N+1,)
            y_array: states [x, z] at time points, shape (n_total, N+1)
            y_target_use: target outputs, shape (N+1, n_outputs)
            p_opt_vals_jax: all parameter values, shape (n_params_total,)
            step_size: gradient descent step size

        Returns:
            p_opt_new: updated optimized parameters, shape (n_params_opt,)
            grad_p_opt: gradient w.r.t. optimized parameters, shape (n_params_opt,)
        """
        # Step 2: Compute loss gradient dL/dy
        y_array_T = y_array.T  # shape: (N+1, n_total)
        dL_dy = self.jac.trajectory_loss_gradient_analytical(t_sol, y_array_T, y_target_use, p_opt_vals_jax)

        # Scale by 1/N if using mean loss
        if self.loss_type == 'mean':
            n_outputs = y_target_use.shape[1] if y_target_use.shape[0] == t_sol.shape[0] else y_target_use.shape[0]
            n_time = t_sol.shape[0]
            N_total = n_outputs * n_time
            dL_dy = dL_dy / N_total

        # Exclude initial condition which is fixed
        dL_dy_adjoint = dL_dy[1:, :]  # shape: (N, n_total)

        # Step 3: Compute Jacobian blocks (pure JAX version)
        if y_array.shape[0] == self.jac.n_total and y_array.shape[1] == t_sol.shape[0]:
            y_array_for_jac = y_array.T
        else:
            y_array_for_jac = y_array

        N = t_sol.shape[0] - 1  # Number of intervals

        # Prepare arrays for vectorized computation
        t_k = t_sol[:-1]
        t_kp1 = t_sol[1:]
        y_k = y_array_for_jac[:-1]
        y_kp1 = y_array_for_jac[1:]

        # Compute Jacobians in parallel using vmapped functions
        J_prev = self.jac._jac_y_k_vmapped(t_k, t_kp1, y_k, y_kp1, p_opt_vals_jax)
        J_curr = self.jac._jac_y_kp1_vmapped(t_k, t_kp1, y_k, y_kp1, p_opt_vals_jax)

        # Step 4: Solve adjoint system using FAST parallel solver
        # This is the key difference from the parent class
        lambda_adjoint = solve_adjoint_system_scan(J_prev, J_curr, dL_dy_adjoint)

        # Step 5: Compute parameter Jacobian dR/dp_opt (pure JAX version)
        J_param = self.jac._jac_p_vmapped(t_k, t_kp1, y_k, y_kp1, p_opt_vals_jax)

        # Reshape to (N*n_total, n_params_opt)
        dR_dp_opt = J_param.reshape(N * self.jac.n_total, -1)

        # Step 6: Compute parameter gradient: dL/dp = -(dR/dp)^T @ λ
        lambda_flat = lambda_adjoint.flatten()
        grad_p_all = -dR_dp_opt.T @ lambda_flat

        # Extract only gradients for optimized parameters
        grad_p_opt = grad_p_all[self.optimize_indices_jax]

        # Step 7: Gradient descent update
        p_opt_current = p_opt_vals_jax[self.optimize_indices_jax]
        p_opt_new = p_opt_current - step_size * grad_p_opt

        return p_opt_new, grad_p_opt
