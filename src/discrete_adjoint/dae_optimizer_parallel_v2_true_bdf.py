"""
Parallel DAE Optimizer V2 with TRUE BDF adjoint (not approximation).

Key improvements:
1. On-the-fly Jacobian computation (memory efficient)
2. TRUE BDF adjoint with proper multi-step coupling
3. Companion matrix construction for O(log N) parallel scan

This module provides the correct discrete adjoint for BDF methods by properly
accounting for the multi-step nature of the discretization.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Tuple, Dict, List
import numpy as np
import time

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

from .dae_jacobian import DAEOptimizer
from src.deer.maths import matmul_recursive

# BDF coefficients (same as in dae_jacobian.py)
BDF_COEFFICIENTS = {
    1: ([1.0, -1.0], 1.0),  # Backward Euler
    2: ([3.0/2.0, -2.0, 1.0/2.0], 1.0),
    3: ([11.0/6.0, -3.0, 3.0/2.0, -1.0/3.0], 1.0),
    4: ([25.0/12.0, -4.0, 3.0, -4.0/3.0, 1.0/4.0], 1.0),
    5: ([137.0/60.0, -5.0, 5.0, -10.0/3.0, 5.0/4.0, -1.0/5.0], 1.0),
    6: ([147.0/60.0, -6.0, 15.0/2.0, -20.0/3.0, 15.0/4.0, -6.0/5.0, 1.0/6.0], 1.0),
}


class DAEOptimizerParallelV2TrueBDF(DAEOptimizer):
    """
    DAE optimizer with true BDF adjoint using companion matrix for parallel scan.

    For BDF-q, the forward discretization is:
        coeffs[0]*y_{k+1} + coeffs[1]*y_k + ... + coeffs[q]*y_{k+1-q} = h * f(y_{k+1})

    The true discrete adjoint couples q+1 consecutive adjoint variables.
    We transform this into a 2-term recurrence using companion matrix.

    Key features:
    - Correct BDF adjoint (not approximation)
    - Memory-efficient parallel Jacobian computation
    - O(log N) parallel scan via companion matrix
    """

    def __init__(self, *args, sequential_fallback_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Default fallback config
        if sequential_fallback_config is None:
            self.sequential_fallback = {'enable': True, 'threshold': 500}
        else:
            self.sequential_fallback = sequential_fallback_config

        # Determine discretization method
        method = self.method.lower()

        if method == 'trapezoidal':
            self.is_bdf = False
            self.bdf_order = None
        elif method == 'backward_euler' or method == 'bdf1':
            self.is_bdf = True
            self.bdf_order = 1
        elif method.startswith('bdf'):
            self.is_bdf = True
            self.bdf_order = int(method[3])
        else:
            raise ValueError(f"Unknown method: {method}")

        if self.is_bdf and self.bdf_order > 1:
            print(f"  Using TRUE BDF{self.bdf_order} adjoint with companion matrix")
            print(f"  Adjoint coupling: {self.bdf_order + 1} consecutive lambda values")
            print(f"  Parallel scan: O(log N) via augmented companion matrix")
        else:
            print(f"  Using {method} with O(log N) parallel adjoint solve")

        # Get BDF coefficients if applicable
        if self.is_bdf:
            self.bdf_coeffs, _ = BDF_COEFFICIENTS[self.bdf_order]
            self.bdf_coeffs = jnp.array(self.bdf_coeffs, dtype=jnp.float64)

        # JIT-compile the combined gradient function
        self._compute_gradient_combined_jit = jit(self._compute_gradient_combined)

    def _compute_trapezoidal_adjoint_onthefly(
        self,
        t_k: jnp.ndarray,
        t_kp1: jnp.ndarray,
        y_k: jnp.ndarray,
        y_kp1: jnp.ndarray,
        p_opt_vals_jax: jnp.ndarray,
        dL_dy_adjoint: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute adjoint matrices for trapezoidal method on-the-fly (memory efficient).

        Adjoint equation: lambda[k] = M[k] @ lambda[k+1] + v[k]
        where:
            M[k] = -(J_curr[k]^T)^{-1} @ J_prev[k+1]^T
            v[k] = (J_curr[k]^T)^{-1} @ dL_dy[k]
        """
        N = t_k.shape[0]

        # Compute J_curr for all intervals (needed for v_all)
        J_curr = self.jac._jac_y_kp1_vmapped(t_k, t_kp1, y_k, y_kp1, p_opt_vals_jax)
        J_curr_T = jnp.transpose(J_curr, (0, 2, 1))

        # Compute v[k] = (J_curr[k]^T)^{-1} @ dL_dy[k]
        v_all = vmap(jnp.linalg.solve)(J_curr_T, dL_dy_adjoint)

        # Only compute shifted J_prev (memory efficient!)
        t_k_shift = t_k[1:]
        t_kp1_shift = t_kp1[1:]
        y_k_shift = y_k[1:]
        y_kp1_shift = y_kp1[1:]

        J_prev_shift = self.jac._jac_y_k_vmapped(t_k_shift, t_kp1_shift,
                                                  y_k_shift, y_kp1_shift, p_opt_vals_jax)
        J_prev_T_shift = jnp.transpose(J_prev_shift, (0, 2, 1))

        # Compute M[k] = -(J_curr[k]^T)^{-1} @ J_prev[k+1]^T
        J_curr_T_m = J_curr_T[:-1]
        M_blocks = -vmap(jnp.linalg.solve)(J_curr_T_m, J_prev_T_shift)

        return M_blocks, v_all

    def _compute_bdf_adjoint_true(
        self,
        t_k: jnp.ndarray,
        t_kp1: jnp.ndarray,
        y_k: jnp.ndarray,
        y_kp1: jnp.ndarray,
        p_opt_vals_jax: jnp.ndarray,
        dL_dy_adjoint: jnp.ndarray,
        y_array_full: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute TRUE BDF adjoint with companion matrix for parallel scan.

        For BDF-q, the residual at step k+1 is:
            R_{k+1} = (coeffs[0]*y_{k+1} + coeffs[1]*y_k + ... + coeffs[q]*y_{k+1-q}) / h - f(y_{k+1})

        The discrete adjoint satisfies:
            lambda[k] = sum_{j=max(0,k-q+1)}^{k} (dR_{j+1}/dy_k)^T @ lambda[j+1]

        This couples q+1 consecutive lambda values. We transform to companion form:
            Lambda_aug[k] = M_aug[k] @ Lambda_aug[k+1] + v_aug[k]

        where Lambda_aug[k] = [lambda[k]; lambda[k+1]; ...; lambda[k+q-1]]

        Args:
            t_k, t_kp1: Time points
            y_k, y_kp1: Current and next states
            p_opt_vals_jax: Parameters
            dL_dy_adjoint: Loss gradient w.r.t. states (N, n)
            y_array_full: Full state trajectory for history (N+1, n)

        Returns:
            M_aug_blocks: Augmented transition matrices (N-1, q*n, q*n)
            v_aug_all: Augmented RHS vectors (N, q*n)
        """
        N = t_k.shape[0]
        n = dL_dy_adjoint.shape[1]
        q = self.bdf_order
        h = t_kp1 - t_k  # time steps

        # Step 1: Compute J_curr (dR_{k+1}/dy_{k+1}) for all k with TRUE BDF formulation
        # For BDF: dR/dy_{k+1} = (coeffs[0] / h) * I_diff - df/dy
        # We need to compute this correctly, not use the backward Euler fallback

        # First, get df/dy and dg/dy at each (t_{k+1}, y_{k+1})
        def compute_bdf_jac_y_kp1(t_kp1_single, y_kp1_single, h_single, p):
            """Compute TRUE BDF Jacobian dR_{k+1}/dy_{k+1}."""
            x_kp1 = y_kp1_single[:self.jac.n_states]
            z_kp1 = y_kp1_single[self.jac.n_states:]

            # Compute df/dy and dg/dy using JAX autodiff
            def f_func(y):
                x = y[:self.jac.n_states]
                z = y[self.jac.n_states:]
                return self.jac.eval_f_jax(t_kp1_single, x, z, p)

            def g_func(y):
                x = y[:self.jac.n_states]
                z = y[self.jac.n_states:]
                return self.jac.eval_g_jax(t_kp1_single, x, z, p)

            df_dy = jax.jacfwd(f_func)(y_kp1_single)  # (n_states, n_total)
            dg_dy = jax.jacfwd(g_func)(y_kp1_single)  # (n_alg, n_total)

            # Build dR/dy_{k+1}
            # Top block (differential): (coeffs[0] / h) * I - df/dy
            # Bottom block (algebraic): dg/dy
            bdf_coeff = self.bdf_coeffs[0] / h_single

            dR_diff = jnp.zeros((self.jac.n_states, n), dtype=jnp.float64)
            # Diagonal part for differential states
            dR_diff = dR_diff.at[:, :self.jac.n_states].set(bdf_coeff * jnp.eye(self.jac.n_states))
            # Subtract df/dy
            dR_diff = dR_diff - df_dy

            # Algebraic part
            dR_alg = dg_dy

            return jnp.vstack([dR_diff, dR_alg])

        # Vectorize over all intervals (map over t, y, h; broadcast p)
        J_curr = vmap(compute_bdf_jac_y_kp1, in_axes=(0, 0, 0, None))(t_kp1, y_kp1, h, p_opt_vals_jax)
        J_curr_T = jnp.transpose(J_curr, (0, 2, 1))

        # Step 2: Compute TRUE BDF history Jacobians
        # For adjoint at k, we need dR_{k+j}/dy_k for j=1..q
        # For BDF: dR_{k+j}/dy_k = (coeffs[j] / h_{k+j}) * I_diff
        # This only affects differential states; algebraic equations don't depend on history

        def compute_bdf_history_jacobian(h_single, coeff_j):
            """Compute dR_{k+j}/dy_k = (coeff_j / h_{k+j}) * I_diff."""
            J_hist = jnp.zeros((n, n), dtype=jnp.float64)
            # Only differential states are affected
            J_hist = J_hist.at[:self.jac.n_states, :self.jac.n_states].set(
                (coeff_j / h_single) * jnp.eye(self.jac.n_states)
            )
            return J_hist

        # Compute history Jacobians for each future residual
        # J_history_list[j-1] contains dR_{k+j}/dy_k for j=1..q-1
        J_history_list = []
        for j in range(1, q + 1):
            # For lambda[k], we need dR_{k+j}/dy_k
            # This depends on h_{k+j} and coeffs[j]
            # Valid for k where k+j <= N (i.e., k <= N-j)
            num_valid = max(0, N - j)

            if num_valid > 0:
                # h[0], h[1], ..., h[N-j-1] but we need h[j], h[j+1], ..., h[N-1]
                # Actually, for lambda[k], dR_{k+j}/dy_k uses h_{k+j}
                # So for k=0, we use h[j]; for k=1, we use h[j+1], etc.
                h_subset = h[j:j+num_valid]  # shape: (num_valid,)
                coeff_j = self.bdf_coeffs[j]
                J_hist_subset = vmap(lambda h_k: compute_bdf_history_jacobian(h_k, coeff_j))(h_subset)

                # Pad to size (N-1, n, n) for uniform indexing
                # Note: we use N-1 because that's the number of lambda values (lambda[0] through lambda[N-2])
                J_hist_padded = jnp.zeros((N - 1, n, n), dtype=jnp.float64)
                J_hist_padded = J_hist_padded.at[:num_valid].set(J_hist_subset)
                J_history_list.append(J_hist_padded)
            else:
                J_history_list.append(jnp.zeros((N - 1, n, n), dtype=jnp.float64))


        # Step 3: Build augmented adjoint system
        # For the companion matrix approach, we need:
        # - Augmented state: Lambda_aug[k] = [lambda[k]; lambda[k+1]; ...; lambda[k+q-1]]
        # - Augmented equation: Lambda_aug[k] = M_aug[k] @ Lambda_aug[k+1] + v_aug[k]

        # The true BDF adjoint equation at step k is:
        # lambda[k] = (dR_{k+1}/dy_k)^T @ lambda[k+1] + (dR_{k+2}/dy_k)^T @ lambda[k+2] + ...

        # For BDF, dR_{j+1}/dy_k = coeffs[j+1-k] / h[j] * I_diff (for k < j+1 <= k+q)

        # Compute base v[k] from direct loss gradient
        v_base = vmap(jnp.linalg.solve)(J_curr_T, dL_dy_adjoint)

        # Build augmented v: v_aug[k] = [v[k]; 0; 0; ...; 0]
        v_aug_all = jnp.zeros((N, q * n), dtype=jnp.float64)
        v_aug_all = v_aug_all.at[:, :n].set(v_base)

        # Step 4: Build companion M_aug matrices with TRUE BDF Jacobians
        # The adjoint equation is: lambda[k] = sum_{j=1}^{q} (dR_{k+j}/dy_k)^T @ lambda[k+j] + (dL/dy_k)
        # In compact form: lambda[k] = -(J_curr[k]^T)^{-1} @ sum_{j=1}^{q} J_hist_j[k]^T @ lambda[k+j]

        # For the companion matrix, we transform to:
        # Lambda_aug[k] = M_aug[k] @ Lambda_aug[k+1] + v_aug[k]
        # where Lambda_aug[k] = [lambda[k]; lambda[k+1]; ...; lambda[k+q-1]]

        # M_aug has structure:
        # [M_00  M_01  M_02  ...  M_0,q-1]  <- lambda[k] coupling
        # [I     0     0     ...  0      ]  <- lambda[k+1] = lambda[k+1]
        # [0     I     0     ...  0      ]  <- lambda[k+2] = lambda[k+2]
        # ...

        # Build M_00, M_01, ..., M_0,q-1 blocks
        # M_0j = -(J_curr[k]^T)^{-1} @ J_hist_j[k]^T for j=1..q-1

        M_coupling_list = []
        for j in range(q):
            # J_hist_j is already padded to (N-1, n, n)
            J_hist_j = J_history_list[j]  # (N-1, n, n)
            J_hist_j_T = jnp.transpose(J_hist_j, (0, 2, 1))

            # For proper adjoint indexing:
            # lambda[k] couples with lambda[k+1] via J_hist_1
            # lambda[k] couples with lambda[k+2] via J_hist_2, etc.
            # So for k in [0, N-2], we need J_curr[k] and J_hist_j for the appropriate shift

            # Use J_curr[:-1] (i.e., J_curr at k=0..N-2) to build M matrices
            J_curr_T_subset = J_curr_T[:-1]  # (N-1, n, n)

            # Compute M_0j[k] = -(J_curr[k]^T)^{-1} @ J_hist_j[k]^T
            M_0j = -vmap(jnp.linalg.solve)(J_curr_T_subset, J_hist_j_T)  # (N-1, n, n)
            M_coupling_list.append(M_0j)

        def build_bdf_companion_matrix(k_idx, *M_coupling_k):
            """Build companion matrix for adjoint at step k.

            Args:
                k_idx: Step index (not used, but kept for vmap structure)
                M_coupling_k: Coupling blocks M_00, M_01, ..., M_0,q-2 (each is (n, n))
            """
            M_aug = jnp.zeros((q * n, q * n), dtype=jnp.float64)

            # Top block row: M_0j blocks go in [:n, j*n:(j+1)*n] for j=0..q-2
            for j, M_j in enumerate(M_coupling_k):
                M_aug = M_aug.at[:n, j*n:(j+1)*n].set(M_j)

            # Shift blocks: Identity matrices for history propagation
            # lambda[k+1] from Lambda_aug[k+1] goes to position [k+1] in Lambda_aug[k]
            for i in range(q - 1):
                M_aug = M_aug.at[(i+1)*n:(i+2)*n, i*n:(i+1)*n].set(jnp.eye(n))

            return M_aug

        # Build all companion matrices using vmap
        # M_coupling_list has q-1 elements: [M_00, M_01, ..., M_0,q-2]
        if len(M_coupling_list) > 0:
            # Unpack M_coupling_list for vmap
            M_aug_blocks = vmap(build_bdf_companion_matrix)(
                jnp.arange(N - 1),
                *M_coupling_list
            )
        else:
            # Fallback for q=1 (shouldn't reach here as q>=2 for BDF)
            M_aug_blocks = jnp.zeros((N - 1, n, n), dtype=jnp.float64)

        return M_aug_blocks, v_aug_all

    def _compute_gradient_combined(
        self,
        t_sol: jnp.ndarray,
        y_array: jnp.ndarray,
        y_target_use: jnp.ndarray,
        p_opt_vals_jax: jnp.ndarray,
        step_size: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Combined optimization step with TRUE BDF adjoint.

        Uses companion matrix to enable O(log N) parallel scan for BDF methods.
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

        # Step 3: Prepare state arrays
        if y_array.shape[0] == self.jac.n_total and y_array.shape[1] == t_sol.shape[0]:
            y_array_for_jac = y_array.T
        else:
            y_array_for_jac = y_array

        N = t_sol.shape[0] - 1

        t_k = t_sol[:-1]
        t_kp1 = t_sol[1:]
        y_k = y_array_for_jac[:-1]
        y_kp1 = y_array_for_jac[1:]

        if self.is_bdf and self.bdf_order > 1:
            # TRUE BDF adjoint with companion matrix
            # Use sequential scan for large N to avoid O(N * (qn)^2) memory usage
            use_sequential = (
                self.sequential_fallback['enable'] and 
                N > self.sequential_fallback['threshold']
            )
            
            if use_sequential:
                lambda_adjoint = self._solve_bdf_adjoint_sequential(
                    t_k, t_kp1, y_k, y_kp1, p_opt_vals_jax, dL_dy_adjoint
                )
            else:
                M_blocks, v_all = self._compute_bdf_adjoint_true(
                    t_k, t_kp1, y_k, y_kp1, p_opt_vals_jax, dL_dy_adjoint, y_array_for_jac
                )
                
                # Step 5: Solve adjoint system using parallel scan
                y0 = v_all[-1]
                vecs = v_all[:-1][::-1]
                mats = M_blocks[::-1]
    
                if N == 1:
                    lambda_aug = v_all
                else:
                    y_rev = matmul_recursive(mats, vecs, y0)
                    lambda_aug = y_rev[::-1]
                    
                n_phys = self.jac.n_total
                lambda_adjoint = lambda_aug[:, :n_phys]
        else:
            # Trapezoidal or BDF1 (backward Euler)
            M_blocks, v_all = self._compute_trapezoidal_adjoint_onthefly(
                t_k, t_kp1, y_k, y_kp1, p_opt_vals_jax, dL_dy_adjoint
            )
            
            # Step 5: Solve adjoint system using parallel scan
            y0 = v_all[-1]
            vecs = v_all[:-1][::-1]
            mats = M_blocks[::-1]
    
            if N == 1:
                lambda_adjoint = v_all
            else:
                y_rev = matmul_recursive(mats, vecs, y0)
                lambda_adjoint = y_rev[::-1]

        # Step 6: Compute parameter Jacobian
        J_param = self.jac._jac_p_vmapped(t_k, t_kp1, y_k, y_kp1, p_opt_vals_jax)
        dR_dp_opt = J_param.reshape(N * self.jac.n_total, -1)

        # Step 7: Compute parameter gradient
        lambda_flat = lambda_adjoint.flatten()
        grad_p_opt = -dR_dp_opt.T @ lambda_flat

        # Step 8: Update
        p_opt_new = p_opt_vals_jax - step_size * grad_p_opt

        return p_opt_new, grad_p_opt

    def _solve_bdf_adjoint_sequential(
        self,
        t_k: jnp.ndarray,
        t_kp1: jnp.ndarray,
        y_k: jnp.ndarray,
        y_kp1: jnp.ndarray,
        p_opt_vals_jax: jnp.ndarray,
        dL_dy_adjoint: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Solve BDF adjoint system sequentially to save memory.
        
        Computes matrices on-the-fly inside jax.lax.scan.
        Memory usage: O(N * n) + O((qn)^2) instead of O(N * (qn)^2).
        """
        N = t_k.shape[0]
        n = dL_dy_adjoint.shape[1]
        q = self.bdf_order
        h_all = t_kp1 - t_k
        
        # Helper to compute Jacobians for a singe step k
        def compute_step_matrices(k, t_kp1_k, y_kp1_k, h_k, dL_dy_k):
            # 1. Compute J_curr (same logic as in _compute_bdf_adjoint_true)
            x_kp1 = y_kp1_k[:self.jac.n_states]
            z_kp1 = y_kp1_k[self.jac.n_states:]

            def f_func(y):
                x = y[:self.jac.n_states]
                z = y[self.jac.n_states:]
                return self.jac.eval_f_jax(t_kp1_k, x, z, p_opt_vals_jax)

            def g_func(y):
                x = y[:self.jac.n_states]
                z = y[self.jac.n_states:]
                return self.jac.eval_g_jax(t_kp1_k, x, z, p_opt_vals_jax)

            df_dy = jax.jacfwd(f_func)(y_kp1_k)
            dg_dy = jax.jacfwd(g_func)(y_kp1_k)
            
            bdf_coeff = self.bdf_coeffs[0] / h_k
            dR_diff = jnp.zeros((self.jac.n_states, n), dtype=jnp.float64)
            dR_diff = dR_diff.at[:, :self.jac.n_states].set(bdf_coeff * jnp.eye(self.jac.n_states))
            dR_diff = dR_diff - df_dy
            dR_alg = dg_dy
            J_curr = jnp.vstack([dR_diff, dR_alg])
            J_curr_T = J_curr.T
            
            # 2. Compute v[k]
            v_base = jnp.linalg.solve(J_curr_T, dL_dy_k)
            v_aug = jnp.zeros(q * n, dtype=jnp.float64)
            v_aug = v_aug.at[:n].set(v_base)
            
            # 3. Compute M[k] blocks
            # We need J_hist matrices for j=1..q-1
            # J_hist_j needs h[k+j]
            # We need to access global h_all. 
            # Check bounds: if k+j >= N, J_hist is zero.
            
            M_aug = jnp.zeros((q * n, q * n), dtype=jnp.float64)
            
            # Fill Identity blocks
            for i in range(q - 1):
                M_aug = M_aug.at[(i+1)*n:(i+2)*n, i*n:(i+1)*n].set(jnp.eye(n))
                
            # Fill M_0j blocks
            # M_0j = -(J_curr^T)^{-1} @ J_hist_j^T
            J_curr_T_inv = jnp.linalg.inv(J_curr_T) # Or solve
            
            for j in range(1, q + 1):
                # We need h[k+j]
                # If k+j >= N, then J_hist is 0
                
                # Check bounds valid
                # For dynamic check in JAX, we can use jnp.where or just array access with clamping
                # but h_all has size N. Indices 0..N-1.
                # if k+j < N: use h[k+j]
                
                def get_J_hist(idx):
                    h_val = h_all[idx]
                    coeff_j = self.bdf_coeffs[j]
                    J_hist = jnp.zeros((n, n), dtype=jnp.float64)
                    J_hist = J_hist.at[:self.jac.n_states, :self.jac.n_states].set(
                        (coeff_j / h_val) * jnp.eye(self.jac.n_states)
                    )
                    return J_hist
                    
                # Use lax.cond to handle bounds
                is_valid = (k + j) < N
                J_hist = jax.lax.cond(
                    is_valid,
                    lambda _: get_J_hist(k+j),
                    lambda _: jnp.zeros((n, n), dtype=jnp.float64),
                    operand=None
                )
                
                J_hist_T = J_hist.T
                M_0j = -J_curr_T_inv @ J_hist_T
                
                # Place in M_aug
                # j starts at 1. Corresponds to lambda[k+j].
                # lambda[k+1] is at index 0 in previous augment (actually index 0 of coupled).
                # Lambda_aug[k+1] = [lambda[k+1], lambda[k+2]...]
                # M_0,0 multiplies lambda[k+1].
                # So j-1 is the index in M_coupling_list.
                # Col indices: (j-1)*n : j*n
                col_idx = j - 1
                M_aug = M_aug.at[:n, col_idx*n:(col_idx+1)*n].set(M_0j)
                
            return M_aug, v_aug

        # Base case: Compute lambda[N-1]
        # This corresponds to step k = N-1.
        # k+j is always >= N for j>=1. So M_0j are all 0.
        # So M_aug has zeros in top row.
        # lambda_aug[N-1] = M[N-1] @ lambda[N] + v[N-1].
        # If we assume lambda[N] = 0 (final condition), then lambda_aug[N-1] = v[N-1].
        
        M_last, v_last = compute_step_matrices(N-1, t_kp1[N-1], y_kp1[N-1], h_all[N-1], dL_dy_adjoint[N-1])
        lambda_aug_last = v_last # assuming lambda_next is 0
        
        # Scan from N-2 down to 0
        def scan_fun(carry, k):
            lambda_next_aug = carry
            M_k, v_k = compute_step_matrices(k, t_kp1[k], y_kp1[k], h_all[k], dL_dy_adjoint[k])
            lambda_curr_aug = M_k @ lambda_next_aug + v_k
            
            # We want to return physical lambda[k] as output
            lambda_phys = lambda_curr_aug[:n]
            return lambda_curr_aug, lambda_phys
            
        # Scan range: N-2, N-3, ..., 0
        scan_indices = jnp.arange(N-2, -1, -1)
        
        if N > 1:
            final_carry, stacked_lambdas = jax.lax.scan(scan_fun, lambda_aug_last, scan_indices)
            # stacked_lambdas is [lambda[N-2], ..., lambda[0]]
            # We need to reverse it to [lambda[0], ..., lambda[N-2]]
            # And append lambda[N-1]
            
            lambdas_rev = stacked_lambdas[::-1]
            lambda_last_phys = lambda_aug_last[:n]
            
            # Concatenate
            lambda_all = jnp.vstack([lambdas_rev, lambda_last_phys[None, :]])
        else:
            lambda_all = lambda_aug_last[:n][None, :]
            
        return lambda_all
