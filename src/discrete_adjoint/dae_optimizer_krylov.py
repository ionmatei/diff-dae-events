"""
Matrix-Free Krylov Adjoint Solver for DAE Optimization.

This module implements a memory-efficient adjoint solver using the GMRES iterative method
combined with JAX's vector-Jacobian products (VJP). This avoids materializing the 
full Jacobian matrices, reducing memory complexity from O(T*N^2) to O(T*N).
"""

import jax
import jax.numpy as jnp
from jax import jit, vjp, vmap
from jax.scipy.sparse.linalg import gmres
import numpy as np
from typing import Tuple, Dict, Optional, Union, Callable
from jax.scipy.sparse.linalg import gmres, bicgstab

from src.discrete_adjoint.dae_jacobian import DAEOptimizer, BDF_COEFFICIENTS

# Enable 64-bit precision for stability
jax.config.update("jax_enable_x64", True)

class DAEOptimizerKrylov(DAEOptimizer):
    """
    DAE Optimizer using Matrix-Free Krylov Subspace Methods (GMRES) with Optimized Matvec.
    
    This optimizer solves the adjoint system A^T * lambda = -dL/dy using GMRES.
    
    OPTIMIZATIONS:
    1. Custom Matvec: Computes A^T * v using local VJPs and JAX scan/vmap, 
       avoiding differentiation through the entire trajectory residual.
    2. Efficient BDF: Pre-computes coefficients and uses vectorized operations 
       instead of Python scanning.
    3. Preconditioning: Uses a block-diagonal preconditioner (approximate inverse of result Jacobian)
       to accelerate convergence for stiff systems.
    
    Memory complexity: O(T*N)
    Computational cost per iter: O(T) (but with small constant factor compared to naive VJP)
    """
    
    def __init__(self, *args, krylov_tol=1e-6, krylov_maxiter=100, solver_type='gmres', preconditioner='block_diag', **kwargs):
        super().__init__(*args, **kwargs)
        self.krylov_tol = krylov_tol
        self.krylov_maxiter = krylov_maxiter
        self.solver_type = solver_type.lower()
        self.use_preconditioner = (preconditioner == 'block_diag')
        
        # BDF Setup
        method = self.method.lower()
        if method == 'trapezoidal':
            self.is_bdf = False
            self.bdf_order = 0
            self.bdf_coeffs = None
        elif method == 'backward_euler':
            self.is_bdf = True
            self.bdf_order = 1
            # BDF1: y_n - y_{n-1} -> coeffs [1, -1]
            self.bdf_coeffs = jnp.array([1.0, -1.0], dtype=jnp.float64)
        elif method.startswith('bdf'):
            self.is_bdf = True
            self.bdf_order = int(method[3])
            # Load coeffs
            coeffs_list, _ = BDF_COEFFICIENTS[self.bdf_order]
            self.bdf_coeffs = jnp.array(coeffs_list, dtype=jnp.float64)
            
            # For efficient gathering, we might need a fixed window size
            # We'll use dynamic slicing in the loop/scan
        else:
            raise ValueError(f"Unknown method for Krylov solver: {method}")
            
        # Compile local VJP functions
        self._compile_local_vjps()
        
        # JIT-compile the combined gradient function
        self._compute_gradient_combined_jit = jit(self._compute_gradient_combined)
        
        print(f"Initialized DAEOptimizerKrylov (Optimized) with method={self.method}")
        print(f"  Solver: {self.solver_type}, Precond: {self.use_preconditioner}")
        print(f"  Tol: {self.krylov_tol}, Max iter: {self.krylov_maxiter}")

    def _compile_local_vjps(self):
        """
        Compile efficient local VJP functions for f and g.
        """
        n_states = self.jac.n_states
        n_alg = self.jac.n_alg
        
        def f_wrapper(t, y, p):
            return self.jac.eval_f_jax(t, y[:n_states], y[n_states:], p)
            
        def g_wrapper(t, y, p):
            return self.jac.eval_g_jax(t, y[:n_states], y[n_states:], p)
            
        # vjp_f returned is a function: v -> (grad_t, grad_y, grad_p)
        # We only need grad_y usually.
        # But we need to pass p.
        
        def vjp_f_y(v, t, y, p):
            """Compute (df/dy)^T * v"""
            # vjp returns (primals, vjp_fun)
            _, vjp_fun = vjp(lambda yy: f_wrapper(t, yy, p), y)
            return vjp_fun(v)[0] # gradient w.r.t y
            
        def vjp_g_y(v, t, y, p):
            """Compute (dg/dy)^T * v"""
            _, vjp_fun = vjp(lambda yy: g_wrapper(t, yy, p), y)
            return vjp_fun(v)[0]

        # Vectorize these over time
        # in_axes: v(0), t(0), y(0), p(None)
        self._vjp_f_y_vmapped = jit(vmap(vjp_f_y, in_axes=(0, 0, 0, None)))
        self._vjp_g_y_vmapped = jit(vmap(vjp_g_y, in_axes=(0, 0, 0, None)))

    def _matvec_adjoint(self, lam_flat, t_sol, y_full, p):
        """
        Compute A^T * lambda efficiently for the adjoint system.
        
        A is the Jacobian of the residuals w.r.t y.
        Returns: A^T * lambda
        """
        N = t_sol.shape[0] - 1
        n_total = self.jac.n_total
        lam = lam_flat.reshape(N, n_total)
        
        # Split lambda into differential and algebraic parts
        # Residual structure: [ R_diff(k), R_alg(k) ]
        # So lambda[k] = [ lam_diff[k], lam_alg[k] ]
        lam_diff = lam[:, :self.jac.n_states]
        lam_alg = lam[:, self.jac.n_states:]
        
        # Calculate time steps
        h = t_sol[1:] - t_sol[:-1]
        
        # Initialize output (gradient w.r.t unknowns y_1...y_N)
        # result size should be same as lambda (corresponding to y_1...y_N)
        # Note: residual k matches equation for step k+1 (in standard notation), 
        # basically R_k involves y_k and y_{k+1}.
        # My code indexing: 
        #   y_full = [y0, y1, ..., yN]
        #   residuals = [r0, ..., r_{N-1}] where r_k couples y_k and y_{k+1}
        #   We solve for unknowns y1...yN.
        #   Matrix A = d(residuals)/d(y_1...y_N)
        
        # Contributions to y_j come from residuals dependent on y_j.
        # For 1-step methods (Trapz/Euler), y_j appears in:
        #   - residual r_{j-1} (current step for y_j)
        #   - residual r_j     (previous step for y_{j+1})
        
        # Compute local VJPs for all steps k=0..N-1
        # relevant points: y_{k+1} (target), y_k (prev)
        # We need f and g evals/VJPs at these points.
        
        # Pre-compute VJPs of f and g at all needed points
        # For simplicity, let's compute at all t_1...t_N (which correspond to y_1...y_N)
        # and t_0...t_{N-1} (y_0...y_{N-1})
        
        # Actually, let's look at the residual structure per method.
        
        check_idx = jnp.arange(N)
        
        if not self.is_bdf:
            # --- Trapezoidal ---
            # r_k = (x_{k+1} - x_k)/h - 0.5(f_{k+1} + f_k)
            # alg_k = g_{k+1}
            
            # Terms involving y_{k+1}:
            #   - diff eq k:  1/h * lam_diff[k] - 0.5 * (df_{k+1}/dy)^T * lam_diff[k]
            #   - alg eq k:   (dg_{k+1}/dy)^T * lam_alg[k]
            
            # Terms involving y_k:
            #   - diff eq k: -1/h * lam_diff[k] - 0.5 * (df_k/dy)^T * lam_diff[k]
            
            # So, the output slot j (for y_j, j=1..N) gathers:
            #   From residual j-1 (where y_j is "k+1"):
            #       1/h_{j-1} * lam_diff[j-1] 
            #       - 0.5 * (df(y_j)/dy)^T * lam_diff[j-1]
            #       + (dg(y_j)/dy)^T * lam_alg[j-1]
            #
            #   From residual j (where y_j is "k"):
            #       - 1/h_j * lam_diff[j]
            #       - 0.5 * (df(y_j)/dy)^T * lam_diff[j]
            #       (Only if j < N, i.e. residual j exists)
            
            # Vectorized approach:
            # 1. Term 1 (Main diagonal block contributions)
            #    Need VJP of f at y_1...y_N with vectors -0.5*lam_diff
            #    Need VJP of g at y_1...y_N with vectors lam_alg
            
            y_1_N = y_full[1:]
            t_1_N = t_sol[1:]
            
            # VJP f at k+1
            # We combine the vectors: v_f = -0.5 * lam_diff
            v_f_main = -0.5 * lam_diff
            term_f_main = self._vjp_f_y_vmapped(v_f_main, t_1_N, y_1_N, p)
            
            # VJP g at k+1
            term_g = self._vjp_g_y_vmapped(lam_alg, t_1_N, y_1_N, p)
            
            # 1/h term for k+1
            term_h_main = jnp.zeros_like(term_f_main)
            h_col = h[:, None]
            term_h_main = term_h_main.at[:, :self.jac.n_states].set(lam_diff / h_col)
            
            res_main = term_f_main + term_g + term_h_main
            
            # 2. Term 2 (Off-diagonal / "Next" lambda contributions affecting current y)
            #    y_j affects residual j as "prev state".
            #    Vector: lam_diff[j] (for j=1..N-1)
            #    Inputs: y_1...y_{N-1} (which are y_j for j=1..N-1)
            #    The last y_N does not appear as a "prev state" in any residual (range is 0..N-1)
            
            # Shifted lambdas: lam_diff[1:] corresponds to residuals 1..N-1
            # These affect y_1...y_{N-1}
            lam_diff_next = lam_diff[1:] # size N-1
            y_affected = y_full[1:-1]    # y_1...y_{N-1}
            t_affected = t_sol[1:-1]
            h_next = h[1:]
            
            # VJP f at k (prev)
            # vector: -0.5 * lam_diff_next
            v_f_next = -0.5 * lam_diff_next
            term_f_next = self._vjp_f_y_vmapped(v_f_next, t_affected, y_affected, p)
            
            # -1/h term
            term_h_next = jnp.zeros_like(term_f_next)
            h_next_col = h_next[:, None]
            term_h_next = term_h_next.at[:, :self.jac.n_states].set(-lam_diff_next / h_next_col)
            
            res_off = term_f_next + term_h_next
            
            # Add to result (padding last element with 0 as y_N is not prev step for anything)
            res_off_padded = jnp.vstack([res_off, jnp.zeros((1, n_total))])
            
            return (res_main + res_off_padded).flatten()
            
        else:
            # --- BDF ---
            # r_k = (1/h) * sum(alpha_m * x_{k+1-m}) - f_{k+1}
            # alg_k = g_{k+1}
            
            # y_{k+1} appears in:
            #   - r_k (as coeff alpha_0)  -> Target y_j is y_{k+1} when k=j-1. Coeff alpha_0.
            #   - r_{k+1} (as coeff alpha_1 if order>=1) -> Target y_j is y_{(k+1)-1+1}.. wait. 
            
            # Let's rephrase:
            # We want derivative wrt y_j (j=1..N).
            # y_j appears in residual r_m if r_m uses time step t_{m+1} and history includes t_j.
            # BDF term approx derivative at t_{m+1} uses y_{m+1}, y_m, y_{m-1}...
            # So y_j appears in residuals r_{j-1}, r_j, r_{j+1}, ... r_{j+q-1}
            
            # Contribution from r_{j-1} (where y_j is y_{m+1} i.e. current):
            #   Deriv: (alpha_0 / h_{j-1}) * I  - df(y_j)/dy
            #   Mul by: lam_diff[j-1]
            #   Also g term: (dg(y_j)/dy)^T * lam_alg[j-1]
            
            # Contribution from r_{j-1+m} (where y_j is history term m steps back, m=1..q):
            #   Note: This requires constant step size assumption or careful handling. 
            #   Function assumes fixed step size h usually for simple BDF coeffs. 
            #   If variable h, coeffs change. We assume h is roughly constant or coeffs provided are for fixed h. 
            #   The code in simple examples usually uses fixed h.
            
            #   Term: (alpha_m / h) * x_j
            #   Deriv w.r.t x_j: (alpha_m / h) * I
            #   Mul by: lam_diff[j-1+m]
            
            # So for y_j, we sum:
            #   Main (from r_{j-1}): 
            #       VJP_f(y_j)^T * (-lam_diff[j-1]) 
            #       + VJP_g(y_j)^T * lam_alg[j-1]
            #       + (alpha_0 / h) * lam_diff[j-1] (for x part)
            #   
            #   History (from r_{j}, r_{j+1}...):
            #       sum_{m=1..q} (alpha_m / h) * lam_diff[j-1+m] (for x part)
            #       (Only if j-1+m < N)
            
            # 1. Main terms (local VJPs at y_j)
            y_1_N = y_full[1:]
            t_1_N = t_sol[1:]
            
            v_f = -lam_diff
            term_f = self._vjp_f_y_vmapped(v_f, t_1_N, y_1_N, p)
            term_g = self._vjp_g_y_vmapped(lam_alg, t_1_N, y_1_N, p)
            
            # Alpha_0 term
            alpha_0 = self.bdf_coeffs[0]
            term_bdf_main = jnp.zeros_like(term_f)
            h_col = h[:, None]
            term_bdf_main = term_bdf_main.at[:, :self.jac.n_states].set((alpha_0 / h_col) * lam_diff)
            
            total_res = term_f + term_g + term_bdf_main
            
            # 2. History terms (Linear accumulation)
            # For each BDF lag m=1..order:
            #   Shift lambda by m steps backward (physically forward in residuals) and add
            for m in range(1, len(self.bdf_coeffs)):
                alpha_m = self.bdf_coeffs[m]
                
                # We need lam_diff[j-1+m] applied to y_j
                # if j=1, we need lam_diff[m]
                # shifts:
                # y indices: 1 .. N
                # lam indices: 0 .. N-1
                # y_j pairs with lam[j-1] (main)
                # y_j pairs with lam[j-1+m] (lag m)
                
                # We take lam_diff, shift it left by m.
                # lam_diff_shifted = lam_diff[m:] (size N-m)
                # This affects y_1 .. y_{N-m}
                
                if m >= N: continue
                
                lam_shifted = lam_diff[m:]
                rows_affected = N - m
                
                # We add to total_res[0:rows_affected]
                # Scale by alpha_m / h (approximated, strictly h should be h_{j-1+m}?) 
                # For fixed step size h is constant. We use h[m:] to align with residuals?
                # Actually usage is (alpha_m * x_{k+1-m})/h. 
                # The h is from the residual equation definition r = ... / h. 
                # So we use h corresponding to the residual where lambda lives.
                h_shifted = h[m:, None]
                
                contribution = (alpha_m / h_shifted) * lam_shifted
                
                # efficient update
                current_slice = total_res[:rows_affected, :self.jac.n_states]
                total_res = total_res.at[:rows_affected, :self.jac.n_states].set(current_slice + contribution)
                
            return total_res.flatten()

    def _apply_preconditioner(self, v_flat, t_sol, y_full, p):
        """
        Apply Block-Diagonal Inverse Preconditioner.
        M ~ BlockDiag(J_0, J_1, ..., J_{N-1})
        Where J_k is the local Jacobian of the residual r_k w.r.t y_{k+1}.
        
        For Trapezoidal:
          J_kk = (1/h)*I - 0.5*df_{k+1}/dy  (top-left block)
                 dg_{k+1}/dy                (bottom block)
        
        For BDF:
          J_kk = (alpha_0/h)*I - df_{k+1}/dy
                 dg_{k+1}/dy
        
        We approximate inverse by solving the linear system block-wise.
        """
        if not self.use_preconditioner:
            return v_flat
            
        N = t_sol.shape[0] - 1
        n_total = self.jac.n_total
        v = v_flat.reshape(N, n_total)
        h = t_sol[1:] - t_sol[:-1]
        
        # Get y_{k+1} points (target of residual k)
        y_loc = y_full[1:] 
        t_loc = t_sol[1:]
        
        # Compute local Jacobian blocks J_loc = d(Res_k)/d(y_{k+1})
        # This involves df/dy and dg/dy at y_{k+1}
        # We can use our pre-compiled Jacobian functions from dae_jacobian.py
        # self.jac._jac_f_vmapped_jit(t, y, p) -> (N, n_x, n_total)
        
        df_dy = self.jac._jac_f_vmapped_jit(t_loc, y_loc, p)
        dg_dy = self.jac._jac_g_vmapped_jit(t_loc, y_loc, p) # (N, n_z, n_total)
        
        # Construct blocks
        # Shape (N, n_x, n_total)
        J_blocks_diff = -df_dy
        
        if not self.is_bdf:
            # Trapezoidal: 1/h * I - 0.5 * df/dy
            J_blocks_diff = -0.5 * df_dy
            diag_val = 1.0 / h[:, None, None]
        else:
            # BDF: alpha_0/h * I - df/dy
            alpha_0 = self.bdf_coeffs[0]
            diag_val = alpha_0 / h[:, None, None]
            
        # Add diagonal to x-part of J_blocks_diff
        # J_blocks_diff is (N, nx, nx+nz)
        # We add diag to J_blocks_diff[:, :, :nx]
        
        # Create identity mask
        eye_x = jnp.eye(self.jac.n_states)
        diag_term = diag_val * eye_x[None, :, :]
        
        J_blocks_diff = J_blocks_diff.at[:, :, :self.jac.n_states].add(diag_term)
        
        # Stack with algebraic part
        # Result: (N, nx+nz, nx+nz)
        J_blocks = jnp.concatenate([J_blocks_diff, dg_dy], axis=1)
        
        # Solve M * z = v  => z = M^{-1} v
        # Since we use Adjoint Krylov, we are solving A^T lambda = rhs.
        # The preconditioner M should approximate A^T.
        # Our J_blocks are approximations of dl/dy_{k+1} blocks of A.
        # A is lower triangular (or block lower). Diagonal blocks are J_kk.
        # A^T is upper triangular. Diagonal blocks are J_kk^T.
        # So we should be inverting J_kk^T.
        
        J_blocks_T = jnp.transpose(J_blocks, (0, 2, 1))
        
        # Solve block-wise
        z = vmap(jnp.linalg.solve)(J_blocks_T, v)
        
        return z.flatten()

    def _compute_gradient_combined(
        self,
        t_sol: jnp.ndarray,
        y_array: jnp.ndarray,
        y_target_use: jnp.ndarray,
        p_opt_vals_jax: jnp.ndarray,
        step_size: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute gradient using Optimized Matrix-Free GMRES adjoint solver.
        """
        # Ensure correct shape
        if y_array.shape[0] == self.jac.n_total and y_array.shape[1] == t_sol.shape[0]:
            y_array_loss = y_array.T
        else:
            y_array_loss = y_array
            
        # 1. Compute dL/dy (RHS)
        dL_dy = self.jac.trajectory_loss_gradient_analytical(
            t_sol, y_array_loss, y_target_use, p_opt_vals_jax
        )
        
        if self.loss_type == 'mean':
            n_outputs = y_target_use.shape[1] if len(y_target_use.shape) > 1 else y_target_use.shape[0]
            n_time = t_sol.shape[0]
            dL_dy = dL_dy / (n_outputs * n_time)
            
        # RHS for adjoint: -dL/dy (unknowns only)
        rhs_vec = -dL_dy[1:].flatten()
        
        # 2. Setup Linear Operator
        # Matvec: v -> A^T v
        def matvec(v):
            # jax.debug.print("M") 
            return self._matvec_adjoint(v, t_sol, y_array_loss, p_opt_vals_jax)
            
        # Preconditioner
        if self.use_preconditioner:
            def precond(v):
                return self._apply_preconditioner(v, t_sol, y_array_loss, p_opt_vals_jax)
            M = precond
        else:
            M = None
            
        # 3. Solve
        if self.solver_type == 'gmres':
            lambda_flat, info = gmres(
                matvec, 
                rhs_vec, 
                M=M,
                tol=self.krylov_tol, 
                maxiter=self.krylov_maxiter,
                restart=30 
            )
        elif self.solver_type == 'bicgstab':
             lambda_flat, info = bicgstab(
                matvec, 
                rhs_vec, 
                M=M,
                tol=self.krylov_tol, 
                maxiter=self.krylov_maxiter
            )
        else:
             raise ValueError(f"Unknown solver: {self.solver_type}")

        # 4. Compute gradient w.r.t parameters
        # grad_p = lambda^T * (dR/dp)
        # We can use VJP of residual w.r.t p.
        # But wait, residual function is global. Can we optimize this too?
        # dR/dp structure: computed locally at each step.
        # r_k depends on p via f(t, y, p) and g(t, y, p).
        # We can accumulate local VJPs w.r.t p.
        
        check_idx = jnp.arange(t_sol.shape[0] - 1)
        lam_reshaped = lambda_flat.reshape(-1, self.jac.n_total)
        lam_diff = lam_reshaped[:, :self.jac.n_states]
        lam_alg = lam_reshaped[:, self.jac.n_states:]
        
        # Need VJP of f and g w.r.t p, contracted with lambda
        # All residuals r_k depend on p directly via f and g terms.
        # r_k includes -0.5*f... or -f...
        # So we just sum up VJP contributions.
        
        t_loc = t_sol[1:]
        y_loc = y_array_loss[1:]
        
        # We need another VJP function for p
        def vjp_f_p(v, t, y, p):
            _, vjp_fun = vjp(lambda pp: self.jac.eval_f_jax(t, y[:self.jac.n_states], y[self.jac.n_states:], pp), p)
            return vjp_fun(v)[0]
            
        def vjp_g_p(v, t, y, p):
            _, vjp_fun = vjp(lambda pp: self.jac.eval_g_jax(t, y[:self.jac.n_states], y[self.jac.n_states:], pp), p)
            return vjp_fun(v)[0]
            
        vjp_f_p_map = vmap(vjp_f_p, in_axes=(0, 0, 0, None))
        vjp_g_p_map = vmap(vjp_g_p, in_axes=(0, 0, 0, None))
        
        # Weights for f-contribution
        if not self.is_bdf:
            # Trapezoidal: r_k has -0.5*f_{k+1} and -0.5*f_k
            # y_loc contains y_1...y_N
            # lam_reshaped contains lam_0...lam_{N-1} corresponding to r_0...r_{N-1}
            # f_{k+1} (at y_{k+1}) hits r_k with -0.5*lam_k
            # f_k (at y_k) hits r_k with -0.5*lam_k
            # Wait, easier to group by time point.
            # f at y_j (time j) contributes to:
            #   r_{j-1} (as f_{k+1}) -> -0.5 * lam_{j-1}
            #   r_j     (as f_k)     -> -0.5 * lam_j
            # (check boundary conditions)
            
            # Vector of weights for each f_j
            # lam_diff is size N.
            # weights = -0.5 * lam_diff (from prev) + -0.5 * [lam_diff[1:], 0] (from next)?
            # No, index logic:
            # r_k uses f_{k+1} and f_k.
            # lam_k multiplies r_k.
            # term: lam_k^T * (-0.5*f_{k+1} - 0.5*f_k)
            # sum_k [ -0.5*lam_k * f_{k+1} - 0.5*lam_k * f_k ]
            # = sum_j f_j * weight_j
            # coeff for f_1: -0.5*lam_0 - 0.5*lam_1
            # coeff for f_N: -0.5*lam_{N-1}
            
            # Let's just vectorize the VJPs of term r_k w.r.t p and sum them.
            # r_k term w.r.t p is: -0.5*df(y_{k+1})/dp - 0.5*df(y_k)/dp + dg/dp...
            
            # Actually simpler: just re-use vjp_f_p_map on chunks
            # Terms: -0.5 * (f(y_1..N) + f(y_0..N-1))
            vec_1 = -0.5 * lam_diff
            vec_0 = -0.5 * lam_diff
            
            grad_p_f1 = jnp.sum(vjp_f_p_map(vec_1, t_sol[1:], y_array_loss[1:], p_opt_vals_jax), axis=0)
            grad_p_f0 = jnp.sum(vjp_f_p_map(vec_0, t_sol[:-1], y_array_loss[:-1], p_opt_vals_jax), axis=0)
            grad_p_g  = jnp.sum(vjp_g_p_map(lam_alg, t_sol[1:], y_array_loss[1:], p_opt_vals_jax), axis=0)
            
            grad_p_opt = grad_p_f1 + grad_p_f0 + grad_p_g
            
            # Don't forget outputs h(t,y,p) might depend on p? 
            # Loss is L(y, p). We already added dL_direct/dp in previous implementations?
            # Base class usually handles partial dL/dp? 
            # In DAEOptimizerKrylov original, it did: grad_p_opt = vjp_fun_p(lambda)[0]
            # which correctly captures "lambda^T * dR/dp".
            # The base class `trajectory_loss_gradient_analytical` gives dL/dy.
            # What about dL/dp (explicit)? 
            # If loss depends explicitly on p, we must add it.
            # DAEOptimizer base doesn't seem to calculate dL/dp explicit.
            # But usually parameters are in physics, not loss function directly (except regularization).
            # Regularization is handled outside usually.
            
        else:
            # BDF: r_k has -f_{k+1}.
            # lam_k multiplies -f_{k+1}.
            # coeff for f_{k+1} is -lam_k.
            
            vec = -lam_diff
            grad_p_f = jnp.sum(vjp_f_p_map(vec, t_sol[1:], y_array_loss[1:], p_opt_vals_jax), axis=0)
            grad_p_g = jnp.sum(vjp_g_p_map(lam_alg, t_sol[1:], y_array_loss[1:], p_opt_vals_jax), axis=0)
            
            grad_p_opt = grad_p_f + grad_p_g
            
        # 5. Update
        p_opt_new = p_opt_vals_jax - step_size * grad_p_opt
        
        return p_opt_new, grad_p_opt 

