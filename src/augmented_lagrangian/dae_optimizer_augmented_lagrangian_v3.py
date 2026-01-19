"""
Augmented Lagrangian DAE Optimizer V3 - Clean Implementation from Scratch

Implements the Augmented Lagrangian method for DAE parameter estimation
as described in algorithm_3.tex (Option C).

Key components:
1. Augmented Lagrangian function L_mu(w, theta, eta)
2. Gradient of L_mu with respect to w (state trajectory)
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import numpy as np
from typing import Tuple, Dict, List, Optional
import time

from src.discrete_adjoint.dae_jacobian import DAEOptimizer

class DAEOptimizerAugmentedLagrangianV3(DAEOptimizer):
    """
    Augmented Lagrangian optimizer for DAE parameter identification.
    
    Minimizes:
        L_mu(w, theta, eta) = Phi(w, theta) + eta^T R(w, theta) + (mu/2) ||R(w, theta)||^2
        
    where:
        w: State trajectory (differential and algebraic variables)
        theta: Parameters
        eta: Lagrange multipliers for the DAE residuals
        Phi: Objective function (trajectory loss)
        R: DAE residuals (defect constraints)
    """
    
    def __init__(
        self,
        dae_data: dict,
        dae_solver=None,
        optimize_params: List[str] = None,
        loss_type: str = 'sum',
        method: str = 'trapezoidal',
        verbose: bool = True
    ):
        """Initialize optimizer."""
        super().__init__(
            dae_data=dae_data,
            dae_solver=dae_solver,
            optimize_params=optimize_params,
            loss_type=loss_type,
            method=method
        )
        
        self.verbose = verbose
        if self.verbose:
            print(f"Augmented Lagrangian V3 Initialized (Method: {self.method})")
            
        # IMPORTANT: DAEJacobian is configured by default (in super().__init__) 
        # to expect reduced parameter vectors if optimize_params is set.
        # However, this class manages parameter reconstruction manually and passes 
        # the FULL parameter vector to residuals.
        # We must disable selective optimization in self.jac so it accepts p_all correctly.
        self.jac.optimize_indices = None
        self.jac.p_all_default = None
            
        # Compile the AL and gradient functions
        self._compile_al_functions()
        
    def _compile_al_functions(self):
        """Compile JAX functions for AL computation."""
        
        # 1. Residual function (vectorized over time)
        # We assume self.jac.residual_single is available (from DAEJacobian)
        # residual_single(t_k, t_kp1, y_k, y_kp1, p)
        self._residual_vmap = jit(vmap(self.jac.residual_single, in_axes=(0, 0, 0, 0, None)))
        
        # 2. Augmented Lagrangian function
        def augmented_lagrangian(
            w: jnp.ndarray,
            theta_vals: jnp.ndarray,
            eta: jnp.ndarray,
            mu: float,
            t_array: jnp.ndarray,
            y_target: jnp.ndarray
        ):
            """
            Compute Augmented Lagrangian value.
            
            Args:
                w: State trajectory, shape (N, n_total)
                theta_vals: Optimized parameter values
                eta: Multipliers, shape (N-1, n_total) corresponding to intervals
                mu: Penalty parameter
                t_array: Time points, shape (N,)
                y_target: Target outputs, shape (N, n_outputs)
            """
            # 1. Map optimized params to full param vector
            p_all = self.p_all.at[self.optimize_indices_jax].set(theta_vals)
            
            # 2. Compute Phi(w, theta) - Trajectory Loss
            # We can reuse self.jac.trajectory_loss or reimplement for clarity/gradient flow
            # Reimplementing to ensure full traceability in one function for grad
            
            # Compute outputs
            if self.jac.h_funcs:
                # Use vmapped eval_h
                outputs = vmap(self.jac.eval_h_jax, in_axes=(0, 0, 0, None))(
                    t_array, 
                    w[:, :self.n_states], 
                    w[:, self.n_states:], 
                    p_all
                )
            else:
                outputs = w[:, :self.n_states]
                
            diff = outputs - y_target
            if self.loss_type == 'mean':
                phi = jnp.mean(diff**2)
            else:
                phi = jnp.sum(diff**2)
                
            # 3. Compute R(w, theta) - Residuals
            t_k = t_array[:-1]
            t_kp1 = t_array[1:]
            w_k = w[:-1]
            w_kp1 = w[1:]
            
            residuals = self._residual_vmap(t_k, t_kp1, w_k, w_kp1, p_all)
            # residuals shape: (N-1, n_total)
            
            # 4. Compute constraint terms
            # Term: eta^T R
            eta_dot_R = jnp.sum(eta * residuals)
            
            # Term: (mu/2) ||R||^2
            R_norm_sq = jnp.sum(residuals**2)
            penalty = (mu / 2.0) * R_norm_sq
            
            return phi + eta_dot_R + penalty
            
        self._augmented_lagrangian_jit = jit(augmented_lagrangian)
        
        # 3. Gradient of AL w.r.t. w
        # We need grad w.r.t first arg (w)
        grad_fn = grad(augmented_lagrangian, argnums=0)
        
        def grad_w_augmented_lagrangian(
            w: jnp.ndarray,
            theta_vals: jnp.ndarray,
            eta: jnp.ndarray,
            mu: float,
            t_array: jnp.ndarray,
            y_target: jnp.ndarray
        ):
            # Compute full gradient
            grad_full = grad_fn(w, theta_vals, eta, mu, t_array, y_target)
            
            # Zero out gradient for w[0] (initial condition is fixed)
            grad_w = grad_full.at[0].set(jnp.zeros_like(grad_full[0]))
            
            return grad_w
            
        self._grad_w_augmented_lagrangian_jit = jit(grad_w_augmented_lagrangian)
        
        # 4. Gradient of AL w.r.t. theta (optimized parameters)
        # We need grad w.r.t second arg (theta_vals)
        grad_theta_fn = grad(augmented_lagrangian, argnums=1)
        self._grad_theta_augmented_lagrangian_jit = jit(grad_theta_fn)

        # 5. Jacobian of Residuals w.r.t. theta
        # Shape: (N-1, n_res, n_theta)
        
        def residual_single_theta(
            t_k: float,
            t_kp1: float,
            w_k: jnp.ndarray, # Combined state
            w_kp1: jnp.ndarray,
            theta_vals: jnp.ndarray
        ):
            # Map optimized params to full param vector
            p_all = self.p_all.at[self.optimize_indices_jax].set(theta_vals)
            # Call residual_single
            return self.jac.residual_single(t_k, t_kp1, w_k, w_kp1, p_all)
            
        # Jacobian w.r.t. theta_vals (arg 4)
        jac_theta_fn = jax.jacfwd(residual_single_theta, argnums=4)
        
        # Vmap over time (args 0,1,2,3 are array, arg 4 is broadcasted)
        self._jac_theta_residual_vmap = jit(vmap(
            jac_theta_fn, 
            in_axes=(0, 0, 0, 0, None)
        ))
        
        # 6. Gradient of Phi (Objective) w.r.t theta
        # Shape: (n_theta,)
        def phi_theta(
            w: jnp.ndarray,
            theta_vals: jnp.ndarray,
            t_array: jnp.ndarray,
            y_target: jnp.ndarray
        ):
            # 1. Map optimized params
            p_all = self.p_all.at[self.optimize_indices_jax].set(theta_vals)
            
            # 2. Compute outputs
            if self.jac.h_funcs:
                outputs = vmap(self.jac.eval_h_jax, in_axes=(0, 0, 0, None))(
                    t_array, 
                    w[:, :self.n_states], 
                    w[:, self.n_states:], 
                    p_all
                )
            else:
                outputs = w[:, :self.n_states]
                
            diff = outputs - y_target
            if self.loss_type == 'mean':
                return jnp.mean(diff**2)
            else:
                return jnp.sum(diff**2)
                
        self._grad_phi_theta_jit = jit(grad(phi_theta, argnums=1))
        
    def compute_grad_phi_theta(
        self,
        t_array: np.ndarray,
        w: np.ndarray,
        theta: np.ndarray,
        y_target: np.ndarray
    ) -> np.ndarray:
        """Compute gradient of trajectory loss w.r.t optimized parameters."""
        if w.shape[0] != len(t_array) and w.shape[1] == len(t_array):
            w = w.T
        return np.array(self._grad_phi_theta_jit(
            jnp.array(w),
            jnp.array(theta),
            jnp.array(t_array),
            jnp.array(y_target)
        ))



    def compute_augmented_lagrangian(
        self,
        t_array: np.ndarray,
        w: np.ndarray,
        theta: np.ndarray,
        eta: np.ndarray,
        y_target: np.ndarray,
        mu: float
    ) -> float:
        """Compute AL value (wrapper for JIT function)."""
        return float(self._augmented_lagrangian_jit(
            jnp.array(w),
            jnp.array(theta),
            jnp.array(eta),
            float(mu),
            jnp.array(t_array),
            jnp.array(y_target)
        ))

    def compute_grad_w_augmented_lagrangian(
        self,
        t_array: np.ndarray,
        w: np.ndarray,
        theta: np.ndarray,
        eta: np.ndarray,
        y_target: np.ndarray,
        mu: float
    ) -> np.ndarray:
        """Compute gradient of AL w.r.t w (wrapper for JIT function)."""
        # Ensure correct shapes
        # w: (N, n_total)
        # eta: (N-1, n_total)
        
        # Check if w needs transpose
        if w.shape[0] != len(t_array) and w.shape[1] == len(t_array):
            w = w.T
            
        # Check if eta matches residuals (N-1)
        if eta.shape[0] == len(t_array):
             # If eta passed as full size, slice it? 
             # Or assume user passed correct size.
             pass
             
        return np.array(self._grad_w_augmented_lagrangian_jit(
            jnp.array(w),
            jnp.array(theta),
            jnp.array(eta),
            float(mu),
            jnp.array(t_array),
            jnp.array(y_target)
        ))

    def compute_grad_theta_augmented_lagrangian(
        self,
        t_array: np.ndarray,
        w: np.ndarray,
        theta: np.ndarray,
        eta: np.ndarray,
        y_target: np.ndarray,
        mu: float
    ) -> np.ndarray:
        """Compute gradient of AL w.r.t theta (wrapper for JIT function)."""
        # Ensure w shape
        if w.shape[0] != len(t_array) and w.shape[1] == len(t_array):
            w = w.T
            
        return np.array(self._grad_theta_augmented_lagrangian_jit(
            jnp.array(w),
            jnp.array(theta),
            jnp.array(eta),
            float(mu),
            jnp.array(t_array),
            jnp.array(y_target)
        ))

    


    def optimize(
        self,
        t_array: np.ndarray,
        y_target: np.ndarray,
        p_init: np.ndarray = None,
        n_iterations: int = 100,
        tol: float = 1e-4,
        verbose: bool = True,
        solver_rtol: float = 1e-6,
        solver_atol: float = 1e-6
    ) -> Dict:
        """
        Run Augmented Lagrangian optimization loop (Algorithm 3 Option C).
        
        Args:
            t_array: Time points (N,)
            y_target: Target trajectory (N, n_outputs)
            p_init: Initial guess for optimized parameters (optional)
            n_iterations: Maximum number of AL iterations
            tol: Convergence tolerance
            verbose: Print progress
            solver_rtol: Relative tolerance for internal DAE solver steps
            solver_atol: Absolute tolerance for internal DAE solver steps
            
        Returns:
            Dictionary with optimization results
        """
        # 1. Initialization
        if p_init is None:
            theta = np.array(self.p_current)
        else:
            theta = np.array(p_init)
            
        mu = getattr(self, 'penalty_mu', 1.0)
        alpha_w = getattr(self, 'alpha_w', 0.01)
        alpha_theta = getattr(self, 'alpha_theta', 0.01)
        n_primal_steps = getattr(self, 'n_primal_steps', 1)
        
        # --- Solve DAE for initial w ---
        # Update self.p for solver with initial theta
        p_all_current = np.array(self.jac.p) 
        for i, idx in enumerate(self.optimize_indices):
            p_all_current[idx] = theta[i]
            
        from src.discrete_adjoint.dae_solver import DAESolver
        dae_data_curr = self.dae_data.copy()
        
        # IMPORTANT: Verify dae_data structure. 
        # It typically has 'parameters': [{'name':..., 'value':...}, ...]
        for i, param in enumerate(dae_data_curr['parameters']):
             # Map values from p_all_current back to dae_data parameters
             # We assume p_all_current matches the order in dae_data['parameters']
             # (This relies on DAEJacobian's p construction logic)
             param['value'] = float(p_all_current[i])
             
        if verbose:
            print("Solving DAE for initial w guess...")
            
        solver = DAESolver(dae_data_curr)
        # Solve with high precision to get a good initial guess
        # Use ncp = number of desired points (approx) or set specific logic
        # Based on previous failure with ncp=N-1 giving N-1 points, we try ncp=N
        try:
            res = solver.solve(t_span=(t_array[0], t_array[-1]), ncp=len(t_array), rtol=solver_rtol, atol=solver_atol)
            x_init = res['x']
            z_init = res['z']
            
            if z_init is not None and z_init.size > 0:
                w_sol = np.vstack([x_init, z_init]).T
            else:
                w_sol = x_init.T
                
            # Verify shape
            if w_sol.shape[0] != len(t_array):
                print(f"Warning: Initial DAE solve length {w_sol.shape[0]} != t_array {len(t_array)}.")
                # Interpolate if needed or raise error. 
                # For now, raise error as grid should match if ncp is set correctly.
                # Note: DAESolver ncp sets number of intervals, so points = ncp + 1.
                # If t_array has N points, ncp should be N-1.
                raise ValueError("Grid mismatch in initialization.")
                
            w = w_sol
            if verbose: print("Initial w computed successfully.")
            
        except Exception as e:
            if verbose: print(f"Initial DAE solve failed: {e}. Fallback to zeros.")
            w = np.zeros((len(t_array), self.n_total))

        # Initialize eta = 0
        eta = np.zeros((len(t_array)-1, self.n_total))
        
        history = {'loss': [], 'mu': [], 'grad_theta_norm': [], 'residual_norm': []}
        start_time = time.time()
        
        for k in range(n_iterations):
            iter_start = time.time()
            
            # 1. Optimal Reset / Feasibility (Skipped as requested)
            
            # 2. Primal Step (w)
            for _ in range(n_primal_steps):
                grad_w = self.compute_grad_w_augmented_lagrangian(
                    t_array, w, theta, eta, y_target, mu
                )
                w = w - alpha_w * grad_w
                
            # Compute new residuals R(w^{k+1/2}, theta^k)
            # Use JAX vmap
            p_all_iter = np.array(self.jac.p)
            for i, idx in enumerate(self.optimize_indices):
                p_all_iter[idx] = theta[i]
                
            # residuals: (N-1, n_total)
            residuals = np.array(self._residual_vmap(
                jnp.array(t_array[:-1]), jnp.array(t_array[1:]), 
                jnp.array(w[:-1]), jnp.array(w[1:]), 
                jnp.array(p_all_iter)
            ))
            
            # 3. Multiplier Update (Dual Ascent)
            # eta^{k+1/2} = eta^k + mu * R
            eta_new = eta + mu * residuals
            
            # 4. Krylov Refinement (Skipped as requested)
            # eta^{k+1} = eta^{k+1/2}
            eta = eta_new
            
            # 5. Parameter Update
            # Minimize L_mu(w^{k+1}, theta, eta^{k+1})
            # Gradient Step
            grad_theta = self.compute_grad_theta_augmented_lagrangian(
                t_array, w, theta, eta, y_target, mu
            )
            
            # Update theta
            theta = theta - alpha_theta * grad_theta
            
            # Logging
            al_val = self.compute_augmented_lagrangian(t_array, w, theta, eta, y_target, mu)
            residual_norm = np.linalg.norm(residuals)
            grad_theta_norm = np.linalg.norm(grad_theta)
            
            history['loss'].append(al_val)
            history['residual_norm'].append(residual_norm)
            history['grad_theta_norm'].append(grad_theta_norm)
            history['mu'].append(mu)
            
            iter_time = time.time() - iter_start
            
            if verbose:
                print(f"Iter {k+1:3d} | AL: {al_val:.4e} | ||R||: {residual_norm:.4e} | ||g_theta||: {grad_theta_norm:.4e} | mu: {mu:.1e} | t: {iter_time:.2f}s")
                
            if residual_norm < tol and grad_theta_norm < tol:
                if verbose: print(f"Converged at iteration {k+1}")
                break
                
        total_time = time.time() - start_time
        if verbose:
            print(f"Optimization finished in {total_time:.2f}s")
            
        return {
            'theta_opt': theta,
            'w_opt': w,
            'history': history
        }
    def compute_jacobian_residual_theta(
        self,
        t_array: np.ndarray,
        w: np.ndarray,
        theta: np.ndarray
    ) -> np.ndarray:
        """
        Compute Jacobian of the residual vector w.r.t. optimized parameters.
        
        Args:
            t_array: Time points (N,)
            w: State trajectory (N, n_total)
            theta: Optimized parameters (n_theta,)
            
        Returns:
            Jacobian tensor of shape (N-1, n_total, n_theta)
        """
        # Ensure w shape
        if w.shape[0] != len(t_array) and w.shape[1] == len(t_array):
            w = w.T
            
        t_k = t_array[:-1]
        t_kp1 = t_array[1:]
        w_k = w[:-1]
        w_kp1 = w[1:]
        
        return np.array(self._jac_theta_residual_vmap(
            jnp.array(t_k),
            jnp.array(t_kp1),
            jnp.array(w_k),
            jnp.array(w_kp1),
            jnp.array(theta)
        ))
