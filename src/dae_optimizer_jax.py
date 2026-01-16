"""
JAX-based DAE Optimizer with End-to-End JIT Compilation

This module provides a fully JIT-compilable optimizer for DAE parameter identification.
Unlike the original DAEOptimizer which uses the IDA solver (not JIT-compatible),
this implementation uses a JAX-based DAE solver that enables end-to-end compilation.

Key differences from dae_jacobian.py:
1. Step 1 (DAE solving) uses a JAX-based RK4 solver with Newton iterations for algebraic constraints
2. All steps (1-7) are JIT-compiled together for maximum performance
3. Uses jax.lax.scan for memory-efficient time stepping

The adjoint-based gradient computation (Steps 2-7) follows the same algorithm as dae_jacobian.py.
"""

import jax
import jax.numpy as jnp
from jax import jit, jacfwd, jacrev
import numpy as np
import re
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from functools import partial

# Enable float64 for better precision
jax.config.update("jax_enable_x64", True)


class DAEOptimizerJax:
    """
    Fully JIT-compilable DAE optimizer using adjoint-based gradient descent.

    This optimizer solves DAEs using a JAX-based solver and computes gradients
    using the adjoint method. The entire optimization step is JIT-compiled.

    System form:
        dx/dt = f(t, x, z, p)
        0 = g(t, x, z, p)
        y = h(t, x, z, p)
    """

    def __init__(
        self,
        dae_data: Dict[str, Any],
        optimize_params: Optional[List[str]] = None,
        loss_type: str = 'sum',
        solver_method: str = 'rk4',
        newton_tol: float = 1e-10,
        newton_max_iter: int = 10
    ):
        """
        Initialize the JAX-based DAE optimizer.

        Args:
            dae_data: Dictionary containing DAE specification with keys:
                - states: List[Dict] with 'name' and 'start' keys
                - alg_vars: List[Dict] with 'name' and optional 'start' keys
                - parameters: List[Dict] with 'name' and 'value' keys
                - f: List[str] of differential equations "der(x) = expr"
                - g: List[str] of algebraic equations "0 = expr"
                - h: List[str] of output equations (or state names)
            optimize_params: List of parameter names to optimize. If None, all parameters.
            loss_type: 'sum' or 'mean' for loss computation
            solver_method: Integration method ('euler', 'rk4')
            newton_tol: Tolerance for Newton solver
            newton_max_iter: Maximum Newton iterations for algebraic constraints
        """
        self.dae_data = dae_data
        self.solver_method = solver_method
        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter

        # Validate loss type
        if loss_type not in ['sum', 'mean']:
            raise ValueError(f"loss_type must be 'sum' or 'mean', got '{loss_type}'")
        self.loss_type = loss_type

        # Extract variable names
        self.state_names = [s['name'] for s in dae_data.get('states', [])]
        self.alg_var_names = [a['name'] for a in dae_data.get('alg_vars', [])]
        self.param_names = [p['name'] for p in dae_data.get('parameters', [])]

        # Dimensions
        self.n_states = len(self.state_names)
        self.n_alg = len(self.alg_var_names)
        self.n_total = self.n_states + self.n_alg
        self.n_params_total = len(self.param_names)

        # Initial conditions
        self.x0 = jnp.array([s['start'] for s in dae_data['states']], dtype=jnp.float64)
        self.z0 = jnp.array([a.get('start', 0.0) for a in dae_data['alg_vars']], dtype=jnp.float64)

        # All parameter values
        self.p_all = jnp.array([p['value'] for p in dae_data['parameters']], dtype=jnp.float64)

        # Determine which parameters to optimize
        if optimize_params is None:
            self.optimize_params = self.param_names.copy()
            self.optimize_indices = list(range(self.n_params_total))
        else:
            self.optimize_params = optimize_params
            self.optimize_indices = []
            for param_name in optimize_params:
                if param_name in self.param_names:
                    idx = self.param_names.index(param_name)
                    self.optimize_indices.append(idx)
                else:
                    print(f"Warning: Parameter '{param_name}' not found in DAE specification")

        self.n_params_opt = len(self.optimize_indices)
        self.optimize_indices_jax = jnp.array(self.optimize_indices, dtype=jnp.int32)

        # Parse equations
        self.f_eqs = self._parse_f_equations()
        self.g_eqs = self._parse_g_equations()
        self.h_eqs = self._parse_h_equations()

        # Determine output dimension
        if self.h_eqs:
            self.n_outputs = len(self.h_eqs)
        else:
            self.n_outputs = self.n_states

        # Build JIT-compiled functions
        self._build_jit_functions()

        # Optimization history
        self.history = {
            'loss': [],
            'gradient_norm': [],
            'params': [],
            'params_all': [],
            'step_size': []
        }

        # Print info
        print(f"\nDAEOptimizerJax initialized:")
        print(f"  Total parameters: {self.n_params_total}")
        print(f"  Parameters to optimize: {self.n_params_opt}")
        print(f"  Optimized parameter names: {self.optimize_params}")
        print(f"  Differential states: {self.n_states}")
        print(f"  Algebraic variables: {self.n_alg}")
        print(f"  Outputs: {self.n_outputs}")
        print(f"  Solver method: {self.solver_method}")
        print(f"  Loss type: {self.loss_type}")

    def _parse_f_equations(self) -> List[Tuple[str, str]]:
        """Parse f equations into (state_name, expression) pairs"""
        equations = []
        if 'f' not in self.dae_data or self.dae_data['f'] is None:
            return equations

        for eq_str in self.dae_data.get('f', []):
            match = re.match(r'der\((\w+)\)\s*=\s*(.+)', eq_str.strip())
            if match:
                state_name = match.group(1)
                expr = match.group(2).strip()
                equations.append((state_name, expr))
            else:
                raise ValueError(f"Cannot parse f equation: {eq_str}")
        return equations

    def _parse_g_equations(self) -> List[str]:
        """Parse g equations into residual expressions"""
        equations = []
        if 'g' not in self.dae_data or self.dae_data['g'] is None:
            return equations

        for eq_str in self.dae_data.get('g', []):
            match = re.match(r'0(?:\.0*)?\s*=\s*(.+)', eq_str.strip())
            if match:
                equations.append(match.group(1).strip())
            else:
                raise ValueError(f"Cannot parse g equation: {eq_str}")
        return equations

    def _parse_h_equations(self) -> List[Tuple[str, str]]:
        """Parse h equations into expressions"""
        equations = []
        h_spec = self.dae_data.get('h', [])
        if not h_spec:
            return equations

        for eq_str in h_spec:
            # Check if it's just a variable name (e.g., state output)
            if eq_str.strip() in self.state_names + self.alg_var_names:
                equations.append(('output', eq_str.strip()))
            else:
                # Try to parse as "name = expr"
                match = re.match(r'(\w+)\s*=\s*(.+)', eq_str.strip())
                if match:
                    equations.append((match.group(1), match.group(2).strip()))
                else:
                    # Just use the expression directly
                    equations.append(('output', eq_str.strip()))
        return equations

    def _build_namespace(self, t: jnp.ndarray, x: jnp.ndarray, z: jnp.ndarray,
                         p: jnp.ndarray) -> Dict[str, Any]:
        """Build namespace for expression evaluation"""
        namespace = {'t': t}

        # Add states
        for i, name in enumerate(self.state_names):
            namespace[name] = x[i]

        # Add algebraic variables
        for i, name in enumerate(self.alg_var_names):
            namespace[name] = z[i]

        # Add parameters
        for i, name in enumerate(self.param_names):
            namespace[name] = p[i]

        return namespace

    def _eval_expr(self, expr: str, namespace: Dict[str, Any]) -> jnp.ndarray:
        """Safely evaluate expression with given namespace"""
        expr = expr.replace('^', '**')
        expr = re.sub(r'\btime\b', 't', expr)

        math_functions = {
            'exp': jnp.exp, 'log': jnp.log, 'log10': jnp.log10,
            'sqrt': jnp.sqrt, 'abs': jnp.abs,
            'sin': jnp.sin, 'cos': jnp.cos, 'tan': jnp.tan,
            'asin': jnp.arcsin, 'acos': jnp.arccos, 'atan': jnp.arctan,
            'sinh': jnp.sinh, 'cosh': jnp.cosh, 'tanh': jnp.tanh,
            'sign': jnp.sign, 'floor': jnp.floor, 'ceil': jnp.ceil,
            'min': jnp.minimum, 'max': jnp.maximum,
            'sigmoid': jax.nn.sigmoid,
        }

        eval_namespace = {**namespace, 'jnp': jnp, **math_functions}
        return eval(expr, {'__builtins__': {}}, eval_namespace)

    def _build_jit_functions(self):
        """Build JIT-compiled core functions"""

        # Vectorized f evaluation
        def eval_f(t, x, z, p):
            """Evaluate f(t, x, z, p) -> dx/dt"""
            namespace = self._build_namespace(t, x, z, p)
            dxdt = jnp.zeros(self.n_states, dtype=jnp.float64)
            for state_name, expr in self.f_eqs:
                idx = self.state_names.index(state_name)
                val = self._eval_expr(expr, namespace)
                dxdt = dxdt.at[idx].set(val)
            return dxdt

        self._eval_f = eval_f

        # Vectorized g evaluation
        def eval_g(t, x, z, p):
            """Evaluate g(t, x, z, p) -> residual"""
            if not self.g_eqs:
                return jnp.zeros(0, dtype=jnp.float64)
            namespace = self._build_namespace(t, x, z, p)
            residuals = []
            for expr in self.g_eqs:
                residuals.append(self._eval_expr(expr, namespace))
            return jnp.array(residuals)

        self._eval_g = eval_g

        # Vectorized h evaluation
        def eval_h(t, x, z, p):
            """Evaluate h(t, x, z, p) -> outputs"""
            if not self.h_eqs:
                return x  # Default: outputs are states
            namespace = self._build_namespace(t, x, z, p)
            outputs = []
            for _, expr in self.h_eqs:
                outputs.append(self._eval_expr(expr, namespace))
            return jnp.array(outputs)

        self._eval_h = eval_h

        # Newton solver for algebraic constraints
        def solve_algebraic_newton(t, x, p, z_init):
            """Solve g(t, x, z, p) = 0 for z using Newton's method"""
            if self.n_alg == 0:
                return jnp.zeros(0, dtype=jnp.float64)

            z = z_init

            def newton_step(carry, _):
                z_curr = carry
                g_val = eval_g(t, x, z_curr, p)
                jac_g_z = jacfwd(lambda zz: eval_g(t, x, zz, p))(z_curr)
                delta_z = jnp.linalg.solve(jac_g_z, -g_val)
                z_new = z_curr + delta_z
                return z_new, jnp.linalg.norm(g_val)

            z_final, _ = jax.lax.scan(newton_step, z, None, length=self.newton_max_iter)
            return z_final

        self._solve_algebraic = solve_algebraic_newton

        # Forward dynamics: given x, solve for z, then compute dx/dt
        def forward_dynamics(t, x, p, z_init):
            """Compute dx/dt by solving algebraic constraints first"""
            z = solve_algebraic_newton(t, x, p, z_init)
            dxdt = eval_f(t, x, z, p)
            return dxdt, z

        self._forward_dynamics = forward_dynamics

        # Single RK4 step
        def rk4_step(t, x, p, z_prev, dt):
            """Single RK4 integration step"""
            k1, z1 = forward_dynamics(t, x, p, z_prev)
            k2, z2 = forward_dynamics(t + 0.5*dt, x + 0.5*dt*k1, p, z1)
            k3, z3 = forward_dynamics(t + 0.5*dt, x + 0.5*dt*k2, p, z2)
            k4, z4 = forward_dynamics(t + dt, x + dt*k3, p, z3)
            x_next = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            return x_next, z4

        self._rk4_step = rk4_step

        # Euler step
        def euler_step(t, x, p, z_prev, dt):
            """Single Euler integration step"""
            dxdt, z = forward_dynamics(t, x, p, z_prev)
            x_next = x + dt * dxdt
            return x_next, z

        self._euler_step = euler_step

        # Full simulation using scan
        def simulate_scan(x0, z0, t_array, p):
            """Simulate DAE over time array using scan"""
            n_time = t_array.shape[0]
            dt = t_array[1] - t_array[0]  # Assume uniform spacing

            def scan_step(carry, t):
                x, z = carry
                if self.solver_method == 'rk4':
                    x_next, z_next = rk4_step(t, x, p, z, dt)
                else:
                    x_next, z_next = euler_step(t, x, p, z, dt)
                y = eval_h(t, x, z, p)
                return (x_next, z_next), (x, z, y)

            init_state = (x0, z0)
            _, (x_traj, z_traj, y_traj) = jax.lax.scan(scan_step, init_state, t_array)

            return x_traj, z_traj, y_traj

        self._simulate_scan = simulate_scan

        # JIT-compile the simulation
        self._simulate_jit = jit(simulate_scan)

        # Build the combined optimization step (Steps 2-7 from original)
        self._build_gradient_computation()

    def _build_gradient_computation(self):
        """Build JIT-compiled gradient computation (adjoint method)"""

        def compute_residual_jacobians(t_k, t_kp1, y_k, y_kp1, p):
            """
            Compute Jacobians of the implicit residual R(y_k, y_{k+1}, p) = 0

            For the trapezoidal rule or RK-like discretization:
            R = y_{k+1} - y_k - dt * F(t, y_k, y_{k+1}, p)

            We need: dR/dy_k, dR/dy_{k+1}, dR/dp
            """
            dt = t_kp1 - t_k
            n_total = self.n_total

            # Define residual as function of y_k, y_{k+1}, p
            def residual_fn(y_k_arg, y_kp1_arg, p_arg):
                # Split into x, z components
                x_k = y_k_arg[:self.n_states]
                z_k = y_k_arg[self.n_states:]
                x_kp1 = y_kp1_arg[:self.n_states]
                z_kp1 = y_kp1_arg[self.n_states:]

                # Evaluate f at both endpoints (trapezoidal-like)
                f_k = self._eval_f(t_k, x_k, z_k, p_arg)
                f_kp1 = self._eval_f(t_kp1, x_kp1, z_kp1, p_arg)

                # Differential equations residual (trapezoidal rule)
                R_x = x_kp1 - x_k - 0.5 * dt * (f_k + f_kp1)

                # Algebraic equations residual at t_{k+1}
                R_z = self._eval_g(t_kp1, x_kp1, z_kp1, p_arg)

                return jnp.concatenate([R_x, R_z])

            # Compute Jacobians
            J_y_k = jacfwd(lambda yk: residual_fn(yk, y_kp1, p))(y_k)
            J_y_kp1 = jacfwd(lambda ykp1: residual_fn(y_k, ykp1, p))(y_kp1)

            # Only compute parameter Jacobian for optimized parameters
            def residual_fn_p_opt(p_opt):
                # Reconstruct full parameter vector
                p_full = self.p_all.at[self.optimize_indices_jax].set(p_opt)
                return residual_fn(y_k, y_kp1, p_full)

            p_opt = p[self.optimize_indices_jax]
            J_p = jacfwd(residual_fn_p_opt)(p_opt)

            return J_y_k, J_y_kp1, J_p

        # Vectorize over time steps
        self._compute_jacobians_vmapped = jax.vmap(
            compute_residual_jacobians,
            in_axes=(0, 0, 0, 0, None)
        )

        def compute_loss_gradient(t_array, y_array, y_target, p):
            """
            Compute dL/dy for the loss function L = sum ||h(y) - y_target||^2

            dL/dy_k = 2 * (dh/dy)^T @ (h(y_k) - y_target_k)
            """
            def single_point_grad(t, y, y_tgt):
                x = y[:self.n_states]
                z = y[self.n_states:]

                # Compute h and its Jacobian
                h_val = self._eval_h(t, x, z, p)

                def h_of_y(yy):
                    return self._eval_h(t, yy[:self.n_states], yy[self.n_states:], p)

                dh_dy = jacfwd(h_of_y)(y)  # (n_outputs, n_total)

                # Error
                error = h_val - y_tgt  # (n_outputs,)

                # Gradient: 2 * dh/dy^T @ error
                grad = 2.0 * dh_dy.T @ error  # (n_total,)

                return grad

            grads = jax.vmap(single_point_grad)(t_array, y_array, y_target)
            return grads

        self._compute_loss_gradient = compute_loss_gradient

        def solve_adjoint_backward(J_curr, J_prev, rhs):
            """
            Solve the adjoint system using backward substitution.

            System: J_curr[k]^T @ lambda[k] + J_prev[k+1]^T @ lambda[k+1] = rhs[k]
            """
            N = J_curr.shape[0]

            # Terminal solve
            lam_N = jnp.linalg.solve(J_curr[-1].T, rhs[-1])

            def backward_step(lam_next, inputs):
                J_curr_k, J_prev_kp1, rhs_k = inputs
                rhs_mod = rhs_k - J_prev_kp1.T @ lam_next
                lam_k = jnp.linalg.solve(J_curr_k.T, rhs_mod)
                return lam_k, lam_k

            # Reverse order inputs
            inputs_rev = (
                J_curr[:-1][::-1],
                J_prev[1:][::-1],
                rhs[:-1][::-1]
            )

            _, lam_rev = jax.lax.scan(backward_step, lam_N, inputs_rev)

            # Concatenate in correct order
            lambda_adj = jnp.concatenate([lam_rev[::-1], lam_N[None, :]], axis=0)

            return lambda_adj

        self._solve_adjoint = solve_adjoint_backward

        def combined_gradient_step(x0, z0, t_array, y_target, p_opt, step_size):
            """
            Combined optimization step: simulate + compute gradient + update.

            This is the fully JIT-compiled optimization step.
            """
            # Reconstruct full parameter vector
            p_full = self.p_all.at[self.optimize_indices_jax].set(p_opt)

            # Step 1: Simulate DAE
            x_traj, z_traj, y_traj = self._simulate_scan(x0, z0, t_array, p_full)

            # Combine x and z into y
            y_array = jnp.concatenate([x_traj, z_traj], axis=1)  # (n_time, n_total)

            # Compute loss
            h_pred = y_traj  # Already computed during simulation
            error = h_pred - y_target
            if self.loss_type == 'mean':
                loss = jnp.mean(error**2)
            else:
                loss = jnp.sum(error**2)

            # Step 2: Compute loss gradient dL/dy
            dL_dy = self._compute_loss_gradient(t_array, y_array, y_target, p_full)

            # Scale by 1/N if using mean loss
            if self.loss_type == 'mean':
                N_total = y_target.shape[0] * y_target.shape[1]
                dL_dy = dL_dy / N_total

            # Exclude initial condition (fixed)
            dL_dy_adj = dL_dy[1:]  # (N, n_total)

            # Step 3: Compute Jacobian blocks
            N = t_array.shape[0] - 1
            t_k = t_array[:-1]
            t_kp1 = t_array[1:]
            y_k = y_array[:-1]
            y_kp1 = y_array[1:]

            J_prev, J_curr, J_param = self._compute_jacobians_vmapped(
                t_k, t_kp1, y_k, y_kp1, p_full
            )

            # Step 4: Solve adjoint system
            lambda_adj = self._solve_adjoint(J_curr, J_prev, dL_dy_adj)

            # Step 5: Compute parameter gradient
            # dR/dp has shape (N, n_total, n_params_opt)
            # Reshape to (N*n_total, n_params_opt)
            dR_dp = J_param.reshape(N * self.n_total, -1)

            # Step 6: dL/dp = -(dR/dp)^T @ lambda
            lambda_flat = lambda_adj.flatten()
            grad_p_opt = -dR_dp.T @ lambda_flat

            # Step 7: Gradient descent update
            p_opt_new = p_opt - step_size * grad_p_opt

            return p_opt_new, loss, grad_p_opt, y_traj

        self._combined_gradient_step = combined_gradient_step
        self._combined_gradient_step_jit = jit(combined_gradient_step)

    def optimization_step(
        self,
        t_array: np.ndarray,
        y_target: np.ndarray,
        p_opt: np.ndarray,
        step_size: float = 0.01
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Perform one optimization step using JIT-compiled gradient computation.

        Args:
            t_array: Time points
            y_target: Target outputs, shape (n_time, n_outputs)
            p_opt: Current optimized parameter values
            step_size: Gradient descent step size

        Returns:
            p_opt_new: Updated optimized parameters
            loss: Loss value
            grad_p_opt: Gradient w.r.t. optimized parameters
        """
        # Convert to JAX arrays
        t_jax = jnp.array(t_array, dtype=jnp.float64)
        y_target_jax = jnp.array(y_target, dtype=jnp.float64)
        p_opt_jax = jnp.array(p_opt, dtype=jnp.float64)

        # Ensure y_target has shape (n_time, n_outputs)
        if y_target_jax.shape[0] != t_jax.shape[0]:
            y_target_jax = y_target_jax.T

        # Run JIT-compiled step
        p_opt_new, loss, grad_p_opt, _ = self._combined_gradient_step_jit(
            self.x0, self.z0, t_jax, y_target_jax, p_opt_jax, step_size
        )

        return np.array(p_opt_new), float(loss), np.array(grad_p_opt)

    def simulate(
        self,
        t_array: np.ndarray,
        p: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate DAE with given parameters.

        Args:
            t_array: Time points
            p: Full parameter vector (if None, uses stored values)

        Returns:
            Dictionary with 't', 'x', 'z', 'y' trajectories
        """
        t_jax = jnp.array(t_array, dtype=jnp.float64)

        if p is None:
            p_jax = self.p_all
        else:
            p_jax = jnp.array(p, dtype=jnp.float64)

        x_traj, z_traj, y_traj = self._simulate_jit(self.x0, self.z0, t_jax, p_jax)

        return {
            't': np.array(t_jax),
            'x': np.array(x_traj).T,  # (n_states, n_time)
            'z': np.array(z_traj).T,  # (n_alg, n_time)
            'y': np.array(y_traj).T   # (n_outputs, n_time)
        }

    def compute_loss(self, y_pred: jnp.ndarray, y_target: jnp.ndarray) -> float:
        """Compute loss between predicted and target outputs."""
        error = y_pred - y_target
        if self.loss_type == 'mean':
            return float(jnp.mean(error**2))
        else:
            return float(jnp.sum(error**2))

    def optimize(
        self,
        t_array: np.ndarray,
        y_target: np.ndarray,
        p_init: Optional[np.ndarray] = None,
        n_iterations: int = 100,
        step_size: float = 0.01,
        tol: float = 1e-6,
        verbose: bool = True
    ) -> Dict:
        """
        Optimize DAE parameters to minimize loss.

        Args:
            t_array: Time points for trajectory
            y_target: Target output trajectory, shape (n_time, n_outputs)
            p_init: Initial values for optimized parameters
            n_iterations: Maximum number of iterations
            step_size: Gradient descent step size
            tol: Convergence tolerance on gradient norm
            verbose: Print progress

        Returns:
            Dictionary with optimization results
        """
        if verbose:
            print("\n" + "=" * 80)
            print("Starting JAX-based DAE Parameter Optimization")
            print("=" * 80)
            print(f"  Iterations: {n_iterations}")
            print(f"  Step size: {step_size}")
            print(f"  Tolerance: {tol}")
            print(f"  Target trajectory points: {len(t_array)}")

        # Initialize parameters
        if p_init is not None:
            p = jnp.array(p_init, dtype=jnp.float64)
        else:
            p = jnp.array([self.p_all[i] for i in self.optimize_indices], dtype=jnp.float64)

        # Reset history
        self.history = {
            'loss': [],
            'gradient_norm': [],
            'params': [],
            'params_all': [],
            'step_size': [],
            'time_per_iter': []
        }

        converged = False

        # Ensure y_target shape
        y_target_use = np.array(y_target)
        if y_target_use.shape[0] != len(t_array):
            y_target_use = y_target_use.T

        # Optimization loop
        for iteration in range(n_iterations):
            t_start = time.time()

            # Perform optimization step
            p_new, loss, grad_p = self.optimization_step(
                t_array, y_target_use, np.array(p), step_size
            )

            t_end = time.time()

            # Compute gradient norm
            grad_norm = float(np.linalg.norm(grad_p))

            # Construct full parameter vector
            p_all_current = np.array(self.p_all)
            for i, opt_idx in enumerate(self.optimize_indices):
                p_all_current[opt_idx] = float(p[i])

            # Store history
            iter_time = t_end - t_start
            self.history['loss'].append(loss)
            self.history['gradient_norm'].append(grad_norm)
            self.history['params'].append(np.array(p))
            self.history['params_all'].append(p_all_current)
            self.history['step_size'].append(step_size)
            self.history['time_per_iter'].append(iter_time)

            # Print progress
            if verbose and (iteration % 10 == 0 or iteration == n_iterations - 1):
                print(f"\nIteration {iteration:4d} ({t_end - t_start:.3f}s):")
                print(f"  Loss:          {loss:.6e}")
                print(f"  Gradient norm: {grad_norm:.6e}")

            # Check convergence
            if grad_norm < tol:
                converged = True
                if verbose:
                    print(f"\n Converged at iteration {iteration}")
                break

            # Update parameters
            p = jnp.array(p_new)

        # Final parameter vector
        p_all_final = np.array(self.p_all)
        for i, opt_idx in enumerate(self.optimize_indices):
            p_all_final[opt_idx] = float(p[i])

        if verbose:
            print("\n" + "=" * 80)
            print("Optimization Complete")
            print("=" * 80)
            print(f"  Converged: {converged}")
            print(f"  Final loss: {self.history['loss'][-1]:.6e}")
            print(f"  Final gradient norm: {self.history['gradient_norm'][-1]:.6e}")

            # Timing statistics
            times = self.history['time_per_iter']
            total_time = sum(times)
            if len(times) > 1:
                # Exclude first iteration (includes JIT compilation)
                times_no_jit = times[1:]
                avg_time = np.mean(times_no_jit)
                std_time = np.std(times_no_jit)
                print(f"\n  Timing statistics:")
                print(f"    Total time:              {total_time:.3f}s")
                print(f"    First iteration (JIT):   {times[0]:.3f}s")
                print(f"    Avg time/iter (no JIT):  {avg_time*1000:.3f}ms")
                print(f"    Std time/iter (no JIT):  {std_time*1000:.3f}ms")
                print(f"    Min time/iter:           {min(times_no_jit)*1000:.3f}ms")
                print(f"    Max time/iter:           {max(times_no_jit)*1000:.3f}ms")
            else:
                print(f"\n  Total time: {total_time:.3f}s")

            print(f"\n  Optimized parameters:")
            for i, (name, val) in enumerate(zip(self.optimize_params, p)):
                print(f"    {name:20s} = {float(val):.6f}")

        return {
            'p_opt': np.array(p),
            'p_all': p_all_final,
            'loss_final': self.history['loss'][-1],
            'history': self.history,
            'converged': converged,
            'n_iterations': len(self.history['loss'])
        }

    def plot_optimization_history(self):
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        ax = axes[0, 0]
        ax.semilogy(self.history['loss'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Function')
        ax.grid(True, alpha=0.3)

        # Gradient norm
        ax = axes[0, 1]
        ax.semilogy(self.history['gradient_norm'], 'r-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm')
        ax.grid(True, alpha=0.3)

        # Parameters
        ax = axes[1, 0]
        params_array = np.array(self.history['params'])
        for i in range(params_array.shape[1]):
            ax.plot(params_array[:, i], label=f'{self.optimize_params[i]}', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Value')
        ax.set_title('Parameter Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Loss vs gradient norm
        ax = axes[1, 1]
        ax.loglog(self.history['gradient_norm'], self.history['loss'], 'go-', linewidth=2, markersize=4)
        ax.set_xlabel('Gradient Norm')
        ax.set_ylabel('Loss')
        ax.set_title('Loss vs Gradient Norm')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    print("DAEOptimizerJax - End-to-End JIT-Compiled Optimizer")
    print("=" * 60)

    # Simple test DAE
    dae_data = {
        'states': [{'name': 'x', 'start': 1.0}],
        'alg_vars': [{'name': 'z', 'start': 1.0}],
        'parameters': [{'name': 'p', 'value': 0.5}],
        'f': ['der(x) = -p * x'],
        'g': ['0 = z - x * x'],
        'h': ['x']
    }

    # Create optimizer
    optimizer = DAEOptimizerJax(dae_data)

    # Test simulation
    t_test = np.linspace(0, 2, 21)
    result = optimizer.simulate(t_test)
    print(f"\nSimulation result shapes:")
    print(f"  x: {result['x'].shape}")
    print(f"  z: {result['z'].shape}")
    print(f"  y: {result['y'].shape}")

    print("\nTest complete!")
