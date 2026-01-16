"""
JAX-based DAE Optimizer using Automatic Differentiation

This module provides a fully JIT-compilable optimizer for DAE parameter identification
using JAX's automatic differentiation (AD) to compute gradients directly through
the simulation, without using the adjoint method.

Key features:
1. Uses jax.value_and_grad to compute loss and gradients in one pass
2. Fully JIT-compiled end-to-end
3. Simpler implementation than adjoint method
4. Custom VJP for algebraic constraint solver (implicit function theorem)

Trade-offs vs adjoint method:
- Simpler code, easier to understand and maintain
- May use more memory for very long trajectories (stores intermediate states)
- Gradient computation is automatic, no need to derive adjoint equations
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev, value_and_grad, custom_vjp
import numpy as np
import re
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from functools import partial

# Enable float64 for better precision
jax.config.update("jax_enable_x64", True)


class DAEOptimizerJaxAD:
    """
    DAE optimizer using automatic differentiation for gradient computation.

    This optimizer solves DAEs using a JAX-based solver and computes gradients
    using JAX's automatic differentiation through the simulation.

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
        Initialize the JAX-based DAE optimizer with AD.

        Args:
            dae_data: Dictionary containing DAE specification
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
            'step_size': [],
            'time_per_iter': []
        }

        print(f"\nDAEOptimizerJaxAD initialized:")
        print(f"  Total parameters: {self.n_params_total}")
        print(f"  Parameters to optimize: {self.n_params_opt}")
        print(f"  Optimized parameter names: {self.optimize_params}")
        print(f"  Differential states: {self.n_states}")
        print(f"  Algebraic variables: {self.n_alg}")
        print(f"  Outputs: {self.n_outputs}")
        print(f"  Solver method: {self.solver_method}")
        print(f"  Loss type: {self.loss_type}")
        print(f"  Gradient method: Automatic Differentiation")

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
            if eq_str.strip() in self.state_names + self.alg_var_names:
                equations.append(('output', eq_str.strip()))
            else:
                match = re.match(r'(\w+)\s*=\s*(.+)', eq_str.strip())
                if match:
                    equations.append((match.group(1), match.group(2).strip()))
                else:
                    equations.append(('output', eq_str.strip()))
        return equations

    def _build_namespace(self, t: jnp.ndarray, x: jnp.ndarray, z: jnp.ndarray,
                         p: jnp.ndarray) -> Dict[str, Any]:
        """Build namespace for expression evaluation"""
        namespace = {'t': t}

        for i, name in enumerate(self.state_names):
            namespace[name] = x[i]

        for i, name in enumerate(self.alg_var_names):
            namespace[name] = z[i]

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
        """Build JIT-compiled core functions with custom VJP for algebraic solver"""

        # Store references for use in nested functions
        state_names = self.state_names
        alg_var_names = self.alg_var_names
        param_names = self.param_names
        f_eqs = self.f_eqs
        g_eqs = self.g_eqs
        h_eqs = self.h_eqs
        n_states = self.n_states
        n_alg = self.n_alg
        newton_max_iter = self.newton_max_iter
        newton_tol = self.newton_tol

        def build_namespace(t, x, z, p):
            namespace = {'t': t}
            for i, name in enumerate(state_names):
                namespace[name] = x[i]
            for i, name in enumerate(alg_var_names):
                namespace[name] = z[i]
            for i, name in enumerate(param_names):
                namespace[name] = p[i]
            return namespace

        def eval_expr(expr, namespace):
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

        def eval_f(t, x, z, p):
            """Evaluate f(t, x, z, p) -> dx/dt"""
            namespace = build_namespace(t, x, z, p)
            dxdt = jnp.zeros(n_states, dtype=jnp.float64)
            for state_name, expr in f_eqs:
                idx = state_names.index(state_name)
                val = eval_expr(expr, namespace)
                dxdt = dxdt.at[idx].set(val)
            return dxdt

        def eval_g(t, x, z, p):
            """Evaluate g(t, x, z, p) -> residual"""
            if not g_eqs:
                return jnp.zeros(0, dtype=jnp.float64)
            namespace = build_namespace(t, x, z, p)
            residuals = []
            for expr in g_eqs:
                residuals.append(eval_expr(expr, namespace))
            return jnp.array(residuals)

        def eval_h(t, x, z, p):
            """Evaluate h(t, x, z, p) -> outputs"""
            if not h_eqs:
                return x
            namespace = build_namespace(t, x, z, p)
            outputs = []
            for _, expr in h_eqs:
                outputs.append(eval_expr(expr, namespace))
            return jnp.array(outputs)

        self._eval_f = eval_f
        self._eval_g = eval_g
        self._eval_h = eval_h

        # Newton solver for algebraic constraints (forward pass only)
        def solve_algebraic_newton_fwd(t, x, p, z_init):
            """Solve g(t, x, z, p) = 0 for z using Newton's method"""
            if n_alg == 0:
                return jnp.zeros(0, dtype=jnp.float64)

            z = z_init

            def newton_step(z_curr, _):
                g_val = eval_g(t, x, z_curr, p)
                jac_g_z = jacfwd(lambda zz: eval_g(t, x, zz, p))(z_curr)
                # Add small regularization for numerical stability
                jac_g_z = jac_g_z + 1e-12 * jnp.eye(n_alg)
                delta_z = jnp.linalg.solve(jac_g_z, -g_val)
                z_new = z_curr + delta_z
                return z_new, jnp.linalg.norm(g_val)

            z_final, _ = jax.lax.scan(newton_step, z, None, length=newton_max_iter)
            return z_final

        # Custom VJP for algebraic solver using implicit function theorem
        @custom_vjp
        def solve_algebraic(t, x, p, z_init):
            """Solve algebraic constraints with custom gradient"""
            return solve_algebraic_newton_fwd(t, x, p, z_init)

        def solve_algebraic_fwd(t, x, p, z_init):
            z = solve_algebraic_newton_fwd(t, x, p, z_init)
            return z, (t, x, z, p)

        def solve_algebraic_bwd(res, g_z):
            """
            Backward pass using implicit function theorem.

            At solution: g(t, x, z, p) = 0
            Differentiating: dg/dx dx + dg/dz dz + dg/dp dp = 0
            Therefore: dz/dx = -(dg/dz)^{-1} @ dg/dx
                       dz/dp = -(dg/dz)^{-1} @ dg/dp

            For VJP: grad_x = g_z @ dz/dx = -g_z @ (dg/dz)^{-1} @ dg/dx
                     grad_p = g_z @ dz/dp = -g_z @ (dg/dz)^{-1} @ dg/dp
            """
            t, x, z, p = res

            if n_alg == 0:
                return (None, jnp.zeros_like(x), jnp.zeros_like(p), None)

            # Compute dg/dz
            dg_dz = jacfwd(lambda zz: eval_g(t, x, zz, p))(z)

            # Solve: dg/dz^T @ lambda = g_z for lambda
            # (equivalent to lambda = (dg/dz)^{-T} @ g_z)
            dg_dz_reg = dg_dz + 1e-12 * jnp.eye(n_alg)
            lambda_val = jnp.linalg.solve(dg_dz_reg.T, g_z)

            # grad_x = -dg/dx^T @ lambda
            dg_dx = jacfwd(lambda xx: eval_g(t, xx, z, p))(x)
            grad_x = -dg_dx.T @ lambda_val

            # grad_p = -dg/dp^T @ lambda
            dg_dp = jacfwd(lambda pp: eval_g(t, x, z, pp))(p)
            grad_p = -dg_dp.T @ lambda_val

            return (None, grad_x, grad_p, None)

        solve_algebraic.defvjp(solve_algebraic_fwd, solve_algebraic_bwd)

        self._solve_algebraic = solve_algebraic

        # Forward dynamics with differentiable algebraic solve
        def forward_dynamics(t, x, p, z_init):
            """Compute dx/dt by solving algebraic constraints first"""
            z = solve_algebraic(t, x, p, z_init)
            dxdt = eval_f(t, x, z, p)
            return dxdt, z

        self._forward_dynamics = forward_dynamics

        # Integration steps
        def rk4_step(t, x, p, z_prev, dt):
            """Single RK4 integration step"""
            k1, z1 = forward_dynamics(t, x, p, z_prev)
            k2, z2 = forward_dynamics(t + 0.5*dt, x + 0.5*dt*k1, p, z1)
            k3, z3 = forward_dynamics(t + 0.5*dt, x + 0.5*dt*k2, p, z2)
            k4, z4 = forward_dynamics(t + dt, x + dt*k3, p, z3)
            x_next = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            return x_next, z4

        def euler_step(t, x, p, z_prev, dt):
            """Single Euler integration step"""
            dxdt, z = forward_dynamics(t, x, p, z_prev)
            x_next = x + dt * dxdt
            return x_next, z

        self._rk4_step = rk4_step
        self._euler_step = euler_step

        # Simulation function (differentiable w.r.t. parameters)
        def simulate(x0, z0, t_array, p):
            """Simulate DAE - differentiable w.r.t. p"""
            dt = t_array[1] - t_array[0]

            def scan_step(carry, t):
                x, z = carry
                if self.solver_method == 'rk4':
                    x_next, z_next = rk4_step(t, x, p, z, dt)
                else:
                    x_next, z_next = euler_step(t, x, p, z, dt)
                y = eval_h(t, x, z, p)
                return (x_next, z_next), (x, z, y)

            _, (x_traj, z_traj, y_traj) = jax.lax.scan(scan_step, (x0, z0), t_array)
            return x_traj, z_traj, y_traj

        self._simulate = simulate
        self._simulate_jit = jit(simulate)

        # Loss function (differentiable)
        def compute_loss(p_opt, x0, z0, t_array, y_target):
            """Compute loss - differentiable w.r.t. p_opt"""
            # Reconstruct full parameter vector
            p_full = self.p_all.at[self.optimize_indices_jax].set(p_opt)

            # Simulate
            _, _, y_traj = simulate(x0, z0, t_array, p_full)

            # Compute loss
            error = y_traj - y_target
            if self.loss_type == 'mean':
                return jnp.mean(error**2)
            else:
                return jnp.sum(error**2)

        self._compute_loss = compute_loss

        # JIT-compiled loss and gradient function
        self._loss_and_grad = jit(value_and_grad(compute_loss))

    def optimization_step(
        self,
        t_array: np.ndarray,
        y_target: np.ndarray,
        p_opt: np.ndarray,
        step_size: float = 0.01
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Perform one optimization step using AD for gradient computation.

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
        t_jax = jnp.array(t_array, dtype=jnp.float64)
        y_target_jax = jnp.array(y_target, dtype=jnp.float64)
        p_opt_jax = jnp.array(p_opt, dtype=jnp.float64)

        # Ensure y_target has shape (n_time, n_outputs)
        if y_target_jax.shape[0] != t_jax.shape[0]:
            y_target_jax = y_target_jax.T

        # Compute loss and gradient using AD
        loss, grad_p_opt = self._loss_and_grad(
            p_opt_jax, self.x0, self.z0, t_jax, y_target_jax
        )

        # Gradient descent update
        p_opt_new = p_opt_jax - step_size * grad_p_opt

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
            'x': np.array(x_traj).T,
            'z': np.array(z_traj).T,
            'y': np.array(y_traj).T
        }

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
        Optimize DAE parameters to minimize loss using AD gradients.

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
            print("Starting JAX-based DAE Parameter Optimization (AD)")
            print("=" * 80)
            print(f"  Iterations: {n_iterations}")
            print(f"  Step size: {step_size}")
            print(f"  Tolerance: {tol}")
            print(f"  Target trajectory points: {len(t_array)}")
            print(f"  Gradient method: Automatic Differentiation")

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
                print(f"\nIteration {iteration:4d} ({iter_time:.3f}s):")
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

        ax = axes[0, 0]
        ax.semilogy(self.history['loss'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Function')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.semilogy(self.history['gradient_norm'], 'r-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        params_array = np.array(self.history['params'])
        for i in range(params_array.shape[1]):
            ax.plot(params_array[:, i], label=f'{self.optimize_params[i]}', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Value')
        ax.set_title('Parameter Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

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
    print("DAEOptimizerJaxAD - Automatic Differentiation Optimizer")
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
    optimizer = DAEOptimizerJaxAD(dae_data)

    # Test simulation
    t_test = np.linspace(0, 2, 21)
    result = optimizer.simulate(t_test)
    print(f"\nSimulation result shapes:")
    print(f"  x: {result['x'].shape}")
    print(f"  z: {result['z'].shape}")
    print(f"  y: {result['y'].shape}")

    print("\nTest complete!")
