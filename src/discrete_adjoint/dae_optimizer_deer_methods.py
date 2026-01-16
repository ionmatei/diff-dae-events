"""
DAE Optimizer using DEER iteration with multiple time discretization methods.

This module implements the DEER (Differentiate-Evaluate-Eliminate-Reuse) iteration
approach for various time discretization schemes:

1. Backward Euler (order 1)
2. Trapezoidal / Crank-Nicolson (order 2)
3. BDF2 - Backward Differentiation Formula order 2 (order 2)
4. BDF3 - Backward Differentiation Formula order 3 (order 3)
5. BDF4 - Backward Differentiation Formula order 4 (order 4)
6. BDF5 - Backward Differentiation Formula order 5 (order 5)
7. BDF6 - Backward Differentiation Formula order 6 (order 6, highest stable)

The DEER approach avoids Newton iteration by:
1. Linearizing the residual around the current iterate
2. Solving the resulting linear system using parallel associative scan
3. Iterating until convergence

This gives O(log N) parallel depth instead of O(N) sequential Newton solves.

Key insight: Any implicit time discretization can be written as:
    L[y] = f(y_shifted, x, params)
where L is a linear operator that can be inverted using matmul_recursive.

BDF coefficients (for constant step size):
- BDF1: y_n - y_{n-1} = h*f_n
- BDF2: (3/2)*y_n - 2*y_{n-1} + (1/2)*y_{n-2} = h*f_n
- BDF3: (11/6)*y_n - 3*y_{n-1} + (3/2)*y_{n-2} - (1/3)*y_{n-3} = h*f_n
- BDF4: (25/12)*y_n - 4*y_{n-1} + 3*y_{n-2} - (4/3)*y_{n-3} + (1/4)*y_{n-4} = h*f_n
- BDF5: (137/60)*y_n - 5*y_{n-1} + 5*y_{n-2} - (10/3)*y_{n-3} + (5/4)*y_{n-4} - (1/5)*y_{n-5} = h*f_n
- BDF6: (147/60)*y_n - 6*y_{n-1} + (15/2)*y_{n-2} - (20/3)*y_{n-3} + (15/4)*y_{n-4} - (6/5)*y_{n-5} + (1/6)*y_{n-6} = h*f_n
"""

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
from jax.lax import scan
import numpy as np
import re
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from functools import partial

# Import DEER utilities
from deer.deer_iter import deer_iteration
from deer.maths import matmul_recursive
from deer.utils import Result

# Enable float64
jax.config.update("jax_enable_x64", True)


# BDF coefficients: coefficients for [y_n, y_{n-1}, y_{n-2}, ...] and divisor for dt
# Format: (coeffs, dt_divisor) where sum(coeffs[1:]) terms go to RHS
BDF_COEFFICIENTS = {
    1: ([1.0, -1.0], 1.0),  # Backward Euler
    2: ([3.0/2.0, -2.0, 1.0/2.0], 1.0),
    3: ([11.0/6.0, -3.0, 3.0/2.0, -1.0/3.0], 1.0),
    4: ([25.0/12.0, -4.0, 3.0, -4.0/3.0, 1.0/4.0], 1.0),
    5: ([137.0/60.0, -5.0, 5.0, -10.0/3.0, 5.0/4.0, -1.0/5.0], 1.0),
    6: ([147.0/60.0, -6.0, 15.0/2.0, -20.0/3.0, 15.0/4.0, -6.0/5.0, 1.0/6.0], 1.0),
}


class DAEOptimizerDEERMethods:
    """
    DAE optimizer using DEER iteration with selectable time discretization.

    Supported methods:
    - 'backward_euler': First-order implicit (A-stable)
    - 'trapezoidal': Second-order implicit (A-stable), also known as Crank-Nicolson
    - 'bdf2': Second-order BDF (A-stable, L-stable)
    - 'bdf3': Third-order BDF (A-stable)
    - 'bdf4': Fourth-order BDF (A-stable)
    - 'bdf5': Fifth-order BDF (A-stable)
    - 'bdf6': Sixth-order BDF (A-stable, highest stable BDF)

    The DEER iteration avoids Newton by iterating:
        y^{k+1} = L^{-1}[f(y^k) + J(y^k) @ y^k]
    where J is the Jacobian and L^{-1} is solved via parallel associative scan.
    """

    def __init__(
        self,
        dae_data: Dict[str, Any],
        optimize_params: Optional[List[str]] = None,
        loss_type: str = 'sum',
        method: str = 'trapezoidal',
        deer_max_iter: int = 100,
        deer_atol: float = 1e-8,
        deer_rtol: float = 1e-6,
    ):
        """
        Initialize the DEER-methods optimizer.

        Args:
            dae_data: DAE specification dictionary
            optimize_params: Parameters to optimize (None = all)
            loss_type: 'sum' or 'mean'
            method: Time discretization method
            deer_max_iter: Maximum DEER iterations
            deer_atol: Absolute tolerance for DEER convergence
            deer_rtol: Relative tolerance for DEER convergence
        """
        self.dae_data = dae_data
        self.method = method
        self.deer_max_iter = deer_max_iter
        self.deer_atol = deer_atol
        self.deer_rtol = deer_rtol

        if loss_type not in ['sum', 'mean']:
            raise ValueError(f"loss_type must be 'sum' or 'mean'")
        self.loss_type = loss_type

        valid_methods = ['backward_euler', 'trapezoidal', 'bdf2', 'bdf3', 'bdf4', 'bdf5', 'bdf6']
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

        # Extract variable info
        self.state_names = [s['name'] for s in dae_data.get('states', [])]
        self.alg_var_names = [a['name'] for a in dae_data.get('alg_vars', [])]
        self.param_names = [p['name'] for p in dae_data.get('parameters', [])]

        self.n_states = len(self.state_names)
        self.n_alg = len(self.alg_var_names)
        self.n_total = self.n_states + self.n_alg
        self.n_params_total = len(self.param_names)

        # Initial conditions
        self.x0 = jnp.array([s['start'] for s in dae_data['states']], dtype=jnp.float64)
        self.z0 = jnp.array([a.get('start', 0.0) for a in dae_data['alg_vars']], dtype=jnp.float64)
        self.y0 = jnp.concatenate([self.x0, self.z0])

        self.p_all = jnp.array([p['value'] for p in dae_data['parameters']], dtype=jnp.float64)

        # Parameters to optimize
        if optimize_params is None:
            self.optimize_params = self.param_names.copy()
            self.optimize_indices = list(range(self.n_params_total))
        else:
            self.optimize_params = optimize_params
            self.optimize_indices = [
                self.param_names.index(name) for name in optimize_params
                if name in self.param_names
            ]

        self.n_params_opt = len(self.optimize_indices)
        self.optimize_indices_jax = jnp.array(self.optimize_indices, dtype=jnp.int32)

        # Parse equations
        self.f_eqs = self._parse_f_equations()
        self.g_eqs = self._parse_g_equations()
        self.h_eqs = self._parse_h_equations()

        self.n_outputs = len(self.h_eqs) if self.h_eqs else self.n_states

        # Build functions
        self._build_jit_functions()

        # History
        self.history = {
            'loss': [], 'gradient_norm': [], 'params': [],
            'params_all': [], 'step_size': [], 'time_per_iter': []
        }

        print(f"\nDAEOptimizerDEERMethods initialized:")
        print(f"  Method: {method}")
        print(f"  DEER max iter: {deer_max_iter}")
        print(f"  Parameters to optimize: {self.n_params_opt}")
        print(f"  States: {self.n_states}, Algebraic: {self.n_alg}")

    def _parse_f_equations(self):
        equations = []
        for eq_str in self.dae_data.get('f', []) or []:
            match = re.match(r'der\((\w+)\)\s*=\s*(.+)', eq_str.strip())
            if match:
                equations.append((match.group(1), match.group(2).strip()))
        return equations

    def _parse_g_equations(self):
        equations = []
        for eq_str in self.dae_data.get('g', []) or []:
            match = re.match(r'0(?:\.0*)?\s*=\s*(.+)', eq_str.strip())
            if match:
                equations.append(match.group(1).strip())
        return equations

    def _parse_h_equations(self):
        equations = []
        for eq_str in self.dae_data.get('h', []) or []:
            if eq_str.strip() in self.state_names + self.alg_var_names:
                equations.append(('output', eq_str.strip()))
            else:
                match = re.match(r'(\w+)\s*=\s*(.+)', eq_str.strip())
                if match:
                    equations.append((match.group(1), match.group(2).strip()))
                else:
                    equations.append(('output', eq_str.strip()))
        return equations

    def _build_jit_functions(self):
        """Build DEER-based simulation and optimization functions."""

        state_names = self.state_names
        alg_var_names = self.alg_var_names
        param_names = self.param_names
        f_eqs = self.f_eqs
        g_eqs = self.g_eqs
        h_eqs = self.h_eqs
        n_states = self.n_states
        n_alg = self.n_alg
        n_total = self.n_total
        method = self.method

        # Expression evaluation helpers
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
            math_funcs = {
                'exp': jnp.exp, 'log': jnp.log, 'log10': jnp.log10,
                'sqrt': jnp.sqrt, 'abs': jnp.abs,
                'sin': jnp.sin, 'cos': jnp.cos, 'tan': jnp.tan,
                'asin': jnp.arcsin, 'acos': jnp.arccos, 'atan': jnp.arctan,
                'sinh': jnp.sinh, 'cosh': jnp.cosh, 'tanh': jnp.tanh,
            }
            return eval(expr, {'__builtins__': {}, 'jnp': jnp, **math_funcs}, namespace)

        def eval_f(t, x, z, p):
            namespace = build_namespace(t, x, z, p)
            dxdt = jnp.zeros(n_states, dtype=jnp.float64)
            for state_name, expr in f_eqs:
                idx = state_names.index(state_name)
                dxdt = dxdt.at[idx].set(eval_expr(expr, namespace))
            return dxdt

        def eval_g(t, x, z, p):
            if not g_eqs:
                return jnp.zeros(0, dtype=jnp.float64)
            namespace = build_namespace(t, x, z, p)
            return jnp.array([eval_expr(expr, namespace) for expr in g_eqs])

        def eval_h(t, x, z, p):
            if not h_eqs:
                return x
            namespace = build_namespace(t, x, z, p)
            return jnp.array([eval_expr(expr, namespace) for _, expr in h_eqs])

        self._eval_f = eval_f
        self._eval_g = eval_g
        self._eval_h = eval_h

        # ================================================================
        # DEER residual functions for different methods
        # ================================================================

        if method == 'backward_euler':
            # Backward Euler: (y_i - y_{i-1})/dt = f(t_i, y_i)
            # Residual: (y_i - y_{i-1})/dt - f(t_i, y_i) = 0
            # Combined with algebraic: g(t_i, y_i) = 0

            def implicit_residual(dydt, y, t, p):
                """F(dy/dt, y, t, p) = 0"""
                x = y[:n_states]
                z = y[n_states:]
                dxdt = dydt[:n_states]

                f_val = eval_f(t, x, z, p)
                g_val = eval_g(t, x, z, p)

                R_x = dxdt - f_val
                return jnp.concatenate([R_x, g_val])

            def deer_func(yshifts, xinput, params):
                """DEER function for backward Euler: f(y_i, y_{i-1}, x, p)"""
                y_i, y_im1 = yshifts
                dt, t = xinput
                dydt = (y_i - y_im1) / dt
                return implicit_residual(dydt, y_i, t, params)

            def shifter_func(y, _):
                """Shift: [y_i, y_{i-1}]"""
                y_im1 = jnp.concatenate([y[:1], y[:-1]], axis=0)
                return [y, y_im1]

            p_num = 2

            def solve_inv_lin(jacs, z, inv_lin_params):
                """Solve M0 @ y_i + M1 @ y_{i-1} = z using parallel scan"""
                M0, M1 = jacs
                y0, = inv_lin_params

                M01 = M0[1:]
                M0invM1 = -vmap(jnp.linalg.solve)(M01, M1[1:])
                M0invz = vmap(jnp.linalg.solve)(M01, z[1:])
                return matmul_recursive(M0invM1, M0invz, y0)

        elif method == 'trapezoidal':
            # Trapezoidal: (y_i - y_{i-1})/dt = 0.5*(f(t_i, y_i) + f(t_{i-1}, y_{i-1}))
            # Residual: (y_i - y_{i-1})/dt - 0.5*f(t_i, y_i) - 0.5*f(t_{i-1}, y_{i-1}) = 0

            def implicit_residual_trap(y_i, y_im1, t_i, t_im1, dt, p):
                """Trapezoidal residual"""
                x_i = y_i[:n_states]
                z_i = y_i[n_states:]
                x_im1 = y_im1[:n_states]
                z_im1 = y_im1[n_states:]

                f_i = eval_f(t_i, x_i, z_i, p)
                f_im1 = eval_f(t_im1, x_im1, z_im1, p)
                g_i = eval_g(t_i, x_i, z_i, p)

                dxdt = (x_i - x_im1) / dt
                R_x = dxdt - 0.5 * (f_i + f_im1)

                return jnp.concatenate([R_x, g_i])

            def deer_func(yshifts, xinput, params):
                """DEER function for trapezoidal"""
                y_i, y_im1 = yshifts
                dt, t_i, t_im1 = xinput
                return implicit_residual_trap(y_i, y_im1, t_i, t_im1, dt, params)

            def shifter_func(y, _):
                """Shift: [y_i, y_{i-1}]"""
                y_im1 = jnp.concatenate([y[:1], y[:-1]], axis=0)
                return [y, y_im1]

            p_num = 2

            def solve_inv_lin(jacs, z, inv_lin_params):
                """Solve trapezoidal linear system"""
                M0, M1 = jacs
                y0, = inv_lin_params

                M01 = M0[1:]
                M0invM1 = -vmap(jnp.linalg.solve)(M01, M1[1:])
                M0invz = vmap(jnp.linalg.solve)(M01, z[1:])
                return matmul_recursive(M0invM1, M0invz, y0)

        elif method.startswith('bdf'):
            # Generic BDF handler for orders 2-6
            bdf_order = int(method[3])
            coeffs, _ = BDF_COEFFICIENTS[bdf_order]
            coeffs = jnp.array(coeffs, dtype=jnp.float64)
            n_history = bdf_order  # Number of past values needed

            def implicit_residual_bdf(y_shifts, t_i, dt, p, step_idx):
                """
                Generic BDF residual.
                y_shifts: list of [y_i, y_{i-1}, y_{i-2}, ...] up to order terms
                For early steps, falls back to lower-order BDF.
                """
                y_i = y_shifts[0]
                x_i = y_i[:n_states]
                z_i = y_i[n_states:]

                f_i = eval_f(t_i, x_i, z_i, p)
                g_i = eval_g(t_i, x_i, z_i, p)

                # Compute derivative approximation based on available history
                # For step k, we can use up to BDF(min(k, order))
                # Compute all possible BDF approximations and select based on step_idx

                x_shifts = [ys[:n_states] for ys in y_shifts]

                # Compute derivative using appropriate BDF order
                def compute_bdf_deriv(order_to_use):
                    c, _ = BDF_COEFFICIENTS[order_to_use]
                    c = jnp.array(c, dtype=jnp.float64)
                    dxdt = c[0] * x_shifts[0]
                    for j in range(1, order_to_use + 1):
                        dxdt = dxdt + c[j] * x_shifts[j]
                    return dxdt / dt

                # Select BDF order based on step index (use lower order for early steps)
                # step_idx: 0=initial, 1=first step (use BDF1), 2=second step (use BDF2), etc.
                dxdt = compute_bdf_deriv(1)  # Default BDF1
                for k in range(2, bdf_order + 1):
                    dxdt = jnp.where(step_idx >= k, compute_bdf_deriv(k), dxdt)

                R_x = dxdt - f_i
                return jnp.concatenate([R_x, g_i])

            def deer_func(yshifts, xinput, params):
                """DEER function for BDF"""
                dt, t_i, step_idx = xinput
                return implicit_residual_bdf(yshifts, t_i, dt, params, step_idx)

            def shifter_func(y, _):
                """Shift: [y_i, y_{i-1}, ..., y_{i-order}]"""
                shifts = [y]
                for k in range(1, n_history + 1):
                    # y_{i-k}: shift by k positions, pad with y[0]
                    y_shifted = jnp.concatenate([jnp.tile(y[:1], (k, 1)), y[:-k]], axis=0)
                    shifts.append(y_shifted)
                return shifts

            p_num = n_history + 1

            def solve_inv_lin(jacs, z, inv_lin_params):
                """Solve BDF linear system with multiple history terms"""
                # jacs is a list of [M0, M1, M2, ...] Jacobians
                y0, = inv_lin_params

                nsamples = z.shape[0]

                # For multi-step methods, solve sequentially
                # (parallel scan only works for 2-term recurrences)
                def solve_step(carry, inputs):
                    y_history = carry  # tuple of (y_{i-1}, y_{i-2}, ..., y_{i-order})
                    jac_inputs = inputs[:-1]  # M0, M1, ..., M_order
                    z_i = inputs[-1]

                    # Solve: M0 @ y_i = z_i - sum(M_k @ y_{i-k})
                    M0_i = jac_inputs[0]
                    rhs = z_i
                    for k in range(1, n_history + 1):
                        rhs = rhs - jac_inputs[k] @ y_history[k-1]
                    y_i = jnp.linalg.solve(M0_i, rhs)

                    # Shift history
                    new_history = (y_i,) + y_history[:-1]
                    return new_history, y_i

                # Initial history: all y0
                init_history = tuple(y0 for _ in range(n_history))

                # Prepare inputs: (M0[1:], M1[1:], ..., z[1:])
                scan_inputs = tuple(jac[1:] for jac in jacs) + (z[1:],)

                _, y_result = scan(solve_step, init_history, scan_inputs)

                return jnp.concatenate([y0[None, :], y_result], axis=0)

        # Store method-specific functions
        self._deer_func = deer_func
        self._shifter_func = shifter_func
        self._solve_inv_lin = solve_inv_lin
        self._p_num = p_num

        # ================================================================
        # Simulation using DEER iteration
        # ================================================================

        def simulate_deer(y0, t_array, p):
            """Simulate using DEER iteration"""
            nsamples = len(t_array)
            ny = n_total

            # Initial guess: constant y0
            yinit_guess = jnp.zeros((nsamples, ny), dtype=jnp.float64) + y0

            # Prepare time inputs
            dt_partial = t_array[1:] - t_array[:-1]
            dt = jnp.concatenate([dt_partial[:1], dt_partial], axis=0)

            if method == 'backward_euler':
                xinput = (dt, t_array)
            elif method == 'trapezoidal':
                t_im1 = jnp.concatenate([t_array[:1], t_array[:-1]], axis=0)
                xinput = (dt, t_array, t_im1)
            elif method.startswith('bdf'):
                # Step index for adaptive order selection
                step_idx = jnp.arange(nsamples, dtype=jnp.float64)
                xinput = (dt, t_array, step_idx)

            inv_lin_params = (y0,)

            result = deer_iteration(
                inv_lin=solve_inv_lin,
                func=deer_func,
                shifter_func=shifter_func,
                p_num=p_num,
                params=p,
                xinput=xinput,
                inv_lin_params=inv_lin_params,
                shifter_func_params=None,
                yinit_guess=yinit_guess,
                max_iter=self.deer_max_iter,
                clip_ytnext=True,
                atol=self.deer_atol,
                rtol=self.deer_rtol,
            )

            return result.value  # (nsamples, ny)

        self._simulate_deer = simulate_deer

        # Compute outputs
        def compute_outputs(y_traj, t_array, p):
            def single_output(y_t, t):
                x = y_t[:n_states]
                z = y_t[n_states:]
                return eval_h(t, x, z, p)
            return vmap(single_output)(y_traj, t_array)

        self._compute_outputs = compute_outputs

        # Full simulation
        def simulate_full(y0, t_array, p):
            y_traj = simulate_deer(y0, t_array, p)
            y_out = compute_outputs(y_traj, t_array, p)
            return y_traj, y_out

        self._simulate_full = simulate_full

        # Loss function
        def compute_loss(p_opt, y0, t_array, y_target):
            p_full = self.p_all.at[self.optimize_indices_jax].set(p_opt)
            _, y_out = simulate_full(y0, t_array, p_full)
            error = y_out - y_target
            if self.loss_type == 'mean':
                return jnp.mean(error**2)
            else:
                return jnp.sum(error**2)

        self._compute_loss = compute_loss
        self._loss_and_grad = jit(value_and_grad(compute_loss))

    def simulate(self, t_array: np.ndarray, p: Optional[np.ndarray] = None) -> Dict:
        """Simulate DAE."""
        t_jax = jnp.array(t_array, dtype=jnp.float64)
        p_jax = self.p_all if p is None else jnp.array(p, dtype=jnp.float64)

        y_traj, y_out = self._simulate_full(self.y0, t_jax, p_jax)

        return {
            't': np.array(t_jax),
            'x': np.array(y_traj[:, :self.n_states]).T,
            'z': np.array(y_traj[:, self.n_states:]).T,
            'y': np.array(y_out).T,
        }

    def optimization_step(self, t_array, y_target, p_opt, step_size=0.01):
        """Single optimization step."""
        t_jax = jnp.array(t_array, dtype=jnp.float64)
        y_target_jax = jnp.array(y_target, dtype=jnp.float64)
        p_opt_jax = jnp.array(p_opt, dtype=jnp.float64)

        if y_target_jax.shape[0] != t_jax.shape[0]:
            y_target_jax = y_target_jax.T

        loss, grad_p = self._loss_and_grad(p_opt_jax, self.y0, t_jax, y_target_jax)
        p_new = p_opt_jax - step_size * grad_p

        return np.array(p_new), float(loss), np.array(grad_p)

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
        """Optimize parameters."""
        if verbose:
            print("\n" + "=" * 80)
            print(f"DEER Optimization ({self.method})")
            print("=" * 80)

        if p_init is not None:
            p = jnp.array(p_init, dtype=jnp.float64)
        else:
            p = jnp.array([self.p_all[i] for i in self.optimize_indices], dtype=jnp.float64)

        self.history = {k: [] for k in self.history}
        converged = False

        y_target_use = np.array(y_target)
        if y_target_use.shape[0] != len(t_array):
            y_target_use = y_target_use.T

        for it in range(n_iterations):
            t_start = time.time()
            p_new, loss, grad_p = self.optimization_step(t_array, y_target_use, np.array(p), step_size)
            iter_time = time.time() - t_start

            grad_norm = float(np.linalg.norm(grad_p))

            self.history['loss'].append(loss)
            self.history['gradient_norm'].append(grad_norm)
            self.history['params'].append(np.array(p))
            self.history['time_per_iter'].append(iter_time)

            if verbose and (it % 10 == 0 or it == n_iterations - 1):
                print(f"Iter {it:4d}: loss={loss:.6e}, grad_norm={grad_norm:.6e}, time={iter_time:.3f}s")

            if grad_norm < tol:
                converged = True
                if verbose:
                    print(f"Converged at iteration {it}")
                break

            p = jnp.array(p_new)

        p_all_final = np.array(self.p_all)
        for i, idx in enumerate(self.optimize_indices):
            p_all_final[idx] = float(p[i])

        if verbose:
            print("=" * 80)
            print("Transformation Complete")
            print("=" * 80)
            print(f"  Converged: {converged}")
            print(f"  Final loss: {self.history['loss'][-1]:.6e}")
            print(f"  Final gradient norm: {self.history['gradient_norm'][-1]:.6e}")

            # Timing statistics
            times = self.history['time_per_iter']
            if times:
                total_time = sum(times)
                first_iter = times[0]
                
                if len(times) > 1:
                    subsequent_times = times[1:]
                    avg_time = np.mean(subsequent_times)
                    std_time = np.std(subsequent_times)
                    min_time = np.min(subsequent_times)
                    max_time = np.max(subsequent_times)
                else:
                    avg_time = first_iter
                    std_time = 0.0
                    min_time = first_iter
                    max_time = first_iter

                print(f"\n  Timing statistics:")
                print(f"    Total time:              {total_time:.3f}s")
                print(f"    First iteration:         {first_iter:.3f}s")
                print(f"    Avg time/iter:           {avg_time*1000:.3f}ms")
                print(f"    Std time/iter:           {std_time*1000:.3f}ms")
                print(f"    Min time/iter:           {min_time*1000:.3f}ms")
                print(f"    Max time/iter:           {max_time*1000:.3f}ms")

            print(f"\n  Optimized parameters (final values):")
            for name, val in zip(self.optimize_params, p):
                print(f"    {name:<20} = {float(val):.6f}")

        return {
            'p_opt': np.array(p),
            'p_all': p_all_final,
            'loss_final': self.history['loss'][-1],
            'history': self.history,
            'converged': converged,
            'n_iterations': len(self.history['loss'])
        }


# Test
if __name__ == "__main__":
    print("=" * 80)
    print("Testing DAEOptimizerDEERMethods")
    print("=" * 80)

    dae_data = {
        'states': [{'name': 'x', 'start': 1.0}],
        'alg_vars': [{'name': 'z', 'start': 1.0}],
        'parameters': [{'name': 'p', 'value': 0.5}],
        'f': ['der(x) = -p * x'],
        'g': ['0 = z - x * x'],
        'h': ['x']
    }

    t_test = np.linspace(0, 2, 51)
    x_exact = np.exp(-0.5 * t_test)

    for method in ['backward_euler', 'trapezoidal', 'bdf2']:
        print(f"\n--- {method} ---")
        opt = DAEOptimizerDEERMethods(dae_data, method=method, deer_max_iter=50)
        result = opt.simulate(t_test)
        error = np.abs(result['x'][0, -1] - x_exact[-1])
        print(f"x[-1] = {result['x'][0, -1]:.6f}, exact = {x_exact[-1]:.6f}, error = {error:.2e}")

    print("\n" + "=" * 80)
    print("Test complete!")
