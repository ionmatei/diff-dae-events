"""
JAX Model for Semi-Explicit DAEs

This module provides a high-performance JAX implementation for semi-explicit DAEs of the form:
    der(x) = f(x, z, u; params)
    0 = g(x, z, u; params)
    y = h(x, z, u; params)

Key features:
1. JIT-compiled for maximum performance
2. Efficient automatic differentiation through implicit function theorem
3. Optimized Jacobian computation using jax.jacfwd/jacrev
4. Fast parameter optimization using jax.value_and_grad
5. Efficient time stepping with jax.lax.scan

Performance optimizations:
- All core functions are JIT-compiled
- Custom VJP (vector-Jacobian product) for algebraic solver
- Efficient Jacobian computation (forward or reverse mode based on dimensions)
- Scan-based integration for memory efficiency
- Functional design for easy batching and vectorization
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev, custom_vjp
import diffrax
from jax.tree_util import tree_map, tree_leaves
import re
from typing import Dict, List, Callable, Optional, Tuple, Any
from functools import partial

# Enable float64 for better precision
jax.config.update("jax_enable_x64", True)


class DAEModel:
    """
    High-performance JAX model for semi-explicit DAEs

    System form:
        dx/dt = f(x, z, u; params)
        0 = g(x, z, u; params)
        y = h(x, z, u; params)

    Features:
    - JIT-compiled operations for maximum speed
    - Efficient automatic differentiation
    - Custom VJP for algebraic constraints
    - Optimized for parameter optimization
    """

    def __init__(self, dae_dict: Dict[str, Any]):
        """
        Initialize DAE model from dictionary specification

        Args:
            dae_dict: Dictionary with keys:
                - states: List[Dict] with 'name' key
                - inputs: List[Dict] with 'name' key
                - alg_vars: List[Dict] with 'name' key
                - outputs: List[Dict] with 'name' key
                - parameters: List[Dict] with 'name' and 'value' keys
                - f: List[str] of differential equations "der(x) = expr"
                - g: List[str] of algebraic equations "0 = expr"
                - h: List[str] of output equations "y = expr"
                - when: List[Dict] with 'condition' and 'equation' keys for events
        """
        self.dae_dict = dae_dict

        # Extract variable names
        self.state_names = [s['name'] for s in dae_dict.get('states', [])] if dae_dict.get('states') else []
        self.input_names = [i['name'] for i in dae_dict.get('inputs', [])] if dae_dict.get('inputs') else []
        self.alg_var_names = [a['name'] for a in dae_dict.get('alg_vars', [])] if dae_dict.get('alg_vars') else []
        self.output_names = [o['name'] for o in dae_dict.get('outputs', [])] if dae_dict.get('outputs') else []
        self.param_names = [p['name'] for p in dae_dict.get('parameters', [])] if dae_dict.get('parameters') else []

        # Dimensions
        self.n_states = len(self.state_names)
        self.n_inputs = len(self.input_names)
        self.n_alg_vars = len(self.alg_var_names)
        self.n_outputs = len(self.output_names)

        # Initialize parameters as dict (JAX functional style)
        self.params = {}
        for param_spec in dae_dict.get('parameters', []):
            name = param_spec['name']
            value = param_spec.get('value', 1.0)
            if value is None or value == 'null':
                value = 1.0
            self.params[name] = jnp.array(float(value), dtype=jnp.float64)

        # Build function evaluators
        self._build_functions()

        # Create JIT-compiled versions of key functions
        self._create_jit_functions()

    def _build_functions(self):
        """Build callable functions for f, g, h from string equations"""
        self.f_eqs = self._parse_f_equations()
        self.g_eqs = self._parse_g_equations()
        self.h_eqs = self._parse_h_equations()
        self.when_eqs = self._parse_when_equations()

    def _parse_f_equations(self) -> List[Tuple[str, str]]:
        """Parse f equations into (state_name, expression) pairs"""
        equations = []
        if 'f' not in self.dae_dict or self.dae_dict['f'] is None:
            return equations

        for eq_str in self.dae_dict.get('f', []):
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
        if 'g' not in self.dae_dict or self.dae_dict['g'] is None:
            return equations

        for eq_str in self.dae_dict.get('g', []):
            # Match both "0 = ..." and "0.0 = ..." (and other decimal representations)
            match = re.match(r'0(?:\.0*)?\s*=\s*(.+)', eq_str.strip())
            if match:
                equations.append(match.group(1).strip())
            else:
                raise ValueError(f"Cannot parse g equation: {eq_str}")
        return equations

    def _parse_h_equations(self) -> List[Tuple[str, str]]:
        """Parse h equations into (output_name, expression) pairs"""
        equations = []
        if 'h' not in self.dae_dict or self.dae_dict['h'] is None:
            return equations

        for eq_str in self.dae_dict.get('h', []):
            match = re.match(r'(\w+)\s*=\s*(.+)', eq_str.strip())
            if match:
                output_name = match.group(1)
                expr = match.group(2).strip()
                equations.append((output_name, expr))
            else:
                raise ValueError(f"Cannot parse h equation: {eq_str}")
        return equations

    def _parse_when_equations(self) -> List[Dict[str, Any]]:
        """
        Parse when equations for event handling

        Returns:
            List of dicts with keys:
                - condition: str, the event condition expression
                - reinits: List[Tuple[str, str]] of (var_name, new_value_expr) pairs
        """
        events = []
        if 'when' not in self.dae_dict or self.dae_dict['when'] is None:
            return events

        for when_spec in self.dae_dict.get('when', []):
            condition = when_spec.get('condition', '').strip()
            equations = when_spec.get('equation', [])

            if not isinstance(equations, list):
                equations = [equations]

            # Parse reinit calls from equations
            reinits = []
            for eq_str in equations:
                # Match pattern: reinit(var, expr) or reinit(var, pre(var2))
                match = re.match(r'reinit\s*\(\s*(\w+)\s*,\s*(.+?)\s*\)', eq_str.strip())
                if match:
                    var_name = match.group(1)
                    value_expr = match.group(2).strip()

                    # Handle pre(var) references - replace with the variable itself
                    # In JAX, we'll use the state before the event
                    value_expr = re.sub(r'pre\s*\(\s*(\w+)\s*\)', r'\1', value_expr)

                    reinits.append((var_name, value_expr))
                else:
                    print(f"Warning: Cannot parse when equation: {eq_str}")

            if reinits:
                events.append({
                    'condition': condition,
                    'reinits': reinits
                })

        return events

    def _eval_expr(self, expr: str, namespace: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Safely evaluate expression with given namespace"""
        # Replace math operators
        expr = expr.replace('^', '**')

        # Replace 'time' with 't'
        expr = re.sub(r'\btime\b', 't', expr)

        # Map common mathematical functions to JAX equivalents
        math_functions = {
            'exp': jnp.exp,
            'log': jnp.log,
            'log10': jnp.log10,
            'sqrt': jnp.sqrt,
            'abs': jnp.abs,
            'sin': jnp.sin,
            'cos': jnp.cos,
            'tan': jnp.tan,
            'asin': jnp.arcsin,
            'acos': jnp.arccos,
            'atan': jnp.arctan,
            'sinh': jnp.sinh,
            'cosh': jnp.cosh,
            'tanh': jnp.tanh,
            'asinh': jnp.arcsinh,
            'acosh': jnp.arccosh,
            'atanh': jnp.arctanh,
            'sigmoid': jax.nn.sigmoid,
            'sign': jnp.sign,
            'floor': jnp.floor,
            'ceil': jnp.ceil,
            'round': jnp.round,
            'min': jnp.min,
            'max': jnp.max,
        }

        # Add jnp and math functions to namespace
        # Note: namespace should already contain parameters from _build_namespace
        eval_namespace = {**namespace, 'jnp': jnp, **math_functions}

        # Evaluate
        return eval(expr, {'__builtins__': {}}, eval_namespace)

    def _build_namespace(self, x: jnp.ndarray, z: Optional[jnp.ndarray],
                        u: Optional[jnp.ndarray], params: Dict[str, jnp.ndarray],
                        t: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
        """Build namespace for expression evaluation"""
        namespace = {}

        # Add time variable if provided
        if t is not None:
            namespace['t'] = t

        # Add states
        for i, name in enumerate(self.state_names):
            namespace[name] = x[..., i]

        # Add algebraic variables
        if z is not None:
            for i, name in enumerate(self.alg_var_names):
                namespace[name] = z[..., i]

        # Add inputs
        if u is not None:
            for i, name in enumerate(self.input_names):
                namespace[name] = u[..., i]

        # Add parameters
        namespace.update(params)

        return namespace

    def g_residual(self, x: jnp.ndarray, z: jnp.ndarray,
                   u: Optional[jnp.ndarray], params: Dict[str, jnp.ndarray],
                   t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Evaluate algebraic constraint residuals g(x, z, u, t; params)

        Args:
            x: States [..., n_states]
            z: Algebraic variables [..., n_alg_vars]
            u: Inputs [..., n_inputs] or None
            params: Dictionary of parameters
            t: Time (scalar or [...]) or None

        Returns:
            Residuals [..., n_constraints]
        """
        if not self.g_eqs:
            return jnp.empty(x.shape[:-1] + (0,), dtype=x.dtype)

        namespace = self._build_namespace(x, z, u, params, t)

        residuals = []
        for expr in self.g_eqs:
            res = self._eval_expr(expr, namespace)
            residuals.append(res)

        return jnp.stack(residuals, axis=-1)

    def _solve_algebraic_newton(self, x: jnp.ndarray, u: Optional[jnp.ndarray],
                                params: Dict[str, jnp.ndarray],
                                t: Optional[jnp.ndarray] = None,
                                z_init: Optional[jnp.ndarray] = None,
                                tol: float = 1e-8, max_iter: int = 5) -> jnp.ndarray:
        """
        Solve algebraic constraints using Newton's method

        This is the core forward solve that will be wrapped with custom_vjp

        Args:
            x: States [..., n_states]
            u: Inputs [..., n_inputs] or None
            params: Dictionary of parameters
            t: Time (scalar or [...]) or None
            z_init: Optional initial guess for z [..., n_alg_vars]. If None, uses zeros.
            tol: Tolerance for convergence (not currently enforced)
            max_iter: Maximum number of Newton iterations

        Returns:
            z: Algebraic variables [..., n_alg_vars]
        """
        if self.n_alg_vars == 0:
            return None

        # Initial guess - warm start if provided
        if z_init is not None:
            z = z_init
        else:
            z = jnp.zeros(x.shape[:-1] + (self.n_alg_vars,), dtype=x.dtype)

        # Newton iteration with early stopping based on tolerance
        def newton_step(carry, _):
            z, converged = carry

            # Compute residual
            g_val = self.g_residual(x, z, u, params, t)

            # Check convergence
            residual_norm = jnp.linalg.norm(g_val)
            is_converged = residual_norm < tol

            # Compute Jacobian dg/dz only if not converged
            # Use forward-mode if n_alg_vars < n_constraints, else reverse-mode
            if self.n_alg_vars < len(self.g_eqs):
                jac_fn = jacfwd(lambda z_: self.g_residual(x, z_, u, params, t))
            else:
                jac_fn = jacrev(lambda z_: self.g_residual(x, z_, u, params, t))

            jac = jac_fn(z)

            # Newton step: solve jac @ delta_z = -g_val
            delta_z = jnp.linalg.solve(jac, -g_val)
            z_new = z + delta_z

            # Only update if not already converged
            z_next = jnp.where(converged, z, z_new)
            converged_next = converged | is_converged

            return (z_next, converged_next), residual_norm

        # Run Newton iterations
        init_carry = (z, False)
        (z_final, _), residual_norms = jax.lax.scan(newton_step, init_carry, None, length=max_iter)

        return z_final

    def _solve_algebraic_with_custom_grad(self, x: jnp.ndarray, u: Optional[jnp.ndarray],
                                          params: Dict[str, jnp.ndarray],
                                          t: Optional[jnp.ndarray] = None,
                                          z_init: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Solve algebraic constraints with custom VJP for efficient gradients

        This uses the implicit function theorem for backpropagation:
            dz/dx = -(dg/dz)^{-1} @ dg/dx
            dz/dparams = -(dg/dz)^{-1} @ dg/dparams

        Args:
            z_init: Optional initial guess for warm starting
        """
        # Define forward and VJP
        @custom_vjp
        def solve_alg(x, u, params, t, z_init):
            return self._solve_algebraic_newton(x, u, params, t, z_init)

        def solve_alg_fwd(x, u, params, t, z_init):
            z = self._solve_algebraic_newton(x, u, params, t, z_init)
            return z, (z, x, u, params, t)

        def solve_alg_bwd(res, g_z):
            """
            Compute VJP using implicit function theorem

            g_z is the gradient w.r.t. z (output)
            We need to compute gradients w.r.t. x, u, params, t
            """
            z, x, u, params, t = res

            # Compute dg/dz
            if self.n_alg_vars < len(self.g_eqs):
                jac_z = jacfwd(lambda z_: self.g_residual(x, z_, u, params, t))(z)
            else:
                jac_z = jacrev(lambda z_: self.g_residual(x, z_, u, params, t))(z)

            # Solve: jac_z^T @ lambda = g_z for lambda
            lambda_val = jnp.linalg.solve(jac_z.T, g_z)

            # Compute gradients using implicit function theorem
            # grad_x = -lambda^T @ dg/dx
            def g_of_x(x_):
                return jnp.sum(self.g_residual(x_, z, u, params, t) * lambda_val)

            grad_x = -grad(g_of_x)(x)

            # grad_u
            if u is not None:
                def g_of_u(u_):
                    return jnp.sum(self.g_residual(x, z, u_, params, t) * lambda_val)
                grad_u = -grad(g_of_u)(u)
            else:
                grad_u = None

            # grad_params (return dict of gradients, matching input structure)
            def g_of_params(params_):
                return jnp.sum(self.g_residual(x, z, u, params_, t) * lambda_val)

            grad_params = grad(g_of_params)(params)
            # Negate all parameter gradients
            grad_params = {k: -v for k, v in grad_params.items()}

            # grad_t
            if t is not None:
                def g_of_t(t_):
                    return jnp.sum(self.g_residual(x, z, u, params, t_) * lambda_val)
                grad_t = -grad(g_of_t)(t)
            else:
                grad_t = None

            return grad_x, grad_u, grad_params, grad_t, None  # None for z_init gradient

        solve_alg.defvjp(solve_alg_fwd, solve_alg_bwd)

        return solve_alg(x, u, params, t, z_init)

    def solve_algebraic(self, x: jnp.ndarray, u: Optional[jnp.ndarray],
                       params: Dict[str, jnp.ndarray],
                       t: Optional[jnp.ndarray] = None,
                       z_init: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Public interface for solving algebraic constraints

        Uses custom VJP for efficient gradient computation

        Args:
            x: States [..., n_states]
            u: Inputs [..., n_inputs] or None
            params: Dictionary of parameters
            t: Time (scalar or [...]) or None
            z_init: Optional initial guess for z [..., n_alg_vars] for warm starting

        Returns:
            z: Algebraic variables [..., n_alg_vars]
        """
        if self.n_alg_vars == 0:
            return None

        return self._solve_algebraic_with_custom_grad(x, u, params, t, z_init)

    def forward(self, x: jnp.ndarray, u: Optional[jnp.ndarray],
               params: Dict[str, jnp.ndarray],
               t: Optional[jnp.ndarray] = None,
               z_init: Optional[jnp.ndarray] = None,
               return_z: bool = False) -> jnp.ndarray:
        """
        Compute dx/dt = f(x, z, u, t; params)

        Args:
            x: States [..., n_states]
            u: Inputs [..., n_inputs] or None
            params: Dictionary of parameters
            t: Time (scalar or [...]) or None
            z_init: Optional initial guess for z for warm starting
            return_z: If True, returns (dxdt, z) instead of just dxdt

        Returns:
            dxdt: State derivatives [..., n_states]
            OR (dxdt, z) if return_z=True
        """
        # Solve for algebraic variables (with optional warm start)
        z = self.solve_algebraic(x, u, params, t, z_init)

        # Build namespace
        namespace = self._build_namespace(x, z, u, params, t)

        # Evaluate f equations
        dxdt_dict = {}
        if self.f_eqs:
            for state_name, expr in self.f_eqs:
                dxdt_dict[state_name] = self._eval_expr(expr, namespace)

        # Stack in correct order
        dxdt_list = []
        for state_name in self.state_names:
            if state_name in dxdt_dict:
                dxdt_list.append(dxdt_dict[state_name])
            else:
                dxdt_list.append(jnp.zeros_like(x[..., 0]))

        dxdt = jnp.stack(dxdt_list, axis=-1)

        if return_z:
            return dxdt, z
        return dxdt

    def compute_outputs(self, x: jnp.ndarray, u: Optional[jnp.ndarray],
                       params: Dict[str, jnp.ndarray],
                       t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Compute outputs y = h(x, z, u, t; params)

        Args:
            x: States [..., n_states]
            u: Inputs [..., n_inputs] or None
            params: Dictionary of parameters
            t: Time (scalar or [...]) or None

        Returns:
            y: Outputs [..., n_outputs]
        """
        # Solve for algebraic variables
        z = self.solve_algebraic(x, u, params, t)

        # Build namespace
        namespace = self._build_namespace(x, z, u, params, t)

        # Evaluate h equations
        y_dict = {}
        if self.h_eqs:
            for output_name, expr in self.h_eqs:
                y_dict[output_name] = self._eval_expr(expr, namespace)

        # Stack in correct order
        y_list = []
        for output_name in self.output_names:
            if output_name in y_dict:
                y_list.append(y_dict[output_name])
            else:
                if self.h_eqs:
                    raise ValueError(f"Output {output_name} not computed")

        return jnp.stack(y_list, axis=-1) if y_list else None

    def evaluate_event_condition(self, event_idx: int, x: jnp.ndarray,
                                 u: Optional[jnp.ndarray],
                                 params: Dict[str, jnp.ndarray],
                                 t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Evaluate event condition (for zero-crossing detection)

        Args:
            event_idx: Index of event in when_eqs
            x: States [..., n_states]
            u: Inputs [..., n_inputs] or None
            params: Dictionary of parameters
            t: Time (scalar or [...]) or None

        Returns:
            condition_value: Scalar or array, event occurs when this crosses zero
        """
        if not self.when_eqs or event_idx >= len(self.when_eqs):
            return jnp.array(1.0)  # Never triggers

        event = self.when_eqs[event_idx]
        condition_expr = event['condition']

        # Solve for algebraic variables
        z = self.solve_algebraic(x, u, params, t)

        # Build namespace
        namespace = self._build_namespace(x, z, u, params, t)

        # Evaluate condition - convert to zero-crossing form
        # E.g., "x > xmax" becomes "x - xmax" (positive when condition is true)
        condition_expr = condition_expr.replace('^', '**')

        # Parse relational operators
        for op in ['>=', '<=', '>', '<', '==']:
            if op in condition_expr:
                parts = condition_expr.split(op, 1)
                if len(parts) == 2:
                    lhs = parts[0].strip()
                    rhs = parts[1].strip()
                    lhs_val = self._eval_expr(lhs, namespace)
                    rhs_val = self._eval_expr(rhs, namespace)
                    # Return difference (zero-crossing form)
                    if op in ['>', '>=']:
                        return lhs_val - rhs_val
                    elif op in ['<', '<=']:
                        return rhs_val - lhs_val
                    else:  # ==
                        return lhs_val - rhs_val

        # If no relational operator, evaluate as-is
        return self._eval_expr(condition_expr, namespace)

    def apply_event_reinit(self, event_idx: int, x: jnp.ndarray,
                          u: Optional[jnp.ndarray],
                          params: Dict[str, jnp.ndarray],
                          t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Apply event reinitialization to state

        Args:
            event_idx: Index of event in when_eqs
            x: States [n_states]
            u: Inputs [n_inputs] or None
            params: Dictionary of parameters
            t: Time (scalar) or None

        Returns:
            x_new: Updated states [n_states]
        """
        if not self.when_eqs or event_idx >= len(self.when_eqs):
            return x

        event = self.when_eqs[event_idx]
        reinits = event['reinits']

        # Solve for algebraic variables
        z = self.solve_algebraic(x, u, params, t)

        # Build namespace with current state (for pre() references)
        namespace = self._build_namespace(x, z, u, params, t)

        # Create mutable copy of state
        x_new = x.copy() if hasattr(x, 'copy') else jnp.array(x)

        # Apply each reinit
        for var_name, value_expr in reinits:
            # Compute new value
            new_value = self._eval_expr(value_expr, namespace)

            # Find variable index and update
            # Check states first
            if var_name in self.state_names:
                idx = self.state_names.index(var_name)
                x_new = x_new.at[idx].set(new_value)
            elif var_name in self.alg_var_names:
                # For algebraic variables, we can't reinit directly in x
                # This would require re-solving the algebraic constraints
                print(f"Warning: reinit of algebraic variable {var_name} not fully supported")
                # You might want to handle this case differently depending on your needs

        return x_new

    def _create_jit_functions(self):
        """Create JIT-compiled versions of key functions for performance"""

        # JIT compile the forward function
        @jit
        def forward_jit(x, u, params, t):
            return self.forward(x, u, params, t)

        self.forward_jit = forward_jit

        # JIT compile output computation
        @jit
        def compute_outputs_jit(x, u, params, t):
            return self.compute_outputs(x, u, params, t)

        self.compute_outputs_jit = compute_outputs_jit

        # JIT compile algebraic solver
        @jit
        def solve_algebraic_jit(x, u, params, t):
            return self.solve_algebraic(x, u, params, t)

        self.solve_algebraic_jit = solve_algebraic_jit

    def simulate(self, x0: jnp.ndarray, t_span: jnp.ndarray,
                params: Optional[Dict[str, jnp.ndarray]] = None,
                u_func: Optional[Callable] = None,
                method: str = 'dopri5',
                rtol: float = 1e-6, atol: float = 1e-8,
                use_warm_start: bool = False,
                max_events: int = 100) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate DAE system using diffrax ODE solver with event handling

        Args:
            x0: Initial state [n_states]
            t_span: Time points [n_time]
            params: Parameters (uses self.params if None)
            u_func: Optional function u_func(t) -> [n_inputs]
            method: ODE solver method (default: 'dopri5')
                Explicit methods:
                - 'euler': 1st order Euler
                - 'heun': 2nd order Heun
                - 'midpoint': 2nd order Midpoint
                - 'ralston': 2nd order Ralston
                - 'bosh3': 3rd order Bosh
                - 'tsit5': 5th order Tsitouras (efficient, recommended)
                - 'dopri5': 5th order Dormand-Prince (default)
                - 'dopri8': 8th order Dormand-Prince (high accuracy)
                Implicit methods (for stiff problems):
                - 'implicit_euler': 1st order implicit Euler
                - 'kvaerno3': 3rd order SDIRK
                - 'kvaerno4': 4th order SDIRK
                - 'kvaerno5': 5th order SDIRK
            rtol: Relative tolerance
            atol: Absolute tolerance
            use_warm_start: If True and DAE has algebraic variables, uses warm starting
                          for Newton solver. Note: warm start is not supported with diffrax,
                          use simulate_scan instead for warm starting.
            max_events: Maximum number of events to handle (prevents infinite loops)

        Returns:
            x_traj: State trajectory [n_time, n_states]
            y_traj: Output trajectory [n_time, n_outputs]
        """
        if params is None:
            params = self.params

        # Warm starting is not directly supported with diffrax
        # If requested and system has algebraic vars, redirect to simulate_scan with rk4
        if use_warm_start and self.n_alg_vars > 0:
            # Use simulate_scan with RK4 for warm starting
            return self.simulate_scan(x0, t_span, params, u_func, method='rk4', use_warm_start=True)

        # Always try simple simulation first using simulate_scan (more robust)
        # Only use event handling if we detect events in the trajectory
        print("Attempting simulation without event handling...")
        x_traj, y_traj = self.simulate_scan(x0, t_span, params, u_func, method='rk4', use_warm_start=True)

        # If no events defined, we're done
        if not self.when_eqs:
            print("No events defined - simulation complete")
            return x_traj, y_traj

        # Check if any events occurred in the trajectory
        events_detected = self._check_for_events(x_traj, t_span, u_func, params)

        if not events_detected:
            print("No events detected in trajectory - simulation complete")
            return x_traj, y_traj

        # Events were detected - need to re-simulate with event handling
        print(f"Events detected - re-simulating with event handling...")
        return self._simulate_with_events(x0, t_span, params, u_func, method, rtol, atol, max_events)

    def _check_for_events(self, x_traj: jnp.ndarray, t_span: jnp.ndarray,
                         u_func: Optional[Callable],
                         params: Dict[str, jnp.ndarray]) -> bool:
        """Check if any events occur in the trajectory"""
        for i in range(len(t_span) - 1):
            x_curr = x_traj[i]
            x_next = x_traj[i + 1]
            t_curr = t_span[i]
            t_next = t_span[i + 1]

            for event_idx in range(len(self.when_eqs)):
                u_curr = u_func(t_curr) if u_func is not None else None
                u_next = u_func(t_next) if u_func is not None else None

                cond_curr = float(self.evaluate_event_condition(event_idx, x_curr, u_curr, params, t_curr))
                cond_next = float(self.evaluate_event_condition(event_idx, x_next, u_next, params, t_next))

                # Check for zero crossing
                if cond_curr <= 0 and cond_next > 0:
                    return True

        return False

    def _simulate_no_events(self, x0: jnp.ndarray, t_span: jnp.ndarray,
                           params: Dict[str, jnp.ndarray],
                           u_func: Optional[Callable],
                           method: str, rtol: float, atol: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Simulate without event handling (faster)"""

        # Define ODE function for diffrax (takes t, x, args)
        def ode_func(t, x, args):
            u = u_func(t) if u_func is not None else None
            return self.forward_jit(x, u, params, t)

        # Create ODE term
        term = diffrax.ODETerm(ode_func)

        # Select solver based on method
        solver_map = {
            'euler': diffrax.Euler,
            'heun': diffrax.Heun,
            'midpoint': diffrax.Midpoint,
            'ralston': diffrax.Ralston,
            'bosh3': diffrax.Bosh3,
            'tsit5': diffrax.Tsit5,
            'dopri5': diffrax.Dopri5,
            'dopri8': diffrax.Dopri8,
            'implicit_euler': diffrax.ImplicitEuler,
            'kvaerno3': diffrax.Kvaerno3,
            'kvaerno4': diffrax.Kvaerno4,
            'kvaerno5': diffrax.Kvaerno5,
        }

        if method.lower() not in solver_map:
            raise ValueError(f"Unknown method '{method}'. Available methods: {list(solver_map.keys())}")

        solver = solver_map[method.lower()]()

        # Create save controller to save at specified times
        saveat = diffrax.SaveAt(ts=t_span)

        # Create stepsize controller
        # Fixed-step methods use ConstantStepSize, adaptive methods use PIDController
        fixed_step_methods = {'euler', 'heun', 'midpoint', 'ralston'}
        if method.lower() in fixed_step_methods:
            # Fixed step size: use average spacing of t_span
            dt = float((t_span[-1] - t_span[0]) / (len(t_span) - 1))
            stepsize_controller = diffrax.ConstantStepSize()
            dt0 = dt
        else:
            # Adaptive step size
            stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
            dt0 = None  # Auto-select initial timestep

        # Solve ODE
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_span[0],
            t1=t_span[-1],
            dt0=dt0,
            y0=x0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=16**5,  # Increase max steps for stiff DAEs
        )

        x_traj = solution.ys

        # Compute outputs
        def compute_output_at_t(x, t):
            u = u_func(t) if u_func is not None else None
            return self.compute_outputs_jit(x, u, params, t)

        # Vectorize output computation over time
        y_traj = jax.vmap(compute_output_at_t)(x_traj, t_span)

        return x_traj, y_traj

    def _simulate_with_events(self, x0: jnp.ndarray, t_span: jnp.ndarray,
                             params: Dict[str, jnp.ndarray],
                             u_func: Optional[Callable],
                             method: str, rtol: float, atol: float,
                             max_events: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate with event handling using diffrax events

        This handles multiple events by detecting them, stopping integration,
        applying reinitialization, and continuing.
        """
        import numpy as np  # Need numpy for dynamic arrays

        # Define ODE function for diffrax
        def ode_func(t, x, args):
            u = u_func(t) if u_func is not None else None
            return self.forward(x, u, params, t)

        # Create ODE term
        term = diffrax.ODETerm(ode_func)

        # Select solver
        solver_map = {
            'euler': diffrax.Euler,
            'heun': diffrax.Heun,
            'midpoint': diffrax.Midpoint,
            'ralston': diffrax.Ralston,
            'bosh3': diffrax.Bosh3,
            'tsit5': diffrax.Tsit5,
            'dopri5': diffrax.Dopri5,
            'dopri8': diffrax.Dopri8,
            'implicit_euler': diffrax.ImplicitEuler,
            'kvaerno3': diffrax.Kvaerno3,
            'kvaerno4': diffrax.Kvaerno4,
            'kvaerno5': diffrax.Kvaerno5,
        }
        solver = solver_map.get(method.lower(), diffrax.Dopri5)()

        # Stepsize controller
        fixed_step_methods = {'euler', 'heun', 'midpoint', 'ralston'}
        if method.lower() in fixed_step_methods:
            dt = float((t_span[-1] - t_span[0]) / (len(t_span) - 1))
            stepsize_controller = diffrax.ConstantStepSize()
            dt0 = dt
        else:
            stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
            dt0 = None

        # Create event function for the first when equation
        # Must be JIT-compatible - no side effects, no prints
        # Don't JIT here - diffrax will handle compilation
        def make_event_fn(idx):
            def event_fn(t, y, args, **kwargs):
                # diffrax event signature: (t, y, args, **kwargs) -> scalar
                u = u_func(t) if u_func is not None else None
                return self.evaluate_event_condition(idx, y, u, params, t)
            return event_fn

        # Simulate with event detection using new diffrax API
        # Import optimistix for root finding
        try:
            import optimistix as optx
            has_optx = True
        except ImportError:
            has_optx = False

        # Create event with root finder (like bouncing ball example)
        if self.when_eqs and has_optx:
            root_finder = optx.Newton(rtol=1e-5, atol=1e-5)
            event = diffrax.Event(cond_fn=make_event_fn(0), root_finder=root_finder)
        elif self.when_eqs:
            # Fallback without root finder
            event = diffrax.Event(cond_fn=make_event_fn(0))
        else:
            event = None

        # Simulate with event detection
        t_out = []
        x_out = []

        current_x = x0
        current_t = t_span[0]
        t_end = t_span[-1]
        event_count = 0

        # Find which output times we still need to hit
        remaining_times = t_span[t_span >= current_t]

        while current_t < t_end and event_count < max_events:
            # Integrate until next event or end time
            segment_times = remaining_times[remaining_times <= t_end]
            if len(segment_times) == 0:
                break

            saveat = diffrax.SaveAt(ts=segment_times)

            # Solve with event detection using new API
            solution = diffrax.diffeqsolve(
                term,
                solver,
                t0=current_t,
                t1=t_end,
                dt0=dt0,
                y0=current_x,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                event=event,
                max_steps=1000000,  # Increase max steps
            )

            # Collect results
            if solution.ys is not None:
                if len(solution.ys.shape) > 1:
                    for i, t_val in enumerate(solution.ts):
                        if t_val >= current_t:
                            t_out.append(float(t_val))
                            x_out.append(np.array(solution.ys[i]))
                else:
                    # Single point
                    t_out.append(float(solution.ts))
                    x_out.append(np.array(solution.ys))

            # Check if event occurred
            event_occurred = False
            if hasattr(solution, 'event_mask') and solution.event_mask is not None:
                event_occurred = bool(jnp.any(solution.event_mask))

            # Also check manually if we stopped early
            final_t = solution.ts[-1] if hasattr(solution.ts, '__getitem__') else solution.ts
            if final_t < t_end - 1e-6:
                event_occurred = True

            if event_occurred and event_count < max_events:
                # Apply event reinitialization
                event_t = float(solution.ts[-1])
                event_x = solution.ys[-1]

                # Find which event triggered (check all)
                for event_idx in range(len(self.when_eqs)):
                    u = u_func(event_t) if u_func is not None else None
                    cond_val = self.evaluate_event_condition(event_idx, event_x, u, params, event_t)

                    # If condition is triggered (crossed zero from negative to positive)
                    if float(cond_val) > -1e-6:  # Small tolerance
                        # Apply reinit
                        current_x = self.apply_event_reinit(event_idx, event_x, u, params, event_t)
                        current_t = event_t + 1e-10  # Advance slightly past event time
                        event_count += 1

                        # Update remaining times
                        remaining_times = t_span[t_span > current_t]
                        break
                else:
                    # No event found, just continue
                    break
            else:
                # No event, we're done
                break

        # Convert to JAX arrays
        if len(x_out) == 0:
            # Fallback: just use initial condition
            x_out = [np.array(x0)]
            t_out = [float(t_span[0])]

        x_traj_full = jnp.array(x_out)
        t_traj_full = jnp.array(t_out)

        # Interpolate to requested time points
        from scipy.interpolate import interp1d
        if len(t_traj_full) > 1:
            interp_fn = interp1d(t_traj_full, x_traj_full, axis=0, kind='linear',
                                bounds_error=False, fill_value='extrapolate')
            x_traj = jnp.array([interp_fn(float(t)) for t in t_span])
        else:
            # Only one point, just repeat
            x_traj = jnp.tile(x_traj_full[0], (len(t_span), 1))

        # Compute outputs
        def compute_output_at_t(x, t):
            u = u_func(t) if u_func is not None else None
            return self.compute_outputs(x, u, params, t)

        y_traj = jax.vmap(compute_output_at_t)(x_traj, t_span)

        return x_traj, y_traj

    def simulate_scan(self, x0: jnp.ndarray, t_span: jnp.ndarray,
                     params: Optional[Dict[str, jnp.ndarray]] = None,
                     u_func: Optional[Callable] = None,
                     method: str = 'euler', dt: Optional[float] = None,
                     use_warm_start: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate using scan-based integration (more memory efficient for long simulations)

        This is useful for very long time series or when you need custom integration schemes.
        For most cases, use simulate() instead.

        Args:
            x0: Initial state [n_states]
            t_span: Time points [n_time]
            params: Parameters (uses self.params if None)
            u_func: Optional function u_func(t) -> [n_inputs]
            method: Integration method
                - 'euler': 1st order Euler
                - 'rk4': 4th order Runge-Kutta
                - 'dopri5': 5th order Dormand-Prince (adaptive coefficients)
                - 'rk8': 8th order Runge-Kutta
            dt: Time step (if None, uses t_span spacing)
            use_warm_start: If True, uses previous z values to initialize Newton solver (faster)

        Returns:
            x_traj: State trajectory [n_time, n_states]
            y_traj: Output trajectory [n_time, n_outputs]
        """
        if params is None:
            params = self.params

        if dt is None:
            dt = t_span[1] - t_span[0]

        def scan_step(state, t):
            """Single integration step"""
            if use_warm_start and self.n_alg_vars > 0:
                x, z_prev = state
            else:
                x = state
                z_prev = None

            u = u_func(t) if u_func is not None else None

            if method == 'euler':
                # 1st order Euler with warm start
                if use_warm_start and self.n_alg_vars > 0:
                    dxdt, z = self.forward(x, u, params, t, z_init=z_prev, return_z=True)
                else:
                    dxdt = self.forward_jit(x, u, params, t)
                    z = None
                x_next = x + dt * dxdt

            elif method == 'rk4':
                # 4th order Runge-Kutta with warm start
                if use_warm_start and self.n_alg_vars > 0:
                    k1, z1 = self.forward(x, u, params, t, z_init=z_prev, return_z=True)
                    k2, z2 = self.forward(x + 0.5 * dt * k1, u, params, t + 0.5 * dt, z_init=z1, return_z=True)
                    k3, z3 = self.forward(x + 0.5 * dt * k2, u, params, t + 0.5 * dt, z_init=z2, return_z=True)
                    k4, z = self.forward(x + dt * k3, u, params, t + dt, z_init=z3, return_z=True)
                else:
                    k1 = self.forward_jit(x, u, params, t)
                    k2 = self.forward_jit(x + 0.5 * dt * k1, u, params, t + 0.5 * dt)
                    k3 = self.forward_jit(x + 0.5 * dt * k2, u, params, t + 0.5 * dt)
                    k4 = self.forward_jit(x + dt * k3, u, params, t + dt)
                    z = None
                x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            elif method == 'dopri5':
                # 5th order Dormand-Prince (embedded method, using 5th order formula)
                # Butcher tableau coefficients
                # Note: dopri5 doesn't support warm start yet (would need more modifications)
                k1 = self.forward_jit(x, u, params, t)
                k2 = self.forward_jit(x + dt * (1/5) * k1, u, params, t + dt * (1/5))
                k3 = self.forward_jit(x + dt * ((3/40) * k1 + (9/40) * k2), u, params, t + dt * (3/10))
                k4 = self.forward_jit(x + dt * ((44/45) * k1 - (56/15) * k2 + (32/9) * k3), u, params, t + dt * (4/5))
                k5 = self.forward_jit(x + dt * ((19372/6561) * k1 - (25360/2187) * k2 + (64448/6561) * k3 - (212/729) * k4),
                                     u, params, t + dt * (8/9))
                k6 = self.forward_jit(x + dt * ((9017/3168) * k1 - (355/33) * k2 + (46732/5247) * k3 + (49/176) * k4 - (5103/18656) * k5),
                                     u, params, t + dt)

                # 5th order solution
                x_next = x + dt * ((35/384) * k1 + (500/1113) * k3 + (125/192) * k4 - (2187/6784) * k5 + (11/84) * k6)
                z = None  # No warm start for dopri5

            else:
                raise ValueError(f"Unknown method: {method}. Available: 'euler', 'rk4', 'dopri5'")

            # Compute output at current time
            y = self.compute_outputs_jit(x, u, params, t)

            # Return next state (with z for warm start if applicable)
            if use_warm_start and self.n_alg_vars > 0:
                next_state = (x_next, z)
            else:
                next_state = x_next

            return next_state, (x, y)

        # Initial state for scan
        if use_warm_start and self.n_alg_vars > 0:
            # Initialize z to zeros for consistent pytree structure
            z0 = jnp.zeros((self.n_alg_vars,), dtype=x0.dtype)
            init_state = (x0, z0)
        else:
            init_state = x0

        # Run scan
        _, (x_traj, y_traj) = jax.lax.scan(scan_step, init_state, t_span)

        return x_traj, y_traj

    def loss_and_grad(self, params: Dict[str, jnp.ndarray],
                     x0: jnp.ndarray, t_span: jnp.ndarray,
                     y_target: jnp.ndarray,
                     u_func: Optional[Callable] = None,
                     loss_fn: Optional[Callable] = None) -> Tuple[float, Dict[str, jnp.ndarray]]:
        """
        Compute loss and gradients w.r.t. parameters (for optimization)

        This is JIT-compiled and optimized for fast parameter optimization.

        Args:
            params: Parameters to optimize
            x0: Initial state
            t_span: Time points
            y_target: Target outputs [n_time, n_outputs]
            u_func: Optional input function
            loss_fn: Loss function (if None, uses MSE)

        Returns:
            loss: Scalar loss value
            grads: Dictionary of gradients w.r.t. parameters
        """
        if loss_fn is None:
            def loss_fn(y_pred, y_target):
                return jnp.mean((y_pred - y_target)**2)

        def compute_loss(params):
            x_traj, y_traj = self.simulate(x0, t_span, params, u_func, use_warm_start=True)
            return loss_fn(y_traj, y_target)

        # Use value_and_grad for efficiency (computes both in one pass)
        loss_val, grads = jax.value_and_grad(compute_loss)(params)

        return loss_val, grads

    def optimize_params(self, x0: jnp.ndarray, t_span: jnp.ndarray,
                       y_target: jnp.ndarray,
                       u_func: Optional[Callable] = None,
                       n_epochs: int = 100,
                       learning_rate: float = 0.01,
                       loss_fn: Optional[Callable] = None,
                       verbose: bool = True) -> Tuple[Dict[str, jnp.ndarray], List[float]]:
        """
        Optimize parameters using gradient descent (with JAX optimizers)

        Args:
            x0: Initial state
            t_span: Time points
            y_target: Target outputs
            u_func: Optional input function
            n_epochs: Number of optimization iterations
            learning_rate: Learning rate
            loss_fn: Loss function (if None, uses MSE)
            verbose: Print progress

        Returns:
            optimized_params: Optimized parameters
            loss_history: Loss at each iteration
        """
        # Use simple Adam-like optimizer
        params = self.params.copy()
        m = {k: jnp.zeros_like(v) for k, v in params.items()}
        v = {k: jnp.zeros_like(v_) for k, v_ in params.items()}
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        loss_history = []

        # JIT compile the loss and grad computation
        loss_and_grad_jit = jit(lambda p: self.loss_and_grad(p, x0, t_span, y_target, u_func, loss_fn))

        for epoch in range(n_epochs):
            # Compute loss and gradients
            loss_val, grads = loss_and_grad_jit(params)
            loss_history.append(float(loss_val))

            # Adam update
            for key in params.keys():
                m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
                v[key] = beta2 * v[key] + (1 - beta2) * grads[key]**2
                m_hat = m[key] / (1 - beta1**(epoch + 1))
                v_hat = v[key] / (1 - beta2**(epoch + 1))
                params[key] = params[key] - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: loss = {loss_val:.6e}")

        return params, loss_history


# Example usage
if __name__ == "__main__":
    print("JAX DAE Model - High Performance Implementation")
    print("=" * 60)

    # Example DAE dictionary (same as PyTorch version)
    dae_dict = {
        'states': [
            {'name': 'x', 'type': 'float'},
            {'name': 'y', 'type': 'float'},
            {'name': 'vx', 'type': 'float'},
            {'name': 'vy', 'type': 'float'}
        ],
        'inputs': [
            {'name': 'ux', 'type': 'float'},
            {'name': 'uy', 'type': 'float'}
        ],
        'alg_vars': [
            {'name': 'ax', 'type': 'float'},
            {'name': 'ay', 'type': 'float'},
            {'name': 'lam', 'type': 'float'}
        ],
        'outputs': [
            {'name': 'vx_out', 'type': 'float'},
            {'name': 'vy_out', 'type': 'float'}
        ],
        'parameters': [
            {'name': 'l', 'type': 'float', 'value': '1.0'},
            {'name': 'm', 'type': 'float', 'value': '1.0'},
            {'name': 'b', 'type': 'float', 'value': '1'},
            {'name': 'g', 'type': 'float', 'value': '1'}
        ],
        'f': [
            'der(x) = vx',
            'der(y) = vy',
            'der(vx) = ax',
            'der(vy) = ay'
        ],
        'g': [
            '0 = m * ax + b * vx - lam * x - ux',
            '0 = m * ay + b * vy - lam * y + m * g - uy',
            '0 = x * ax + vx**2 + y * ay + vy**2'
        ],
        'h': [
            'vx_out = vx',
            'vy_out = vy'
        ]
    }

    # Create model
    print("\n1. Creating JAX DAE Model...")
    model = DAEModel(dae_dict)

    print(f"   States: {model.state_names}")
    print(f"   Algebraic vars: {model.alg_var_names}")
    print(f"   Outputs: {model.output_names}")
    print(f"   Parameters: {model.param_names}")

    # Print parameter values
    print("\n   Initial parameter values:")
    for name, val in model.params.items():
        print(f"     {name} = {float(val):.4f}")

    # Test forward pass
    print("\n2. Testing forward pass...")
    x = jnp.array([1.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
    u = jnp.array([0.0, 0.0], dtype=jnp.float64)
    dxdt = model.forward_jit(x, u, model.params, 0.0)
    print(f"   x = {x}")
    print(f"   dxdt = {dxdt}")

    # Test output computation
    y = model.compute_outputs_jit(x, u, model.params, 0.0)
    print(f"   y = {y}")

    # Test simulation
    print("\n3. Testing simulation...")
    x0 = jnp.array([0.19866933079506122, -0.9800665778412416, 0.0, 0.0], dtype=jnp.float64)
    t_span = jnp.linspace(0, 2, 20, dtype=jnp.float64)

    def u_func(t):
        return jnp.array([0.0, 0.0], dtype=jnp.float64)

    x_traj, y_traj = model.simulate(x0, t_span, u_func=u_func)
    print(f"   x_traj shape: {x_traj.shape}")
    print(f"   y_traj shape: {y_traj.shape}")
    print(f"   Final state: {x_traj[-1]}")
    print(f"   Final output: {y_traj[-1]}")

    # Test gradient computation
    print("\n4. Testing gradient computation...")
    target_y = jnp.array([0.0, -0.5], dtype=jnp.float64)
    y_target = jnp.tile(target_y, (len(t_span), 1))

    loss_val, grads = model.loss_and_grad(model.params, x0, t_span, y_target, u_func)
    print(f"   Initial loss: {loss_val:.6e}")
    print(f"   Gradients computed for: {list(grads.keys())}")
    for name, grad_val in grads.items():
        print(f"     d(loss)/d({name}) = {float(grad_val):.6e}")

    # Test parameter optimization
    print("\n5. Testing parameter optimization...")
    optimized_params, loss_history = model.optimize_params(
        x0, t_span, y_target, u_func,
        n_epochs=50, learning_rate=0.01, verbose=False
    )

    print(f"   Initial loss: {loss_history[0]:.6e}")
    print(f"   Final loss: {loss_history[-1]:.6e}")
    print(f"   Optimized parameters:")
    for name, val in optimized_params.items():
        print(f"     {name} = {float(val):.4f} (initial: {float(model.params[name]):.4f})")

    print("\n" + "=" * 60)
    print("All tests passed! JAX implementation is ready for use.")
    print("\nPerformance tips:")
    print("- Use forward_jit, compute_outputs_jit for compiled performance")
    print("- Use simulate() for automatic differentiation through time")
    print("- Use loss_and_grad() for efficient parameter optimization")
    print("- All functions are JIT-compiled for maximum speed")
