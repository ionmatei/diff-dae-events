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

import ast
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


# =============================================================================
# AST-based Expression Compiler for JAX-traceable functions
# =============================================================================

# Allowed JAX math functions mapping
_JAX_MATH_FUNCS = {
    'exp': 'jnp.exp', 'log': 'jnp.log', 'log10': 'jnp.log10',
    'sqrt': 'jnp.sqrt', 'abs': 'jnp.abs',
    'sin': 'jnp.sin', 'cos': 'jnp.cos', 'tan': 'jnp.tan',
    'asin': 'jnp.arcsin', 'acos': 'jnp.arccos', 'atan': 'jnp.arctan',
    'sinh': 'jnp.sinh', 'cosh': 'jnp.cosh', 'tanh': 'jnp.tanh',
    'arcsin': 'jnp.arcsin', 'arccos': 'jnp.arccos', 'arctan': 'jnp.arctan',
    'pow': 'jnp.power', 'sign': 'jnp.sign', 'floor': 'jnp.floor',
    'ceil': 'jnp.ceil', 'min': 'jnp.minimum', 'max': 'jnp.maximum',
}


class _ExpressionTransformer(ast.NodeTransformer):
    """
    AST transformer that converts expression strings into JAX-compatible code.

    Transforms variable names to array indexing:
        x -> x_arr[idx]
        z -> z_arr[idx]
        param -> p_arr[idx]
        t/time -> t

    Transforms function calls to JAX equivalents:
        sin(x) -> jnp.sin(x)
        exp(x) -> jnp.exp(x)
    """

    def __init__(self, state_names: List[str], alg_var_names: List[str],
                 param_names: List[str]):
        self.state_idx = {name: i for i, name in enumerate(state_names)}
        self.alg_idx = {name: i for i, name in enumerate(alg_var_names)}
        self.param_idx = {name: i for i, name in enumerate(param_names)}

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Transform variable names to array indexing."""
        name = node.id

        # Time variable
        if name in ('t', 'time'):
            return ast.Name(id='t', ctx=ast.Load())

        # State variable: x -> x_arr[idx]
        if name in self.state_idx:
            return ast.Subscript(
                value=ast.Name(id='x', ctx=ast.Load()),
                slice=ast.Constant(value=self.state_idx[name]),
                ctx=node.ctx
            )

        # Algebraic variable: z -> z_arr[idx]
        if name in self.alg_idx:
            return ast.Subscript(
                value=ast.Name(id='z', ctx=ast.Load()),
                slice=ast.Constant(value=self.alg_idx[name]),
                ctx=node.ctx
            )

        # Parameter: p -> p_arr[idx]
        if name in self.param_idx:
            return ast.Subscript(
                value=ast.Name(id='p', ctx=ast.Load()),
                slice=ast.Constant(value=self.param_idx[name]),
                ctx=node.ctx
            )

        # Unknown name - keep as is (might be a constant like pi)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Transform function calls to JAX equivalents."""
        # First transform arguments
        node = self.generic_visit(node)

        # Check if it's a known math function
        if isinstance(node.func, ast.Name) and node.func.id in _JAX_MATH_FUNCS:
            jax_func = _JAX_MATH_FUNCS[node.func.id]
            # Parse the JAX function reference (e.g., 'jnp.sin' -> Attribute)
            parts = jax_func.split('.')
            func_node = ast.Name(id=parts[0], ctx=ast.Load())
            for part in parts[1:]:
                func_node = ast.Attribute(value=func_node, attr=part, ctx=ast.Load())
            node.func = func_node

        return node


def _transform_expr_to_jax(expr_str: str, state_names: List[str],
                           alg_var_names: List[str], param_names: List[str]) -> str:
    """
    Transform an expression string into JAX-compatible Python code string.

    Returns the transformed expression as a string (not a function).
    """
    # Preprocess: replace ^ with ** and 'time' with 't'
    expr_str = expr_str.replace('^', '**')
    expr_str = re.sub(r'\btime\b', 't', expr_str)

    # Parse into AST
    try:
        tree = ast.parse(expr_str, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {expr_str}") from e

    # Transform AST
    transformer = _ExpressionTransformer(state_names, alg_var_names, param_names)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    # Convert back to source code
    return ast.unparse(tree)


def compile_expression(expr_str: str, state_names: List[str],
                       alg_var_names: List[str], param_names: List[str]) -> Callable:
    """
    Compile an expression string into a JAX-traceable function.

    Uses exec to create a real Python function, avoiding eval() overhead at runtime.
    """
    transformed = _transform_expr_to_jax(expr_str, state_names, alg_var_names, param_names)

    # Create a real function using exec (no eval overhead at runtime)
    func_code = f"def _expr_func(t, x, z, p):\n    return {transformed}"
    local_ns = {}
    exec(func_code, {'jnp': jnp}, local_ns)

    return local_ns['_expr_func']


def compile_expression_vectorized(expr_strs: List[str], state_names: List[str],
                                   alg_var_names: List[str], param_names: List[str]) -> Callable:
    """
    Compile multiple expressions into a single JAX-traceable function returning a vector.

    Creates a single fused function to avoid function call overhead.
    """
    if not expr_strs:
        def empty_func(t, x, z, p):
            return jnp.zeros(0, dtype=jnp.float64)
        return empty_func

    # Transform all expressions
    transformed_exprs = [
        _transform_expr_to_jax(expr, state_names, alg_var_names, param_names)
        for expr in expr_strs
    ]

    # Build a single fused function that computes all expressions at once
    # This avoids function call overhead and list comprehension overhead
    expr_list = ', '.join(transformed_exprs)
    func_code = f"def _combined_func(t, x, z, p):\n    return jnp.array([{expr_list}], dtype=jnp.float64)"

    local_ns = {}
    exec(func_code, {'jnp': jnp}, local_ns)

    return local_ns['_combined_func']


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

        # Parse and compile equations into JAX-traceable functions
        self.f_eqs, self._eval_f = self._parse_f_equations()
        self.g_eqs, self._eval_g = self._parse_g_equations()
        self.h_eqs, self._eval_h = self._parse_h_equations()

        self.n_outputs = len(self.h_eqs) if self.h_eqs else self.n_states

        # Build functions
        self._build_jit_functions()

        # History
        self.history = {
            'loss': [], 'gradient_norm': [], 'params': [],
            'params_all': [], 'step_size': [], 'time_per_iter': []
        }
        
        # Algorithm configuration (SGD or ADAM)
        self.algorithm_type = None
        self.algorithm_params = None
        
        # Adam state
        self.adam_m = None
        self.adam_v = None
        self.adam_t = 0

        print(f"\nDAEOptimizerDEERMethods initialized:")
        print(f"  Method: {method}")
        print(f"  DEER max iter: {deer_max_iter}")
        print(f"  Parameters to optimize: {self.n_params_opt}")
        print(f"  States: {self.n_states}, Algebraic: {self.n_alg}")

    def _parse_f_equations(self) -> Tuple[List[Tuple[str, str]], Callable]:
        """
        Parse f equations and compile into a single JAX-traceable function.

        Returns:
            Tuple of (equations list for reference, compiled eval_f function)
        """
        equations = []
        expr_strs = []
        state_indices = []

        for eq_str in self.dae_data.get('f', []) or []:
            match = re.match(r'der\((\w+)\)\s*=\s*(.+)', eq_str.strip())
            if match:
                state_name = match.group(1)
                expr = match.group(2).strip()
                equations.append((state_name, expr))
                expr_strs.append(expr)
                state_indices.append(self.state_names.index(state_name))

        # Compile all expressions into individual functions
        compiled_exprs = [
            compile_expression(expr, self.state_names, self.alg_var_names, self.param_names)
            for expr in expr_strs
        ]

        # Pre-compute for closure
        n_states = self.n_states

        def eval_f(t, x, z, p):
            """Compiled f evaluation - JAX traceable."""
            dxdt = jnp.zeros(n_states, dtype=jnp.float64)
            for idx, func in zip(state_indices, compiled_exprs):
                dxdt = dxdt.at[idx].set(func(t, x, z, p))
            return dxdt

        return equations, eval_f

    def _parse_g_equations(self) -> Tuple[List[str], Callable]:
        """
        Parse g equations and compile into a single JAX-traceable function.

        Returns:
            Tuple of (equations list for reference, compiled eval_g function)
        """
        equations = []
        expr_strs = []

        for eq_str in self.dae_data.get('g', []) or []:
            match = re.match(r'0(?:\.0*)?\s*=\s*(.+)', eq_str.strip())
            if match:
                expr = match.group(1).strip()
                equations.append(expr)
                expr_strs.append(expr)

        if not expr_strs:
            def eval_g(t, x, z, p):
                return jnp.zeros(0, dtype=jnp.float64)
            return equations, eval_g

        # Compile vectorized function
        eval_g = compile_expression_vectorized(
            expr_strs, self.state_names, self.alg_var_names, self.param_names
        )

        return equations, eval_g

    def _parse_h_equations(self) -> Tuple[List[Tuple[str, str]], Callable]:
        """
        Parse h (output) equations and compile into a single JAX-traceable function.

        Returns:
            Tuple of (equations list for reference, compiled eval_h function)
        """
        equations = []
        expr_strs = []
        n_states = self.n_states

        for eq_str in self.dae_data.get('h', []) or []:
            stripped = eq_str.strip()
            # Check if it's just a variable name
            if stripped in self.state_names:
                equations.append(('output', stripped))
                expr_strs.append(stripped)
            elif stripped in self.alg_var_names:
                equations.append(('output', stripped))
                expr_strs.append(stripped)
            else:
                match = re.match(r'(\w+)\s*=\s*(.+)', stripped)
                if match:
                    equations.append((match.group(1), match.group(2).strip()))
                    expr_strs.append(match.group(2).strip())
                else:
                    equations.append(('output', stripped))
                    expr_strs.append(stripped)

        if not expr_strs:
            # Default: output all states
            def eval_h(t, x, z, p):
                return x
            return equations, eval_h

        # Compile vectorized function
        eval_h = compile_expression_vectorized(
            expr_strs, self.state_names, self.alg_var_names, self.param_names
        )

        return equations, eval_h

    def _build_jit_functions(self):
        """Build DEER-based simulation and optimization functions."""

        # Use pre-compiled JAX-traceable functions from parsing stage
        eval_f = self._eval_f
        eval_g = self._eval_g
        eval_h = self._eval_h

        n_states = self.n_states
        n_alg = self.n_alg
        n_total = self.n_total
        method = self.method

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
            # Generic BDF handler for orders 2-6 using COMPANION MATRIX construction
            # This transforms the q-step recurrence into a first-order block system
            # suitable for parallel associative scan via matmul_recursive
            #
            # OPTIMIZATION: Exploit block structure of augmented Jacobians.
            # The augmented residual R_aug = [R_phys; R_hist] has Jacobian M0 = dR/dY_i:
            #   M0 = [B0, B1, B2, ..., B_{q-1}]   (physics depends on all of Y_i)
            #        [ I,  0,  0, ...,  0     ]   (shift constraints are identity)
            #        [ 0,  I,  0, ...,  0     ]
            #        ...
            # Note: B1..B_{q-1} are generally nonzero (BDF derivative uses history).
            # But the lower rows are pure identity shifts, enabling block elimination:
            # solve bottom trivially (h_i = shift), substitute into top, solve ny×ny.
            bdf_order = int(method[3])
            coeffs, _ = BDF_COEFFICIENTS[bdf_order]
            coeffs = jnp.array(coeffs, dtype=jnp.float64)
            n_history = bdf_order  # Number of past values needed

            def implicit_residual_bdf(y_shifts, t_i, dt, p, step_idx):
                """
                Generic BDF residual.
                y_shifts: list of [y_i, y_{i-1}, y_{i-2}, ...] up to order terms extracted from augmented state
                For early steps, falls back to lower-order BDF.
                """
                y_i = y_shifts[0]
                x_i = y_i[:n_states]
                z_i = y_i[n_states:]

                f_i = eval_f(t_i, x_i, z_i, p)
                g_i = eval_g(t_i, x_i, z_i, p)

                # Compute derivative approximation based on available history
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
                dxdt = compute_bdf_deriv(1)  # Default BDF1
                for k in range(2, bdf_order + 1):
                    dxdt = jnp.where(step_idx >= k, compute_bdf_deriv(k), dxdt)

                R_x = dxdt - f_i
                return jnp.concatenate([R_x, g_i])

            def deer_func(yshifts, xinput, params):
                """
                DEER function for BDF with companion matrix.

                yshifts: [Y_i, Y_{i-1}] where each Y is augmented state of size (q*n_y)
                         Y_i = [y_i; y_{i-1}; ...; y_{i-q+1}]

                Returns: Augmented residual of size (q*n_y) matching the augmented state.
                """
                dt, t_i, step_idx = xinput
                Y_i, Y_im1 = yshifts  # Each is augmented state of size (q*n_y)

                ny = n_states + n_alg

                # Extract individual history terms from augmented states
                y_history = []
                for k in range(n_history):
                    y_k = Y_i[k*ny:(k+1)*ny]
                    y_history.append(y_k)
                # Get the last term y_{i-q} from Y_{i-1}
                y_iq = Y_im1[-ny:]
                y_history.append(y_iq)

                # Compute physical residual for y_i
                R_phys = implicit_residual_bdf(y_history, t_i, dt, params, step_idx)

                # Compute history consistency residuals
                R_hist = []
                for k in range(n_history - 1):
                    y_from_Yi = Y_i[(k+1)*ny:(k+2)*ny]
                    y_from_Yim1 = Y_im1[k*ny:(k+1)*ny]
                    R_hist.append(y_from_Yi - y_from_Yim1)

                return jnp.concatenate([R_phys] + R_hist)

            def shifter_func(y, _):
                """Shifter for companion matrix approach."""
                y_im1 = jnp.concatenate([y[:1], y[:-1]], axis=0)
                return [y, y_im1]

            p_num = 2  # Companion form is always 2-term recurrence

            def solve_inv_lin(jacs, z, inv_lin_params):
                """
                Solve BDF linear system using block elimination + parallel scan.

                The augmented system M0 @ Y_i + M1 @ Y_{i-1} = z has block structure:

                M0 = -dR/dY_i (NEGATED Jacobian from DEER):
                     [B0, B1, ..., B_{q-1}]   <- first ny rows (physics)
                     [-I,  0,  ..., 0     ]   <- shift constraints (negated)
                     [0,  -I,  ..., 0     ]
                     ...

                M1 = -dR/dY_{i-1} (NEGATED Jacobian from DEER):
                     [C0, C1, ..., C_{q-1}]   <- physics coupling to Y_{i-1}
                     [+I, 0,  ..., 0      ]   <- shift: from -(-I)
                     [0, +I,  ..., 0      ]
                     ...

                Partition Y_i = [y_i; h_i] where h_i has (q-1)*ny components.

                Bottom equations give: h_i = S @ Y_{i-1} + r_i  (shift + RHS)
                Substitute into top: B0 @ y_i = z_top - [B1..B_{q-1}] @ h_i - M1_top @ Y_{i-1}

                This reduces each timestep to one ny×ny solve instead of (q*ny)×(q*ny).
                """
                y0, = inv_lin_params
                nsamples = z.shape[0]
                ny = y0.shape[0]
                q = n_history
                ny_aug = q * ny
                ny_hist = (q - 1) * ny  # Size of history part h_i

                M0, M1 = jacs  # M0, M1: (nsamples, q*ny, q*ny)

                # Build initial augmented state: Y0 = [y0; y0; ...; y0]
                Y0 = jnp.tile(y0, q)  # (q*ny,)

                # Extract block structure from M0 and M1 (first row of blocks)
                # B0 = M0[:, :ny, :ny]           - Jacobian of physics w.r.t. y_i
                # Bh = M0[:, :ny, ny:]           - Jacobian of physics w.r.t. h_i (history in Y_i)
                # M1_top = M1[:, :ny, :]         - Jacobian of physics w.r.t. Y_{i-1}

                # The bottom (q-1)*ny rows encode shift constraints:
                # h_i[k] - Y_{i-1}[k] = z_bottom[k]  =>  h_i = Y_{i-1}[:ny_hist] + z_bottom
                # In terms of full Y_{i-1}: h_i = S_select @ Y_{i-1} + z_bottom
                # where S_select picks the first (q-1) blocks of Y_{i-1}

                # Build shift/selection matrix for history: picks first (q-1)*ny of Y_{i-1}
                S_hist = jnp.zeros((ny_hist, ny_aug), dtype=jnp.float64)
                S_hist = S_hist.at[:ny_hist, :ny_hist].set(jnp.eye(ny_hist))

                def compute_Ab(M0_i, M1_i, z_i):
                    """
                    Compute A, b for Y_i = A @ Y_{i-1} + b using block elimination.
                    """
                    # Extract blocks
                    B0 = M0_i[:ny, :ny]              # (ny, ny)
                    Bh = M0_i[:ny, ny:]              # (ny, ny_hist)
                    M1_top = M1_i[:ny, :]            # (ny, q*ny)
                    z_top = z_i[:ny]                 # (ny,)
                    z_bottom = z_i[ny:]              # (ny_hist,)

                    # DEER passes NEGATED Jacobians: M0 = -dR/dY_i, M1 = -dR/dY_{i-1}
                    #
                    # For R_hist[k] = Y_i[(k+1)*ny:...] - Y_{i-1}[k*ny:...]:
                    #   dR_hist/dY_i has +I => M0_bottom has -I
                    #   dR_hist/dY_{i-1} has -I => M1_bottom has +I
                    #
                    # Bottom equations: (-I) @ h_i + (+I) @ Y_{i-1}[:ny_hist] = z_bottom
                    # => h_i = Y_{i-1}[:ny_hist] - z_bottom = S_hist @ Y_{i-1} - z_bottom

                    # Top equation: B0 @ y_i + Bh @ h_i + M1_top @ Y_{i-1} = z_top
                    # Substitute h_i = S_hist @ Y_{i-1} - z_bottom:
                    # B0 @ y_i + Bh @ (S_hist @ Y_{i-1} - z_bottom) + M1_top @ Y_{i-1} = z_top
                    # B0 @ y_i = z_top + Bh @ z_bottom - (Bh @ S_hist + M1_top) @ Y_{i-1}
                    #
                    # Let: rhs_y = z_top + Bh @ z_bottom
                    #      C_y = Bh @ S_hist + M1_top
                    # Then: y_i = B0^{-1} @ rhs_y - B0^{-1} @ C_y @ Y_{i-1}

                    rhs_y = z_top + Bh @ z_bottom                    # (ny,)
                    C_y = Bh @ S_hist + M1_top                       # (ny, q*ny)

                    # Solve the ny×ny system
                    B0_inv_rhs = jnp.linalg.solve(B0, rhs_y)         # (ny,)
                    B0_inv_Cy = jnp.linalg.solve(B0, C_y)            # (ny, q*ny)

                    # Now construct full A and b for Y_i = [y_i; h_i]
                    # y_i = -B0_inv_Cy @ Y_{i-1} + B0_inv_rhs
                    # h_i = S_hist @ Y_{i-1} - z_bottom
                    #
                    # So: A = [-B0_inv_Cy]    b = [B0_inv_rhs ]
                    #         [S_hist    ]        [-z_bottom  ]

                    A = jnp.zeros((ny_aug, ny_aug), dtype=jnp.float64)
                    A = A.at[:ny, :].set(-B0_inv_Cy)
                    A = A.at[ny:, :].set(S_hist)

                    b = jnp.zeros(ny_aug, dtype=jnp.float64)
                    b = b.at[:ny].set(B0_inv_rhs)
                    b = b.at[ny:].set(-z_bottom)

                    return A, b

                # Compute A, b for timesteps 1 to nsamples-1
                A_all, b_all = vmap(compute_Ab)(M0[1:], M1[1:], z[1:])

                # Parallel scan
                Y_result = matmul_recursive(A_all, b_all, Y0)

                # Prepend Y0
                Y_full = jnp.concatenate([Y0[None, :], Y_result], axis=0)

                return Y_full[:nsamples]

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

            # Prepare time inputs
            dt_partial = t_array[1:] - t_array[:-1]
            dt = jnp.concatenate([dt_partial[:1], dt_partial], axis=0)

            if method == 'backward_euler':
                # Initial guess: constant y0
                yinit_guess = jnp.zeros((nsamples, ny), dtype=jnp.float64) + y0
                xinput = (dt, t_array)
                inv_lin_params = (y0,)
            elif method == 'trapezoidal':
                # Initial guess: constant y0
                yinit_guess = jnp.zeros((nsamples, ny), dtype=jnp.float64) + y0
                t_im1 = jnp.concatenate([t_array[:1], t_array[:-1]], axis=0)
                xinput = (dt, t_array, t_im1)
                inv_lin_params = (y0,)
            elif method.startswith('bdf'):
                # For BDF with companion matrix, we work with AUGMENTED state
                # yinit_guess has shape (nsamples, q*ny) where q is BDF order
                ny_aug = n_history * ny
                yinit_guess = jnp.zeros((nsamples, ny_aug), dtype=jnp.float64)
                # Initialize with y0 replicated: [y0; y0; ...; y0]
                for k in range(n_history):
                    yinit_guess = yinit_guess.at[:, k*ny:(k+1)*ny].set(y0)

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

            # Extract physical state from result
            # For BDF, result.value is augmented state (nsamples, q*ny)
            # We need to extract just y_i from each augmented state
            if method.startswith('bdf'):
                y_aug = result.value  # (nsamples, q*ny)
                return y_aug[:, :ny]  # Extract first ny components
            else:
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

    def _adam_update_step(self, p, grad, m, v, t, beta1, beta2, epsilon, step_size):
        """Adam optimizer update step."""
        t_new = t + 1
        m_new = beta1 * m + (1 - beta1) * grad
        v_new = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m_new / (1 - beta1 ** t_new)
        v_hat = v_new / (1 - beta2 ** t_new)
        p_new = p - step_size * m_hat / (jnp.sqrt(v_hat) + epsilon)
        return p_new, m_new, v_new, t_new


    def optimize(
        self,
        t_array: np.ndarray,
        y_target: np.ndarray,
        p_init: Optional[np.ndarray] = None,
        n_iterations: int = 100,
        step_size: float = 0.01,
        tol: float = 1e-6,
        verbose: bool = True,
        algorithm_config: Optional[Dict] = None
    ) -> Dict:
        """Optimize parameters."""
        # Parse algorithm configuration
        if algorithm_config is None:
            self.algorithm_type = 'SGD'
            self.algorithm_params = {'step_size': step_size}
        else:
            self.algorithm_type = algorithm_config.get('type', 'SGD').upper()
            self.algorithm_params = algorithm_config.get('params', {})
            if 'step_size' not in self.algorithm_params:
                self.algorithm_params['step_size'] = step_size
        
        algo_step_size = self.algorithm_params.get('step_size', step_size)
        
        if verbose:
            print("\n" + "=" * 80)
            print(f"DEER Optimization ({self.method})")
            print("=" * 80)
            print(f"  Algorithm: {self.algorithm_type}")
            if self.algorithm_type == 'ADAM':
                print(f"  Beta1: {self.algorithm_params.get('beta1', 0.9)}")
                print(f"  Beta2: {self.algorithm_params.get('beta2', 0.999)}")
                print(f"  Epsilon: {self.algorithm_params.get('epsilon', 1e-8)}")

        if p_init is not None:
            p = jnp.array(p_init, dtype=jnp.float64)
        else:
            p = jnp.array([self.p_all[i] for i in self.optimize_indices], dtype=jnp.float64)
        
        # Initialize Adam state if using Adam
        if self.algorithm_type == 'ADAM':
            self.adam_m = jnp.zeros_like(p)
            self.adam_v = jnp.zeros_like(p)
            self.adam_t = 0

        self.history = {k: [] for k in self.history}
        converged = False

        y_target_use = np.array(y_target)
        if y_target_use.shape[0] != len(t_array):
            y_target_use = y_target_use.T

        for it in range(n_iterations):
            t_start = time.time()
            
            # Compute gradient
            _, loss, grad_p = self.optimization_step(t_array, y_target_use, np.array(p), algo_step_size)
            
            # Apply parameter update based on algorithm
            if self.algorithm_type == 'ADAM':
                beta1 = self.algorithm_params.get('beta1', 0.9)
                beta2 = self.algorithm_params.get('beta2', 0.999)
                epsilon = self.algorithm_params.get('epsilon', 1e-8)
                
                p_new, self.adam_m, self.adam_v, self.adam_t = self._adam_update_step(
                    p, grad_p, self.adam_m, self.adam_v, self.adam_t,
                    beta1, beta2, epsilon, algo_step_size
                )
            else:  # SGD
                p_new = p - algo_step_size * jnp.array(grad_p)
                
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
