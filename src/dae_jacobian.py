"""
DAE Jacobian Computation using JAX

Computes Jacobians of trapezoidal residual functions with respect to states
at time points, parallelized over time using JAX vmap.

For a trapezoidal residual r(y_k, y_{k+1}), we compute:
  - dr/dy_k: Jacobian with respect to previous state
  - dr/dy_{k+1}: Jacobian with respect to current state

These form the block-diagonal and block-superdiagonal of the full Jacobian matrix.
"""

import numpy as np
from typing import Dict, List, Tuple
import re

try:
    import jax
    import jax.numpy as jnp
    from jax import vmap, jacfwd, jacrev, jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np
    print("Error: JAX is required for Jacobian computation")
    raise


class DAEJacobian:
    """
    Computes Jacobians of DAE residual functions using JAX.

    The trapezoidal residual for interval [k, k+1] is:
        r(y_k, y_{k+1}) = [
            (x_{k+1} - x_k) / h - 0.5 * (f(t_k, y_k) + f(t_{k+1}, y_{k+1})),
            g(t_{k+1}, y_{k+1})
        ]

    We compute:
        - J_k = dr/dy_k: Jacobian with respect to previous state
        - J_{k+1} = dr/dy_{k+1}: Jacobian with respect to current state
    """

    def __init__(self, dae_data: dict):
        """
        Initialize from DAE specification.

        Args:
            dae_data: Dictionary containing DAE specification with keys:
                - states: list of differential state variables
                - alg_vars: list of algebraic variables
                - parameters: list of parameters
                - f: list of differential equations (dx/dt = f)
                - g: list of algebraic equations (0 = g)
        """
        # Extract variables
        self.states = dae_data['states']
        self.alg_vars = dae_data['alg_vars']
        self.parameters = dae_data['parameters']

        # Extract equations
        self.f_eqs = dae_data['f']
        self.g_eqs = dae_data['g']

        # Create name mappings
        self.state_names = [s['name'] for s in self.states]
        self.alg_names = [a['name'] for a in self.alg_vars]
        self.param_names = [p['name'] for p in self.parameters]

        # Get parameter values
        self.p = jnp.array([p['value'] for p in self.parameters])

        # Dimensions
        self.n_states = len(self.state_names)
        self.n_alg = len(self.alg_names)
        self.n_total = self.n_states + self.n_alg

        # Compile equations
        self._compile_equations()

        # Compile vectorized Jacobian functions
        self._compile_jacobian_functions()

    def _compile_equations(self):
        """Compile equation strings into Python functions."""
        # Compile f equations (derivatives)
        self.f_funcs = []
        for eq in self.f_eqs:
            match = re.match(r'der\((\w+)\)\s*=\s*(.+)', eq)
            if match:
                _, rhs = match.groups()
                self.f_funcs.append(rhs)
            else:
                raise ValueError(f"Invalid f equation format: {eq}")

        # Compile g equations (algebraic constraints)
        self.g_funcs = []
        for eq in self.g_eqs:
            if '=' in eq:
                lhs, rhs = eq.split('=', 1)
                lhs, rhs = lhs.strip(), rhs.strip()

                if lhs == '0' or lhs == '0.0':
                    expr = rhs
                elif rhs == '0' or rhs == '0.0':
                    expr = lhs
                else:
                    expr = f"({lhs}) - ({rhs})"

                self.g_funcs.append(expr)
            else:
                self.g_funcs.append(eq)

    def _create_jax_eval_namespace(self, t, x, z):
        """Create namespace for JAX equation evaluation (deprecated - use version with params)."""
        return self._create_jax_eval_namespace_with_params(t, x, z, self.p)

    def _create_jax_eval_namespace_with_params(self, t, x, z, p):
        """Create namespace for JAX equation evaluation with explicit parameters."""
        ns = {
            'sin': jnp.sin, 'cos': jnp.cos, 'tan': jnp.tan,
            'sinh': jnp.sinh, 'cosh': jnp.cosh, 'tanh': jnp.tanh,
            'exp': jnp.exp, 'log': jnp.log, 'log10': jnp.log10,
            'sqrt': jnp.sqrt, 'abs': jnp.abs, 'pow': jnp.power,
            'min': jnp.minimum, 'max': jnp.maximum,
            'time': t, 't': t,
        }

        # Add states
        for i, name in enumerate(self.state_names):
            ns[name] = x[i]

        # Add algebraic variables
        for i, name in enumerate(self.alg_names):
            ns[name] = z[i]

        # Add parameters
        for i, name in enumerate(self.param_names):
            ns[name] = p[i]

        return ns

    def eval_f_jax(self, t, x, z, p):
        """Evaluate f(t, x, z, p) using JAX."""
        ns = self._create_jax_eval_namespace_with_params(t, x, z, p)
        dxdt_list = [eval(expr, ns) for expr in self.f_funcs]
        return jnp.array(dxdt_list)

    def eval_g_jax(self, t, x, z, p):
        """Evaluate g(t, x, z, p) using JAX."""
        ns = self._create_jax_eval_namespace_with_params(t, x, z, p)
        g_list = [eval(expr, ns) for expr in self.g_funcs]
        return jnp.array(g_list)

    def residual_trapezoidal_single(self, t_k, t_kp1, y_k, y_kp1, p):
        """
        Trapezoidal residual for a single time interval [t_k, t_{k+1}].

        Args:
            t_k: time at step k
            t_kp1: time at step k+1
            y_k: combined state [x_k, z_k] at step k
            y_kp1: combined state [x_{k+1}, z_{k+1}] at step k+1
            p: parameter vector

        Returns:
            residual: residual vector for the trapezoidal scheme
        """
        h = t_kp1 - t_k

        # Split states
        x_k = y_k[:self.n_states]
        z_k = y_k[self.n_states:]
        x_kp1 = y_kp1[:self.n_states]
        z_kp1 = y_kp1[self.n_states:]

        # Evaluate f at both time points
        f_k = self.eval_f_jax(t_k, x_k, z_k, p)
        f_kp1 = self.eval_f_jax(t_kp1, x_kp1, z_kp1, p)

        # Evaluate g at time k+1
        g_kp1 = self.eval_g_jax(t_kp1, x_kp1, z_kp1, p)

        # Residual
        r_diff = (x_kp1 - x_k) / h - 0.5 * (f_k + f_kp1)
        r_alg = g_kp1

        return jnp.concatenate([r_diff, r_alg])

    def _compile_jacobian_functions(self):
        """
        Compile vectorized Jacobian functions using JAX vmap.

        Creates vmapped versions of:
            - dr/dy_k: Jacobian with respect to previous state
            - dr/dy_{k+1}: Jacobian with respect to current state
            - dr/dp: Jacobian with respect to parameters
        """
        # Single-interval Jacobian function with respect to y_k
        def jac_y_k_single(t_k, t_kp1, y_k, y_kp1, p):
            # Fix y_kp1, p and differentiate with respect to y_k
            return jacfwd(lambda yk: self.residual_trapezoidal_single(t_k, t_kp1, yk, y_kp1, p))(y_k)

        # Single-interval Jacobian function with respect to y_{k+1}
        def jac_y_kp1_single(t_k, t_kp1, y_k, y_kp1, p):
            # Fix y_k, p and differentiate with respect to y_{k+1}
            return jacfwd(lambda ykp1: self.residual_trapezoidal_single(t_k, t_kp1, y_k, ykp1, p))(y_kp1)

        # Single-interval Jacobian function with respect to parameters
        def jac_p_single(t_k, t_kp1, y_k, y_kp1, p):
            # Fix y_k, y_kp1 and differentiate with respect to p
            return jacfwd(lambda pp: self.residual_trapezoidal_single(t_k, t_kp1, y_k, y_kp1, pp))(p)

        # Vectorize over all intervals
        # in_axes=(0, 0, 0, 0, None) means vmap over first axis of times and states, broadcast p
        self._jac_y_k_vmapped = vmap(jac_y_k_single, in_axes=(0, 0, 0, 0, None))
        self._jac_y_kp1_vmapped = vmap(jac_y_kp1_single, in_axes=(0, 0, 0, 0, None))
        self._jac_p_vmapped = vmap(jac_p_single, in_axes=(0, 0, 0, 0, None))

        # Create JIT-compiled versions for better performance
        self._jac_y_k_vmapped_jit = jit(self._jac_y_k_vmapped)
        self._jac_y_kp1_vmapped_jit = jit(self._jac_y_kp1_vmapped)
        self._jac_p_vmapped_jit = jit(self._jac_p_vmapped)

        # Compile f and g Jacobian functions for analytical approach
        self._compile_f_g_jacobian_functions()

        print("JAX Jacobian vmap functions compiled successfully!")

    def _compile_f_g_jacobian_functions(self):
        """
        Compile vectorized Jacobian functions for f and g separately.

        This allows analytical construction of residual Jacobians:
        For trapezoidal residual:
            r = [(x_{k+1} - x_k)/h - 0.5*(f_k + f_{k+1}), g_{k+1}]

        The Jacobians are:
            dr/dy_k = [[-I/h - 0.5*df_k/dy_k, 0],
                       [0, 0]]

            dr/dy_{k+1} = [[I/h - 0.5*df_{k+1}/dy_{k+1}, 0],
                           [dg_{k+1}/dy_{k+1}, I]]

        where the identity matrices are EXACT (no roundoff).
        """
        # Single-point Jacobian of f with respect to y
        def jac_f_single(t, y, p):
            x = y[:self.n_states]
            z = y[self.n_states:]
            return jacfwd(lambda yy: self.eval_f_jax(t, yy[:self.n_states], yy[self.n_states:], p))(y)

        # Single-point Jacobian of g with respect to y
        def jac_g_single(t, y, p):
            x = y[:self.n_states]
            z = y[self.n_states:]
            return jacfwd(lambda yy: self.eval_g_jax(t, yy[:self.n_states], yy[self.n_states:], p))(y)

        # Vectorize over time points, broadcast p
        self._jac_f_vmapped = vmap(jac_f_single, in_axes=(0, 0, None))
        self._jac_g_vmapped = vmap(jac_g_single, in_axes=(0, 0, None))

        # JIT compile
        self._jac_f_vmapped_jit = jit(self._jac_f_vmapped)
        self._jac_g_vmapped_jit = jit(self._jac_g_vmapped)

    def compute_jacobian_blocks_analytical(self, t_array: np.ndarray, y_array: np.ndarray, p: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobian blocks analytically from f and g Jacobians.

        This method constructs the residual Jacobian blocks using:
        1. Exact identity matrices (no autodiff roundoff)
        2. Jacobians of f and g computed via vmap

        For trapezoidal residual r = [(x_{k+1} - x_k)/h - 0.5*(f_k + f_{k+1}), g_{k+1}]:

        dr/dy_k has structure:
            [[-I/h - 0.5*df_k/dy_k]_{n_states x n_total}]
            [[0]_{n_alg x n_total}]

        dr/dy_{k+1} has structure:
            [[I/h - 0.5*df_{k+1}/dy_{k+1}]_{n_states x n_total}]
            [[dg_{k+1}/dy_{k+1}]_{n_alg x n_total}]

        Args:
            t_array: time points, shape (N+1,)
            y_array: states at time points, shape (N+1, n_total) or (n_total, N+1)
            p: parameter vector, if None uses self.p

        Returns:
            J_prev: list of N Jacobian matrices, where J_prev[i] = dr_{i+1}/dy_i
            J_curr: list of N Jacobian matrices, where J_curr[i] = dr_{i+1}/dy_{i+1}
        """
        # Use default parameters if not provided
        if p is None:
            p = self.p
        p = jnp.array(p)

        # Detect and transpose if needed
        if y_array.shape[0] == self.n_total and y_array.shape[1] == len(t_array):
            y_array = y_array.T

        N = len(t_array) - 1
        if N <= 0:
            return [], []

        # Compute all f and g Jacobians
        t_all = jnp.array(t_array)
        y_all = jnp.array(y_array)

        # df/dy at all time points, shape: (N+1, n_states, n_total)
        df_dy = self._jac_f_vmapped_jit(t_all, y_all, p)

        # dg/dy at all time points, shape: (N+1, n_alg, n_total)
        dg_dy = self._jac_g_vmapped_jit(t_all, y_all, p)

        # Create identity matrices (EXACT - no roundoff)
        I_states = np.eye(self.n_states)
        I_alg = np.eye(self.n_alg)

        # Initialize lists
        J_prev_list = []
        J_curr_list = []

        # Build Jacobian blocks for each interval
        for k in range(N):
            h = t_array[k+1] - t_array[k]

            # Extract Jacobians at k and k+1
            df_k = np.array(df_dy[k])      # shape: (n_states, n_total)
            df_kp1 = np.array(df_dy[k+1])  # shape: (n_states, n_total)
            dg_kp1 = np.array(dg_dy[k+1])  # shape: (n_alg, n_total)

            # Construct J_prev[k] = dr_{k+1}/dy_k
            # Top block (differential): -I/h - 0.5*df_k/dy_k for first n_states columns, then -0.5*df_k/dz_k
            J_prev_diff = np.zeros((self.n_states, self.n_total))
            J_prev_diff[:, :self.n_states] = -I_states / h  # Exact identity contribution
            J_prev_diff -= 0.5 * df_k  # Add f Jacobian contribution

            # Bottom block (algebraic): all zeros
            J_prev_alg = np.zeros((self.n_alg, self.n_total))

            # Combine
            J_prev = np.vstack([J_prev_diff, J_prev_alg])
            J_prev_list.append(J_prev)

            # Construct J_curr[k] = dr_{k+1}/dy_{k+1}
            # Top block (differential): I/h - 0.5*df_{k+1}/dy_{k+1}
            J_curr_diff = np.zeros((self.n_states, self.n_total))
            J_curr_diff[:, :self.n_states] = I_states / h  # Exact identity contribution
            J_curr_diff -= 0.5 * df_kp1  # Subtract f Jacobian contribution

            # Bottom block (algebraic): dg_{k+1}/dy_{k+1}
            J_curr_alg = dg_kp1.copy()

            # Combine
            J_curr = np.vstack([J_curr_diff, J_curr_alg])
            J_curr_list.append(J_curr)

        return J_prev_list, J_curr_list

    def compute_jacobian_blocks(self, t_array: np.ndarray, y_array: np.ndarray, p: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobian blocks for all time intervals in parallel.

        Given time points t_0, t_1, ..., t_N and states y_0, y_1, ..., y_N,
        computes Jacobians for intervals [t_0, t_1], [t_1, t_2], ..., [t_{N-1}, t_N].

        Note: y_0 is treated as initial condition (fixed), so we compute N Jacobian blocks
        for y_1, y_2, ..., y_N.

        Args:
            t_array: time points, shape (N+1,) for N intervals
            y_array: states at time points, shape (N+1, n_total) or (n_total, N+1)
            p: parameter vector, if None uses self.p

        Returns:
            J_prev: list of N Jacobian matrices, where J_prev[i] = dr_{i+1}/dy_i
            J_curr: list of N Jacobian matrices, where J_curr[i] = dr_{i+1}/dy_{i+1}

            Each matrix has shape (n_total, n_total).
            Note: r_1, r_2, ..., r_N are residuals for intervals [t_0,t_1], [t_1,t_2], ..., [t_{N-1},t_N]
        """
        # Use default parameters if not provided
        if p is None:
            p = self.p
        p = jnp.array(p)

        # Detect and transpose if needed
        if y_array.shape[0] == self.n_total and y_array.shape[1] == len(t_array):
            y_array = y_array.T

        N = len(t_array) - 1  # Number of intervals

        if N <= 0:
            return [], []

        # Prepare arrays for vectorized computation
        t_k = jnp.array(t_array[:-1])      # t_0, t_1, ..., t_{N-1}
        t_kp1 = jnp.array(t_array[1:])     # t_1, t_2, ..., t_N
        y_k = jnp.array(y_array[:-1])      # y_0, y_1, ..., y_{N-1}
        y_kp1 = jnp.array(y_array[1:])     # y_1, y_2, ..., y_N

        # Compute Jacobians in parallel using vmapped functions
        # For interval i: residual r_{i+1} depends on (y_i, y_{i+1})
        # J_prev[i] = dr_{i+1}/dy_i (Jacobian w.r.t. first argument of residual)
        # J_curr[i] = dr_{i+1}/dy_{i+1} (Jacobian w.r.t. second argument of residual)
        J_prev = self._jac_y_k_vmapped(t_k, t_kp1, y_k, y_kp1, p)      # shape: (N, n_total, n_total)
        J_curr = self._jac_y_kp1_vmapped(t_k, t_kp1, y_k, y_kp1, p)    # shape: (N, n_total, n_total)

        # Convert to numpy and return as list of matrices
        J_prev_list = [np.array(J_prev[i]) for i in range(N)]
        J_curr_list = [np.array(J_curr[i]) for i in range(N)]

        return J_prev_list, J_curr_list

    def compute_jacobian_blocks_jit(self, t_array: np.ndarray, y_array: np.ndarray, p: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobian blocks for all time intervals in parallel using JIT-compiled functions.

        This method is identical to compute_jacobian_blocks() but uses JIT-compiled vmapped functions
        for improved performance, especially on repeated calls.

        Given time points t_0, t_1, ..., t_N and states y_0, y_1, ..., y_N,
        computes Jacobians for intervals [t_0, t_1], [t_1, t_2], ..., [t_{N-1}, t_N].

        Note: y_0 is treated as initial condition (fixed), so we compute N Jacobian blocks
        for y_1, y_2, ..., y_N.

        Args:
            t_array: time points, shape (N+1,) for N intervals
            y_array: states at time points, shape (N+1, n_total) or (n_total, N+1)
            p: parameter vector, if None uses self.p

        Returns:
            J_prev: list of N Jacobian matrices, where J_prev[i] = dr_{i+1}/dy_i
            J_curr: list of N Jacobian matrices, where J_curr[i] = dr_{i+1}/dy_{i+1}

            Each matrix has shape (n_total, n_total).
            Note: r_1, r_2, ..., r_N are residuals for intervals [t_0,t_1], [t_1,t_2], ..., [t_{N-1},t_N]
        """
        # Use default parameters if not provided
        if p is None:
            p = self.p
        p = jnp.array(p)

        # Detect and transpose if needed
        if y_array.shape[0] == self.n_total and y_array.shape[1] == len(t_array):
            y_array = y_array.T

        N = len(t_array) - 1  # Number of intervals

        if N <= 0:
            return [], []

        # Prepare arrays for vectorized computation
        t_k = jnp.array(t_array[:-1])      # t_0, t_1, ..., t_{N-1}
        t_kp1 = jnp.array(t_array[1:])     # t_1, t_2, ..., t_N
        y_k = jnp.array(y_array[:-1])      # y_0, y_1, ..., y_{N-1}
        y_kp1 = jnp.array(y_array[1:])     # y_1, y_2, ..., y_N

        # Compute Jacobians in parallel using JIT-compiled vmapped functions
        # For interval i: residual r_{i+1} depends on (y_i, y_{i+1})
        # J_prev[i] = dr_{i+1}/dy_i (Jacobian w.r.t. first argument of residual)
        # J_curr[i] = dr_{i+1}/dy_{i+1} (Jacobian w.r.t. second argument of residual)
        J_prev = self._jac_y_k_vmapped_jit(t_k, t_kp1, y_k, y_kp1, p)      # shape: (N, n_total, n_total)
        J_curr = self._jac_y_kp1_vmapped_jit(t_k, t_kp1, y_k, y_kp1, p)    # shape: (N, n_total, n_total)

        # Convert to numpy and return as list of matrices
        J_prev_list = [np.array(J_prev[i]) for i in range(N)]
        J_curr_list = [np.array(J_curr[i]) for i in range(N)]

        return J_prev_list, J_curr_list

    def compute_parameter_jacobian(self, t_array: np.ndarray, y_array: np.ndarray, p: np.ndarray = None) -> List[np.ndarray]:
        """
        Compute Jacobian blocks with respect to parameters for all time intervals.

        Given time points t_0, t_1, ..., t_N and states y_0, y_1, ..., y_N,
        computes dr/dp for intervals [t_0, t_1], [t_1, t_2], ..., [t_{N-1}, t_N].

        Args:
            t_array: time points, shape (N+1,) for N intervals
            y_array: states at time points, shape (N+1, n_total) or (n_total, N+1)
            p: parameter vector, if None uses self.p

        Returns:
            J_param: list of N Jacobian matrices, where J_param[i] = dr_{i+1}/dp
                     Each matrix has shape (n_total, n_params)
        """
        # Use default parameters if not provided
        if p is None:
            p = self.p
        p = jnp.array(p)

        # Detect and transpose if needed
        if y_array.shape[0] == self.n_total and y_array.shape[1] == len(t_array):
            y_array = y_array.T

        N = len(t_array) - 1  # Number of intervals

        if N <= 0:
            return []

        # Prepare arrays for vectorized computation
        t_k = jnp.array(t_array[:-1])      # t_0, t_1, ..., t_{N-1}
        t_kp1 = jnp.array(t_array[1:])     # t_1, t_2, ..., t_N
        y_k = jnp.array(y_array[:-1])      # y_0, y_1, ..., y_{N-1}
        y_kp1 = jnp.array(y_array[1:])     # y_1, y_2, ..., y_N

        # Compute parameter Jacobians in parallel using JIT-compiled vmapped function
        J_param = self._jac_p_vmapped_jit(t_k, t_kp1, y_k, y_kp1, p)  # shape: (N, n_total, n_params)

        # Convert to numpy and return as list of matrices
        J_param_list = [np.array(J_param[i]) for i in range(N)]

        return J_param_list

    def assemble_full_parameter_jacobian(self, t_array: np.ndarray, y_array: np.ndarray, p: np.ndarray = None) -> np.ndarray:
        """
        Compute and assemble full parameter Jacobian matrix.

        The full parameter Jacobian dR/dp where R = [r_1, r_2, ..., r_N] has structure:

            J_param_full = [
                dr_1/dp
                dr_2/dp
                dr_3/dp
                ...
                dr_N/dp
            ]

        Args:
            t_array: time points, shape (N+1,) for N intervals
            y_array: states at time points, shape (N+1, n_total) or (n_total, N+1)
            p: parameter vector, if None uses self.p

        Returns:
            J_param_full: full parameter Jacobian matrix, shape (N*n_total, n_params)
        """
        # Use default parameters if not provided
        if p is None:
            p = self.p
        p = jnp.array(p)

        # Detect and transpose if needed
        if y_array.shape[0] == self.n_total and y_array.shape[1] == len(t_array):
            y_array = y_array.T

        N = len(t_array) - 1
        if N <= 0:
            return np.zeros((0, len(p)))

        # Prepare arrays for vectorized computation
        t_k = jnp.array(t_array[:-1])      # t_0, t_1, ..., t_{N-1}
        t_kp1 = jnp.array(t_array[1:])     # t_1, t_2, ..., t_N
        y_k = jnp.array(y_array[:-1])      # y_0, y_1, ..., y_{N-1}
        y_kp1 = jnp.array(y_array[1:])     # y_1, y_2, ..., y_N

        # Compute parameter Jacobians in parallel using JIT-compiled vmapped function
        J_param = self._jac_p_vmapped_jit(t_k, t_kp1, y_k, y_kp1, p)  # shape: (N, n_total, n_params)

        # Reshape: stack all blocks vertically
        # From (N, n_total, n_params) to (N*n_total, n_params)
        J_param_full = np.array(J_param).reshape(N * self.n_total, len(p))

        return J_param_full

    def assemble_full_jacobian(self, J_prev: List[np.ndarray], J_curr: List[np.ndarray]) -> np.ndarray:
        """
        Assemble full sparse Jacobian from block lists.

        The full Jacobian dR/dY where R = [r_1, r_2, ..., r_N] and Y = [y_1, y_2, ..., y_N]
        (y_0 is fixed initial condition) has block-bidiagonal structure:

            J_full = [
                dr_1/dy_1,  0,          0,          ...,  0
                dr_2/dy_1,  dr_2/dy_2,  0,          ...,  0
                0,          dr_3/dy_2,  dr_3/dy_3,  ...,  0
                ...
                0,          0,          ..., dr_N/dy_{N-1}, dr_N/dy_N
            ]

        which in terms of our lists is:

            J_full = [
                J_curr[0],  0,          0,          ...,  0
                J_prev[1],  J_curr[1],  0,          ...,  0
                0,          J_prev[2],  J_curr[2],  ...,  0
                ...
                0,          0,          ..., J_prev[N-1], J_curr[N-1]
            ]

        Args:
            J_prev: list where J_prev[i] = dr_{i+1}/dy_i
            J_curr: list where J_curr[i] = dr_{i+1}/dy_{i+1}

        Returns:
            J_full: full Jacobian matrix, shape (N*n_total, N*n_total)
        """
        N = len(J_curr)
        n = self.n_total

        J_full = np.zeros((N * n, N * n))

        for k in range(N):
            # Diagonal blocks: dr_{k+1}/dy_{k+1} goes to row block k, column block k
            J_full[k*n:(k+1)*n, k*n:(k+1)*n] = J_curr[k]

            # Sub-diagonal blocks: dr_{k+1}/dy_k goes to row block k, column block k-1 (for k>0)
            if k > 0:
                J_full[k*n:(k+1)*n, (k-1)*n:k*n] = J_prev[k]

        return J_full
