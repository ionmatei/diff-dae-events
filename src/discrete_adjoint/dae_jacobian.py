"""
DAE Jacobian Computation using JAX

Computes Jacobians of implicit time discretization residual functions with respect
to states at time points, parallelized over time using JAX vmap.

Supported discretization methods:
1. Backward Euler (order 1) - A-stable, L-stable
2. Trapezoidal / Crank-Nicolson (order 2) - A-stable
3. BDF2 - Backward Differentiation Formula order 2 (order 2) - A-stable, L-stable
4. BDF3 - Backward Differentiation Formula order 3 (order 3) - A-stable
5. BDF4 - Backward Differentiation Formula order 4 (order 4) - A-stable
6. BDF5 - Backward Differentiation Formula order 5 (order 5) - A-stable
7. BDF6 - Backward Differentiation Formula order 6 (order 6, highest stable BDF)

For a general implicit residual r(y_k, y_{k+1}, ...), we compute:
  - dr/dy_k: Jacobian with respect to previous state
  - dr/dy_{k+1}: Jacobian with respect to current state

These form the block-diagonal and block-superdiagonal of the full Jacobian matrix.

BDF coefficients (for constant step size):
- BDF1: y_n - y_{n-1} = h*f_n
- BDF2: (3/2)*y_n - 2*y_{n-1} + (1/2)*y_{n-2} = h*f_n
- BDF3: (11/6)*y_n - 3*y_{n-1} + (3/2)*y_{n-2} - (1/3)*y_{n-3} = h*f_n
- BDF4: (25/12)*y_n - 4*y_{n-1} + 3*y_{n-2} - (4/3)*y_{n-3} + (1/4)*y_{n-4} = h*f_n
- BDF5: (137/60)*y_n - 5*y_{n-1} + 5*y_{n-2} - (10/3)*y_{n-3} + (5/4)*y_{n-4} - (1/5)*y_{n-5} = h*f_n
- BDF6: (147/60)*y_n - 6*y_{n-1} + (15/2)*y_{n-2} - (20/3)*y_{n-3} + (15/4)*y_{n-4} - (6/5)*y_{n-5} + (1/6)*y_{n-6} = h*f_n
"""

import numpy as np
from typing import Dict, List, Tuple
import re
import time
import os
from jax import lax

# Check if JAX should use CPU from environment variable
# Set JAX_PLATFORM_NAME=cpu before running your script to force CPU usage
_USE_CPU = os.environ.get('JAX_PLATFORM_NAME', '').lower() == 'cpu'

try:
    import jax
    import jax.numpy as jnp
    from jax import vmap, jacfwd, jacrev, jit
    JAX_AVAILABLE = True

    # Print device info on import
    try:
        devices = jax.devices()
        device_type = devices[0].platform
        print(f"JAX initialized with device: {device_type} ({len(devices)} device(s))")
    except:
        pass

except ImportError:
    JAX_AVAILABLE = False
    jnp = np
    print("Error: JAX is required for Jacobian computation")
    raise


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

# Valid time discretization methods
VALID_METHODS = ['backward_euler', 'trapezoidal', 'bdf2', 'bdf3', 'bdf4', 'bdf5', 'bdf6']


def configure_jax_device(use_cpu: bool = False):
    """
    Configure JAX to use CPU or GPU.

    Args:
        use_cpu: If True, forces JAX to use CPU. If False, uses default (GPU if available).

    Note:
        IMPORTANT: This function should be called BEFORE importing this module for the first time,
        or BEFORE any JAX operations are performed. JAX configuration is global and affects all modules.

        RECOMMENDED METHODS (in order of preference):

        1. Set environment variable before running Python (most reliable):
           $ JAX_PLATFORM_NAME=cpu python your_script.py

        2. Set environment variable at the top of your main script (before any imports):
           import os
           os.environ['JAX_PLATFORM_NAME'] = 'cpu'
           # Now import modules that use JAX
           from src.dae_jacobian import DAEOptimizer

        3. Use this function (may not work if JAX already compiled functions):
           from src.dae_jacobian import configure_jax_device
           configure_jax_device(use_cpu=True)

        Once JAX is configured, it affects ALL files that import JAX, so you only need
        to configure it ONCE at the start of your program.
    """
    if use_cpu:
        try:
            jax.config.update('jax_platform_name', 'cpu')
            print("JAX configured to use CPU")
            devices = jax.devices()
            print(f"Current JAX devices: {devices}")
        except Exception as e:
            print(f"Warning: Could not update JAX config. Error: {e}")
            print("Try setting JAX_PLATFORM_NAME=cpu environment variable before running Python.")
    else:
        # Check what device JAX is using
        try:
            devices = jax.devices()
            print(f"JAX using devices: {devices}")
        except:
            pass


class DAEJacobian:
    """
    Computes Jacobians of DAE residual functions using JAX.

    Supports multiple time discretization methods:
    - 'backward_euler': First-order implicit (A-stable, L-stable)
    - 'trapezoidal': Second-order implicit (A-stable), also known as Crank-Nicolson
    - 'bdf2': Second-order BDF (A-stable, L-stable)
    - 'bdf3': Third-order BDF (A-stable)
    - 'bdf4': Fourth-order BDF (A-stable)
    - 'bdf5': Fifth-order BDF (A-stable)
    - 'bdf6': Sixth-order BDF (A-stable, highest stable BDF)

    For a general implicit residual r(y_k, y_{k+1}, ...), we compute:
        - J_k = dr/dy_k: Jacobian with respect to previous state
        - J_{k+1} = dr/dy_{k+1}: Jacobian with respect to current state
    """

    def __init__(self, dae_data: dict, method: str = 'trapezoidal'):
        """
        Initialize from DAE specification.

        Args:
            dae_data: Dictionary containing DAE specification with keys:
                - states: list of differential state variables
                - alg_vars: list of algebraic variables
                - parameters: list of parameters
                - f: list of differential equations (dx/dt = f)
                - g: list of algebraic equations (0 = g)
            method: Time discretization method. One of:
                - 'backward_euler': First-order implicit
                - 'trapezoidal': Second-order (default)
                - 'bdf2' through 'bdf6': Higher-order BDF methods
        """
        # Validate method
        if method not in VALID_METHODS:
            raise ValueError(f"method must be one of {VALID_METHODS}, got '{method}'")
        self.method = method

        # Extract variables
        self.states = dae_data['states']
        self.alg_vars = dae_data['alg_vars']
        self.parameters = dae_data['parameters']

        # Extract equations
        self.f_eqs = dae_data['f']
        self.g_eqs = dae_data['g']
        self.h_eqs = dae_data.get('h', None)  # Output equations (optional)

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

        # For selective parameter optimization (set by DAEOptimizer)
        self.optimize_indices = None  # Indices of parameters to optimize
        self.p_all_default = None     # Default values for all parameters

        # Compile equations
        self._compile_equations()

        # Compile vectorized Jacobian functions
        self._compile_jacobian_functions()

    def set_selective_optimization(self, optimize_indices, p_all_default):
        """
        Configure selective parameter optimization.

        Args:
            optimize_indices: List/array of indices of parameters to optimize
            p_all_default: Array of default values for ALL parameters
        """
        self.optimize_indices = optimize_indices
        self.p_all_default = jnp.array(p_all_default)

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

        # Compile h equations (outputs)
        # If not provided, default to returning state vectors
        self.h_funcs = []
        if self.h_eqs:
            for eq in self.h_eqs:
                # Format: output_name = expression
                if '=' in eq:
                    _, rhs = eq.split('=', 1)
                    self.h_funcs.append(rhs.strip())
                else:
                    self.h_funcs.append(eq)
        else:
            # Default: outputs are the differential states only
            self.h_funcs = self.state_names.copy()

    def _create_jax_eval_namespace(self, t, x, z):
        """Create namespace for JAX equation evaluation (deprecated - use version with params)."""
        return self._create_jax_eval_namespace_with_params(t, x, z, self.p)

    def _create_jax_eval_namespace_with_params(self, t, x, z, p, optimize_indices=None, p_all_default=None):
        """
        Create namespace for JAX equation evaluation with explicit parameters.

        Args:
            t: time
            x: differential states
            z: algebraic variables
            p: parameter vector - can be either:
                - Full parameter vector (all parameters) when optimize_indices is None
                - Optimized parameters only when optimize_indices is provided
            optimize_indices: Optional list of indices indicating which parameters are in p
            p_all_default: Optional full parameter vector with default values for fixed params
        """
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
        if optimize_indices is None:
            # p contains all parameters in order
            for i, name in enumerate(self.param_names):
                ns[name] = p[i]
        else:
            # p contains only optimized parameters
            # Use default values for fixed parameters
            if p_all_default is None:
                raise ValueError("p_all_default must be provided when optimize_indices is specified")

            # Start with all default values
            param_values = list(p_all_default)

            # Override with optimized parameter values
            for opt_idx, param_idx in enumerate(optimize_indices):
                param_values[param_idx] = p[opt_idx]

            # Add to namespace
            for i, name in enumerate(self.param_names):
                ns[name] = param_values[i]

        return ns

    def eval_f_jax(self, t, x, z, p):
        """Evaluate f(t, x, z, p) using JAX."""
        ns = self._create_jax_eval_namespace_with_params(
            t, x, z, p,
            optimize_indices=self.optimize_indices,
            p_all_default=self.p_all_default
        )
        dxdt_list = [eval(expr, ns) for expr in self.f_funcs]
        return jnp.array(dxdt_list)

    def eval_g_jax(self, t, x, z, p):
        """Evaluate g(t, x, z, p) using JAX."""
        ns = self._create_jax_eval_namespace_with_params(
            t, x, z, p,
            optimize_indices=self.optimize_indices,
            p_all_default=self.p_all_default
        )
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

    def residual_backward_euler_single(self, t_k, t_kp1, y_k, y_kp1, p):
        """
        Backward Euler residual for a single time interval [t_k, t_{k+1}].

        Backward Euler: (x_{k+1} - x_k) / h = f(t_{k+1}, x_{k+1}, z_{k+1})

        Args:
            t_k: time at step k
            t_kp1: time at step k+1
            y_k: combined state [x_k, z_k] at step k
            y_kp1: combined state [x_{k+1}, z_{k+1}] at step k+1
            p: parameter vector

        Returns:
            residual: residual vector for the backward Euler scheme
        """
        h = t_kp1 - t_k

        # Split states
        x_k = y_k[:self.n_states]
        x_kp1 = y_kp1[:self.n_states]
        z_kp1 = y_kp1[self.n_states:]

        # Evaluate f at time k+1 only
        f_kp1 = self.eval_f_jax(t_kp1, x_kp1, z_kp1, p)

        # Evaluate g at time k+1
        g_kp1 = self.eval_g_jax(t_kp1, x_kp1, z_kp1, p)

        # Residual: (x_{k+1} - x_k)/h - f_{k+1} = 0
        r_diff = (x_kp1 - x_k) / h - f_kp1
        r_alg = g_kp1

        return jnp.concatenate([r_diff, r_alg])

    def residual_bdf2_single(self, t_k, t_kp1, y_k, y_kp1, y_km1, p):
        """
        BDF2 residual for a single time interval.

        BDF2: (3/2)*x_{k+1} - 2*x_k + (1/2)*x_{k-1} = h * f(t_{k+1}, x_{k+1}, z_{k+1})

        Args:
            t_k: time at step k
            t_kp1: time at step k+1
            y_k: combined state [x_k, z_k] at step k
            y_kp1: combined state [x_{k+1}, z_{k+1}] at step k+1
            y_km1: combined state [x_{k-1}, z_{k-1}] at step k-1
            p: parameter vector

        Returns:
            residual: residual vector for the BDF2 scheme
        """
        h = t_kp1 - t_k

        # Split states
        x_km1 = y_km1[:self.n_states]
        x_k = y_k[:self.n_states]
        x_kp1 = y_kp1[:self.n_states]
        z_kp1 = y_kp1[self.n_states:]

        # BDF2 coefficients: (3/2)*y_n - 2*y_{n-1} + (1/2)*y_{n-2} = h*f_n
        coeffs = BDF_COEFFICIENTS[2][0]

        # Evaluate f at time k+1
        f_kp1 = self.eval_f_jax(t_kp1, x_kp1, z_kp1, p)

        # Evaluate g at time k+1
        g_kp1 = self.eval_g_jax(t_kp1, x_kp1, z_kp1, p)

        # Residual: (coeffs[0]*x_{k+1} + coeffs[1]*x_k + coeffs[2]*x_{k-1})/h - f_{k+1} = 0
        r_diff = (coeffs[0] * x_kp1 + coeffs[1] * x_k + coeffs[2] * x_km1) / h - f_kp1
        r_alg = g_kp1

        return jnp.concatenate([r_diff, r_alg])

    def residual_bdf_single(self, t_kp1, h, y_history, p, order):
        """
        Generic BDF residual for orders 2-6.

        BDF formula: sum_j(coeffs[j] * x_{k+1-j}) / h = f(t_{k+1}, x_{k+1}, z_{k+1})

        Args:
            t_kp1: time at step k+1
            h: step size
            y_history: list of states [y_{k+1}, y_k, y_{k-1}, ...] up to order+1 elements
            p: parameter vector
            order: BDF order (2-6)

        Returns:
            residual: residual vector for the BDF scheme
        """
        coeffs = BDF_COEFFICIENTS[order][0]

        # Current state
        y_kp1 = y_history[0]
        x_kp1 = y_kp1[:self.n_states]
        z_kp1 = y_kp1[self.n_states:]

        # Compute BDF derivative approximation
        dxdt_approx = coeffs[0] * x_kp1
        for j in range(1, order + 1):
            x_j = y_history[j][:self.n_states]
            dxdt_approx = dxdt_approx + coeffs[j] * x_j
        dxdt_approx = dxdt_approx / h

        # Evaluate f at time k+1
        f_kp1 = self.eval_f_jax(t_kp1, x_kp1, z_kp1, p)

        # Evaluate g at time k+1
        g_kp1 = self.eval_g_jax(t_kp1, x_kp1, z_kp1, p)

        # Residual: dxdt_approx - f_{k+1} = 0
        r_diff = dxdt_approx - f_kp1
        r_alg = g_kp1

        return jnp.concatenate([r_diff, r_alg])

    def residual_single(self, t_k, t_kp1, y_k, y_kp1, p):
        """
        Compute residual using the selected method.

        This is a unified interface that dispatches to the appropriate residual function
        based on self.method. For BDF methods of order > 2, this uses the two-point
        interface with fallback to backward Euler for early steps.

        Args:
            t_k: time at step k
            t_kp1: time at step k+1
            y_k: combined state [x_k, z_k] at step k
            y_kp1: combined state [x_{k+1}, z_{k+1}] at step k+1
            p: parameter vector

        Returns:
            residual: residual vector for the selected scheme
        """
        if self.method == 'backward_euler':
            return self.residual_backward_euler_single(t_k, t_kp1, y_k, y_kp1, p)
        elif self.method == 'trapezoidal':
            return self.residual_trapezoidal_single(t_k, t_kp1, y_k, y_kp1, p)
        elif self.method == 'bdf2':
            # For BDF2 with only two points, fall back to backward Euler
            # (proper BDF2 needs three points handled in _compile_jacobian_functions)
            return self.residual_backward_euler_single(t_k, t_kp1, y_k, y_kp1, p)
        else:
            # BDF3-6: fall back to backward Euler in this simple interface
            # (proper multi-step handled in _compile_jacobian_functions)
            return self.residual_backward_euler_single(t_k, t_kp1, y_k, y_kp1, p)

    def _compile_jacobian_functions(self):
        """
        Compile vectorized Jacobian functions using JAX vmap.

        Creates vmapped versions of:
            - dr/dy_k: Jacobian with respect to previous state
            - dr/dy_{k+1}: Jacobian with respect to current state
            - dr/dp: Jacobian with respect to parameters

        The residual function used depends on self.method.
        """
        # Select the appropriate residual function based on method
        if self.method == 'backward_euler':
            residual_fn = self.residual_backward_euler_single
        elif self.method == 'trapezoidal':
            residual_fn = self.residual_trapezoidal_single
        else:
            # For BDF methods, use the unified residual_single which handles dispatch
            # Note: For higher-order BDF, proper multi-step implementation would require
            # different Jacobian structure. Here we use a simplified two-step approach.
            residual_fn = self.residual_single

        # Single-interval Jacobian function with respect to y_k
        def jac_y_k_single(t_k, t_kp1, y_k, y_kp1, p):
            # Fix y_kp1, p and differentiate with respect to y_k
            return jacfwd(lambda yk: residual_fn(t_k, t_kp1, yk, y_kp1, p))(y_k)

        # Single-interval Jacobian function with respect to y_{k+1}
        def jac_y_kp1_single(t_k, t_kp1, y_k, y_kp1, p):
            # Fix y_k, p and differentiate with respect to y_{k+1}
            return jacfwd(lambda ykp1: residual_fn(t_k, t_kp1, y_k, ykp1, p))(y_kp1)

        # Single-interval Jacobian function with respect to parameters
        def jac_p_single(t_k, t_kp1, y_k, y_kp1, p):
            # Fix y_k, y_kp1 and differentiate with respect to p
            return jacfwd(lambda pp: residual_fn(t_k, t_kp1, y_k, y_kp1, pp))(p)

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

        # Create JIT-compiled loss and gradient functions
        self._compile_loss_functions()

        print("JAX Jacobian vmap functions compiled successfully!")

    def _compile_f_g_jacobian_functions(self):
        """
        Compile vectorized Jacobian functions for f and g separately.

        This allows analytical construction of residual Jacobians for all methods.
        The Jacobians df/dy and dg/dy are computed via vmap and then combined
        with exact identity matrices (no roundoff) based on the selected method.

        The Jacobian structure depends on the method:
        - backward_euler: only uses df_{k+1}/dy and dg_{k+1}/dy
        - trapezoidal: uses df_k/dy, df_{k+1}/dy, and dg_{k+1}/dy
        - bdf2-6: uses df_{k+1}/dy and dg_{k+1}/dy with BDF coefficients
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

    def _compile_loss_functions(self):
        """
        Compile JIT versions of loss and gradient functions.

        Creates JIT-compiled methods for efficient loss computation and gradient evaluation.
        """
        # Create JIT-compiled versions of loss functions
        self.trajectory_loss_jit = jit(self.trajectory_loss)
        self.trajectory_loss_gradient_jit = jit(self.trajectory_loss_gradient)
        self.trajectory_loss_gradient_analytical_jit = jit(self.trajectory_loss_gradient_analytical)

        # Create JIT-compiled versions of parameter Jacobian functions
        self.compute_residual_jacobian_wrt_params_jit = jit(self.compute_residual_jacobian_wrt_params)
        self.compute_residual_jacobian_wrt_params_analytical_jit = jit(self.compute_residual_jacobian_wrt_params_analytical)

    def compute_jacobian_blocks_analytical(self, t_array: np.ndarray, y_array: np.ndarray, p: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobian blocks analytically from f and g Jacobians.

        This method constructs the residual Jacobian blocks using:
        1. Exact identity matrices (no autodiff roundoff)
        2. Jacobians of f and g computed via vmap

        The structure depends on the discretization method:

        For backward Euler: r = [(x_{k+1} - x_k)/h - f_{k+1}, g_{k+1}]
            dr/dy_k = [[-I/h, 0], [0, 0]]
            dr/dy_{k+1} = [[I/h - df_{k+1}/dy_{k+1}], [dg_{k+1}/dy_{k+1}]]

        For trapezoidal: r = [(x_{k+1} - x_k)/h - 0.5*(f_k + f_{k+1}), g_{k+1}]
            dr/dy_k = [[-I/h - 0.5*df_k/dy_k], [0]]
            dr/dy_{k+1} = [[I/h - 0.5*df_{k+1}/dy_{k+1}], [dg_{k+1}/dy_{k+1}]]

        For BDF methods: Uses BDF coefficients for derivative approximation
            dr/dy_k = [[c1/h, 0], [0, 0]]  (coefficients depend on BDF order)
            dr/dy_{k+1} = [[c0/h - df_{k+1}/dy_{k+1}], [dg_{k+1}/dy_{k+1}]]

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

        # Initialize lists
        J_prev_list = []
        J_curr_list = []

        # Get BDF order if applicable
        bdf_order = None
        if self.method.startswith('bdf'):
            bdf_order = int(self.method[3])
            coeffs = BDF_COEFFICIENTS[bdf_order][0]

        # Build Jacobian blocks for each interval
        for k in range(N):
            h = t_array[k+1] - t_array[k]

            # Extract Jacobians at k and k+1
            df_k = np.array(df_dy[k])      # shape: (n_states, n_total)
            df_kp1 = np.array(df_dy[k+1])  # shape: (n_states, n_total)
            dg_kp1 = np.array(dg_dy[k+1])  # shape: (n_alg, n_total)

            if self.method == 'backward_euler':
                # Backward Euler: r = (x_{k+1} - x_k)/h - f_{k+1}
                # dr/dy_k = [-I/h, 0; 0, 0]
                J_prev_diff = np.zeros((self.n_states, self.n_total))
                J_prev_diff[:, :self.n_states] = -I_states / h

                # dr/dy_{k+1} = [I/h - df_{k+1}/dy; dg_{k+1}/dy]
                J_curr_diff = np.zeros((self.n_states, self.n_total))
                J_curr_diff[:, :self.n_states] = I_states / h
                J_curr_diff -= df_kp1

            elif self.method == 'trapezoidal':
                # Trapezoidal: r = (x_{k+1} - x_k)/h - 0.5*(f_k + f_{k+1})
                # dr/dy_k = [-I/h - 0.5*df_k/dy; 0]
                J_prev_diff = np.zeros((self.n_states, self.n_total))
                J_prev_diff[:, :self.n_states] = -I_states / h
                J_prev_diff -= 0.5 * df_k

                # dr/dy_{k+1} = [I/h - 0.5*df_{k+1}/dy; dg_{k+1}/dy]
                J_curr_diff = np.zeros((self.n_states, self.n_total))
                J_curr_diff[:, :self.n_states] = I_states / h
                J_curr_diff -= 0.5 * df_kp1

            elif self.method.startswith('bdf'):
                # BDF methods: derivative approx = sum(c_j * x_{k+1-j}) / h
                # For two-step interface, we use adaptive order based on step index
                # For early steps (k < order-1), use lower order BDF

                # Determine effective order for this step
                effective_order = min(bdf_order, k + 1)  # k+1 because we need k+1 points for BDF(k+1)

                if effective_order == 1:
                    # BDF1 = Backward Euler
                    c0, c1 = 1.0, -1.0
                    J_prev_diff = np.zeros((self.n_states, self.n_total))
                    J_prev_diff[:, :self.n_states] = c1 * I_states / h

                    J_curr_diff = np.zeros((self.n_states, self.n_total))
                    J_curr_diff[:, :self.n_states] = c0 * I_states / h
                    J_curr_diff -= df_kp1
                else:
                    # Higher order BDF: only use c0 and c1 for the two-step Jacobian
                    # (full multi-step would require more history points)
                    eff_coeffs = BDF_COEFFICIENTS[effective_order][0]
                    c0, c1 = eff_coeffs[0], eff_coeffs[1]

                    # dr/dy_k coefficient is c1/h (second coefficient)
                    J_prev_diff = np.zeros((self.n_states, self.n_total))
                    J_prev_diff[:, :self.n_states] = c1 * I_states / h

                    # dr/dy_{k+1} = c0/h * I - df_{k+1}/dy
                    J_curr_diff = np.zeros((self.n_states, self.n_total))
                    J_curr_diff[:, :self.n_states] = c0 * I_states / h
                    J_curr_diff -= df_kp1

            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Bottom block (algebraic): all zeros for J_prev
            J_prev_alg = np.zeros((self.n_alg, self.n_total))

            # Combine J_prev
            J_prev = np.vstack([J_prev_diff, J_prev_alg])
            J_prev_list.append(J_prev)

            # Bottom block (algebraic): dg_{k+1}/dy_{k+1} for J_curr
            J_curr_alg = dg_kp1.copy()

            # Combine J_curr
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

    def compute_parameter_jacobian_jit(self, t_array: np.ndarray, y_array: np.ndarray, p: np.ndarray = None) -> List[np.ndarray]:
        """
        Compute Jacobian blocks with respect to parameters using JIT-compiled functions.

        This method is identical to compute_parameter_jacobian() but returns the result
        without converting to numpy, maintaining JAX arrays for better performance in JIT contexts.

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

        # Convert to list but keep as numpy arrays (not JAX) for consistency
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
        # Keep as JAX array for JIT compatibility
        J_param_full = J_param.reshape(N * self.n_total, p.shape[0])

        return J_param_full

    def compute_residual_jacobian_wrt_params(
        self,
        t_array: np.ndarray,
        y_array: np.ndarray,
        p: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute Jacobian of full residual vector with respect to parameters.

        This computes dR/dp where R = [r_1, r_2, ..., r_N] is the stacked residual vector
        for all time intervals using the selected discretization method.

        Args:
            t_array: time points, shape (N+1,) for N intervals
            y_array: states at time points, shape (N+1, n_total) or (n_total, N+1)
            p: parameter vector, if None uses self.p

        Returns:
            dR_dp: full Jacobian matrix, shape (N*n_total, n_params)
                   This is the concatenation of all dr_k/dp blocks:
                   dR_dp = [dr_1/dp; dr_2/dp; ...; dr_N/dp]

        Notes:
            - Uses vmap for efficient parallel computation of all intervals
            - Each dr_k/dp has shape (n_total, n_params)
            - The result is reshaped from (N, n_total, n_params) to (N*n_total, n_params)
            - This is the same as assemble_full_parameter_jacobian but with clearer naming

        Example:
            >>> jac = DAEJacobian(dae_data, method='trapezoidal')
            >>> t_array = np.linspace(0, 1, 11)  # 11 time points, 10 intervals
            >>> y_array = ...  # trajectory of shape (11, n_total)
            >>> dR_dp = jac.compute_residual_jacobian_wrt_params(t_array, y_array)
            >>> print(dR_dp.shape)  # (10*n_total, n_params)
        """
        return self.assemble_full_parameter_jacobian(t_array, y_array, p)

    def compute_residual_jacobian_wrt_params_analytical(
        self,
        t_array: np.ndarray,
        y_array: np.ndarray,
        p: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute Jacobian of residual with respect to parameters analytically.

        This computes dr_k/dp directly from f and g Jacobians, which is more efficient
        and numerically accurate than autodiff through the residual function.

        The structure depends on the discretization method:

        For backward Euler:
            dr_k/dp = [[-df_{k+1}/dp], [dg_{k+1}/dp]]

        For trapezoidal:
            dr_k/dp = [[-0.5*(df_k/dp + df_{k+1}/dp)], [dg_{k+1}/dp]]

        For BDF methods:
            dr_k/dp = [[-df_{k+1}/dp], [dg_{k+1}/dp]]  (BDF methods evaluate f only at k+1)

        Args:
            t_array: time points, shape (N+1,) for N intervals
            y_array: states at time points, shape (N+1, n_total) or (n_total, N+1)
            p: parameter vector, if None uses self.p

        Returns:
            dR_dp: full Jacobian matrix, shape (N*n_total, n_params)

        Notes:
            - Computes df/dp and dg/dp separately using vmap
            - Constructs dr/dp analytically for better numerical properties
            - More efficient than automatic differentiation through residual
        """
        # Use default parameters if not provided
        if p is None:
            p = self.p
        p = jnp.array(p)

        # Detect and transpose if needed
        if y_array.shape[0] == self.n_total and y_array.shape[1] == len(t_array):
            y_array = y_array.T

        N = len(t_array) - 1
        n_params = len(p)

        if N <= 0:
            return np.zeros((0, n_params))

        # Convert to JAX arrays
        t_all = jnp.array(t_array)
        y_all = jnp.array(y_array)

        # Compute df/dp and dg/dp at all time points using vmap
        def df_dp_single(t, y):
            x = y[:self.n_states]
            z = y[self.n_states:]
            return jax.jacfwd(lambda pp: self.eval_f_jax(t, x, z, pp))(p)

        def dg_dp_single(t, y):
            x = y[:self.n_states]
            z = y[self.n_states:]
            return jax.jacfwd(lambda pp: self.eval_g_jax(t, x, z, pp))(p)

        # Vectorize over all time points
        df_dp_vec = jax.vmap(df_dp_single, in_axes=(0, 0))
        dg_dp_vec = jax.vmap(dg_dp_single, in_axes=(0, 0))

        df_dp_all = df_dp_vec(t_all, y_all)  # shape: (N+1, n_states, n_params)
        dg_dp_all = dg_dp_vec(t_all, y_all)  # shape: (N+1, n_alg, n_params)

        # Build dr_k/dp for each interval
        dR_dp_blocks = []

        for k in range(N):
            # Algebraic part is always: dg_{k+1}/dp
            dr_alg_dp = dg_dp_all[k + 1]  # shape: (n_alg, n_params)

            if self.method == 'backward_euler':
                # Backward Euler: dr/dp = [-df_{k+1}/dp, dg_{k+1}/dp]
                dr_diff_dp = -df_dp_all[k + 1]

            elif self.method == 'trapezoidal':
                # Trapezoidal: dr/dp = [-0.5*(df_k/dp + df_{k+1}/dp), dg_{k+1}/dp]
                dr_diff_dp = -0.5 * (df_dp_all[k] + df_dp_all[k + 1])

            elif self.method.startswith('bdf'):
                # BDF methods: evaluate f only at k+1
                # dr/dp = [-df_{k+1}/dp, dg_{k+1}/dp]
                dr_diff_dp = -df_dp_all[k + 1]

            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Stack vertically
            dr_k_dp = jnp.vstack([dr_diff_dp, dr_alg_dp])  # shape: (n_total, n_params)
            dR_dp_blocks.append(dr_k_dp)

        # Stack all blocks vertically
        dR_dp = jnp.vstack(dR_dp_blocks)  # shape: (N*n_total, n_params)

        return dR_dp

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

    def trajectory_loss(
        self,
        t_array: np.ndarray,
        y_array: np.ndarray,
        y_target_array: np.ndarray,
        p: np.ndarray = None
    ) -> float:
        """
        Compute trajectory tracking loss as sum of squared errors.

        Loss = sum_k ||h(t_k, y_k) - y_target_k||^2

        where h is the output function from the DAE specification.

        Args:
            t_array: time points, shape (N+1,) for N intervals
            y_array: states at time points, shape (N+1, n_total) or (n_total, N+1)
            y_target_array: target outputs at time points, shape (N+1, n_outputs) or (n_outputs, N+1)
            p: parameter vector, if None uses self.p

        Returns:
            loss: scalar loss value (sum of squared errors)

        Notes:
            - If h function is not defined, uses identity (returns differential states x)
            - Uses JAX for vectorized computation
        """
        # Use default parameters if not provided
        if p is None:
            p = self.p
        p = jnp.array(p)

        # Detect and transpose if needed
        if y_array.shape[0] == self.n_total and y_array.shape[1] == len(t_array):
            y_array = y_array.T

        # Determine output dimension
        if self.h_funcs:
            n_outputs = len(self.h_funcs)
        else:
            n_outputs = self.n_states

        # Detect and transpose target if needed
        if y_target_array.shape[0] == n_outputs and y_target_array.shape[1] == len(t_array):
            y_target_array = y_target_array.T

        # Convert to JAX arrays
        t_jax = jnp.array(t_array)
        y_jax = jnp.array(y_array)
        y_target_jax = jnp.array(y_target_array)

        # Compute outputs using vectorized eval_h
        def eval_h_single(t, y):
            x = y[:self.n_states]
            z = y[self.n_states:]
            return self.eval_h_jax(t, x, z, p)

        # Vectorize over time points
        eval_h_vec = jax.vmap(eval_h_single, in_axes=(0, 0))
        y_pred = eval_h_vec(t_jax, y_jax)  # shape: (N+1, n_outputs)

        # Compute squared error
        errors = y_pred - y_target_jax
        squared_errors = errors ** 2

        # Sum over all time points and outputs
        loss = jnp.sum(squared_errors)

        return loss

    def trajectory_loss_gradient(
        self,
        t_array: np.ndarray,
        y_array: np.ndarray,
        y_target_array: np.ndarray,
        p: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute gradient of trajectory loss with respect to states y.

        Computes dL/dy where L = sum_k ||h(t_k, y_k) - y_target_k||^2

        Args:
            t_array: time points, shape (N+1,) for N intervals
            y_array: states at time points, shape (N+1, n_total) or (n_total, N+1)
            y_target_array: target outputs at time points, shape (N+1, n_outputs) or (n_outputs, N+1)
            p: parameter vector, if None uses self.p

        Returns:
            grad: gradient dL/dy, shape (N+1, n_total)
                  grad[k, :] = dL/dy_k contains gradient w.r.t. state at time k

        Notes:
            - Uses JAX automatic differentiation (grad)
            - Returns gradient in same orientation as y_array input
        """
        # Use default parameters if not provided
        if p is None:
            p = self.p
        p = jnp.array(p)

        # Detect and transpose if needed
        transposed_input = False
        if y_array.shape[0] == self.n_total and y_array.shape[1] == len(t_array):
            y_array = y_array.T
            transposed_input = True

        # Determine output dimension
        if self.h_funcs:
            n_outputs = len(self.h_funcs)
        else:
            n_outputs = self.n_states

        # Detect and transpose target if needed
        if y_target_array.shape[0] == n_outputs and y_target_array.shape[1] == len(t_array):
            y_target_array = y_target_array.T

        # Convert to JAX arrays
        t_jax = jnp.array(t_array)
        y_jax = jnp.array(y_array)
        y_target_jax = jnp.array(y_target_array)

        # Define loss function that takes y_array as argument
        def loss_fn(y):
            # Compute outputs
            def eval_h_single(t, y_single):
                x = y_single[:self.n_states]
                z = y_single[self.n_states:]
                return self.eval_h_jax(t, x, z, p)

            eval_h_vec = jax.vmap(eval_h_single, in_axes=(0, 0))
            y_pred = eval_h_vec(t_jax, y)

            # Compute loss
            errors = y_pred - y_target_jax
            return jnp.sum(errors ** 2)

        # Compute gradient using JAX
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(y_jax)

        # Convert back to numpy
        grad_np = np.array(grad)

        # Transpose back if input was transposed
        if transposed_input:
            grad_np = grad_np.T

        return grad_np

    def trajectory_loss_gradient_analytical(
        self,
        t_array: np.ndarray,
        y_array: np.ndarray,
        y_target_array: np.ndarray,
        p: np.ndarray = None
    ):
        """
        Compute gradient of trajectory loss analytically using chain rule.

        For L = sum_k ||h(t_k, y_k) - y_target_k||^2:
            dL/dy_k = 2 * (dh/dy_k)^T @ (h(t_k, y_k) - y_target_k)

        Args:
            t_array: time points, shape (N+1,) for N intervals
            y_array: states at time points, shape (N+1, n_total) or (n_total, N+1)
            y_target_array: target outputs at time points, shape (N+1, n_outputs) or (n_outputs, N+1)
            p: parameter vector, if None uses self.p

        Returns:
            grad: gradient dL/dy, shape (N+1, n_total) or (n_total, N+1) matching input

        Notes:
            - More efficient than automatic differentiation for this specific loss
            - Computes Jacobians dh/dy and output errors explicitly
            - Returns JAX array (not NumPy) for JIT compatibility
        """
        # Use default parameters if not provided
        if p is None:
            p = self.p
        p = jnp.array(p)

        # Convert to JAX arrays first
        t_jax = jnp.array(t_array)
        y_jax = jnp.array(y_array)
        y_target_jax = jnp.array(y_target_array)

        # Detect and transpose if needed
        transposed_input = False
        if y_jax.shape[0] == self.n_total and y_jax.shape[1] == t_jax.shape[0]:
            y_jax = y_jax.T
            transposed_input = True

        # Determine output dimension
        if self.h_funcs:
            n_outputs = len(self.h_funcs)
        else:
            n_outputs = self.n_states

        # Detect and transpose target if needed
        if y_target_jax.shape[0] == n_outputs and y_target_jax.shape[1] == t_jax.shape[0]:
            y_target_jax = y_target_jax.T

        N_points = t_jax.shape[0]

        # Compute outputs h(t_k, y_k)
        def eval_h_single(t, y):
            x = y[:self.n_states]
            z = y[self.n_states:]
            return self.eval_h_jax(t, x, z, p)

        eval_h_vec = jax.vmap(eval_h_single, in_axes=(0, 0))
        y_pred = eval_h_vec(t_jax, y_jax)  # shape: (N+1, n_outputs)

        # Compute errors
        errors = y_pred - y_target_jax  # shape: (N+1, n_outputs)

        # Compute Jacobians dh/dy at each time point
        def jac_h_single(t, y):
            x = y[:self.n_states]
            z = y[self.n_states:]
            return jax.jacfwd(lambda yy: self.eval_h_jax(t, yy[:self.n_states], yy[self.n_states:], p))(y)

        jac_h_vec = jax.vmap(jac_h_single, in_axes=(0, 0))
        dh_dy = jac_h_vec(t_jax, y_jax)  # shape: (N+1, n_outputs, n_total)

        # Compute gradient: dL/dy_k = 2 * (dh/dy_k)^T @ error_k
        # For each time point k: grad[k, :] = 2 * dh_dy[k, :, :].T @ errors[k, :]
        def compute_grad_single(dh_dy_k, error_k):
            return 2.0 * dh_dy_k.T @ error_k

        grad_vec = jax.vmap(compute_grad_single, in_axes=(0, 0))
        grad = grad_vec(dh_dy, errors)  # shape: (N+1, n_total)

        # Transpose back if input was transposed
        if transposed_input:
            grad = grad.T

        return grad

    def eval_h_jax(self, t, x, z, p):
        """
        JAX version of eval_h with explicit parameters - evaluates h(t, x, z, p).

        Args:
            t: scalar time
            x: array of differential states
            z: array of algebraic variables
            p: array of parameters

        Returns:
            y = h(t, x, z, p) as JAX array
            If h is not defined, returns x (differential states only)
        """
        if not self.h_funcs:
            # If h is not defined, return the state vector x (identity mapping)
            return x

        # If h_funcs contains state names (default case), just return x
        if self.h_funcs == self.state_names:
            return x

        ns = self._create_jax_eval_namespace_with_params(
            t, x, z, p,
            optimize_indices=self.optimize_indices,
            p_all_default=self.p_all_default
        )

        y_list = []
        for i, expr in enumerate(self.h_funcs):
            val = eval(expr, ns)
            y_list.append(val)

        return jnp.array(y_list)


class DAEOptimizer:
    """
    Iterative optimizer for DAE parameters using adjoint-based gradient descent.

    Minimizes a loss function by adjusting DAE parameters to match an output trajectory.
    Uses the adjoint method for efficient gradient computation.

    Supports multiple time discretization methods through the DAEJacobian class:
    - 'backward_euler': First-order implicit (A-stable, L-stable)
    - 'trapezoidal': Second-order implicit (A-stable) - default
    - 'bdf2' through 'bdf6': Higher-order BDF methods
    """

    def __init__(self, dae_data: dict, dae_solver=None, optimize_params: List[str] = None,
                 loss_type: str = 'sum', method: str = 'trapezoidal',
                 rtol: float = 1e-4, atol: float = 1e-4):
        """
        Initialize the DAE optimizer.

        Args:
            dae_data: Dictionary containing DAE specification
            dae_solver: Optional DAESolver instance. If None, creates a new one.
            optimize_params: List of parameter names to optimize. If None, optimizes all parameters.
                           Example: ['C1', 'C2', 'C3'] to optimize only capacitor parameters
            loss_type: Type of loss function - 'sum' or 'mean'. Default is 'sum'.
                      'sum': loss = sum((y_pred - y_target)^2)
                      'mean': loss = mean((y_pred - y_target)^2)
            method: Time discretization method. One of:
                    'backward_euler', 'trapezoidal' (default), 'bdf2' through 'bdf6'
            rtol: Relative tolerance for DAE solver. Default is 1e-4.
            atol: Absolute tolerance for DAE solver. Default is 1e-4.
        """
        from .dae_solver import DAESolver
        from .adjoint_solver import solve_adjoint_system_jit

        self.dae_data = dae_data
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.jac = DAEJacobian(dae_data, method=method)

        if dae_solver is None:
            self.solver = DAESolver(dae_data)
        else:
            self.solver = dae_solver

        # Validate and store loss type
        if loss_type not in ['sum', 'mean']:
            raise ValueError(f"loss_type must be 'sum' or 'mean', got '{loss_type}'")
        self.loss_type = loss_type

        # Store dimensions
        self.n_params_total = len(self.jac.param_names)
        self.n_states = self.jac.n_states
        self.n_alg = self.jac.n_alg
        self.n_total = self.jac.n_total

        # Initialize all parameters
        self.p_all = jnp.array([p['value'] for p in dae_data['parameters']])

        # Determine which parameters to optimize
        if optimize_params is None:
            # Optimize all parameters
            self.optimize_params = self.jac.param_names.copy()
            self.optimize_indices = list(range(self.n_params_total))
        else:
            # Only optimize specified parameters
            self.optimize_params = optimize_params
            self.optimize_indices = []
            for param_name in optimize_params:
                if param_name in self.jac.param_names:
                    idx = self.jac.param_names.index(param_name)
                    self.optimize_indices.append(idx)
                else:
                    print(f"Warning: Parameter '{param_name}' not found in DAE specification")

        self.n_params = len(self.optimize_indices)

        # Store as JAX array for use in JIT-compiled functions
        self.optimize_indices_jax = jnp.array(self.optimize_indices)

        # Create mask for parameter gradient (only non-zero for optimized params)
        self.param_mask = np.zeros(self.n_params_total, dtype=bool)
        self.param_mask[self.optimize_indices] = True

        # Initialize current values for optimized parameters
        self.p_current = jnp.array([self.p_all[i] for i in self.optimize_indices])

        # Configure the Jacobian object for selective optimization
        # This tells the Jacobian to expect a smaller parameter vector
        self.jac.set_selective_optimization(self.optimize_indices, self.p_all)

        # Create JIT-compiled version of combined gradient computation (steps 2-7)
        # This combines all gradient steps into one JIT-compiled function for better performance
        self._compute_gradient_combined_jit = jit(self._compute_gradient_combined)
        # self._compute_gradient_combined_jit = jit(lambda *args, **kwargs: lax.stop_gradient(self._compute_gradient_combined(*args, **kwargs)))

        # Optimization history
        self.history = {
            'loss': [],
            'gradient_norm': [],
            'params': [],
            'params_all': [],
            'step_size': []
        }

        print(f"DAE Optimizer initialized")
        print(f"  Method: {self.method}")
        print(f"  Total parameters: {self.n_params_total}")
        print(f"  Parameters to optimize: {self.n_params}")
        print(f"  Optimized parameter names: {self.optimize_params}")
        print(f"  Fixed parameters: {[name for i, name in enumerate(self.jac.param_names) if i not in self.optimize_indices]}")
        print(f"  Differential states: {self.n_states}")
        print(f"  Algebraic variables: {self.n_alg}")
        print(f"  Loss type: {self.loss_type}")

    def compute_loss(self, y_pred: jnp.ndarray, y_target: jnp.ndarray) -> float:
        """
        Compute loss function (sum or mean of squared errors).

        Args:
            y_pred: Predicted outputs, shape (n_outputs, n_time_points)
            y_target: Target outputs, shape (n_outputs, n_time_points)

        Returns:
            loss: Scalar loss value
        """
        errors = y_pred - y_target
        squared_errors = errors ** 2

        if self.loss_type == 'sum':
            loss = jnp.sum(squared_errors)
        else:  # mean
            loss = jnp.mean(squared_errors)

        return loss

    def compute_loss_gradient(
        self,
        t_array: np.ndarray,
        y_array: np.ndarray,
        y_target: np.ndarray,
        p: np.ndarray
    ) -> jnp.ndarray:
        """
        Compute gradient of loss with respect to states using JAX.

        For sum loss: L = sum_k ||h(t_k, y_k, p) - y_target_k||^2
            dL/dy_k = 2 * (dh/dy_k)^T @ (h(t_k, y_k, p) - y_target_k)

        For mean loss: L = mean_k ||h(t_k, y_k, p) - y_target_k||^2
            dL/dy_k = (2/N) * (dh/dy_k)^T @ (h(t_k, y_k, p) - y_target_k)
            where N = total number of elements

        Args:
            t_array: time points, shape (N+1,)
            y_array: states [x, z] at time points, shape (N+1, n_total)
            y_target: target outputs, shape (N+1, n_outputs)
            p: parameter vector

        Returns:
            dL_dy: gradient of loss w.r.t. states, shape (N+1, n_total)
        """
        # Use the JIT-compiled gradient function from DAEJacobian
        grad = self.jac.trajectory_loss_gradient_analytical_jit(t_array, y_array, y_target, p)

        # Scale by 1/N if using mean loss
        if self.loss_type == 'mean':
            # Compute total number of elements
            n_outputs = y_target.shape[1] if y_target.shape[0] == len(t_array) else y_target.shape[0]
            n_time = len(t_array)
            N_total = n_outputs * n_time
            grad = grad / N_total

        return grad

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

        This function combines all gradient computation steps into a single JIT-compiled
        function for potentially better performance by reducing Python overhead and
        enabling cross-step optimizations.

        Args:
            t_sol: time points, shape (N+1,)
            y_array: states [x, z] at time points, shape (n_total, N+1)
            y_target_use: target outputs, shape (N+1, n_outputs)
            p_opt_vals_jax: optimized parameter values, shape (n_params_opt,)
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
        # Detect and transpose if needed
        if y_array.shape[0] == self.jac.n_total and y_array.shape[1] == t_sol.shape[0]:
            y_array_for_jac = y_array.T
        else:
            y_array_for_jac = y_array

        N = t_sol.shape[0] - 1  # Number of intervals

        # Prepare arrays for vectorized computation
        t_k = t_sol[:-1]      # t_0, t_1, ..., t_{N-1}
        t_kp1 = t_sol[1:]     # t_1, t_2, ..., t_N
        y_k = y_array_for_jac[:-1]      # y_0, y_1, ..., y_{N-1}
        y_kp1 = y_array_for_jac[1:]     # y_1, y_2, ..., y_N

        # Compute Jacobians in parallel using vmapped functions
        J_prev = self.jac._jac_y_k_vmapped(t_k, t_kp1, y_k, y_kp1, p_opt_vals_jax)      # shape: (N, n_total, n_total)
        J_curr = self.jac._jac_y_kp1_vmapped(t_k, t_kp1, y_k, y_kp1, p_opt_vals_jax)    # shape: (N, n_total, n_total)

        # Step 4: Solve adjoint system (pure JAX version)
        # Implementation of backward substitution for block tridiagonal system
        # J_curr[k]^T * lambda[k] + J_prev[k+1]^T * lambda[k+1] = b[k]
        # Working backwards from k = N-1 to k = 0
        # Note: b = dL_dy_adjoint (already the RHS, no negation needed)

        # Terminal solve: J_curr[N-1]^T * lambda[N-1] = b[N-1]
        lam_N = jnp.linalg.solve(J_curr[-1].T, dL_dy_adjoint[-1])

        # Backward substitution for k = N-2, N-3, ..., 0
        def backward_step(lam_next, inputs):
            """
            Single backward substitution step.

            Solves: J_curr[k]^T @ λ[k] + J_prev[k+1]^T @ λ[k+1] = b[k]
                    => λ[k] = (J_curr[k]^T)^{-1} @ (b[k] - J_prev[k+1]^T @ λ[k+1])
            """
            J_curr_k, J_prev_kp1, b_k = inputs

            # Compute modified RHS: b[k] - J_prev[k+1]^T @ λ[k+1]
            rhs = b_k - J_prev_kp1.T @ lam_next

            # Solve J_curr[k]^T @ λ[k] = rhs
            lam_k = jnp.linalg.solve(J_curr_k.T, rhs)

            return lam_k, lam_k

        # Prepare inputs in reverse order (k = N-2 down to 0)
        inputs_reversed = (
            J_curr[:-1][::-1],      # J_curr[N-2], J_curr[N-3], ..., J_curr[0]
            J_prev[1:][::-1],       # J_prev[N-1], J_prev[N-2], ..., J_prev[1]
            dL_dy_adjoint[:-1][::-1]  # b[N-2], b[N-3], ..., b[0]
        )

        # Run backward scan
        _, lam_reversed = jax.lax.scan(backward_step, lam_N, inputs_reversed)

        # Concatenate: [λ[0], λ[1], ..., λ[N-2], λ[N-1]]
        lambda_adjoint = jnp.concatenate([lam_reversed[::-1], lam_N[None, :]], axis=0)

        # Step 5: Compute parameter Jacobian dR/dp_opt (pure JAX version)
        # Prepare arrays for vectorized computation
        J_param = self.jac._jac_p_vmapped(t_k, t_kp1, y_k, y_kp1, p_opt_vals_jax)  # shape: (N, n_total, n_params_opt)

        # Reshape to (N*n_total, n_params_opt)
        dR_dp_opt = J_param.reshape(N * self.jac.n_total, -1)

        # Step 6: Compute parameter gradient: dL/dp = -(dR/dp)^T @ λ
        lambda_flat = lambda_adjoint.flatten()  # shape: (N*n_total,)
        grad_p_opt = -dR_dp_opt.T @ lambda_flat  # shape: (n_params_opt,)

        # Step 7: Gradient descent update
        p_opt_new = p_opt_vals_jax - step_size * grad_p_opt

        return p_opt_new, grad_p_opt

    def optimization_step(
        self,
        t_array: np.ndarray,
        y_target: np.ndarray,
        p_opt: np.ndarray,
        step_size: float = 0.01
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Perform one optimization step using adjoint-based gradient descent.

        Steps:
        1. Solve DAE with current parameters to get y(p)
        2. Compute loss gradient dL/dy
        3. Solve adjoint system: J^T λ = dL/dy
        4. Compute parameter gradient: dL/dp = -(dR/dp)^T λ
        5. Update parameters: p_new = p - step_size * dL/dp

        Args:
            t_array: time points for trajectory
            y_target: target output trajectory, shape (N+1, n_outputs) or (n_outputs, N+1)
            p_opt: current values of optimized parameters only
            step_size: gradient descent step size

        Returns:
            p_opt_new: updated optimized parameters
            loss: loss value at current parameters
            grad_p_opt: gradient with respect to optimized parameters only
        """
        timings = {}

        # Step 1: Construct full parameter vector (optimized + fixed)
        t1_start = time.time()

        p_all = np.array(self.p_all)  # Start with all default values
        for i, opt_idx in enumerate(self.optimize_indices):
            p_all[opt_idx] = float(p_opt[i])  # Update optimized parameters

        # Update solver parameters
        for i in range(self.n_params_total):
            self.solver.p[i] = float(p_all[i])

        # Reset initial conditions to original values from DAE specification
        # This ensures each iteration starts from the same initial state
        self.solver.x0 = np.array([s['start'] for s in self.dae_data['states']])

        # For algebraic variables, use 'start' if provided, otherwise zero
        self.solver.z0 = np.array([
            a.get('start', 0.0) for a in self.dae_data['alg_vars']
        ])

        # Solve DAE
        t_span = (float(t_array[0]), float(t_array[-1]))
        ncp = len(t_array)
        result = self.solver.solve(t_span=t_span, ncp=ncp, rtol=self.rtol, atol=self.atol)

        # Extract solution
        t_sol = result['t']
        x_sol = result['x']  # shape: (n_states, n_time)
        z_sol = result['z']  # shape: (n_alg, n_time)
        y_pred = result['y']  # shape: (n_outputs, n_time)

        # Combine into full state vector
        y_array = np.vstack([x_sol, z_sol])  # shape: (n_total, n_time)

        # Ensure y_target has correct shape
        if y_target.shape[0] == len(t_array) and y_target.shape[1] != len(t_array):
            # Already in (n_time, n_outputs) format
            y_target_use = y_target
        elif y_target.shape[1] == len(t_array):
            # Transpose from (n_outputs, n_time) to (n_time, n_outputs)
            y_target_use = y_target.T
        else:
            y_target_use = y_target

        # Compute loss
        # y_pred has shape (n_outputs, n_time)
        # y_target_use has shape (n_time, n_outputs)
        # Convert y_target_use to (n_outputs, n_time) to match y_pred
        y_target_for_loss = y_target_use.T  # (n_outputs, n_time)
        loss = self.compute_loss(jnp.array(y_pred), jnp.array(y_target_for_loss))

        # Create JAX array with ONLY optimized parameter values
        # The Jacobian methods will use default values for fixed parameters (configured in set_selective_optimization)
        p_opt_vals_jax = jnp.array([p_all[i] for i in self.optimize_indices])

        t1_end = time.time()
        timings['step_1'] = t1_end - t1_start

        # Step 2: Compute loss gradient dL/dy (this is the "b" vector for adjoint)
        t2_start = time.time()

        # compute_loss_gradient expects y_array in (n_time, n_total) format
        # but y_array is currently (n_total, n_time), so transpose it
        y_array_T = y_array.T  # shape: (n_time, n_total)
        dL_dy = self.compute_loss_gradient(t_sol, y_array_T, y_target_use, p_opt_vals_jax)

        # dL_dy has shape (n_time, n_total), but adjoint system needs (N, n_total)
        # where N = n_time - 1 (excludes initial condition which is fixed)
        dL_dy_adjoint = jnp.array(dL_dy[1:, :])  # shape: (N, n_total)

        t2_end = time.time()
        timings['step_2'] = t2_end - t2_start

        # Step 3: Compute Jacobian blocks
        t3_start = time.time()

        J_prev_list, J_curr_list = self.jac.compute_jacobian_blocks_jit(t_sol, y_array, p_opt_vals_jax)

        t3_end = time.time()
        timings['step_3'] = t3_end - t3_start

        # Step 4: Solve adjoint system
        t4_start = time.time()

        from .adjoint_solver import solve_adjoint_system_jit
        lambda_adjoint = solve_adjoint_system_jit(J_prev_list, J_curr_list, dL_dy_adjoint)

        t4_end = time.time()
        timings['step_4'] = t4_end - t4_start

        # Step 5: Compute parameter Jacobian dR/dp_opt (only for optimized parameters)
        t5_start = time.time()

        # Compute parameter Jacobian - this now computes w.r.t. optimized params only
        J_param_list = self.jac.compute_parameter_jacobian_jit(t_sol, y_array, p_opt_vals_jax)

        # Stack into a single matrix: dR/dp_opt with shape (N*n_total, n_params_opt)
        dR_dp_opt = np.vstack(J_param_list)  # Each J_param_list[i] has shape (n_total, n_params_opt)

        t5_end = time.time()
        timings['step_5'] = t5_end - t5_start

        # Step 6: Compute parameter gradient: dL/dp = -(dR/dp)^T @ λ
        t6_start = time.time()

        # lambda_adjoint has shape (N, n_total)
        # Flatten lambda to (N*n_total,)
        lambda_flat = lambda_adjoint.flatten()

        # Gradient computation for optimized parameters only
        grad_p_opt = -dR_dp_opt.T @ lambda_flat  # shape: (n_params_opt,)
        grad_p_opt = np.array(grad_p_opt)

        t6_end = time.time()
        timings['step_6'] = t6_end - t6_start

        # Step 7: Gradient descent update (only for optimized parameters)
        t7_start = time.time()

        p_opt_new = p_opt - step_size * grad_p_opt

        t7_end = time.time()
        timings['step_7'] = t7_end - t7_start

        return np.array(p_opt_new), float(loss), np.array(grad_p_opt), timings

    def optimization_step_combined(
        self,
        t_array: np.ndarray,
        y_target: np.ndarray,
        p_opt: np.ndarray,
        step_size: float = 0.01,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Perform one optimization step using COMBINED JIT-compiled gradient computation.

        This version combines steps 2-7 into a single JIT-compiled function for better
        performance by reducing Python overhead and enabling JAX to optimize across steps.

        Steps:
        1. Solve DAE with current parameters to get y(p) [NOT JIT-compiled]
        2-7. Combined JIT-compiled gradient computation:
             - Compute loss gradient dL/dy
             - Compute Jacobian blocks
             - Solve adjoint system
             - Compute parameter Jacobian
             - Compute parameter gradient
             - Update parameters

        Args:
            t_array: time points for trajectory
            y_target: target output trajectory, shape (N+1, n_outputs) or (n_outputs, N+1)
            p_opt: current values of optimized parameters only
            step_size: gradient descent step size

        Returns:
            p_opt_new: updated optimized parameters
            loss: loss value at current parameters
            grad_p_opt: gradient with respect to optimized parameters only
        """
        timings = {}

        # Step 1: Construct full parameter vector and solve DAE (NOT JIT-compiled)
        t1_start = time.time()

        p_all = np.array(self.p_all)  # Start with all default values
        for i, opt_idx in enumerate(self.optimize_indices):
            p_all[opt_idx] = float(p_opt[i])  # Update optimized parameters

        # Update solver parameters (must use numpy/float for DAE solver)
        for i in range(self.n_params_total):
            self.solver.p[i] = float(p_all[i])

        # Reset initial conditions to original values from DAE specification
        self.solver.x0 = np.array([s['start'] for s in self.dae_data['states']])
        self.solver.z0 = np.array([
            a.get('start', 0.0) for a in self.dae_data['alg_vars']
        ])

        # Solve DAE
        t_span = (float(t_array[0]), float(t_array[-1]))
        ncp = len(t_array)
        result = self.solver.solve(t_span=t_span, ncp=ncp, rtol=self.rtol, atol=self.atol)

        # Extract solution (numpy arrays from DAE solver)
        t_sol = result['t']
        x_sol = result['x']  # shape: (n_states, n_time)
        z_sol = result['z']  # shape: (n_alg, n_time)
        y_pred = result['y']  # shape: (n_outputs, n_time)

        # Combine into full state vector
        y_array = np.vstack([x_sol, z_sol])  # shape: (n_total, n_time)

        # Ensure y_target has correct shape
        if y_target.shape[0] == len(t_array) and y_target.shape[1] != len(t_array):
            y_target_use = y_target
        elif y_target.shape[1] == len(t_array):
            y_target_use = y_target.T
        else:
            y_target_use = y_target

        # Compute loss
        y_target_for_loss = y_target_use.T  # (n_outputs, n_time)
        loss = self.compute_loss(jnp.array(y_pred), jnp.array(y_target_for_loss))

        t1_end = time.time()
        timings['step_1_dae_solve'] = t1_end - t1_start

        # Steps 2-7: Combined JIT-compiled gradient computation
        t27_start = time.time()

        # Convert numpy arrays to JAX arrays for JIT function
        t_sol_jax = jnp.array(t_sol)
        y_array_jax = jnp.array(y_array)
        y_target_use_jax = jnp.array(y_target_use)
        p_opt_vals_jax = jnp.array([p_all[i] for i in self.optimize_indices])

        # Call combined JIT-compiled function
        p_opt_new_jax, grad_p_opt_jax = self._compute_gradient_combined_jit(
            t_sol_jax,
            y_array_jax,
            y_target_use_jax,
            p_opt_vals_jax,
            step_size
        )

        # Convert back to numpy for return
        p_opt_new = np.array(p_opt_new_jax)
        grad_p_opt = np.array(grad_p_opt_jax)

        t27_end = time.time()
        timings['steps_2_7_combined_jit'] = t27_end - t27_start

        return p_opt_new, float(loss), grad_p_opt, timings

    def optimize(
        self,
        t_array: np.ndarray,
        y_target: np.ndarray,
        p_init: np.ndarray = None,
        n_iterations: int = 100,
        step_size: float = 0.01,
        tol: float = 1e-6,
        combined: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Optimize DAE parameters to minimize loss.

        Args:
            t_array: time points for trajectory
            y_target: target output trajectory
            p_init: initial parameter values (if None, uses current values)
            n_iterations: maximum number of iterations
            step_size: gradient descent step size
            tol: convergence tolerance on gradient norm
            verbose: whether to print progress

        Returns:
            Dictionary containing:
                - p_opt: optimized parameters
                - loss_final: final loss value
                - history: optimization history
                - converged: whether optimization converged
        """
        if verbose:
            print("\n" + "=" * 80)
            print("Starting DAE Parameter Optimization")
            print("=" * 80)
            print(f"  Iterations: {n_iterations}")
            print(f"  Step size: {step_size}")
            print(f"  Tolerance: {tol}")
            print(f"  Target trajectory points: {len(t_array)}")

        # Initialize parameters
        if p_init is not None:
            p = jnp.array(p_init)
        else:
            p = self.p_current

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

        # Optimization loop
        for iteration in range(n_iterations):
            t_start = time.time()

            # Perform optimization step
            p_new, loss, grad_p, step_timings = self.optimization_step_combined(
                t_array, y_target, p, step_size
            ) if combined else self.optimization_step(
                t_array, y_target, p, step_size
            )

            t_end = time.time()
            iter_time = t_end - t_start

            # Compute gradient norm
            grad_norm = float(jnp.linalg.norm(grad_p))

            # Construct full parameter vector for history
            p_all_current = np.array(self.p_all)
            for i, opt_idx in enumerate(self.optimize_indices):
                p_all_current[opt_idx] = float(p[i])

            # Store history
            self.history['loss'].append(loss)
            self.history['gradient_norm'].append(grad_norm)
            self.history['params'].append(np.array(p))  # Only optimized params
            self.history['params_all'].append(p_all_current)  # All params
            self.history['step_size'].append(step_size)
            self.history['time_per_iter'].append(iter_time)

            if verbose and (iteration % 1 == 0):  # Print every iteration for now if verbose
                print(f"\nIteration {iteration:4d} ({iter_time:.3f}s):")
                print(f"  Loss:          {loss:.6e}")
                print(f"  Gradient norm: {grad_norm:.6e}")
                
                # Print timing breakdown
                t_forward = step_timings.get('step_1', 0.0) + step_timings.get('step_1_dae_solve', 0.0)
                t_adjoint = step_timings.get('steps_2_7_combined_jit', 0.0)
                if t_adjoint == 0:
                    t_adjoint = sum(step_timings.get(f'step_{i}', 0.0) for i in range(2, 8))
                
                print(f"  Forward solve: {t_forward*1000:.1f}ms")
                print(f"  Adjoint step:  {t_adjoint*1000:.1f}ms")
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
                    print(f"\n✓ Converged at iteration {iteration}")
                    print(f"  Final gradient norm: {grad_norm:.6e}")
                break

            # Update parameters
            p = jnp.array(p_new)

        # Store final parameters
        self.p_current = p

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
                times_no_first = times[1:]
                avg_time = np.mean(times_no_first)
                std_time = np.std(times_no_first)
                print(f"\n  Timing statistics:")
                print(f"    Total time:              {total_time:.3f}s")
                print(f"    First iteration:         {times[0]:.3f}s")
                print(f"    Avg time/iter:           {avg_time*1000:.3f}ms")
                print(f"    Std time/iter:           {std_time*1000:.3f}ms")
                print(f"    Min time/iter:           {min(times_no_first)*1000:.3f}ms")
                print(f"    Max time/iter:           {max(times_no_first)*1000:.3f}ms")
            else:
                print(f"\n  Total time: {total_time:.3f}s")

            # Final full parameter vector
            p_all_final = np.array(self.p_all)
            for i, opt_idx in enumerate(self.optimize_indices):
                p_all_final[opt_idx] = float(p[i])

            print(f"\n  Optimized parameters (final values):")
            for i, (name, val) in enumerate(zip(self.optimize_params, p)):
                print(f"    {name:20s} = {float(val):.6f}")

        # Construct final full parameter vector
        p_all_final = np.array(self.p_all)
        for i, opt_idx in enumerate(self.optimize_indices):
            p_all_final[opt_idx] = float(p[i])

        return {
            'p_opt': np.array(p),  # Only optimized parameters
            'p_all': p_all_final,  # All parameters (optimized + fixed)
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
            print("Matplotlib not available, skipping plot")
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

        # Parameters (only optimized ones)
        ax = axes[1, 0]
        params_array = np.array(self.history['params'])
        for i in range(params_array.shape[1]):
            ax.plot(params_array[:, i], label=f'{self.optimize_params[i]}', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Value')
        ax.set_title('Parameter Evolution (Optimized)')
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
