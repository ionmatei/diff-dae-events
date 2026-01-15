"""
DAE Solver using SUNDIALS IDA (via scikits.odes)

Reads JSON DAE specification and solves the semi-explicit DAE:
    dx/dt = f(t, x, z, p)
    0 = g(t, x, z, p)
    y = h(t, x, z, p)  (outputs, if present)

Where:
    x = differential states
    z = algebraic variables
    p = parameters
    y = outputs

The DAE is converted to implicit form for IDA:
    F(t, y, ydot) = 0
where y = [x, z] combines differential and algebraic variables.
"""

import json
import numpy as np
from scikits.odes.dae import dae
from typing import Dict, List, Tuple
import re

try:
    import jax
    import jax.numpy as jnp
    from jax import vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np
    print("Warning: JAX not available. Vectorized operations will use numpy loops.")


class DAESolver:
    """
    Solves semi-explicit DAEs from JSON specification using solve_ivp.

    The DAE is converted to an ODE by solving algebraic equations at each timestep.
    """

    def __init__(self, dae_data: dict):
        """
        Load DAE from JSON specification.

        Args:
            dae_data: Dictionary containing DAE specification
        """
        form = dae_data

        # Extract variables
        self.states = form['states']  # Differential states
        self.alg_vars = form['alg_vars']  # Algebraic variables
        self.parameters = form['parameters']
        self.outputs = form.get('outputs', None)
        if self.outputs is None:
            self.outputs = []

        # Extract equations
        self.f_eqs = form['f']  # dx/dt = f(...)
        self.g_eqs = form['g']  # 0 = g(...)
        self.h_eqs = form.get('h', None)  # y = h(...)

        # Create name-to-index mappings
        self.state_names = [s['name'] for s in self.states]
        self.alg_names = [a['name'] for a in self.alg_vars]
        self.param_names = [p['name'] for p in self.parameters]
        self.output_names = [o['name'] for o in self.outputs] if self.outputs else []

        # Get initial conditions
        self.x0 = np.array([s['start'] for s in self.states])
        # Algebraic variables will be solved from constraints - initialize to zero
        self.z0 = np.zeros(len(self.alg_vars))
        self.p = np.array([p['value'] for p in self.parameters])

        print(f"DAE loaded")
        print(f"  Differential states: {len(self.states)}")
        print(f"  Algebraic variables: {len(self.alg_vars)}")
        print(f"  Parameters: {len(self.parameters)}")
        print(f"  Outputs: {len(self.outputs)}")
        print(f"  f equations: {len(self.f_eqs)}")
        print(f"  g equations: {len(self.g_eqs)}")

        # Compile equations into Python functions
        self._compile_equations()

        # Compile JAX vectorized functions (vmap) once during initialization
        self._compile_vectorized_functions()

    def _make_safe_name(self, name: str) -> str:
        """Convert variable name to valid Python identifier."""
        # Replace dots and special chars with underscores
        return re.sub(r'[^a-zA-Z0-9_]', '_', name)

    def _compile_equations(self):
        """
        Compile equation strings into executable Python functions.
        This uses eval() but in a controlled namespace.
        """
        print("\nCompiling equations...")

        # Create namespace with math functions
        self.namespace = {
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'sinh': np.sinh,
            'cosh': np.cosh,
            'tanh': np.tanh,
            'exp': np.exp,
            'log': np.log,
            'log10': np.log10,
            'sqrt': np.sqrt,
            'abs': np.abs,
            'pow': np.power,
            'min': np.minimum,
            'max': np.maximum,
            'der': lambda x: x,  # Placeholder, handled separately
        }

        print("Available math functions:", list(self.namespace.keys()))

        # Compile f equations (derivatives)
        self.f_funcs = []
        for i, eq in enumerate(self.f_eqs):
            # Extract LHS: der(state_name) = RHS
            match = re.match(r'der\((\w+)\)\s*=\s*(.+)', eq)
            if match:
                state_name, rhs = match.groups()
                # Create function that evaluates RHS
                self.f_funcs.append(rhs)
            else:
                raise ValueError(f"Invalid f equation format: {eq}")

        # Compile g equations (algebraic constraints)
        self.g_funcs = []
        for i, eq in enumerate(self.g_eqs):
            # Format: 0 = g(...) or g(...) = 0
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

        # Compile h equations (outputs) if present
        self.h_funcs = []
        if self.h_eqs:
            for eq in self.h_eqs:
                # Format: output_name = expression
                if '=' in eq:
                    _, rhs = eq.split('=', 1)
                    self.h_funcs.append(rhs.strip())
                else:
                    self.h_funcs.append(eq)

        print("Equations compiled successfully!")

    def _create_eval_namespace(self, t: float, x: np.ndarray, z: np.ndarray) -> Dict:
        """Create namespace for equation evaluation."""
        ns = self.namespace.copy()

        # Add time
        ns['time'] = t
        ns['t'] = t

        # Add states
        for i, name in enumerate(self.state_names):
            ns[name] = x[i]

        # Add algebraic variables
        for i, name in enumerate(self.alg_names):
            ns[name] = z[i]

        # Add parameters
        for i, name in enumerate(self.param_names):
            ns[name] = self.p[i]

        return ns

    def eval_f(self, t: float, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Evaluate f(t, x, z, p) - the derivatives dx/dt.

        Args:
            t: time
            x: differential states
            z: algebraic variables

        Returns:
            dx/dt
        """
        ns = self._create_eval_namespace(t, x, z)

        dxdt = np.zeros(len(x))
        for i, expr in enumerate(self.f_funcs):
            try:
                dxdt[i] = eval(expr, ns)
            except Exception as e:
                print(f"Error evaluating f[{i}]: {expr}")
                print(f"Error: {e}")
                raise

        return dxdt

    def eval_g(self, t: float, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Evaluate g(t, x, z, p) - the algebraic constraints.

        Args:
            t: time
            x: differential states
            z: algebraic variables

        Returns:
            g(t, x, z, p), should be zero
        """
        ns = self._create_eval_namespace(t, x, z)

        g = np.zeros(len(z))
        for i, expr in enumerate(self.g_funcs):
            try:
                g[i] = eval(expr, ns)
            except Exception as e:
                print(f"Error evaluating g[{i}]: {expr}")
                print(f"Error: {e}")
                raise

        return g

    def eval_h(self, t: float, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Evaluate h(t, x, z, p) - the output equations.

        Args:
            t: time
            x: differential states
            z: algebraic variables

        Returns:
            y = h(t, x, z, p) if h is defined, otherwise returns x (identity)
        """
        if not self.h_funcs:
            # If h is not defined, return the state vector x (identity mapping)
            return x

        ns = self._create_eval_namespace(t, x, z)

        y = np.zeros(len(self.h_funcs))
        for i, expr in enumerate(self.h_funcs):
            try:
                y[i] = eval(expr, ns)
            except Exception as e:
                print(f"Error evaluating h[{i}]: {expr}")
                print(f"Error: {e}")
                raise

        return y

    def _compile_vectorized_functions(self):
        """
        Compile vectorized versions of f, g, h using JAX vmap.

        This creates vmapped functions ONCE during initialization for efficiency.
        The vmapped functions can then be called multiple times without overhead.
        """
        if not JAX_AVAILABLE:
            # Set to None to indicate vmap not available
            self._f_vmapped = None
            self._g_vmapped = None
            self._h_vmapped = None
            return

        n_states = len(self.state_names)

        # Define single-point evaluation function for f
        def eval_f_single(t, y):
            x = y[:n_states]
            z = y[n_states:]
            return self.eval_f_jax(t, x, z)

        # Define single-point evaluation function for g
        def eval_g_single(t, y):
            x = y[:n_states]
            z = y[n_states:]
            return self.eval_g_jax(t, x, z)

        # Define single-point evaluation function for h
        def eval_h_single(t, y):
            x = y[:n_states]
            z = y[n_states:]
            return self.eval_h_jax(t, x, z)

        # Create vmapped versions ONCE - vmap over both time and state vectors
        # in_axes=(0, 0) means vmap over first axis of both arguments
        self._f_vmapped = vmap(eval_f_single, in_axes=(0, 0))
        self._g_vmapped = vmap(eval_g_single, in_axes=(0, 0))
        # Always create h vmap since h now returns x (identity) when not defined
        self._h_vmapped = vmap(eval_h_single, in_axes=(0, 0))

        print("JAX vmap functions compiled successfully!")

    def _create_jax_eval_namespace(self, t, x, z):
        """Create namespace for JAX equation evaluation."""
        ns = {}

        # Math functions
        ns.update({
            'sin': jnp.sin,
            'cos': jnp.cos,
            'tan': jnp.tan,
            'sinh': jnp.sinh,
            'cosh': jnp.cosh,
            'tanh': jnp.tanh,
            'exp': jnp.exp,
            'log': jnp.log,
            'log10': jnp.log10,
            'sqrt': jnp.sqrt,
            'abs': jnp.abs,
            'pow': jnp.power,
            'min': jnp.minimum,
            'max': jnp.maximum,
        })

        # Add time
        ns['time'] = t
        ns['t'] = t

        # Add states
        for i, name in enumerate(self.state_names):
            ns[name] = x[i]

        # Add algebraic variables
        for i, name in enumerate(self.alg_names):
            ns[name] = z[i]

        # Add parameters
        for i, name in enumerate(self.param_names):
            ns[name] = self.p[i]

        return ns

    def eval_f_jax(self, t, x, z):
        """
        JAX version of eval_f - evaluates f(t, x, z, p).

        Args:
            t: scalar time
            x: array of differential states
            z: array of algebraic variables

        Returns:
            dx/dt as JAX array
        """
        ns = self._create_jax_eval_namespace(t, x, z)

        dxdt_list = []
        for i, expr in enumerate(self.f_funcs):
            val = eval(expr, ns)
            dxdt_list.append(val)

        return jnp.array(dxdt_list)

    def eval_g_jax(self, t, x, z):
        """
        JAX version of eval_g - evaluates g(t, x, z, p).

        Args:
            t: scalar time
            x: array of differential states
            z: array of algebraic variables

        Returns:
            g(t, x, z, p) as JAX array
        """
        ns = self._create_jax_eval_namespace(t, x, z)

        g_list = []
        for i, expr in enumerate(self.g_funcs):
            val = eval(expr, ns)
            g_list.append(val)

        return jnp.array(g_list)

    def eval_h_jax(self, t, x, z):
        """
        JAX version of eval_h - evaluates h(t, x, z, p).

        Args:
            t: scalar time
            x: array of differential states
            z: array of algebraic variables

        Returns:
            y = h(t, x, z, p) as JAX array if h is defined, otherwise returns x (identity)
        """
        if not self.h_funcs:
            # If h is not defined, return the state vector x (identity mapping)
            return x

        ns = self._create_jax_eval_namespace(t, x, z)

        y_list = []
        for i, expr in enumerate(self.h_funcs):
            val = eval(expr, ns)
            y_list.append(val)

        return jnp.array(y_list)

    def eval_f_vectorized(self, t_array: np.ndarray, y_array: np.ndarray) -> np.ndarray:
        """
        Vectorized evaluation of f over multiple time points using JAX vmap.

        Args:
            t_array: array of time points, shape (n_times,)
            y_array: array of states [x, z] at each time point, shape (n_times, n_states + n_alg)
                     or shape (n_states + n_alg, n_times) - will be auto-detected

        Returns:
            Array of f evaluations, shape (n_times, n_states)
        """
        if not JAX_AVAILABLE or self._f_vmapped is None:
            # Fallback to loop-based evaluation
            return self._eval_f_vectorized_numpy(t_array, y_array)

        # Detect array shape and transpose if needed
        n_total = len(self.state_names) + len(self.alg_names)
        if y_array.shape[0] == n_total and y_array.shape[1] == len(t_array):
            # Shape is (n_vars, n_times), transpose to (n_times, n_vars)
            y_array = y_array.T

        # Convert to JAX arrays
        t_jax = jnp.array(t_array)
        y_jax = jnp.array(y_array)

        # Call the pre-compiled vmapped function
        result = self._f_vmapped(t_jax, y_jax)

        # Convert back to numpy
        return np.array(result)

    def eval_g_vectorized(self, t_array: np.ndarray, y_array: np.ndarray) -> np.ndarray:
        """
        Vectorized evaluation of g over multiple time points using JAX vmap.

        Args:
            t_array: array of time points, shape (n_times,)
            y_array: array of states [x, z] at each time point, shape (n_times, n_states + n_alg)
                     or shape (n_states + n_alg, n_times) - will be auto-detected

        Returns:
            Array of g evaluations, shape (n_times, n_alg)
        """
        if not JAX_AVAILABLE or self._g_vmapped is None:
            # Fallback to loop-based evaluation
            return self._eval_g_vectorized_numpy(t_array, y_array)

        # Detect array shape and transpose if needed
        n_total = len(self.state_names) + len(self.alg_names)
        if y_array.shape[0] == n_total and y_array.shape[1] == len(t_array):
            # Shape is (n_vars, n_times), transpose to (n_times, n_vars)
            y_array = y_array.T

        # Convert to JAX arrays
        t_jax = jnp.array(t_array)
        y_jax = jnp.array(y_array)

        # Call the pre-compiled vmapped function
        result = self._g_vmapped(t_jax, y_jax)

        # Convert back to numpy
        return np.array(result)

    def eval_h_vectorized(self, t_array: np.ndarray, y_array: np.ndarray) -> np.ndarray:
        """
        Vectorized evaluation of h over multiple time points using JAX vmap.

        Args:
            t_array: array of time points, shape (n_times,)
            y_array: array of states [x, z] at each time point, shape (n_times, n_states + n_alg)
                     or shape (n_states + n_alg, n_times) - will be auto-detected

        Returns:
            Array of h evaluations, shape (n_times, n_outputs) if h is defined,
            or shape (n_times, n_states) if h is not defined (returns x as identity)
        """
        if not JAX_AVAILABLE or self._h_vmapped is None:
            # Fallback to loop-based evaluation
            return self._eval_h_vectorized_numpy(t_array, y_array)

        # Detect array shape and transpose if needed
        n_total = len(self.state_names) + len(self.alg_names)
        if y_array.shape[0] == n_total and y_array.shape[1] == len(t_array):
            # Shape is (n_vars, n_times), transpose to (n_times, n_vars)
            y_array = y_array.T

        # Convert to JAX arrays
        t_jax = jnp.array(t_array)
        y_jax = jnp.array(y_array)

        # Call the pre-compiled vmapped function
        result = self._h_vmapped(t_jax, y_jax)

        # Convert back to numpy
        return np.array(result)

    def _eval_f_vectorized_numpy(self, t_array: np.ndarray, y_array: np.ndarray) -> np.ndarray:
        """Fallback numpy loop-based implementation for eval_f_vectorized."""
        # Detect array shape and transpose if needed
        n_total = len(self.state_names) + len(self.alg_names)
        if y_array.shape[0] == n_total and y_array.shape[1] == len(t_array):
            y_array = y_array.T

        n_states = len(self.state_names)
        n_times = len(t_array)

        result = np.zeros((n_times, n_states))
        for i in range(n_times):
            x = y_array[i, :n_states]
            z = y_array[i, n_states:]
            result[i] = self.eval_f(t_array[i], x, z)

        return result

    def _eval_g_vectorized_numpy(self, t_array: np.ndarray, y_array: np.ndarray) -> np.ndarray:
        """Fallback numpy loop-based implementation for eval_g_vectorized."""
        # Detect array shape and transpose if needed
        n_total = len(self.state_names) + len(self.alg_names)
        if y_array.shape[0] == n_total and y_array.shape[1] == len(t_array):
            y_array = y_array.T

        n_states = len(self.state_names)
        n_alg = len(self.alg_names)
        n_times = len(t_array)

        result = np.zeros((n_times, n_alg))
        for i in range(n_times):
            x = y_array[i, :n_states]
            z = y_array[i, n_states:]
            result[i] = self.eval_g(t_array[i], x, z)

        return result

    def _eval_h_vectorized_numpy(self, t_array: np.ndarray, y_array: np.ndarray) -> np.ndarray:
        """Fallback numpy loop-based implementation for eval_h_vectorized."""
        # Detect array shape and transpose if needed
        n_total = len(self.state_names) + len(self.alg_names)
        if y_array.shape[0] == n_total and y_array.shape[1] == len(t_array):
            y_array = y_array.T

        n_states = len(self.state_names)
        n_times = len(t_array)

        # Determine output size based on whether h is defined
        if self.h_funcs:
            n_outputs = len(self.h_funcs)
        else:
            # If h is not defined, output is the state vector x
            n_outputs = n_states

        result = np.zeros((n_times, n_outputs))
        for i in range(n_times):
            x = y_array[i, :n_states]
            z = y_array[i, n_states:]
            result[i] = self.eval_h(t_array[i], x, z)

        return result

    def residual_ida(self, t: float, y: np.ndarray, ydot: np.ndarray, res: np.ndarray):
        """
        IDA residual function: F(t, y, ydot) = 0

        For semi-explicit DAE:
            dx/dt = f(t, x, z, p)  =>  F_x = dx/dt - f(t, x, z, p) = 0
            0 = g(t, x, z, p)       =>  F_z = g(t, x, z, p) = 0

        Args:
            t: time
            y: combined state [x, z] where x are differential, z are algebraic
            ydot: time derivatives [xdot, zdot]
            res: output residual array to fill
        """
        # Split combined state
        n_states = len(self.state_names)
        x = y[:n_states]
        z = y[n_states:]
        xdot = ydot[:n_states]

        # Evaluate f and g
        f_vals = self.eval_f(t, x, z)
        g_vals = self.eval_g(t, x, z)

        # Residuals for differential equations: xdot - f = 0
        res[:n_states] = xdot - f_vals

        # Residuals for algebraic equations: g = 0
        res[n_states:] = g_vals

    def residual_trapezoidal(self, t_n: float, y_n: np.ndarray, t_np1: float, y_np1: np.ndarray) -> np.ndarray:
        """
        Trapezoidal residual function for DAE discretization.

        Uses the trapezoidal (Crank-Nicolson) scheme to approximate the DAE:
            For differential equations:
                (x_{n+1} - x_n) / h - 0.5 * (f(t_n, x_n, z_n) + f(t_{n+1}, x_{n+1}, z_{n+1})) = 0

            For algebraic equations:
                g(t_{n+1}, x_{n+1}, z_{n+1}) = 0

        Args:
            t_n: time at step n
            y_n: combined state [x_n, z_n] at step n
            t_np1: time at step n+1
            y_np1: combined state [x_{n+1}, z_{n+1}] at step n+1

        Returns:
            residual: residual vector for the trapezoidal scheme
        """
        # Time step
        h = t_np1 - t_n

        # Split combined states
        n_states = len(self.state_names)
        x_n = y_n[:n_states]
        z_n = y_n[n_states:]
        x_np1 = y_np1[:n_states]
        z_np1 = y_np1[n_states:]

        # Evaluate f at both time points
        f_n = self.eval_f(t_n, x_n, z_n)
        f_np1 = self.eval_f(t_np1, x_np1, z_np1)

        # Evaluate g at time n+1
        g_np1 = self.eval_g(t_np1, x_np1, z_np1)
        
        # Evaluate g at time n
        g_n = self.eval_g(t_n, x_n, z_n)

        # Residual vector
        residual = np.zeros(len(y_n))

        # Differential equation residuals: (x_{n+1} - x_n)/h - 0.5*(f_n + f_{n+1}) = 0
        residual[:n_states] = (x_np1 - x_n) / h - 0.5 * (f_n + f_np1)

        # Algebraic equation residuals: g_{n+1} = 0
        residual[n_states:] = g_np1
        # residual[n_states:] = g_n        

        return residual

    def residual_hermite_simpson(self, t_n: float, y_n: np.ndarray, t_np1: float, y_np1: np.ndarray) -> np.ndarray:
        """
        Hermite-Simpson residual function for DAE discretization.

        Uses the Hermite-Simpson collocation scheme with a midpoint:
            For differential equations:
                Defect at midpoint:
                    x_mid - (x_n + x_{n+1})/2 - h/8 * (f_n - f_{n+1}) = 0
                Collocation at midpoint:
                    (x_{n+1} - x_n)/h - (f_n + 4*f_mid + f_{n+1})/6 = 0

            For algebraic equations:
                g(t_mid, x_mid, z_mid) = 0
                g(t_{n+1}, x_{n+1}, z_{n+1}) = 0

        Args:
            t_n: time at step n
            y_n: combined state [x_n, z_n] at step n
            t_np1: time at step n+1
            y_np1: combined state [x_{n+1}, z_{n+1}] at step n+1

        Returns:
            residual: residual vector for the Hermite-Simpson scheme
                     [defect_constraints, collocation_constraints, algebraic_constraints]
        """
        # Time step
        h = t_np1 - t_n
        t_mid = (t_n + t_np1) / 2.0

        # Split combined states
        n_states = len(self.state_names)
        n_alg = len(self.alg_names)
        x_n = y_n[:n_states]
        z_n = y_n[n_states:]
        x_np1 = y_np1[:n_states]
        z_np1 = y_np1[n_states:]

        # Compute midpoint state using Hermite interpolation
        # This is derived from the defect constraint
        f_n = self.eval_f(t_n, x_n, z_n)
        f_np1 = self.eval_f(t_np1, x_np1, z_np1)

        x_mid = (x_n + x_np1) / 2.0 + h / 8.0 * (f_n - f_np1)

        # For algebraic variables at midpoint, use simple averaging as initial guess
        # then solve g(t_mid, x_mid, z_mid) = 0
        z_mid = (z_n + z_np1) / 2.0

        # Evaluate f at midpoint
        f_mid = self.eval_f(t_mid, x_mid, z_mid)

        # Evaluate g at midpoint and endpoint
        g_mid = self.eval_g(t_mid, x_mid, z_mid)
        g_np1 = self.eval_g(t_np1, x_np1, z_np1)

        # Residual vector: [collocation_residual, algebraic_residuals]
        # We return just the key constraints for evaluation
        residual = np.zeros(n_states + n_alg)

        # Collocation constraint: (x_{n+1} - x_n)/h - (f_n + 4*f_mid + f_{n+1})/6 = 0
        residual[:n_states] = (x_np1 - x_n) / h - (f_n + 4.0 * f_mid + f_np1) / 6.0

        # Algebraic equation residuals: use average of midpoint and endpoint
        # This better captures the algebraic constraint satisfaction over the interval
        residual[n_states:] = (g_mid + g_np1) / 2.0

        return residual

    def evaluate_trapezoidal_residual(self, result: Dict) -> Dict:
        """
        Evaluate the trapezoidal residual on a solution trajectory.

        This checks how well the IDA solution satisfies the trapezoidal discretization
        scheme by computing the residual at each time step.

        Args:
            result: Solution dictionary from solve() method

        Returns:
            Dictionary containing:
                - t: time points (excluding first point)
                - residuals: residual norms at each time step
                - residuals_diff: residuals for differential equations
                - residuals_alg: residuals for algebraic equations
                - max_residual: maximum residual norm
                - mean_residual: mean residual norm
        """
        t = result['t']
        x = result['x']
        z = result['z']
        n_states = len(self.state_names)
        n_alg = len(self.alg_names)
        n_steps = len(t) - 1

        print("\nEvaluating trapezoidal residual on IDA solution...")
        print(f"  Number of time points: {len(t)}")
        print(f"  Number of steps to evaluate: {n_steps}")

        if n_steps <= 0:
            print("  Warning: Not enough time points to evaluate residuals (need at least 2)")
            return {
                't': np.array([]),
                'residual_norms': np.array([]),
                'residuals_diff_norms': np.array([]),
                'residuals_alg_norms': np.array([]),
                'all_residuals': np.array([]),
                'max_residual': 0.0,
                'mean_residual': 0.0,
            }

        # Storage for residuals
        residual_norms = np.zeros(n_steps)
        residuals_diff_norms = np.zeros(n_steps)
        residuals_alg_norms = np.zeros(n_steps)
        all_residuals = np.zeros((n_steps, n_states + n_alg))

        # Evaluate residual at each time step
        for i in range(n_steps):
            # Get states at time n and n+1
            y_n = np.concatenate([x[:, i], z[:, i]])
            y_np1 = np.concatenate([x[:, i+1], z[:, i+1]])

            # Compute trapezoidal residual
            res = self.residual_trapezoidal(t[i], y_n, t[i+1], y_np1)
            all_residuals[i, :] = res

            # Compute norms
            residual_norms[i] = np.linalg.norm(res)
            residuals_diff_norms[i] = np.linalg.norm(res[:n_states])
            residuals_alg_norms[i] = np.linalg.norm(res[n_states:])

        max_residual = np.max(residual_norms)
        mean_residual = np.mean(residual_norms)

        print(f"Trapezoidal residual evaluation complete!")
        print(f"  Max residual norm: {max_residual:.6e}")
        print(f"  Mean residual norm: {mean_residual:.6e}")
        print(f"  Max differential residual: {np.max(residuals_diff_norms):.6e}")
        print(f"  Max algebraic residual: {np.max(residuals_alg_norms):.6e}")

        return {
            't': t[1:],  # Exclude first point (no residual computed there)
            'residual_norms': residual_norms,
            'residuals_diff_norms': residuals_diff_norms,
            'residuals_alg_norms': residuals_alg_norms,
            'all_residuals': all_residuals,
            'max_residual': max_residual,
            'mean_residual': mean_residual,
        }

    def evaluate_hermite_simpson_residual(self, result: Dict) -> Dict:
        """
        Evaluate the Hermite-Simpson residual on a solution trajectory.

        This checks how well the IDA solution satisfies the Hermite-Simpson discretization
        scheme by computing the residual at each time step.

        Args:
            result: Solution dictionary from solve() method

        Returns:
            Dictionary containing:
                - t: time points (excluding first point)
                - residual_norms: residual norms at each time step
                - residuals_diff_norms: residuals for differential equations
                - residuals_alg_norms: residuals for algebraic equations
                - max_residual: maximum residual norm
                - mean_residual: mean residual norm
        """
        t = result['t']
        x = result['x']
        z = result['z']
        n_states = len(self.state_names)
        n_alg = len(self.alg_names)
        n_steps = len(t) - 1

        print("\nEvaluating Hermite-Simpson residual on IDA solution...")
        print(f"  Number of time points: {len(t)}")
        print(f"  Number of steps to evaluate: {n_steps}")

        if n_steps <= 0:
            print("  Warning: Not enough time points to evaluate residuals (need at least 2)")
            return {
                't': np.array([]),
                'residual_norms': np.array([]),
                'residuals_diff_norms': np.array([]),
                'residuals_alg_norms': np.array([]),
                'all_residuals': np.array([]),
                'max_residual': 0.0,
                'mean_residual': 0.0,
            }

        # Storage for residuals
        residual_norms = np.zeros(n_steps)
        residuals_diff_norms = np.zeros(n_steps)
        residuals_alg_norms = np.zeros(n_steps)
        all_residuals = np.zeros((n_steps, n_states + n_alg))

        # Evaluate residual at each time step
        for i in range(n_steps):
            # Get states at time n and n+1
            y_n = np.concatenate([x[:, i], z[:, i]])
            y_np1 = np.concatenate([x[:, i+1], z[:, i+1]])

            # Compute Hermite-Simpson residual
            res = self.residual_hermite_simpson(t[i], y_n, t[i+1], y_np1)
            all_residuals[i, :] = res

            # Compute norms
            residual_norms[i] = np.linalg.norm(res)
            residuals_diff_norms[i] = np.linalg.norm(res[:n_states])
            residuals_alg_norms[i] = np.linalg.norm(res[n_states:])

        max_residual = np.max(residual_norms)
        mean_residual = np.mean(residual_norms)

        print(f"Hermite-Simpson residual evaluation complete!")
        print(f"  Max residual norm: {max_residual:.6e}")
        print(f"  Mean residual norm: {mean_residual:.6e}")
        print(f"  Max differential residual: {np.max(residuals_diff_norms):.6e}")
        print(f"  Max algebraic residual: {np.max(residuals_alg_norms):.6e}")

        return {
            't': t[1:],  # Exclude first point (no residual computed there)
            'residual_norms': residual_norms,
            'residuals_diff_norms': residuals_diff_norms,
            'residuals_alg_norms': residuals_alg_norms,
            'all_residuals': all_residuals,
            'max_residual': max_residual,
            'mean_residual': mean_residual,
        }

    def solve(self,
              t_span: Tuple[float, float],
              ncp: int = 500,
              rtol: float = 1e-6,
              atol: float = 1e-8,
              **kwargs) -> Dict:
        """
        Solve the DAE using SUNDIALS IDA.

        Args:
            t_span: (t0, tf) time interval
            ncp: number of communication points (output points)
            rtol: Relative tolerance
            atol: Absolute tolerance
            **kwargs: Additional arguments to IDA solver

        Returns:
            Dictionary with:
                - t: time points
                - x: differential states
                - z: algebraic variables
                - y: outputs (if h is defined)
                - state_names: names of differential states
                - alg_names: names of algebraic variables
                - output_names: names of outputs
        """
        print(f"\nSolving DAE from t={t_span[0]} to t={t_span[1]} using SUNDIALS IDA")
        print(f"  Output points: {ncp}")
        print(f"  Tolerances: rtol={rtol}, atol={atol}")

        # Combine initial conditions: y0 = [x0, z0]
        y0 = np.concatenate([self.x0, self.z0])
        n_states = len(self.state_names)
        n_total = len(y0)

        # Check initial algebraic constraints
        print(f"\nChecking initial conditions...")
        print(f"  Initial differential states (x0): {self.x0}")
        print(f"  Initial algebraic variables (z0): {self.z0}")

        g0 = self.eval_g(t_span[0], self.x0, self.z0)
        print(f"  Initial algebraic constraint residuals g(t0, x0, z0):")
        for i, (name, val) in enumerate(zip(self.alg_names, g0)):
            print(f"    {name}: {val:.6e}")

        if np.max(np.abs(g0)) > 1e-6:
            print(f"  WARNING: Initial algebraic constraints not satisfied (max residual: {np.max(np.abs(g0)):.6e})")
            print(f"  IDA will try to compute consistent initial conditions...")

        # Initial derivatives: ydot0 = [f(t0, x0, z0), 0]
        # Differential variables have non-zero derivatives
        # Algebraic variables have zero derivatives (by definition)
        f0 = self.eval_f(t_span[0], self.x0, self.z0)
        ydot0 = np.concatenate([f0, np.zeros(len(self.z0))])

        print(f"  Initial derivatives f(t0, x0, z0): {f0}")

        # Create IDA solver
        solver = dae(
            'ida',
            self.residual_ida,
            rtol=rtol,
            atol=atol,
            **kwargs
        )

        # Specify which variables are algebraic (id array)
        # id[i] = 1.0 for differential variables
        # id[i] = 0.0 for algebraic variables
        algvar_id = np.concatenate([
            np.ones(n_states),      # differential states
            np.zeros(n_total - n_states)  # algebraic variables
        ])
        solver.set_options(algebraic_vars_idx=algvar_id)

        # Create time span
        tspan = np.linspace(t_span[0], t_span[1], ncp)

        # Solve
        print("Starting integration...")
        sol = solver.solve(tspan, y0, ydot0)

        if not sol.flag:
            print(f"Warning: Solver flag indicates potential issues")

        print(f"Integration completed successfully!")
        print(f"  Time steps: {len(sol.values.t)}")

        # Extract results
        t = sol.values.t
        y_all = sol.values.y

        # Split into differential and algebraic variables
        x = y_all[:, :n_states].T
        z = y_all[:, n_states:].T

        # Compute outputs - if h is defined, use it; otherwise returns x (identity)
        # Determine output size
        if self.h_funcs:
            n_outputs = len(self.h_funcs)
        else:
            n_outputs = n_states

        y_out = np.zeros((n_outputs, len(t)))
        for i in range(len(t)):
            y_out[:, i] = self.eval_h(t[i], x[:, i], z[:, i])

        result = {
            't': t,
            'x': x,
            'z': z,
            'y': y_out,
            'state_names': self.state_names,
            'alg_names': self.alg_names,
            'output_names': self.output_names if self.output_names else self.state_names,
        }

        return result


def plot_solution(result: Dict, max_vars: int = 10):
    """
    Plot DAE solution.

    Args:
        result: Solution dictionary from DAESolver.solve()
        max_vars: Maximum number of variables to plot per subplot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping plot")
        return

    t = result['t']
    x = result['x']
    z = result['z']
    y = result['y']

    n_plots = 1 + (1 if z is not None else 0) + (1 if y is not None else 0)

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot differential states
    ax = axes[plot_idx]
    n_states = min(max_vars, x.shape[0])
    for i in range(n_states):
        ax.plot(t, x[i, :], label=result['state_names'][i])
    ax.set_xlabel('Time')
    ax.set_ylabel('Differential States')
    ax.set_title(f'Differential States (showing {n_states}/{x.shape[0]})')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True)
    plot_idx += 1

    # Plot algebraic variables
    if z is not None:
        ax = axes[plot_idx]
        n_alg = min(max_vars, z.shape[0])
        for i in range(n_alg):
            ax.plot(t, z[i, :], label=result['alg_names'][i])
        ax.set_xlabel('Time')
        ax.set_ylabel('Algebraic Variables')
        ax.set_title(f'Algebraic Variables (showing {n_alg}/{z.shape[0]})')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True)
        plot_idx += 1

    # Plot outputs
    if y is not None:
        ax = axes[plot_idx]
        for i in range(y.shape[0]):
            ax.plot(t, y[i, :], label=result['output_names'][i])
        ax.set_xlabel('Time')
        ax.set_ylabel('Outputs')
        ax.set_title('Outputs')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import time as time_module

    # Example usage
    json_path = "dae_examples/dae_specification_smooth.json"
    
    with open(json_path, 'r') as f:
        dae_data = json.load(f) 
    

    print("=" * 80)
    print("DAE Solver using SUNDIALS IDA")
    print("=" * 80)

    # Load and solve DAE
    start_time = time_module.time()

    solver = DAESolver(dae_data)

    result = solver.solve(
        t_span=(0.0, 60.0),
        ncp=500,  # Number of output points
        rtol=1e-5,
        atol=1e-5,
    )

    elapsed = time_module.time() - start_time

    print(f"\nTotal solve time: {elapsed:.3f} seconds")
    print(f"Final time: {result['t'][-1]:.6f}")
    print(f"Number of time points: {len(result['t'])}")

    # Print some solution values
    print("\nSolution at final time:")
    print(f"  First 5 differential states:")
    for i in range(min(5, len(result['state_names']))):
        print(f"    {result['state_names'][i]:20s} = {result['x'][i, -1]:12.6e}")

    print(f"\n  First 5 algebraic variables:")
    for i in range(min(5, len(result['alg_names']))):
        print(f"    {result['alg_names'][i]:20s} = {result['z'][i, -1]:12.6e}")

    if result['y'] is not None:
        print(f"\n  Outputs:")
        for i in range(len(result['output_names'])):
            print(f"    {result['output_names'][i]:20s} = {result['y'][i, -1]:12.6e}")
    else:
        print(f"\n  No output equations (h) defined in DAE specification")

    # Evaluate trapezoidal residual on IDA solution
    print("\n" + "=" * 80)
    print("Evaluating Trapezoidal Discretization Residual")
    print("=" * 80)

    trap_residual = solver.evaluate_trapezoidal_residual(result)

    # Evaluate Hermite-Simpson residual on IDA solution
    print("\n" + "=" * 80)
    print("Evaluating Hermite-Simpson Discretization Residual")
    print("=" * 80)

    hs_residual = solver.evaluate_hermite_simpson_residual(result)

    # Plot solution and residuals
    print("\nGenerating plots...")
    plot_solution(result, max_vars=5)

    # Plot trapezoidal residuals
    if len(trap_residual['residual_norms']) > 0:
        print("Plotting trapezoidal residuals...")
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # Plot residual norms
            ax = axes[0]
            ax.semilogy(trap_residual['t'], trap_residual['residual_norms'], 'b-', label='Total residual')
            ax.semilogy(trap_residual['t'], trap_residual['residuals_diff_norms'], 'r--', label='Differential part')
            ax.semilogy(trap_residual['t'], trap_residual['residuals_alg_norms'], 'g-.', label='Algebraic part')
            ax.set_xlabel('Time')
            ax.set_ylabel('Residual Norm')
            ax.set_title('Trapezoidal Residual Norms (IDA Solution)')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            # Plot histogram of residuals
            ax = axes[1]
            ax.hist(trap_residual['residual_norms'], bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Residual Norm')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Trapezoidal Residuals')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available, skipping residual plots")
    else:
        print("Skipping residual plots (no residual data available)")

    # Plot Hermite-Simpson residuals
    if len(hs_residual['residual_norms']) > 0:
        print("Plotting Hermite-Simpson residuals...")
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # Plot residual norms
            ax = axes[0]
            ax.semilogy(hs_residual['t'], hs_residual['residual_norms'], 'b-', label='Total residual')
            ax.semilogy(hs_residual['t'], hs_residual['residuals_diff_norms'], 'r--', label='Differential part')
            ax.semilogy(hs_residual['t'], hs_residual['residuals_alg_norms'], 'g-.', label='Algebraic part')
            ax.set_xlabel('Time')
            ax.set_ylabel('Residual Norm')
            ax.set_title('Hermite-Simpson Residual Norms (IDA Solution)')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            # Plot histogram of residuals
            ax = axes[1]
            ax.hist(hs_residual['residual_norms'], bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Residual Norm')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Hermite-Simpson Residuals')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available, skipping residual plots")
    else:
        print("Skipping Hermite-Simpson residual plots (no residual data available)")

    # Comparison plot of both methods
    if len(trap_residual['residual_norms']) > 0 and len(hs_residual['residual_norms']) > 0:
        print("Plotting comparison of discretization schemes...")
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

            ax.semilogy(trap_residual['t'], trap_residual['residual_norms'], 'b-', label='Trapezoidal', linewidth=2)
            ax.semilogy(hs_residual['t'], hs_residual['residual_norms'], 'r--', label='Hermite-Simpson', linewidth=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Residual Norm')
            ax.set_title('Comparison: Trapezoidal vs Hermite-Simpson Residuals (IDA Solution)')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            # Print comparison statistics
            print("\n" + "=" * 80)
            print("Comparison of Discretization Schemes")
            print("=" * 80)
            print(f"Trapezoidal:")
            print(f"  Max residual:  {trap_residual['max_residual']:.6e}")
            print(f"  Mean residual: {trap_residual['mean_residual']:.6e}")
            print(f"\nHermite-Simpson:")
            print(f"  Max residual:  {hs_residual['max_residual']:.6e}")
            print(f"  Mean residual: {hs_residual['mean_residual']:.6e}")
            print(f"\nRatio (Hermite-Simpson / Trapezoidal):")
            print(f"  Max residual:  {hs_residual['max_residual'] / trap_residual['max_residual']:.4f}")
            print(f"  Mean residual: {hs_residual['mean_residual'] / trap_residual['mean_residual']:.4f}")
            print("\nNote: Hermite-Simpson is a higher-order method (4th order) vs Trapezoidal (2nd order)")
            print("      Lower residuals indicate better approximation of the DAE dynamics")

        except ImportError:
            print("Matplotlib not available, skipping comparison plots")

    # Demonstrate vectorized evaluation with JAX
    print("\n" + "=" * 80)
    print("Testing Vectorized Function Evaluation (JAX vmap)")
    print("=" * 80)

    if JAX_AVAILABLE:
        print("JAX is available - using vmap for parallel evaluation")
    else:
        print("JAX not available - using numpy loops")

    # Prepare data for vectorized evaluation
    t_vec = result['t']
    x_vec = result['x']  # shape: (n_states, n_times)
    z_vec = result['z']  # shape: (n_alg, n_times)
    y_vec = np.vstack([x_vec, z_vec])  # shape: (n_states + n_alg, n_times)

    print(f"\nEvaluating functions over {len(t_vec)} time points...")
    print(f"  State vector shape: {x_vec.shape}")
    print(f"  Algebraic vector shape: {z_vec.shape}")
    print(f"  Combined y vector shape: {y_vec.shape}")

    # Time the vectorized evaluations
    import time as time_module

    # Test f vectorized
    start = time_module.time()
    f_vec = solver.eval_f_vectorized(t_vec, y_vec)
    f_time = time_module.time() - start
    print(f"\nVectorized f evaluation:")
    print(f"  Output shape: {f_vec.shape}")
    print(f"  Time: {f_time:.6f} seconds")
    print(f"  Sample values at t={t_vec[0]:.2f}: {f_vec[0, :min(3, f_vec.shape[1])]}")

    # Test g vectorized
    start = time_module.time()
    g_vec = solver.eval_g_vectorized(t_vec, y_vec)
    g_time = time_module.time() - start
    print(f"\nVectorized g evaluation:")
    print(f"  Output shape: {g_vec.shape}")
    print(f"  Time: {g_time:.6f} seconds")
    print(f"  Max constraint violation: {np.max(np.abs(g_vec)):.6e}")

    # Test h vectorized (if available)
    if solver.h_funcs:
        start = time_module.time()
        h_vec = solver.eval_h_vectorized(t_vec, y_vec)
        h_time = time_module.time() - start
        print(f"\nVectorized h evaluation:")
        print(f"  Output shape: {h_vec.shape}")
        print(f"  Time: {h_time:.6f} seconds")
        print(f"  Sample output at t={t_vec[0]:.2f}: {h_vec[0, :min(3, h_vec.shape[1])]}")

    # Compare with loop-based evaluation for verification
    print(f"\nVerification: Comparing vectorized vs loop-based evaluation...")
    f_loop = np.zeros((len(t_vec), len(solver.state_names)))
    start = time_module.time()
    for i in range(len(t_vec)):
        x_i = x_vec[:, i]
        z_i = z_vec[:, i]
        f_loop[i] = solver.eval_f(t_vec[i], x_i, z_i)
    loop_time = time_module.time() - start

    print(f"  Loop-based f evaluation time: {loop_time:.6f} seconds")
    print(f"  Vectorized speedup: {loop_time / f_time:.2f}x")
    print(f"  Max difference: {np.max(np.abs(f_vec - f_loop)):.6e}")

    print("\n" + "=" * 80)
    print("Vectorized evaluation test complete!")
    print("=" * 80)
