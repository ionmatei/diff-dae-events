"""
DAE Solver using scipy.integrate.solve_ivp

This version converts the semi-explicit DAE to an ODE by solving the
algebraic equations at each timestep using scipy.optimize.fsolve.

Solves the semi-explicit DAE:
    dx/dt = f(t, x, z, p)
    0 = g(t, x, z, p)
    y = h(t, x, z, p)  (outputs, if present)

Where:
    x = differential states
    z = algebraic variables
    p = parameters
    y = outputs
"""

import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from typing import Dict, List, Tuple, Optional
import re


class DAESolverScipy:
    """
    Solves semi-explicit DAEs from JSON specification using scipy solve_ivp.

    The DAE is converted to an ODE by solving algebraic equations at each timestep.
    """

    def __init__(self, dae_data: dict):
        """
        Load DAE from JSON specification.

        Args:
            dae_data: Dictionary containing DAE specification (simplified_form)
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

        print(f"DAE loaded (scipy version)")
        print(f"  Differential states: {len(self.states)}")
        print(f"  Algebraic variables: {len(self.alg_vars)}")
        print(f"  Parameters: {len(self.parameters)}")
        print(f"  Outputs: {len(self.outputs)}")
        print(f"  f equations: {len(self.f_eqs)}")
        print(f"  g equations: {len(self.g_eqs)}")

        # Compile equations into Python functions
        self._compile_equations()

    def _make_safe_name(self, name: str) -> str:
        """Convert variable name to valid Python identifier."""
        return re.sub(r'[^a-zA-Z0-9_]', '_', name)

    def _compile_equations(self):
        """
        Compile equation strings into executable Python functions.
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

        # Compile f equations (derivatives)
        self.f_funcs = []
        for i, eq in enumerate(self.f_eqs):
            # Extract LHS: der(state_name) = RHS
            match = re.match(r'der\((\w+)\)\s*=\s*(.+)', eq)
            if match:
                state_name, rhs = match.groups()
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

    def eval_h(self, t: float, x: np.ndarray, z: np.ndarray) -> Optional[np.ndarray]:
        """
        Evaluate h(t, x, z, p) - the output equations.

        Args:
            t: time
            x: differential states
            z: algebraic variables

        Returns:
            y = h(t, x, z, p) or None if no outputs
        """
        if not self.h_funcs:
            return None

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

    def solve_algebraic(self, t: float, x: np.ndarray, z_guess: np.ndarray) -> np.ndarray:
        """
        Solve algebraic equations 0 = g(t, x, z, p) for z.

        Uses Newton's method via fsolve with better tolerances.

        Args:
            t: time
            x: differential states (fixed)
            z_guess: initial guess for algebraic variables

        Returns:
            z that satisfies g(t, x, z, p) = 0
        """
        def residual(z):
            return self.eval_g(t, x, z)

        # Use reasonable tolerances
        z_sol, info, ier, msg = fsolve(residual, z_guess, full_output=True,
                                        xtol=1e-6, maxfev=2000)

        if ier != 1:
            # Check if residual is actually small
            res_norm = np.linalg.norm(info['fvec'])
            if res_norm > 1e-4:  # Only warn if residual is actually large
                if not hasattr(self, '_conv_warnings'):
                    self._conv_warnings = 0
                self._conv_warnings += 1
                if self._conv_warnings <= 5:  # Print first few warnings
                    print(f"Warning: Algebraic solver at t={t:.6f}")
                    print(f"  Message: {msg}")
                    print(f"  Residual norm: {res_norm:.6e}")

        return z_sol

    def ode_rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Right-hand side for ODE solver.

        At each timestep:
        1. Solve 0 = g(t, x, z) for z
        2. Compute dx/dt = f(t, x, z)

        Args:
            t: time
            x: differential states

        Returns:
            dx/dt
        """
        # Solve for algebraic variables
        # Use previous solution as initial guess (stored in self.z_current)
        z = self.solve_algebraic(t, x, self.z_current)

        # Update current algebraic variables for next step
        self.z_current = z

        # Evaluate derivatives
        dxdt = self.eval_f(t, x, z)

        return dxdt

    def solve(self,
              t_span: Tuple[float, float],
              method: str = 'BDF',
              rtol: float = 1e-6,
              atol: float = 1e-8,
              max_step: float = np.inf,
              t_eval: Optional[np.ndarray] = None,
              **kwargs) -> Dict:
        """
        Solve the DAE using scipy solve_ivp.

        Args:
            t_span: (t0, tf) time interval
            method: Integration method ('RK45', 'BDF', 'Radau', etc.)
            rtol: Relative tolerance
            atol: Absolute tolerance
            max_step: Maximum step size
            t_eval: Times at which to store solution (if None, solver chooses)
            **kwargs: Additional arguments to solve_ivp

        Returns:
            Dictionary with:
                - t: time points
                - x: differential states
                - z: algebraic variables
                - y: outputs (if h is defined)
                - state_names: names of differential states
                - alg_names: names of algebraic variables
                - output_names: names of outputs
                - sol: ODE solution object
        """
        print(f"\nSolving DAE from t={t_span[0]} to t={t_span[1]} using scipy.solve_ivp")
        print(f"  Method: {method}")
        print(f"  Tolerances: rtol={rtol}, atol={atol}")

        # Compute consistent initial conditions for algebraic variables
        print("Computing consistent initial conditions...")
        self._conv_warnings = 0
        z0_consistent = self.solve_algebraic(t_span[0], self.x0, self.z0)
        g0_norm = np.linalg.norm(self.eval_g(t_span[0], self.x0, z0_consistent))
        print(f"Initial algebraic residual: {g0_norm:.6e}")

        # Initialize current algebraic variables
        self.z_current = z0_consistent.copy()

        # Solve using solve_ivp
        print("Starting integration...")
        sol = solve_ivp(
            fun=self.ode_rhs,
            t_span=t_span,
            y0=self.x0,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            t_eval=t_eval,
            **kwargs
        )

        print(f"Integration completed: {sol.message}")
        print(f"  Time steps: {len(sol.t)}")
        print(f"  Function evaluations: {sol.nfev}")
        if hasattr(self, '_conv_warnings'):
            print(f"  Algebraic solver warnings: {self._conv_warnings}")

        # Extract time and states
        t = sol.t
        x = sol.y

        # Compute algebraic variables at all time points
        print("Computing algebraic variables at output points...")
        z = np.zeros((len(self.alg_vars), len(t)))
        for i, ti in enumerate(t):
            z_guess = z0_consistent if i == 0 else z[:, i-1]
            z[:, i] = self.solve_algebraic(ti, x[:, i], z_guess)

        # Compute outputs if h is defined
        y = None
        if self.h_funcs:
            print("Computing outputs...")
            y = np.zeros((len(self.h_funcs), len(t)))
            for i in range(len(t)):
                y[:, i] = self.eval_h(t[i], x[:, i], z[:, i])

        result = {
            't': t,
            'x': x,
            'z': z,
            'y': y,
            'sol': sol,
            'state_names': self.state_names,
            'alg_names': self.alg_names,
            'output_names': self.output_names,
        }

        return result


if __name__ == "__main__":
    import time as time_module

    # Example usage
    json_path = "dae_examples/dae_specification_smooth.json"

    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    print("=" * 80)
    print("DAE Solver using scipy.integrate.solve_ivp")
    print("=" * 80)

    # Load and solve DAE
    start_time = time_module.time()

    solver = DAESolverScipy(dae_data)

    # Create evaluation points (same as IDA version)
    t_eval = np.linspace(0.0, 60.0, 500)

    result = solver.solve(
        t_span=(0.0, 60.0),
        method='Radau',  # Good for stiff DAEs
        rtol=1e-6,
        atol=1e-6,
        t_eval=t_eval,
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
