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
from typing import Dict, List, Tuple, NamedTuple, Optional
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


class TrajectorySegment(NamedTuple):
    """A continuous run of the DAE between two events."""
    t: np.ndarray      # Shape (N,)
    x: np.ndarray      # Shape (N, n_states)
    z: np.ndarray      # Shape (N, n_alg)
    xp: np.ndarray     # Shape (N, n_states) - Derivative (Needed for Hermite Interp)

class EventInfo(NamedTuple):
    """Explicit capture of the discontinuity."""
    t_event: float           # tau
    event_idx: int           # Which condition triggered
    
    # State immediately BEFORE reinit (x-)
    x_pre: np.ndarray
    z_pre: np.ndarray
    
    # State immediately AFTER reinit (x+)
    x_post: np.ndarray
    z_post: np.ndarray
    
    # Sensitivity matrices for the jump (Optional, but good to store if calculated)
    # J_x = d(x+)/d(x-)
    # J_p = d(x+)/d(p)

class AugmentedSolution(NamedTuple):
    """The full output required by the Adjoint Optimizer."""
    segments: List[TrajectorySegment]
    events: List[EventInfo]


class DAESolver:
    """
    Solves semi-explicit DAEs from JSON specification using solve_ivp.

    The DAE is converted to an ODE by solving algebraic equations at each timestep.
    """

    def __init__(self, dae_data: dict, verbose: bool = True,
                 zeno_dt_threshold: float = 1.0e-4,
                 zeno_window_events: int = 10,
                 zeno_window_seconds: float = 5.0e-3,
                 use_compiled_residual: bool = True):
        """
        Load DAE from JSON specification.

        Args:
            dae_data: Dictionary containing DAE specification
            verbose: Whether to print loading/compilation messages
            zeno_dt_threshold: Time gap (seconds) below which two consecutive
                *same-index* events are treated as Zeno chattering and the
                simulation is terminated early.
            zeno_window_events: Number of recent events used by the
                rate-based Zeno detector (default 10).
            zeno_window_seconds: If `zeno_window_events + 1` consecutive
                events span less than this many seconds, the simulation
                terminates (default 5 ms). Catches sustained chatter even
                when several balls bounce against the floor in alternation
                (which the same-index check alone misses), without
                triggering on legitimate close-clusters of distinct events
                (e.g. several balls landing within ~1 ms of each other).
        """
        self.verbose = verbose
        self.zeno_dt_threshold = float(zeno_dt_threshold)
        self.zeno_window_events = int(zeno_window_events)
        self.zeno_window_seconds = float(zeno_window_seconds)
        self.use_compiled_residual = bool(use_compiled_residual)
        self._f_impl = None
        self._g_impl = None
        self._zc_impl = None
        form = dae_data

        # Extract variables
        self.states = form['states']  # Differential states
        self.alg_vars = form.get('alg_vars', [])  # Algebraic variables (may be empty for ODEs)
        self.parameters = form['parameters']
        self.outputs = form.get('outputs', None)
        if self.outputs is None:
            self.outputs = []

        # Extract equations
        self.f_eqs = form['f']  # dx/dt = f(...)
        self.g_eqs = form.get('g', [])  # 0 = g(...) (may be empty/null for ODEs)
        if self.g_eqs is None:
            self.g_eqs = []
        self.h_eqs = form.get('h', None)  # y = h(...)
        self.when_clauses = form.get('when', None)  # Event handling

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

        if self.verbose:
            print(f"DAE loaded")
            print(f"  Differential states: {len(self.states)}")
            print(f"  Algebraic variables: {len(self.alg_vars)}")
            print(f"  Parameters: {len(self.parameters)}")
            print(f"  Outputs: {len(self.outputs)}")
            print(f"  f equations: {len(self.f_eqs)}")
            print(f"  g equations: {len(self.g_eqs)}")
            if self.when_clauses:
                print(f"  Event clauses (when): {len(self.when_clauses)}")

        # Compile equations into Python functions
        self._compile_equations()

        # Compile event handling (when clauses)
        self._compile_events()

        # Compile a single-function fast path for f, g, zc. This consolidates
        # the per-equation eval() loop into one compiled function that writes
        # the entire residual / event vector in one call. Auto-detected:
        # falls back transparently to the eval-based path if state / alg /
        # parameter names collide with Python keywords or namespace entries.
        if self.use_compiled_residual:
            self._compile_residual_fast_path()

        # Compile JAX vectorized functions (vmap) once during initialization
        self._compile_vectorized_functions()

    def update_parameters(self, p: np.ndarray):
        """Update the parameter values used by the solver."""
        self.p = np.array(p).copy()

    def update_initial_conditions(self, x0: np.ndarray):
        """Update the initial conditions used by the solver."""
        self.x0 = np.array(x0).copy()

    def _make_safe_name(self, name: str) -> str:
        """Convert variable name to valid Python identifier."""
        # Replace dots and special chars with underscores
        return re.sub(r'[^a-zA-Z0-9_]', '_', name)

    def _compile_equations(self):
        """
        Compile equation strings into executable Python functions.
        This uses eval() but in a controlled namespace.
        """
        if self.verbose:
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

        if self.verbose:
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

        if self.verbose:
            print("Equations compiled successfully!")

    def _compile_events(self):
        """
        Compile event handling (when clauses) into executable functions.
        
        Creates:
        - Zero-crossing functions (zc) for event detection
        - Reinitialization expressions
        - Mapping of which variables are affected by each event
        """
        if not self.when_clauses:
            self.n_events = 0
            self.zc_funcs = []
            self.event_reinit_exprs = []
            self.event_reinit_vars = []
            self.event_reinit_var_names = []
            if self.verbose:
                print("\nNo event clauses (when) found - events disabled")
            return
        
        self.n_events = len(self.when_clauses)
        self.zc_funcs = []  # Zero-crossing function expressions
        self.event_reinit_exprs = []  # Reinitialization expressions
        self.event_reinit_vars = []  # Variable indices to reinitialize
        self.event_reinit_var_names = []  # Variable names for tracking
        
        if self.verbose:
            print(f"\nCompiling {self.n_events} event clause(s)...")
        
        for i, when_clause in enumerate(self.when_clauses):
            condition = when_clause['condition']
            reinit = when_clause['reinit']

            zc_expr = self._parse_condition_to_zc(condition)
            self.zc_funcs.append(zc_expr)

            # Normalize reinit to list (support both string and list of strings)
            reinit_list = reinit if isinstance(reinit, list) else [reinit]

            exprs_i, names_i, vars_i = [], [], []
            for reinit_str in reinit_list:
                reinit_expr, var_name = self._parse_reinit(reinit_str)
                exprs_i.append(reinit_expr)
                names_i.append(var_name)

                if var_name in self.state_names:
                    vars_i.append(('state', self.state_names.index(var_name)))
                elif var_name in self.alg_names:
                    vars_i.append(('alg', self.alg_names.index(var_name)))
                else:
                    raise ValueError(f"Reinit variable '{var_name}' not found in states or algebraic variables")

            self.event_reinit_exprs.append(exprs_i)
            self.event_reinit_var_names.append(names_i)
            self.event_reinit_vars.append(vars_i)

            if self.verbose:
                print(f"  Event {i}: when {condition} then reinit {names_i}")
                print(f"    Zero-crossing: zc = {zc_expr}")
                for expr, vn, (vt, vi) in zip(exprs_i, names_i, vars_i):
                    print(f"    Reinit: {expr} -> {vt}[{vi}] ({vn})")
        
        if self.verbose:
            print("Event clauses compiled successfully!")

    # ---------------------------------------------------------------- #
    # Compiled residual fast path
    # ---------------------------------------------------------------- #
    def _compile_residual_fast_path(self) -> bool:
        """
        Build single compiled Python functions for f, g, and zc that write
        the entire output vector in one call. Replaces the per-equation
        eval(expr, ns) loop on the IDA hot path (one Python eval per
        equation, plus a fresh namespace dict per call).

        Auto-detect: requires state / alg / parameter names to be valid
        Python identifiers that do not collide with the math-fn closure or
        with reserved local names. If any name is incompatible, the fast
        path is left disabled and the existing eval-based path is used.

        Sets self._f_impl, self._g_impl, self._zc_impl on success. They
        each have signature (t, x, z, p, out) and write into out in place.
        """
        import keyword

        ns_keys = {
            'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
            'exp', 'log', 'log10', 'sqrt', 'abs', 'pow',
            'min', 'max',
            't', 'time', '_x', '_z', '_p', '_out',
        }

        def _safe(name: str) -> bool:
            return (
                isinstance(name, str)
                and name.isidentifier()
                and not keyword.iskeyword(name)
                and name not in ns_keys
            )

        all_names = self.state_names + self.alg_names + self.param_names
        bad = [n for n in all_names if not _safe(n)]
        if bad:
            if self.verbose:
                preview = ", ".join(bad[:5]) + (" ..." if len(bad) > 5 else "")
                print(f"Compiled residual fast path: disabled "
                      f"(incompatible name(s): {preview})")
            return False

        # Build the shared preamble: unpack t, states, algebraics, params
        # into local variables so the per-equation expressions can reference
        # them by their original symbolic names.
        preamble = ["    time = t"]
        for i, name in enumerate(self.state_names):
            preamble.append(f"    {name} = _x[{i}]")
        for i, name in enumerate(self.alg_names):
            preamble.append(f"    {name} = _z[{i}]")
        for i, name in enumerate(self.param_names):
            preamble.append(f"    {name} = _p[{i}]")
        preamble_str = "\n".join(preamble) if preamble else "    pass"

        def _build(fn_name: str, exprs):
            if not exprs:
                return f"def {fn_name}(t, _x, _z, _p, _out):\n    return\n"
            lines = [f"def {fn_name}(t, _x, _z, _p, _out):", preamble_str]
            for i, expr in enumerate(exprs):
                lines.append(f"    _out[{i}] = {expr}")
            return "\n".join(lines) + "\n"

        zc_funcs = self.zc_funcs if hasattr(self, 'zc_funcs') else []
        src = (
            _build("_f_impl", self.f_funcs)
            + _build("_g_impl", self.g_funcs)
            + _build("_zc_impl", zc_funcs)
        )

        # Closure: math fns route through numpy, identical to the eval path.
        glb = {
            '__builtins__': {},
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
            'exp': np.exp, 'log': np.log, 'log10': np.log10,
            'sqrt': np.sqrt, 'abs': np.abs, 'pow': np.power,
            'min': np.minimum, 'max': np.maximum,
        }
        locs: Dict = {}
        try:
            exec(compile(src, "<dae_compiled_residual>", "exec"), glb, locs)
        except Exception as e:
            if self.verbose:
                print(f"Compiled residual fast path: disabled "
                      f"(compilation failed: {e})")
            return False

        f_impl = locs['_f_impl']
        g_impl = locs['_g_impl']
        zc_impl = locs['_zc_impl']

        # Numerical-equivalence check at the initial conditions: ensures
        # the compiled path produces identical results to the eval-based
        # path before we let it into the hot loop. If it disagrees we
        # disable transparently and the user sees the eval fallback.
        n_states = len(self.state_names)
        n_alg = len(self.alg_names)
        n_events = self.n_events
        try:
            x_test = np.asarray(self.x0, dtype=float)
            z_test = np.asarray(self.z0, dtype=float)
            t_test = 0.0

            out_f = np.zeros(n_states)
            f_impl(t_test, x_test, z_test, self.p, out_f)
            ref_f = self.eval_f(t_test, x_test, z_test)
            if not np.allclose(out_f, ref_f, rtol=1e-12, atol=1e-12):
                raise RuntimeError("f mismatch on initial conditions")

            if n_alg > 0:
                out_g = np.zeros(n_alg)
                g_impl(t_test, x_test, z_test, self.p, out_g)
                ref_g = self.eval_g(t_test, x_test, z_test)
                if not np.allclose(out_g, ref_g, rtol=1e-12, atol=1e-12):
                    raise RuntimeError("g mismatch on initial conditions")

            if n_events > 0:
                out_zc = np.zeros(n_events)
                zc_impl(t_test, x_test, z_test, self.p, out_zc)
                ref_zc = self.eval_zc(t_test, x_test, z_test)
                if not np.allclose(out_zc, ref_zc, rtol=1e-12, atol=1e-12):
                    raise RuntimeError("zc mismatch on initial conditions")
        except Exception as e:
            if self.verbose:
                print(f"Compiled residual fast path: disabled "
                      f"(validation failed: {e})")
            return False

        self._f_impl = f_impl
        self._g_impl = g_impl
        self._zc_impl = zc_impl

        if self.verbose:
            print("Compiled residual fast path: enabled "
                  f"(f:{len(self.f_funcs)}, g:{len(self.g_funcs)}, "
                  f"zc:{len(zc_funcs)})")
        return True

    def _parse_condition_to_zc(self, condition: str) -> str:
        """
        Parse a condition like 'h<0' into a zero-crossing function 'h - 0'.
        
        The zero-crossing function zc is designed so that:
        - zc < 0 means condition is TRUE
        - zc > 0 means condition is FALSE
        """
        condition = condition.strip()
        
        # Handle comparison operators
        for op in ['<=', '>=', '<', '>', '==']:
            if op in condition:
                lhs, rhs = condition.split(op, 1)
                lhs, rhs = lhs.strip(), rhs.strip()
                
                # For <, <=: zc = lhs - rhs (zc < 0 when condition true)
                if op in ['<', '<=']:
                    return f"({lhs}) - ({rhs})"
                # For >, >=: zc = rhs - lhs (zc < 0 when condition true)
                elif op in ['>', '>=']:
                    return f"({rhs}) - ({lhs})"
                # For ==: zc = lhs - rhs (zc = 0 when condition true)
                else:  # ==
                    return f"({lhs}) - ({rhs})"
        
        raise ValueError(f"Could not parse condition: {condition}")
    
    def _parse_reinit(self, reinit: str) -> Tuple[str, str]:
        """
        Parse a reinit expression like 'v + e*prev(v) = 0'.
        
        The format is an equation that equals zero, and we need to:
        1. Identify which variable is being reinitialized (the one without prev())
        2. Rearrange to solve for that variable
        
        Returns:
            (expression, variable_name)
        """
        if '=' not in reinit:
            raise ValueError(f"Invalid reinit format (missing '='): {reinit}")
        
        lhs, rhs = reinit.split('=', 1)
        lhs, rhs = lhs.strip(), rhs.strip()
        
        # Handle both "expr = 0" and "0 = expr" forms
        if lhs == '0' or lhs == '0.0':
            # Form: 0 = expr, so expr should be the equation
            equation = rhs
        elif rhs == '0' or rhs == '0.0':
            # Form: expr = 0, so expr is the equation
            equation = lhs
        else:
            # Form: lhs = rhs, convert to lhs - rhs = 0
            equation = f"({lhs}) - ({rhs})"
        
        # Find the variable being reinitialized (appears without prev())
        # Strategy: look for state/alg variable names in the equation
        var_name = None
        for name in self.state_names + self.alg_names:
            # Check if variable appears in equation without prev() around it
            # Simple heuristic: if name appears but "prev(" + name + ")" doesn't replace all occurrences
            if name in equation:
                # Check if it's not always inside prev()
                import re
                # Find all occurrences of the variable
                pattern = r'\b' + re.escape(name) + r'\b'
                matches = list(re.finditer(pattern, equation))
                
                # Check if any match is NOT preceded by "prev("
                for match in matches:
                    start = match.start()
                    # Look back to see if preceded by "prev("
                    if start >= 5:
                        prefix = equation[start-5:start]
                        if not prefix.endswith('prev('):
                            var_name = name
                            break
                    else:
                        # Not enough space for "prev(", so it's not inside prev()
                        var_name = name
                        break
                
                if var_name:
                    break
        
        if not var_name:
            raise ValueError(f"Could not identify variable being reinitialized in: {reinit}")
        
        # The expression should solve the equation for var_name
        # For "v + e*prev(v) = 0", we want v = -e*prev(v)
        # We'll keep the equation form and solve it during evaluation
        
        return equation, var_name



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

    def eval_zc(self, t: float, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Evaluate zero-crossing functions for event detection.
        
        Args:
            t: time
            x: differential states
            z: algebraic variables
        
        Returns:
            Array of zero-crossing values (length = n_events)
            zc[i] < 0 means event condition i is TRUE
            zc[i] > 0 means event condition i is FALSE
        """
        if self.n_events == 0:
            return np.array([])
        
        ns = self._create_eval_namespace(t, x, z)
        
        zc = np.zeros(self.n_events)
        for i, expr in enumerate(self.zc_funcs):
            try:
                zc[i] = eval(expr, ns)
            except Exception as e:
                print(f"Error evaluating zero-crossing function {i}: {expr}")
                print(f"Error: {e}")
                raise
        
        return zc

    def _root_fn_wrapper(self, t: float, y: np.ndarray, ydot: np.ndarray, out: np.ndarray):
        """
        Wrapper for IDA root-finding.

        This is called by IDA to detect zero-crossings.
        Signature must match IDA's expected interface: (t, y, ydot, out)
        """
        n_states = len(self.state_names)
        x = y[:n_states]
        z = y[n_states:]

        if self._zc_impl is not None and self.n_events > 0:
            # Compiled fast path: writes directly into IDA's out buffer,
            # no per-equation eval(), no per-call namespace dict.
            self._zc_impl(t, x, z, self.p, out)
            return

        zc = self.eval_zc(t, x, z)
        out[:] = zc


    def _apply_reinit(self, event_idx: int, t: float, x: np.ndarray, z: np.ndarray, 
                      x_pre: np.ndarray, z_pre: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply reinitialization for a triggered event.
        
        Args:
            event_idx: Index of the event that was triggered
            t: Current time
            x: Current differential states
            z: Current algebraic variables
            x_pre: Pre-event differential states (for prev() operator)
            z_pre: Pre-event algebraic variables (for prev() operator)
        
        Returns:
            (x_new, z_new): Updated state and algebraic variables
        """
        from scipy.optimize import fsolve
        import re

        x_new = x.copy()
        z_new = z.copy()

        # Lists of reinit equations / variables for this event
        reinit_equations = self.event_reinit_exprs[event_idx]
        reinit_var_list = self.event_reinit_vars[event_idx]
        reinit_name_list = self.event_reinit_var_names[event_idx]

        # Build base namespace ONCE from pre-event state
        base_ns = self._create_eval_namespace(t, x_pre, z_pre)
        for i, name in enumerate(self.state_names):
            base_ns[f'prev_{name}'] = x_pre[i]
        for i, name in enumerate(self.alg_names):
            base_ns[f'prev_{name}'] = z_pre[i]

        def prev(vn):
            if vn in self.state_names:
                return x_pre[self.state_names.index(vn)]
            elif vn in self.alg_names:
                return z_pre[self.alg_names.index(vn)]
            raise ValueError(f"Unknown variable in prev(): {vn}")
        base_ns['prev'] = prev

        prev_pattern = r'prev\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)'

        # Solve each reinit equation independently
        for reinit_eq, (var_type, var_idx), var_name in zip(
                reinit_equations, reinit_var_list, reinit_name_list):
            eq_mod = re.sub(prev_pattern, r'prev_\1', reinit_eq)

            def residual(var_value_array, _eq=eq_mod, _vn=var_name):
                current_ns = base_ns.copy()
                current_ns[_vn] = var_value_array[0]
                try:
                    return [eval(_eq, current_ns)]
                except Exception as e:
                    print(f"Error evaluating reinit for event {event_idx}: {_eq}")
                    raise

            initial_guess = x_pre[var_idx] if var_type == 'state' else z_pre[var_idx]
            try:
                new_value = fsolve(residual, [initial_guess], full_output=False)[0]
            except Exception as e:
                print(f"Error solving reinit for event {event_idx}: {reinit_eq}")
                raise

            if var_type == 'state':
                x_new[var_idx] = new_value
            else:
                z_new[var_idx] = new_value

        return x_new, z_new



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

        if self.verbose:
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

        if self._f_impl is not None:
            # Compiled fast path: write -f into res[:n_states], then add
            # xdot in place. No temporaries, no per-equation eval().
            res_x = res[:n_states]
            self._f_impl(t, x, z, self.p, res_x)
            np.subtract(xdot, res_x, out=res_x)
            if self._g_impl is not None and len(z) > 0:
                self._g_impl(t, x, z, self.p, res[n_states:])
            return

        # Evaluate f and g (eval-based fallback)
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
                verbose: bool = False,
                **kwargs) -> Dict:
            """
            Solve the DAE using SUNDIALS IDA.
            """
            if verbose:
                print(f"\nSolving DAE from t={t_span[0]} to t={t_span[1]} using SUNDIALS IDA")
                print(f"  Output points: {ncp}")
                print(f"  Tolerances: rtol={rtol}, atol={atol}")

            # Combine initial conditions: y0 = [x0, z0]
            y0 = np.concatenate([self.x0, self.z0])
            n_states = len(self.state_names)
            n_total = len(y0)

            # Check initial algebraic constraints
            if verbose:
                print(f"\nChecking initial conditions...")
                print(f"  Initial differential states (x0): {self.x0}")
                print(f"  Initial algebraic variables (z0): {self.z0}")

            g0 = self.eval_g(t_span[0], self.x0, self.z0)
            if verbose:
                print(f"  Initial algebraic constraint residuals g(t0, x0, z0):")
                for i, (name, val) in enumerate(zip(self.alg_names, g0)):
                    print(f"    {name}: {val:.6e}")

                if np.max(np.abs(g0)) > 1e-6:
                    print(f"  WARNING: Initial algebraic constraints not satisfied (max residual: {np.max(np.abs(g0)):.6e})")
                    print(f"  IDA will try to compute consistent initial conditions...")

            # Initial derivatives: ydot0 = [f(t0, x0, z0), 0]
            f0 = self.eval_f(t_span[0], self.x0, self.z0)
            ydot0 = np.concatenate([f0, np.zeros(len(self.z0))])

            if verbose:
                print(f"  Initial derivatives f(t0, x0, z0): {f0}")

            # Create IDA solver
            solver = dae(
                'ida',
                self.residual_ida,
                rtol=rtol,
                atol=atol,
                **kwargs
            )

            # ---------------------------------------------------------
            # FIX START: Generate Indices, not a Mask
            # ---------------------------------------------------------
            # scikits.odes expects a list of INTEGER INDICES for algebraic vars.
            # Example: if n_states=3 and n_total=5, we want [3, 4]
            algvar_indices = np.arange(n_states, n_total)
            
            solver.set_options(algebraic_vars_idx=algvar_indices)
            
            # FIX 2: Enable Consistent IC Calculation
            # This is required because ydot0 for algebraic vars is 0, which 
            # might be inconsistent with the derivatives of the states.
            solver.set_options(compute_initcond='yp0')
            # ---------------------------------------------------------
            # FIX END
            # ---------------------------------------------------------

            # Create time span
            tspan = np.linspace(t_span[0], t_span[1], ncp)

            # Solve
            if verbose:
                print("Starting integration...")
            
            # Note: compute_initcond happens automatically inside .solve() 
            # when the option is set above.
            sol = solver.solve(tspan, y0, ydot0)

            if not sol.flag and verbose:
                print(f"Warning: Solver flag indicates potential issues: {sol.message}")

            if verbose:
                print(f"Integration completed successfully!")
                print(f"  Time steps: {len(sol.values.t)}")

            # Extract results
            t = sol.values.t
            y_all = sol.values.y

            # Split into differential and algebraic variables
            x = y_all[:, :n_states].T
            z = y_all[:, n_states:].T

            # Compute outputs
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

    def solve_with_events(self,
                          t_span: Tuple[float, float],
                          ncp: int = 500,
                          rtol: float = 1e-6,
                          atol: float = 1e-8,
                          min_event_delta: float = None,
                          verbose: bool = False,
                          **kwargs) -> Dict:
        """
        Solve the DAE with event handling using SUNDIALS IDA.
        
        Events are detected via zero-crossing functions and trigger reinitialization.
        
        Args:
            t_span: Time span (t_start, t_end)
            ncp: Approximate number of output points
            rtol: Relative tolerance
            atol: Absolute tolerance
            min_event_delta: Minimum time between events (stops if violated)
            verbose: Print progress information
            **kwargs: Additional options passed to IDA
        
        Returns:
            Dictionary containing:
                - t: time points
                - x: differential states
                - z: algebraic variables
                - y: outputs
                - event_times: list of event times
                - event_indices: list of event indices that triggered
                - event_vars_changed: list of (var_name, old_val, new_val) for each event
                - early_termination: True if stopped due to event frequency threshold
                - state_names, alg_names, output_names: variable names
        """
        if verbose:
            print(f"\nSolving DAE with events from t={t_span[0]} to t={t_span[1]}")
            print(f"  Events enabled: {self.n_events > 0}")
            print(f"  Event frequency threshold: {min_event_delta if min_event_delta else 'None'}")
        
        # If no events, use regular solve
        if self.n_events == 0:
            if verbose:
                print("  No events detected, using regular solve method")
            result = self.solve(t_span, ncp, rtol, atol, verbose, **kwargs)
            result['event_times'] = []
            result['event_indices'] = []
            result['event_vars_changed'] = []
            result['early_termination'] = False
            return result
        
        # Initialize
        t_start, t_end = t_span
        t_curr = t_start
        y_curr = np.concatenate([self.x0, self.z0])
        n_states = len(self.state_names)
        
        # Calculate initial derivatives
        f0 = self.eval_f(t_curr, self.x0, self.z0)
        yp_curr = np.concatenate([f0, np.zeros(len(self.z0))])
        
        # Initialize event tracking
        zc_vals = self.eval_zc(t_curr, self.x0, self.z0)
        cond_active_prev = (zc_vals <= 0)  # Boolean state memory (treat boundary as active to avoid re-fire after a position clamp)
        
        event_times = []
        event_indices = []
        event_vars_changed = []
        last_event_time = -np.inf
        
        # Storage for results
        t_all = [t_curr]
        y_all = [y_curr.copy()]
        
        early_termination = False
        iteration = 0
        max_iterations = 10000  # Safety limit
        
        if verbose:
            print(f"  Initial zero-crossing values: {zc_vals}")
            print(f"  Initial condition states: {cond_active_prev}")
        
        while t_curr < t_end and iteration < max_iterations:
            iteration += 1
            
            # Configure solver
            n_total = len(y_curr)
            algvar_indices = np.arange(n_states, n_total)
            
            solver = dae(
                'ida',
                self.residual_ida,
                rtol=rtol,
                atol=atol,
                **kwargs
            )
            
            solver.set_options(algebraic_vars_idx=algvar_indices)
            solver.set_options(compute_initcond='yp0')
            
            # Set up root finding for events
            if self.n_events > 0:
                solver.set_options(
                    rootfn=self._root_fn_wrapper,
                    nr_rootfns=self.n_events
                )
            
            # Create time segment
            remaining_time = t_end - t_curr
            n_seg = max(int(ncp * remaining_time / (t_end - t_start)), 2)
            t_segment = np.linspace(t_curr, t_end, n_seg)
            
            # Solve until next event or end time
            if verbose and iteration % 10 == 1:
                print(f"  Iteration {iteration}: Solving from t={t_curr:.6f} to t={t_end:.6f}")
            
            sol = solver.solve(t_segment, y_curr, yp_curr)
            
            # Append results (skip first point if not the initial time)
            start_idx = 0 if iteration == 1 else 1
            if len(sol.values.t) > start_idx:
                t_all.extend(sol.values.t[start_idx:])
                y_all.extend(sol.values.y[start_idx:, :])
            
            # Update current state
            t_curr = sol.values.t[-1]
            y_curr = sol.values.y[-1, :]
            yp_curr = sol.values.ydot[-1, :]
            
            
            # Check solver status
            if sol.flag == 2:  # Root found (event detected)
                if verbose:
                    print(f"    Event detected at t={t_curr:.6f}")
                
                # Extract current state
                x_curr = y_curr[:n_states]
                z_curr = y_curr[n_states:]
                
                # -----------------------------------------------------------
                # FIX: Check for "Zeno" behavior (consecutive events without flow)
                # -----------------------------------------------------------
                # We declare Zeno only when (a) the solver returned <= 2 points
                # (no intermediate evolution) AND (b) the time gap to the prior
                # event is below `zeno_dt_threshold`. Distinct legitimate events
                # firing within a small but finite gap (e.g. several balls
                # landing within a few ms) used to terminate the run; the
                # time-gap check now lets those proceed.
                if (len(sol.values.t) <= 2
                        and last_event_time > -np.inf
                        and (t_curr - last_event_time) < self.zeno_dt_threshold):
                    if verbose:
                        print(f"    Consecutive events detected without intermediate samples!")
                        print(f"    Terminating simulation early at t={t_curr:.6f} to prevent Zeno bottleneck.")
                    early_termination = True
                    break

                # -----------------------------------------------------------
                # FIX 1: Use IDA's root info instead of manual 'abs < 0.1' check
                # -----------------------------------------------------------
                # sol.roots.val is usually an array of integers (1, -1, 0)
                # indicating which root function had a sign change.
                triggered_indices = []
                if hasattr(sol, 'roots') and hasattr(sol.roots, 'val'):
                    # Find indices where root info is non-zero
                    triggered_indices = [idx for idx, val in enumerate(sol.roots.val) if val != 0]
                    if verbose:
                        print(f"      IDA roots info: {sol.roots.val}")
                        print(f"      Triggered indices from IDA: {triggered_indices}")
                else:
                    # Fallback if wrapper doesn't expose roots
                    # IDA only returns flag=2 if it found a root, so at least one zc must be near 0
                    # IDA may return state slightly past the root, so use reasonable tolerance
                    zc_check = self.eval_zc(t_curr, x_curr, z_curr)
                    triggered_indices = []
                    zc_tolerance = 0.05  # Tighter than old 0.1, but accounts for IDA behavior
                    for idx, val in enumerate(zc_check):
                        # If we're near the root and it was previously positive (inactive),
                        # then this is a falling edge crossing
                        if abs(val) < zc_tolerance and not cond_active_prev[idx]:
                            triggered_indices.append(idx)
                        # Alternatively, if the sign actually flipped to negative
                        elif (val < 0) and not cond_active_prev[idx]:
                            triggered_indices.append(idx)
                    if verbose:
                        print(f"      Zero-crossing values: {zc_check}")
                        print(f"      Triggered indices (tolerance={zc_tolerance}): {triggered_indices}")
                
                if verbose:
                    print(f"      cond_active_prev (before event): {cond_active_prev}")
                
                # Check for event frequency violation BEFORE processing events
                if min_event_delta is not None and last_event_time > -np.inf:
                    time_since_last = t_curr - last_event_time
                    if time_since_last < min_event_delta:
                        if verbose:
                            print(f"    Event frequency threshold violated!")
                            print(f"    Time since last event: {time_since_last:.6e} < {min_event_delta:.6e}")
                            print(f"    Terminating simulation early at t={t_curr:.6f}")
                        early_termination = True
                        break
                
                event_occurred = False
                
                # Process Triggered Events
                # Capture the "Pre-Event" state ONCE for this time step.
                # This ensures all simultaneous events see the un-modified state 
                # entering the event point (order-independent behavior).
                x_frozen_pre = x_curr.copy()
                z_frozen_pre = z_curr.copy()
                
                for i in triggered_indices:
                    # Check direction: We only want falling edge (False -> True)
                    # i.e., previously Positive (False), now Negative/Zero (True)
                    if not cond_active_prev[i]:
                        event_occurred = True
                        
                        if verbose:
                            print(f"    Event {i} triggered: {self.when_clauses[i]['condition']}")
                        
                        # Apply reinitialization using the FROZEN pre-state
                        # This prevents Event B from reacting to Event A's immediate changes
                        # All simultaneous events see the same pre-event state
                        x_new, z_new = self._apply_reinit(
                            i, t_curr, x_curr, z_curr, x_frozen_pre, z_frozen_pre
                        )
                        
                        # Track what changed (compare against frozen pre-state)
                        event_times.append(t_curr)
                        event_indices.append(i)
                        for var_name, (var_type, var_idx) in zip(
                                self.event_reinit_var_names[i], self.event_reinit_vars[i]):
                            old_val = x_frozen_pre[var_idx] if var_type == 'state' else z_frozen_pre[var_idx]
                            new_val = x_new[var_idx] if var_type == 'state' else z_new[var_idx]
                            event_vars_changed.append((var_name, old_val, new_val))
                            if verbose:
                                print(f"      Reinitialized {var_name}: {old_val:.6e} -> {new_val:.6e}")
                        
                        # Update current state (cumulative updates for simultaneous events)
                        x_curr = x_new
                        z_curr = z_new
                
                # -----------------------------------------------------------
                # FIX 2 & 3: Correct Post-Event State Update
                # -----------------------------------------------------------
                if event_occurred:
                    last_event_time = t_curr
                    
                    # 1. Update y_curr for the solver
                    y_curr = np.concatenate([x_curr, z_curr])
                    
                    # 2. Update derivative guess
                    f_new = self.eval_f(t_curr, x_curr, z_curr)
                    yp_curr = np.concatenate([f_new, np.zeros(len(z_curr))])
                    
                    # 3. CRITICAL: Re-evaluate ALL event conditions at the NEW state.
                    # This ensures 'cond_active_prev' is correct for the start 
                    # of the next continuous segment.
                    zc_post = self.eval_zc(t_curr, x_curr, z_curr)
                    cond_active_prev = (zc_post <= 0)

                    if verbose:
                        print(f"      Re-init complete. New condition states: {cond_active_prev}")
                
                
            elif sol.flag == 0:  # Success - reached t_end
                if verbose:
                    print(f"  Successfully reached t_end = {t_end:.6f}")
                break
            else:
                print(f"Warning: Solver returned flag {sol.flag}: {sol.message}")
                break
        
        if iteration >= max_iterations:
            print(f"Warning: Maximum iterations ({max_iterations}) reached")
            early_termination = True
        
        # Convert lists to arrays
        t_all = np.array(t_all)
        y_all = np.array(y_all)
        
        # Split into differential and algebraic
        x = y_all[:, :n_states].T
        z = y_all[:, n_states:].T
        
        # Compute outputs
        if self.h_funcs:
            n_outputs = len(self.h_funcs)
        else:
            n_outputs = n_states
        
        y_out = np.zeros((n_outputs, len(t_all)))
        for i in range(len(t_all)):
            y_out[:, i] = self.eval_h(t_all[i], x[:, i], z[:, i])
        
        if verbose:
            print(f"\nSimulation complete!")
            print(f"  Final time: {t_all[-1]:.6f}")
            print(f"  Total events: {len(event_times)}")
            print(f"  Early termination: {early_termination}")
        
        result = {
            't': t_all,
            'x': x,
            'z': z,
            'y': y_out,
            'event_times': event_times,
            'event_indices': event_indices,
            'event_vars_changed': event_vars_changed,
            'early_termination': early_termination,
            'state_names': self.state_names,
            'alg_names': self.alg_names,
            'output_names': self.output_names if self.output_names else self.state_names,
        }
        
        return result

    def solve_augmented(self, 
                        t_span: Tuple[float, float],
                        rtol: float = 1e-6, 
                        atol: float = 1e-8,
                        ncp: int = None, # Added ncp to derive max_step
                        max_segments: int = None, # Added to limit output segments
                        max_points_per_seg: int = None, # Added to limit samples per segment
                        **kwargs) -> AugmentedSolution:
        """
        Solves DAE and returns the 'Natural Grid' trajectory split by events.
        This format is required for Discrete Adjoint Optimization.
        
        Args:
            ncp: If provided, used to set the maximum step size (max_step) 
                 ensuring segments have sufficient density for plotting/analysis.
                 max_step = (t_end - t_start) / ncp
            max_segments: Maximum number of trajectory segments to return.
                          Events are also filtered to include only those relevant
                          to the returned segments.
            max_points_per_seg: Maximum number of time points per segment.
                                If exceeded, the segment is truncated.
        """
        t_curr, t_end = t_span
        
        # Determine max_step if ncp is provided
        max_step = np.inf
        if ncp:
            max_step = (t_end - t_curr) / float(ncp)
        
        # 1. Initialize State
        y_curr = np.concatenate([self.x0, self.z0])
        f0 = self.eval_f(t_curr, self.x0, self.z0)
        yp_curr = np.concatenate([f0, np.zeros(len(self.z0))])
        
        # Initialize event tracking condition
        # We only trigger events when condition goes False -> True (Falling edge of zc).
        # Boundary (zc == 0) is treated as already-active so a freshly-clamped
        # state does not re-fire its own event.
        zc_vals = self.eval_zc(t_curr, self.x0, self.z0)
        cond_active_prev = (zc_vals <= 0)

        # 2. Setup Loop Storage
        segments = []
        events = []
        
        # Current Segment Buffers
        cur_t = [t_curr]
        cur_x = [self.x0]
        cur_z = [self.z0]
        cur_xp = [f0] # Store x_dot for accurate interpolation later
        
        # 3. Configure Solver
        n_total = len(y_curr)
        n_states = len(self.state_names)
        algvar_indices = np.arange(n_states, n_total)

        # Initialize IDA wrapper
        # Note: We do NOT rely on solver internal max_step options as they can be unreliable/variable across versions.
        # We enforce density by controlling the integration horizon in the loop.
        solver = dae(
            'ida',
            self.residual_ida,
            rtol=rtol, atol=atol,
            algebraic_vars_idx=algvar_indices,
            compute_initcond='yp0',
            # Event detection
            rootfn=self._root_fn_wrapper if self.n_events > 0 else None,
            nr_rootfns=self.n_events,
            **kwargs
        )
        
        # Initialize the internal memory of IDA
        solver.init_step(t_curr, y_curr, yp_curr)

        max_events = 1000 # Safety break
        event_count = 0

        while t_curr < t_end:
            # ----------------------------------------------------------------
            # CRITICAL: Control Step Size via Horizon
            # ----------------------------------------------------------------
            # If we just say solver.step(t_end), IDA takes huge steps.
            # We urge it to return sooner by setting a closer horizon.
            # solver.step(tout) integrates toward tout, but stops after one internal step.
            # However, if that internal step is huge, we still miss points.
            # BETTER: We don't force 'step(tout)' to be the end.
            # But wait, 'step(tout)' usually means "take one step towards tout". 
            # It DOES NOT mean "stop exactly at tout" unless we use 'run(tout)'.
            # BUT, changing tout *guides* the heuristic max step size in some solvers.
            
            # Actually, to guarantee points at least every max_step, we should check:
            # If the solver takes a huge step, we can't prevent it easily without options.
            # scikits.odes doesn't easily expose 'max_step'.
            
            # ALTERNATIVE: Use solver.step(t_end) but checking the result.
            # If result t_new is too far, that's bad.
            
            # Let's try passing 'max_step' to the constructor again but verified?
            # No, 'max_step_size' was ignored.
            
            # Let's try controlling the upper bound of integration time passed to step.
            # If we say step(t_curr + max_step), it CANNOT go past that.
            # This effectively limits the step size returned.
            
            next_target = min(t_end, t_curr + max_step)
            
            # scikits.odes syntax for step: returns (flag, values, ...)
            step_result = solver.step(next_target)
            
            flag = int(step_result.flag)
            t_new = step_result.values.t
            y_new = step_result.values.y
            yp_new = step_result.values.ydot

            # Extract state
            x_new = y_new[:n_states]
            z_new = y_new[n_states:]
            xp_new = yp_new[:n_states]

            # ----------------------------------------------------------------
            # CASE A: Event Detected (Flag 2)
            # ----------------------------------------------------------------
            if flag == 2:

                # 1. Capture the exact Event Time (tau)
                tau = t_new
                
                # 2. Identify Event Candidates
                triggered_candidates = []
                
                # 2. Identify events that actually crossed at this tau.
                # We always check the state directly: an event is triggered
                # iff zc[i] <= 0 at tau AND it was inactive before. Relying
                # on solver.roots.val is not reliable when several events
                # cross within one integrator step (e.g. multiple balls
                # landing at the floor in close succession): IDA may mark
                # all of them in roots.val even though only one actually
                # crossed at tau, which historically caused wrong-event
                # attribution and skipped reinits.
                zc = self.eval_zc(tau, x_new, z_new)
                triggered_candidates = [i for i in range(self.n_events)
                                        if zc[i] <= 0]

                # 3. Filter by direction: a real event requires the
                # condition to have just transitioned from inactive (zc>0)
                # to active (zc<=0).
                real_events = [idx for idx in triggered_candidates
                               if not cond_active_prev[idx]]
                
                if real_events:
                    # Handle Primary Event (Simultaneous handling could be added, prioritizing first for now)
                    event_idx = real_events[0] 
                    event_count += 1
                    
                    if event_count > max_events:
                        print(f"Warning: Maximum event count ({max_events}) reached. Terminating.")
                        break

                    # ----------------------------------------------------------------
                    # FIX: Zeno Clamping (Consecutive events without intermediate samples)
                    # ----------------------------------------------------------------
                    # We declare Zeno only when the *same* event index re-fires
                    # with a small time gap. Distinct events from different
                    # balls firing close in time (e.g. several balls landing on
                    # the floor within ~1 ms) are legitimate and must not
                    # terminate the run. A ball settling to rest, on the other
                    # hand, produces a geometric series of same-index Floor
                    # events with dt halving each step -- that is real Zeno
                    # and we stop the simulation there to keep the state
                    # physically valid (otherwise gravity drags the ball
                    # through the floor once the rebound velocity reaches 0).
                    prev_event = events[-1] if events else None
                    if (len(cur_t) <= 2
                            and prev_event is not None
                            and prev_event.event_idx == event_idx
                            and (tau - prev_event.t_event) < self.zeno_dt_threshold):
                        print(f"Warning: Zeno barrier detected (Event triggered immediately after previous event).")
                        print(f"  Terminating simulation at t={tau:.6f} to prevent infinite loop.")
                        
                        # We must finalize the current segment and return
                        # Add the event point to close the segment if not already added
                        # (If len is 2, the second point might be tau, or close to it? 
                        #  Actually cur_t was appeneded via cur_t.append(tau) further down?
                        #  No, we are inside 'if flag == 2'. cur_t currently has partial steps?
                        #  Let's check logic above: cur_t initialized at loop start.
                        #  If flag=2, we haven't appended tau yet.)
                        
                        # Add the event point to close the segment ONLY if time advanced
                        if len(cur_t) == 0 or tau > cur_t[-1] + 1e-9:
                            cur_t.append(tau)
                            cur_x.append(x_new)
                            cur_z.append(z_new)
                            cur_xp.append(xp_new)
                        else:
                            # If zero step, we can update the last point, but better to just keep the consistent one
                            # Updating might introduce a jump. Let's strictly only append if advanced.
                            # However, to capture the event time exactly:
                            cur_t[-1] = tau
                            # Do NOT update states if it implies a jump. 
                            # But x_new is the state at tau. 
                            # If x_new differs from x_old, we have a problem.
                            # For safety in Zeno, let's just terminate with what we have.
                            pass
                        
                        # Add partial segment (ensure at least 1 point)
                        if len(cur_t) > 0:
                            segments.append(TrajectorySegment(
                                np.array(cur_t), np.array(cur_x), np.array(cur_z), np.array(cur_xp)
                            ))
                        
                        # Return current solution (truncated)
                        return self._finalize_augmented_solution(segments, events, max_segments, max_points_per_seg)


                    # 4. Finalize Previous Segment
                    # Add the event node (x-) to the current segment end
                    # Only add if we moved forward in time
                    if len(cur_t) == 0 or tau > cur_t[-1] + 1e-14:
                        cur_t.append(tau)
                        cur_x.append(x_new)
                        cur_z.append(z_new)
                        cur_xp.append(xp_new)
                    else:
                        # We are at the same time point (or extremely close), override the last point
                        # to ensure exact event time capture
                        cur_t[-1] = tau
                        cur_x[-1] = x_new
                        cur_z[-1] = z_new
                        cur_xp[-1] = xp_new
                    
                    # Store segment
                    segments.append(TrajectorySegment(
                        np.array(cur_t), np.array(cur_x), np.array(cur_z), np.array(cur_xp)
                    ))
                    
                    # CHECK FOR EARLY STOP: Max Segments
                    # If we just finished a segment, check if we reached the limit
                    if max_segments is not None and len(segments) >= max_segments:
                         # We reached the limit. We do NOT process the event (jump) or start a new segment.
                         # We stop here.
                         return self._finalize_augmented_solution(segments, events, max_segments, max_points_per_seg)

                    # 5. Perform Reinitialization (Jump)
                    # We need x- (x_new) and x+ (reinitialized)
                    x_pre, z_pre = x_new.copy(), z_new.copy()
                    
                    # Apply your existing logic
                    x_post, z_post = self._apply_reinit(
                        event_idx, tau, x_new, z_new, x_pre, z_pre
                    )
                    
                    # 6. Store Event Info
                    events.append(EventInfo(
                        t_event=tau,
                        event_idx=event_idx,
                        x_pre=x_pre, z_pre=z_pre,
                        x_post=x_post, z_post=z_post
                    ))

                    # 6b. Rate-based Zeno guard: if many recent events fit in
                    # a tiny time window, this is sustained chatter (a ball
                    # settling onto the floor, or a small group of balls
                    # alternating bounces). Beyond this point velocities
                    # collapse and the simplified physics (no normal force)
                    # would let gravity drag balls through the boundary, so
                    # we stop the simulation here.
                    K = self.zeno_window_events
                    if len(events) > K:
                        recent_span = events[-1].t_event - events[-1 - K].t_event
                        if recent_span < self.zeno_window_seconds:
                            print(f"Warning: Zeno chatter detected "
                                  f"({K + 1} events within {recent_span:.3e}s). "
                                  f"Terminating simulation at t={tau:.6f}.")
                            # Close current segment (segment was started
                            # right after the previous event).
                            segments.append(TrajectorySegment(
                                np.array(cur_t), np.array(cur_x),
                                np.array(cur_z), np.array(cur_xp)
                            ))
                            return AugmentedSolution(segments=segments, events=events)
                    
                    # 7. Update State & Reset Solver
                    t_curr = tau
                    y_curr = np.concatenate([x_post, z_post])
                    
                    # Re-evaluate derivatives f(x+, z+) consistent with new state
                    f_post = self.eval_f(t_curr, x_post, z_post)
                    yp_curr = np.concatenate([f_post, np.zeros(len(z_post))])
                    
                    # Update active condition map using POST-EVENT state.
                    # Boundary (zc == 0) is treated as still-active, otherwise
                    # a position-clamp reinit (which sets zc to exactly 0) would
                    # let the same event re-fire on the next root scan.
                    zc_post = self.eval_zc(t_curr, x_post, z_post)
                    cond_active_prev = (zc_post <= 0)
                    
                    # Restart solver
                    solver.init_step(t_curr, y_curr, yp_curr)
                    
                    # Start buffers for new segment
                    cur_t = [t_curr]
                    cur_x = [x_post]
                    cur_z = [z_post]
                    cur_xp = [f_post]
                    
                else:
                    # Flag 2 but no valid failing edge (Rising edge or spurious)
                    # Treat as a step, update state, but do not reinit
                    # We accept the point
                    # Treat as a step, update state, but do not reinit
                    # We accept the point if unique
                    if len(cur_t) == 0 or tau > cur_t[-1] + 1e-14:
                        cur_t.append(tau)
                        cur_x.append(x_new)
                        cur_z.append(z_new)
                        cur_xp.append(xp_new)
                        t_curr = tau
                    
                    # Update active conditions based on current state (so we correctly track being 'active')
                    # e.g. if we just crossed to positive, cond_active_prev becomes False.
                    # Boundary (zc == 0) is treated as still-active.
                    zc_new = self.eval_zc(t_curr, x_new, z_new)
                    cond_active_prev = (zc_new <= 0)
                    
                    # Note: We do NOT call init_step here, we let the solver continue from this point.
                    # IDA (scikits.odes) state should be valid to continue.

            # ----------------------------------------------------------------
            # CASE B: Standard Step (Flag 0)
            # ----------------------------------------------------------------
            elif flag == 0:
                # Append to current buffers if time advanced
                if t_new > cur_t[-1] + 1e-14:
                    cur_t.append(t_new)
                    cur_x.append(x_new)
                    cur_z.append(z_new)
                    cur_xp.append(xp_new)
                    t_curr = t_new

                # Check target stop condition
                if t_curr >= t_end:
                    break

                # Refresh cond_active_prev from the current state. Without this,
                # an event that was clamped (zc=0, marked active) will never be
                # un-marked when the state lifts off the boundary, because IDA
                # does not reliably emit a +1 root when the function starts at
                # exactly 0. As a result the same event would never re-fire on
                # subsequent crossings (e.g. a ball failing to bounce after the
                # final small bounce of a settling sequence).
                zc_new = self.eval_zc(t_curr, x_new, z_new)
                cond_active_prev = (zc_new <= 0)

                t_curr = t_new

                # Append to current buffers
                cur_t.append(t_new)
                cur_x.append(x_new)
                cur_z.append(z_new)
                cur_xp.append(xp_new)
                
            else:
                print(f"Solver Error: Flag {flag}")
                break

        # Finalize the last segment
        segments.append(TrajectorySegment(
            np.array(cur_t), np.array(cur_x), np.array(cur_z), np.array(cur_xp)
        ))
        
        return self._finalize_augmented_solution(segments, events, max_segments, max_points_per_seg)

    def _finalize_augmented_solution(self, segments, events, max_segments, max_points_per_seg):
        """Helper to post-process segments and events before returning."""
        
        # 1. Truncate segments if needed
        if max_segments is not None and len(segments) > max_segments:
            segments = segments[:max_segments]

        # 2. Modify segments (Gap Logic + Max Points Truncation + Duplicate Filter)
        modified_segments = []
        tol_dup = 1e-9
        
        for i, seg in enumerate(segments):
            # Start with original data
            t, x, z, xp = seg.t, seg.x, seg.z, seg.xp
            
            # 1. Filter duplicates (dt < tol)
            if len(t) > 1:
                # Keep first point, and any point where t[k] > t[k-1] + tol
                valid_mask = np.concatenate(([True], np.diff(t) > tol_dup))
                t = t[valid_mask]
                x = x[valid_mask]
                z = z[valid_mask]
                xp = xp[valid_mask]
            
            # 2. Apply truncation if configured
            if max_points_per_seg is not None and len(t) > max_points_per_seg:
                 t = t[:max_points_per_seg]
                 x = x[:max_points_per_seg]
                 z = z[:max_points_per_seg]
                 xp = xp[:max_points_per_seg]
            
            
            modified_segments.append(TrajectorySegment(t, x, z, xp))
        
        segments = modified_segments

        # 3. Filter events 
        # Ensure no events exist after the end of the last retained segment
        # In particular, we cannot have an event without a following segment (dangling event),
        # as this leads to undefined adjoint residuals (no x_post variables to enforce jump).
        if segments:
            # We expect N segments and N-1 events.
            # If we have N events, the last one is dangling.
            expected_events = len(segments) - 1
            if len(events) > expected_events:
                 events = events[:expected_events]
        else:
            events = []

        return AugmentedSolution(segments, events)



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
