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

import os
import jax
import jax.numpy as jnp
from jax import jit, jacfwd, value_and_grad, custom_jvp, vmap
from jax.scipy.linalg import lu_factor, lu_solve
import numpy as np
import re
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from functools import partial

# Enable float64 for better precision
jax.config.update("jax_enable_x64", True)

# Default to CPU unless the caller already set `JAX_PLATFORMS` (e.g.
# "cuda" / "gpu"). Without this, any script that imports this module
# before pinning the platform will silently latch whatever backend
# JAX auto-detects — i.e. GPU if CUDA is visible. The env-var check
# preserves the opt-in to GPU for users who set it deliberately.
if not os.environ.get("JAX_PLATFORMS"):
    jax.config.update("jax_platform_name", "cpu")

# Persistent JIT compilation cache: subsequent runs reusing the same
# (n_states, n_alg, n_params, dt, t-grid) skip the seconds-long XLA
# compile step. Override location via JAX_COMPILATION_CACHE_DIR.
_jax_cache_dir = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "jax_dae_optim"),
)
os.makedirs(_jax_cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _jax_cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# Names that DAE specs must not collide with for the fused codegen path.
_PROTECTED_NAMES = frozenset({
    't', 'x', 'z', 'p', 'jnp', 'time',
    'exp', 'log', 'log10', 'sqrt', 'abs',
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
    'sinh', 'cosh', 'tanh', 'sign', 'floor', 'ceil',
    'min', 'max', 'sigmoid',
})

_MATH_FUNCS = {
    'exp': jnp.exp, 'log': jnp.log, 'log10': jnp.log10,
    'sqrt': jnp.sqrt, 'abs': jnp.abs,
    'sin': jnp.sin, 'cos': jnp.cos, 'tan': jnp.tan,
    'asin': jnp.arcsin, 'acos': jnp.arccos, 'atan': jnp.arctan,
    'sinh': jnp.sinh, 'cosh': jnp.cosh, 'tanh': jnp.tanh,
    'sign': jnp.sign, 'floor': jnp.floor, 'ceil': jnp.ceil,
    'min': jnp.minimum, 'max': jnp.maximum,
    'sigmoid': jax.nn.sigmoid,
}


def _normalize_expr(expr: str) -> str:
    """Modelica-style ^ -> **, `time` -> `t`."""
    return re.sub(r'\btime\b', 't', expr.replace('^', '**'))


def _build_fused_function(
    fn_name: str,
    state_names: List[str],
    alg_var_names: List[str],
    param_names: List[str],
    rhs_exprs: List[str],
):
    """Codegen one fused Python function returning the whole residual /
    rhs vector via `jnp.stack`. Returns `None` if the spec collides with
    a protected name (caller falls back to the per-equation `eval` path).
    """
    if any(n in _PROTECTED_NAMES for n in state_names + alg_var_names + param_names):
        return None
    if not rhs_exprs:
        src = (
            f"def {fn_name}(t, x, z, p):\n"
            "    return jnp.zeros(0, dtype=jnp.float64)\n"
        )
    else:
        lines = [f"def {fn_name}(t, x, z, p):"]
        for i, n in enumerate(state_names):
            lines.append(f"    {n} = x[{i}]")
        for i, n in enumerate(alg_var_names):
            lines.append(f"    {n} = z[{i}]")
        for i, n in enumerate(param_names):
            lines.append(f"    {n} = p[{i}]")
        lines.append("    return jnp.stack([")
        for rhs in rhs_exprs:
            lines.append(f"        ({_normalize_expr(rhs)}),")
        lines.append("    ])")
        src = "\n".join(lines) + "\n"

    ns = {'jnp': jnp, **_MATH_FUNCS}
    try:
        exec(compile(src, f"<{fn_name}>", "exec"), ns)
    except (SyntaxError, NameError):
        return None
    return ns[fn_name]


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
        newton_max_iter: int = 10,
        diffrax_solver: str = 'Tsit5',
        rtol: float = 1.0e-6,
        atol: float = 1.0e-6,
        dtmax: Optional[float] = None,
        diffrax_max_steps: int = 16384,
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
        self.diffrax_solver = diffrax_solver
        self.rtol = rtol
        self.atol = atol
        self.dtmax = dtmax
        self.diffrax_max_steps = diffrax_max_steps

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

        def eval_f_slow(t, x, z, p):
            """Per-equation eval fallback (only when codegen can't fire)."""
            namespace = build_namespace(t, x, z, p)
            dxdt = jnp.zeros(n_states, dtype=jnp.float64)
            for state_name, expr in f_eqs:
                idx = state_names.index(state_name)
                val = eval_expr(expr, namespace)
                dxdt = dxdt.at[idx].set(val)
            return dxdt

        def eval_g_slow(t, x, z, p):
            if not g_eqs:
                return jnp.zeros(0, dtype=jnp.float64)
            namespace = build_namespace(t, x, z, p)
            residuals = [eval_expr(expr, namespace) for expr in g_eqs]
            return jnp.array(residuals)

        def eval_h_slow(t, x, z, p):
            if not h_eqs:
                return x
            namespace = build_namespace(t, x, z, p)
            outputs = [eval_expr(expr, namespace) for _, expr in h_eqs]
            return jnp.array(outputs)

        # Codegen fused versions: builds one Python function per residual
        # group whose body is a single `jnp.stack([...])` over inlined RHS
        # expressions. The fused graph is O(n_eq) smaller for XLA, which
        # dramatically cuts first-JIT compile time on bigger DAEs.
        # f-equations: re-order to match `state_names` so the result aligns
        # with the differential-state layout regardless of source order.
        f_rhs_by_state = dict(f_eqs)
        f_rhs_ordered = [f_rhs_by_state.get(n, '0.0') for n in state_names]

        eval_f_fast = _build_fused_function(
            "_fused_f", state_names, alg_var_names, param_names, f_rhs_ordered
        )
        eval_g_fast = _build_fused_function(
            "_fused_g", state_names, alg_var_names, param_names, list(g_eqs)
        )
        h_rhs = [expr for _, expr in h_eqs]
        eval_h_fast = (
            _build_fused_function(
                "_fused_h", state_names, alg_var_names, param_names, h_rhs
            )
            if h_eqs else None
        )

        eval_f = eval_f_fast if eval_f_fast is not None else eval_f_slow
        eval_g = eval_g_fast if eval_g_fast is not None else eval_g_slow
        if h_eqs:
            eval_h = eval_h_fast if eval_h_fast is not None else eval_h_slow
        else:
            # No `h` block: outputs == states (unchanged behavior).
            eval_h = eval_h_slow

        self._eval_f = eval_f
        self._eval_g = eval_g
        self._eval_h = eval_h

        # Hoist Jacobian closures: trace these once at construction time
        # rather than re-creating from inside Newton each iter. The
        # IFT jvp below uses `jax.jvp` through `eval_g` directly, so we
        # only need the square Jacobian `dg/dz` here.
        _dg_dz_fn = jacfwd(eval_g, argnums=2)

        # Modified-Newton (chord) solver for the algebraic constraint:
        # factor `dg/dz` once at the initial guess and reuse the LU across
        # all iterations. For nearly-linear DAEs (e.g. circuit Cauer this
        # converges in 1 iter) this collapses the inner loop to one
        # back-substitution per step. Convergence drops from quadratic to
        # linear, but per-iter cost drops by an order of magnitude — net
        # speedup on every DAE we've measured.
        # `solve_algebraic` is wrapped in a `custom_vjp`, so AD never
        # traces through this loop; we get to use lu_factor/lu_solve
        # without worrying about backward-mode rules for them.
        eye_alg = jnp.eye(n_alg) if n_alg > 0 else None

        def solve_algebraic_newton_fwd(t, x, p, z_init):
            if n_alg == 0:
                return jnp.zeros(0, dtype=jnp.float64)

            jac_g_z0 = _dg_dz_fn(t, x, z_init, p) + 1e-12 * eye_alg
            lu_piv = lu_factor(jac_g_z0)

            def chord_step(z_curr, _):
                g_val = eval_g(t, x, z_curr, p)
                delta_z = lu_solve(lu_piv, -g_val)
                return z_curr + delta_z, jnp.linalg.norm(g_val)

            z_final, _ = jax.lax.scan(chord_step, z_init, None,
                                      length=newton_max_iter)
            return z_final

        # Implicit-function-theorem rule, expressed as a `custom_jvp`.
        # We use jvp (not vjp) because diffrax's *implicit* Runge-Kutta
        # solvers (Kvaerno*, etc.) take a forward-mode derivative of the
        # vector field to build their Newton iteration; a `custom_vjp`
        # leaves jvp undefined and breaks them. Reverse-mode AD is
        # synthesized by JAX from this jvp via tape transposition, so
        # gradients of the loss continue to flow analytically through
        # the algebraic solve.
        #
        # IFT: at convergence, g(t, x, z, p) = 0. Differentiating along
        # any tangent direction:
        #     ∂g/∂t·dt + ∂g/∂x·dx + ∂g/∂z·dz + ∂g/∂p·dp = 0
        # ⇒ dz = -(∂g/∂z)^{-1} · (∂g/∂t·dt + ∂g/∂x·dx + ∂g/∂p·dp)
        # `z_init` does not appear in the IFT — at convergence z is
        # independent of the initial guess, so its tangent contribution
        # is zero.
        @custom_jvp
        def solve_algebraic(t, x, p, z_init):
            return solve_algebraic_newton_fwd(t, x, p, z_init)

        @solve_algebraic.defjvp
        def solve_algebraic_jvp(primals, tangents):
            t, x, p, z_init = primals
            dt, dx, dp, _ = tangents
            z = solve_algebraic_newton_fwd(t, x, p, z_init)
            if n_alg == 0:
                return z, jnp.zeros_like(z)

            dg_dz = _dg_dz_fn(t, x, z, p) + 1e-12 * eye_alg
            # ∂g/∂t·dt + ∂g/∂x·dx + ∂g/∂p·dp via a single jvp through g
            # (cheaper than building the three Jacobians separately).
            _, dg_dot = jax.jvp(
                lambda tt, xx, pp: eval_g(tt, xx, z, pp),
                (t, x, p),
                (dt, dx, dp),
            )
            dz = -jnp.linalg.solve(dg_dz, dg_dot)
            return z, dz

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

        # ----------------------------------------------------------------
        # Simulation function (differentiable w.r.t. parameters).
        #
        # Three integration backends:
        #   - 'rk4'     : fixed-step RK4 with per-stage Newton on `g`
        #   - 'euler'   : fixed-step explicit Euler with one Newton on `g`
        #   - 'diffrax' : adaptive-step diffrax solver (Tsit5 / Dopri5 /
        #                 Dopri8 / Heun / Kvaerno3 / Kvaerno5). The
        #                 algebraic constraint is still resolved by the
        #                 same `solve_algebraic` (custom_vjp + chord
        #                 Newton + IFT backward), so AD gradients through
        #                 the diffrax adjoint stay analytically sharp on
        #                 the `g` side. For stiff systems use Kvaerno5
        #                 (singly-diagonally-implicit RK).
        # ----------------------------------------------------------------
        if self.solver_method == 'diffrax':
            import diffrax  # imported lazily so non-diffrax users skip it

            _solver_map = {
                'Tsit5': diffrax.Tsit5,
                'Dopri5': diffrax.Dopri5,
                'Dopri8': diffrax.Dopri8,
                'Heun': diffrax.Heun,
                'Kvaerno3': diffrax.Kvaerno3,
                'Kvaerno5': diffrax.Kvaerno5,
            }
            if self.diffrax_solver not in _solver_map:
                raise ValueError(
                    f"Unknown diffrax_solver {self.diffrax_solver!r}; "
                    f"valid: {sorted(_solver_map)}"
                )
            _diff_solver = _solver_map[self.diffrax_solver]()

            _pid_kwargs = dict(rtol=self.rtol, atol=self.atol)
            if self.dtmax is not None:
                _pid_kwargs['dtmax'] = float(self.dtmax)
            _controller = diffrax.PIDController(**_pid_kwargs)

            # Cold-start z guess for every internal substep. The chord
            # Newton inside `solve_algebraic` converges in ~1-2 iters for
            # near-linear DAEs, so this is cheap; for nonlinear DAEs it
            # may need bumping `newton_max_iter`.
            def _vector_field(t, x, args):
                p_, z_guess = args
                z = solve_algebraic(t, x, p_, z_guess)
                return eval_f(t, x, z, p_)

            _term = diffrax.ODETerm(_vector_field)
            _max_steps = int(self.diffrax_max_steps)

            # Decide once whether the per-output-point reconstruct is
            # actually needed. When `h` is the identity (no `h` block in
            # the spec) `y == x`, so we can skip the vmapped algebraic
            # solves entirely — that's `len(t_array)` solves per call,
            # both forward and through the gradient. For specs with a
            # real `h`, we still need fresh `z` and `y` at every saved
            # point.
            _h_is_identity = not h_eqs
            _n_alg_static = n_alg

            def simulate(x0, z0, t_array, p):
                sol = diffrax.diffeqsolve(
                    _term, _diff_solver,
                    t0=t_array[0], t1=t_array[-1],
                    dt0=t_array[1] - t_array[0],
                    y0=x0,
                    args=(p, z0),
                    saveat=diffrax.SaveAt(ts=t_array),
                    stepsize_controller=_controller,
                    max_steps=_max_steps,
                )
                x_traj = sol.ys  # (n_steps, n_states)

                if _h_is_identity:
                    # y == x; emit a zero placeholder for z (loss path
                    # doesn't read it, public `simulate()` accepts it as
                    # "unused" since the explicit RK4 path already only
                    # ever stored a warm-start carry, not the on-manifold
                    # z at saved times).
                    z_traj = jnp.zeros(
                        (t_array.shape[0], _n_alg_static),
                        dtype=x_traj.dtype,
                    )
                    return x_traj, z_traj, x_traj

                def _reconstruct(t, x):
                    z = solve_algebraic(t, x, p, z0)
                    y = eval_h(t, x, z, p)
                    return z, y

                z_traj, y_traj = vmap(_reconstruct)(t_array, x_traj)
                return x_traj, z_traj, y_traj

        else:
            def simulate(x0, z0, t_array, p):
                """Simulate DAE - differentiable w.r.t. p (fixed-step)."""
                dt = t_array[1] - t_array[0]

                def scan_step(carry, t):
                    x, z = carry
                    if self.solver_method == 'rk4':
                        x_next, z_next = rk4_step(t, x, p, z, dt)
                    else:
                        x_next, z_next = euler_step(t, x, p, z, dt)
                    y = eval_h(t, x, z, p)
                    return (x_next, z_next), (x, z, y)

                _, (x_traj, z_traj, y_traj) = jax.lax.scan(
                    scan_step, (x0, z0), t_array
                )
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
