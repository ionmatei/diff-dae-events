"""
Explicit Discrete Adjoint Optimizer for DAEs with Events.

Uses the discretization viewpoint where the entire hybrid trajectory is treated
as a solution to a large nonlinear algebraic system R(U, θ) = 0.

The unknown vector U contains:
- All state values at grid points: {Y_{i,k}} for segment i, time point k
- All event times: {τ_i} for i = 1..M

The residual R includes:
- Trapezoidal timestep residuals within each segment
- Event guard constraints (ψ = 0 at event surface)
- Event reset constraints (state jump equations)

Gradient computation via discrete adjoint:
    R_U^T λ = J_U^T
    ∇_θ J = J_θ - R_θ^T λ
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Tuple, Dict, List, NamedTuple, Optional
import numpy as np
import time

jax.config.update("jax_enable_x64", True)

# =============================================================================
# Equation Compiler
# =============================================================================

def compile_equations_to_jax(eqn_strings, state_names, alg_names, param_names, extra_args=None):
    """
    Parses string equations into a single JAX-jittable function.
    Eliminates dictionary creation and eval() overhead during runtime.

    Args:
        eqn_strings: List of equation strings to compile
        state_names: List of state variable names
        alg_names: List of algebraic variable names
        param_names: List of parameter names
        extra_args: Dict mapping {var_name: replacement_code} for custom variables (e.g. prev_x)

    Returns:
        Callable fn(t, x, z, p) -> jnp.array
    """
    if not eqn_strings:
        # Return a dummy function returning empty array
        def dummy_fn(t, x, z, p):
            return jnp.array([])
        return dummy_fn

    import re

    # Build substitution map
    # Sort names by length (descending) to avoid partial matches on prefixes
    subs = []
    
    # Custom args first
    if extra_args:
        for name, repl in extra_args.items():
            subs.append((name, repl))

    # States: x[i]
    for i, name in enumerate(state_names):
        subs.append((name, f"x[{i}]"))
    
    # Algebraic: z[i]
    for i, name in enumerate(alg_names):
        subs.append((name, f"z[{i}]"))
        
    # Parameters: p[i]
    for i, name in enumerate(param_names):
        subs.append((name, f"p[{i}]"))
        
    # Time
    subs.append(('time', 't'))
    subs.append(('t', 't'))

    # Sort substitutions by length of variable name descending
    # This prevents replacing 'v' inside 'prev_v' if 'prev_v' is also in the list
    subs.sort(key=lambda item: len(item[0]), reverse=True)

    # Function template
    # We use a closure pattern via exec to bind jax numpy (jnp) and math functions
    
    # Build the returned list string: "[expr1, expr2, ...]"
    processed_eqns = []
    
    # Math namespace keys to protect from substitution
    # e.g. don't replace 'min' in 'min(a, b)' with a variable named 'min'
    math_funcs = {
        'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 
        'exp', 'log', 'sqrt', 'abs', 'pow', 'min', 'max'
    }
    
    for eq in eqn_strings:
        # We need to robustly replace variables
        # Using regex \bNAME\b
        
        processed_eq = eq
        for name, repl in subs:
            # Skip if name matches a math function
            if name in math_funcs:
                continue
                
            # Regex replace full words only
            pattern = r'(?<!\.)\b' + re.escape(name) + r'\b'
            processed_eq = re.sub(pattern, repl, processed_eq)
            
        processed_eqns.append(processed_eq)
        
    # Join into a single list expression
    return_expr = "[" + ", ".join(processed_eqns) + "]"
    
    # Generate function source
    # We include 'jnp' in the closure namespace
    fname = f"compiled_fn_{abs(hash(return_expr))}"
    
    source = f"""
def {fname}(t, x, z, p):
    return jnp.array({return_expr})
"""
    
    # Define execution namespace with JAX and math functions
    exec_ns = {
        'jnp': jnp,
        'sin': jnp.sin, 'cos': jnp.cos, 'tan': jnp.tan,
        'sinh': jnp.sinh, 'cosh': jnp.cosh, 'tanh': jnp.tanh,
        'exp': jnp.exp, 'log': jnp.log, 'sqrt': jnp.sqrt,
        'abs': jnp.abs, 'pow': jnp.power,
        'min': jnp.minimum, 'max': jnp.maximum,
    }
    
    # Compile
    try:
        exec(source, exec_ns)
        return exec_ns[fname]
    except Exception as e:
        print(f"Error compiling equations: {e}")
        print(f"Source:\n{source}")
        raise

from .dae_solver import DAESolver, AugmentedSolution


# =============================================================================
# Data Structures
# =============================================================================

class DiscreteTrajectory(NamedTuple):
    """Flattened discrete trajectory for adjoint computation."""
    # Segment data (list of arrays, one per segment)
    t_segments: List[jnp.ndarray]      # Time grids per segment
    y_segments: List[jnp.ndarray]      # States [x, z] per segment, shape (N_i, n_total)

    # Event data
    tau: jnp.ndarray                   # Event times, shape (M,)
    event_indices: List[int]           # Which event triggered at each transition
    y_pre: List[jnp.ndarray]           # States just before each event
    y_post: List[jnp.ndarray]          # States just after each event


# =============================================================================
# Interpolation (reused from original)
# =============================================================================

def linear_interpolate_segment(t_query, t_seg, y_seg):
    """Linear interpolation on a segment."""
    def interp_single(t_q):
        idx = jnp.searchsorted(t_seg, t_q, side='right') - 1
        idx = jnp.clip(idx, 0, len(t_seg) - 2)

        t0, t1 = t_seg[idx], t_seg[idx + 1]
        y0, y1 = y_seg[idx], y_seg[idx + 1]

        h = t1 - t0
        h_safe = jnp.where(h < 1e-12, 1e-12, h)
        s = jnp.clip((t_q - t0) / h_safe, 0.0, 1.0)

        return jnp.where(h < 1e-12, y0, y0 * (1 - s) + y1 * s)

    return vmap(interp_single)(t_query)


def predict_outputs_from_x(aug_sol_jax, target_times, n_outputs, eval_h_fn, p_all, blend_sharpness=100.0):
    """
    Predict outputs at target times using segment-based interpolation.

    For each target time, finds the correct segment (using event times as boundaries)
    and interpolates within that segment. Uses very sharp sigmoid blending to
    approximate hard segment selection while maintaining differentiability.

    Args:
        aug_sol_jax: Dict with 'segments' and 'events'
        target_times: Query times
        n_outputs: Number of outputs from h function
        eval_h_fn: Function h(t, x, z, p) -> y
        p_all: Full parameter vector
        blend_sharpness: Sigmoid sharpness (higher = sharper segment boundaries)
    """
    segments = aug_sol_jax['segments']
    events = aug_sol_jax['events']
    n_segments = len(segments)

    if n_segments == 0:
        return jnp.zeros((len(target_times), n_outputs))

    M = len(target_times)
    y_pred = jnp.zeros((M, n_outputs))
    total_weight = jnp.zeros((M, 1))

    # Build event time boundaries
    # Segment i covers: events[i-1].tau (or t_start) to events[i].tau (or t_end)
    for i, seg in enumerate(segments):
        t_seg_start = seg['t'][0]
        t_seg_end = seg['t'][-1]

        # Determine effective boundaries based on events
        if i == 0:
            t_lower = t_seg_start
        else:
            t_lower = jnp.asarray(events[i-1]['tau'])

        if i == n_segments - 1:
            t_upper = t_seg_end
        else:
            t_upper = jnp.asarray(events[i]['tau'])

        # Clamp query times to segment grid range for interpolation
        t_clamped = jnp.clip(target_times, t_seg_start, t_seg_end)

        # Interpolate x and z
        x_interp = linear_interpolate_segment(t_clamped, seg['t'], seg['x'])
        z_interp = linear_interpolate_segment(t_clamped, seg['t'], seg['z']) if seg['z'].size > 0 else jnp.zeros((M, 0))

        # Apply output function h(t, x, z, p)
        def apply_h(t_x_z):
            n_x = x_interp.shape[1]
            t_q = t_x_z[0]
            x_q = t_x_z[1:1+n_x]
            z_q = t_x_z[1+n_x:]
            return eval_h_fn(t_q, x_q, z_q, p_all)

        if z_interp.size > 0:
            inputs = jnp.concatenate([t_clamped[:, None], x_interp, z_interp], axis=1)
        else:
            inputs = jnp.concatenate([t_clamped[:, None], x_interp], axis=1)

        h_interp = vmap(apply_h)(inputs)

        # Sharp mask: 1 if t_lower <= t < t_upper, 0 otherwise
        # Use very sharp sigmoid to approximate step function
        mask_lower = jax.nn.sigmoid(blend_sharpness * (target_times - t_lower))
        mask_upper = jax.nn.sigmoid(blend_sharpness * (t_upper - target_times))
        mask = mask_lower * mask_upper

        mask = mask[:, None]
        y_pred = y_pred + mask * h_interp
        total_weight = total_weight + mask

    return y_pred / (total_weight + 1e-8)


# =============================================================================
# Main Optimizer Class
# =============================================================================

class DAEOptimizerExplicitAdjoint:
    """
    Discrete adjoint optimizer using explicit residual formulation.

    The key insight: treat the entire discretized trajectory as unknowns U
    satisfying R(U, θ) = 0, then use implicit differentiation for gradients.
    """

    def __init__(
        self,
        dae_data: Dict,
        optimize_params: List[str],
        solver: DAESolver = None,
        verbose: bool = True
    ):
        self.dae_data = dae_data
        self.optimize_params = optimize_params
        self.verbose = verbose

        # Create solver if not provided
        if solver is None:
            self.solver = DAESolver(dae_data, verbose=verbose)
        else:
            self.solver = solver

        # Parameter setup
        self.param_names = [p['name'] for p in dae_data['parameters']]
        self.n_params_total = len(self.param_names)
        self.optimize_indices = [self.param_names.index(p) for p in optimize_params]
        self.n_opt_params = len(optimize_params)

        self.p_all = np.array([p['value'] for p in dae_data['parameters']])
        self.p_opt = np.array([self.p_all[i] for i in self.optimize_indices])

        # State dimensions
        self.n_states = len(dae_data['states'])
        self.n_alg = len(dae_data.get('alg_vars', []))
        self.n_total = self.n_states + self.n_alg

        # Compile JAX functions
        self._compile_jax_functions()

        if verbose:
            print(f"Explicit Adjoint Optimizer initialized")
            print(f"  States: {self.n_states}, Algebraic: {self.n_alg}")
            print(f"  Optimizing {self.n_opt_params} parameters: {optimize_params}")

    def _compile_jax_functions(self):
        """Compile JAX-differentiable versions of DAE functions."""

        # Compile h_funcs: if h is not defined, default to state names
        h_eqs = self.dae_data.get('h', None)
        if h_eqs:
            self.h_funcs = []
            for eq in h_eqs:
                if '=' in eq:
                    _, rhs = eq.split('=', 1)
                    self.h_funcs.append(rhs.strip())
                else:
                    self.h_funcs.append(eq)
        else:
            # Default: outputs are the differential states
            self.h_funcs = self.solver.state_names.copy()
            
        self.n_outputs = len(self.h_funcs)

        # f(t, x, z, p) - differential equations
        self._eval_f = compile_equations_to_jax(
            self.solver.f_funcs, 
            self.solver.state_names, 
            self.solver.alg_names, 
            self.param_names
        )

        # g(t, x, z, p) - algebraic equations
        self._eval_g = compile_equations_to_jax(
            self.solver.g_funcs, 
            self.solver.state_names, 
            self.solver.alg_names, 
            self.param_names
        )

        # Zero-crossing functions
        self._eval_zc = compile_equations_to_jax(
            self.solver.zc_funcs if self.solver.zc_funcs else [], 
            self.solver.state_names, 
            self.solver.alg_names, 
            self.param_names
        )

        # h(t, x, z, p) - output function
        self._eval_h = compile_equations_to_jax(
            self.h_funcs,
            self.solver.state_names,
            self.solver.alg_names,
            self.param_names
        )
        
        # Optimization: Identity function if h is just state names
        if self.h_funcs == self.solver.state_names:
            def eval_h_identity(t, x, z, p):
                return x
            self._eval_h = eval_h_identity

        # Jump map
        # Pre-compile jump residual functions
        self.jump_residual_funcs = []
        if hasattr(self.solver, 'event_reinit_exprs'):
            import re
            
            for i, reinit_expr in enumerate(self.solver.event_reinit_exprs):
                var_name = self.solver.event_reinit_var_names[i]
                
                # Replace prev(name) with prev_name
                reinit_modified = re.sub(r'prev\(\s*(\w+)\s*\)', r'prev_\1', reinit_expr)
                
                # Manual compilation for jump residual
                subs = []
                for j, name in enumerate(self.solver.state_names):
                    subs.append((f"prev_{name}", f"x[{j}]")) # Maps to x argument (x_pre)
                for j, name in enumerate(self.solver.alg_names):
                    subs.append((f"prev_{name}", f"z[{j}]")) # Maps to z argument (z_pre)
                for j, name in enumerate(self.param_names):
                    subs.append((name, f"p[{j}]"))
                
                subs.append((var_name, "val_new")) # The new value we solve for
                subs.append(('time', 't'))
                subs.append(('t', 't'))
                
                subs.sort(key=lambda item: len(item[0]), reverse=True)
                
                math_funcs = {
                    'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 
                    'exp', 'log', 'sqrt', 'abs', 'pow', 'min', 'max'
                }
                
                processed_eq = reinit_modified
                for name, repl in subs:
                    if name in math_funcs: continue
                    pattern = r'(?<!\.)\b' + re.escape(name) + r'\b'
                    processed_eq = re.sub(pattern, repl, processed_eq)
                
                fname = f"jump_res_{i}_{abs(hash(processed_eq))}"
                source = f"""
def {fname}(t, x, z, p, val_new):
    return jnp.array([{processed_eq}])[0]
"""
                exec_ns = {
                    'jnp': jnp,
                    'sin': jnp.sin, 'cos': jnp.cos, 'tan': jnp.tan,
                    'sinh': jnp.sinh, 'cosh': jnp.cosh, 'tanh': jnp.tanh,
                    'exp': jnp.exp, 'log': jnp.log, 'sqrt': jnp.sqrt,
                    'abs': jnp.abs, 'pow': jnp.power,
                    'min': jnp.minimum, 'max': jnp.maximum,
                }
                try:
                    exec(source, exec_ns)
                    self.jump_residual_funcs.append(exec_ns[fname])
                except Exception as e:
                    print(f"Error compiling jump residual {i}: {e}")
                    raise

        def eval_jump_jax(x_pre, z_pre, tau, p_all, event_idx):
            """Compute x_post via compiled reinit equation."""
            x_post = x_pre.copy()  # JAX array copy
            
            if not self.jump_residual_funcs:
                return x_post

            var_type, var_idx = self.solver.event_reinit_vars[event_idx]
            res_fn = self.jump_residual_funcs[event_idx]
            
            # Linear solve
            val_at_0 = res_fn(tau, x_pre, z_pre, p_all, 0.0)
            val_at_1 = res_fn(tau, x_pre, z_pre, p_all, 1.0)
            
            coeff = val_at_1 - val_at_0
            safe_coeff = jnp.where(jnp.abs(coeff) < 1e-12, 1.0, coeff)
            new_val = -val_at_0 / safe_coeff
            
            if var_type == 'state':
                x_post = x_post.at[var_idx].set(new_val)
            
            return x_post

        self._eval_jump = eval_jump_jax


    def _p_all_from_p_opt(self, p_opt):
        """Construct full parameter vector from optimized subset."""
        p_all = jnp.array(self.p_all)
        for i, idx in enumerate(self.optimize_indices):
            p_all = p_all.at[idx].set(p_opt[i])
        return p_all

    def _convert_aug_sol_to_jax(self, aug_sol: AugmentedSolution) -> Dict:
        """Convert AugmentedSolution to JAX-compatible dict."""
        segments = []
        for seg in aug_sol.segments:
            y = np.concatenate([seg.x, seg.z], axis=1)
            segments.append({
                't': jnp.array(seg.t),
                'y': jnp.array(y),
                'x': jnp.array(seg.x),
                'z': jnp.array(seg.z)
            })

        events = []
        for ev in aug_sol.events:
            events.append({
                'tau': ev.t_event,
                'event_idx': ev.event_idx,
                'x_pre': jnp.array(ev.x_pre),
                'z_pre': jnp.array(ev.z_pre),
                'x_post': jnp.array(ev.x_post),
                'z_post': jnp.array(ev.z_post)
            })

        return {'segments': segments, 'events': events}

    def predict_outputs(self, aug_sol: AugmentedSolution, target_times) -> np.ndarray:
        """
        Predict outputs at target times from an AugmentedSolution.

        This is a public wrapper that converts the solution to JAX format
        and uses the same interpolation logic as the optimization loss.

        Args:
            aug_sol: AugmentedSolution from DAESolver.solve_augmented()
            target_times: Array of times at which to predict outputs

        Returns:
            y_pred: (n_times, n_outputs) array of predicted outputs
        """
        aug_sol_jax = self._convert_aug_sol_to_jax(aug_sol)
        p_all = jnp.array(self.p_all)
        target_times_jnp = jnp.array(target_times)
        y_pred = self._predict_outputs(aug_sol_jax, target_times_jnp, p_all)
        return np.array(y_pred)

    # =========================================================================
    # Trapezoidal Residuals
    # =========================================================================

    def _trapezoidal_residual(self, t_k, t_kp1, y_k, y_kp1, p_all):
        """
        Trapezoidal rule residual for one timestep.

        For differential states:
            r_x = x_{k+1} - x_k - (h/2)(f_k + f_{k+1}) = 0

        For algebraic states:
            r_z = g(t_{k+1}, x_{k+1}, z_{k+1}) = 0
        """
        h = t_kp1 - t_k

        x_k, z_k = y_k[:self.n_states], y_k[self.n_states:]
        x_kp1, z_kp1 = y_kp1[:self.n_states], y_kp1[self.n_states:]

        f_k = self._eval_f(t_k, x_k, z_k, p_all)
        f_kp1 = self._eval_f(t_kp1, x_kp1, z_kp1, p_all)

        # Differential residual
        r_x = x_kp1 - x_k - (h / 2) * (f_k + f_kp1)

        # Algebraic residual
        r_z = self._eval_g(t_kp1, x_kp1, z_kp1, p_all)

        return jnp.concatenate([r_x, r_z]) if len(r_z) > 0 else r_x

    # =========================================================================
    # Event Residuals
    # =========================================================================

    def _event_guard_residual(self, tau, x_pre, z_pre, p_all, event_idx):
        """Guard constraint: ψ(τ, y^-) = 0."""
        zc = self._eval_zc(tau, x_pre, z_pre, p_all)
        return zc[event_idx]

    def _event_reset_residual(self, tau, x_pre, z_pre, x_post, z_post, p_all, event_idx):
        """Reset constraint: x^+ = J(x^-, τ, p)."""
        x_expected = self._eval_jump(x_pre, z_pre, tau, p_all, event_idx)
        return x_post - x_expected

    # =========================================================================
    # Loss and Gradient Computation
    # =========================================================================

    def _predict_outputs(self, aug_sol_jax, target_times, p_all):
        """Predict outputs at target times using h function."""
        return predict_outputs_from_x(
            aug_sol_jax, target_times, self.n_outputs,
            self._eval_h, p_all
        )

    def compute_loss(self, aug_sol_jax, target_times, target_outputs, p_all):
        """Compute MSE loss."""
        y_pred = self._predict_outputs(aug_sol_jax, target_times, p_all)
        diff = y_pred - target_outputs
        return jnp.mean(diff ** 2)

    def compute_gradient(self, aug_sol: AugmentedSolution, target_times, target_outputs, p_opt):
        """
        Compute gradient using discrete adjoint method.

        The gradient formula:
            ∇_θ J = J_θ - R_θ^T λ

        where λ solves R_U^T λ = J_U^T

        For trapezoidal discretization, this becomes a backward sweep
        through the trajectory, solving local linear systems at each step.
        """
        aug_sol_jax = self._convert_aug_sol_to_jax(aug_sol)
        p_all = self._p_all_from_p_opt(p_opt)

        segments = aug_sol_jax['segments']
        events = aug_sol_jax['events']
        n_segments = len(segments)

        # Compute loss
        loss = self.compute_loss(aug_sol_jax, target_times, target_outputs, p_all)

        # Compute dL/dy at all grid points via VJP through prediction
        def loss_fn(aug_dict):
            y_pred = self._predict_outputs(aug_dict, target_times, p_all)
            return jnp.mean((y_pred - target_outputs) ** 2)

        _, vjp_fn = jax.vjp(loss_fn, aug_sol_jax)
        (grad_aug,) = vjp_fn(1.0)

        # Initialize gradient accumulator
        grad_p = jnp.zeros(self.n_opt_params)

        # Terminal adjoint (zero for tracking problems)
        lambda_curr = jnp.zeros(self.n_total)

        # Backward sweep through segments
        for i in range(n_segments - 1, -1, -1):
            seg = segments[i]
            t_seg = seg['t']
            y_seg = seg['y']

            # Get forcing terms (dL/dy) for this segment
            dL_dx = grad_aug['segments'][i]['x']
            dL_dz = grad_aug['segments'][i]['z']

            # Handle empty algebraic case
            if dL_dz.size == 0:
                dL_dy = dL_dx
            else:
                dL_dy = jnp.concatenate([dL_dx, dL_dz], axis=1)

            # Backward sweep through timesteps in this segment
            lambda_seg, grad_p_seg = self._backward_segment_trapezoidal(
                t_seg, y_seg, dL_dy, lambda_curr, p_all
            )

            grad_p = grad_p + grad_p_seg

            # Handle event transition if not first segment
            if i > 0:
                ev = events[i - 1]
                # Get direct loss gradient w.r.t. event time from VJP
                dL_dtau = grad_aug['events'][i - 1].get('tau', 0.0)

                lambda_pre, grad_p_ev = self._backward_event(
                    lambda_seg,
                    ev['x_pre'], ev['z_pre'],
                    ev['x_post'], ev['z_post'],
                    ev['tau'], p_all, ev['event_idx'],
                    dL_dtau=dL_dtau
                )

                grad_p = grad_p + grad_p_ev
                lambda_curr = jnp.concatenate([lambda_pre, jnp.zeros(self.n_alg)]) if self.n_alg > 0 else lambda_pre
            else:
                lambda_curr = lambda_seg

        return grad_p, float(loss)

    def _backward_segment_trapezoidal(self, t_seg, y_seg, dL_dy, lambda_terminal, p_all):
        """
        Backward adjoint sweep through a segment using trapezoidal rule.

        For trapezoidal discretization:
            x_{k+1} = x_k + (h/2) * (f_k + f_{k+1})

        Residual: r_k = x_{k+1} - x_k - (h/2) * (f_k + f_{k+1}) = 0

        Adjoint equation propagates backward:
            λ_k = λ_{k+1} + (h/2) * (J_k^T + J_{k+1}^T) @ λ_{k+1} + dL/dx_k
        """
        N = len(t_seg)

        if N <= 1:
            return lambda_terminal, jnp.zeros(self.n_opt_params)

        # Filter duplicate time points (preprocessing in numpy)
        t_np = np.array(t_seg)
        y_np = np.array(y_seg)
        dL_np = np.array(dL_dy)

        dt = np.diff(t_np)
        keep = np.concatenate([[True], dt > 1e-12])

        t_seg = jnp.array(t_np[keep])
        y_seg = jnp.array(y_np[keep])

        # Accumulate forcing for dropped points
        dL_accum = []
        curr = dL_np[0].copy()
        for j in range(1, len(keep)):
            if keep[j]:
                dL_accum.append(curr)
                curr = dL_np[j].copy()
            else:
                curr += dL_np[j]
        dL_accum.append(curr)
        dL_dy = jnp.array(np.stack(dL_accum))

        N = len(t_seg)
        if N <= 1:
            return lambda_terminal, jnp.zeros(self.n_opt_params)

        n_y = y_seg.shape[1]
        p_opt = jnp.array([p_all[i] for i in self.optimize_indices])

        # Use lax.scan for efficient backward sweep
        # Reverse the arrays for backward scan
        t_rev = t_seg[::-1]
        y_rev = y_seg[::-1]
        dL_rev = dL_dy[::-1]

        # Initial state: lambda at terminal time
        lambda_init = lambda_terminal[:n_y] + dL_rev[0][:self.n_states]
        grad_p_init = jnp.zeros(self.n_opt_params)

        def scan_step(carry, inputs):
            lambda_kp1, grad_p_acc = carry
            t_kp1, t_k, y_kp1, y_k, dL_k = inputs

            h = t_kp1 - t_k
            x_k = y_k[:self.n_states]
            x_kp1 = y_kp1[:self.n_states]
            z = jnp.zeros(self.n_alg) if self.n_alg > 0 else jnp.array([])

            # Compute Jacobians df/dx
            def f_of_x_k(x):
                return self._eval_f(t_k, x, z, p_all)
            def f_of_x_kp1(x):
                return self._eval_f(t_kp1, x, z, p_all)

            J_k = jax.jacfwd(f_of_x_k)(x_k)
            J_kp1 = jax.jacfwd(f_of_x_kp1)(x_kp1)

            # Adjoint propagation
            lambda_k = lambda_kp1 + (h / 2) * (J_k.T @ lambda_kp1 + J_kp1.T @ lambda_kp1) + dL_k[:self.n_states]

            # Parameter gradient
            def f_of_p(p):
                p_full = self._p_all_from_p_opt(p)
                f_k_p = self._eval_f(t_k, x_k, z, p_full)
                f_kp1_p = self._eval_f(t_kp1, x_kp1, z, p_full)
                return -(h / 2) * (f_k_p + f_kp1_p)

            dr_dp = jax.jacfwd(f_of_p)(p_opt)
            grad_p_new = grad_p_acc - dr_dp.T @ lambda_kp1

            return (lambda_k, grad_p_new), lambda_k

        # Prepare scan inputs (pairs of consecutive points, reversed)
        # t_kp1 is t_rev[:-1], t_k is t_rev[1:]
        scan_inputs = (t_rev[:-1], t_rev[1:], y_rev[:-1], y_rev[1:], dL_rev[1:])

        (lambda_final, grad_p), _ = jax.lax.scan(
            scan_step,
            (lambda_init, grad_p_init),
            scan_inputs
        )

        # Pad lambda if needed
        if self.n_alg > 0:
            lambda_final = jnp.concatenate([lambda_final, jnp.zeros(self.n_alg)])

        return lambda_final, grad_p

    def _backward_event(self, lambda_post, x_pre, z_pre, x_post, z_post, tau, p_all, event_idx, dL_dtau=0.0):
        """
        Propagate adjoint backward through an event.

        Event constraints:
            1. Guard: ψ(τ, x^-, z^-) = 0
            2. Reset: x^+ = J(x^-, z^-, τ, p)

        Adjoint equations:
            λ^- = (∂J/∂x^-)^T λ^+ + γ (∂ψ/∂x^-)

        where γ (timing sensitivity) comes from:
            γ = (H^+ - H^-) / (dψ/dt)
            H = λ^T f (Hamiltonian)

        Args:
            dL_dtau: Direct gradient of loss w.r.t. event time (from VJP through prediction)
        """
        # Jacobian of jump map w.r.t. x_pre
        def jump_x(x):
            return self._eval_jump(x, z_pre, tau, p_all, event_idx)

        J_jump_x = jax.jacfwd(jump_x)(x_pre)

        # Gradients of guard constraint
        def zc_x(x):
            return self._eval_zc(tau, x, z_pre, p_all)[event_idx]

        def zc_t(t):
            return self._eval_zc(t, x_pre, z_pre, p_all)[event_idx]

        grad_psi_x = jax.grad(zc_x)(x_pre)
        grad_psi_t = jax.grad(zc_t)(tau)

        # Dynamics before and after
        f_pre = self._eval_f(tau, x_pre, z_pre, p_all)
        f_post = self._eval_f(tau, x_post, z_post, p_all)

        # Only state part of lambda_post matters for event
        lambda_post_x = lambda_post[:self.n_states]

        # Pull adjoint through jump
        lambda_tilde = J_jump_x.T @ lambda_post_x

        # Compute gamma (timing sensitivity)
        # This includes both the adjoint contribution and the direct loss contribution
        H_post = jnp.dot(lambda_post_x, f_post)
        H_pre = jnp.dot(lambda_tilde, f_pre)

        dψ_dt = grad_psi_t + jnp.dot(grad_psi_x, f_pre)
        dψ_dt_safe = jnp.where(jnp.abs(dψ_dt) < 1e-8, 1e-8 * jnp.sign(dψ_dt + 1e-12), dψ_dt)

        # gamma from adjoint dynamics
        gamma_adj = (H_post - H_pre) / dψ_dt_safe

        # gamma from direct loss dependence on tau
        gamma_loss = dL_dtau / dψ_dt_safe

        gamma = gamma_adj + gamma_loss
        gamma = jnp.clip(gamma, -1e6, 1e6)

        # Adjoint before event
        lambda_pre = lambda_tilde - gamma * grad_psi_x

        # Parameter gradient from event
        def jump_p(p):
            p_full = self._p_all_from_p_opt(p)
            return self._eval_jump(x_pre, z_pre, tau, p_full, event_idx)

        def zc_p(p):
            p_full = self._p_all_from_p_opt(p)
            return self._eval_zc(tau, x_pre, z_pre, p_full)[event_idx]

        p_opt = jnp.array([p_all[i] for i in self.optimize_indices])
        J_jump_p = jax.jacfwd(jump_p)(p_opt)
        grad_psi_p = jax.grad(zc_p)(p_opt)

        grad_p_event = J_jump_p.T @ lambda_post_x + gamma * grad_psi_p

        return lambda_pre, grad_p_event

    # =========================================================================
    # Optimization Interface
    # =========================================================================

    def forward_solve(self, t_span, p_opt, ncp=200):
        """Run forward simulation with given parameters."""
        # Update solver parameters
        p_all = np.array(self.p_all)
        for i, idx in enumerate(self.optimize_indices):
            p_all[idx] = float(p_opt[i])

        for i in range(self.n_params_total):
            self.solver.p[i] = float(p_all[i])

        # Reset initial conditions
        self.solver.x0 = np.array([s['start'] for s in self.dae_data['states']])
        self.solver.z0 = np.array([a.get('start', 0.0) for a in self.dae_data.get('alg_vars', [])])

        return self.solver.solve_augmented(t_span=t_span, ncp=ncp)

    def optimization_step(self, t_span, target_times, target_outputs, p_opt, ncp=200):
        """
        Single optimization step: forward solve + gradient computation.

        Returns:
            p_opt: Current parameters (unchanged)
            loss: Loss value
            grad: Parameter gradient
        """
        p_opt_jax = jnp.array(p_opt)
        target_times_jax = jnp.array(target_times)
        target_outputs_jax = jnp.array(target_outputs)

        # Forward solve
        aug_sol = self.forward_solve(t_span, p_opt, ncp)

        # Gradient computation
        grad, loss = self.compute_gradient(aug_sol, target_times_jax, target_outputs_jax, p_opt_jax)

        return p_opt, loss, np.array(grad)

    def optimize(
        self,
        t_span: Tuple[float, float],
        target_times: np.ndarray,
        target_outputs: np.ndarray,
        max_iterations: int = 100,
        step_size: float = 0.01,
        tol: float = 1e-6,
        ncp: int = 200,
        print_every: int = 10,
        algorithm: str = 'adam'
    ) -> Dict:
        """
        Run optimization loop.

        Args:
            t_span: (t_start, t_end)
            target_times: Target measurement times
            target_outputs: Target values at those times
            max_iterations: Max iterations
            step_size: Learning rate
            tol: Gradient norm tolerance
            ncp: Collocation points for solver
            print_every: Print interval
            algorithm: 'sgd' or 'adam'

        Returns:
            Dictionary with results
        """
        p_opt = np.array(self.p_opt)
        history = {'loss': [], 'gradient_norm': [], 'params': []}

        # Adam state
        if algorithm.lower() == 'adam':
            m = np.zeros_like(p_opt)
            v = np.zeros_like(p_opt)
            beta1, beta2, eps = 0.9, 0.999, 1e-8

        if self.verbose:
            print(f"\nStarting optimization")
            print(f"  Algorithm: {algorithm}")
            print(f"  Max iterations: {max_iterations}")
            print(f"  Step size: {step_size}")
            print(f"  Parameters: {self.optimize_params}")
            print(f"  Initial values: {p_opt}")
            print()

        start_time = time.time()
        iter_times = []

        for it in range(max_iterations):
            iter_start = time.time()
            _, loss, grad = self.optimization_step(t_span, target_times, target_outputs, p_opt, ncp)
            iter_time = time.time() - iter_start
            iter_times.append(iter_time)
            grad_norm = np.linalg.norm(grad)

            history['loss'].append(loss)
            history['gradient_norm'].append(grad_norm)
            history['params'].append(p_opt.copy())

            if it % print_every == 0 or it == max_iterations - 1:
                avg_time = np.mean(iter_times[-min(5, len(iter_times)):])
                print(f"  Iter {it:4d}: Loss = {loss:.6e}, |grad| = {grad_norm:.6e}, p = {p_opt}, time = {iter_time:.2f}s")

            if grad_norm < tol:
                print(f"\nConverged at iteration {it}")
                break

            # Update parameters
            if algorithm.lower() == 'adam':
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1**(it + 1))
                v_hat = v / (1 - beta2**(it + 1))
                p_opt = p_opt - step_size * m_hat / (np.sqrt(v_hat) + eps)
            else:
                p_opt = p_opt - step_size * grad

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"\nOptimization complete in {elapsed:.2f}s")
            print(f"  Final loss: {history['loss'][-1]:.6e}")
            print(f"  Final params: {p_opt}")

        return {
            'params': p_opt,
            'history': history,
            'elapsed_time': elapsed,
            'converged': grad_norm < tol
        }
