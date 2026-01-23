"""
Event-Aware DAE Optimizer using Discrete Adjoint Method.

Extends the parallel optimized adjoint solver to handle DAEs with events (discontinuities).
Uses a hybrid approach:
- Python loop for iterating through variable-topology segments
- JIT-compiled inner operations for scan-based adjoints and event block solves

Key Components:
1. AugmentedSolution handling (variable number of segments/events)
2. Differentiable interpolation for loss computation at target times
3. Event block adjoint solver (handles state jumps and timing sensitivities)
4. Segment adjoint solver (reuses existing trapezoidal scan implementation)
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax import lax
from jax.scipy.linalg import lu_factor, lu_solve
from typing import Tuple, Dict, List, NamedTuple, Optional, Callable
import numpy as np
import time

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

from .dae_optimizer_parallel_optimized import DAEOptimizerParallelOptimized
from .dae_solver import DAESolver, TrajectorySegment, EventInfo, AugmentedSolution


# =============================================================================
# Module 1: JAX-Compatible Data Structures
# =============================================================================

class SegmentAdjointResult(NamedTuple):
    """Result from running adjoint on a single segment."""
    lambda_start: jnp.ndarray    # Adjoint at start of segment (N_x,)
    grad_params: jnp.ndarray     # Parameter gradients from this segment


class EventAdjointResult(NamedTuple):
    """Result from solving event adjoint block."""
    lambda_pre: jnp.ndarray      # Adjoint before event (lambda^-)
    gamma: float                 # Event timing sensitivity
    grad_params: jnp.ndarray     # Parameter gradients from event


# =============================================================================
# Module 2: Differentiable Interpolation Layer
# =============================================================================

def linear_interpolate_segment(
    t_query: jnp.ndarray,
    t_seg: jnp.ndarray,
    x_seg: jnp.ndarray
) -> jnp.ndarray:
    """
    Linear interpolation on a segment using state info only.
    
    Args:
        t_query: Query times, shape (M,)
        t_seg: Segment time points, shape (N,)
        x_seg: Segment states, shape (N, n_states)
        
    Returns:
        Interpolated states at query times, shape (M, n_states)
    """
    
    def interp_single_time(t_q):
        # Find interval index (clamp to valid range)
        idx = jnp.searchsorted(t_seg, t_q, side='right') - 1
        idx = jnp.clip(idx, 0, len(t_seg) - 2)
        
        # Get interval endpoints
        t0 = t_seg[idx]
        t1 = t_seg[idx + 1]
        x0 = x_seg[idx]
        x1 = x_seg[idx + 1]
        
        # Robust h calculation (avoid division by zero for duplicate points)
        h = t1 - t0
        h_safe = jnp.where(h < 1e-12, 1e-12, h)
        
        s = (t_q - t0) / h_safe  # Normalized parameter [0, 1]
        s = jnp.clip(s, 0.0, 1.0) # Ensure bounded for linear interp
        
        # Linear Interpolation
        x_interp = x0 * (1.0 - s) + x1 * s
        
        # If h < epsilon, just return x0
        return jnp.where(h < 1e-12, x0, x_interp)
    
    return vmap(interp_single_time)(t_query)


def predict_from_augmented_solution(
    aug_sol_jax: Dict,
    target_times: jnp.ndarray,
    blend_sharpness: float = 50.0  # Reduced sharpness for better gradients
) -> jnp.ndarray:
    """
    Predict states at target times from augmented solution using soft blending.
    
    This implements differentiable interpolation that correctly propagates
    gradients through event times. Uses normalized weighting to handle
    overlapping segment masks.
    
    Args:
        aug_sol_jax: Dictionary containing JAX arrays for segments and events
        target_times: Target times for prediction, shape (M,)
        blend_sharpness: Sharpness of sigmoid blending (higher = sharper transitions)
        
    Returns:
        Predicted states at target times, shape (M, n_states)
    """
    segments = aug_sol_jax['segments']
    events = aug_sol_jax['events']
    n_segments = len(segments)
    
    # Check if empty solution
    if n_segments == 0:
        return jnp.zeros((len(target_times), 1))
    
    n_states = segments[0]['x'].shape[1]
    M = len(target_times)
    
    # Initialize output and weight accumulator
    y_pred = jnp.zeros((M, n_states))
    total_weight = jnp.zeros((M, 1))

    # Compute contribution from each segment
    for i, seg in enumerate(segments):
        t_start = seg['t'][0]
        t_end = seg['t'][-1]
        
        # FIX: Clamp query times to segment bounds before interpolation
        # This prevents cubic Hermite polynomials from exploding to Inf for distant targets
        # The mask will zero out these values anyway, but clamping avoids 0 * Inf = NaN
        t_clamped = jnp.clip(target_times, t_start, t_end)
        
        # Interpolate this segment onto clamped target times
        x_interp = linear_interpolate_segment(
            t_clamped, seg['t'], seg['x']
        )
        
        # Soft mask using sigmoid for differentiability
        mask_start = jax.nn.sigmoid(blend_sharpness * (target_times - t_start))
        mask_end = jax.nn.sigmoid(blend_sharpness * (t_end - target_times))
        mask = mask_start * mask_end  # (M,)
        
        # For interior segments, refine mask based on event boundaries
        if i > 0:
            tau_prev = events[i-1]['tau']
            mask = mask * jax.nn.sigmoid(blend_sharpness * (target_times - tau_prev))
        if i < n_segments - 1:
            tau_next = events[i]['tau']
            mask = mask * jax.nn.sigmoid(blend_sharpness * (tau_next - target_times))
        
        mask = mask[:, None]  # Broadcast for state dimension
        
        # Accumulate weighted contribution
        y_pred = y_pred + mask * x_interp
        total_weight = total_weight + mask
    
    # Normalize to avoid scaling artifacts in overlapping regions
    y_pred = y_pred / (total_weight + 1e-8)
    
    return y_pred


# =============================================================================
# Module 3: Event Block Adjoint Solver (Factory Pattern for JIT)
# =============================================================================

def create_event_adjoint_solver(eval_f_fn, eval_zc_fn, eval_jump_fn):
    """
    Factory function that creates a JIT-compiled event adjoint solver.
    
    This pattern allows us to JIT-compile the solver while capturing the
    evaluation functions in a closure (avoiding the issue of passing
    functions as arguments to JIT-compiled code).
    
    Args:
        eval_f_fn: Function to evaluate dynamics f(t, x, z, p)
        eval_zc_fn: Function to evaluate zero-crossing g(t, x, z, p)
        eval_jump_fn: Function to compute jump x_post = J(x_pre, z_pre, tau, p, event_idx)
        
    Returns:
        JIT-compiled function that solves the event adjoint block
    """
    
    def solve_event_adjoint_block(
        lambda_plus: jnp.ndarray,
        x_pre: jnp.ndarray,
        z_pre: jnp.ndarray,
        x_post: jnp.ndarray,
        z_post: jnp.ndarray,
        tau: float,
        p: jnp.ndarray,
        event_idx: int
    ) -> Tuple[jnp.ndarray, float, jnp.ndarray]:
        """
        Solve the event adjoint block to propagate adjoint through discontinuity.
        
        Given:
            lambda_plus: adjoint from the segment AFTER the event
            Event transition: x_post = J(x_pre, p) where J is the jump map
            Event condition: g(x, t, p) = 0 defines the event surface
            
        Solve for:
            lambda_minus: adjoint BEFORE the event (to pass to previous segment)
            gamma: sensitivity of loss to event time tau
            grad_p: parameter gradient contribution from this event
            
        The key equations (from Bryson & Ho / optimal control theory):
            lambda_minus = (dJ/dx)^T @ lambda_plus - gamma * (dg/dx)
            gamma = [(dJ/dx)^T @ lambda_plus]·f_pre / (dg/dt + (dg/dx)·f_pre)
        """
        n_states = len(x_pre)
        n_params = len(p)
        
        # =====================================================================
        # 1. Compute Jacobian of Jump Map: dJ/dx_pre
        # =====================================================================
        
        def jump_wrapper(x):
            """Wrapper for jump map differentiation."""
            return eval_jump_fn(x, z_pre, tau, p, event_idx)
        
        # dJ/dx_pre: shape (n_states, n_states)
        jac_J_x = jax.jacfwd(jump_wrapper)(x_pre)
        
        # =====================================================================
        # 2. Compute Jacobian of Event Condition: dg/dx, dg/dt
        # =====================================================================
        
        def zc_wrapper_x(x):
            """Zero-crossing as function of x for differentiation."""
            zc_all = eval_zc_fn(tau, x, z_pre, p)
            return zc_all[event_idx]
        
        def zc_wrapper_t(t):
            """Zero-crossing as function of t for differentiation."""
            zc_all = eval_zc_fn(t, x_pre, z_pre, p)
            return zc_all[event_idx]
        
        # dg/dx: shape (n_states,)
        grad_g_x = jax.grad(zc_wrapper_x)(x_pre)
        
        # dg/dt: scalar
        grad_g_t = jax.grad(zc_wrapper_t)(tau)
        
        # =====================================================================
        # 3. Compute dynamics at pre-event and post-event states
        # =====================================================================
        
        f_pre = eval_f_fn(tau, x_pre, z_pre, p)   # dx/dt just before event
        f_post = eval_f_fn(tau, x_post, z_post, p)  # dx/dt just after event
        
        # =====================================================================
        # 4. Solve for gamma (event timing sensitivity)
        # =====================================================================
        
        # Intermediate: pull lambda back through jump
        lambda_tilde = jac_J_x.T @ lambda_plus
        
        # Hamiltonian before and after (H = lambda · f)
        H_post = jnp.dot(lambda_plus, f_post)
        H_pre_tilde = jnp.dot(lambda_tilde, f_pre)
        
        # Total time derivative of g along trajectory
        dg_dt_total = grad_g_t + jnp.dot(grad_g_x, f_pre)
        
        # gamma = (H_post - H_pre) / (dg/dt_total)
        # Robust regularization: prevent NaN when trajectory is tangent to event surface
        # For Zeno-like behavior (v->0), dg_dt_total -> 0, causing gamma -> infinity.
        # We increase regularization and clamp gamma to prevent gradient explosion.
        reg_eps = 1e-6
        denom = jnp.where(jnp.abs(dg_dt_total) < reg_eps, 
                          reg_eps * jnp.sign(dg_dt_total + 1e-12), 
                          dg_dt_total)
        denom = jnp.where(denom == 0.0, reg_eps, denom)
        
        gamma = (H_post - H_pre_tilde) / denom
        
        # Clamp gamma to avoid exploding gradients in Zeno regime
        gamma = jnp.clip(gamma, -1e6, 1e6)
        
        # =====================================================================
        # 5. Compute lambda_minus (adjoint before event)
        # =====================================================================
        
        lambda_minus = lambda_tilde - gamma * grad_g_x
        
        # =====================================================================
        # 6. Compute parameter gradients from event
        # =====================================================================
        
        def jump_wrapper_p(params):
            """Jump map as function of parameters."""
            return eval_jump_fn(x_pre, z_pre, tau, params, event_idx)
        
        def zc_wrapper_p(params):
            """Zero-crossing as function of parameters."""
            zc_all = eval_zc_fn(tau, x_pre, z_pre, params)
            return zc_all[event_idx]
        
        # dJ/dp: shape (n_states, n_params)
        jac_J_p = jax.jacfwd(jump_wrapper_p)(p)
        
        # dg/dp: shape (n_params,)
        grad_g_p = jax.grad(zc_wrapper_p)(p)
        
        # Parameter gradient contribution:
        # grad_p = lambda_plus^T @ (dJ/dp) + gamma * (dg/dp)
        grad_p_event = jac_J_p.T @ lambda_plus + gamma * grad_g_p
        
        return lambda_minus, gamma, grad_p_event
    
    # Return the function (can be JIT-compiled by caller if desired)
    return solve_event_adjoint_block


# =============================================================================
# Module 4: Main Event-Aware Optimizer Class
# =============================================================================

class DAEOptimizerEventAware(DAEOptimizerParallelOptimized):
    """
    Event-aware DAE optimizer using discrete adjoint method.
    
    Extends DAEOptimizerParallelOptimized to handle DAEs with events.
    Uses hybrid approach: Python loop for segment iteration, JIT for inner ops.
    
    Key differences from base class:
    1. Uses AugmentedSolution instead of regular time arrays
    2. Backward pass iterates through segments in reverse, solving event blocks
    3. Loss computed via differentiable interpolation to handle variable grids
    """
    
    def __init__(self, *args, verbose=True, **kwargs):
        # Initialize base class
        super().__init__(*args, verbose=verbose, **kwargs)
        
        if self.verbose:
            print("  Event-aware extension enabled")
            print("  Using hybrid JIT strategy (Python loop + JIT inner ops)")
        
        # Compile JAX versions of event-related functions
        self._compile_event_functions()
    
    def _compile_event_functions(self):
        """
        Compile JAX-differentiable versions of event-related functions.
        
        These are needed for computing Jacobians through the event transitions.
        """
        n_states = self.jac.n_states
        n_alg = self.jac.n_alg
        
        # =====================================================================
        # JAX version of zero-crossing evaluation
        # =====================================================================
        
        def eval_zc_jax(t, x, z, p):
            """
            Evaluate zero-crossing functions using JAX.
            
            Returns array of zero-crossing values (one per event type).
            """
            # Build namespace for equation evaluation
            ns = self.jac._create_jax_eval_namespace_with_params(
                t, x, z, p,
                optimize_indices=self.jac.optimize_indices,
                p_all_default=self.jac.p_all_default
            )
            
            # Evaluate each zero-crossing expression
            zc_list = []
            for expr in self.solver.zc_funcs:
                val = eval(expr, ns)
                zc_list.append(val)
            
            return jnp.array(zc_list) if zc_list else jnp.array([0.0])
        
        self._eval_zc_jax = eval_zc_jax
        
        # =====================================================================
        # JAX version of jump map (reinitialization)
        # =====================================================================
        
        def eval_jump_jax(x_pre, z_pre, tau, p, event_idx):
            """
            Evaluate jump map: x_post = J(x_pre, z_pre, tau, p).
            
            For explicit reinit like "v = -e*prev(v)", this is straightforward.
            Implements the reinitialization logic in JAX for differentiability.
            """
            x_post = x_pre.copy()
            
            if not hasattr(self.solver, 'event_reinit_exprs') or len(self.solver.event_reinit_exprs) == 0:
                return x_post
            
            # Get reinit info for this event
            reinit_expr = self.solver.event_reinit_exprs[event_idx]
            var_type, var_idx = self.solver.event_reinit_vars[event_idx]
            var_name = self.solver.event_reinit_var_names[event_idx]
            
            # Build namespace with prev() values
            ns = self.jac._create_jax_eval_namespace_with_params(
                tau, x_pre, z_pre, p,
                optimize_indices=self.jac.optimize_indices, 
                p_all_default=self.jac.p_all_default
            )
            
            # Add prev_ prefixed values
            for i, name in enumerate(self.solver.state_names):
                ns[f'prev_{name}'] = x_pre[i]
            for i, name in enumerate(self.solver.alg_names):
                ns[f'prev_{name}'] = z_pre[i]
            
            # Replace prev(var) with prev_var in expression
            import re
            reinit_modified = re.sub(r'prev\(\s*(\w+)\s*\)', r'prev_\1', reinit_expr)
            
            # FIX: Robust coefficient extraction for linear reinit equations
            # Solve linear equation: coeff * var + offset = 0
            # Use two-point evaluation to find coefficient and offset
            
            ns[var_name] = 0.0
            val_at_0 = eval(reinit_modified, ns)  # offset
            
            ns[var_name] = 1.0
            val_at_1 = eval(reinit_modified, ns)
            
            # Coefficient is the slope
            coeff = val_at_1 - val_at_0
            
            # Solution: var = -offset / coeff
            # Guard against division by zero (unlikely for valid equations)
            safe_coeff = jnp.where(jnp.abs(coeff) < 1e-12, 1.0, coeff)
            new_val = -val_at_0 / safe_coeff
            
            if var_type == 'state':
                x_post = x_post.at[var_idx].set(new_val)
            
            return x_post
        
        self._eval_jump_jax = eval_jump_jax
        
        # =====================================================================
        # JAX version of f evaluation with parameters
        # =====================================================================
        
        def eval_f_jax_with_params(t, x, z, p):
            """Evaluate f(t, x, z, p) using JAX."""
            return self.jac.eval_f_jax(t, x, z, p)
        
        self._eval_f_jax = eval_f_jax_with_params
        
        # =====================================================================
        # Create Event Adjoint Solver using Factory Pattern
        # =====================================================================
        
        # Create the event adjoint solver with eval functions captured in closure
        self._solve_event_adjoint = create_event_adjoint_solver(
            eval_f_fn=eval_f_jax_with_params,
            eval_zc_fn=eval_zc_jax,
            eval_jump_fn=eval_jump_jax
        )
    
    def _convert_augmented_to_jax(self, aug_sol: AugmentedSolution) -> Dict:
        """
        Convert AugmentedSolution (NumPy) to JAX-compatible dictionary.
        
        Args:
            aug_sol: AugmentedSolution from solver
            
        Returns:
            Dictionary with JAX arrays for segments and events
        """
        segments = []
        for seg in aug_sol.segments:
            segments.append({
                't': jnp.array(seg.t),
                'x': jnp.array(seg.x),
                'z': jnp.array(seg.z),
                'xp': jnp.array(seg.xp)
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
    
    def compute_segment_adjoint_trapezoidal(
        self,
        segment: Dict,
        lambda_terminal: jnp.ndarray,
        forcing_terms: jnp.ndarray,
        p: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Run trapezoidal adjoint on a single segment.
        
        Reuses the efficient scan-based implementation from base class.
        
        Args:
            segment: Dictionary with 't', 'x', 'z', 'xp' arrays
            lambda_terminal: Terminal adjoint condition, shape (n_total,)
            forcing_terms: dL/dy at each grid point, shape (N, n_total)
            p: Parameter vector
            
        Returns:
            lambda_start: Adjoint at start of segment
            grad_params: Parameter gradients from this segment
        """
        t = segment['t']
        x = segment['x']
        z = segment['z']
        
        # Filter out duplicate/near-duplicate time points to avoid division by zero
        # This can happen at event boundaries where solver outputs near-identical times
        dt = jnp.diff(t)
        valid_mask = dt > 1e-12
        valid_indices = jnp.concatenate([jnp.array([0]), jnp.where(valid_mask, jnp.arange(1, len(t)), -1)])
        valid_indices = valid_indices[valid_indices >= 0]
        
        # If using NumPy for indexing (static), convert to Python list
        # For JAX compatibility, we need to handle this carefully
        # Use a simple approach: filter in Python before JIT
        t_np = np.array(t)
        x_np = np.array(x)
        z_np = np.array(z)
        
        dt_np = np.diff(t_np)
        keep_mask = np.concatenate([[True], dt_np > 1e-12])
        t = jnp.array(t_np[keep_mask])
        x = jnp.array(x_np[keep_mask])
        z = jnp.array(z_np[keep_mask])
        
        N = len(t)
        
        if N <= 1:
            # Degenerate segment (single point)
            return lambda_terminal, jnp.zeros(len(p))
        
        # Also filter forcing_terms to match, BUT accumulate dropped forcing into kept points
        # If we drop a point i (because t[i] == t[i-1]), the loss gradient at i should contribute to i-1
        # otherwise we lose gradient information.
        
        if len(forcing_terms) == len(keep_mask):
            # We need to accumulate forcing
            # Convert to numpy for iteration (easier) - this is outside JIT scan usually, or handled as static
            u_np = np.array(forcing_terms)
            
            # Create a new array for kept forcing
            u_filtered = []
            
            current_accum = u_np[0].copy()
            
            for i in range(1, len(keep_mask)):
                if keep_mask[i]:
                    # Push the accumulator
                    u_filtered.append(current_accum)
                    # Start new accumulator
                    current_accum = u_np[i].copy()
                else:
                    # Accumulate into current
                    current_accum += u_np[i]
            
            # Append the last accumulator
            u_filtered.append(current_accum)
            
            forcing_terms_filtered = jnp.array(np.stack(u_filtered))
        else:
             forcing_terms_filtered = forcing_terms
        
        # Sanity check
        if len(forcing_terms_filtered) != len(t):
             # Mismatch in accumulation logic vs filtering logic?
             # keep_mask has a True for the first element.
             # Loop above:
             # i=1..N-1.
             # If keep_mask[i] is True, we append accumulated(i-1) and start accumulation at i.
             # If keep_mask[i] is False, we add to accumulated(i-1).
             # At end, we append accumulated(last).
             # This means u_filtered will have (count of True in keep_mask[1:]) + 1.
             # keep_mask[0] is Always True.
             # Total count = 1 + count(True in 1..). = count(True).
             # This matches len(t) logic above.
             pass
        
        # Combine x and z into full state
        y = jnp.concatenate([x, z], axis=1)  # (N, n_total)
        
        # Prepare arrays for adjoint computation
        t_k = t[:-1]
        t_kp1 = t[1:]
        y_k = y[:-1]
        y_kp1 = y[1:]
        
        # Prepare arrays (already created above)
        
        # Use Standard Dense Adjoint Solver (Reference Implementation)
        # This is more robust than the matrix-free path which was returning zero gradients
        from .adjoint_solver import solve_adjoint_system_jit
        
        # Compute Jacobian blocks
        J_prev_list, J_curr_list = self.jac.compute_jacobian_blocks_jit(t, y.T, p)
        
        # Prepare dL/dy for adjoint solve
        # forcing_terms is (N, n_total). We need (N-1, n_total) for steps
        dL_dy_adjoint = forcing_terms_filtered[1:]
        
        # Solve adjoint system
        # solve_adjoint_system_jit expects J lists and dL_dy
        lambda_adjoint = solve_adjoint_system_jit(J_prev_list, J_curr_list, dL_dy_adjoint)
        
        # lambda_adjoint is (N-1, n_total) corresponding to t[1:] points
        
        # We need to incorporate terminal condition into the backpropagation?
        # solve_adjoint_system_jit solves J^T lambda = b.
        # It assumes terminal lambda is handling boundary?
        # Actually standard solver assumes independent steps or simple chain.
        # But we have `lambda_terminal` from next segment.
        # The standard solver typically does backward substitution:
        # J_curr[k]^T lambda[k] + J_prev[k+1]^T lambda[k+1] = b[k]
        # For the last step k=N-1:
        # J_curr[N-1]^T lambda[N-1] + J_prev[N]^T lambda[N] = b[N-1]
        # lambda[N] is `lambda_terminal`.
        # solve_adjoint_system_jit assumes lambda[N] (outside array) is zero?
        # Let's check solve_adjoint_system_jit signature/behavior (it is imported).
        # Usually it returns lambda[k] for k=0..N-1.
        
        # To handle lambda_terminal, we must modify b[N-1].
        # b[N-1] -= J_prev[N]^T @ lambda_terminal
        # J_prev_list has indices k=0 (step 0..1).
        # We need J_prev associated with the "next" step which is outside this segment?
        # No, J_prev[k+1] term.
        # If we are at the end of segment, lambda[N] is the boundary value.
        # We need to compute J_prev[N] term?
        # But J_prev is computed from segment data.
        # Wait, adjoint eqn: J_{k,k}^T * lam_k + J_{k+1,k}^T * lam_{k+1} = -dL/dx_k.
        # Last point N. lam_N is lambda_terminal.
        # Eq for N-1 involves lam_N-1 and lam_N.
        # J^T_{N-1, N-1} lam_{N-1} + J^T_{N, N-1} lam_N = ...
        # J^T_{N, N-1} corresponds to dependence of x_N on x_{N-1}. This is J_prev (of step leading to N).
        # J_prev_list[N-1] is roughly J_prev for step N-1->N.
        # Wait, reference solver assumes `solves J_curr[k].T @ x[k] + J_prev[k+1].T @ x[k+1] = b`.
        # If we treat the last step explicitly:
        # We need to subtract J_prev_term from RHS.
        
        # However, simpler approach: Append a dummy step or just modify RHS.
        # Let's inspect J_prev_list structure. it corresponds to blocks.
        # Use manual backward substitution or modify RHS for last element.
        
        # Modification:
        # The solver solves for lambda[0]...lambda[N-1].
        # The last equation (index N-2? no N-1 steps). index k=N-2.
        # (N points t[0]..t[N-1]. N-1 intervals. Adjoint has N-1 unknowns lambda[0]..lambda[N-2] corresponding to intervals?
        # No, lambda is usually defined at time points t[1]...t[N]?
        # Adjoint variables usually match state count.
        # Standard: lambda has shape (N-1, n). Corresponds to t[0]..t[N-2]? Or t[1]..t[N-1]?
        # `solve_adjoint_system_jit` returns (N, n)?
        # Doc in `dae_jacobian.py` says `lambda_adjoint` shape `(N, n_total)` where `N = n_time - 1`.
        # So it returns lambda for `t[0]...t[N-2]`?
        # This implies it solves strictly inside.
        
        # To support lambda_terminal, we need to correct the RHS of the last equation.
        # But for now, let's just use the result and assume terminal effect is small/handled,
        # OR better: Add `lambda_terminal` to the initial guess? No, it's linear linear solve.
        
        # We will assume solve_adjoint_system_jit handles the chain.
        # But we need to feed it `lambda_terminal`.
        # Standard `optimization_step` sets `dL_dy_adjoint` and calls solver.
        # It assumes `lambda[end]` is 0.
        
        # Correct logic for chaining:
        # We need `J_prev_last` term. `J_prev_list` has length N-1.
        # But we need `J_prev` for the hypothetical boundary step?
        # Actually, `lambda_terminal` is at `t[N-1]`.
        # The solver computes `lambda` up to `t[N-2]`.
        # The last equation connects `lambda[N-2]` and `lambda[N-1]`.
        # `J_curr[N-2].T @ lam[N-2] + J_prev[N-1].T @ lam[N-1] = b[N-2]`.
        # `solve_adjoint_system_jit` solves this.
        # It expects `lambda[N-1]` to be PART of the system?
        # Usually it solves for ALL lambda.
        # But `Standard` optimizer computes `dL_dy[1:, :]` -> `(N-1, n)`.
        # So it produces `N-1` lambdas.
        # So `lambda_terminal` corresponds to the value AT `t[N-1]`.
        # This is `y0` in the recursive matmul.
        # `solve_adjoint_system_jit` likely assumes `y0=0`.
        
        # We cannot easily modify `solve_adjoint_system_jit` as it is imported.
        # BUT we can modify the RHS `dL_dy_adjoint[-1]`.
        # `b[-1]` (last eqn) should include `- J_prev[last].T @ lambda_terminal`.
        # `J_prev_list` has `N-1` matrices. `J_prev_list[-1]` is `J_prev[N-1]`.
        # So:
        # b[-1] -= J_prev_list[-1].T @ lambda_terminal
        
        # Let's do that.
        
        # Update RHS with terminal adjoint info
        rhs_mod = dL_dy_adjoint.at[-1].add(-J_prev_list[-1].T @ lambda_terminal)
        
        lambda_internal = solve_adjoint_system_jit(J_prev_list, J_curr_list, rhs_mod)
        
        # lambda_internal is lambda at t[0]...t[N-2]
        # We need full lambda history: [lambda[0]...lambda[N-2], lambda_terminal]
        # To match lambda_all structure (usually [t[1]...t[N-1], t[N]]?)
        # Base matrixfree returned `lambda_all` of size `N-1`?
        # Matrixfree returns `lambda` matching `forcing_terms_filtered[1:]` length.
        
        lambda_all = jnp.vstack([lambda_internal, lambda_terminal])
        
        # Now lambda_all has N rows (matching t).

        
        lambda_start = lambda_all[0]
        
        # Compute parameter gradients using internal lambdas (corresponding to N-1 steps)
        # lambda_all has N elements, J_param has N-1 elements
        grad_params = self._compute_parameter_gradient_matrixfree(
            t_k, t_kp1, y_k, y_kp1, p, lambda_all[:-1]
        )
        
        return lambda_start, grad_params
    
    def _compute_parameter_gradient_matrixfree(
        self,
        t_k: jnp.ndarray,
        t_kp1: jnp.ndarray,
        y_k: jnp.ndarray,
        y_kp1: jnp.ndarray,
        p_opt_vals_jax: jnp.ndarray,
        lambda_adjoint: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute parameter gradient using efficient einsum contraction.

        Avoids forming large (N*n_total, n_params) reshaped matrix.
        Instead computes: grad[p] = -sum_k lambda[k,n]^T @ J_param[k,n,p]
        """
        # Compute parameter Jacobian: shape (N, n_total, n_params)
        J_param = self.jac._jac_p_vmapped(t_k, t_kp1, y_k, y_kp1, p_opt_vals_jax)

        # Efficient contraction: sum_k (lambda[k]^T @ J_param[k])
        # lambda_adjoint: (N, n_total)
        # J_param: (N, n_total, n_params)
        # Result: (n_params,)
        grad_p_opt = -jnp.einsum('kn,knp->p', lambda_adjoint, J_param)

        return grad_p_opt

    def compute_loss_and_forcing(
        self,
        aug_sol_jax: Dict,
        target_times: jnp.ndarray,
        target_outputs: jnp.ndarray,
        p: jnp.ndarray
    ) -> Tuple[float, List[jnp.ndarray]]:
        """
        Compute loss and forcing terms (dL/dy) for each segment using VJP.
        
        Uses global VJP (Vector-Jacobian Product) to propagate gradients from
        the loss back through differentiable interpolation to segment grid points.
        This correctly handles the gradient chain without non-differentiable ops.
        
        Args:
            aug_sol_jax: JAX-converted augmented solution
            target_times: Target times for loss computation
            target_outputs: Target output values at target_times
            p: Parameter vector
            
        Returns:
            loss: Scalar loss value
            segment_forcing: List of forcing arrays, one per segment
        """
        n_states = self.jac.n_states
        n_alg = self.jac.n_alg
        n_total = self.jac.n_total
        
        # Determine simulation horizon
        segments = aug_sol_jax['segments']
        if len(segments) > 0:
            t_final = segments[-1]['t'][-1]
            
            # Mask target data to keep only points within horizon
            # Add small tolerance to include end point
            mask = target_times <= (t_final + 1e-10)
            
            # Filter targets to match simulation horizon
            target_times_used = target_times[mask]
            
            # Handle target_outputs shape: (N, n_out)
            if target_outputs.shape[0] == len(target_times):
                target_outputs_used = target_outputs[mask]
            elif target_outputs.ndim == 2 and target_outputs.shape[1] == len(target_times):
                # Transposed case (n_out, N)
                target_outputs_used = target_outputs[:, mask]
            else:
                # Fallback, let JAX broadcast or error
                target_outputs_used = target_outputs
        else:
            t_final = 0.0
            target_times_used = jnp.array([])
            target_outputs_used = jnp.array([])
            
        def prediction_loss_fn(aug_sol_primal):
            """
            Pure function: AugmentedSolution dict -> Loss scalar.
            
            This is the function we differentiate through to get forcing terms.
            """
            # Predict states at target times via differentiable interpolation
            y_pred = predict_from_augmented_solution(aug_sol_primal, target_times_used)
            
            # Compute loss (MSE)
            diff = y_pred - target_outputs_used
            if self.loss_type == 'sum':
                loss_val = jnp.sum(diff ** 2)
            else:
                loss_val = jnp.mean(diff ** 2)
            return loss_val
        
        # Compute loss and VJP setup
        loss, vjp_fn = jax.vjp(prediction_loss_fn, aug_sol_jax)
        
        # Compute gradients w.r.t. everything in aug_sol_jax
        # We pass 1.0 as the incoming gradient for the loss scalar
        (aug_sol_grads,) = vjp_fn(1.0)
        
        # Extract forcing terms (dL/dx, dL/dz) for each segment
        # aug_sol_grads has the exact same structure as aug_sol_jax
        # The 'x' field in each segment contains dL/dx at each grid point
        segment_forcing = []
        
        for i, seg_grad in enumerate(aug_sol_grads['segments']):
            # Get gradient w.r.t. state 'x' and algebraic 'z'
            dL_dx = seg_grad['x']  # shape: (N, n_states)
            dL_dz = seg_grad['z']  # shape: (N, n_alg)
            
            # Concatenate to match (N, n_total) shape expected by adjoint solver
            forcing_k = jnp.concatenate([dL_dx, dL_dz], axis=1)
            segment_forcing.append(forcing_k)
        
        return loss, segment_forcing
    
    def backward_pass_events(
        self,
        aug_sol: AugmentedSolution,
        target_times: jnp.ndarray,
        target_outputs: jnp.ndarray,
        p: jnp.ndarray
    ) -> Tuple[jnp.ndarray, float]:
        """
        Hybrid backward pass for event-aware optimization.
        
        Uses Python loop to iterate through segments/events in reverse,
        with JIT-compiled inner operations.
        
        Args:
            aug_sol: AugmentedSolution from forward solve
            target_times: Target times for loss
            target_outputs: Target outputs
            p: Parameter vector
            
        Returns:
            grad_params: Total parameter gradients
            loss: Loss value
        """
        # Convert to JAX-compatible format
        aug_sol_jax = self._convert_augmented_to_jax(aug_sol)
        segments = aug_sol_jax['segments']
        events = aug_sol_jax['events']
        n_segments = len(segments)
        n_params = len(p)
        
        # Compute loss and forcing terms
        loss, segment_forcing = self.compute_loss_and_forcing(
            aug_sol_jax, target_times, target_outputs, p
        )
        
        # Initialize gradient accumulator
        total_grad = jnp.zeros(n_params)
        
        # Initialize terminal adjoint (zero at final time for tracking problems)
        last_seg = segments[-1]
        n_total = self.jac.n_total
        lambda_curr = jnp.zeros(n_total)
        
        # =====================================================================
        # REVERSE LOOP: Iterate through segments backward
        # =====================================================================
        
        for i in range(n_segments - 1, -1, -1):
            segment = segments[i]
            forcing = segment_forcing[i]
            
            # -----------------------------------------------------------------
            # Step 1: Run continuous adjoint on this segment
            # -----------------------------------------------------------------
            
            lambda_start, seg_grad = self.compute_segment_adjoint_trapezoidal(
                segment, lambda_curr, forcing, p
            )
            
            # Accumulate gradients
            total_grad = total_grad + seg_grad
            
            # -----------------------------------------------------------------
            # Step 2: Handle event transition (if not first segment)
            # -----------------------------------------------------------------
            
            if i > 0:
                event_idx = i - 1
                event = events[event_idx]
                
                # The lambda_start from segment i becomes lambda_plus for event
                # We only care about state variables, not algebraics
                lambda_plus = lambda_start[:self.jac.n_states]
                
                # Solve event adjoint block using factory-created solver
                lambda_minus, gamma, event_grad = self._solve_event_adjoint(
                    lambda_plus=lambda_plus,
                    x_pre=event['x_pre'],
                    z_pre=event['z_pre'],
                    x_post=event['x_post'],
                    z_post=event['z_post'],
                    tau=event['tau'],
                    p=p,
                    event_idx=event['event_idx']
                )
                
                # Update current adjoint for next iteration
                # Pad lambda_minus to full n_total (algebraics get zero)
                lambda_curr = jnp.concatenate([
                    lambda_minus,
                    jnp.zeros(self.jac.n_alg)
                ])
                
                # Accumulate event gradients
                total_grad = total_grad + event_grad
            
        return total_grad, float(loss)

    def _adam_update_step(self, p, grad, m, v, t, beta1, beta2, epsilon, step_size):
        """
        Adam optimizer update step.
        """
        t_new = t + 1
        
        # Update biased first moment estimate
        m_new = beta1 * m + (1 - beta1) * grad
        
        # Update biased second moment estimate
        v_new = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = m_new / (1 - beta1 ** t_new)
        
        # Compute bias-corrected second moment estimate
        v_hat = v_new / (1 - beta2 ** t_new)
        
        # Update parameters
        p_new = p - step_size * m_hat / (jnp.sqrt(v_hat) + epsilon)
        
        return p_new, m_new, v_new, t_new
    
    def optimization_step_events(
        self,
        t_span: Tuple[float, float],
        target_times: np.ndarray,
        target_outputs: np.ndarray,
        p_opt: np.ndarray,
        step_size: float = 0.01,
        ncp: int = 200
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Perform one optimization step for DAE with events.
        
        Args:
            t_span: Time span for simulation
            target_times: Target times for loss computation
            target_outputs: Target output values
            p_opt: Current optimized parameter values
            step_size: Gradient descent step size
            ncp: Number of collocation points for solver density
            
        Returns:
            p_opt_new: Updated parameters
            loss: Loss value
            grad: Parameter gradients
        """
        # Step 1: Update solver parameters
        p_all = np.array(self.p_all)
        for i, opt_idx in enumerate(self.optimize_indices):
            p_all[opt_idx] = float(p_opt[i])
        
        for i in range(self.n_params_total):
            self.solver.p[i] = float(p_all[i])
        
        # Reset initial conditions
        self.solver.x0 = np.array([s['start'] for s in self.dae_data['states']])
        self.solver.z0 = np.array([
            a.get('start', 0.0) for a in self.dae_data.get('alg_vars', [])
        ])
        
        # Step 2: Forward solve with events
        aug_sol = self.solver.solve_augmented(
            t_span=t_span,
            rtol=self.rtol,
            atol=self.atol,
            ncp=ncp
        )
        
        # Step 3: Backward pass
        p_opt_jax = jnp.array([p_all[i] for i in self.optimize_indices])
        target_times_jax = jnp.array(target_times)
        target_outputs_jax = jnp.array(target_outputs)
        
        grad, loss = self.backward_pass_events(
            aug_sol, target_times_jax, target_outputs_jax, p_opt_jax
        )
        
        return p_opt, loss, np.array(grad)
    
    def optimize_events(
        self,
        t_span: Tuple[float, float],
        target_times: np.ndarray,
        target_outputs: np.ndarray,
        max_iterations: int = 100,
        step_size: float = 0.01,
        tol: float = 1e-6,
        ncp: int = 200,
        print_every: int = 10,
        algorithm_config: Dict = None
    ) -> Dict:
        """
        Run full optimization loop for DAE with events.
        
        Args:
            t_span: Time span (t_start, t_end)
            target_times: Times at which outputs are measured
            target_outputs: Target output values, shape (M, n_outputs)
            max_iterations: Maximum optimization iterations
            step_size: Learning rate
            tol: Convergence tolerance on gradient norm
            ncp: Collocation points for solver
            print_every: Print interval
            
        Returns:
            Dictionary with optimization results
        """
        # Initialize
        p_opt = np.array(self.p_current)
        history = {
            'loss': [],
            'gradient_norm': [],
            'params': [],
            'n_events': []
        }
        
        # Configure Algorithm
        if algorithm_config is None:
            algorithm_type = 'SGD'
            algorithm_params = {'step_size': step_size}
        else:
            algorithm_type = algorithm_config.get('type', 'SGD').upper()
            algorithm_params = algorithm_config.get('params', {})
            
        algo_step_size = algorithm_params.get('step_size', step_size)
        
        # Initialize Adam state if needed
        if algorithm_type == 'ADAM':
            adam_m = jnp.zeros_like(p_opt)
            adam_v = jnp.zeros_like(p_opt)
            adam_t = 0
        
        print(f"\nStarting event-aware optimization")
        print(f"  Algorithm: {algorithm_type}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Step size: {step_size}")
        print(f"  Target times: {len(target_times)} points")
        
        start_time = time.time()
        
        for iteration in range(max_iterations):
            # Optimization step
            p_opt_new, loss, grad = self.optimization_step_events(
                t_span, target_times, target_outputs, p_opt, step_size, ncp
            )
            
            grad_norm = np.linalg.norm(grad)
            
            # Update history
            history['loss'].append(loss)
            history['gradient_norm'].append(grad_norm)
            history['params'].append(p_opt.copy())
            
            # Print progress
            if iteration % print_every == 0 or iteration == max_iterations - 1:
                print(f"  Iter {iteration:4d}: Loss = {loss:.6e}, |grad| = {grad_norm:.6e}")
            
            # Check convergence
            if grad_norm < tol:
                print(f"\nConverged at iteration {iteration} (|grad| = {grad_norm:.6e} < {tol})")
                break
            
            # Update parameters
            if algorithm_type == 'ADAM':
                beta1 = algorithm_params.get('beta1', 0.9)
                beta2 = algorithm_params.get('beta2', 0.999)
                epsilon = algorithm_params.get('epsilon', 1e-8)
                p_opt, adam_m, adam_v, adam_t = self._adam_update_step(
                    p_opt, grad, adam_m, adam_v, adam_t, beta1, beta2, epsilon, algo_step_size
                )
            else:
                p_opt = p_opt - algo_step_size * grad
        
        elapsed = time.time() - start_time
        print(f"\nOptimization complete in {elapsed:.2f}s")
        print(f"  Final loss: {history['loss'][-1]:.6e}")
        print(f"  Final params: {p_opt}")
        
        return {
            'params': p_opt,
            'history': history,
            'elapsed_time': elapsed,
            'converged': grad_norm < tol
        }

    def plot_optimization_history(self, history: Dict):
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available, skipping plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        ax = axes[0, 0]
        ax.semilogy(history['loss'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Function')
        ax.grid(True, alpha=0.3)

        # Gradient norm
        ax = axes[0, 1]
        ax.semilogy(history['gradient_norm'], 'r-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm')
        ax.grid(True, alpha=0.3)

        # Parameters (only optimized ones)
        ax = axes[1, 0]
        params_array = np.array(history['params'])
        for i in range(params_array.shape[1]):
            ax.plot(params_array[:, i], label=f'{self.optimize_params[i]}', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Value')
        ax.set_title('Parameter Evolution (Optimized)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Loss vs gradient norm
        ax = axes[1, 1]
        ax.loglog(history['gradient_norm'], history['loss'], 'go-', linewidth=2, markersize=4)
        ax.set_xlabel('Gradient Norm')
        ax.set_ylabel('Loss')
        ax.set_title('Loss vs Gradient Norm')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
