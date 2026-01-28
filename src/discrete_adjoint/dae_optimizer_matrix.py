"""
Matrix-Based Discrete Adjoint Optimizer for DAEs with Events.

This optimizer uses the full matrix form for computing gradients via the discrete
adjoint method, similar to the approach in debug/verify_residual.py.

The entire hybrid trajectory is treated as a solution to a large nonlinear 
algebraic system R(W, p) = 0, where:
- W contains all state/algebraic variables at grid points and event times
- p contains the parameters to optimize

Gradient computation:
    [dR/dW]^T λ = -dL/dW
    dJ/dp = dL/dp + λ^T dR/dp
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Tuple, Dict, List, Optional, Callable
import numpy as np
import time
from functools import partial
import re
import copy

jax.config.update("jax_enable_x64", True)

from .dae_solver import DAESolver, AugmentedSolution

# =============================================================================
# 1. Equation Compiler
# =============================================================================

def compile_equations_to_jax(eqn_strings, state_names, alg_names, param_names, extra_args=None):
    """Parses string equations into a single JAX-jittable function."""
    if not eqn_strings:
        return lambda t, x, z, p: jnp.array([])

    subs = []
    if extra_args:
        for name, repl in extra_args.items(): subs.append((name, repl))
    for i, name in enumerate(state_names): subs.append((name, f"x[{i}]"))
    for i, name in enumerate(alg_names):   subs.append((name, f"z[{i}]"))
    for i, name in enumerate(param_names): subs.append((name, f"p[{i}]"))
    subs.append(('time', 't'))
    subs.append(('t', 't'))
    subs.sort(key=lambda item: len(item[0]), reverse=True)

    processed_eqns = []
    math_funcs = {'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'exp', 'log', 'sqrt', 'abs', 'pow', 'min', 'max'}
    
    for eq in eqn_strings:
        processed_eq = eq
        for name, repl in subs:
            if name in math_funcs: continue
            pattern = r'(?<!\.)\b' + re.escape(name) + r'\b'
            processed_eq = re.sub(pattern, repl, processed_eq)
        processed_eqns.append(processed_eq)

    return_expr = "[" + ", ".join(processed_eqns) + "]"
    
    def create_closure():
        exec_ns = {'jnp': jnp}
        for f in math_funcs:
            if hasattr(jnp, f): exec_ns[f] = getattr(jnp, f)
        exec_ns['power'] = jnp.power 
        source = f"def compiled_fn(t, x, z, p): return jnp.array({return_expr})"
        exec(source, exec_ns)
        return exec_ns['compiled_fn']

    return create_closure()

# =============================================================================
# 2. DAE Function Creation
# =============================================================================

def create_jax_functions(dae_data):
    # This matches logic in DAEOptimizerPadded, simplified here for separate verification
    state_names = [s['name'] for s in dae_data['states']]
    alg_names = [a['name'] for a in dae_data.get('alg_vars', [])]
    param_names = [p['name'] for p in dae_data['parameters']]
    
    # Compile f
    f_eqs = dae_data['f']
    # Extract RHS
    f_exprs = []
    for eq in f_eqs:
        # format: der(x) = ...
        if '=' in eq:
            f_exprs.append(eq.split('=', 1)[1].strip())
        else:
            raise ValueError(f"Invalid f equation: {eq}")
            
    # Compile g
    g_eqs = dae_data.get('g', [])
    g_exprs = []
    for eq in g_eqs:
        if '=' in eq:
            lhs, rhs = eq.split('=', 1)
            g_exprs.append(f"({lhs.strip()}) - ({rhs.strip()})")
        else:
            g_exprs.append(eq)
            
    # Compile Guard (Event Condition)
    # Assumes single event for bouncing ball (h=0) or we check all?
    # User example implies specific events te1, te2 corresponding to h=0
    # We will grab the first event condition "h < 0" -> "h" = 0
    # Actually DAESolver converts "h<0" to "h - 0 < 0". Surface is h=0.
    when_clauses = dae_data.get('when', [])
    guard_exprs = []
    reinit_exprs = [] # Functions that return the NEW value for the reset variable
    reinit_vars = []  # Index of variable being reset
    
    for wc in when_clauses:
        cond = wc['condition']
        if '<' in cond:
            lhs, rhs = cond.split('<', 1)
            expr = f"({lhs}) - ({rhs})"
        elif '>' in cond:
            lhs, rhs = cond.split('>', 1)
            expr = f"({lhs}) - ({rhs})"
        else: # Handle = 0 case if present
             lhs, rhs = cond.split('=', 1)
             expr = f"({lhs}) - ({rhs})"
            
        guard_exprs.append(expr)
        
        # Reinit
        # Spec format: "v + e*prev(v) = 0" -> Res = (v_post + e*v_pre) - 0
        reinit_str = wc['reinit']
        if '=' in reinit_str:
            lhs, rhs = reinit_str.split('=', 1)
            raw_expr = f"({lhs}) - ({rhs})"
            
            # Identify reinitialized variable using regex
            # Find which state name appears as a full word in LHS
            lhs_clean = lhs
            for i, name in enumerate(state_names):
                 if re.search(r'\b' + re.escape(name) + r'\b', lhs_clean):
                     reinit_vars.append(('state', i))
                     break
        else:
            raw_expr = reinit_str # Assume = 0 if no equals?

        # Substitution Logic:
        # 1. prev(name) -> variable in x_pre/z_pre
        # 2. name -> variable in x_post/z_post
        
        # We need a custom compiler that takes TWO state vectors (pre and post).
        reinit_exprs.append(raw_expr)
    
    print(f"DEBUG: Guard Exprs: {guard_exprs}")
    print(f"DEBUG: Reinit Exprs: {reinit_exprs}")

    h_exprs = dae_data.get('h', [])
    # If no h, default to state (x)
    use_default_h = (len(h_exprs) == 0)
    
    # ... (existing setup code for variables)
    
    # Common compiler
    def compile_to_jax(expr_list, is_reinit=False):
        if not expr_list:
            if is_reinit:
                return lambda t, x_post, z_post, x_pre, z_pre, p: jnp.array([])
            else:
                return lambda t, x, z, p: jnp.array([])
            
        subs = []
        # Normal vars -> Post (or current for f/g)
        for i, n in enumerate(state_names): 
            target = f"x_post[{i}]" if is_reinit else f"x[{i}]"
            subs.append((n, target))
        for i, n in enumerate(alg_names):
            target = f"z_post[{i}]" if is_reinit else f"z[{i}]"
            subs.append((n, target))
            
        # Param is same
        for i, n in enumerate(param_names): subs.append((n, f"p[{i}]"))
        
        # Pre vars (only for reinit)
        if is_reinit:
            # We use a placeholder first? No, we can just assume `prev(n)` pattern exists.
            # But we must handle `prev( n )` regex first.
            pass
            
        subs.append(('time', 't'))
        subs.sort(key=lambda x: len(x[0]), reverse=True)
        
        jax_exprs = []
        for e in expr_list:
            final_e = e
            
            # 1. Handle prev(name) -> x_pre[i]
            if is_reinit:
                def replace_prev(match):
                    var = match.group(1)
                    if var in state_names:
                        return f"x_pre[{state_names.index(var)}]"
                    if var in alg_names:
                        return f"z_pre[{alg_names.index(var)}]"
                    return f"prev_{var}" # Fallback
                
                final_e = re.sub(r'prev\s*\(\s*(\w+)\s*\)', replace_prev, final_e)
            
            # 2. Handle standard names -> x_post[i]
            for name, repl in subs:
                pattern = r'(?<!\.)\b' + re.escape(name) + r'\b'
                # Ensure we don't replace parts of already replaced "x_pre" or "x_post" (unlikely given naming)
                # But "x" is in "x_pre"? No, "x" is not a variable name usually. variable is "h", "v".
                final_e = re.sub(pattern, repl, final_e)
                
            jax_exprs.append(final_e)
            
        if is_reinit:
            code = f"def func(t, x_post, z_post, x_pre, z_pre, p): return jnp.array([{', '.join(jax_exprs)}])"
        else:
            code = f"def func(t, x, z, p): return jnp.array([{', '.join(jax_exprs)}])"
            
        local_scope = {'jnp': jnp}
        exec(code, local_scope)
        return local_scope['func']

    f_fn = compile_to_jax(f_exprs, is_reinit=False)
    g_fn = compile_to_jax(g_exprs, is_reinit=False)
    guard_fn = compile_to_jax(guard_exprs, is_reinit=False)
    reinit_res_fn = compile_to_jax(reinit_exprs, is_reinit=True) # Returns Residuals
    
    # Compile h_fn
    if use_default_h:
        # returns x
        h_fn = lambda t, x, z, p: x
    else:
        h_fn = compile_to_jax(h_exprs, is_reinit=False)

    return f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, (len(state_names), len(alg_names), len(param_names))

# --- New Packing/Jacobian Logic ---

def pack_solution(sol: AugmentedSolution, dae_data: dict) -> Tuple[jnp.ndarray, list, list]:
    """
    Pack solution into flat vector W.
    Structure: [Seg0_Nodes, Event0_Time, Seg1_Nodes, Event1_Time, ...]
    Returns: (W_flat, structure, grid_taus)
    """
    structure = []
    grid_taus = []
    W_parts = []
    
    num_seg = len(sol.segments)
    num_events = len(sol.events)
    
    for i in range(num_seg):
        seg = sol.segments[i]
        n_pts = len(seg.t)
        n_x = seg.x.shape[1]
        n_z = seg.z.shape[1] if seg.z is not None and seg.z.ndim > 1 else 0
        n_w = n_x + n_z
        
        # Calculate normalized time grid
        t_start = seg.t[0]
        t_end = seg.t[-1]
        denom = t_end - t_start
        if denom < 1e-12:
            denom = 1.0
        tau = (seg.t - t_start) / denom
        grid_taus.append(jnp.array(tau))
        
        # Flatten segment data
        seg_start_idx = len(W_parts)
        if n_z > 0:
            seg_data = np.column_stack([seg.x, seg.z])
        else:
            seg_data = seg.x
        W_parts.append(seg_data.flatten())
        
        seg_len = len(seg_data.flatten())
        structure.append(('segment', n_pts, seg_len))
        
        # Add event time immediately after segment (if exists)
        if i < num_events:
            ev = sol.events[i]
            W_parts.append(np.array([ev.t_event]))
            structure.append(('event_time', 1))
    
    W_flat = jnp.concatenate([jnp.array(p) for p in W_parts])
    return W_flat, structure, grid_taus

def unpack_solution_structure(W_flat, structure, dims, grid_taus):
    """
    Unpack W into trajectory segments.
    Returns: (segs_t, segs_x, segs_z, event_times)
    """
    n_x, n_z, _ = dims
    n_w = n_x + n_z
    
    segs_t = []
    segs_x = []
    segs_z = []
    event_times = []
    
    idx = 0
    seg_counter = 0
    t_start = 0.0
    
    for item in structure:
        kind = item[0]
        count = item[1] if len(item) > 1 else 1
        length = item[2] if len(item) > 2 else 1
        if kind == 'segment':
            n_pts = count
            seg_data = W_flat[idx:idx+length].reshape((n_pts, n_w))
            idx += length
            
            # Get event time (end of segment)
            # Find next event or use t_final
            t_end = 2.0  # Default
            for j in range(len(structure)):
                if structure[j][0] == 'event_time':
                    # Check if this event comes after current segment
                    # Simple heuristic: count segments before this event
                    segments_before = sum(1 for k in range(j) if structure[k][0] == 'segment')
                    if segments_before == seg_counter:
                        # Find event time index in W
                        ev_idx = sum(s[2] if len(s) > 2 else s[1] for s in structure[:j])
                        t_end = W_flat[ev_idx]
                        break
            
            # Reconstruct time grid
            tau = grid_taus[seg_counter]
            t_seg = t_start + tau * (t_end - t_start)
            
            xs = seg_data[:, :n_x]
            zs = seg_data[:, n_x:] if n_z > 0 else jnp.zeros((n_pts, 0))
            
            segs_t.append(t_seg)
            segs_x.append(xs)
            segs_z.append(zs)
            
            t_start = t_end
            seg_counter += 1
            
        elif kind == 'event_time':
            te = W_flat[idx]
            event_times.append(te)
            idx += 1
    
    return segs_t, segs_x, segs_z, jnp.array(event_times)

# =============================================================================
# 4. Residual Function
# =============================================================================

def unpack_and_compute_residual(W_flat, p_opt, dae_data, structure, funcs, param_mapping, grid_taus):
    """
    Compute global residual vector R(W, p) for the discretized DAE system.
    """
    f_fn, g_fn, _, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_total_p = dims
    n_w = n_x + n_z
    
    # Reconstruct full parameter vector
    p_all_default, opt_indices = param_mapping
    p_all = p_all_default
    if len(opt_indices) > 0:
        p_all = p_all.at[jnp.array(opt_indices)].set(p_opt)
    
    residuals = []
    
    # Track event indices
    event_indices = []
    idx_scan = 0
    for i, item in enumerate(structure):
        kind = item[0]
        count = item[1] if len(item) > 1 else 1
        length = item[2] if len(item) > 2 else 1
        if kind == 'event_time':
            event_indices.append(idx_scan)
        idx_scan += length
    
    # Process structure
    event_counter = 0
    seg_counter = 0
    idx_scan = 0
    t_start_seg = 0.0
    t_final = 2.0
    
    last_x, last_z = None, None
    
    for i, item in enumerate(structure):
        kind = item[0]
        count = item[1] if len(item) > 1 else 1
        length = item[2] if len(item) > 2 else 1
        if kind == 'segment':
            n_pts = count
            segment_data = W_flat[idx_scan:idx_scan+length].reshape((n_pts, n_w))
            idx_scan += length
            
            xs = segment_data[:, :n_x]
            zs = segment_data[:, n_x:] if n_z > 0 else jnp.zeros((n_pts, 0))
            
            # Determine segment end time
            t0 = t_start_seg
            if event_counter < len(event_indices):
                te_idx = event_indices[event_counter]
                te = W_flat[te_idx]
            else:
                te = t_final
            
            # Time grid
            current_tau = grid_taus[seg_counter]
            ts = t0 + current_tau * (te - t0)
            
            # Initial condition (first segment only)
            if i == 0:
                x0_fixed = jnp.array([s['start'] for s in dae_data['states']])
                residuals.extend(xs[0] - x0_fixed)
            
            # Flow residuals (trapezoidal rule)
            for k in range(n_pts - 1):
                t_k = ts[k]
                t_kp1 = ts[k+1]
                x_k = xs[k]
                x_kp1 = xs[k+1]
                z_k = zs[k] if n_z > 0 else jnp.array([])
                z_kp1 = zs[k+1] if n_z > 0 else jnp.array([])
                
                h = t_kp1 - t_k
                
                f_k = f_fn(t_k, x_k, z_k, p_all)
                f_kp1 = f_fn(t_kp1, x_kp1, z_kp1, p_all)
                
                res = -x_kp1 + x_k + (h/2.0)*(f_k + f_kp1)
                residuals.extend(res)
                
                if n_z > 0:
                    residuals.extend(g_fn(t_k, x_k, z_k, p_all))
            
            # Algebraic at last point
            if n_z > 0:
                residuals.extend(g_fn(ts[-1], xs[-1], zs[-1], p_all))
            
            last_x = xs[-1]
            last_z = zs[-1] if n_z > 0 else jnp.array([])
            
            t_start_seg = te
            seg_counter += 1
            
        elif kind == 'event_time':
            idx_scan += 1
            te = W_flat[idx_scan - 1]
            
            # Get post-event state from next segment
            if i + 1 < len(structure):
                next_item = structure[i+1]
                next_kind = next_item[0]
                next_count = next_item[1]
                next_len = next_item[2] if len(next_item) > 2 else next_item[1]
                
                # Extract next segment data
                next_seg_data = W_flat[idx_scan:idx_scan+next_len].reshape((next_count, n_w))
                x_post = next_seg_data[0, :n_x]
                z_post = next_seg_data[0, n_x:] if n_z > 0 else jnp.array([])
            
                x_pre = last_x
                z_pre = last_z
            
                # Guard constraint
                val_guard = guard_fn(te, x_pre, z_pre, p_all)
                residuals.extend(val_guard)
            
                # Reset map
                val_reset = reinit_res_fn(te, x_post, z_post, x_pre, z_pre, p_all)
                residuals.extend(val_reset)
            
                # Continuity for non-reinitialized variables
                for k in range(n_x):
                    is_reinit = any(True for (t, idx) in reinit_vars if t == 'state' and idx == k)
                    if not is_reinit:
                        residuals.extend(x_post[k:k+1] - x_pre[k:k+1])
            
                if n_z > 0:
                    residuals.extend(g_fn(te, x_post, z_post, p_all))
            
                event_counter += 1
    
    return jnp.concatenate([jnp.array(r).flatten() for r in residuals])

# =============================================================================
# 5. Trajectory Prediction (for Loss)
# =============================================================================

@jit
def predict_trajectory_sigmoid(segs_t, segs_x, segs_z, event_times, target_times, blend_sharpness=100.0):
    """
    Predict trajectory at target_times using sigmoid blending at events.
    """
    def predict_single(t_target):
        y_accum = jnp.zeros_like(segs_x[0][0])
        w_accum = 0.0
        
        for seg_idx, (ts, xs) in enumerate(zip(segs_t, segs_x)):
            # Segment bounds
            t0 = ts[0]
            t1 = ts[-1]
            
            # Compute sigmoid weight
            if seg_idx == 0:
                w_left = 1.0
            else:
                te_left = event_times[seg_idx - 1]
                w_left = jax.nn.sigmoid(blend_sharpness * (t_target - te_left))
            
            if seg_idx == len(segs_t) - 1:
                w_right = 1.0
            else:
                te_right = event_times[seg_idx]
                w_right = jax.nn.sigmoid(blend_sharpness * (te_right - t_target))
            
            mask = w_left * w_right
            
            # Interpolate within segment
            t_clip = jnp.clip(t_target, t0, t1)
            idx = jnp.searchsorted(ts, t_clip, side='right') - 1
            idx = jnp.clip(idx, 0, len(ts) - 2)
            
            t0_grid = ts[idx]
            t1_grid = ts[idx + 1]
            denom = t1_grid - t0_grid
            denom = jnp.where(jnp.abs(denom) < 1e-12, 1e-12, denom)
            s = (t_clip - t0_grid) / denom
            s = jnp.clip(s, 0.0, 1.0)
            
            val = xs[idx] * (1.0 - s) + xs[idx+1] * s
            
            y_accum += mask * val
            w_accum += mask
        
        return y_accum / (w_accum + 1e-8)
    
    return jax.vmap(predict_single)(target_times)

# =============================================================================
# 6. Matrix-Based Optimizer
# =============================================================================

class DAEOptimizerMatrix:
    """
    DAE optimizer using full matrix-based adjoint gradients.
    """
    
    def __init__(self, dae_data: dict, solver_config: dict, verbose: bool = True):
        self.dae_data = dae_data
        self.solver_config = solver_config
        self.verbose = verbose
        
        # Extract parameters
        self.param_names = [p['name'] for p in dae_data['parameters']]
        self.p_all_default = jnp.array([p['value'] for p in dae_data['parameters']])
        
        # Create solver
        self.solver = DAESolver(dae_data, verbose=False)
        
    def optimize(self, 
                 target_times: np.ndarray,
                 target_data: np.ndarray,
                 optimize_params: List[str],
                 learning_rate: float = 0.01,
                 max_iterations: int = 100,
                 tol: float = 1e-6,
                 ncp: int = 10):
        """
        Optimize parameters to match target data using adjoint gradients.
        
        Args:
            target_times: Time points for target data
            target_data: Target trajectory values
            optimize_params: List of parameter names to optimize
            learning_rate: Gradient descent step size
            max_iterations: Maximum iterations
            tol: Convergence tolerance
            ncp: Number of collocation points per segment
        """
        # Setup parameter mapping
        opt_indices = [self.param_names.index(name) for name in optimize_params]
        p_opt = self.p_all_default[jnp.array(opt_indices)].copy()
        param_mapping = (self.p_all_default, opt_indices)
        
        target_times = jnp.array(target_times)
        target_data = jnp.array(target_data)
        
        # Optimization history
        history = {
            'loss': [],
            'grad_norm': [],
            'params': [],
            'time_per_iter': []
        }
        
        if self.verbose:
            print(f"\nOptimizing parameters: {optimize_params}")
            print(f"Initial values: {p_opt}")
        
        # Compile JAX functions once
        funcs = create_jax_functions(self.dae_data)
        f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
        n_x, n_z, n_p = dims

        # Performance caching
        last_structure = None
        compute_grads_and_jacs = None

        for iter_idx in range(max_iterations):
            iter_start = time.time()
            
            # Update solver parameters
            p_current = self.p_all_default.at[jnp.array(opt_indices)].set(p_opt)
            self.solver.update_parameters(np.array(p_current))
            
            # Solve forward problem
            t_span = (self.solver_config['start_time'], self.solver_config['stop_time'])
            sol = self.solver.solve_augmented(t_span, ncp=ncp)
            
            # Pack solution
            W_flat, structure, grid_taus = pack_solution(sol, self.dae_data)
            structure_tuple = tuple(structure)

            # Check if we need to re-jit (structure changed)
            if structure_tuple != last_structure:
                if self.verbose and last_structure is not None:
                    print(f"DEBUG: Structure changed at iter {iter_idx}, recompiling...")

                # Define the functions to be jitted
                def R_global_internal(W, p):
                    return unpack_and_compute_residual(
                        W, p, self.dae_data, structure, funcs, param_mapping, grid_taus
                    )
                
                def loss_function_internal(W, p):
                    segs_t, segs_x, segs_z, ev_tau = unpack_solution_structure(W, structure, dims, grid_taus)
                    y_pred = predict_trajectory_sigmoid(segs_t, segs_x, segs_z, ev_tau, target_times, blend_sharpness=300.0)
                    
                    # Weighting: height is at index 0, velocity at index 1
                    # Focus more on height to avoid velocity timing noise
                    diff = y_pred - target_data
                    weights = jnp.array([1.0, 0.0])[:diff.shape[1]]
                    weighted_diff = diff * weights
                    return jnp.mean(weighted_diff**2)

                @jit
                def _compute_all(W, p):
                    L = loss_function_internal(W, p)
                    dL_dW = jax.grad(loss_function_internal, argnums=0)(W, p)
                    dL_dp = jax.grad(loss_function_internal, argnums=1)(W, p)
                    dR_dW = jax.jacfwd(R_global_internal, argnums=0)(W, p)
                    dR_dp = jax.jacfwd(R_global_internal, argnums=1)(W, p)
                    return L, dL_dW, dL_dp, dR_dW, dR_dp

                compute_grads_and_jacs = _compute_all
                last_structure = structure_tuple

            # Compute gradients and Jacobians (JIT cached if structure is same)
            L, dL_dW, dL_dp, dR_dW, dR_dp = compute_grads_and_jacs(W_flat, p_opt)
            
            # Solve adjoint system
            lambda_sol = jnp.linalg.solve(dR_dW.T, -dL_dW)
            
            # Total gradient
            dJ_dp = dL_dp + jnp.dot(lambda_sol, dR_dp)
            grad_norm = jnp.linalg.norm(dJ_dp)
            
            iter_time = time.time() - iter_start
            
            # Store history
            history['loss'].append(float(L))
            history['grad_norm'].append(float(grad_norm))
            history['params'].append(p_opt.copy())
            history['time_per_iter'].append(iter_time)
            
            if self.verbose:
                print(f"Iter {iter_idx:3d}: Loss={L:.6e}, GradNorm={grad_norm:.6e}, Time={iter_time:.3f}s")
                for i, name in enumerate(optimize_params):
                    print(f"  {name}={p_opt[i]:.6f}, dJ/d{name}={dJ_dp[i]:.6e}")
            
            # Check convergence
            if grad_norm < tol:
                if self.verbose:
                    print(f"\\nConverged after {iter_idx+1} iterations")
                break
            
            # Update parameters
            p_opt = p_opt - learning_rate * dJ_dp
        
        return {
            'params': p_opt,
            'param_names': optimize_params,
            'history': history,
            'success': grad_norm < tol,
            'iterations': iter_idx + 1
        }
