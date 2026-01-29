
import numpy as np
import yaml
import json
import sys
import os
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

# Add src to path
sys.path.append(os.getcwd())

from src.discrete_adjoint.dae_solver import DAESolver
import re
import matplotlib.pyplot as plt

def eval_f_python(f_funcs, t, x, z, p, ns_base):
    ns = ns_base.copy()
    ns['t'] = t
    ns['time'] = t
    
    # Update state/alg/param values
    # Assumes ns_base already has 'sin', 'cos', etc.
    # We need to map variable names to values
    # This is tricky without the variable names available here easily.
    # Better to use the DAESolver's eval methods or recreate the JAX functions.
    pass

def load_system(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    solver_cfg = config['dae_solver']
    dae_spec_file = solver_cfg['dae_specification_file']
    
    with open(dae_spec_file, 'r') as f:
        dae_data = json.load(f)
        
    return dae_data, solver_cfg

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

def pack_solution(sol, dae_data):
    """
    Packs the AugmentedSolution into a flat vector W.
    Structure:
    [ [Seg0_Nodes], [te1], [Seg1_Nodes], [te2], ... ]
    
    Nodes are flattened: [x0, z0, x1, z1, ...]
    
    Returns: W_flat, structure, grid_taus
    """
    
    # Primal variables list
    w_list = []
    
    # Helper to pack a node
    def pack_node(x, z):
        w_list.extend(x)
        w_list.extend(z)
        
    num_seg = len(sol.segments)
    num_events = len(sol.events)
    
    # We will assume the structure:
    # Seg 0 (all points) -> Event 0 Time -> Seg 1 (all points) -> ...
    # This matches user description: [w(t0)...w(prev)] [te] [w(after)...]
    
    # Note: AugmentedSolution segments are:
    # Seg i: t[0]...t[end]. x[0]...x[end].
    # Events are separate.
    # However, sol.segments[i].x[-1] IS w_prev(event i).
    # sol.segments[i+1].x[0] IS w_after(event i).
    # And sol.events[i].t_event IS te.
    
    # So we can just iterate segments and events in order.
    
    structure = []
    grid_taus = [] 
    
    for i in range(num_seg):
        seg = sol.segments[i]
        n_points = len(seg.t)
        
        # Calculate Tau Grid (Normalized)
        t_start = seg.t[0]
        t_end = seg.t[-1]
        denom = t_end - t_start
        if denom < 1e-12: denom = 1.0 # Single point or zero duration
        tau = (seg.t - t_start) / denom
        grid_taus.append(tau)
        
        seg_start_idx = len(w_list)
        
        for k in range(n_points):
            # Check if this point is "part of the optimization variables" or just derived?
            # User implies W contains ALL node values w(tk).
            w_list.extend(seg.x[k])
            w_list.extend(seg.z[k] if len(seg.z) > 0 else [])
            
        seg_len = len(w_list) - seg_start_idx
        structure.append(('segment', n_points, seg_len))
        
        # If there is an event following this segment
        if i < num_events:
            ev = sol.events[i]
            # Add Event Time
            w_list.append(ev.t_event)
            structure.append(('event_time', 1))
            
    return jnp.array(w_list), structure, grid_taus

def unpack_and_compute_residual(W_flat, p_opt, dae_data, structure, funcs, param_mapping, grid_taus, t_final=2.0):
    """
    Reconstructs trajectory from W and computes global Residual vector.
    Uses grid_taus (normalized) to reconstruct time grid.
    """
    # Updated unpacking to include h_fn (ignored here)
    f_fn, g_fn, _, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_total_p = dims
    n_w = n_x + n_z
    
    # Reconstruct Parameters
    p_all_default, opt_indices = param_mapping
    p_all = p_all_default
    if len(opt_indices) > 0:
        p_all = p_all.at[jnp.array(opt_indices)].set(p_opt)
    
    residuals = []
    
    # Identify event time indices in W first
    event_indices_in_W = []
    idx_scan = 0
    t_start_seg = 0.0
    # t_final used from arg
    
    for i, (kind, count, *_) in enumerate(structure):
        if kind == 'event_time':
            event_indices_in_W.append(idx_scan)
        length = count if kind == 'event_time' else _[0]
        idx_scan += length
            
    event_counter = 0
    seg_counter = 0
    idx_scan = 0
    
    # State tracking for events
    last_x, last_z = None, None
    
    for i, (kind, count, *_) in enumerate(structure):
        if kind == 'segment':
            n_pts = count
            length = _[0]
            # Extract Nodes
            segment_data = W_flat[idx_scan : idx_scan + length].reshape((n_pts, n_w))
            idx_scan += length
            
            xs = segment_data[:, :n_x]
            zs = segment_data[:, n_x:]
            
            # Get Segment Bounds
            t0 = t_start_seg
            if event_counter < len(event_indices_in_W):
                te_idx = event_indices_in_W[event_counter]
                te = W_flat[te_idx]
            else:
                te = t_final
                
            # Construct Time Grid (Normalized)
            current_tau = grid_taus[seg_counter]
            ts = t0 + current_tau * (te - t0)
            
            # Construct Time Grid (Normalized)
            current_tau = grid_taus[seg_counter]
            ts = t0 + current_tau * (te - t0)
            
            # --- Initial Condition Constraint (First Segment Only) ---
            if i == 0:
                # Get initial state from dae_data
                # dae_data structure: 'states' list with 'start'
                x0_fixed = jnp.array([s['start'] for s in dae_data['states']])
                # Residual: x[0] - x0_fixed = 0
                residuals.extend(xs[0] - x0_fixed)

            # --- Flow Residuals ---
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
                
                # Res = -x_next + x_curr + h/2...
                res = -x_kp1 + x_k + (h/2.0)*(f_k + f_kp1)
                residuals.extend(res)
                
                if n_z > 0:
                    residuals.extend(g_fn(t_k, x_k, z_k, p_all))
            
            # G at last point
            if n_z > 0:
                residuals.extend(g_fn(ts[-1], xs[-1], zs[-1] if n_z>0 else [], p_all))

            last_x = xs[-1]
            last_z = zs[-1] if n_z>0 else []
            
            t_start_seg = te
            seg_counter += 1
            
        elif kind == 'event_time':
            idx_scan += 1 
            te = W_flat[idx_scan - 1]
            
            # Guard and Reset (only if there is a next segment)
            if i + 1 < len(structure):
                next_kind, next_count, *next_extra = structure[i+1]
                next_len = next_extra[0]
                next_seg_data = W_flat[idx_scan : idx_scan + next_len].reshape((next_count, n_w))
                x_post = next_seg_data[0, :n_x]
                z_post = next_seg_data[0, n_x:]
                
                x_pre = last_x
                z_pre = last_z
                
                val_guard = guard_fn(te, x_pre, z_pre, p_all)
                residuals.extend(val_guard)
                
                val_reset = reinit_res_fn(te, x_post, z_post, x_pre, z_pre, p_all)
                residuals.extend(val_reset)
                
                # Enforce continuity for variables NOT in reinit_vars
                # reinit_vars is list of (type, index)
                # Check states
                for k in range(n_x):
                    is_reinit = any(True for (t, idx) in reinit_vars if t == 'state' and idx == k)
                    if not is_reinit:
                        residuals.extend(x_post[k:k+1] - x_pre[k:k+1])
                        
                # Ensure algebraic variables satisfy constraints at event time (consistency)
                if n_z > 0:
                    residuals.extend(g_fn(te, x_post, z_post, p_all))
                
                event_counter += 1
    
    return jnp.concatenate([jnp.array(r).flatten() for r in residuals])
    


# --- Prediction and Loss Logic ---

def unpack_solution_structure(W_flat, structure, n_dims, grid_taus, t_final=2.0):
    """
    Unpacks W into lists of (t, x, z) arrays for each segment and event times.
    """
    n_x, n_z, n_w = n_dims
    
    segments_t = []
    segments_x = []
    segments_z = []
    events_tau = []
    
    idx_scan = 0
    t_start_seg = 0.0
    
    # Identify event indices first to get values
    event_indices = [i for i, (kind, _, *_) in enumerate(structure) if kind == 'event_time']
    # But we need the values from W to build time grids
    
    # Extract all event times first
    temp_scan = 0
    extracted_events = []
    for kind, count, *extra in structure:
        length = extra[0] if kind == 'segment' else count
        if kind == 'event_time':
            extracted_events.append(W_flat[temp_scan])
        temp_scan += length
        
    event_counter = 0
    seg_counter = 0
    idx_scan = 0
    
    for kind, count, *extra in structure:
        if kind == 'segment':
            n_pts = count
            length = extra[0]
            
            chunk = W_flat[idx_scan : idx_scan + length]
            idx_scan += length
            
            nodes = chunk.reshape((n_pts, n_w))
            xs = nodes[:, :n_x]
            zs = nodes[:, n_x:]
            
            # Time Grid
            t0 = t_start_seg
            if event_counter < len(extracted_events):
                te = extracted_events[event_counter]
            else:
                te = t_final
                
            current_tau = grid_taus[seg_counter]
            ts = t0 + current_tau * (te - t0)
            
            segments_t.append(ts)
            segments_x.append(xs)
            segments_z.append(zs)
            
            t_start_seg = te
            seg_counter += 1
            
        elif kind == 'event_time':
            # Store event time
            val = W_flat[idx_scan]
            events_tau.append(val)
            idx_scan += 1
            event_counter += 1
            
    return segments_t, segments_x, segments_z, jnp.array(events_tau)

def predict_trajectory_sigmoid(segments_t, segments_x, segments_z, events_tau, target_times, blend_sharpness=300.0):
    """
    Sigmoid-blended prediction.
    """
    n_outputs = segments_x[0].shape[1] # Assume output = state for now (all states)
    
    # Vectorized prediction over target_times
    def predict_single(t_q):
        y_accum = jnp.zeros(n_outputs)
        w_accum = 0.0
        
        for i in range(len(segments_t)):
            ts = segments_t[i]
            xs = segments_x[i]
            # zs ignored for now, assuming output is full state x
            
            t_start = ts[0]
            t_end = ts[-1]
            
            # Effective window
            lower = t_start if i == 0 else events_tau[i-1]
            upper = t_end if i == len(segments_t)-1 else events_tau[i]
            
            # Sigmoid Mask
            # Use jax.nn.sigmoid
            mask = jax.nn.sigmoid(blend_sharpness * (t_q - lower)) * jax.nn.sigmoid(blend_sharpness * (upper - t_q))
            
            # Interpolation
            # Clip t_q to segment range to avoid extrapolation errors
            t_clip = jnp.clip(t_q, t_start, t_end)
            
            # Index lookup (differentiable-ish via clip but index is integer)
            # searchsorted is not diff w.r.t grid, but grid moves with W.
            # gradients flow through values at indices.
            idx = jnp.searchsorted(ts, t_clip, side='right') - 1
            idx = jnp.clip(idx, 0, len(ts)-2)
            
            t0_grid, t1_grid = ts[idx], ts[idx+1]
            denom = t1_grid - t0_grid
            denom = jnp.where(jnp.abs(denom) < 1e-12, 1e-12, denom)
            s = (t_clip - t0_grid) / denom
            s = jnp.clip(s, 0.0, 1.0)
            
            val = xs[idx] * (1.0 - s) + xs[idx+1] * s
            
            y_accum += mask * val
            w_accum += mask
            
        return y_accum / (w_accum + 1e-8)

    return jax.vmap(predict_single)(target_times)

def prepare_loss_targets(sol, state_names, t_start, t_end):
    """
    Extract target times and data from solution, filtering out event boundaries.
    Returns (target_times, target_data) as JAX arrays.
    """
    all_t_targets = []
    all_x_targets = []
    
    num_segs = len(sol.segments)
    for i, seg in enumerate(sol.segments):
        t_arr = seg.t
        n = len(t_arr)
        if n == 0: continue
        
        # Determine strict start/end indices to exclude boundary VALUES
        # We use a small tolerance to handle numerical noise in duplicates
        tol_dup = 1e-9
        
        # Left Boundary
        start_idx = 0
        # if i > 0: # Exclude start (te) - COMMENTED OUT for Right Continuity
        #     t_start_val = t_arr[0]
        #     # Skip all points roughly equal to t_start_val
        #     while start_idx < n and abs(t_arr[start_idx] - t_start_val) < tol_dup:
        #         start_idx += 1
                
        # Right Boundary
        end_idx = n
        if i < num_segs - 1: # Exclude end (te next)
            t_end_val = t_arr[-1]
            # Skip all points roughly equal to t_end_val from end
            while end_idx > start_idx and abs(t_arr[end_idx-1] - t_end_val) < tol_dup:
                end_idx -= 1
        
        if end_idx > start_idx:
             t_slice = seg.t[start_idx:end_idx]
             x_slice = seg.x[start_idx:end_idx]
             all_t_targets.append(t_slice)
             all_x_targets.append(x_slice)
             
    target_times = jnp.concatenate([jnp.array(t) for t in all_t_targets])
    target_data = jnp.concatenate([jnp.array(x) for x in all_x_targets])
    
    print("DEBUG: Segment Structure:")
    for i, seg in enumerate(sol.segments):
        print(f"  Seg {i}: len={len(seg.t)}, range=[{seg.t[0]:.4f}, {seg.t[-1]:.4f}]")
        print(f"    Values: {seg.t}")
        
    print(f"DEBUG: Filtered Target Times ({len(target_times)}):")
    print(target_times)
    
    return target_times, target_data

def verify_jacobian_cli():
    # 1. Load and Solve
    dae_data, solver_cfg = load_system('config/config_bouncing_ball.yaml')
    solver = DAESolver(dae_data, verbose=False)
    t_span = (0.0, 2.0)
    ncp = 10 
    sol = solver.solve_augmented(t_span, ncp=ncp)
    
    funcs = create_jax_functions(dae_data)
    f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_p = dims
    n_w = n_x + n_z
    
    # 2. Pack W
    W_flat, structure, grid_taus = pack_solution(sol, dae_data)
    print(f"Packed W size: {W_flat.shape}")
    
    # 3. Parameters
    opt_params = ['g', 'e']
    p_names = [p['name'] for p in dae_data['parameters']]
    p_vals = jnp.array([p['value'] for p in dae_data['parameters']])
    opt_indices = [p_names.index(op) for op in opt_params]
    p_opt = p_vals[jnp.array(opt_indices)]
    
    # 4. Residual and Loss setup
    
    # Ground Truth Data for Loss
    target_times, target_data = prepare_loss_targets(sol, dae_data['states'], solver_cfg['start_time'], solver_cfg['stop_time'])
    
    # Check if empty
    if len(target_times) == 0:
        print("[WARN] Target set empty after filtering! Falling back to raw centers.")
        # Fallback logic if needed, but ncp=10 should suffice.
    
    # Subsample to test interpolation?
    # Use every 2nd point?
    # valid_indices = jnp.arange(0, len(target_times), 2)
    # target_times = target_times[valid_indices]
    # target_data = target_data[valid_indices]

    print(f"Target Data Points: {len(target_times)}")
    
    def loss_function(W, p_cur):
        # 1. Unpack W -> Trajectory
        segs_t, segs_x, segs_z, ev_tau = unpack_solution_structure(W, structure, (n_x, n_z, n_w), grid_taus)
        
        # 2. Map State to Output using h_fn
        # h_fn signature: (t, x, z, p)
        # We need to map over all segments
        segs_y = []
        for i in range(len(segs_t)):
            ts = segs_t[i]
            xs = segs_x[i]
            zs = segs_z[i] if len(segs_z) > i else jnp.zeros((len(ts), 0))
            
            # Vectorize h evaluaton
            # p_cur is (n_opt,). We need p_all.
            p_all = p_vals # Default
            if len(opt_indices) > 0:
                 p_all = p_all.at[jnp.array(opt_indices)].set(p_cur)
                 
            # h_fn expects single points. vmap it.
            h_vmap = jax.vmap(lambda t, x, z: h_fn(t, x, z, p_all))
            ys = h_vmap(ts, xs, zs)
            segs_y.append(ys)
            
        # 3. Predict at target times using OUTPUTS (segs_y)
        # Note: predict_trajectory_sigmoid expects x and z. Now we pass y as "x" and empty "z"?
        # Or update predict signature?
        # Predict signature: (segs_t, segs_x, segs_z, ...).
        # We can pass segs_y as segs_x, and empty segs_z.
        # The blending will happen on y.
        segs_dummy_z = [jnp.zeros((len(t),0)) for t in segs_t]
        y_pred = predict_trajectory_sigmoid(segs_t, segs_y, segs_dummy_z, ev_tau, target_times)
        
        # 4. MSE Loss
        diff = y_pred - target_data
        return jnp.mean(diff**2)
        
    print("Computing Loss Gradients...")
    # DEBUG: Diagnose High Loss (Commented out after verification)
    # segs_t, segs_x, segs_z, ev_tau = unpack_solution_structure(W_flat, structure, (n_x, n_z, n_w), grid_taus)
    # y_pred_debug = predict_trajectory_sigmoid(segs_t, segs_x, segs_z, ev_tau, target_times)
    # ...
    
    L0 = loss_function(W_flat, p_opt)
    print(f"Loss(W_0): {L0:.6e}")
    
    dL_dW_fn = jax.jit(jax.grad(loss_function, argnums=0))
    # dL_dp_fn = jax.jit(jax.grad(loss_function, argnums=1))
    
    start_t = time.time()
    grad_W = dL_dW_fn(W_flat, p_opt)
    end_t = time.time()
    
    print(f"Loss Gradient Time: {end_t - start_t:.4f}s")
    print(f"Norm dL/dW: {jnp.linalg.norm(grad_W):.6e}")
    
    # Also Jacobians of R as requested before
    print("Computing Residual Jacobians...")
    def R_global(W, p_cur):
        return unpack_and_compute_residual(
            W, p_cur, dae_data, structure, funcs, 
            (p_vals, opt_indices), grid_taus
        )
            
    dR_dW_fn = jax.jit(jax.jacfwd(R_global, argnums=0))
    dR_dp_fn = jax.jit(jax.jacfwd(R_global, argnums=1))
    
    J_W = dR_dW_fn(W_flat, p_opt)
    J_p = dR_dp_fn(W_flat, p_opt)
    
    print(f"Norm dR/dW: {jnp.linalg.norm(J_W):.6e}")
    print(f"Norm dR/dp: {jnp.linalg.norm(J_p):.6e}")

    # Solve Adjoint System
    # dR/dW^T * lambda = -dL/dW
    print("Solving Adjoint System...")
    RHS = -grad_W
    # J_W is (n_w, n_w).
    # lambda is (n_w,)
    lambda_adj = jnp.linalg.solve(J_W.T, RHS)
    
    print(f"Norm lambda: {jnp.linalg.norm(lambda_adj):.6e}")
    
    # Verify
    adjoint_residual = jnp.dot(J_W.T, lambda_adj) + grad_W
    print(f"Adjoint Residual Norm: {jnp.linalg.norm(adjoint_residual):.6e}")

    # --- Visualization ---
    print("Generating Interpolation Comparison Plot...")
    
    # 1. Unpack structure for plotting
    segs_t, segs_x, segs_z, ev_tau = unpack_solution_structure(W_flat, structure, (n_x, n_z, n_w), grid_taus)
    
    # Compute segs_y using h_fn
    segs_y = []
    # p_opt is passed here. Assuming constant p for display.
    # Note: h_fn(t,x,z,p). p needs to be full if optimized.
    # We reconstruct p_all
    p_all = p_vals 
    if len(opt_indices) > 0:
         p_all = p_all.at[jnp.array(opt_indices)].set(p_opt)
         
    h_vmap = jax.vmap(lambda t, x, z: h_fn(t, x, z, p_all))
    
    for i in range(len(segs_t)):
        ts = segs_t[i]
        xs = segs_x[i]
        zs = segs_z[i] if len(segs_z) > i else jnp.zeros((len(ts), 0))
        ys = h_vmap(ts, xs, zs)
        segs_y.append(ys)
        
    # 2. Dense Grid for Interpolated Prediction
    t_dense = jnp.linspace(0.0, 2.0, 500)
    # Pass segs_y as "x" for interpolation
    segs_dummy_z = [jnp.zeros((len(t),0)) for t in segs_t]
    y_interp = predict_trajectory_sigmoid(segs_t, segs_y, segs_dummy_z, ev_tau, t_dense)
    
    # Map target_data (which is currently just x) to y
    # target_data currently effectively x_target (concatenated)
    # We should re-extract x/z from filtered indices if possible, or assume identity?
    # Actually, we have target_times and target_data (x).
    # Since target_data was built from seg.x, we need seg.z too to Map.
    # But for Bouncing Ball n_z=0.
    # Generally, we should have kept x and z targets.
    # Re-map target data on the fly?
    # target_data is currently (N, n_x).
    # We can assume z is empty for now or re-extract properly.
    # Re-Extraction is Safer.
    # Or just rely on Bouncing Ball logic?
    # Use h_vmap on the filtered target_data (assuming it is x)
    # Default Bouncing Ball h(x)=x.
    # If h is complex, we need z.
    # For now, apply h_vmap assuming z=empty. This matches current Bouncing Ball config.
    target_y = h_vmap(target_times, target_data, jnp.zeros((len(target_times), 0)))
    
    plt.figure(figsize=(10, 6))
    
    # 3. Plot Non-Interpolated (Raw SEGS_Y)
    n_outputs = segs_y[0].shape[1]
    colors = ['r', 'b', 'g', 'c'] # Extend colors
    labels = [f'y{k}' for k in range(n_outputs)]
    if n_outputs == 2: labels = ['h', 'v'] # Special case name
    
    for i, seq_t in enumerate(segs_t):
        seq_y = segs_y[i]
        for var_idx in range(n_outputs):
            label = f"Raw {labels[var_idx]}" if i == 0 else None
            # Use dashed line for raw
            plt.plot(seq_t, seq_y[:, var_idx], '--', color=colors[var_idx], alpha=0.5, label=label)
            # Add points
            plt.scatter(seq_t, seq_y[:, var_idx], s=10, color=colors[var_idx], alpha=0.5)

    # 4. Plot Interpolated
    for var_idx in range(n_outputs):
        plt.plot(t_dense, y_interp[:, var_idx], '-', color=colors[var_idx], linewidth=2, label=f"Interpolated {labels[var_idx]}")
        
    # 5. Plot Target Data (Filtered Y)
    for var_idx in range(n_outputs):
        # We verify alignment
        plt.scatter(target_times, target_y[:, var_idx], marker='x', s=50, color='k', label="Target (Loss)" if var_idx==0 else None)

    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.title('Approximation: Interpolated Output vs Raw Output')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = 'interpolation_comparison.png'
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")

def verify_residual_cli():
    # 1. Run Solver to get Ground Truth trajectory
    dae_data, solver_cfg = load_system('config/config_bouncing_ball.yaml')
    solver = DAESolver(dae_data, verbose=False)
    
    t_span = (0.0, 2.0)
    # Use small ncp to make vector readable sized
    ncp = 10 
    
    print(f"Solving DAE (g={dae_data['parameters'][0]['value']}...)")
    sol = solver.solve_augmented(t_span, ncp=ncp)
    
    # 2. Compile JAX functions
    funcs = create_jax_functions(dae_data)
    f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_p = dims
    p_val = jnp.array([p['value'] for p in dae_data['parameters']])
    
    # 3. Compute Residuals iterating over the solution structure
    # We don't need to flatten to a single W vector to 'verify' the residual norm,
    # checking the components individually is sufficient and equivalent.
    
    print("\nVerifying Residuals constraint-by-constraint...")
    
    total_res_norm_sq = 0.0
    count = 0
    
    # Iterate Segments
    for i, seg in enumerate(sol.segments):
        ts = seg.t
        xs = seg.x
        zs = seg.z
        
        n_points = len(ts)
        print(f"Segment {i}: {n_points} points")
        
        # Check Flow Residuals (Trapezoidal)
        # 0 = -x(k+1) + x(k) + h/2 * (f(k) + f(k+1))
        # 0 = g(k)
        
        for k in range(n_points - 1):
            t_k = ts[k]
            t_kp1 = ts[k+1]
            x_k = xs[k]
            x_kp1 = xs[k+1]
            z_k = zs[k] if len(zs) > 0 else jnp.array([])
            z_kp1 = zs[k+1] if len(zs) > 0 else jnp.array([]) # g(k) check usually enough? User prompt asks for g at node.
            
            h = t_kp1 - t_k
            
            # f evaluations
            f_k = f_fn(t_k, x_k, z_k, p_val)
            f_kp1 = f_fn(t_kp1, x_kp1, z_kp1, p_val)
            
            # Trapezoidal Res
            # res_diff = x_kp1 - x_k - (h/2)*(f_k + f_kp1) 
            # User notation: 0 = -x(t1) + x(t0) + ...  =>  0 = x(t0) - x(t1) + ...
            # Equivalent norm.
            res_diff = -x_kp1 + x_k + (h/2.0)*(f_k + f_kp1)
            
            norm_diff = jnp.linalg.norm(res_diff)
            total_res_norm_sq += norm_diff**2
            count += 1
            
            if norm_diff > 1e-4:
                 print(f"  [FAIL] Flow Res k={k} ({t_k:.4f}->{t_kp1:.4f}): {norm_diff:.6e}")
            
            # g check (at k)
            if n_z > 0:
                g_val = g_fn(t_k, x_k, z_k, p_val)
                norm_g = jnp.linalg.norm(g_val)
                total_res_norm_sq += norm_g**2
                count += 1
                if norm_g > 1e-4:
                    print(f"  [FAIL] Alg Res k={k}: {norm_g:.6e}")
        
        # Check g at last point too
        if n_z > 0:
             g_val = g_fn(ts[-1], xs[-1], zs[-1], p_val)
             norm_g = jnp.linalg.norm(g_val)
             total_res_norm_sq += norm_g**2
             count += 1
        
        # Check Event Constraints if consistent with next segment
        # User Logic:
        # Segment 0 ends at w_prev(te1). 
        # Verify Guard(w_prev(te1)) = 0
        # Verify Reset(w_prev, w_after) = 0
        
        if i < len(sol.events):
            event = sol.events[i]
            te = event.t_event
            x_pre = event.x_pre
            z_pre = event.z_pre
            x_post = event.x_post
            z_post = event.z_post # z needs to be solved for x_post (usually consistently)
            
            # 1. Guard Condition
            # guard(w_prev) = 0
            # Note: DAESolver triggers when guard CROSSES zero. 
            # Depending on tolerance, it might be 1e-6, not exactly 0.
            val_guard = guard_fn(te, x_pre, z_pre, p_val)
            # The event index that triggered
            ev_idx = event.event_idx
            # We compiled all guards. Check the one relevant?
            # Or all? For now check specific index.
            g_scalar = val_guard[ev_idx]
            
            norm_guard = abs(g_scalar)
            total_res_norm_sq += norm_guard**2
            count += 1
            print(f"  Event {i} Guard ({te:.6f}): {norm_guard:.6e}")
            if norm_guard > 1e-3: # Settle for tolerance
                 print(f"  [WARN] Guard residual > 1e-3")
            
            # 2. Reset Residual
            # Calculate Residual using implicit function
            # reinit_res_fn returns array of residuals for all 'when' clauses.
            reinit_residuals = reinit_res_fn(te, x_post, z_post, x_pre, z_pre, p_val)
            
            # Select the one corresponding to current event
            val_reset = reinit_residuals[ev_idx]
            
            norm_reset = abs(val_reset)
            total_res_norm_sq += norm_reset**2
            count += 1
            
            print(f"  Event {i} Reset: {norm_reset:.6e}")
            
    total_norm = np.sqrt(total_res_norm_sq)
    print(f"\nTotal Residual Norm: {total_norm:.6e} (over {count} constraints)")
    
    # 4. Verify Loss and Direct Gradients (as requested)
    print("\nVerifying Loss and Direct Gradients at Ground Truth...")
    target_times, target_data = prepare_loss_targets(sol, dae_data['states'], solver_cfg['start_time'], solver_cfg['stop_time'])
    W_flat, structure, grid_taus = pack_solution(sol, dae_data)
    n_w = n_x + n_z
    
    def loss_function(W, p_cur):
        segs_t, segs_x, segs_z, ev_tau = unpack_solution_structure(W, structure, (n_x, n_z, n_w), grid_taus)
        y_pred = predict_trajectory_sigmoid(segs_t, segs_x, segs_z, ev_tau, target_times)
        diff = y_pred - target_data
        return jnp.mean(diff**2)
    
    L_val = loss_function(W_flat, p_val)
    dL_dp = jax.grad(loss_function, argnums=1)(W_flat, p_val)
    
    print(f"Loss at Ground Truth: {L_val:.6e}")
    param_names = [p['name'] for p in dae_data['parameters']]
    for i, name in enumerate(param_names):
        print(f"Direct dL/d{name} at Ground Truth: {dL_dp[i]:.6e}")

    if total_norm < 1e-2: # Relaxed tolerance for numerical integration error
        print("\nPASS: Residuals are consistent with trajectory.")
    else:
        print("\nFAIL: Residuals are too high.")

def verify_sensitivity_cli():
    print("--- Sensitivity Analysis ---")
    # 1. Run Solver to get Ground Truth trajectory
    dae_data, solver_cfg = load_system('config/config_bouncing_ball.yaml')
    solver = DAESolver(dae_data, verbose=False)
    
    # We need p_opt
    p_opt = jnp.array([p['value'] for p in dae_data['parameters']])
    param_names = [p['name'] for p in dae_data['parameters']]
    print(f"Parameters: {param_names} = {p_opt}")
    
    # Run Reference Simulation
    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    ncp = 10
    sol = solver.solve_augmented(t_span, ncp=ncp)
    
    # 2. Get Targets (Ground Truth)
    target_times, target_data = prepare_loss_targets(sol, dae_data['states'], solver_cfg['start_time'], solver_cfg['stop_time'])
    
    # 3. Setup Functions
    W_flat, structure, grid_taus = pack_solution(sol, dae_data)
    funcs = create_jax_functions(dae_data)
    f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_p = dims
    n_w = n_x + n_z
    
    # Setup parameter mapping
    opt_params = ['g', 'e']
    param_names = [p['name'] for p in dae_data['parameters']]
    p_all_default = jnp.array([p['value'] for p in dae_data['parameters']])
    opt_indices = [param_names.index(op) for op in opt_params]
    param_mapping = (p_all_default, opt_indices)
    
    def R_global(W, p):
        return unpack_and_compute_residual(
            W, p, dae_data, structure, funcs, param_mapping, grid_taus
        )
        
    def loss_function(W, p_cur):
        segs_t, segs_x, segs_z, ev_tau = unpack_solution_structure(W, structure, (n_x, n_z, n_w), grid_taus)
        y_pred = predict_trajectory_sigmoid(segs_t, segs_x, segs_z, ev_tau, target_times)
        diff = y_pred - target_data
        return jnp.mean(diff**2)
        
    print("Computing Gradients at Optimum...")
    L_opt = loss_function(W_flat, p_opt)
    dL_dp = jax.grad(loss_function, argnums=1)(W_flat, p_opt)
    dL_dW = jax.grad(loss_function, argnums=0)(W_flat, p_opt)
    
    print(f"Loss at Optimum: {L_opt:.6e}")
    for i, name in enumerate(param_names):
        print(f"Direct dL/d{name} at Optimum: {dL_dp[i]:.6e}")
    
    print("Computing Jacobians...")
    dR_dW = jax.jacfwd(R_global, argnums=0)(W_flat, p_opt)
    dR_dp = jax.jacfwd(R_global, argnums=1)(W_flat, p_opt)
    
    print("Solving Adjoint System...")
    # dR_dW.T * lambda = -dL_dW
    lambda_sol = jnp.linalg.solve(dR_dW.T, -dL_dW)
    
    print("Computing Total Derivatives...")
    total_grad = dL_dp + jnp.dot(lambda_sol, dR_dp)
    
    for i, name in enumerate(param_names):
        print(f"dJ/d{name} (Adjoint): {total_grad[i]:.6e}")
        
    print("\n--- Finite Difference Verification (Perturbation 1e-6) ---")
    epsilon = 1e-6
    L_ref = loss_function(W_flat, p_opt)
    
    for i, name in enumerate(param_names):
        p_pert = p_opt.at[i].add(epsilon)
        
        # 1. Update Parameters (Deep Copy)
        import copy
        dae_data_pert = copy.deepcopy(dae_data)
        dae_data_pert['parameters'][i]['value'] = float(p_pert[i])
        
        # 2. Solve Forward with perturbed params
        solver_pert = DAESolver(dae_data_pert, verbose=False)
        sol_pert = solver_pert.solve_augmented(t_span, ncp=ncp)
        
        # 3. Pack W_pert
        W_pert, struct_pert, grid_pert = pack_solution(sol_pert, dae_data_pert)
        
        # 4. Compute Loss (Manually unpack)
        segs_t, segs_x, segs_z, ev_tau = unpack_solution_structure(W_pert, struct_pert, (n_x, n_z, n_w), grid_pert)
        y_pred = predict_trajectory_sigmoid(segs_t, segs_x, segs_z, ev_tau, target_times)
        diff = y_pred - target_data
        L_pert = jnp.mean(diff**2)
        
        fd_grad = (L_pert - L_ref) / epsilon
        adj_grad = total_grad[i]
        
        print(f"{name}: Adj={adj_grad:.6e}, FD={fd_grad:.6e}, Diff={abs(adj_grad - fd_grad):.6e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'jac':
        verify_jacobian_cli()
    elif len(sys.argv) > 1 and sys.argv[1] == 'sens':
        verify_sensitivity_cli()
    else:
        verify_residual_cli()
