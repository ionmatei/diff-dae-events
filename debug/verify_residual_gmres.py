import numpy as np
import yaml
import json
import sys
import os
import time
import jax
import jax.numpy as jnp
from jax import config
from functools import partial

config.update("jax_enable_x64", True)

# Add src to path
sys.path.append(os.getcwd())

from src.discrete_adjoint.dae_solver import DAESolver
import re
import matplotlib.pyplot as plt

# --- 1. Loader & Compiler Functions (Unchanged) ---

def load_system(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    solver_cfg = config['dae_solver']
    dae_spec_file = solver_cfg['dae_specification_file']
    
    with open(dae_spec_file, 'r') as f:
        dae_data = json.load(f)
        
    return dae_data, solver_cfg

def create_jax_functions(dae_data):
    state_names = [s['name'] for s in dae_data['states']]
    alg_names = [a['name'] for a in dae_data.get('alg_vars', [])]
    param_names = [p['name'] for p in dae_data['parameters']]
    
    # Compile f
    f_eqs = dae_data['f']
    f_exprs = [eq.split('=', 1)[1].strip() if '=' in eq else eq for eq in f_eqs]
            
    # Compile g
    g_eqs = dae_data.get('g', [])
    g_exprs = []
    for eq in g_eqs:
        if '=' in eq:
            lhs, rhs = eq.split('=', 1)
            g_exprs.append(f"({lhs.strip()}) - ({rhs.strip()})")
        else:
            g_exprs.append(eq)
            
    # Compile Guard & Reinit
    when_clauses = dae_data.get('when', [])
    guard_exprs = []
    reinit_exprs = [] 
    reinit_vars = [] 
    
    for wc in when_clauses:
        cond = wc['condition']
        if '<' in cond: lhs, rhs = cond.split('<', 1)
        elif '>' in cond: lhs, rhs = cond.split('>', 1)
        else: lhs, rhs = cond.split('=', 1)
        guard_exprs.append(f"({lhs}) - ({rhs})")
        
        reinit_str = wc['reinit']
        if '=' in reinit_str:
            lhs, rhs = reinit_str.split('=', 1)
            raw_expr = f"({lhs}) - ({rhs})"
            lhs_clean = lhs
            for i, name in enumerate(state_names):
                 if re.search(r'\b' + re.escape(name) + r'\b', lhs_clean):
                     reinit_vars.append(('state', i))
                     break
        else:
            raw_expr = reinit_str 
        reinit_exprs.append(raw_expr)
    
    h_exprs = dae_data.get('h', [])
    use_default_h = (len(h_exprs) == 0)
    
    def compile_to_jax(expr_list, is_reinit=False):
        if not expr_list:
            if is_reinit: return lambda t, xp, zp, x, z, p: jnp.array([])
            else: return lambda t, x, z, p: jnp.array([])
            
        subs = []
        for i, n in enumerate(state_names): 
            target = f"x_post[{i}]" if is_reinit else f"x[{i}]"
            subs.append((n, target))
        for i, n in enumerate(alg_names):
            target = f"z_post[{i}]" if is_reinit else f"z[{i}]"
            subs.append((n, target))
        for i, n in enumerate(param_names): subs.append((n, f"p[{i}]"))
        subs.append(('time', 't'))
        subs.sort(key=lambda x: len(x[0]), reverse=True)
        
        jax_exprs = []
        for e in expr_list:
            final_e = e
            if is_reinit:
                def replace_prev(match):
                    var = match.group(1)
                    if var in state_names: return f"x_pre[{state_names.index(var)}]"
                    if var in alg_names: return f"z_pre[{alg_names.index(var)}]"
                    return f"prev_{var}"
                final_e = re.sub(r'prev\s*\(\s*(\w+)\s*\)', replace_prev, final_e)
            
            for name, repl in subs:
                pattern = r'(?<!\.)\b' + re.escape(name) + r'\b'
                final_e = re.sub(pattern, repl, final_e)
            jax_exprs.append(final_e)
            
        args = "t, x_post, z_post, x_pre, z_pre, p" if is_reinit else "t, x, z, p"
        code = f"def func({args}): return jnp.array([{', '.join(jax_exprs)}])"
        local_scope = {'jnp': jnp}
        exec(code, local_scope)
        return local_scope['func']

    f_fn = compile_to_jax(f_exprs, False)
    g_fn = compile_to_jax(g_exprs, False)
    guard_fn = compile_to_jax(guard_exprs, False)
    reinit_res_fn = compile_to_jax(reinit_exprs, True)
    h_fn = lambda t, x, z, p: x if use_default_h else compile_to_jax(h_exprs, False)(t, x, z, p)

    return f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, tuple(reinit_vars), (len(state_names), len(alg_names), len(param_names))


# --- 2. Optimized Residual Logic (SCAN) ---

@partial(jax.jit, static_argnames=['funcs', 'dims'])
def compute_segment_residual_scan(segment_data, t_start, t_end, grid_taus, funcs, dims, p_val):
    """
    Computes residuals for a SINGLE continuous segment using jax.lax.scan.
    Replaces the Python loop over time steps.
    """
    f_fn, g_fn, _, _, _, _, _ = funcs
    n_x, n_z, n_p = dims
    
    # 1. Prepare Inputs
    # segment_data: (N, n_x + n_z)
    w_curr = segment_data[:-1] 
    w_next = segment_data[1:]
    tau_curr = grid_taus[:-1]
    tau_next = grid_taus[1:]
    
    # 2. Define Scan Step
    def step_fn(carry, inputs):
        w_k, w_kp1, tau_k, tau_kp1 = inputs
        
        # Recover time
        t_k = t_start + tau_k * (t_end - t_start)
        t_kp1 = t_start + tau_kp1 * (t_end - t_start)
        h = t_kp1 - t_k
        
        x_k, z_k = w_k[:n_x], w_k[n_x:]
        x_kp1, z_kp1 = w_kp1[:n_x], w_kp1[n_x:]
        
        # Dynamics
        f_k = f_fn(t_k, x_k, z_k, p_val)
        f_kp1 = f_fn(t_kp1, x_kp1, z_kp1, p_val)
        
        # Trapezoidal Rule
        res_flow = -x_kp1 + x_k + (h / 2.0) * (f_k + f_kp1)
        
        # Algebraics (at k)
        res_alg = jnp.array([])
        if n_z > 0:
            res_alg = g_fn(t_k, x_k, z_k, p_val)
            
        return carry, jnp.concatenate([res_flow, res_alg])

    # 3. Execute Scan
    # We iterate N-1 times
    scan_inputs = (w_curr, w_next, tau_curr, tau_next)
    _, residuals_stacked = jax.lax.scan(step_fn, None, scan_inputs)
    
    # 4. Handle Boundary (g at last point N-1)
    extra_res = jnp.array([])
    if n_z > 0:
        w_last = segment_data[-1]
        t_last = t_end 
        extra_res = g_fn(t_last, w_last[:n_x], w_last[n_x:], p_val)
        
    return jnp.concatenate([residuals_stacked.flatten(), extra_res])

def pack_solution(sol, dae_data):
    w_list = []
    num_seg = len(sol.segments)
    num_events = len(sol.events)
    structure = []
    grid_taus = [] 
    
    for i in range(num_seg):
        seg = sol.segments[i]
        n_points = len(seg.t)
        
        t_start = seg.t[0]
        t_end = seg.t[-1]
        denom = t_end - t_start
        if denom < 1e-12: denom = 1.0
        tau = (seg.t - t_start) / denom
        grid_taus.append(tau)
        
        seg_start_idx = len(w_list)
        for k in range(n_points):
            w_list.extend(seg.x[k])
            w_list.extend(seg.z[k] if len(seg.z) > 0 else [])
            
        seg_len = len(w_list) - seg_start_idx
        structure.append(('segment', n_points, seg_len))
        
        if i < num_events:
            ev = sol.events[i]
            w_list.append(ev.t_event)
            structure.append(('event_time', 1))
            
    return jnp.array(w_list), structure, grid_taus

def unpack_and_compute_residual(W_flat, p_opt, dae_data, structure, funcs, param_mapping, grid_taus):
    """
    Computes global residual vector.
    Uses SCAN for segments (Fast) and standard logic for events.
    """
    f_fn, g_fn, _, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, _ = dims
    n_w = n_x + n_z
    
    # Reconstruct Parameters
    p_all_default, opt_indices = param_mapping
    p_all = p_all_default
    if len(opt_indices) > 0:
        p_all = p_all.at[jnp.array(opt_indices)].set(p_opt)
    
    residuals_list = []
    idx_scan = 0
    seg_counter = 0
    
    # Extract event times for time grid reconstruction
    event_times_vals = []
    temp_idx = 0
    for kind, count, *extra in structure:
        length = extra[0] if kind == 'segment' else count
        if kind == 'event_time':
            event_times_vals.append(W_flat[temp_idx])
        temp_idx += length
        
    t_curr = 0.0
    t_final_fixed = 2.0
    ev_ptr = 0
    
    last_x, last_z = None, None
    
    for i, (kind, count, *_) in enumerate(structure):
        length = _[0] if kind == 'segment' else count
        data_block = W_flat[idx_scan : idx_scan + length]
        
        if kind == 'segment':
            n_pts = count
            segment_nodes = data_block.reshape((n_pts, n_w))
            
            # Determine bounds
            t_start = t_curr
            t_end = event_times_vals[ev_ptr] if ev_ptr < len(event_times_vals) else t_final_fixed
            
            # 1. IC Constraint
            if i == 0:
                 x0_fixed = jnp.array([s['start'] for s in dae_data['states']])
                 residuals_list.append(segment_nodes[0, :n_x] - x0_fixed)

            # 2. Scanned Dynamics (Optimized)
            taus = grid_taus[seg_counter]
            seg_res = compute_segment_residual_scan(
                segment_nodes, t_start, t_end, taus, funcs, dims, p_all
            )
            residuals_list.append(seg_res)

            last_x = segment_nodes[-1, :n_x]
            last_z = segment_nodes[-1, n_x:]
            
            t_curr = t_end
            seg_counter += 1
            
        elif kind == 'event_time':
            te = data_block[0]
            
            # 3. Event Constraints
            if i + 1 < len(structure):
                # Peek next segment
                next_kind, next_count, *next_extra = structure[i+1]
                next_len = next_extra[0]
                next_data = W_flat[idx_scan + length : idx_scan + length + next_len]
                next_nodes = next_data.reshape((next_count, n_w))
                
                x_post = next_nodes[0, :n_x]
                z_post = next_nodes[0, n_x:]
                
                # Guard Condition
                residuals_list.append(guard_fn(te, last_x, last_z, p_all))
                
                # Reset Map
                residuals_list.append(reinit_res_fn(te, x_post, z_post, last_x, last_z, p_all))
                
                # Continuity for non-reset variables
                for k in range(n_x):
                     is_reinit = any(True for (t, idx) in reinit_vars if t == 'state' and idx == k)
                     if not is_reinit:
                         residuals_list.append(x_post[k:k+1] - last_x[k:k+1])
                         
                if n_z > 0:
                    residuals_list.append(g_fn(te, x_post, z_post, p_all))

            ev_ptr += 1
            
        idx_scan += length

    return jnp.concatenate([r.flatten() for r in residuals_list])


# --- 3. Helpers (Loss & Prediction) ---

def unpack_solution_structure(W_flat, structure, n_dims, grid_taus):
    n_x, n_z, n_w = n_dims
    segments_t = []
    segments_x = []
    segments_z = []
    events_tau = []
    
    # Extract event times first
    event_times_vals = []
    temp_idx = 0
    for kind, count, *extra in structure:
        length = extra[0] if kind == 'segment' else count
        if kind == 'event_time':
            event_times_vals.append(W_flat[temp_idx])
        temp_idx += length

    idx_scan = 0
    t_curr = 0.0
    t_final = 2.0
    ev_ptr = 0
    seg_ctr = 0

    for kind, count, *extra in structure:
        length = extra[0] if kind == 'segment' else count
        data = W_flat[idx_scan : idx_scan + length]
        
        if kind == 'segment':
            nodes = data.reshape((count, n_w))
            t_end = event_times_vals[ev_ptr] if ev_ptr < len(event_times_vals) else t_final
            
            tau = grid_taus[seg_ctr]
            ts = t_curr + tau * (t_end - t_curr)
            
            segments_t.append(ts)
            segments_x.append(nodes[:, :n_x])
            segments_z.append(nodes[:, n_x:])
            
            t_curr = t_end
            seg_ctr += 1
        elif kind == 'event_time':
            events_tau.append(data[0])
            ev_ptr += 1
        
        idx_scan += length
        
    return segments_t, segments_x, segments_z, jnp.array(events_tau)

def predict_trajectory_sigmoid(segments_t, segments_x, segments_z, events_tau, target_times, blend_sharpness=300.0):
    n_outputs = segments_x[0].shape[1]
    
    def predict_single(t_q):
        y_accum = jnp.zeros(n_outputs)
        w_accum = 0.0
        
        for i in range(len(segments_t)):
            ts = segments_t[i]
            xs = segments_x[i]
            t_start, t_end = ts[0], ts[-1]
            
            lower = t_start if i == 0 else events_tau[i-1]
            upper = t_end if i == len(segments_t)-1 else events_tau[i]
            
            mask = jax.nn.sigmoid(blend_sharpness * (t_q - lower)) * jax.nn.sigmoid(blend_sharpness * (upper - t_q))
            t_clip = jnp.clip(t_q, t_start, t_end)
            
            idx = jnp.searchsorted(ts, t_clip, side='right') - 1
            idx = jnp.clip(idx, 0, len(ts)-2)
            
            t0_g, t1_g = ts[idx], ts[idx+1]
            denom = jnp.where(jnp.abs(t1_g - t0_g) < 1e-12, 1e-12, t1_g - t0_g)
            s = jnp.clip((t_clip - t0_g) / denom, 0.0, 1.0)
            
            val = xs[idx] * (1.0 - s) + xs[idx+1] * s
            y_accum += mask * val
            w_accum += mask
            
        return y_accum / (w_accum + 1e-8)

    return jax.vmap(predict_single)(target_times)

def prepare_loss_targets(sol, state_names, t_start, t_end):
    all_t, all_x = [], []
    for i, seg in enumerate(sol.segments):
        n = len(seg.t)
        if n == 0: continue
        start_idx, end_idx = 0, n
        
        # Simple boundary filtering
        if i > 0: start_idx += 1
        if i < len(sol.segments) - 1: end_idx -= 1
        
        if end_idx > start_idx:
            all_t.append(seg.t[start_idx:end_idx])
            all_x.append(seg.x[start_idx:end_idx])
            
    if not all_t: return jnp.array([]), jnp.array([])
    return jnp.concatenate([jnp.array(t) for t in all_t]), jnp.concatenate([jnp.array(x) for x in all_x])


# --- 4. CLI Entry Points ---

def verify_sensitivity_cli():
    print("--- Sensitivity Analysis (Optimized Scan) ---")
    dae_data, solver_cfg = load_system('config/config_bouncing_ball.yaml')
    solver = DAESolver(dae_data, verbose=False)
    
    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    sol = solver.solve_augmented(t_span, ncp=100) # Increased ncp to show scan benefit
    
    target_times, target_data = prepare_loss_targets(sol, dae_data['states'], *t_span)
    
    funcs = create_jax_functions(dae_data)
    _, _, _, _, _, _, dims = funcs
    n_x, n_z, n_p = dims
    n_w = n_x + n_z
    
    W_flat, structure, grid_taus = pack_solution(sol, dae_data)
    print(f"Packed W size: {W_flat.shape[0]} (ncp={100})")
    
    p_opt = jnp.array([p['value'] for p in dae_data['parameters']])
    param_names = [p['name'] for p in dae_data['parameters']]
    opt_indices = [0, 1] # Assume first two params optimized
    
    def R_global(W, p):
        return unpack_and_compute_residual(
            W, p, dae_data, structure, funcs, (p_opt, opt_indices), grid_taus
        )
        
    def loss_function(W, p):
        segs_t, segs_x, segs_z, ev_tau = unpack_solution_structure(W, structure, (n_x, n_z, n_w), grid_taus)
        y_pred = predict_trajectory_sigmoid(segs_t, segs_x, segs_z, ev_tau, target_times)
        return jnp.mean((y_pred - target_data)**2)

    print("Computing Gradients...")
    start = time.time()
    dL_dp = jax.grad(loss_function, 1)(W_flat, p_opt)
    dL_dW = jax.grad(loss_function, 0)(W_flat, p_opt)
    
    # Adjoint Solve
    # A^T * lambda = -dL/dW
    # Use GMRES with VJP operator to avoid instantiating matrix
    from jax.scipy.sparse.linalg import gmres
    
    def At_operator(v):
        _, vjp_fun = jax.vjp(lambda w: R_global(w, p_opt), W_flat)
        return vjp_fun(v)[0]
    
    print("Solving linear system (GMRES)...")
    lambda_adj, info = gmres(At_operator, -dL_dW, tol=1e-6, maxiter=2000)
    
    # dR/dp^T * lambda
    # We use VJP of R_global w.r.t p
    _, vjp_p = jax.vjp(lambda p: R_global(W_flat, p), p_opt)
    term_2 = vjp_p(lambda_adj)[0]
    
    total_grad = dL_dp + term_2
    end = time.time()
    
    print(f"Total Computation Time: {end - start:.4f}s")
    for i, name in enumerate(param_names):
        if i < len(total_grad):
             print(f"dJ/d{name}: {total_grad[i]:.6e}")

if __name__ == "__main__":
    verify_sensitivity_cli()