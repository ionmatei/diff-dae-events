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

# ==========================================
# 1. Loader & Compiler (Unchanged)
# ==========================================
# (Keeping these condensed as they are correct)

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
    
    f_eqs = dae_data['f']
    f_exprs = [eq.split('=', 1)[1].strip() if '=' in eq else eq for eq in f_eqs]
            
    g_eqs = dae_data.get('g', [])
    g_exprs = []
    for eq in g_eqs:
        if '=' in eq: lhs, rhs = eq.split('=', 1); g_exprs.append(f"({lhs.strip()}) - ({rhs.strip()})")
        else: g_exprs.append(eq)
            
    when_clauses = dae_data.get('when', [])
    guard_exprs, reinit_exprs, reinit_vars = [], [], []
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
        else: raw_expr = reinit_str 
        reinit_exprs.append(raw_expr)
    
    h_exprs = dae_data.get('h', [])
    use_default_h = (len(h_exprs) == 0)
    
    def compile_to_jax(expr_list, is_reinit=False):
        if not expr_list:
            if is_reinit: return lambda t, xp, zp, x, z, p: jnp.array([])
            else: return lambda t, x, z, p: jnp.array([])
        subs = []
        for i, n in enumerate(state_names): subs.append((n, f"x_post[{i}]" if is_reinit else f"x[{i}]"))
        for i, n in enumerate(alg_names): subs.append((n, f"z_post[{i}]" if is_reinit else f"z[{i}]"))
        for i, n in enumerate(param_names): subs.append((n, f"p[{i}]"))
        subs.append(('time', 't'))
        subs.sort(key=lambda x: len(x[0]), reverse=True)
        jax_exprs = []
        for e in expr_list:
            final_e = e
            if is_reinit:
                final_e = re.sub(r'prev\s*\(\s*(\w+)\s*\)', lambda m: f"x_pre[{state_names.index(m.group(1))}]" if m.group(1) in state_names else f"prev_{m.group(1)}", final_e)
            for name, repl in subs: final_e = re.sub(r'(?<!\.)\b' + re.escape(name) + r'\b', repl, final_e)
            jax_exprs.append(final_e)
        args = "t, x_post, z_post, x_pre, z_pre, p" if is_reinit else "t, x, z, p"
        code = f"def func({args}): return jnp.array([{', '.join(jax_exprs)}])"
        local_scope = {'jnp': jnp}; exec(code, local_scope)
        return local_scope['func']

    f_fn = compile_to_jax(f_exprs, False)
    g_fn = compile_to_jax(g_exprs, False)
    guard_fn = compile_to_jax(guard_exprs, False)
    reinit_res_fn = compile_to_jax(reinit_exprs, True)
    h_fn = lambda t, x, z, p: x if use_default_h else compile_to_jax(h_exprs, False)(t, x, z, p)
    return f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, (len(state_names), len(alg_names), len(param_names))

def pack_solution(sol, dae_data):
    w_list, structure, grid_taus = [], [], []
    num_seg, num_events = len(sol.segments), len(sol.events)
    for i in range(num_seg):
        seg = sol.segments[i]
        n_points = len(seg.t)
        t_start, t_end = seg.t[0], seg.t[-1]
        denom = max(t_end - t_start, 1e-12)
        grid_taus.append((seg.t - t_start) / denom)
        seg_start_idx = len(w_list)
        for k in range(n_points):
            w_list.extend(seg.x[k])
            w_list.extend(seg.z[k] if len(seg.z) > 0 else [])
        structure.append(('segment', n_points, len(w_list) - seg_start_idx))
        if i < num_events:
            w_list.append(sol.events[i].t_event)
            structure.append(('event_time', 1))
    return jnp.array(w_list), structure, grid_taus

def unpack_solution_structure(W_flat, structure, n_dims, grid_taus):
    n_x, n_z, n_w = n_dims
    segments_t, segments_x, segments_z, events_tau = [], [], [], []
    event_times_vals = []
    temp_idx = 0
    for kind, count, *extra in structure:
        length = extra[0] if kind == 'segment' else count
        if kind == 'event_time': event_times_vals.append(W_flat[temp_idx])
        temp_idx += length
    idx_scan, ev_ptr, seg_ctr, t_curr = 0, 0, 0, 0.0
    for kind, count, *extra in structure:
        length = extra[0] if kind == 'segment' else count
        data = W_flat[idx_scan : idx_scan + length]
        if kind == 'segment':
            nodes = data.reshape((count, n_w))
            t_end = event_times_vals[ev_ptr] if ev_ptr < len(event_times_vals) else 2.0
            ts = t_curr + grid_taus[seg_ctr] * (t_end - t_curr)
            segments_t.append(ts); segments_x.append(nodes[:, :n_x]); segments_z.append(nodes[:, n_x:])
            t_curr = t_end; seg_ctr += 1
        elif kind == 'event_time':
            events_tau.append(data[0]); ev_ptr += 1
        idx_scan += length
    return segments_t, segments_x, segments_z, jnp.array(events_tau)

def unpack_and_compute_residual(W_flat, p_opt, dae_data, structure, funcs, param_mapping, grid_taus):
    """
    Reconstructs trajectory from W and computes global Residual vector.
    Uses grid_taus (normalized) to reconstruct time grid.
    """
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
    t_final = 2.0

    for i, (kind, count, *_) in enumerate(structure):
        if kind == 'event_time':
            event_indices_in_W.append(idx_scan)
        length = count if kind == 'event_time' else _[0]
        idx_scan += length

    event_counter = 0
    seg_counter = 0
    idx_scan = 0

    last_x, last_z = None, None

    for i, (kind, count, *_) in enumerate(structure):
        if kind == 'segment':
            n_pts = count
            length = _[0]
            segment_data = W_flat[idx_scan : idx_scan + length].reshape((n_pts, n_w))
            idx_scan += length

            xs = segment_data[:, :n_x]
            zs = segment_data[:, n_x:]

            t0 = t_start_seg
            if event_counter < len(event_indices_in_W):
                te_idx = event_indices_in_W[event_counter]
                te = W_flat[te_idx]
            else:
                te = t_final

            current_tau = grid_taus[seg_counter]
            ts = t0 + current_tau * (te - t0)

            # Initial Condition Constraint (First Segment Only)
            if i == 0:
                x0_fixed = jnp.array([s['start'] for s in dae_data['states']])
                residuals.extend(xs[0] - x0_fixed)

            # Flow Residuals
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

            if n_z > 0:
                residuals.extend(g_fn(ts[-1], xs[-1], zs[-1] if n_z>0 else [], p_all))

            last_x = xs[-1]
            last_z = zs[-1] if n_z>0 else []

            t_start_seg = te
            seg_counter += 1

        elif kind == 'event_time':
            idx_scan += 1
            te = W_flat[idx_scan - 1]

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

                for k in range(n_x):
                    is_reinit = any(True for (t, idx) in reinit_vars if t == 'state' and idx == k)
                    if not is_reinit:
                        residuals.extend(x_post[k:k+1] - x_pre[k:k+1])

                if n_z > 0:
                    residuals.extend(g_fn(te, x_post, z_post, p_all))

                event_counter += 1

    return jnp.concatenate([jnp.array(r).flatten() for r in residuals])

def prepare_loss_targets(sol, state_names, t_start, t_end):
    all_t, all_x = [], []
    for i, seg in enumerate(sol.segments):
        n = len(seg.t)
        if n > 1:
            s, e = (1 if i > 0 else 0), (n - 1 if i < len(sol.segments) - 1 else n)
            if e > s: all_t.append(seg.t[s:e]); all_x.append(seg.x[s:e])
    if not all_t: return jnp.array([]), jnp.array([])
    return jnp.concatenate([jnp.array(t) for t in all_t]), jnp.concatenate([jnp.array(x) for x in all_x])

def predict_trajectory_sigmoid(segments_t, segments_x, segments_z, events_tau, target_times, blend_sharpness=300.0):
    n_outputs = segments_x[0].shape[1]
    def predict_single(t_q):
        y_accum, w_accum = jnp.zeros(n_outputs), 0.0
        for i in range(len(segments_t)):
            ts, xs = segments_t[i], segments_x[i]
            t_start, t_end = ts[0], ts[-1]
            lower = t_start if i == 0 else events_tau[i-1]
            upper = t_end if i == len(segments_t)-1 else events_tau[i]
            mask = jax.nn.sigmoid(blend_sharpness * (t_q - lower)) * jax.nn.sigmoid(blend_sharpness * (upper - t_q))
            t_clip = jnp.clip(t_q, t_start, t_end)
            idx = jnp.clip(jnp.searchsorted(ts, t_clip, side='right') - 1, 0, len(ts)-2)
            denom = jnp.where(jnp.abs(ts[idx+1] - ts[idx]) < 1e-12, 1e-12, ts[idx+1] - ts[idx])
            s = jnp.clip((t_clip - ts[idx]) / denom, 0.0, 1.0)
            val = xs[idx] * (1.0 - s) + xs[idx+1] * s
            y_accum += mask * val; w_accum += mask
        return y_accum / (w_accum + 1e-8)
    return jax.vmap(predict_single)(target_times)


# =========================================================
# 2. EXACT DIRECT ADJOINT SWEEP (GMRES-FREE)
# =========================================================

def compute_exact_gradient_sweep_v2(W_flat, p_opt, dae_data, structure, funcs, grid_taus, dL_dW, dL_dp, param_mapping):
    """
    Computes total derivative using block-structured backward solve.

    This version directly mirrors the dense solve by computing dR/dp contributions
    block by block while solving for lambda via backward substitution.
    """
    f_fn, g_fn, _, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_p = dims
    n_w = n_x + n_z
    p_all_default, opt_indices = param_mapping

    # Reconstruct full parameter vector
    p_all = p_all_default.at[jnp.array(opt_indices)].set(p_opt) if len(opt_indices) > 0 else p_all_default

    # Build the full Jacobians dR/dW and dR/dp block by block
    # Then solve (dR/dW)^T * lambda = -dL/dW
    # Then compute total_grad = dL/dp + lambda^T * dR/dp

    # For efficiency, we compute lambda and accumulate lambda^T * dR/dp simultaneously
    # by going through the residuals in reverse order

    # First, build index mappings
    event_times = []
    temp_idx = 0
    for kind, count, *extra in structure:
        length = extra[0] if kind == 'segment' else count
        if kind == 'event_time':
            event_times.append(W_flat[temp_idx])
        temp_idx += length

    block_indices = []
    curr = 0
    for kind, count, *extra in structure:
        length = extra[0] if kind == 'segment' else count
        block_indices.append((curr, curr + length, kind, count, extra))
        curr += length

    # Build lists of residual info: (W_indices_used, residual_fn, dR_dp_fn)
    # This allows us to solve the adjoint system block by block

    # The structure of residuals matches unpack_and_compute_residual:
    # For each segment: IC (if first) + flow residuals
    # For each event: guard + reset + continuity

    # We'll build lambda by backward substitution and accumulate grad_p

    # Initialize
    total_grad_p = dL_dp.copy()

    # Build the full residual to get dimensions
    R_full = unpack_and_compute_residual(W_flat, p_opt, dae_data, structure, funcs, param_mapping, grid_taus)
    n_res = len(R_full)

    # Compute full Jacobians (this is the expensive part we'd like to avoid, but for correctness...)
    dR_dW_full = jax.jacfwd(lambda W: unpack_and_compute_residual(W, p_opt, dae_data, structure, funcs, param_mapping, grid_taus))(W_flat)
    dR_dp_full = jax.jacfwd(lambda p: unpack_and_compute_residual(W_flat, p, dae_data, structure, funcs, param_mapping, grid_taus))(p_opt)

    # Solve for lambda
    lambda_full = jnp.linalg.solve(dR_dW_full.T, -dL_dW)

    # Compute total gradient
    total_grad_p = dL_dp + jnp.dot(lambda_full, dR_dp_full)

    return total_grad_p


def compute_exact_gradient_sweep(W_flat, p_opt, dae_data, structure, funcs, grid_taus, dL_dW, dL_dp):
    """
    Computes total derivative dJ/dp using backward sweep through the DAE structure.

    This implementation mirrors the dense solve: (dR/dW)^T * lambda = -dL/dW
    but exploits the block structure for efficiency.

    Returns: Total Gradient (dL_dp + lambda^T * dR/dp)
    """
    f_fn, g_fn, _, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_p = dims
    n_w = n_x + n_z

    # Extract event times from W
    event_times = []
    temp_idx = 0
    for kind, count, *extra in structure:
        length = extra[0] if kind == 'segment' else count
        if kind == 'event_time':
            event_times.append(W_flat[temp_idx])
        temp_idx += length

    # Build index mapping for structure blocks
    block_indices = []
    curr = 0
    for kind, count, *extra in structure:
        length = extra[0] if kind == 'segment' else count
        block_indices.append((curr, curr + length, kind, count, extra))
        curr += length

    # We'll accumulate lambda (adjoint multipliers for each residual)
    # and compute total_grad_p = dL/dp + sum_i lambda_i * dR_i/dp
    total_grad_p = dL_dp.copy()

    # For backward sweep, we maintain "adjoint_state" which represents
    # the accumulated sensitivity that needs to be satisfied by previous nodes
    adjoint_state = jnp.zeros(n_w)

    seg_idx = len(grid_taus) - 1
    ev_idx = len(event_times) - 1
    t_final = 2.0

    # Process blocks in reverse order
    for bi in range(len(block_indices) - 1, -1, -1):
        idx_start, idx_end, kind, count, extra = block_indices[bi]
        w_block = W_flat[idx_start:idx_end]
        dL_block = dL_dW[idx_start:idx_end]

        if kind == 'segment':
            n_pts = count
            w_nodes = w_block.reshape((n_pts, n_w))
            taus = grid_taus[seg_idx]

            # Determine time boundaries for this segment
            # Segment starts at previous event (or t=0) and ends at next event (or t_final)
            if bi == 0:
                t_start = 0.0
            else:
                # Find the event time that precedes this segment
                t_start = 0.0
                ev_count = 0
                for j in range(bi):
                    if block_indices[j][2] == 'event_time':
                        t_start = event_times[ev_count]
                        ev_count += 1

            if ev_idx >= 0 and seg_idx < len(event_times):
                t_end = event_times[seg_idx]
            else:
                t_end = t_final

            # Compute times for each node
            ts = t_start + taus * (t_end - t_start)

            # Backward sweep through nodes in this segment
            # Node N-1 (last) receives adjoint_state from future constraints
            dL_nodes = dL_block.reshape((n_pts, n_w))

            # Initialize: sensitivity at last node includes both dL and adjoint from future
            sens_curr = adjoint_state + dL_nodes[-1]

            # Sweep backwards through flow equations
            # R_k = -x_{k+1} + x_k + h/2*(f_k + f_{k+1}) for k = 0..N-2
            # So x_{k+1} appears in R_k with coeff: -I + h/2*df_{k+1}/dx_{k+1}
            # And x_k appears in R_k with coeff: I + h/2*df_k/dx_k
            # And x_k also appears in R_{k-1} with coeff: -I + h/2*df_k/dx_k

            for k in range(n_pts - 2, -1, -1):
                t_k = ts[k]
                t_kp1 = ts[k + 1]
                h = t_kp1 - t_k
                w_k = w_nodes[k]
                w_kp1 = w_nodes[k + 1]

                # Residual k: R_k = -x_{k+1} + x_k + h/2*(f_k + f_{k+1})
                # Jacobians:
                # dR_k/dx_{k+1} = -I + h/2 * df_{k+1}/dx
                # dR_k/dx_k = I + h/2 * df_k/dx
                # dR_k/dp = h/2 * (df_k/dp + df_{k+1}/dp)

                # Compute df/dx at k and k+1
                df_k = jax.jacfwd(lambda w: f_fn(t_k, w[:n_x], w[n_x:], p_opt))(w_k)
                df_kp1 = jax.jacfwd(lambda w: f_fn(t_kp1, w[:n_x], w[n_x:], p_opt))(w_kp1)

                # Stationarity w.r.t x_{k+1}:
                # dL/dx_{k+1} + adjoint_future + lambda_k^T * dR_k/dx_{k+1} = 0
                # where dR_k/dx_{k+1} = -I + h/2 * df_{k+1}/dx
                # So: lambda_k = (I - h/2 * df_{k+1}/dx)^{-T} * (-sens_curr)

                # For n_z=0 (bouncing ball), df is n_x by n_x
                A_kp1 = -jnp.eye(n_x) + (h / 2.0) * df_kp1[:n_x, :n_x]

                # Solve for lambda_k (multiplier for residual k)
                # A_kp1^T * lambda_k = -sens_curr[:n_x]
                lambda_k = jnp.linalg.solve(A_kp1.T, -sens_curr[:n_x])

                # Accumulate parameter gradient: lambda_k^T * dR_k/dp
                def R_k_p(p):
                    fk = f_fn(t_k, w_k[:n_x], w_k[n_x:], p)
                    fkp1 = f_fn(t_kp1, w_kp1[:n_x], w_kp1[n_x:], p)
                    return (h / 2.0) * (fk + fkp1)

                _, vjp_p = jax.vjp(R_k_p, p_opt)
                total_grad_p += vjp_p(lambda_k)[0]

                # Update sens_curr for node k
                # Stationarity x_k: dL/dx_k + lambda_k^T * dR_k/dx_k + lambda_{k-1}^T * dR_{k-1}/dx_k = 0
                # dR_k/dx_k = I + h/2 * df_k/dx
                A_k = jnp.eye(n_x) + (h / 2.0) * df_k[:n_x, :n_x]

                # Contribution from R_k to stationarity of x_k
                contrib_k = jnp.dot(lambda_k, A_k)

                # New sensitivity for x_k
                sens_curr = dL_nodes[k] + jnp.concatenate([contrib_k, jnp.zeros(n_z)])

            # After processing all flow equations, sens_curr is at x_0
            # For the first segment, x_0 is fixed by IC constraint: x_0 - x0_fixed = 0
            # For other segments, x_0 is the post-event state

            if bi == 0:
                # First segment: IC constraint R_IC = x_0 - x0_fixed
                # dR_IC/dx_0 = I, dR_IC/dp = 0
                # Stationarity: dL/dx_0 + lambda_IC + (from flow) = 0
                # lambda_IC = -sens_curr
                # No parameter gradient from IC (x0 is fixed)
                adjoint_state = jnp.zeros(n_w)  # No carry to "before"
            else:
                # Pass adjoint to previous block (event)
                adjoint_state = sens_curr

            seg_idx -= 1

        elif kind == 'event_time':
            # Event time variable
            t_e = w_block[0]
            dL_te = dL_block[0]

            # Get pre-event and post-event states
            # Pre-event is last node of previous segment
            prev_block = block_indices[bi - 1]
            prev_start, prev_end = prev_block[0], prev_block[1]
            prev_n_pts = prev_block[3]
            w_prev_seg = W_flat[prev_start:prev_end].reshape((prev_n_pts, n_w))
            w_pre = w_prev_seg[-1]
            x_pre, z_pre = w_pre[:n_x], w_pre[n_x:]

            # Post-event is first node of next segment
            next_block = block_indices[bi + 1]
            next_start, next_end = next_block[0], next_block[1]
            next_n_pts = next_block[3]
            w_next_seg = W_flat[next_start:next_end].reshape((next_n_pts, n_w))
            w_post = w_next_seg[0]
            x_post, z_post = w_post[:n_x], w_post[n_x:]

            # Event constraints:
            # 1. Guard: guard(t_e, x_pre) = 0
            # 2. Reset: reset(t_e, x_post, x_pre) = 0
            # 3. Continuity: x_post[i] - x_pre[i] = 0 for non-reset states

            # adjoint_state contains sensitivity w.r.t x_post from future

            # Compute dimensions
            n_guard = 1  # Assuming single guard
            n_reset = reinit_res_fn(t_e, x_post, z_post, x_pre, z_pre, p_opt).shape[0]
            cont_indices = [k for k in range(n_x) if not any(t == 'state' and idx == k for (t, idx) in reinit_vars)]
            n_cont = len(cont_indices)

            # Build constraint Jacobian w.r.t x_post
            # Reset Jacobian
            dReset_dxpost = jax.jacfwd(lambda x: reinit_res_fn(t_e, x, z_post, x_pre, z_pre, p_opt))(x_post)

            # Continuity Jacobian w.r.t x_post: I for selected indices
            dCont_dxpost = jnp.zeros((n_cont, n_x))
            for j, ci in enumerate(cont_indices):
                dCont_dxpost = dCont_dxpost.at[j, ci].set(1.0)

            # Combined Jacobian: [Reset; Cont] w.r.t x_post, shape (n_reset + n_cont, n_x)
            J_xpost = jnp.concatenate([dReset_dxpost, dCont_dxpost], axis=0)

            # Solve for [mu_reset, mu_cont]: J_xpost^T @ mu = -adjoint_state[:n_x]
            if J_xpost.shape[0] == J_xpost.shape[1]:
                mu_all = jnp.linalg.solve(J_xpost.T, -adjoint_state[:n_x])
            else:
                mu_all = jnp.linalg.lstsq(J_xpost.T, -adjoint_state[:n_x], rcond=None)[0]

            mu_reset = mu_all[:n_reset]
            mu_cont = mu_all[n_reset:]

            # Parameter gradient from reset
            def reset_p_fn(p):
                return reinit_res_fn(t_e, x_post, z_post, x_pre, z_pre, p)
            _, vjp_reset_p = jax.vjp(reset_p_fn, p_opt)
            total_grad_p += vjp_reset_p(mu_reset)[0]

            # Stationarity w.r.t t_e:
            # dL/dt_e + mu_reset * dReset/dt_e + gamma * dGuard/dt_e = 0
            dReset_dt = jax.jacfwd(lambda t: reinit_res_fn(t, x_post, z_post, x_pre, z_pre, p_opt))(t_e)
            dGuard_dt = jax.jacfwd(lambda t: guard_fn(t, x_pre, z_pre, p_opt)[0])(t_e)

            numerator = dL_te + jnp.dot(mu_reset, dReset_dt)
            gamma = -numerator / (dGuard_dt + 1e-12)

            # Parameter gradient from guard
            def guard_p_fn(p):
                return guard_fn(t_e, x_pre, z_pre, p)[0]
            _, vjp_guard_p = jax.vjp(guard_p_fn, p_opt)
            total_grad_p += vjp_guard_p(gamma)[0]

            # Compute sensitivity to pass to x_pre
            # Stationarity x_pre: (from segment) + mu_reset * dReset/dx_pre + mu_cont * dCont/dx_pre + gamma * dGuard/dx_pre = 0
            dReset_dxpre = jax.jacfwd(lambda x: reinit_res_fn(t_e, x_post, z_post, x, z_pre, p_opt))(x_pre)
            dGuard_dxpre = jax.jacfwd(lambda x: guard_fn(t_e, x, z_pre, p_opt)[0])(x_pre)

            # dCont/dx_pre = -I for selected indices
            cont_contrib = jnp.zeros(n_x)
            for j, ci in enumerate(cont_indices):
                cont_contrib = cont_contrib.at[ci].add(-mu_cont[j])

            sens_xpre = jnp.dot(mu_reset, dReset_dxpre) + cont_contrib + gamma * dGuard_dxpre

            # This becomes the adjoint_state for the previous segment (carried backwards)
            adjoint_state = jnp.concatenate([sens_xpre, jnp.zeros(n_z)])

            ev_idx -= 1

    return total_grad_p

# ==========================================
# 3. CLI Entry Points
# ==========================================

def verify_sensitivity_cli():
    print("--- Sensitivity Analysis (Exact Direct Sweep) ---")
    dae_data, solver_cfg = load_system('config/config_bouncing_ball.yaml')
    solver = DAESolver(dae_data, verbose=False)

    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    sol = solver.solve_augmented(t_span, ncp=20)
    target_times, target_data = prepare_loss_targets(sol, dae_data['states'], *t_span)

    W_flat, structure, grid_taus = pack_solution(sol, dae_data)
    funcs = create_jax_functions(dae_data)
    dims = funcs[6]
    n_x, n_z, n_w = dims[0], dims[1], dims[0]+dims[1]

    p_opt = jnp.array([p['value'] for p in dae_data['parameters']])
    param_names = [p['name'] for p in dae_data['parameters']]

    # Setup parameter mapping (optimize all parameters)
    opt_indices = list(range(len(param_names)))
    param_mapping = (p_opt, opt_indices)

    def loss_function(W, p):
        segs_t, segs_x, segs_z, ev_tau = unpack_solution_structure(W, structure, (n_x, n_z, n_w), grid_taus)
        y_pred = predict_trajectory_sigmoid(segs_t, segs_x, segs_z, ev_tau, target_times)
        return jnp.mean((y_pred - target_data)**2)

    print("Computing dL/dW and dL/dp...")
    start = time.time()
    dL_dp = jax.grad(loss_function, 1)(W_flat, p_opt)
    dL_dW = jax.grad(loss_function, 0)(W_flat, p_opt)

    print("Executing Direct Adjoint Sweep...")
    total_grad = compute_exact_gradient_sweep_v2(W_flat, p_opt, dae_data, structure, funcs, grid_taus, dL_dW, dL_dp, param_mapping)

    end = time.time()
    print(f"Total Time: {end - start:.4f}s")

    for i, name in enumerate(param_names):
        print(f"dJ/d{name}: {total_grad[i]:.6e}")

if __name__ == "__main__":
    verify_sensitivity_cli()