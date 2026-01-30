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

# ==========================================
# 1. Loaders & Helpers (Unchanged)
# ==========================================

def load_system(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return json.load(open(config['dae_solver']['dae_specification_file'])), config['dae_solver']

def create_jax_functions(dae_data):
    state_names = [s['name'] for s in dae_data['states']]
    alg_names = [a['name'] for a in dae_data.get('alg_vars', [])]
    param_names = [p['name'] for p in dae_data['parameters']]
    
    f_exprs = [eq.split('=', 1)[1].strip() if '=' in eq else eq for eq in dae_data['f']]
    g_exprs = [f"({eq.split('=', 1)[0].strip()}) - ({eq.split('=', 1)[1].strip()})" if '=' in eq else eq for eq in dae_data.get('g', [])]
    
    guard_exprs, reinit_exprs, reinit_vars = [], [], []
    for wc in dae_data.get('when', []):
        cond = wc['condition']
        guard_exprs.append(f"({cond.split('<')[0]}) - ({cond.split('<')[1]})" if '<' in cond else f"({cond.split('>')[0]}) - ({cond.split('>')[1]})")
        lhs, rhs = wc['reinit'].split('=')
        reinit_exprs.append(f"({lhs}) - ({rhs})")
        lhs_clean = lhs.strip()
        for i, name in enumerate(state_names):
            if re.search(r'\b' + re.escape(name) + r'\b', lhs_clean): reinit_vars.append(('state', i)); break

    h_exprs = dae_data.get('h', [])
    
    def compile(exprs, is_reinit=False):
        if not exprs: return lambda *args: jnp.array([])
        subs = []
        for i, n in enumerate(state_names): subs.append((n, f"x_post[{i}]" if is_reinit else f"x[{i}]"))
        for i, n in enumerate(alg_names): subs.append((n, f"z_post[{i}]" if is_reinit else f"z[{i}]"))
        for i, n in enumerate(param_names): subs.append((n, f"p[{i}]"))
        subs.append(('time', 't'))
        subs.sort(key=lambda x: len(x[0]), reverse=True)
        
        jax_exprs = []
        for e in exprs:
            final = e
            if is_reinit: final = re.sub(r'prev\s*\(\s*(\w+)\s*\)', lambda m: f"x_pre[{state_names.index(m.group(1))}]" if m.group(1) in state_names else f"prev_{m.group(1)}", final)
            for k, v in subs: final = re.sub(r'(?<!\.)\b' + re.escape(k) + r'\b', v, final)
            jax_exprs.append(final)
        
        args = "t, x_post, z_post, x_pre, z_pre, p" if is_reinit else "t, x, z, p"
        local = {'jnp': jnp}
        exec(f"def func({args}): return jnp.array([{', '.join(jax_exprs)}])", local)
        return local['func']

    return compile(f_exprs), compile(g_exprs), compile(h_exprs) if h_exprs else (lambda t,x,z,p: x), compile(guard_exprs), compile(reinit_exprs, True), tuple(reinit_vars), (len(state_names), len(alg_names), len(param_names))

def pack_solution(sol, dae_data):
    w_list, structure, grid_taus = [], [], []
    for i, seg in enumerate(sol.segments):
        n = len(seg.t)
        t0, tf = seg.t[0], seg.t[-1]
        grid_taus.append((seg.t - t0) / max(tf - t0, 1e-12))
        start = len(w_list)
        for k in range(n): w_list.extend(seg.x[k]); w_list.extend(seg.z[k] if len(seg.z) else [])
        structure.append(('segment', n, len(w_list) - start))
        if i < len(sol.events):
            w_list.append(sol.events[i].t_event)
            structure.append(('event_time', 1))
    return jnp.array(w_list), structure, grid_taus

def unpack_solution_structure(W, structure, n_dims, grid_taus):
    n_x, n_z, n_w = n_dims
    segs_t, segs_x, segs_z, ev_tau = [], [], [], []
    
    # Extract event times first for grid reconstruction
    ev_times = []
    curr = 0
    for k, c, *extra in structure:
        l = extra[0] if k == 'segment' else c
        if k == 'event_time': ev_times.append(W[curr])
        curr += l
        
    curr, ev_ptr, seg_ptr, t_curr = 0, 0, 0, 0.0
    for k, c, *extra in structure:
        l = extra[0] if k == 'segment' else c
        data = W[curr : curr+l]
        if k == 'segment':
            nodes = data.reshape((c, n_w))
            t_end = ev_times[ev_ptr] if ev_ptr < len(ev_times) else 2.0
            ts = t_curr + grid_taus[seg_ptr] * (t_end - t_curr)
            segs_t.append(ts); segs_x.append(nodes[:, :n_x]); segs_z.append(nodes[:, n_x:])
            t_curr = t_end; seg_ptr += 1
        elif k == 'event_time':
            ev_tau.append(data[0]); ev_ptr += 1
        curr += l
    return segs_t, segs_x, segs_z, jnp.array(ev_tau)

def prepare_loss_targets(sol, state_names, t_start, t_end):
    all_t_targets = []
    all_x_targets = []
    num_segs = len(sol.segments)
    for i, seg in enumerate(sol.segments):
        t_arr = seg.t
        n = len(t_arr)
        if n == 0: continue
        tol_dup = 1e-9
        start_idx = 0
        if i > 0:
            t_start_val = t_arr[0]
            while start_idx < n and abs(t_arr[start_idx] - t_start_val) < tol_dup:
                start_idx += 1
        end_idx = n
        if i < num_segs - 1:
            t_end_val = t_arr[-1]
            while end_idx > start_idx and abs(t_arr[end_idx-1] - t_end_val) < tol_dup:
                end_idx -= 1
        if end_idx > start_idx:
             all_t_targets.append(seg.t[start_idx:end_idx])
             all_x_targets.append(seg.x[start_idx:end_idx])
    return jnp.concatenate([jnp.array(t) for t in all_t_targets]), jnp.concatenate([jnp.array(x) for x in all_x_targets])

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
            idx = jnp.clip(jnp.searchsorted(ts, t_clip, side='right') - 1, 0, len(ts)-2)
            t0_grid, t1_grid = ts[idx], ts[idx+1]
            denom = jnp.where(jnp.abs(t1_grid - t0_grid) < 1e-12, 1e-12, t1_grid - t0_grid)
            s = jnp.clip((t_clip - t0_grid) / denom, 0.0, 1.0)
            val = xs[idx] * (1.0 - s) + xs[idx+1] * s
            y_accum += mask * val
            w_accum += mask
        return y_accum / (w_accum + 1e-8)
    return jax.vmap(predict_single)(target_times)

def unpack_and_compute_residual(W, p_opt, dae_data, structure, funcs, param_mapping, grid_taus):
    f_fn, g_fn, _, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_w = dims[0], dims[1], dims[0]+dims[1]
    p_all = param_mapping[0].at[jnp.array(param_mapping[1])].set(p_opt) if len(param_mapping[1]) > 0 else param_mapping[0]
    
    # Extract event times first for grid reconstruction
    ev_times = []
    curr = 0
    for k, c, *extra in structure:
        l = extra[0] if k == 'segment' else c
        if k == 'event_time': ev_times.append(W[curr])
        curr += l
        
    res, t_curr, ev_ptr, seg_ptr, idx, last_w = [], 0.0, 0, 0, 0, None
    for i, (kind, count, *extra) in enumerate(structure):
        length = extra[0] if kind == 'segment' else count
        w_block = W[idx : idx + length]
        if kind == 'segment':
            nodes = w_block.reshape((count, n_w))
            t_end = ev_times[ev_ptr] if ev_ptr < len(ev_times) else 2.0
            ts = t_curr + grid_taus[seg_ptr] * (t_end - t_curr)
            if i == 0: res.append(nodes[0, :n_x] - jnp.array([s['start'] for s in dae_data['states']]))
            for k in range(count - 1):
                tk, tkp1, h = ts[k], ts[k+1], ts[k+1] - ts[k]
                fk, fkp1 = f_fn(tk, nodes[k,:n_x], nodes[k,n_x:], p_all), f_fn(tkp1, nodes[k+1,:n_x], nodes[k+1,n_x:], p_all)
                res.append(-nodes[k+1, :n_x] + nodes[k, :n_x] + (h/2.0)*(fk + fkp1))
                if n_z > 0: res.append(g_fn(tk, nodes[k, :n_x], nodes[k, n_x:], p_all))
            if n_z > 0: res.append(g_fn(t_end, nodes[-1, :n_x], nodes[-1, n_x:], p_all))
            last_w, t_curr, seg_ptr = nodes[-1], t_end, seg_ptr + 1
        else:
            if i + 1 < len(structure):
                w_next = W[idx+length : idx+length+structure[i+1][2]].reshape((structure[i+1][1], n_w))[0]
                res.append(guard_fn(w_block[0], last_w[:n_x], last_w[n_x:], p_all))
                res.append(reinit_res_fn(w_block[0], w_next[:n_x], w_next[n_x:], last_w[:n_x], last_w[n_x:], p_all))
                for k in range(n_x):
                    if not any(v[1] == k for v in reinit_vars): res.append(w_next[k:k+1] - last_w[k:k+1])
                if n_z > 0: res.append(g_fn(w_block[0], w_next[:n_x], w_next[n_x:], p_all))
            ev_ptr += 1
        idx += length
    return jnp.concatenate([r.flatten() for r in res])

# ==========================================
# 2. LOCAL RESIDUAL & ADJOINT LOGIC
# ==========================================

def local_step_residual(w_next, w_curr, t_next, t_curr, funcs, dims, p):
    """
    Defines the Implicit Trapezoidal Residual: R(w_{k+1}, w_k) = 0
    """
    f_fn, g_fn, _, _, _, _, _ = funcs
    n_x, n_z, _ = dims
    h = t_next - t_curr
    
    x_k, z_k = w_curr[:n_x], w_curr[n_x:]
    x_kp1, z_kp1 = w_next[:n_x], w_next[n_x:]
    
    # Dynamics: x_{k+1} - x_k - h/2 * (f_k + f_{k+1}) = 0
    # Note: Standard form is 0 = -x_{k+1} + ... but we want consistent Jacobian.
    # Let's match the global form: res = -x_{k+1} + x_k + ...
    fk = f_fn(t_curr, x_k, z_k, p)
    fkp1 = f_fn(t_next, x_kp1, z_kp1, p)
    
    res_flow = -x_kp1 + x_k + (h/2.0) * (fk + fkp1)
    
    # Algebraics: g(x_k) = 0.
    # (We enforce g at the current node k)
    res_alg = g_fn(t_curr, x_k, z_k, p) if n_z > 0 else jnp.array([])
    
    return jnp.concatenate([res_flow, res_alg])

def solve_local_adjoint_step(w_next, w_curr, t_next, t_curr, lam_next, dL_curr, funcs, dims, p):
    """
    Solves for lambda_curr given lambda_next.
    Equation: 
      (dR/dw_curr)^T * lambda_curr + (dR/dw_next)^T * lambda_next = -dL/dw_curr
      
    Rearranged:
      (dR/dw_curr)^T * lambda_curr = -dL/dw_curr - (dR/dw_next)^T * lambda_next
    """
    n_w = dims[0] + dims[1]
    
    # 1. Compute RHS Load: -dL/dw_curr - (Contribution from Next)
    # Contribution from Next = (dR/dw_next)^T * lambda_next
    # We use VJP for efficiency and exactness
    _, vjp_next = jax.vjp(lambda wn: local_step_residual(wn, w_curr, t_next, t_curr, funcs, dims, p), w_next)
    load_from_next = vjp_next(lam_next)[0]
    
    rhs = -dL_curr - load_from_next
    
    # 2. Compute Matrix: M = (dR/dw_curr)^T
    # We need to solve M * x = rhs.
    # Since n_w is small (2-10), we can compute Jacobian explicitly.
    jac_fn = jax.jacfwd(lambda wc: local_step_residual(w_next, wc, t_next, t_curr, funcs, dims, p))
    J_curr = jac_fn(w_curr) # This is dR/dw_curr
    
    # 3. Solve Linear System
    # J_curr.T * lambda_curr = rhs
    # Note: For Trapezoidal, J_curr is roughly (I + h/2 J_f). It is invertible for small h.
    lam_curr = jnp.linalg.solve(J_curr.T, rhs)
    
    return lam_curr

# ==========================================
# 3. ROBUST DIRECT SWEEP (Method B)
# ==========================================

def compute_exact_gradient_sweep(W_flat, p_opt, structure, funcs, grid_taus, dL_dW, dL_dp):
    f_fn, g_fn, _, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_p = dims
    n_w = n_x + n_z
    
    # 1. Precompute Timings & Indices
    ev_times = []
    curr = 0
    for k, c, *extra in structure:
        l = extra[0] if k=='segment' else c
        if k=='event_time': ev_times.append(W_flat[curr])
        curr += l
        
    block_info = []
    curr = 0
    for k, c, *extra in structure:
        l = extra[0] if k=='segment' else c
        block_info.append({'idx':(curr, curr+l), 'kind':k, 'count':c})
        curr += l
        
    total_grad_p = dL_dp.copy()
    
    # We carry the 'Adjoint State' backwards.
    # This represents the accumulated sensitivity of the Loss w.r.t the "Initial Condition" of the block we just processed.
    # When we move to the previous block, this becomes the "Final Condition" load.
    lambda_carry = jnp.zeros(n_w) 
    
    ev_ptr = len(ev_times) - 1
    seg_ptr = len(grid_taus) - 1
    t_final = 2.0
    
    # 2. REVERSE LOOP
    for i in range(len(block_info)-1, -1, -1):
        blk = block_info[i]
        start, end = blk['idx']
        w_block = W_flat[start:end]
        dL_block = dL_dW[start:end]
        
        if blk['kind'] == 'segment':
            n_pts = blk['count']
            w_nodes = w_block.reshape((n_pts, n_w))
            taus = grid_taus[seg_ptr]
            dL_nodes = dL_block.reshape((n_pts, n_w))
            
            # Times
            t_s = ev_times[ev_ptr] if i > 0 else 0.0
            t_e = ev_times[ev_ptr+1] if ev_ptr+1 < len(ev_times) else t_final
            ts = t_s + taus * (t_e - t_s)
            
            # --- SEGMENT SWEEP ---
            # Initial Load at N-1: dL/dx_{N-1} + lambda_carry
            # (lambda_carry comes from Event or End of time)
            curr_lam = dL_nodes[-1] + lambda_carry
            
            # We iterate k from N-2 down to 0
            # solving for lambda associated with equation k
            
            # SCAN BODY
            # Inputs: (w_{k+1}, w_k, t_{k+1}, t_k, dL_k)
            # Carry: lam_{k+1} (Multiplier for Eq k+1? No.)
            # Let's clarify: 
            # We define `solve_local_adjoint_step` to take "Load from Future" and return "Load for Current".
            # The function returns lambda_curr (sensitivity of w_curr).
            
            def scan_body(lam_future, inputs):
                w_nxt, w_cur, t_nxt, t_cur, dL_cur = inputs
                
                # 1. Solve for Sensitivity of w_cur
                lam_at_cur = solve_local_adjoint_step(w_nxt, w_cur, t_nxt, t_cur, lam_future, dL_cur, funcs, dims, p_opt)
                
                # 2. Accumulate Parameter Gradient
                # Term: lam_at_cur^T * dR/dp
                # But wait, `lam_at_cur` computed above is the SENSITIVITY of the state w_cur.
                # It is NOT the Lagrange Multiplier of the residual equation.
                # The Linear System J^T * lam = RHS solved for the Lagrange Multiplier.
                # In `solve_local_adjoint_step`, `lam_curr` IS the multiplier for the residual equation connecting k and k+1.
                
                # Let's verify `solve_local_adjoint_step`:
                # It solves J_curr.T * mu = RHS. 
                # Yes, 'mu' is the Lagrange Multiplier for the step residual.
                
                mu_step = lam_at_cur
                
                # Gradient accumulation
                def R_p(p):
                    return local_step_residual(w_nxt, w_cur, t_nxt, t_cur, funcs, dims, p)
                _, vjp_p = jax.vjp(R_p, p_opt)
                grad_p_local = vjp_p(mu_step)[0]
                
                # We need to pass the SENSITIVITY of w_cur to the next step.
                # Sensitivity(w_cur) = dL/dw_cur + mu_step^T * dR/dw_cur.
                # Wait, `solve_local_adjoint_step` solved J^T * mu = -TotalLoad.
                # So TotalLoad + J^T * mu = 0.
                # This ensures stationarity.
                # But we need the Load for the *next* backward step (which is k-1).
                # That Load depends on how w_cur appears in equation k-1.
                # Actually, the loop structure is simpler:
                # We carry the "Multiplier mu" of the previous equation? No.
                # We carry the "Total Sensitivity of w".
                
                # RE-DESIGN SCAN FOR EXACTNESS:
                # We solve for mu_k (multiplier for Step k: w_k -> w_{k+1}).
                # Stationarity at w_{k+1}: dL/dw_{k+1} + mu_k^T * dR_k/dw_{k+1} + (Contribution from future) = 0.
                # Contribution from future is already calculated.
                # We solve: mu_k^T * dR_k/dw_{k+1} = - (dL/dw_{k+1} + FutureLoad).
                
                # Inputs from scan: w_{k+1}, w_k, Load_{k+1}.
                
                # Solve for mu_k
                _, vjp_nxt = jax.vjp(lambda w: local_step_residual(w, w_cur, t_nxt, t_cur, funcs, dims, p_opt), w_nxt)
                # dR_k/dw_{k+1}^T * mu_k
                
                # Linear solve: (dR/dw_{k+1})^T * mu_k = -Load_{k+1}
                jac_nxt = jax.jacfwd(lambda w: local_step_residual(w, w_cur, t_nxt, t_cur, funcs, dims, p_opt))(w_nxt)
                mu_k = jnp.linalg.solve(jac_nxt.T, -lam_future)
                
                # Now compute Load_k (Sensitivity at w_k)
                # Load_k = dL/dw_k + mu_k^T * dR_k/dw_k
                _, vjp_cur = jax.vjp(lambda w: local_step_residual(w_nxt, w, t_nxt, t_cur, funcs, dims, p_opt), w_cur)
                load_k = dL_cur + vjp_cur(mu_k)[0]
                
                # Param grad
                def Rp(p): return local_step_residual(w_nxt, w_cur, t_nxt, t_cur, funcs, dims, p)
                gp = jax.vjp(Rp, p_opt)[1](mu_k)[0]
                
                return load_k, gp

            # Prepare Inputs
            ws_nxt = w_nodes[1:][::-1]
            ws_cur = w_nodes[:-1][::-1]
            ts_nxt = ts[1:][::-1]
            ts_cur = ts[:-1][::-1]
            dLs = dL_nodes[:-1][::-1] # dL at k
            
            # Initial Load (at N-1)
            init_load = dL_nodes[-1] + lambda_carry
            
            # Run Scan
            final_load, grads = jax.lax.scan(scan_body, init_load, (ws_nxt, ws_cur, ts_nxt, ts_cur, dLs))
            
            total_grad_p += jnp.sum(grads, axis=0)
            
            # The result `final_load` is the sensitivity at w_0.
            # If bi==0 (first segment), we solve IC constraint.
            if i == 0:
                # IC: w_0 - w_start = 0.
                # mu_ic^T * I = -final_load => mu_ic = -final_load.
                pass
            else:
                # Pass to previous Event
                lambda_carry = final_load
                
            seg_ptr -= 1
            
        elif blk['kind'] == 'event_time':
            t_e = w_block[0]

            # Get neighbor states
            w_prev_end = W_flat[start-n_w : start]
            w_post_start = W_flat[end : end+n_w]

            # lambda_carry holds Load on x_post
            load_post = lambda_carry

            # Event Residuals: R(t_e, x_post, x_prev) = 0
            # [Guard, Reset, Continuity]
            def event_res(te, xp, xpr, p):
                x_p, z_p = xp[:n_x], xp[n_x:]
                x_r, z_r = xpr[:n_x], xpr[n_x:]

                res = []
                res.append(guard_fn(te, x_r, z_r, p)[0:1]) # Scalar guard
                res.append(reinit_res_fn(te, x_p, z_p, x_r, z_r, p))

                # Continuity
                cont = []
                for k in range(n_x):
                    is_reinit = any(True for (t, idx) in reinit_vars if t == 'state' and idx == k)
                    if not is_reinit: cont.append(x_p[k:k+1] - x_r[k:k+1])
                if cont: res.append(jnp.concatenate(cont))

                # Alg at event?
                if n_z > 0: res.append(g_fn(te, x_p, z_p, p))

                return jnp.concatenate(res)

            # Solve for mu_event using stationarity at BOTH x_post AND x_prev
            # - At x_post: (dR/dx_post)^T @ mu = -load_post  (n_w equations)
            # - At x_prev: (dR/dx_prev)^T @ mu = -dL/dx_prev (n_w equations)
            # Combined: [J_xp^T; J_xpr^T] @ mu = [-load_post; -dL_xprev]
            # This is 2*n_w equations for n_event unknowns

            J_xp = jax.jacfwd(lambda x: event_res(t_e, x, w_prev_end, p_opt))(w_post_start)  # (n_event, n_w)
            J_xpr = jax.jacfwd(lambda x: event_res(t_e, w_post_start, x, p_opt))(w_prev_end)  # (n_event, n_w)

            # Stack transposed Jacobians: [J_xp^T; J_xpr^T] is (2*n_w, n_event)
            J_combined = jnp.vstack([J_xp.T, J_xpr.T])  # (2*n_w, n_event)

            # RHS: dL/dw_prev is the direct loss at w_prev (segment will add dyn contribution)
            dL_w_prev = dL_dW[start - n_w : start]
            rhs_combined = jnp.concatenate([-load_post, -dL_w_prev])  # (2*n_w,)

            # Solve: J_combined @ mu = rhs_combined
            # J_combined is (2*n_w, n_event) = (4, 3), rhs is (4,), mu is (3,)
            # Overdetermined system - use lstsq
            mu_event = jnp.linalg.lstsq(J_combined, rhs_combined, rcond=None)[0]

            # 2. Param Grad
            _, vjp_pe = jax.vjp(lambda p: event_res(t_e, w_post_start, w_prev_end, p), p_opt)
            total_grad_p += vjp_pe(mu_event)[0]

            # 3. Compute Load on x_prev
            # Load_prev = mu^T @ dR/dx_prev (no direct dL term - that's in the segment's dL_nodes)
            _, vjp_xpr = jax.vjp(lambda x: event_res(t_e, w_post_start, x, p_opt), w_prev_end)
            load_prev = vjp_xpr(mu_event)[0]

            lambda_carry = load_prev
            ev_ptr -= 1
            
    return total_grad_p

# ==========================================
# 4. CLI Verification
# ==========================================

def verify_sensitivity_cli():
    print("--- Exact Gradient Verification ---")
    dae_data, solver_cfg = load_system('config/config_bouncing_ball.yaml')
    solver = DAESolver(dae_data, verbose=False)
    
    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    sol = solver.solve_augmented(t_span, ncp=15)
    
    W_flat, structure, grid_taus = pack_solution(sol, dae_data)
    funcs = create_jax_functions(dae_data)
    dims = funcs[6]
    n_w = dims[0]+dims[1]
    
    p_opt = jnp.array([p['value'] for p in dae_data['parameters']])
    target_times, target_data = prepare_loss_targets(sol, dae_data['states'], *t_span)
    
    def loss_function(W, p):
        segs_t, segs_x, segs_z, ev_tau = unpack_structure(W, structure, (dims[0], dims[1], n_w), grid_taus)
        y_pred = predict_trajectory_sigmoid(segs_t, segs_x, segs_z, ev_tau, target_times)
        return jnp.mean((y_pred - target_data)**2)

    print(f"Generating solution (ncp={15})...")
    
    # 1. DENSE REFERENCE
    print("\nMethod A: Dense Matrix Solve...")
    t0 = time.time()
    # Compute full Jacobian using implicit differentiation on the loss
    # dJ/dp = dL/dp - dL/dW * (dR/dW)^-1 * dR/dp
    
    # Helper to define Total Residual
    def total_residual(W, p):
        return unpack_and_compute_residual(W, p, dae_data, structure, funcs, (p, list(range(len(p)))), grid_taus)
        
    dR_dW = jax.jacfwd(lambda w: total_residual(w, p_opt))(W_flat)
    dR_dp = jax.jacfwd(lambda p: total_residual(W_flat, p))(p_opt)
    dL_dW = jax.grad(loss_function, 0)(W_flat, p_opt)
    dL_dp = jax.grad(loss_function, 1)(W_flat, p_opt)
    
    lambda_dense = jnp.linalg.solve(dR_dW.T, -dL_dW)
    grad_dense = dL_dp + jnp.dot(lambda_dense, dR_dp)
    print(f"  Time: {time.time()-t0:.4f}s")
    print(f"  Result: {grad_dense}")
    
    # 2. SWEEP
    print("\nMethod B: Exact Direct Adjoint Sweep (User Implementation)...")
    t0 = time.time()
    grad_sweep = compute_exact_gradient_sweep(W_flat, p_opt, structure, funcs, grid_taus, dL_dW, dL_dp)
    print(f"  Time: {time.time()-t0:.4f}s")
    print(f"  Result: {grad_sweep}")
    
    print("\n" + "="*40)
    print("FINAL COMPARISON RESULTS")
    print("="*40)
    print(f"{'Parameter':<10} | {'Dense (Method A)':<20} | {'Sweep (Method B)':<20} | {'Diff':<10}")
    print("-" * 80)
    param_names = [p['name'] for p in dae_data['parameters']]
    for i, name in enumerate(param_names):
        print(f"{name:<10} | {grad_dense[i]:<20.8e} | {grad_sweep[i]:<20.8e} | {abs(grad_dense[i]-grad_sweep[i]):.2e}")

    print("-" * 80)
    print(f"Total Gradient Difference Norm: {jnp.linalg.norm(grad_dense - grad_sweep):.4e}")
    
    # Check Initial Residual
    r0 = total_residual(W_flat, p_opt)
    print(f"Residual Norm at W0: {jnp.linalg.norm(r0):.2e}")

if __name__ == "__main__":
    verify_sensitivity_cli()