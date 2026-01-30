import jax
import jax.numpy as jnp
from jax import config
from functools import partial
import numpy as np
import time

config.update("jax_enable_x64", True)

# [Assuming Loader, Helper, and Prediction functions from your provided code are present here]
# ... (Load_system, create_jax_functions, pack/unpack, etc. same as your snippet) ...

# =========================================================
# 3. ROBUST BACKWARD SWEEP (Block-Triangular Solver)
# =========================================================

# =========================================================
# 3. ROBUST BACKWARD SWEEP (Block-Triangular Solver)
# =========================================================

def solve_local_step_adjoint(lam_next, w_curr, w_next, t_curr, t_next, dL_curr, funcs, dims, p):
    """
    Solves for lambda_curr (multiplier for step k) given lambda_next (multiplier for step k+1).
    AND computes:
      1. Sensitivity w.r.t the state x_k (Load for previous step).
      2. Sensitivity w.r.t time boundaries (t_curr, t_next).
    """
    n_x, n_z, _ = dims
    n_w = n_x + n_z
    
    # 1. Define Local Residual R_k(w_curr, w_next)
    def res_k_fn(wc, wn):
        h = t_next - t_curr
        xc, zc = wc[:n_x], wc[n_x:]
        xn, zn = wn[:n_x], wn[n_x:]
        
        f_c = funcs[0](t_curr, xc, zc, p)
        f_n = funcs[0](t_next, xn, zn, p)
        res_dyn = -xn + xc + (h/2.0)*(f_c + f_n)
        
        res_alg = funcs[1](t_curr, xc, zc, p) if n_z > 0 else jnp.array([])
        
        return jnp.concatenate([res_dyn, res_alg])

    # 2. RHS of the linear system
    rhs = -dL_curr 
    
    # 3. Build Matrix A = dR_k / dw_{next}
    jac_wn_fn = jax.jacfwd(lambda wn: res_k_fn(w_curr, wn))
    A_T = jac_wn_fn(w_next).T
    
    # 4. Linear Solve
    lam_k = jnp.linalg.solve(A_T, rhs)
    
    # 5. Compute Load for Previous Step (Sensitivity w.r.t w_curr)
    _, vjp_wc = jax.vjp(lambda wc: res_k_fn(wc, w_next), w_curr)
    partial_load_k = vjp_wc(lam_k)[0]
    
    # 6. Parameter Gradient Contribution
    def res_p_fn(p_arg):
        h = t_next - t_curr
        xc, zc = w_curr[:n_x], w_curr[n_x:]
        xn, zn = w_next[:n_x], w_next[n_x:]
        f_c = funcs[0](t_curr, xc, zc, p_arg)
        f_n = funcs[0](t_next, xn, zn, p_arg)
        r_d = -xn + xc + (h/2.0)*(f_c + f_n)
        r_a = funcs[1](t_curr, xc, zc, p_arg) if n_z > 0 else jnp.array([])
        return jnp.concatenate([r_d, r_a])
        
    _, vjp_p = jax.vjp(res_p_fn, p)
    grad_p_contrib = vjp_p(lam_k)[0]

    # 7. Time Gradient Contribution (Breathing Mesh)
    def res_t_fn(times):
        tc, tn = times[0], times[1]
        h_local = tn - tc
        
        fx, zx = w_curr[:n_x], w_curr[n_x:]
        nx, nz = w_next[:n_x], w_next[n_x:]
        
        fc = funcs[0](tc, fx, zx, p)
        fn = funcs[0](tn, nx, nz, p)
        
        r_d = -nx + fx + (h_local/2.0)*(fc + fn)
        r_a = funcs[1](tc, fx, zx, p) if n_z > 0 else jnp.array([])
        return jnp.concatenate([r_d, r_a])

    _, vjp_t = jax.vjp(res_t_fn, jnp.array([t_curr, t_next]))
    grad_times = vjp_t(lam_k)[0] # [dL/dt_curr, dL/dt_next]
    
    return lam_k, partial_load_k, grad_p_contrib, grad_times

@partial(jax.jit, static_argnames=['funcs', 'dims'])
def run_segment_backward_sweep(w_nodes, ts, dL_nodes, load_at_end, funcs, dims, p):
    """
    Scans backwards through a segment.
    """
    ws_curr = w_nodes[:-1]
    ws_next = w_nodes[1:]
    ts_curr = ts[:-1]
    ts_next = ts[1:]
    taus_curr = (ts_curr - ts[0]) / (ts[-1] - ts[0])
    taus_next = (ts_next - ts[0]) / (ts[-1] - ts[0])

    dLs = dL_nodes[:-1]
    
    scan_ws_c = ws_curr[::-1]
    scan_ws_n = ws_next[::-1]
    scan_ts_c = ts_curr[::-1]
    scan_ts_n = ts_next[::-1]
    scan_taus_c = taus_curr[::-1]
    scan_taus_n = taus_next[::-1]
    scan_dLs  = dLs[::-1]
    
    init_load = dL_nodes[-1] + load_at_end
    

    
    # Helper to capture updated solve_local
    def solve_helper(load_future, w_c, w_n, t_c, t_n):
        # We need to call the updated solve_local_step_adjoint
        # But wait, it uses 'vjp_time' inside (typo in my previous reflection, fixed in impl below).
        # Need to ensure correct scope.
        return solve_local_step_adjoint(None, w_c, w_n, t_c, t_n, load_future, funcs, dims, p)

    def scan_body(carry, inputs):
        load_future, acc_t_start, acc_t_end = carry
        w_c, w_n, t_c, t_n, dL_at_n, tau_c, tau_n = inputs
        
        # Call Local Step
        # RE-IMPLEMENTING solve_local_step_adjoint INLINE TO AVOID SCOPE/CLOSURE ISSUES WITH VJP
        n_x, n_z, _ = dims
        
        def res_k_fn_closure(wc_in, wn_in):
            h = t_n - t_c
            xc, zc = wc_in[:n_x], wc_in[n_x:]
            xn, zn = wn_in[:n_x], wn_in[n_x:]
            f_c = funcs[0](t_c, xc, zc, p)
            f_n = funcs[0](t_n, xn, zn, p)
            res_dyn = -xn + xc + (h/2.0)*(f_c + f_n)
            res_alg = funcs[1](t_c, xc, zc, p) if n_z > 0 else jnp.array([])
            return jnp.concatenate([res_dyn, res_alg])

        rhs = -load_future 
        jac_wn = jax.jacfwd(lambda wn: res_k_fn_closure(w_c, wn))(w_n)
        lam_k = jnp.linalg.solve(jac_wn.T, rhs)
        
        _, vjp_wc = jax.vjp(lambda wc: res_k_fn_closure(wc, w_n), w_c)
        partial_load_k = vjp_wc(lam_k)[0]
        
        # Param Grad
        def res_p_fn_closure(p_arg):
            h = t_n - t_c
            xc, zc = w_c[:n_x], w_c[n_x:]
            xn, zn = w_n[:n_x], w_n[n_x:]
            f_c = funcs[0](t_c, xc, zc, p_arg)
            f_n = funcs[0](t_n, xn, zn, p_arg)
            r_d = -xn + xc + (h/2.0)*(f_c + f_n)
            r_a = funcs[1](t_c, xc, zc, p_arg) if n_z > 0 else jnp.array([])
            return jnp.concatenate([r_d, r_a])
        _, vjp_p = jax.vjp(res_p_fn_closure, p)
        grad_p_step = vjp_p(lam_k)[0]
        
        # Time Gradient
        def res_t_fn_closure(times):
            tc_in, tn_in = times[0], times[1]
            h_local = tn_in - tc_in
            xc, zc = w_c[:n_x], w_c[n_x:]
            xn, zn = w_n[:n_x], w_n[n_x:]
            f_c = funcs[0](tc_in, xc, zc, p)
            f_n = funcs[0](tn_in, xn, zn, p)
            r_d = -xn + xc + (h_local/2.0)*(f_c + f_n)
            r_a = funcs[1](tc_in, xc, zc, p) if n_z > 0 else jnp.array([])
            return jnp.concatenate([r_d, r_a])
            
        _, vjp_t = jax.vjp(res_t_fn_closure, jnp.array([t_c, t_n]))
        grad_times = vjp_t(lam_k)[0]
        
        # Accumulate Time Gradients (Chain Rule)
        # t_k = t_start + tau_k * (t_end - t_start)
        d_tc, d_tn = grad_times[0], grad_times[1]
        
        dt_start_local = d_tc * (1.0 - tau_c) + d_tn * (1.0 - tau_n)
        dt_end_local   = d_tc * (tau_c)       + d_tn * (tau_n)
        
        new_acc_start = acc_t_start + dt_start_local
        new_acc_end   = acc_t_end   + dt_end_local
        
        # Accumulate dL/dNode for the next step's load
        next_load = partial_load_k + dL_at_n

        return (next_load, new_acc_start, new_acc_end), grad_p_step

    # Execute Scan
    # Carry: (current_load, acc_t_start, acc_t_end)
    init_carry = (init_load, 0.0, 0.0)
    
    # Execute Scan
    # Carry: (current_load, acc_t_start, acc_t_end)
    init_carry = (init_load, 0.0, 0.0)
    
    (final_partial_load, grad_t_start_total, grad_t_end_total), grads_p = jax.lax.scan(
        scan_body, init_carry, 
        (scan_ws_c, scan_ws_n, scan_ts_c, scan_ts_n, scan_dLs, scan_taus_c, scan_taus_n)
    )
    
    total_gp = jnp.sum(grads_p, axis=0)
    sens_w0 = final_partial_load
    

    
    return sens_w0, total_gp, grad_t_start_total, grad_t_end_total



@partial(jax.jit, static_argnames=['funcs', 'dims'])
def solve_event_system(t_event, w_prev_end, w_post_start, adjoint_state, dL_t, mesh_sens_next_seg, funcs, dims, p_opt):
    """
    Solves the event system logic including finding the null-space direction and particular solution.
    Returns: mu_0, v_null, terms_for_c_equation
    """
    f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, _ = funcs
    n_x, n_z, n_p = dims
    n_w = n_x + n_z

    def event_res_fn(t, xp, xpr, p):
        x_p, z_p = xp[:n_x], xp[n_x:]
        x_r, z_r = xpr[:n_x], xpr[n_x:]
        r_g = guard_fn(t, x_r, z_r, p)[0:1]
        r_r = reinit_res_fn(t, x_p, z_p, x_r, z_r, p)
        r_c = []
        for k in range(n_x):
            is_r = False
            for (type_, idx_) in reinit_vars:
                    if type_=='state' and idx_==k: is_r=True
            if not is_r: r_c.append(x_p[k:k+1] - x_r[k:k+1])
        r_a = g_fn(t, x_p, z_p, p) if n_z > 0 else jnp.array([])
        return jnp.concatenate([r_g, r_r] + r_c + [r_a])

    # 1. Jacobians
    J_xp = jax.jacfwd(lambda xp: event_res_fn(t_event, xp, w_prev_end, p_opt))(w_post_start)
    J_t_partial = jax.jacfwd(lambda t: event_res_fn(t, w_post_start, w_prev_end, p_opt))(t_event)
    J_xpr = jax.jacfwd(lambda xpr: event_res_fn(t_event, w_post_start, xpr, p_opt))(w_prev_end)

    # w_dot removed: In discrete adjoint, x_pre dependence on t_e is handled 
    # by the Segment Residuals (grad_t_end_prev).
    # The Event Residual R(t, x_post, x_pre) partial w.r.t t should NOT include dx_pre/dt.
    
    J_te_total = J_t_partial.reshape((-1, 1))

    # 2. Decompose J_xp^T to find mu_0 and v_null
    # J_xp is (N_res, N_w). N_res = N_x + 1 + N_z (approx). N_w = N_x + N_z.
    # Usually N_res = N_w + 1.
    matrix_for_mu = J_xp.T # (N_w, N_res)
    # We want mu such that matrix_for_mu @ mu = -adjoint_state
    
    # QR Decomposition of J_xp (N_res, N_w)
    # Q: (N_res, N_res). R: (N_res, N_w).
    # J_xp = Q @ R
    # J_xp^T = R^T @ Q^T
    # R^T @ (Q^T mu) = -adjoint_state
    # Let y = Q^T mu. R^T y = -adjoint_state.
    # R^T is (N_w, N_res).
    # Since N_res > N_w, R looks like [U; 0].
    # Wait, J_xp usually has full rank N_w.
    # Q, R = jax.scipy.linalg.qr(J_xp, mode='full')
    # Use standard lstsq to find mu_0
    mu_0, residuals, rank, s = jnp.linalg.lstsq(matrix_for_mu, -adjoint_state, rcond=1e-9)
    
    # Find Null Space (Kernel of J_xp^T = Left Kernel of J_xp)
    # J_xp^T v = 0.
    # This corresponds to the last column of Q in QR of J_xp.
    Q, _ = jax.scipy.linalg.qr(J_xp, mode='full')
    v_null = Q[:, -1] # Last column
    
    # Equation for c:
    # Stationarity t_e: dL/dte + mesh_sens_next + mu^T J_te_total + grad_t_end_prev(mu) = 0
    # mu = mu_0 + c * v_null
    # grad_t_end_prev depends on Load = J_xpr^T mu
    
    # Precompute terms for scalar equation (excluding grad_t_end_prev)
    t_e_rhs_base = dL_t + mesh_sens_next_seg + jnp.dot(mu_0, J_te_total.flatten())
    t_e_slope    = jnp.dot(v_null, J_te_total.flatten())
    
    # Compute Loads for Previous Segment
    # _, vjp_xpr = jax.vjp(lambda xpr: event_res_fn(t_event, w_post_start, xpr, p_opt), w_prev_end)
    # load_prev_mu0 = vjp_xpr(mu_0)[0]
    # load_prev_v   = vjp_xpr(v_null)[0]
    # Manual VJP using J_xpr
    load_prev_mu0 = J_xpr.T @ mu_0
    load_prev_v   = J_xpr.T @ v_null
    
    # Gradients w.r.t p from Event
    _, vjp_p = jax.vjp(lambda p: event_res_fn(t_event, w_post_start, w_prev_end, p), p_opt)
    gp_mu0 = vjp_p(mu_0)[0]
    gp_v   = vjp_p(v_null)[0]
    
    return load_prev_mu0, load_prev_v, t_e_rhs_base, t_e_slope, gp_mu0, gp_v, mu_0, v_null

def compute_adjoint_sweep_direct(W_flat, p_opt, dae_data, structure, funcs, grid_taus, dL_dW, dL_dp):
    f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_p = dims
    n_w = n_x + n_z
    
    block_indices = []
    curr = 0
    for kind, count, *extra in structure:
        length = extra[0] if kind == 'segment' else count
        block_indices.append((curr, curr + length, kind, count))
        curr += length
        
    ev_vals = [W_flat[b[0]] for b in block_indices if b[2] == 'event_time']
    
    total_grad_p = dL_dp.copy()
    adjoint_state = jnp.zeros(n_w)
    
    mesh_sens_next_seg = 0.0 
    
    seg_idx = len(grid_taus) - 1
    ev_idx = len(ev_vals) - 1
    t_final = 2.0
    
    # Reverse loop logic
    i = len(block_indices) - 1
    
    while i >= 0:
        idx_s, idx_e, kind, count = block_indices[i]
        
        if kind == 'segment':
            # Identify if this is a "Standalone" segment (Last one) or coupled with an Event
            # If it is the very last block, we process it normally.
            # If it is preceded by an Event (which is Block i-1, handled in next iter), 
            # we skip it here? No, the loop decrements.
            # We want to handle [Event i-1, Segment i] together if possible.
            # But the dependencies are: Event i-1 needs Segment i? No.
            # Event i-1 is "Start" of Segment i.
            # "Previous Segment" for Event i-1 is Segment i-2.
            # "Next Segment" for Event i-1 is Segment i.
            # Wait.
            # Event k is between Seg k and Seg k+1.
            # Backward sweep: Seg k+1 -> Event k -> Seg k.
            # Cycle is: Event k needs Seg k. Seg k needs Event k.
            
            # So, when we are at index i (Segment k+1), we just do the sweep.
            # Then we go to i-1 (Event k).
            # We look ahead to i-2 (Segment k). AND PROCESS BOTH.
            
            # Verify Block i is Segment k+1.
            # If i == len-1, it is Seg N. No Event after it.
            # Just do Sweep.
            
            # Basic Sweep for Segment
            w_block = W_flat[idx_s:idx_e]
            dL_block = dL_dW[idx_s:idx_e]
            n_pts = count
            w_nodes = w_block.reshape((n_pts, n_w))
            taus = grid_taus[seg_idx]
            dL_nodes = dL_block.reshape((n_pts, n_w))
            
            t_s = ev_vals[ev_idx] if ev_idx >= 0 else 0.0
            # If this is the last segment, t_e is final.
            # But if we are inside the 'Joint' step (Seg k), t_e is Event k's time.
            # We need to be careful with ev_idx.
            
            # Let's rely on standard logic but detect if we should MERGE.
            # Merge if current is Event.
            
            t_e = ev_vals[ev_idx+1] if ev_idx+1 < len(ev_vals) else t_final
            ts = t_s + taus * (t_e - t_s)
            
            sens_w0, grad_p_seg, grad_t_start, grad_t_end = run_segment_backward_sweep(
                w_nodes, ts, dL_nodes, adjoint_state, funcs, dims, p_opt
            )
            
            total_grad_p += grad_p_seg
            adjoint_state = sens_w0
            mesh_sens_next_seg = grad_t_start
            
            seg_idx -= 1
            i -= 1
            
        elif kind == 'event_time':
            # Event k. Next block (already processed) was Seg k+1.
            # Previous block (i-1) is Seg k.
            # We MUST process Seg k here to resolve cycle.
            
            t_event = W_flat[idx_s]
            dL_t = dL_dW[idx_s]
            
            # Data for Event
            prev_blk = block_indices[i-1] # Seg k
            next_blk = block_indices[i+1] # Seg k+1
            w_prev_end = W_flat[prev_blk[1]-n_w : prev_blk[1]]
            w_post_start = W_flat[next_blk[0] : next_blk[0]+n_w]
            
            # 1. Event Analysis
            l_prev_mu0, l_prev_v, te_rhs, te_slope, gp_mu0, gp_v, mu_0, v_null = solve_event_system(
                t_event, w_prev_end, w_post_start, adjoint_state, dL_t, mesh_sens_next_seg, funcs, dims, p_opt
            )
            
            # Verify Event Multipliers
            # We solve for mu s.t. ... 
            # But we didn't finish solving for mu yet! We only have affine form.
            # We compute c_val later.
            # We should verify mu AFTER c_val is found.
            
            # 2. Setup Segment k (Previous Seg)
            idx_s_prev, idx_e_prev, k_prev, count_prev = prev_blk
            w_block_prev = W_flat[idx_s_prev:idx_e_prev]
            dL_block_prev = dL_dW[idx_s_prev:idx_e_prev]
            n_pts_prev = count_prev
            w_nodes_prev = w_block_prev.reshape((n_pts_prev, n_w))
            taus_prev = grid_taus[seg_idx] # seg_idx was decremented once after Seg k+1
            dL_nodes_prev = dL_block_prev.reshape((n_pts_prev, n_w))
            
            # Times for Seg k
            # End time is t_event. Start time is ev_vals[ev_idx-1] or 0.
            t_e_prev = t_event
            t_s_prev = ev_vals[ev_idx-1] if ev_idx > 0 else 0.0
            ts_prev = t_s_prev + taus_prev * (t_e_prev - t_s_prev)
            
            # 3. Double Sweep (Vectorized)
            # Stack inputs
            # w_nodes, ts, p_opt are same.
            # dL_nodes: [dL_nodes_prev, Zeros]
            # load_at_end: [l_prev_mu0, l_prev_v]
            
            dL_stack = jnp.stack([dL_nodes_prev, jnp.zeros_like(dL_nodes_prev)])
            load_stack = jnp.stack([l_prev_mu0, l_prev_v])
            
            sweep_vmap = jax.vmap(run_segment_backward_sweep, in_axes=(None, None, 0, 0, None, None, None))
            sens_w0_stack, grad_p_stack, grad_ts_stack, grad_te_stack = sweep_vmap(
                w_nodes_prev, ts_prev, dL_stack, load_stack, funcs, dims, p_opt
            )
            
            # Extract
            # 0 -> corresponds to mu0 solution
            # 1 -> corresponds to v_null solution
            grad_te_mu0 = grad_te_stack[0]
            grad_te_v   = grad_te_stack[1]
            
            # 4. Solve for c
            # Equation: te_rhs + te_slope * c + (grad_te_mu0 + grad_te_v * c) = 0
            # c * (te_slope + grad_te_v) = - (te_rhs + grad_te_mu0)
            
            denom = te_slope + grad_te_v
            # Avoid div by zero?
            c_val = -(te_rhs + grad_te_mu0) / (denom + 1e-12)
            
            # 5. Accumulate Results
            # Final Event P grad
            total_grad_p += gp_mu0 + c_val * gp_v
            # Segment P grad
            total_grad_p += grad_p_stack[0] + c_val * grad_p_stack[1]
            
            # Final Adjoint State (Start of Seg k)
            adjoint_state = sens_w0_stack[0] + c_val * sens_w0_stack[1]
            
            # Mesh Sens for next event (Ev k-1)
            # grad_t_start from Seg k
            mesh_sens_next_seg = grad_ts_stack[0] + c_val * grad_ts_stack[1]
            
            # Complete processing
            # We handled Ev i (idx i) and Seg i-1 (idx i-1).
            # Decrement i by 2.
            # Update seg_idx and ev_idx.
            seg_idx -= 1 # Seg k processed
            ev_idx -= 1  # Ev k processed
            i -= 2
            
    return total_grad_p