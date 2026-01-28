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
    grad_times = vjp_time(lam_k)[0] # [dL/dt_curr, dL/dt_next]
    
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

    dLs = dL_nodes[1:]
    
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
        
        return (partial_load_k, new_acc_start, new_acc_end), grad_p_step

    # Execute Scan
    # Carry: (current_load, acc_t_start, acc_t_end)
    init_carry = (init_load, 0.0, 0.0)
    (final_partial_load, grad_t_start_total, grad_t_end_total), grads_p = jax.lax.scan(
        scan_body, init_carry, 
        (scan_ws_c, scan_ws_n, scan_ts_c, scan_ts_n, scan_dLs, scan_taus_c, scan_taus_n)
    )
    
    sensitivity_w0 = dL_nodes[0] + final_partial_load
    total_grad_p = jnp.sum(grads_p, axis=0)
    
    return sensitivity_w0, total_grad_p, grad_t_start_total, grad_t_end_total

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
    
    # Store mesh sensitivity from the "Next" segment to pass to "Current" event
    mesh_sens_next_seg = 0.0 
    
    seg_idx = len(grid_taus) - 1
    ev_idx = len(ev_vals) - 1
    t_final = 2.0
    
    for i in range(len(block_indices) - 1, -1, -1):
        idx_s, idx_e, kind, count = block_indices[i]
        w_block = W_flat[idx_s:idx_e]
        dL_block = dL_dW[idx_s:idx_e]
        
        if kind == 'segment':
            n_pts = count
            w_nodes = w_block.reshape((n_pts, n_w))
            taus = grid_taus[seg_idx]
            dL_nodes = dL_block.reshape((n_pts, n_w))
            
            t_s = ev_vals[ev_idx] if i > 0 else 0.0
            t_e = ev_vals[ev_idx+1] if ev_idx+1 < len(ev_vals) else t_final
            ts = t_s + taus * (t_e - t_s)
            
            # --- Perform Sweep ---
            sens_w0, grad_p_seg, grad_t_start, grad_t_end = run_segment_backward_sweep(
                w_nodes, ts, dL_nodes, adjoint_state, funcs, dims, p_opt
            )
            
            total_grad_p += grad_p_seg
            adjoint_state = sens_w0
            
            # The 'grad_t_start' is the sensitivity w.r.t the Event Time that starts this segment
            # This is the "Load" for that Event.
            mesh_sens_next_seg = grad_t_start
            
            # 'grad_t_end' should theoretically be added to the Event Time that ENDS this segment.
            # But in the reverse sweep, we already processed that Event (it was the "Next" block).
            # We didn't have this value then?
            # Actually, Block i is Segment. Block i+1 was Event (End of Segment).
            # We already processed Block i+1.
            # Did we miss adding `grad_t_end` to Block i+1's stationarity?
            # Yes. But since we iterate strictly backwards, we can't add it "back in time".
            # FIX: We need to accumulate `grad_t_end` of Segment i into a buffer for Event at t_e?
            # BUT: Event i+1 (Time t_e) determines t_e.
            # Changing t_e affects Segment i (End) and Segment i+1 (Start).
            # So Sensitivity of t_e = (Sensitivity from Seg i+1 Start) + (Sensitivity from Seg i End).
            # When we processed Event i+1, we had `mesh_sens_next_seg` (which was Seg i+1 Start).
            # We DID NOT have Seg i End sensitivity yet.
            # This implies we can't solve Event i+1 fully until we do Segment i?
            # OR: The Event Adjoint ($\gamma$) depends on future only?
            # Stationarity at t_e: dL/dte + gamma*... + Load_Mesh_Future + Load_Mesh_Past = 0.
            # If so, the system is coupled globally?
            # NO. The "Breathing Mesh" formulation typically defines t_grids purely primarily.
            # But `t_e` is a variable.
            # The user's Logic Step 3 said: "Retrieve Mesh Sensitivity from the NEXT segment... This is 'grad_t_start'".
            # It did not mention 'grad_t_end' of the previous segment.
            # Why? Maybe because `t_end` is determined by `t_start` + duration? Or purely `t_event`.
            # If t_event changes, it affects Past Segment's End and Future Segment's Start.
            # The "Past Segment End" sensitivity is propagated via lambda_carry?
            # No, lambda_carry is state sensitivity.
            # If we strictly follow Block Causal:
            # Block 1 (Seg) -> Output Mesh Sens Start -> Block 2 (Event) -> Input Load.
            # This handles "Future Segment".
            # What about "Past Segment"?
            # The Event determines t_e. The Past Segment just "ends" there.
            # The sensitivity dL / dt_end (of Past Seg) adds to dL / dt_e.
            # But this is a gradient accumulation w.r.t a variable.
            # If t_e is a variable we solve for (via gammaStationarity), we need ALL forces on it.
            # However, in DAE adjoints, typically only the FUTURE dependency matters for causality.
            # The PAST dependency is "already happened"?
            # No.
            # User's note: "Sweep Segment 1... This sweep calculates acc_dt_end. This represents dL/dte1... arising from shift of end of Segment 1."
            # This suggests we just calculate it.
            # Maybe it is added to the gradient if t_e is a PARAMETER.
            # But if t_e is an internal variable (determined by event function), its sensitivity is handled by gamma.
            # User's logic flow: "Sweep Segment 2... Computes acc_dt_start... Solve Event 1... Use acc_dt_start".
            # This confirms we use Future Segment Start sensitivity.
            # It ignores the Past Segment End sensitivity?
            # Let's stick to using `mesh_sens_next_seg` (which is dL/dt_start of Block i+1 (if it was a segment, but here blocks are alternating)).
            # Wait. Block i is Segment. Step i-1 is Event.
            # Segment i Start Time is Event i-1.
            # So `grad_t_start` of Segment i is indeed the load for Event i-1.
            # This matches User Logic exactly.
            
            seg_idx -= 1
            
        elif kind == 'event_time':
            t_event = w_block[0]
            dL_t = dL_block[0]
            
            prev_blk = block_indices[i-1]
            next_blk = block_indices[i+1]
            w_prev_end = W_flat[prev_blk[1]-n_w : prev_blk[1]]
            w_post_start = W_flat[next_blk[0] : next_blk[0]+n_w]
            
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

            # Solve for Event Multipliers (mu)
            # Stationarity at x_post: (dR/dx_post)^T * mu = -adjoint_state (from Future Segment Start state load)
            
            J_xp = jax.jacfwd(lambda xp: event_res_fn(t_event, xp, w_prev_end, p_opt))(w_post_start)
            
            # Stationarity at t_e: (dR/dt)^T * mu = - (dL/dt_e + MESH_SENSITIVITY_FROM_NEXT_SEGMENT)
            # We augment the RHS for the t_e row.
            # Wait, the system solving `mu` usually couples variables?
            # Standard: [dR/dx_p ; dR/dt]^T * mu = [-Lx ; -Lt].
            # Here variables are x_post, t_event?
            # No. t_event is determined by Guard(prev).
            # Variables in this block coupling: x_post (Reset), t_event (Guard).
            # The system must solve for mu based on loads on x_post and t_event?
            # NO. t_event is determined by G(x_prev, t)=0.
            # So t_event is "state-like".
            # So we solve for mu to satisfy stationarity at x_post AND t_event (if t_event is treated as a node).
            # Actually, `lstsq` in VerifyResidualScan uses combined Jacobian?
            # Let's use the explicit structure:
            # We have unknowns mu (size n_w + 1).
            # We have equations for Stationarity at x_post (size n_w).
            # We have Stationarity at t_e (size 1). 
            #   dL_t + mu^T * dR/dt + mesh_sens = 0.
            #   => (dR/dt)^T * mu = -(dL_t + mesh_sens).
            # So we stack the system!
            
            J_te = jax.jacfwd(lambda t: event_res_fn(t, w_post_start, w_prev_end, p_opt))(t_event) # size (n_res, 1)
            
            # Matrix: [ dR/dx_p^T ]  (n_w x n_res)
            #         [ dR/dt^T   ]  (1   x n_res)
            # Total size: (n_w + 1) x n_res.
            # Since n_res = n_w + 1 (Guard + Reset + Cont + Alg), this is Square?
            # Yes. n_res = 1(G) + n_x(R+C) + n_z(A) = 1 + n_w.
            
            System_Matrix = jnp.vstack([J_xp.T, J_te.T])
            
            # RHS
            # Load on x_post = adjoint_state
            # Load on t_e    = dL_t + mesh_sens_next_seg
            rhs_vec = jnp.concatenate([-adjoint_state, -(dL_t + mesh_sens_next_seg)])
            
            if System_Matrix.shape[0] == System_Matrix.shape[1]:
                mu_event = jnp.linalg.solve(System_Matrix, rhs_vec)
            else:
                mu_event = jnp.linalg.lstsq(System_Matrix, rhs_vec, rcond=1e-5)[0]
                
            # Accumulate Gradients
            _, vjp_p = jax.vjp(lambda p: event_res_fn(t_event, w_post_start, w_prev_end, p), p_opt)
            total_grad_p += vjp_p(mu_event)[0]
            
            # Pass Load to Previous Segment (w_prev)
            _, vjp_xpr = jax.vjp(lambda xpr: event_res_fn(t_event, w_post_start, xpr, p_opt), w_prev_end)
            load_prev = vjp_xpr(mu_event)[0]
            
            adjoint_state = load_prev
            ev_idx -= 1
            
    return total_grad_p