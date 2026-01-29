
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# Import helper functions from original file if needed, or re-implement
# We need run_segment_backward_sweep primarily.

@partial(jax.jit, static_argnames=['funcs', 'dims', 'max_blocks'])
def compute_adjoint_sweep_padded(
    W_padded, TS_padded, p_opt, 
    # Structure Arrays
    block_types,     # (MAX_BLOCKS,) int: 0=Pad, 1=Seg, 2=Event
    block_indices,   # (MAX_BLOCKS, 2) int: [start, end]
    block_param,     # (MAX_BLOCKS,) float: e.g. t_event for Event, or 0.0
    dL_padded,       # (MAX_W,) float
    dL_dp,           # (n_p,) float
    funcs, 
    dims,
    max_blocks
):
    f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_p = dims

    # Helper for extracting block data properly mapped to JIT static size
    
    # Scan Carry State: ... [Same as before]
    
    init_adjoint = jnp.zeros(n_x + n_z)
    init_mesh_sens = 0.0
    init_grad_p = dL_dp
    
    # Pending Event State Structure: ... [Same]
    n_mu_max = n_x + n_z + 1
    
    init_pending = (
        0.0, # active
        jnp.zeros(n_mu_max), # mu_0 (padded/max size)
        jnp.zeros(n_mu_max), # v_null
        jnp.zeros(n_x + n_z), # load_prev_mu0
        jnp.zeros(n_x + n_z), # load_prev_v
        0.0, # te_rhs
        0.0, # te_slope
        jnp.zeros(n_p), # gp_mu0
        jnp.zeros(n_p)  # gp_v
    )
    
    init_carry = (init_adjoint, init_mesh_sens, init_grad_p, init_pending)
    
    # Scan Indices: Reverse iteration
    scan_indices = jnp.arange(max_blocks)[::-1]
    
    def scan_body(carry, block_idx):
        adj_load, mesh_sens, grad_p, pending = carry
        
        # Unpack Block Info using dynamic slice
        b_type = block_types[block_idx] # 0=Pad, 1=Seg, 2=Event
        # block_indices: [start_idx, length] ?
        # Step 127: n_pts_valid = block_indices[block_idx, 1]
        
        W_block = W_padded[block_idx]
        T_block = TS_padded[block_idx] # New!
        dL_block = dL_padded[block_idx]
        n_pts_valid = block_indices[block_idx, 1] 
        
        # --- Logic Switch ---
        
        # Case 0: Padding / Inactive
        def case_pad(c_args):
            # Identity pass
            return c_args[0], c_args[1], c_args[2], c_args[3]
            
        # Case 1: Segment
        def case_segment(c_args):
            curr_adj, curr_mesh, curr_gp, curr_pend = c_args
            
            # Check if we have a Pending Event (Joint Solve)
            pending_active = curr_pend[0]
            # jax.debug.print("Blk {} Pending {}", block_idx, pending_active)
            
            def branch_joint_solve(args):
                p_adj, p_mesh, p_gp, p_pend = args
                
                # Unpack pending event data
                _, mu_0, v_null, lp_mu0, lp_v, te_rhs, te_slope, gp_mu0, gp_v = p_pend
                
                # 1. Prepare Stacked Sweep Inputs
                # CRITICAL: Order must be [mu_0, v_null] to match how results are used
                dL_zeros = jnp.zeros_like(dL_block)
                dL_stack = jnp.stack([dL_block, dL_zeros])  # [mu_0 has dL, v_null has zeros]
                load_stack = jnp.stack([lp_mu0, lp_v])      # [mu_0 load, v_null load]
                
                # VMAP Sweep
                sweep_vmap = jax.vmap(run_segment_backward_sweep, in_axes=(None, None, 0, 0, None, None, None, None))
                
                # Use extracted T_block
                sens_w0_stack, grad_p_stack, grad_ts_stack, grad_te_stack = sweep_vmap(
                    W_block, T_block, dL_stack, load_stack, n_pts_valid, funcs, dims, p_opt
                )
                
                # 2. Solve for c
                grad_te_mu0 = grad_te_stack[0]
                grad_te_v   = grad_te_stack[1]
                
                denom = te_slope + grad_te_v
                # Regularize (same as v2)
                c_val = -(te_rhs + grad_te_mu0) / (denom + 1e-12)
                
                # 3. Combine
                total_adj = sens_w0_stack[0] + c_val * sens_w0_stack[1]
                
                total_gp_seg = grad_p_stack[0] + c_val * grad_p_stack[1]
                inc_gp = gp_mu0 + c_val * gp_v + total_gp_seg
                
                jax.debug.print("Joint Blk {}: gp_stack[0]={} c_val={} inc_gp={}", block_idx, grad_p_stack[0], c_val, inc_gp)

                new_mesh = grad_ts_stack[0] + c_val * grad_ts_stack[1]
                
                new_pend = init_pending
                
                return total_adj, new_mesh, p_gp + inc_gp, new_pend

            def branch_standard_solve(args):
                p_adj, p_mesh, p_gp, p_pend = args
                
                # Single Sweep
                sens_w0, grad_p_seg, grad_ts, grad_te = run_segment_backward_sweep(
                    W_block, T_block, dL_block, p_adj, n_pts_valid, funcs, dims, p_opt
                )
                
                jax.debug.print("Std Solve Blk {} GP_seg {}", block_idx, grad_p_seg)
                
                return sens_w0, grad_ts, p_gp + grad_p_seg, p_pend
                
            return jax.lax.cond(
                pending_active > 0.5,
                branch_joint_solve,
                branch_standard_solve,
                c_args
            )

        # Case 2: Event
        def case_event(c_args):
             curr_adj, curr_mesh, curr_gp, curr_pend = c_args
             
             # Need Pre and Post states
             prev_block_idx = block_idx - 1 
             next_block_idx = block_idx + 1 
             
             # Assumed valid structure (padded safely)
             # NOTE: block_idx-1 might be -1. We expect inputs padded enough?
             # Standard scan blocks: Pad, ..., Event, Seg.
             # If block_idx=0 is Event?
             # For robustness, we clamp or use safe gather?
             # JAX handles negative index wrap, which is bad here.
             # Assume blocks sorted carefully.
             
             # Extract Pre-Event End State
             w_pre_seg = W_padded[prev_block_idx] # (MAX_PTS, NW)
             # n_pts_pre depends on indices
             n_pts_pre = block_indices[prev_block_idx, 1]
             w_prev_end = w_pre_seg[n_pts_pre - 1] 
             
             # Extract Post-Event Start State
             w_post_seg = W_padded[next_block_idx]
             w_post_start = w_post_seg[0]
             
             t_event = block_param[block_idx] 
             # Extract mapped dL_t (stored in first element of block)
             dL_t = dL_block[0, 0] 
             
             tuple_res = solve_event_system_affine_jit(
                 t_event, w_prev_end, w_post_start, curr_adj, dL_t, curr_mesh, funcs, dims, p_opt
             )
             
             lp_mu0, lp_v, rhs, slope, gp_mu0, gp_v, mu_0, v_null = tuple_res
             
             new_pending = (
                 1.0, 
                 mu_0, v_null,
                 lp_mu0, lp_v,
                 rhs, slope,
                 gp_mu0, gp_v
             )
             
             zeros_adj = jnp.zeros_like(curr_adj)
             return zeros_adj, curr_mesh, curr_gp, new_pending
             
        # Switch
        new_carry = jax.lax.switch(
            b_type,
            [case_pad, case_segment, case_event],
            carry
        )
        jax.debug.print("Blk {}: Type {} GP_in {} GP_out {}", block_idx, b_type, carry[2], new_carry[2])
        return new_carry, None

    final_carry, _ = jax.lax.scan(scan_body, init_carry, scan_indices)
    
    final_grad_p = final_carry[2]
    
    return final_grad_p

# Placeholder for now, will fill logic next

@partial(jax.jit, static_argnames=['funcs', 'dims'])
def run_segment_backward_sweep(w_nodes, ts, dL_nodes, load_at_end, n_pts_valid, funcs, dims, p_opt):
    """
    Backward sweep for a single segment (JIT compatible) with Padding Support.
    n_pts_valid: int, actual number of points in the segment.
    """
    f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_p = dims
    n_w = n_x + n_z

    # Time steps in MAX buffer (indices 0 .. MAX-1)
    # We scan indices k from MAX-2 down to 0.
    # Total scan steps = MAX_PTS - 1
    
    # 1. Determine Terminal Load (at n_pts_valid - 1)
    # We need to extract dL_nodes[n_pts_valid - 1]
    # Since n_pts_valid is dynamic (tracer), we use indexing.
    dL_terminal = dL_nodes[n_pts_valid - 1]
    
    init_load = dL_terminal + load_at_end
    
    # 2. Prepare Scan Inputs (Reversed)
    # CRITICAL: w_nodes, ts, dL_nodes are PADDED arrays of size MAX_PTS
    # Valid data is in indices [0:n_pts_valid]
    # When we reverse, valid data moves to the END of the array
    # So if n_pts_valid=8 and MAX_PTS=100:
    #   - Original: [v0, v1, ..., v7, pad, pad, ..., pad]
    #   - Reversed: [pad, pad, ..., pad, v7, v6, ..., v0]
    # The valid steps are now at indices [MAX_PTS - n_pts_valid : MAX_PTS-1]
    
    # Scan goes from k = N_max-2 down to 0.
    scan_ws_c = w_nodes[:-1][::-1]
    scan_ws_n = w_nodes[1:][::-1]
    scan_ts_c = ts[:-1][::-1]
    scan_ts_n = ts[1:][::-1]
    scan_dLs = dL_nodes[:-1][::-1] 
    
    # 3. Create Validity Mask
    # After reversal, valid indices are at the END
    # Original valid step indices: 0, 1, ..., n_pts_valid-2 (n_pts_valid-1 steps total)
    # After reversal of array[:-1], these map to:
    #   reversed_idx = (MAX_PTS-2) - original_idx
    # So valid reversed indices are: (MAX_PTS-2) - (n_pts_valid-2), ..., (MAX_PTS-2) - 0
    #                              = MAX_PTS - n_pts_valid, ..., MAX_PTS-2
    
    max_steps = w_nodes.shape[0] - 1  # MAX_PTS - 1
    indices_reversed = jnp.arange(max_steps)  # 0 to MAX_PTS-2
    # Valid if: index >= (MAX_PTS - 1) - (n_pts_valid - 1) = MAX_PTS - n_pts_valid
    mask_valid = indices_reversed >= (max_steps + 1 - n_pts_valid)
    scan_mask = mask_valid
    
    #                 n=n_pts_valid, m=max_steps, nv=jnp.sum(scan_mask))

    # Extract Boundary Times for Chain Rule
    # Dynamic access for tf
    t0 = ts[0]
    # tf = ts[n_pts_valid - 1] # Dynamic slice
    # JAX dynamic index
    tf = ts[n_pts_valid - 1]
    duration = tf - t0
    dur_inv = jnp.where(jnp.abs(duration) < 1e-12, 0.0, 1.0/duration)
    
    init_carry = (init_load, 0.0, 0.0, 0.0) # load, (unused), gTs, gTe
    
    def scan_body_seg(carry, inputs):
        # Inputs: w_c, w_n, t_c, t_n, dL_at_n, is_valid
        w_c, w_n, t_c, t_n, dL_at_n, is_valid = inputs
        
        load_n, _, gTs, gTe = carry
        
        # --- Branch: Valid vs Invalid ---
        
        def branch_valid(_):
            grad_p_step, lam_k, partial_load_k = solve_local_step_adjoint_jit(
                 w_c, w_n, t_c, t_n, load_n, funcs, dims, p_opt
            )
            next_load = partial_load_k + dL_at_n
            
            # Time Gradients [d/dt_c, d/dt_n]
            # Re-compute dot_res_t? No, we need output from solve_local_
            # solve_local_step_adjoint_jit was modified (viewed code) to return:
            # lam_k, partial_load_k, grad_p, grad_times
            # Wait, the signature in `solve_local_step_adjoint_jit` output earlier?
            # Viewed snippet: return grad_p_step, lam_combined, partial_load_k
            # It DOES NOT return grad_times yet! The V2 snippet had it.
            # I need to CHECK if solve_local_step_adjoint_jit calculates grad_times.
            
            # Re-implement grad_times calc here to be safe or assuming I update `solve_local`?
            # Better to calc here since `solve_local` in padded might have skipped it.
            
            def dot_res_t(t_c_val, t_n_val):
                 xn, zn = w_n[:n_x], w_n[n_x:]
                 xc, zc = w_c[:n_x], w_c[n_x:]
                 fc = f_fn(t_c_val, xc, zc, p_opt)
                 fn = f_fn(t_n_val, xn, zn, p_opt)
                 h_var = t_n_val - t_c_val
                 r_flow = -xn + xc + (h_var/2.0)*(fc + fn)
                 r_alg = g_fn(t_n_val, xn, zn, p_opt) if n_z > 0 else jnp.array([])
                 lam_f = lam_k[:n_x]
                 lam_a = lam_k[n_x:] if n_z > 0 else jnp.array([])
                 term = jnp.dot(r_flow, lam_f)
                 if n_z>0: term += jnp.dot(r_alg, lam_a)
                 return term
                 
            d_tc, d_tn = jax.grad(dot_res_t, argnums=(0, 1))(t_c, t_n)
            
            # Chain Rule
            # t_k = t0 + tau_k * duration
            # dt_k / dt0 = 1 - tau_k
            # dt_k / dtf = tau_k
            
            tau_c = (t_c - t0) * dur_inv
            tau_n = (t_n - t0) * dur_inv
            
            # Accumulate gTs (Start)
            # Contribution from d_tn and d_tc
            # But scan processes steps. A step involves t_c and t_n.
            # Each node t_k participates in STEP k-1 (as t_n) and STEP k (as t_c).
            # We just sum all partials.
            
            inc_start = d_tc * (1.0 - tau_c) + d_tn * (1.0 - tau_n)
            inc_end   = d_tc * tau_c         + d_tn * tau_n
            
            return (next_load, 0.0, gTs + inc_start, gTe + inc_end), grad_p_step

        def branch_invalid(_):
            # Identity: load passes through
            # next_load = load_n
            # grad_p = 0
            # time grads = 0
            zeros_p = jnp.zeros(n_p)
            return (load_n, 0.0, gTs, gTe), zeros_p

        result = jax.lax.cond(is_valid, branch_valid, branch_invalid, operand=None)
        return result

    # Scan
    inputs = (scan_ws_c, scan_ws_n, scan_ts_c, scan_ts_n, scan_dLs, scan_mask)
    final_carry, grads_p = jax.lax.scan(scan_body_seg, init_carry, inputs)
    
    sens_w0 = final_carry[0]
    total_gp = jnp.sum(grads_p, axis=0)
    jax.debug.print("Seg Total GP: {}", total_gp)
    total_gTs = final_carry[2]
    total_gTe = final_carry[3]
    
    
    return sens_w0, total_gp, total_gTs, total_gTe


@partial(jax.jit, static_argnames=['funcs', 'dims'])
def solve_event_system_affine_jit(t_event, w_prev_end, w_post_start, adjoint_state, dL_t, mesh_sens_prev, funcs, dims, p_opt):
    """
    Computes Affine coefficients for Event Multipliers.
    Returns: (load_prev_mu0, load_prev_v, t_e_rhs_base, t_e_slope, gp_mu0, gp_v, mu_0, v_null)
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
    # Note: w_post_start corresponds to xp (x+), w_prev_end to xpr (x-)
    J_xp = jax.jacfwd(lambda xp: event_res_fn(t_event, xp, w_prev_end, p_opt))(w_post_start)
    J_t_partial = jax.jacfwd(lambda t: event_res_fn(t, w_post_start, w_prev_end, p_opt))(t_event)
    J_xpr = jax.jacfwd(lambda xpr: event_res_fn(t_event, w_post_start, xpr, p_opt))(w_prev_end)
    
    J_te_total = J_t_partial.reshape((-1, 1))

    # 2. Decompose J_xp^T to find mu_0 and v_null
    matrix_for_mu = J_xp.T # (N_w, N_res)
    
    # Solve for mu_0
    mu_0, residuals, rank, s = jnp.linalg.lstsq(matrix_for_mu, -adjoint_state, rcond=1e-9)
    
    # Null Space (Last col of Q)
    Q, _ = jax.scipy.linalg.qr(J_xp, mode='full')
    v_null = Q[:, -1]
    
    # PAD Output to n_mu_max (n_x + n_z + 1)
    # Why +1? Just in case? Or matching some logic?
    # init_pending used n_x + n_z + 1.
    n_mu_max = n_x + n_z + 1
    
    def safe_pad(arr):
        pad_len = n_mu_max - arr.shape[0]
        if pad_len > 0:
            return jnp.pad(arr, (0, pad_len))
        return arr[:n_mu_max] # clip if larger?
        
    mu_0_padded = safe_pad(mu_0)
    v_null_padded = safe_pad(v_null)
    
    # 3. Precompute terms
    # mesh_sens_prev here is actually mesh_sens of NEXT segment (forward time)
    # Passed as 'mesh_sens_next_seg' in v3.
    # In scan state, it is `curr_mesh`.
    
    t_e_rhs_base = dL_t + mesh_sens_prev + jnp.dot(mu_0, J_te_total.flatten())
    t_e_slope    = jnp.dot(v_null, J_te_total.flatten())
    
    # Loads for Previous Segment
    load_prev_mu0 = J_xpr.T @ mu_0
    load_prev_v   = J_xpr.T @ v_null
    
    # Param Gradients
    _, vjp_p = jax.vjp(lambda p: event_res_fn(t_event, w_post_start, w_prev_end, p), p_opt)
    gp_mu0 = vjp_p(mu_0)[0]
    gp_v   = vjp_p(v_null)[0]
    
    return (
        load_prev_mu0, load_prev_v, 
        t_e_rhs_base, t_e_slope, 
        gp_mu0, gp_v, 
        mu_0_padded, v_null_padded
    )



@partial(jax.jit, static_argnames=['funcs', 'dims'])
def solve_local_step_adjoint_jit(w_c, w_n, t_c, t_n, load_at_n, funcs, dims, p_opt):
    """
    Solves for lambda_k and computes partial_load_k.
    Adheres to the logic:
       (dR_k / dw_{k+1})^T * lambda_k = - load_at_n
       
    Wait, if (nx+nz) equations > nx constraints (Flow only), how did v2 do it?
    Let's re-read v2 logic via memory.
    v2: `solve_local_step_adjoint`
    It likely included the "Last Algebraic Constraint" from the next step?
    Or it solved a square system of size (nx+nz) assuming w_n includes z_n?
    
    In Implicit Trapz:
    R_flow(k) = -x_{k+1} + x_k + ...
    R_alg(k+1) = g(t_{k+1}, x_{k+1}, z_{k+1})
    
    The step k+1 satisfies BOTH R_flow(k)=0 and R_alg(k+1)=0.
    The variables are (x_{k+1}, z_{k+1}).
    So stationarity w.r.t (x_{k+1}, z_{k+1}) involves multipliers for BOTH.
    
    Stationarity at k+1:
      Load_from_future + (dR_flow_k / dw_{k+1})^T * lam_flow_k + (dR_alg_{k+1} / dw_{k+1})^T * lam_alg_{k+1} = 0
    
    So `load_at_n` (which is dL/dw_{k+1} + future terms) is BALANCED by:
      1. Flow_k (past step)
      2. Alg_{k+1} (current step constraint)
      
    So we solve for (lam_flow_k, lam_alg_{k+1}).
    The Residuals R_total_at_boundary = [ R_flow_k, R_alg_{k+1} ].
    The Variables = w_{k+1}.
    Jacobian J = d(R_total) / d(w_{k+1}).
    J^T * [lam_flow_k; lam_alg_{k+1}] = - Load_{k+1}.
    
    This is the system!
    
    So `lambda_k` in our return should actually be `[lam_flow_k, lam_alg_{k+1}]`.
    Wait, `lam_alg_{k+1}` belongs to step k+1 logically?
    Yes, but it is solved here.
    
    What about `lam_alg_k`? It will be solved in the NEXT backward step (k-1).
    So we iterate k from N-1 down to 0.
    At step k (connecting k to k+1):
      We find `lam_flow_k` and `lam_alg_{k+1}`.
      We return `partial_load_k` which is contribution to `w_k`.
      Contribution to `w_k` is:
        dL/dw_k (handled in loop accumulation)
        + (dR_flow_k / dw_k)^T * lam_flow_k
        + (dR_alg_k / dw_k)^T * lam_alg_k  <-- Wait.
        
    We don't know `lam_alg_k` yet!
    So `partial_load_k` should ONLY include `(dR_flow_k / dw_k)^T * lam_flow_k`.
    The term `(dR_alg_k / dw_k)^T * lam_alg_k` will be computed in Step k-1?
    NO. Step k-1 computes `lam_alg_k` (index k).
    
    So:
    Loop k (N-1 -> 0):
      1. Inputs: w_k, w_{k+1}, Load_{k+1}.
      2. Solve for X = [lam_flow_k, lam_alg_{k+1}].
         System: J_{k+1}^T * X = - Load_{k+1}.
         Where J_{k+1} = d(Flow_k, Alg_{k+1}) / d(w_{k+1}).
      3. Compute `Load_contribution_to_k` from X.
         (dR_total / dw_k)^T * X ?
         R_total has Flow_k (depends on w_k) and Alg_{k+1} (No depend on w_k).
         So only Flow_k contributes.
         Load_contrib = (dFlow_k / dw_k)^T * lam_flow_k.
      4. Accumulate: next_load_for_k = dL/dw_k + Load_contrib.
      5. BUT, we also need `lam_alg_k` contribution for `Load_for_k` eventually?
         The term `(dAlg_k / dw_k)^T * lam_alg_k` is MISSING from `next_load_for_k`.
         
         BUT `lam_alg_k` is computed in the NEXT iteration (k-1)!
         So Step k-1 will solve for `lam_alg_k` and `lam_flow_{k-1}`.
         Essentially `Load_for_k` (input to k-1) *MUST* include the `dAlg_k` term?
         
         Wait.
         If Step k-1 solves for `lam_alg_k`, it needs `Load_for_k` to be "complete" regarding w_k stationarity?
         Equation at w_k:
           Load_from_k (flow_k-1, alg_k) = 0?
           
         Let's trace:
         Stationarity at w_k:
           dL/dw_k + (Flow_{k-1})' * l_f_{k-1} + (Alg_k)' * l_a_k + (Flow_k)' * l_f_k + (Alg_{k+1})' * l_a_{k+1} = 0?
           No. Alg_{k+1} depends on w_{k+1}.
           Structure:
             w_0 -- Flow_0 --> w_1 -- Flow_1 --> w_2
             |                 |                 |
             Alg_0             Alg_1             Alg_2
             
         Stationarity w_k (for k=1):
           dL/dw_1 
           + (dFlow_0/dw_1)^T * lam_flow_0
           + (dAlg_1 /dw_1)^T * lam_alg_1
           + (dFlow_1/dw_1)^T * lam_flow_1
           = 0
           
         In Step k=1 (processing w_1 -> w_2, finding lam_flow_1, lam_alg_2):
           We know `lam_flow_1`.
           We compute `Load_contribution` = (dFlow_1/dw_1)^T * lam_flow_1.
           This is the "Forward" term acting on w_1.
           
         In Step k=0 (processing w_0 -> w_1, finding lam_flow_0, lam_alg_1):
           We start with `Load_at_1` = dL/dw_1 + Load_contribution (from step 1).
           We solve for `lam_flow_0` and `lam_alg_1`.
           System:
             [ dFlow_0/dw_1   dAlg_1/dw_1 ]^T [ lam_flow_0 ] = - Load_at_1
             [                             ]   [ lam_alg_1  ]
             
       This matches exactly!
       
    So:
    `solve_local_step_adjoint` should solve for `[lam_flow_k, lam_alg_{k+1}]`.
    Using Jacobian w.r.t `w_{k+1}`.
    
    Warning: `g_fn` for Alg_{k+1} uses `t_{k+1}`.
    Check `verify_residual.py`.
    Residuals structure:
      ...
      Flow_{N-1} (step N-1, connects N-1 to N)
      Alg_{N-1}  (at node N-1)
      Alg_{N}    (at node N)
      
    Wait. The block structure in verifying is:
    Segment:
      steps 0..N-2: Flow_k, Alg_k
      step N-1: Flow_{N-1}, Alg_{N-1} ??
      
    Actually: `n_steps` usually means intervals.
    If `n_pts`=N, intervals = N-1.
    Indices 0..N-1.
    verify_residual loop k in 0..N-2:
       Flow_k
       Alg_k
    Then last point: Alg_{N-1}.
    
    So Alg constraint is at START of the step?
    If so, Stationarity at w_{k+1} (node k+1) involves `Alg_{k+1}`.
    Which is in the NEXT block (Next 'k').
    
    So my logic holds:
    At step k (Flow_k), we interact with w_{k+1}.
    At w_{k+1}, we have Alg_{k+1}.
    So we solve for [lam_flow_k, lam_alg_{k+1}].
    
    Implementation:
    """
    f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_p = dims

    # 1. Define Residual function R(w_n) = [ Flow_k(w_n), Alg_{k+1}(w_n) ]
    # Flow_k depends on w_c and w_n.
    # Alg_{k+1} depends on w_n.
    
    def boundary_res(w_n_var):
        xn, zn = w_n_var[:n_x], w_n_var[n_x:]
        
        # Flow Part (only w_n dependency)
        # R_flow = -x_n + (h/2)*f(t_n, x_n, z_n)
        # (plus consitituent from w_c, which is constant here)
        fn_val = f_fn(t_n, xn, zn, p_opt)
        h_step = t_n - t_c
        r_flow = -xn + (h_step/2.0) * fn_val
        
        # Alg Part
        r_alg = g_fn(t_n, xn, zn, p_opt)
        
        return jnp.concatenate([r_flow, r_alg])
        
    # 2. Jacobian J = dR / dw_n
    # Shape: (nx + nz) x (nx + nz)
    J = jax.jacrev(boundary_res)(w_n)
    
    # Check singularity?
    # Ideally checking rcond approx.
    # We solve J^T * lambda = - load
    
    # 3. Solve
    # lam_combined = [lam_flow, lam_alg]
    # Solve A x = b  => J.T x = -load
    # x = solve(J.T, -load)
    
    # Use lstsq for robustness or solve if square
    lam_combined = jnp.linalg.solve(J.T, -load_at_n)
    
    lam_flow = lam_combined[:n_x]
    
    # 4. Compute partial Load contribution to w_c
    # Only Flow_k depends on w_c.
    def flow_res_c(w_c_var):
        xc, zc = w_c_var[:n_x], w_c_var[n_x:]
        fc_val = f_fn(t_c, xc, zc, p_opt)
        h_step = t_n - t_c
        # R_flow part dependent on c
        return xc + (h_step/2.0) * fc_val
        
    # VJP to get (dFlow/dw_c)^T * lam_flow
    # This is efficiently computed via vjp
    _, vjp_fun = jax.vjp(flow_res_c, w_c)
    partial_load_k = vjp_fun(lam_flow)[0]
    
    # 5. Parameter Gradients for this Step
    # Both Flow and Alg depend on p
    def step_res_p(p_var):
        # Flow
        xc, zc = w_c[:n_x], w_c[n_x:]
        xn, zn = w_n[:n_x], w_n[n_x:]
        h_step = t_n - t_c
        fc = f_fn(t_c, xc, zc, p_var)
        fn = f_fn(t_n, xn, zn, p_var)
        r_f = -xn + xc + (h_step/2.0)*(fc + fn)
        
        # Alg (at k+1) - corresponds to lam_alg_{k+1}
        r_a = g_fn(t_n, xn, zn, p_var)
        
        return jnp.concatenate([r_f, r_a])
        
    _, vjp_p = jax.vjp(step_res_p, p_opt)
    grad_p_step = vjp_p(lam_combined)[0]
    jax.debug.print("Grad P Step: {}", grad_p_step) # Debug only

    #                 ln=jnp.linalg.norm(load_at_n), lmn=jnp.linalg.norm(lam_combined), gp=grad_p_step)
    

    return grad_p_step, lam_combined, partial_load_k
    
# =============================================================================
# Helper Functions
# =============================================================================

@partial(jax.jit, static_argnames=['dims']) 
def pad_problem_data_jit(
    ts_all_ragged, ys_all_ragged, event_ts_ragged, 
    max_blocks, max_pts, dims
):
     """
     JIT-compatible padding is hard because inputs are ragged.
     This helper is meant to be called from Python (non-JIT) to prepare arrays.
     """
     pass

def pad_problem_data(ts_all, ys_all, event_infos, max_blocks, max_pts, dims):
    """
    Converts dynamic trajectory data into padded fixed-size arrays.
    """
    n_x, n_z, n_p = dims
    n_w = n_x + n_z
    
    W_padded = np.zeros((max_blocks, max_pts, n_w))
    TS_padded = np.zeros((max_blocks, max_pts))
    block_types = np.zeros(max_blocks, dtype=int)
    block_indices = np.zeros((max_blocks, 2), dtype=int)
    block_param = np.zeros(max_blocks)
    dL_padded = np.zeros((max_blocks, max_pts, n_w))
    
    curr_blk = 0
    n_segs = len(ts_all)
    n_evs = len(event_infos)
    
    for i in range(n_segs):
        # 1. Segment
        if curr_blk >= max_blocks: raise ValueError("Exceeded max_blocks")
        n_pts = ts_all[i].shape[0]
        if n_pts > max_pts: raise ValueError(f"Segment {i} points {n_pts} > max_pts {max_pts}")
            
        W_padded[curr_blk, :n_pts, :] = ys_all[i]
        TS_padded[curr_blk, :n_pts] = ts_all[i]
        block_types[curr_blk] = 1 # Segment
        block_indices[curr_blk] = [0, n_pts] 
        curr_blk += 1
        

        # 2. Event
        if i < n_evs:
            if curr_blk >= max_blocks: raise ValueError("Exceeded max_blocks")
            t_ev, ev_idx = event_infos[i]
            block_types[curr_blk] = 2 # Event
            block_indices[curr_blk] = [0, 1]
            block_param[curr_blk] = t_ev
            curr_blk += 1
            
    return W_padded, TS_padded, block_types, block_indices, block_param, dL_padded

