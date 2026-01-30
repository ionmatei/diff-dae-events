
"""
Test for equivalence between DAEMatrixGradient (Direct Matrix method) and 
DAEPaddedGradient (Padded Adjoint Sweep method).

This test verifies:
1. Norm of the residual matrix (using Matrix formulation)
2. Jacobian of residual w.r.t W (J_W)
3. Jacobian of residual w.r.t p (J_p)
4. Solution of the adjoint (lambda)
5. Final Parameter Gradient

It uses DAESolver to generate a solution, then evaluates these objects.
"""

import numpy as np
import yaml
import json
import os
import sys
import jax
import jax.numpy as jnp
from jax import config as jax_config
from functools import partial

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_platform_name", "cpu")

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from src.discrete_adjoint.dae_solver import DAESolver
from src.discrete_adjoint.dae_matrix_gradient import DAEMatrixGradient
from src.discrete_adjoint.dae_padded_gradient import DAEPaddedGradient
from src.discrete_adjoint.utils import create_jax_functions, run_segment_backward_sweep, solve_event_system_affine_jit

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    solver_cfg = cfg['dae_solver']
    opt_cfg = cfg['optimizer']
    dae_spec_path = solver_cfg['dae_specification_file']
    if not os.path.isabs(dae_spec_path):
        # Assume relative to project root, not config dir
        dae_spec_path = os.path.join(root_dir, dae_spec_path)
    with open(dae_spec_path, 'r') as f:
        dae_data = json.load(f)
    return dae_data, solver_cfg, opt_cfg

def prepare_loss_targets(sol):
    """Simple extraction of all points."""
    all_t = []
    all_x = []
    for seg in sol.segments:
        if len(seg.t) > 0:
            all_t.append(seg.t)
            all_x.append(seg.x)
    if not all_t:
        return jnp.array([]), jnp.array([])
    return jnp.concatenate([jnp.array(t[:-1]) for t in all_t]), jnp.concatenate([jnp.array(x[:-1]) for x in all_x])

# Helper: Re-implement Matrix Residual exposed for testing
# (Removed: We now use get_internals from DAEMatrixGradient)


# =============================================================================
# Helper: Detailed Multiplier Extraction
# =============================================================================

@partial(jax.jit, static_argnames=['funcs', 'dims'])
def solve_local_step_adjoint_jit(w_c, w_n, t_c, t_n, load_at_n, funcs, dims, p_opt):
    f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_p = dims

    def boundary_res(w_n_var):
        xn, zn = w_n_var[:n_x], w_n_var[n_x:]
        fn_val = f_fn(t_n, xn, zn, p_opt)
        h_step = t_n - t_c
        r_flow = -xn + (h_step/2.0) * fn_val
        r_alg = g_fn(t_n, xn, zn, p_opt) if n_z > 0 else jnp.array([])
        return jnp.concatenate([r_flow, r_alg])
        
    J = jax.jacrev(boundary_res)(w_n)
    
    # Solve J.T * lam = -load
    # lam contains [lam_flow, lam_alg]
    # For singularity robustness, we could use lstsq, but solve is standard for DAEMatrix
    lam_combined = jnp.linalg.solve(J.T, -load_at_n)
    lam_flow = lam_combined[:n_x]
    
    # Partial Load to w_c
    def flow_res_c(w_c_var):
        xc, zc = w_c_var[:n_x], w_c_var[n_x:]
        fc_val = f_fn(t_c, xc, zc, p_opt)
        h_step = t_n - t_c
        return xc + (h_step/2.0) * fc_val
        
    _, vjp_fun = jax.vjp(flow_res_c, w_c)
    partial_load_k = vjp_fun(lam_flow)[0]
    
    # Param Gradients
    def step_res_p(p_var):
        xc, zc = w_c[:n_x], w_c[n_x:]
        xn, zn = w_n[:n_x], w_n[n_x:]
        h_step = t_n - t_c
        fc = f_fn(t_c, xc, zc, p_var)
        fn = f_fn(t_n, xn, zn, p_var)
        r_f = -xn + xc + (h_step/2.0)*(fc + fn)
        r_a = g_fn(t_n, xn, zn, p_var) if n_z > 0 else jnp.array([])
        return jnp.concatenate([r_f, r_a])
        
    _, vjp_p = jax.vjp(step_res_p, p_opt)
    grad_p_step = vjp_p(lam_combined)[0]
    
    return grad_p_step, lam_combined, partial_load_k

@partial(jax.jit, static_argnames=['funcs', 'dims'])
def run_segment_backward_sweep_with_history(w_nodes, ts, dL_nodes, load_at_end, n_pts_valid, funcs, dims, p_opt):
    f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_p = dims
    
    dL_terminal = dL_nodes[n_pts_valid - 1]
    init_load = dL_terminal + load_at_end
    
    scan_ws_c = w_nodes[:-1][::-1]
    scan_ws_n = w_nodes[1:][::-1]
    scan_ts_c = ts[:-1][::-1]
    scan_ts_n = ts[1:][::-1]
    scan_dLs = dL_nodes[:-1][::-1] 
    
    max_steps = w_nodes.shape[0] - 1
    indices_reversed = jnp.arange(max_steps)
    mask_valid = indices_reversed >= (max_steps + 1 - n_pts_valid)
    
    t0, tf = ts[0], ts[n_pts_valid - 1]
    duration = tf - t0
    dur_inv = jnp.where(jnp.abs(duration) < 1e-12, 0.0, 1.0/duration)
    
    init_carry = (init_load, 0.0, 0.0, 0.0)
    
    def scan_body_seg(carry, inputs):
        w_c, w_n, t_c, t_n, dL_at_n, is_valid = inputs
        load_n, _, gTs, gTe = carry
        
        def branch_valid(_):
            grad_p_step, lam_k, partial_load_k = solve_local_step_adjoint_jit(
                 w_c, w_n, t_c, t_n, load_n, funcs, dims, p_opt
            )
            next_load = partial_load_k + dL_at_n
            
            # Time gradients (simplified for this test, needed for continuity logic)
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
            tau_c = (t_c - t0) * dur_inv
            tau_n = (t_n - t0) * dur_inv
            inc_start = d_tc * (1.0 - tau_c) + d_tn * (1.0 - tau_n)
            inc_end   = d_tc * tau_c         + d_tn * tau_n
            
            return (next_load, 0.0, gTs + inc_start, gTe + inc_end), (grad_p_step, lam_k)

        def branch_invalid(_):
             # Dummy return
             return (load_n, 0.0, gTs, gTe), (jnp.zeros(n_p), jnp.zeros(n_x + n_z))
             
        return jax.lax.cond(is_valid, branch_valid, branch_invalid, None)

    scan_inputs = (scan_ws_c, scan_ws_n, scan_ts_c, scan_ts_n, scan_dLs, mask_valid)
    final_carry, (grad_p_hist, lam_hist) = jax.lax.scan(scan_body_seg, init_carry, scan_inputs)
    
    sens_w0, _, gTs, gTe = final_carry
    grad_p_total = jnp.sum(grad_p_hist, axis=0)
    
    return sens_w0, grad_p_total, gTs, gTe, lam_hist

# Helper: Padded Sweep with History for Testing
@partial(jax.jit, static_argnames=['funcs', 'dims', 'max_blocks'])
def compute_adjoint_sweep_padded_with_history(W_padded, TS_padded, p_val, block_types, block_indices, block_param, dL_padded, dL_dp, funcs, dims, max_blocks):
    """
    Runs the adjoint sweep on padded data and returns the FULL adjoint history (per block).
    Returns: (final_grad_p, adjoint_history_list)
    """
    
    # Imports for inner functions
    # from src.discrete_adjoint.utils import solve_event_system_affine_jit # Already imported at top

    n_x, n_z, n_p = dims
    
    # Initial Carry
    init_adjoint = jnp.zeros(n_x + n_z)
    init_mesh_sens = 0.0
    init_grad_p = dL_dp
    
    n_mu_max = n_x + n_z + 1
    init_pending = (
        0.0, jnp.zeros(n_mu_max), jnp.zeros(n_mu_max), 
        jnp.zeros(n_x + n_z), jnp.zeros(n_x + n_z), 
        0.0, 0.0, jnp.zeros(n_p), jnp.zeros(n_p)
    )
    
    init_carry = (init_adjoint, init_mesh_sens, init_grad_p, init_pending)
    scan_indices = jnp.arange(max_blocks)[::-1]
    
    def scan_body(carry, block_idx):
        adj_load, mesh_sens, grad_p, pending = carry
        
        b_type = block_types[block_idx]
        W_block = W_padded[block_idx]
        T_block = TS_padded[block_idx]
        dL_block = dL_padded[block_idx]
        n_pts_valid = block_indices[block_idx, 1]
        
        # 1. Segment Case
        def case_segment(c_args):
            curr_adj, curr_mesh, curr_gp, curr_pend = c_args
            pending_active = curr_pend[0]
            
            def branch_joint_solve(args):
                p_adj, p_mesh, p_gp, p_pend = args
                _, mu_0, v_null, lp_mu0, lp_v, te_rhs, te_slope, gp_mu0, gp_v = p_pend
                
                dL_zeros = jnp.zeros_like(dL_block)
                dL_stack = jnp.stack([dL_block, dL_zeros])
                load_stack = jnp.stack([lp_mu0, lp_v])
                
                # Use WITH HISTORY sweeper
                sweep_vmap = jax.vmap(run_segment_backward_sweep_with_history, in_axes=(None, None, 0, 0, None, None, None, None))
                sens_w0_stack, grad_p_stack, grad_ts_stack, grad_te_stack, lam_hist_stack = sweep_vmap(
                    W_block, T_block, dL_stack, load_stack, n_pts_valid, funcs, dims, p_val
                )
                
                grad_te_mu0, grad_te_v = grad_te_stack[0], grad_te_stack[1]
                denom = te_slope + grad_te_v
                c_val = -(te_rhs + grad_te_mu0) / (denom + 1e-12)
                
                total_adj = sens_w0_stack[0] + c_val * sens_w0_stack[1]
                inc_gp = gp_mu0 + c_val * gp_v + (grad_p_stack[0] + c_val * grad_p_stack[1])
                new_mesh = grad_ts_stack[0] + c_val * grad_ts_stack[1]
                
                # Combine histories
                # lam_hist_stack: (2, MAX_STEP, n_w)
                # lam_total = lam_mu0 + c * lam_v
                lam_hist_comb = lam_hist_stack[0] + c_val * lam_hist_stack[1]
                
                return total_adj, new_mesh, p_gp + inc_gp, init_pending, lam_hist_comb

            def branch_standard_solve(args):
                p_adj, p_mesh, p_gp, p_pend = args
                sens_w0, grad_p_seg, grad_ts, grad_te, lam_hist = run_segment_backward_sweep_with_history(
                    W_block, T_block, dL_block, p_adj, n_pts_valid, funcs, dims, p_val
                )
                return sens_w0, grad_ts, p_gp + grad_p_seg, p_pend, lam_hist
            
            new_carry = jax.lax.cond(pending_active > 0.5, branch_joint_solve, branch_standard_solve, c_args)
            return new_carry[0], new_carry[1], new_carry[2], new_carry[3], new_carry[4]

        # 2. Event Case
        def case_event(c_args):
             curr_adj, curr_mesh, curr_gp, curr_pend = c_args
             prev_block_idx = block_idx - 1 
             next_block_idx = block_idx + 1 
             
             w_pre_seg = W_padded[prev_block_idx]
             n_pts_pre = block_indices[prev_block_idx, 1]
             w_prev_end = w_pre_seg[n_pts_pre - 1] 
             
             w_post_seg = W_padded[next_block_idx]
             w_post_start = w_post_seg[0]
             
             t_event = block_param[block_idx] 
             dL_t = dL_block[0, 0] 
             
             tuple_res = solve_event_system_affine_jit(
                 t_event, w_prev_end, w_post_start, curr_adj, dL_t, curr_mesh, funcs, dims, p_val
             )
             lp_mu0, lp_v, rhs, slope, gp_mu0, gp_v, mu_0, v_null = tuple_res
             
             new_pending = (1.0, mu_0, v_null, lp_mu0, lp_v, rhs, slope, gp_mu0, gp_v)
             zeros_adj = jnp.zeros_like(curr_adj)
             
             # The multiplier for the event is mu_0? 
             # Matrix method includes event residuals. 
             # mu_0 is the multiplier for the event condition R_event = 0 ??
             # Actually mu_0 is multiplier for Jump Cond + Guard?
             # In Matrix method `Event` block yields `residuals.append(val_guard)` and `residuals.append(val_reset)` etc.
             # mu_0 size is n_x + n_z + 1.
             # This likely matches `[guard (1), reset (n_x + n_z)]` or similar.
             # We return mu_0 as the "history" for this block.
             # Pad shape to (MAX_STEPS, n_w) for consistency?
             # Or use separate output channels.
             # No, scan requires uniform output types.
             # We'll use a wrapper struct or just a large array.
             # Segment hist: (MAX, n_w).
             # Event hist: (n_mu). 
             # This is tricky.
             # Let's return a pytree dict: {'seg': ..., 'evt': ...}
             
             # We'll use a wrapper struct or just a large array.
             # Segment hist: (MAX, n_w). Event: (n_w+1,)
             # We'll return a Pytree: `{'kind': ..., 'seg_data': ..., 'evt_data': ...}`
             return zeros_adj, curr_mesh, curr_gp, new_pending, {'kind': 2.0, 'seg_data': jnp.zeros((W_padded.shape[1]-1, n_x+n_z)), 'evt_data': mu_0}

        # 3. Pad Case
        def case_pad(c_args):
            return c_args[0], c_args[1], c_args[2], c_args[3], {'kind': 0.0, 'seg_data': jnp.zeros((W_padded.shape[1]-1, n_x+n_z)), 'evt_data': jnp.zeros(n_x+n_z+1)}

        # Update case_segment to return dict
        def case_segment_wrapper(c_args):
             na, nm, ng, np, hist = case_segment(c_args)
             # hist is (MAX_PTS-1, n_w).
             return na, nm, ng, np, {'kind': 1.0, 'seg_data': hist, 'evt_data': jnp.zeros(n_x+n_z+1)}

        # Re-wrap event/pad
        def case_pad_wrapper(c_args):
             na, nm, ng, np, hist_dict = case_pad(c_args)
             return na, nm, ng, np, hist_dict

        def case_event_wrapper(c_args):
             na, nm, ng, np, hist_dict = case_event(c_args) # hist_dict already has mu_0
             return na, nm, ng, np, hist_dict

        # Switch
        new_adj, new_mesh, new_gp, new_pend, saved_history = jax.lax.switch(
            b_type, [case_pad_wrapper, case_segment_wrapper, case_event_wrapper], carry
        )
        
        return (new_adj, new_mesh, new_gp, new_pend), saved_history

    final_carry, history = jax.lax.scan(scan_body, init_carry, scan_indices)
    return final_carry[2], final_carry[0], history

# =============================================================================
# Main Test
# =============================================================================
def run_test():
    print("=" * 80)
    print("GRADIENT EQUIVALENCE TEST (REAL FUNCTIONS)")
    print("=" * 80)

    # 1. Setup
    config_path = os.path.join(root_dir, 'config/config_bouncing_ball.yaml')
    dae_data, solver_cfg, opt_cfg = load_config(config_path)
    
    # Use non-default parameters
    param_names = [p['name'] for p in dae_data['parameters']]
    true_p = [p['value'] for p in dae_data['parameters']]
    p_test = list(true_p)
    p_test[0] += 1.0 # Change g
    p_test[1] += 0.1 # Change coef
    print(f"Testing with params: {dict(zip(param_names, p_test))}")

    solver = DAESolver(dae_data, verbose=False)
    
    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    ncp = solver_cfg['ncp']
    
    # A. Generate Targets from True Params
    print(f"Generating Target Data (True Params: {dict(zip(param_names, true_p))})...")
    solver.update_parameters(true_p)
    sol_true = solver.solve_augmented(t_span, ncp=ncp)
    target_times, target_data = prepare_loss_targets(sol_true)
    
    # B. Solve at Biased Params
    print(f"Solving DAE at Biased Params: {dict(zip(param_names, p_test))}...")
    solver.update_parameters(p_test)
    sol = solver.solve_augmented(t_span, ncp=ncp)
    t_final = sol.segments[-1].t[-1]
    
    # 2. Matrix Gradient Objects
    print("\n--- MATRIX GRADIENT OBJECTS ---")
    matrix_comp = DAEMatrixGradient(dae_data)
    
    # Get INTERNAL functions actually used
    residual_fn, loss_fn, W_flat, grid_taus_tuple, structure = matrix_comp.get_internals(sol)
    
    # A) Residual
    R = residual_fn(W_flat, jnp.array(p_test), grid_taus_tuple, t_final)
    norm_R = jnp.linalg.norm(R)
    print(f"1. Norm of Residual Matrix: {norm_R:.6e}")
    if norm_R < 1e-8:
        print("   [PASS] Residual norm is close to zero.")
    else:
        print("   [FAIL] Residual norm is too high.")

    # B) Jacobians (Forward AD on REAL function)
    print("Computing Jacobians (Forward AutoDiff)...")
    J_W = jax.jacfwd(residual_fn, argnums=0)(W_flat, jnp.array(p_test), grid_taus_tuple, t_final)
    J_p = jax.jacfwd(residual_fn, argnums=1)(W_flat, jnp.array(p_test), grid_taus_tuple, t_final)
    
    print(f"2. J_W shape: {J_W.shape}, Norm: {jnp.linalg.norm(J_W):.4e}")
    print(f"3. J_p shape: {J_p.shape}, Norm: {jnp.linalg.norm(J_p):.4e}")

    # C) Matrix Loss and Gradients
    blend_sharpness = 150.0
    # Use real loss_fn
    loss_val_matrix, dL_dW = jax.value_and_grad(loss_fn, argnums=0)(W_flat, jnp.array(p_test), grid_taus_tuple, target_times, target_data, t_final, blend_sharpness, True)
    dL_dp = jax.grad(loss_fn, argnums=1)(W_flat, jnp.array(p_test), grid_taus_tuple, target_times, target_data, t_final, blend_sharpness, True)
    
    print(f"Matrix Loss Value: {loss_val_matrix:.6e}")
    
    # Solve Adjoint
    print("Solving Matrix Adjoint System...")
    lam_matrix = jnp.linalg.solve(J_W.T, -dL_dW)
    grad_matrix = dL_dp + jnp.dot(lam_matrix, J_p)
    
    print(f"4. Matrix Adjoint Solution Norm: {jnp.linalg.norm(lam_matrix):.4e}")
    print(f"5. Matrix Total Gradient: {grad_matrix}")


    # 3. Padded Gradient Comparisons
    print("\n--- PADDED GRADIENT COMPARISON ---")
    padded_comp = DAEPaddedGradient(dae_data, max_targets=300)
    
    # Pad Data
    ts_all = [s.t for s in sol.segments]
    ys_all = [jnp.concatenate([s.x, s.z], axis=1) for s in sol.segments]
    event_infos = [e.t_event for e in sol.events]
    
    W_p, TS_p, b_types, b_indices, b_param, _ = padded_comp._pad_problem_data(
        ts_all, ys_all, event_infos
    )
    
    # Compute Padded Gradient (using compute_total_gradient, which calls jit_total_grad)
    tt_padded, td_padded, n_tgt = padded_comp._pad_targets(target_times, target_data)
    n_tgt = jnp.int32(n_tgt)
    
    W_p_d, TS_p_d, b_types_d, b_indices_d, b_param_d, tt_d, td_d = jax.device_put(
        (W_p, TS_p, b_types, b_indices, b_param, tt_padded, td_padded)
    )

    loss_val_padded, grad_padded = padded_comp.jit_total_grad(
        W_p_d, TS_p_d, jnp.array(p_test),
        b_types_d, b_indices_d, b_param_d,
        tt_d, td_d, n_tgt, t_final, blend_sharpness,
        adaptive_horizon=False, soft_interp=True
    )
    
    print(f"Padded Loss Value: {loss_val_padded:.6e}")
    print(f"Padded Total Gradient: {grad_padded}")
    
    # --- COMPARISONS ---
    
    # 1. Loss Value Match
    diff_loss = abs(loss_val_matrix - loss_val_padded)
    print(f"\nLoss Difference: {diff_loss:.6e}")
    if diff_loss < 1e-6:
        print("   [PASS] Loss values match.")
    else:
        print("   [FAIL] Loss values do not match!")

    # 2. Gradient Match
    diff_grad = jnp.linalg.norm(grad_matrix - grad_padded)
    print(f"Gradient Difference: {diff_grad:.6e}")
    if diff_grad < 1e-4:
        print("   [PASS] Gradients match.")
    else:
        print("   [FAIL] Gradients do not match!")

    # 4. Full Multiplier Comparison
    # Recalculate Padded History using simplified sweeper
    dL_dW_np = np.asarray(dL_dW)
    # _, structure, _ = matrix_comp.pack_solution(sol) # structure is already returned by get_internals
    
    # Map dL_dW (Matrix Flat) to Padded Blocks
    # We construct dL_p which matches W_p layout
    dL_p_np = np.zeros(W_p.shape)
    cursor = 0
    blk = 0
    for kind, count, *extra in structure:
        if kind == 'segment':
            length = extra[0] 
            chunk = dL_dW_np[cursor : cursor + length]
            chunk_reshaped = chunk.reshape((count, matrix_comp.dims[0]+matrix_comp.dims[1]))
            dL_p_np[blk, :count, :] = chunk_reshaped
            blk += 1
            cursor += length
        elif kind == 'event_time':
            val = dL_dW_np[cursor]
            dL_p_np[blk, 0, 0] = val
            blk += 1
            cursor += 1
    
    dL_p = jnp.array(dL_p_np)
    dL_p_d = jax.device_put(dL_p)
    
    # Run Padded Sweep for History
    _, adj_t0, adj_history_pytree = compute_adjoint_sweep_padded_with_history(
        W_p_d, TS_p_d, jnp.array(p_test),
        b_types_d, b_indices_d, b_param_d,
        dL_p_d, jnp.zeros_like(grad_padded),
        padded_comp.funcs, padded_comp.dims, padded_comp.max_blocks
    )

    # Reconstruct Matrix-Layout Adjoint Vector
    reconstructed_lam = []
    
    # Add IC Multiplier First (Matrix Layout: [IC, Seg0...])
    # adj_t0 is (nx+nz). Matrix IC is (nx).
    lam_ic = adj_t0[:matrix_comp.dims[0]]
    # Note: Matrix Adjoint definition for IC might have sign flip based on convention?
    # R_ic = x(0) - x_init.
    # Usually we solve J^T lam = -dL/dW.
    # J_ic w.r.t x(0) is I.
    # So lam_ic = -dL/dx(0) (from future terms).
    # Padded adj_t0 IS dL/dx(0).
    # If J^T * lam = -dL/dx, and J=I, then lam = -dL/dx.
    # So we expect `lam_matrix` approx `-adj_padded`.
    # Let's verify sign during test.
    reconstructed_lam.append(-lam_ic)

    # Helper to get block history
    def get_hist_at_scan_index(idx):
        return jax.tree_util.tree_map(lambda x: x[idx], adj_history_pytree)
    
    num_blocks_total = len(structure)
    
    for i, (kind, count, *extra) in enumerate(structure):
        scan_idx = num_blocks_total - 1 - i
        block_hist = get_hist_at_scan_index(scan_idx)
        
        if kind == 'segment':
            n_steps = count - 1
            if n_steps > 0:
                raw_hist = block_hist['seg_data']
                valid_hist = raw_hist[-(n_steps):] # (n_steps, n_w)
                ordered_lam = valid_hist[::-1]
                # Padded adj are sensitivities dL/dx.
                # Matrix lams are multipliers.
                # For dynamic steps: -x_next + x_curr + ... = 0.
                # dR/dx_next = -I.
                # J^T lam = -dL/dx.
                # lam * (-I) = -dL/dx => lam = dL/dx.
                # So for internal steps, lam ~ adj.
                # Let's allow for flexible sign check or just use + for now.
                reconstructed_lam.append(ordered_lam.flatten())
                
        elif kind == 'event_time':
            mu_0 = block_hist['evt_data'] # (nx+nz+1)
            # Matrix Event Residuals Size:
            # 1 (Guard) + (nx+nz) (Reset/Map)
            # Total 1 + nx + nz.
            # Padded mu_0 matches this structure?
            # solve_event_... returns mu_0 used to satisfy:
            # dR/d... * mu = ...
            # Yes. Sign?
            # Usually multipliers have consistent sign with flow.
            # We'll use +mu_0.
            n_mu_real = matrix_comp.dims[0] + matrix_comp.dims[1] + 1
            real_mu = mu_0[:n_mu_real]
            reconstructed_lam.append(real_mu)

    lam_recon_vec = jnp.concatenate(reconstructed_lam)
    
    # Align shapes
    if lam_recon_vec.shape[0] != lam_matrix.shape[0]:
        print(f"Shape Mismatch! Recon: {lam_recon_vec.shape}, Matrix: {lam_matrix.shape}")
        # Pad or truncation issue?
    else:
        diff_lam = jnp.linalg.norm(lam_recon_vec - lam_matrix)
        print(f"\nFull Multiplier Vector Camparison:")
        print(f"   Reconstructed Norm: {jnp.linalg.norm(lam_recon_vec):.4e}")
        print(f"   Matrix Lambda Norm: {jnp.linalg.norm(lam_matrix):.4e}")
        print(f"   Difference Norm:    {diff_lam:.6e}")
        
        if diff_lam < 1e-4:
             print("   [PASS] Full Multiplier vectors match.")
        else:
             print("   [FAIL] Multiplier vectors diverge.")
             # Debug sign?
             diff_plus = jnp.linalg.norm(lam_recon_vec + lam_matrix)
             if diff_plus < 1e-4:
                 print("   [NOTE] Vector matches with SIGN FLIP.")

if __name__ == "__main__":
    run_test()
