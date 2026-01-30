
import numpy as np
import yaml
import json
import os
import sys
import time
import jax
import jax.numpy as jnp
from jax import config
from functools import partial

config.update("jax_enable_x64", True)

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from src.discrete_adjoint.dae_solver import DAESolver

# Import both versions of residual logic
import debug.verify_residual as dense
import debug.verify_residual_gmres as gmres_impl
from debug.verify_residual_direct_padded import compute_adjoint_sweep_padded

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
            ev_t = event_infos[i]
            block_types[curr_blk] = 2 # Event
            block_indices[curr_blk] = [0, 1] 
            block_param[curr_blk] = ev_t
            TS_padded[curr_blk, 0] = ev_t
            curr_blk += 1
            
    return W_padded, TS_padded, block_types, block_indices, block_param, dL_padded

def compare_methods():
    print("="*80)
    print("COMPARISON: Dense vs. Padded JIT Adjoint")
    print("="*80)

    # 1. Setup system
    config_path = os.path.join(root_dir, 'config/config_bouncing_ball.yaml')
    dae_data, solver_cfg = gmres_impl.load_system(config_path)
    solver = DAESolver(dae_data, verbose=False)

    t_span = (0.0, 2.0)
    ncp = 15

    print(f"Generating solution (ncp={ncp})...")
    sol = solver.solve_augmented(t_span, ncp=ncp)
    
    # Extract
    W_flat, structure, grid_taus = gmres_impl.pack_solution(sol, dae_data)
    funcs = gmres_impl.create_jax_functions(dae_data)
    f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_p = dims
    n_w = n_x + n_z

    p_opt = jnp.array([p['value'] for p in dae_data['parameters']])
    
    # 3. Preparation for Loss (actual production loss)
    target_times, target_data = gmres_impl.prepare_loss_targets(sol, dae_data['states'], *t_span)

    def loss_function(W, p):
        segs_t, segs_x, segs_z, ev_tau = gmres_impl.unpack_solution_structure(W, structure, (n_x, n_z, n_w), grid_taus)
        y_pred = gmres_impl.predict_trajectory_sigmoid(segs_t, segs_x, segs_z, ev_tau, target_times, blend_sharpness=150.0)
        return jnp.mean((y_pred - target_data)**2)

    def R_global(W, p):
        return gmres_impl.unpack_and_compute_residual(W, p, dae_data, structure, funcs, (p_opt, [0, 1]), grid_taus)

    # Gradients of Loss
    dL_dp = jax.grad(loss_function, 1)(W_flat, p_opt)
    dL_dW = jax.grad(loss_function, 0)(W_flat, p_opt)
    
    print(f"\ndL/dp (direct from loss): {dL_dp}")
    print(f"dL/dW nonzero: {jnp.count_nonzero(dL_dW)} / {dL_dW.shape[0]}")

    # =========================================================================
    # METHOD A: Dense Matrix Solve
    # =========================================================================
    print("\nMethod A: Dense Matrix Solve...")
    dR_dW_dense = jax.jacfwd(R_global, 0)(W_flat, p_opt)
    dR_dp_dense = jax.jacfwd(R_global, 1)(W_flat, p_opt)

    start_dense = time.time()
    # Regularized solve
    lambda_dense = jnp.linalg.solve(dR_dW_dense.T + 1e-12*jnp.eye(dR_dW_dense.shape[0]), -dL_dW)
    grad_A = dL_dp + jnp.dot(lambda_dense, dR_dp_dense)
    time_dense = time.time() - start_dense
    print(f"  Time: {time_dense:.4f}s")
    print(f"  Result: {grad_A}")

    # =========================================================================
    # METHOD C: Padded JIT Solver
    # =========================================================================
    print("\nMethod C: Padded JIT Solver...")
    
    # Pad Data
    ts_all = [s.t for s in sol.segments]
    ys_all = [jnp.concatenate([s.x, s.z], axis=1) for s in sol.segments]
    event_infos = [e.t_event for e in sol.events]
    
    MAX_BLOCKS = 20
    MAX_PTS = 100
    
    W_p, TS_p, b_types, b_indices, b_param, dL_p = pad_problem_data(
        ts_all, ys_all, event_infos, MAX_BLOCKS, MAX_PTS, dims
    )
    
    # Map dL_dW (flat) to dL_p (padded)
    # The Dense solver computes dL/dW where W is the packed solution
    # We need to unpack this into the padded block structure
    
    # CRITICAL: dL_dW corresponds to the SOLUTION VECTOR W_flat, not the residual
    # So we need to map solution gradients, not residual gradients
    
    # Structure tells us how W_flat is packed:
    # - Segments: [x0, z0, x1, z1, ..., xN, zN] (count points, each n_w dims)
    # - Events: [t_event] (1 scalar)
    
    print(f"  Mapping dL_dW (shape {dL_dW.shape}) to padded format...")
    print(f"  n_w: {n_w}")
    print(f"  Structure: {structure}")

    curr_idx = 0
    blk_idx = 0
    for kind, count, *extra in structure:
         length = extra[0] if kind == 'segment' else count
         end_idx = curr_idx + length 
         
         if kind == 'segment':
              # Extract gradient chunk for this segment
              dL_chunk = dL_dW[curr_idx : end_idx]
              # Reshape to (n_pts, n_w)
              dL_chunk_reshaped = dL_chunk.reshape((count, n_w))
              # Copy to padded array
              dL_p[blk_idx, :count, :] = dL_chunk_reshaped
              blk_idx += 1
         elif kind == 'event_time':
              # Events don't contribute to dL_p states, BUT we need dL/dt_event
              # We store it in the first element of the padding block for the event
              # dL_dW[curr_idx] is the scalar gradient w.r.t event time
              dL_t = dL_dW[curr_idx]
              dL_p[blk_idx, 0, 0] = dL_t
              
              # print(f"  Mapping Event {i}: dL_t={dL_t}")
              blk_idx += 1
              
         curr_idx = end_idx
    
    print(f"  Mapped {blk_idx} blocks, dL_p nonzero: {jnp.count_nonzero(dL_p)}")
    print(f"  Active blocks (non-pad): {jnp.sum(b_types > 0)}")
    
    # Debug: Compare dL_dW vs dL_p
    print(f"  dL_dW stats: min={jnp.min(dL_dW):.4f}, max={jnp.max(dL_dW):.4f}, mean={jnp.mean(jnp.abs(dL_dW)):.4f}")
    print(f"  dL_p stats: min={jnp.min(dL_p):.4f}, max={jnp.max(dL_p):.4f}, mean={jnp.mean(jnp.abs(dL_p)):.4f}")

    # Run JIT Solver
    print("  Compiling...")
    jit_fun = jax.jit(compute_adjoint_sweep_padded, static_argnames=['funcs', 'dims', 'max_blocks'])
    
    start_jit = time.time()
    # First Run (Compile)
    grad_C_raw = jit_fun(
        jnp.array(W_p), jnp.array(TS_p), p_opt,
        jnp.array(b_types), jnp.array(b_indices), jnp.array(b_param),
        jnp.array(dL_p), dL_dp,
        funcs, dims, MAX_BLOCKS
    )
    # Block until done
    _ = grad_C_raw.block_until_ready()
    time_compile = time.time() - start_jit
    print(f"  Compile Time: {time_compile:.4f}s")
    
    start_exec = time.time()
    grad_C = jit_fun(
         jnp.array(W_p), jnp.array(TS_p), p_opt,
        jnp.array(b_types), jnp.array(b_indices), jnp.array(b_param),
        jnp.array(dL_p), dL_dp,
        funcs, dims, MAX_BLOCKS
    )
    grad_C.block_until_ready()
    time_exec = time.time() - start_exec
    print(f"  Exec Time: {time_exec:.4f}s")
    print(f"  Result: {grad_C}")
    
    # Comparison
    err = jnp.linalg.norm(grad_C - grad_A)
    print(f"\nDifference (C vs A): {err:.6e}")
    if err < 1e-4:
        print("SUCCESS: JIT Solver matches Dense baseline.")
    else:
        print("FAILURE: Gradients mismatch.")

if __name__ == "__main__":
    compare_methods()
