
import jax
import jax.numpy as jnp
import numpy as np
from jax import config
config.update("jax_enable_x64", True)

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from discrete_adjoint.dae_solver import DAESolver
from debug.verify_residual_direct_padded import compute_adjoint_sweep_padded, solve_local_step_adjoint_jit

# Import Bouncing Ball Model
# We can redefine it here or import. 
# Let's import from examples if possible or simpler to define inline.
# Inline is safer for standalone test.

def create_bouncing_ball_problem():
    # Parameters: g, gam, k
    p_opt = jnp.array([9.8, 0.1, 1000.0])
    
    def f_fn(t, x, z, p):
        # x = [h, v]
        # h_dot = v
        # v_dot = -g - gam*v
        g, gam, k = p[0], p[1], p[2]
        h, v = x[0], x[1]
        return jnp.array([v, -g - gam*v])
    
    def g_fn(t, x, z, p):
        return jnp.array([]) # No algebraic vars for simple ball
        
    def h_fn(t, x, z, p): # Event guard
        return jnp.array([x[0]]) # h=0
        
    def guard_fn(t, x, z, p):
        return x[0]
        
    def reinit_res_fn(t, x_p, z_p, x_r, z_r, p):
        # v_new = -0.8 * v_old
        # h_new = h_old
        v_old = x_r[1]
        v_new = x_p[1]
        return jnp.array([v_new + 0.8 * v_old]) 
        
    reinit_vars = [('state', 1)] # v is reinitialized directly
    
    dims = (2, 0, 3) # nx, nz, np
    
    funcs = (f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, dims)
    return funcs, dims, p_opt

def pad_problem_data(ts_all, ys_all, event_infos, max_blocks, max_pts, dims):
    """
    Converts dynamic trajectory data into padded fixed-size arrays.
    event_infos: List OF FLOATS (times)
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
        n_pts = ts_all[i].shape[0]
        W_padded[curr_blk, :n_pts, :] = ys_all[i]
        TS_padded[curr_blk, :n_pts] = ts_all[i]
        block_types[curr_blk] = 1 # Segment
        block_indices[curr_blk] = [0, n_pts] 
        curr_blk += 1
        
        # 2. Event
        if i < n_evs:
            ev_t = event_infos[i]
            block_types[curr_blk] = 2 # Event
            block_indices[curr_blk] = [0, 1] 
            block_param[curr_blk] = ev_t
            TS_padded[curr_blk, 0] = ev_t
            curr_blk += 1
            
    return W_padded, TS_padded, block_types, block_indices, block_param, dL_padded

def run_test():
    print("Initializing Problem...")
    funcs, dims, p_opt = create_bouncing_ball_problem()
    n_x, n_z, n_p = dims
    
    # 1. Generate Trajectory (Dense)
    solver = DAESolver(funcs, atol=1e-8, rtol=1e-8)
    y0 = jnp.array([1.0, 0.0])
    z0 = jnp.array([])
    t_span = (0.0, 1.5) # Should trigger matches
    
    print("Solving DAE...")
    sol = solver.solve_augmented(y0, z0, t_span, p_opt)
    
    # Extract Segments
    ts_all = [s.ts for s in sol.segments]
    ys_all = [s.ys for s in sol.segments]
    event_infos = [e.t for e in sol.events]
    
    print(f"Generated {len(ts_all)} segments and {len(event_infos)} events.")
    
    # 2. Pad Data
    MAX_BLOCKS = 10
    MAX_PTS = 200 # Ensure enough
    
    print("Padding Data...")
    W_pad, TS_pad, b_types, b_indices, b_param, dL_pad = pad_problem_data(
        ts_all, ys_all, event_infos, MAX_BLOCKS, MAX_PTS, dims
    )
    
    # 3. Define Loss (e.g. minimize final velocity)
    # We put dL/dw at the last point of the last segment.
    # We find the last valid segment index.
    last_seg_idx = -1
    for i in range(MAX_BLOCKS-1, -1, -1):
        if b_types[i] == 1:
            last_seg_idx = i
            break
            
    if last_seg_idx >= 0:
        n_pts_last = b_indices[last_seg_idx, 1]
        # dL/dv = 1.0 (Minimize v?)
        # W = [h, v]
        dL_pad[last_seg_idx, n_pts_last-1, 1] = 1.0
        print(f"Injected loss at Block {last_seg_idx}, Index {n_pts_last-1}")
        
    dL_dp = jnp.zeros(n_p)
    
    # 4. Run Padded Adjoint
    print("Compiling JIT Solver...")
    # Wrap to check JIT
    jit_solver = compute_adjoint_sweep_padded
    
    # Convert inputs to JAX arrays
    W_j = jnp.array(W_pad)
    TS_j = jnp.array(TS_pad)
    b_types_j = jnp.array(b_types)
    b_indices_j = jnp.array(b_indices)
    b_param_j = jnp.array(b_param)
    dL_pad_j = jnp.array(dL_pad)
    dL_dp_j = jnp.array(dL_dp)
    
    print("Running Solver...")
    grads = jit_solver(
        W_j, TS_j, p_opt,
        b_types_j, b_indices_j, b_param_j,
        dL_pad_j, dL_dp_j,
        funcs, dims, MAX_BLOCKS
    )
    
    print("Detailed Parameter Gradients:", grads)
    print("Success! Gradients computed.")

if __name__ == "__main__":
    run_test()
