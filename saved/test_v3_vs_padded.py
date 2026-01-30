#!/usr/bin/env python3
"""Quick test to compare v2 (direct) vs padded solver with simple loss."""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
import sys
import os

config.update("jax_enable_x64", True)

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from src.discrete_adjoint.dae_solver import DAESolver
import debug.verify_residual_gmres as gmres_impl
from debug.verify_residual_direct_v2 import compute_adjoint_sweep_direct
from debug.verify_residual_direct_padded import compute_adjoint_sweep_padded
from debug.compare_adjoint_methods_padded import pad_problem_data

# Setup
config_path = os.path.join(root_dir, 'config/config_bouncing_ball.yaml')
dae_data, solver_cfg = gmres_impl.load_system(config_path)
solver = DAESolver(dae_data, verbose=False)

t_span = (0.0, 2.0)
ncp = 15

print("Generating solution...")
sol = solver.solve_augmented(t_span, ncp=ncp)

W_flat, structure, grid_taus = gmres_impl.pack_solution(sol, dae_data)
funcs = gmres_impl.create_jax_functions(dae_data)
f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
n_x, n_z, n_p = dims
n_w = n_x + n_z
p_opt = jnp.array([p['value'] for p in dae_data['parameters']])

# Interpolation loss (production)
target_times, target_data = gmres_impl.prepare_loss_targets(sol, dae_data['states'], *t_span)

def loss_interp(W, p):
    segs_t, segs_x, segs_z, ev_tau = gmres_impl.unpack_solution_structure(W, structure, (n_x, n_z, n_w), grid_taus)
    y_pred = gmres_impl.predict_trajectory_sigmoid(segs_t, segs_x, segs_z, ev_tau, target_times, blend_sharpness=150.0)
    return jnp.mean((y_pred - target_data)**2)

dL_dp = jax.grad(loss_interp, 1)(W_flat, p_opt)
dL_dW = jax.grad(loss_interp, 0)(W_flat, p_opt)

print(f"\nInterpolation Loss Test")
print(f"dL/dp: {dL_dp}")
print(f"dL/dW nonzero: {jnp.count_nonzero(dL_dW)}/{len(dL_dW)}")

# Method 1: v2 Direct
print("\n=== V2 Direct Solver ===")
grad_v2 = compute_adjoint_sweep_direct(W_flat, p_opt, dae_data, structure, funcs, grid_taus, dL_dW, dL_dp)
print(f"Result: {grad_v2}")

# Method 2: Padded
print("\n=== Padded Solver ===")
ts_all = [s.t for s in sol.segments]
ys_all = [jnp.concatenate([s.x, s.z], axis=1) for s in sol.segments]
event_infos = [e.t_event for e in sol.events]

MAX_BLOCKS = 20
MAX_PTS = 100

W_p, TS_p, b_types, b_indices, b_param, dL_p = pad_problem_data(
    ts_all, ys_all, event_infos, MAX_BLOCKS, MAX_PTS, dims
)

# Map dL_dW to dL_p
curr_idx = 0
blk_idx = 0
for kind, count, *extra in structure:
    length = extra[0] if kind == 'segment' else count
    end_idx = curr_idx + length 
    
    if kind == 'segment':
        dL_chunk = dL_dW[curr_idx : end_idx]
        dL_chunk_reshaped = dL_chunk.reshape((count, n_w))
        dL_p[blk_idx, :count, :] = dL_chunk_reshaped
        blk_idx += 1
    elif kind == 'event_time':
        dL_t = dL_dW[curr_idx]
        dL_p[blk_idx, 0, 0] = dL_t
        blk_idx += 1
        
    curr_idx = end_idx

grad_padded = compute_adjoint_sweep_padded(
    jnp.array(W_p), jnp.array(TS_p), p_opt,
    jnp.array(b_types), jnp.array(b_indices), jnp.array(b_param),
    jnp.array(dL_p), dL_dp,
    funcs, dims, MAX_BLOCKS
)
print(f"Result: {grad_padded}")

# Compare
diff = jnp.linalg.norm(grad_padded - grad_v2)
print(f"\n=== Comparison ===")
print(f"V2:     {grad_v2}")
print(f"Padded: {grad_padded}")
print(f"Diff:   {diff:.6e}")
if diff < 1e-8:
    print("SUCCESS: Padded matches V2!")
else:
    print("FAILURE: Mismatch detected")
