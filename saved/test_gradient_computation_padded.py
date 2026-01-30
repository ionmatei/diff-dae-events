
import numpy as np
import yaml
import os
import sys
import time
import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from src.discrete_adjoint.dae_solver import DAESolver

# Import both versions of residual logic
import debug.verify_residual as dense
from debug.dae_padded_gradient import DAEPaddedGradient



def test_gradient_sweep():
    print("="*80)
    print("TEST: Gradient Computation Sweep (Varying Parameters)")
    print("="*80)

    # 0. Verify JAX Device
    print(f"JAX Default Backend: {jax.default_backend()}")
    print(f"JAX Devices: {jax.devices()}")
    
    # 1. Setup system
    config_path = os.path.join(root_dir, 'config/config_bouncing_ball.yaml')
    dae_data, solver_cfg = dense.load_system(config_path)
    solver = DAESolver(dae_data, verbose=False) # Helper prints off

    t_span = (0.0, 2.0)
    ncp = 15
    
    # Parameters to sweep
    # We will test a set of (g, e) pairs
    # Base: g=9.81, e=0.8
    test_cases = [
        # Vary e (g=9.81)
        {'g': 9.81, 'e': 0.8},
        {'g': 9.81, 'e': 0.5},
        {'g': 9.81, 'e': 0.9},
        {'g': 9.81, 'e': 0.3},
        # Vary g (e=0.8)
        {'g': 5.0,  'e': 0.8},
        {'g': 15.0, 'e': 0.8},
        {'g': 2.0,  'e': 0.8}, # Low gravity
    ]
    
    # Find indices
    param_names = [p['name'] for p in dae_data['parameters']]
    try:
        e_idx = param_names.index('e')
        g_idx = param_names.index('g')
    except ValueError:
        print("Error: parameters 'e' or 'g' not found in system.")
        return

    # JIT Compile the padded solver ONCE (using max sizes)
    MAX_BLOCKS = 30 # MAX_BLOCKS — n_segments + n_events. 
    MAX_PTS = 100 # max collocation points per segment = ncp + 1
    MAX_TARGETS = 200 # MAX_TARGETS — number of time-points returned by prepare_loss_targets. 
    funcs = dense.create_jax_functions(dae_data)
    f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, dims = funcs

    # Instantiate Padded Gradient Computer
    padded_gradient_computer = DAEPaddedGradient(dae_data, max_blocks=MAX_BLOCKS, max_pts=MAX_PTS, max_targets=MAX_TARGETS)
    
    # --- Generate Fixed Target Data (Reference) ---
    print("Generating Fixed Target Data (e=0.8, g=9.81)...")
    reference_p = [p['value'] for p in dae_data['parameters']]
    solver.update_parameters(reference_p)
    sol_ref = solver.solve_augmented(t_span, ncp=ncp)
    target_times, target_data = dense.prepare_loss_targets(sol_ref, dae_data['states'], *t_span)
    print(f"  Target Data Points: {len(target_times)}")
    # -----------------------------------------------
    
    for case in test_cases:
        val_g = case['g']
        val_e = case['e']
        print(f"\n--- Testing parameters g={val_g}, e={val_e} ---")
        
        # Update parameter
        current_p = [p['value'] for p in dae_data['parameters']]
        current_p[e_idx] = val_e
        current_p[g_idx] = val_g
        solver.update_parameters(current_p)
        
        p_opt = jnp.array(current_p)
        
        # Solve Forward
        # print(f"  Solving forward...")
        try:
            sol = solver.solve_augmented(t_span, ncp=ncp)
        except Exception as e:
            print(f"  Solver failed for e={val_e}: {e}")
            continue


        t_loss_start = time.perf_counter()
        # Compute Gradients & Structure internally (needed for Baseline Method A)
        dL_dW, dL_dp, structure = padded_gradient_computer.compute_loss_gradients(
            sol, p_opt, target_times, target_data, blend_sharpness=150.0
        )
        t_loss = (time.perf_counter() - t_loss_start) * 1000.0
        print(f"  Loss Gradient Time: {t_loss:.3f} ms")
        
        # NOTE: For debugging R_global locally, we might still need W_flat. 
        # But if we want to be fully class-reliant, we should probably SKIP local R_global debug 
        # OR ask the class to give us W_flat.
        # Given the request to "remove processing", let's trust the class.
        # However, R_global check is useful. 
        # Let's see if we can get W_flat from packing again? No, that's inefficient.
        # But for 'test' script, correctness check is key.
        # Let's use the class's pack_solution just for local debug if needed, 
        # OR just acknowledge we don't need to manually check residual in the loop if we trust the solver works.
        # The prompt implies refactoring to use the class.
        
        # If we really want to check forward residual here:
        W_flat, structure_dbg, grid_taus_dbg = dense.pack_solution(sol, dae_data)
        
        def R_global(W, p):
             # We need grid_taus for this... 
             return dense.unpack_and_compute_residual(
                W, p, dae_data, structure_dbg, funcs, (p_opt, [0, 1]), grid_taus_dbg, t_final=sol.segments[-1].t[-1] if sol.segments else 2.0
            )

        # DEBUG: Check Forward Residual
        R_val = R_global(W_flat, p_opt)
        R_norm = jnp.linalg.norm(R_val)
        print(f"  Trajectory Stats:")
        print(f"    Segments: {len(sol.segments)}")
        print(f"    Events: {len(sol.events)}")
        min_seg_dur = min([s.t[-1] - s.t[0] for s in sol.segments])
        print(f"    Min Seg Duration: {min_seg_dur:.6e}")
        print(f"    Residual Norm |R(W)|: {R_norm:.6e}")

        if R_norm > 1e-4:
            print(f"    WARNING: Forward solution does not satisfy residual! This invalidates adjoint assumption.")
            # Breakdown
            print(f"    Residual Breakdown:")
            # We need to manually unpack R_val based on structure to see where the error is
            curr_idx = 0
            # 1. Initial Condition Residual (first n_x elements)
            n_x, n_z, _ = dims
            R_init = R_val[curr_idx : curr_idx + n_x]
            print(f"      R_init limit: {jnp.linalg.norm(R_init):.6e}")
            curr_idx += n_x

            seg_idx = 0
            ev_idx = 0
            
            for kind, count, *extra in structure:
                if kind == 'segment':
                    # Segment Residuals:
                    # - ODE residuals (at collocation points)
                    # - Continuity residuals (at end of segment / start of next) - WAIT, check verify_residual_gmres logic
                    # Usually verify_residual computes huge vector. 
                    
                    # Actually, let's just use the structure to slice R_val
                    # The structure of W_flat matches the structure of R_val generally?
                    # NO. R_val structure is DIFFERENT.
                    # R_val includes:
                    # - Initial condition (n_x)
                    # - Per segment:
                    #   - Collocation equations (dims * ncp)
                    #   - Continuity/Event match (dims)
                    pass
            
            # Since checking exact structure is complex, let's just inspect the first few errors
            # Or use a heuristic if we don't have R_structure easily available.
            
            # ALTERNATIVE: Use verification logic from verify_residual_gmres if possible
            # But simpler: Print first global residual
            print(f"      R_val[:10]: {R_val[:10]}")
            
            # Check if it's the LAST segment?
            # R_val size vs W_flat size?
            print(f"      R_val shape: {R_val.shape}, W_flat shape: {W_flat.shape}")
            print(f"      R_val[-20:]: {R_val[-20:]}")
            
            # Inspect Last Segment Data
            last_seg = sol.segments[-1]
            print(f"    Last Segment Details:")
            print(f"      Time points: {last_seg.t}")
            print(f"      States (end): {last_seg.x[-2:]}")
        


        # METHOD A: Dense Matrix Solve (Baseline)
        # print("  Computing Dense Adjoint...")
        t_a_start = time.perf_counter()
        dR_dW_dense = jax.jacfwd(R_global, 0)(W_flat, p_opt)
        dR_dp_dense = jax.jacfwd(R_global, 1)(W_flat, p_opt)

        lambda_dense = jnp.linalg.solve(dR_dW_dense.T + 1e-12*jnp.eye(dR_dW_dense.shape[0]), -dL_dW)
        grad_A = dL_dp + jnp.dot(lambda_dense, dR_dp_dense)
        t_baseline = (time.perf_counter() - t_a_start) * 1000.0
        print(f"  Baseline (Method A) Time: {t_baseline:.3f} ms")
        
        # METHOD C: Padded JIT Solver (Unified)
        # print("  Computing Padded Adjoint (Unified JIT)...")
        t0 = time.perf_counter()
        loss_val, grad_C = padded_gradient_computer.compute_total_gradient(
             sol, p_opt, target_times, target_data, blend_sharpness=150.0
        )
        grad_C.block_until_ready()
        t_jit = (time.perf_counter() - t0) * 1000.0
        
        # Compare
        err = jnp.linalg.norm(grad_C - grad_A)
        print(f"  Dense (Baseline): {grad_A}")
        print(f"  Padded (Adjoint): {grad_C}")
        print(f"  Padded Runtime: {t_jit:.3f} ms")
        print(f"  Gradient Difference: {err:.6e}")
        
        if err < 1e-4:
            print(f"  > SUCCESS for g={val_g}, e={val_e}")
        else:
            print(f"  > FAILURE for g={val_g}, e={val_e}")

if __name__ == "__main__":
    test_gradient_sweep()
