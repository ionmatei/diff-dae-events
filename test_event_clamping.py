
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import json
import numpy as np
import jax.numpy as jnp
from src.discrete_adjoint.dae_solver import DAESolver
from src.discrete_adjoint.dae_optimizer_event_aware import DAEOptimizerEventAware

def run_test():
    print("============================================================")
    print("TEST: Zeno Clamping Verification")
    print("============================================================")
    
    with open('dae_examples/dae_specification_bouncing_ball.json', 'r') as f:
        dae_data = json.load(f)

    # 1. Generate target trajectory (safe params)
    print("\nGenerating target trajectory (e=0.75)...")
    solver_true = DAESolver(dae_data, verbose=False)
    solver_true.p[1] = 0.75
    aug_sol_true = solver_true.solve_augmented(t_span=(0.0, 1.5), ncp=50)
    
    t_target = []
    y_target = []
    for seg in aug_sol_true.segments:
        t_target.extend(seg.t)
        y_target.extend(seg.x)
    t_target = np.array(t_target)
    y_target = np.array(y_target)
    print(f"Target points: {len(t_target)}, Horizon: {t_target[-1]:.4f}")

    # 2. Test Zeno parameters (e=0.1) with varying gravity to FORCE clamping
    # Low restitution coefficient guarantees rapid event accumulation (Zeno)
    g_values = [9.81, 15.0]
    
    for g_val in g_values:
        print(f"\n------------------------------------------------")
        print(f"Testing with g = {g_val}, e = 0.1 (Zeno forced)")
        print(f"------------------------------------------------")
        
        solver = DAESolver(dae_data, verbose=False)
        solver.p[0] = g_val
        solver.p[1] = 0.1
        
        optimizer = DAEOptimizerEventAware(dae_data, solver, optimize_params=['e'], method='trapezoidal', verbose=False)
        
        # Run forward check first
        print("Running forward solve...")
        aug_sol = solver.solve_augmented(t_span=(0.0, 1.5), ncp=50)
        
        t_final = aug_sol.segments[-1].t[-1]
        n_events = len(aug_sol.events)
        print(f"Result: t_final={t_final:.4f} (Target 1.5000), Events={n_events}")
        
        # Verify Clamping
        if t_final < 1.498:
            print(f"SUCCESS: Simulation clamped early at t={t_final:.4f}.")
            print("  (Zeno protection successfully triggered!)")
        else:
            print("WARNING: Simulation reached end time. Zeno effect NOT detected!")
            print("  This is potentially a FAILURE of the Zeno detection logic if e=0.1.")
            
        # Verify Loss Calculation with Clamped Horizon
        print("Testing loss calculation with clamped target...")
        try:
            # Convert to JAX format
            aug_sol_jax = optimizer._convert_augmented_to_jax(aug_sol)
            
            # Compute loss - this should handle the mismatched horizon via filtering
            # We call the internal method directly to verify it doesn't crash
            # Note: We provide p array needed for evaluation
            loss_val, _ = optimizer.compute_loss_and_forcing(
                aug_sol_jax, 
                jnp.array(t_target), 
                jnp.array(y_target), 
                jnp.array([0.5])
            )
            
            print(f"Loss check PASSED!")
            print(f"  Loss Value: {loss_val:.4f}")
            
            if jnp.isnan(loss_val):
                print("FAILURE: NaN detected in loss!")
            else:
                print("SUCCESS: Loss is finite.")
                
        except Exception as e:
            print(f"FAILURE: Loss calculation crashed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_test()
