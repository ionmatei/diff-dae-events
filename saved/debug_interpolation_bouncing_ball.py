"""
Debug script to test and visualize the interpolation feature of the discrete adjoint optimizer.
Run this from the project root: python debug/debug_interpolation_bouncing_ball.py
"""

import os
import sys
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import jax

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.discrete_adjoint.dae_solver import DAESolver
from src.discrete_adjoint.dae_optimizer_implicit_adjoint import DAEOptimizerImplicitAdjoint

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_debug_interpolation():
    print("=" * 80)
    print("Debug: Interpolation Bouncing Ball")
    print("=" * 80)

    # 1. Load configuration
    config_path = 'config/config_bouncing_ball.yaml'
    print(f"Loading config from {config_path}...")
    config = load_config(config_path)

    solver_cfg = config['dae_solver']
    opt_cfg = config['optimizer']

    # Set device
    device = opt_cfg.get('device', 'cpu')
    os.environ['JAX_PLATFORM_NAME'] = device
    print(f"JAX Platform: {jax.devices()}")

    # 2. Load DAE Specification
    json_path = solver_cfg['dae_specification_file']
    print(f"Loading DAE spec from {json_path}...")
    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    # 3. Reference Simulation (Forward Solve)
    print("\nRunning Forward Simulation...")
    solver = DAESolver(dae_data, verbose=True)
    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    ncp = solver_cfg['ncp']
    
    # Store start time for metrics
    import time
    t0 = time.time()
    
    # Extract config parameters for solver
    max_segments = opt_cfg.get('max_segments', 20)
    max_points_per_seg = opt_cfg.get('max_points_per_segment', 500)
    
    print(f"Solving with max_segments={max_segments}, max_points_per_seg={max_points_per_seg}")
    
    aug_sol = solver.solve_augmented(
        t_span=t_span, 
        ncp=ncp,
        max_segments=max_segments,
        max_points_per_seg=max_points_per_seg
    )
    
    print(f"Simulation done in {time.time() - t0:.4f}s")
    print(f"Number of events: {len(aug_sol.events)}")

    # Details about segments and events
    print("\nAugmented Solution Details:")
    for i, seg in enumerate(aug_sol.segments):
        print(f"  Segment {i}: t_start={seg.t[0]:.4f}, t_end={seg.t[-1]:.4f}, points={len(seg.t)}")
    

    print("\nEvent Details & Time Gaps:")
    for i, ev in enumerate(aug_sol.events):
        print(f"  Event {i}: t={ev.t_event:.15f}")
        print(f"    x_pre:  {ev.x_pre}")
        print(f"    x_post: {ev.x_post}")
        
        # Gap analysis
        t_seg_end = aug_sol.segments[i].t[-1]
        
        # Only compute gap to right if next segment exists
        if i + 1 < len(aug_sol.segments):
            t_next_seg_start = aug_sol.segments[i+1].t[0]
            diff_right = t_next_seg_start - ev.t_event
            diff_left = ev.t_event - t_seg_end
            
            print(f"    Gap Left (Event - Seg{i} End):    {diff_left:.4e}")
            print(f"    Gap Right (Seg{i+1} Start - Event): {diff_right:.4e}")
        else:
             print(f"    (Last Event - no next segment)")


    # 4. Generate Target Times for Interpolation
    # Test Extrapolation: Target range larger than solution range
    # Solution ends at ~1.73s (from previous runs)
    # We extend target to 2.5s to see what happens
    
    t_final_sol = aug_sol.segments[-1].t[-1] if aug_sol.segments else t_span[1]
    t_final_target = t_final_sol + 0.5 # Extend by 0.5s
    
    n_targets = 200
    t_target = np.linspace(t_span[0], t_final_target, n_targets)
    print(f"\nInterpolating at {n_targets} points from {t_span[0]:.4f} to {t_final_target:.4f}...")
    print(f"Solution ends at {t_final_sol:.4f}")

    # 5. Initialize Optimizer (for interpolation method)
    # We use the configurations from config file
    blend_sharpness = opt_cfg.get('blend_sharpness', 100.0)
    prediction_method = opt_cfg.get('prediction_method', 'sigmoid')
    
    print(f"Prediction Method: {prediction_method}")
    if prediction_method == 'sigmoid':
        print(f"Blend Sharpness: {blend_sharpness}")

    optimizer = DAEOptimizerImplicitAdjoint(
        dae_data=dae_data,
        optimize_params=opt_cfg['opt_params'],
        verbose=False, # Reduce noise
        blend_sharpness=blend_sharpness,
        max_segments=opt_cfg.get('max_segments', 20),
        max_points_per_seg=opt_cfg.get('max_points_per_segment', 500),
        prediction_method=prediction_method
    )

    t0 = time.time()
    
    # Check optimization_step filtering
    # We pass the Extended Target (which goes beyond simulation end)
    # The optimizer should truncate it internally to compute loss.
    # We pass dummy initial parameters.
    print(f"\nCalling optimization_step with extended target range ({t_span[0]} to {t_final_target:.4f})...")
    
    # Dummy target outputs (just zeros for testing)
    y_target_dummy = np.zeros((n_targets, 1)) 
    
    p_opt_new, loss, grads, n_seg, n_seg_jit, n_pts, n_pts_jit, t_fwd, t_adj = optimizer.optimization_step(
        t_span=t_span,
        target_times=t_target,
        target_outputs=y_target_dummy,
        p_opt=optimizer.p_all[np.array(optimizer.optimize_indices)],
        ncp=ncp
    )
    
    print(f"Optimization step completed.")
    print(f"Loss: {loss}")
    
    # Just used for plotting
    y_interp = optimizer.predict_outputs(aug_sol, t_target)
    print(f"Interpolation for plot done in {time.time() - t0:.4f}s")

    # 6. Visualization
    print("\nGenerating Plots...")
    
    # Extract raw simulation data for comparison
    t_raw_list = []
    h_raw_list = []
    
    # Height is state index 0 usually, but let's verify
    state_names = [s['name'] for s in dae_data['states']]
    h_idx = state_names.index('h')
    
    for seg in aug_sol.segments:
        t_raw_list.extend(seg.t)
        h_raw_list.extend(seg.x[:, h_idx])
    
    t_raw = np.array(t_raw_list)
    h_raw = np.array(h_raw_list)
    
    # Also get event times
    event_times = [ev.t_event for ev in aug_sol.events]

    plt.figure(figsize=(12, 6))
    
    # Plot Raw Simulation (lines)
    plt.plot(t_raw, h_raw, 'k-', linewidth=1.5, alpha=0.5, label='Raw Simulation (DAESolver)')
    
    # Plot Interpolated Points (dots)
    # y_interp shape is (n_targets, n_outputs). 
    # Typically n_outputs matches state count or specific output function.
    # In bouncing ball, h eq is usually the state itself.
    # If h is defined in 'h' field of dae_spec, DAEOptimizer uses it.
    # Default behavior if 'h' is missing is all states.
    # Let's check dae_data for 'h'
    if 'h' in dae_data: # If output equations defined
        # 'h' in dae_data is usually a list of strings ["h", "v"] or similar
        # We need to find which output corresponds to height 'h'
        # But wait, config says 'dae_specification_bouncing_ball.json'.
        # Usually bouncing ball output is just states.
        # Assuming output 0 corresponds to 'h' for now.
        h_interp = y_interp[:, 0]
    else:
        # If no h, returns all states. h is likely index 0
        h_interp = y_interp[:, h_idx]

    plt.scatter(t_target, h_interp, color='r', s=15, alpha=0.8, label='Interpolated (Optimizer)', zorder=5)
    
    # Plot Events
    for et in event_times:
        plt.axvline(x=et, color='b', linestyle='--', alpha=0.6, linewidth=1, label='Event' if et == event_times[0] else "")

    plt.title(f'Bouncing Ball: Raw Simulation vs Interpolation\nMethod: {prediction_method}, Sharpness: {blend_sharpness}')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (h)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = 'debug/interpolation_debug.png'
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")
    # plt.show() # Commented out for headless

if __name__ == "__main__":
    run_debug_interpolation()
