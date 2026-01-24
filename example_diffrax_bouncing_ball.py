
"""
Test the Diffrax-based Optimizer on the Bouncing Ball example.
"""

import os
import argparse
import yaml
import numpy as np
import json
import matplotlib.pyplot as plt
# Add repo root to path if needed (implicitly handled by running from root)
# from src.discrete_adjoint.dae_optimizer_diffrax import DAEOptimizerDiffrax
# MOVED INSIDE MAIN to allow env vars to be set first

# Reusing some setup logic from the original example
# from src.discrete_adjoint.dae_solver import DAESolver # Needed for Generating Truth
# MOVED INSIDE MAIN

def setup_jax_device(config: dict):
    """Set JAX platform from config."""
    # Ensure config structure exists
    if 'optimizer' not in config:
        print("Warning: 'optimizer' section missing in config")
        return 'cpu'
        
    device = config.get('optimizer', {}).get('device', 'cpu')
    os.environ['JAX_PLATFORM_NAME'] = device
    
    if device == 'gpu':
        gpu_mem_fraction = config.get('optimizer', {}).get('gpu_memory_fraction')
        if gpu_mem_fraction is not None:
             os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(gpu_mem_fraction)
             print(f"GPU memory fraction set to: {gpu_mem_fraction}")
    
    # Verify
    try:
        import jax # Import here to ensure env vars set first? 
        # Actually JAX is already imported at top level. 
        # We must set env vars BEFORE `import jax` or `jax` calls ideally, or use jax.config.
        # But the original file imported `jax.numpy` then set env vars?
        # Wait, the original file had `import os` then `setup_jax_device` then `run...`.
        # And imports of solver were delayed.
        # I should assume `import jax` at top level might lock backend?
        # JAX usually initializes on first use.
        print(f"JAX Platform: {jax.default_backend()}")
    except:
        pass
    return device

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_diffrax_bouncing_ball(config: dict):
    print("=" * 80)
    print("Bouncing Ball - Diffrax Optimizer Test")
    print("=" * 80)
    
    # 1. Setup
    solver_cfg = config['dae_solver']
    opt_cfg = config['optimizer']
    json_path = solver_cfg['dae_specification_file']
    
    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    # 2. Generate Reference (Truth) using the Original Solver 
    # (to ensure we match the problem definition exactly)
    print("\nStep 1: Generate Reference Trajectory (using DAESolver)")
    from src.discrete_adjoint.dae_solver import DAESolver
    solver_true = DAESolver(dae_data, verbose=False)
    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    aug_sol_true = solver_true.solve_augmented(t_span=t_span, ncp=solver_cfg['ncp'])
    
    # Extract targets
    # We want uniform targets
    t_duration = t_span[1] - t_span[0]
    n_targets = max(20, int(20 * t_duration))
    # Avoid extremely close to event boundaries for fairness in discrete consistency
    t_target = np.linspace(t_span[0] + 0.1, t_span[1] - 0.1, n_targets)
    
    # Extract reference values from DAESolver solution
    # Logic: iterate segments
    y_target_list = []
    # Just use interpolation manually
    from scipy.interpolate import interp1d
    
    # Construct global arrays
    t_all = []
    x_all = [] # dim=2
    for seg in aug_sol_true.segments:
        t_all.extend(seg.t)
        x_all.extend(seg.x)
    
    t_all = np.array(t_all)
    x_all = np.array(x_all)
    
    # Interpolate
    # Handle duplicates at jumps by careful interpolation or just simple assumption
    # Here we just want a "noisy measurement"
    # Note: Bouncing ball has unique x(t), v(t) is discontinuous. 
    # State 0 is h (continuous), State 1 is v (discontinuous).
    # Interpolation of v across jump is physically wrong but fine for "noisy data".
    # Better: finding valid segment. But for simplicity, we interpolate.
    
    # Sort t just in case
    sort_idx = np.argsort(t_all)
    t_all = t_all[sort_idx]
    x_all = x_all[sort_idx]
    
    # Simple linear interp
    y_target = np.zeros((len(t_target), 2))
    for i in range(2):
        y_target[:, i] = np.interp(t_target, t_all, x_all[:, i])
        
    print(f"  Generated {len(t_target)} target points.")

    # 3. Optimization Setup
    print("\nStep 2: Setup Diffrax Optimizer")
    
    # Perturb parameters
    # Copy DAE data to modify starts
    dae_data_init = json.loads(json.dumps(dae_data))
    
    p_true = {p['name']: p['value'] for p in dae_data['parameters']}
    optimize_params = opt_cfg['opt_params']
    
    # Apply perturbations (same as original example)
    for p in dae_data_init['parameters']:
        if p['name'] == 'g':
             p['value'] = p_true['g'] * 0.7
        elif p['name'] == 'e':
             p['value'] = p_true['e'] * 0.7
             
    print(f"  Initial params: g={next(p['value'] for p in dae_data_init['parameters'] if p['name']=='g')}")
    print(f"  Initial params: e={next(p['value'] for p in dae_data_init['parameters'] if p['name']=='e')}")

    # Import here to respect JAX platform settings
    from src.discrete_adjoint.dae_optimizer_diffrax import DAEOptimizerDiffrax

    optimizer = DAEOptimizerDiffrax(
        dae_data=dae_data_init,
        optimize_params=optimize_params,
        verbose=True
    )
    
    # 3.1 Debug: Check Initial Trajectory
    print("\nStep 2.5: Verify Initial Trajectory (Debug)")
    p_init = optimizer.p_opt
    try:
        y_debug = optimizer.predict_outputs(p_init, t_span, t_target)
        loss_debug = np.mean((y_debug - y_target)**2)
        print(f"  Initial Pred Loss: {loss_debug}")
        if np.isnan(loss_debug):
            print("  WARNING: Initial loss is NaN! Simulation failed.")
            print(f"  y_debug sample: {y_debug[:5]}")
    except Exception as e:
        print(f"  Simulation failed with error: {e}")
    
    # 4. Run Optimization
    print("\nStep 3: Optimize")
    
    res = optimizer.optimize(
        t_span=t_span,
        target_times=t_target,
        target_outputs=y_target,
        max_iterations=opt_cfg['max_iterations'],
        step_size=opt_cfg['algorithm']['params']['step_size'],
        tol=opt_cfg['tol'],
        print_every=opt_cfg['print_every']
    )
    
    # 5. Results
    print("\nStep 4: Results")
    p_opt = res['params']
    print(f"  Optimized params: {p_opt}")
    
    # Plotting
    try:
        # Predict full trajectory with optimized params
        y_pred_opt = optimizer.predict_outputs(p_opt, t_span, t_target)
        
        # We also want a dense plot for visualization
        t_dense = np.linspace(t_span[0], t_span[1], 500)
        y_pred_dense = optimizer.predict_outputs(p_opt, t_span, t_dense)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Height
        ax = axes[0]
        ax.scatter(t_target, y_target[:, 0], c='k', marker='x', label='Target')
        ax.plot(t_dense, y_pred_dense[:, 0], 'r-', label='Diffrax Opt')
        ax.set_title("Height")
        ax.legend()
        
        # Velocity
        ax = axes[1]
        ax.scatter(t_target, y_target[:, 1], c='k', marker='x', label='Target')
        ax.plot(t_dense, y_pred_dense[:, 1], 'r-', label='Diffrax Opt')
        ax.set_title("Velocity")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig("bouncing_ball_diffrax_result.png")
        print("  Saved plot to bouncing_ball_diffrax_result.png")
        
    except ImportError:
        print("skipped plotting")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config_bouncing_ball.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_jax_device(config)
    run_diffrax_bouncing_ball(config)
