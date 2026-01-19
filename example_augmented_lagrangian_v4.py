"""
Example demonstrating Augmented Lagrangian V4 for DAE parameter estimation.

This uses the DAEOptimizerAugmentedLagrangianV4 class which implements
Algorithm 3 (Option C) from the documentation, with the modification of
enforcing a feasible path (solving DAE at each iteration).

Key components:
1. Augmented Lagrangian with VJP-based parameter update.
2. Robust initialization via DAE solve.
3. Feasible path enforcement: solving DAE at each primal step.
"""

import os
import argparse
import yaml
import json
import numpy as np
import time

# Set JAX to float64 for precision
import jax
jax.config.update("jax_enable_x64", True)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_jax_device(config: dict):
    """Set JAX platform from config."""
    device = config.get('optimizer', {}).get('device', 'cpu')
    os.environ['JAX_PLATFORM_NAME'] = device
    
    if device == 'gpu':
        gpu_mem_fraction = config.get('optimizer', {}).get('gpu_memory_fraction')
        if gpu_mem_fraction is not None:
             os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(gpu_mem_fraction)
             print(f"GPU memory fraction set to: {gpu_mem_fraction}")
             
    return device

def _init():
    parser = argparse.ArgumentParser(description="DAE Parameter Identification (Augmented Lagrangian V4)")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config_cauer.yaml',
        help='Path to configuration YAML file'
    )
    args, _ = parser.parse_known_args()
    config = load_config(args.config)
    device = setup_jax_device(config)
    return config, device

_config, _device = _init()

from src.discrete_adjoint.dae_solver import DAESolver
from src.augmented_lagrangian.dae_optimizer_augmented_lagrangian_v4 import DAEOptimizerAugmentedLagrangianV4

def example_augmented_lagrangian_v4(config: dict):
    
    print("=" * 80)
    print("DAE Parameter Identification with Augmented Lagrangian V4")
    print("=" * 80)
    print("Using: DAEOptimizerAugmentedLagrangianV4 (Feasible Path)")
    
    # Extract config
    # Extract config sections
    solver_cfg = config['dae_solver']
    opt_cfg = config['optimizer']
    al_cfg = config.get('augmented_lagrangian', {})
    
    # Load DAE
    json_path = solver_cfg['dae_specification_file']
    with open(json_path, 'r') as f:
        dae_data = json.load(f)
        
    print(f"\nLoaded DAE from: {json_path}")
    print(f"  Differential states: {len(dae_data['states'])}")
    print(f"  Algebraic variables: {len(dae_data['alg_vars'])}")
    print(f"  Parameters: {len(dae_data['parameters'])}")
    
    # Step 1: Generate reference
    print("\n" + "=" * 80)
    print("Step 1: Generate Reference Trajectory")
    print("=" * 80)
    
    p_true = np.array([p['value'] for p in dae_data['parameters']])
    param_names = [p['name'] for p in dae_data['parameters']]
    
    print("\nTrue parameters:")
    for name, val in zip(param_names[:10], p_true[:10]):
        print(f"  {name:20s} = {val:.6f}")
        
    solver_true = DAESolver(dae_data)
    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    ncp = 2000 # Use fixed high grid density for reference
    rtol = solver_cfg['rtol']
    atol = solver_cfg['atol']
    
    print(f"\nSolving DAE for reference...")
    result_true = solver_true.solve(t_span=t_span, ncp=ncp, rtol=rtol, atol=atol)
    t_ref = result_true['t']
    
    x_ref = result_true['x']
    z_ref = result_true['z']
    if z_ref is None or z_ref.size == 0:
        w_ref = x_ref.T
    else:
        w_ref = np.vstack([x_ref, z_ref]).T
        
    y_target = w_ref[:, :len(dae_data['states'])] 
    
    # Step 2: Select parameters
    print("\n" + "=" * 80)
    print("Step 2: Select Parameters to Optimize")
    print("=" * 80)
    
    opt_params = opt_cfg['opt_params']
    if not opt_params:
        print("No optimize params in config?")
        return
        
    print(f"Parameters to optimize: {opt_params}")
    optimize_indices = [param_names.index(name) for name in opt_params]
    
    # Step 3: Create Initial Guess
    print("\n" + "=" * 80)
    print("Step 3: Create Initial Guess")
    print("=" * 80)
    
    p_init = p_true.copy()
    np.random.seed(42)
    perturbation = 0.2
    for idx in optimize_indices:
        p_init[idx] = p_true[idx] * (1 + perturbation * (2 * np.random.rand() - 1))
        
    p_init_opt = np.array([p_init[idx] for idx in optimize_indices])
    
    print(f"Initial guess (perturbed {perturbation*100}%):")
    print(p_init_opt)
    
    # Prepare DAE data with INITIAL parameters
    dae_data_init = dae_data.copy()
    for i, p_dict in enumerate(dae_data_init['parameters']):
        p_dict['value'] = float(p_init[i])
        
    # Step 4: Optimize
    print("\n" + "=" * 80)
    print("Step 4: Optimize with Augmented Lagrangian V4")
    print("=" * 80)
    
    optimizer = DAEOptimizerAugmentedLagrangianV4(
        dae_data_init,
        optimize_params=opt_params,
        method='trapezoidal', 
        loss_type=opt_cfg.get('loss_type', 'sum'),
        verbose=True
    )
    
    # Set hyperparameters
    optimizer.penalty_mu = al_cfg.get('penalty_mu', 1.0)
    optimizer.alpha_w = al_cfg.get('alpha_w', 0.01)
    optimizer.alpha_theta = al_cfg.get('alpha_theta', 0.01)
    optimizer.n_primal_steps = al_cfg.get('n_primal_steps', 5)
    
    # Increase max iterations for convergence testing
    max_iters = opt_cfg.get('max_iterations', 100)
    
    print("\nOptimization Configuration:")
    print(f"  Penalty mu: {optimizer.penalty_mu}")
    print(f"  Alpha w: {optimizer.alpha_w}")
    print(f"  Alpha theta: {optimizer.alpha_theta}")
    print(f"  Primal steps per iter: {optimizer.n_primal_steps}")
    print(f"  Max AL iterations: {max_iters}")
    print(f"  Tolerance: {opt_cfg.get('tol', 1e-4)}")

    result_opt = optimizer.optimize(
        t_array=t_ref,
        y_target=y_target,
        p_init=p_init_opt,
        n_iterations=max_iters,
        tol=opt_cfg.get('tol', 1e-4),
        verbose=True,
        solver_rtol=solver_cfg['rtol'],
        solver_atol=solver_cfg['atol']
    )
    
    p_opt = result_opt['theta_opt']
    w_opt = result_opt['w_opt']
    
    # Step 5: Results
    print("\n" + "=" * 80)
    print("Step 5: Results")
    print("=" * 80)
    
    print("\nParameter comparison:")
    print(f"{'Parameter':<20} {'True':>12} {'Initial':>12} {'Optimized':>12} {'Error (%)':>12}")
    print("-" * 80)
    
    p_true_opt = np.array([p_true[idx] for idx in optimize_indices])
    p_init_disp = p_init_opt
    
    for i, name in enumerate(opt_params):
        val_true = p_true_opt[i]
        val_init = p_init_disp[i]
        val_opt = p_opt[i]
        error_opt = abs(val_opt - val_true) / abs(val_true) * 100 if val_true != 0 else 0
        print(f"{name:<20} {val_true:>12.6f} {val_init:>12.6f} {val_opt:>12.6f} {error_opt:>11.2f}%")
        
    # Plotting
    try:
        import matplotlib.pyplot as plt
        
        # History
        history = result_opt['history']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(history['loss'], 'b-o')
        axes[0, 0].set_title('Augmented Lagrangian Value')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].grid(True)
        
        # Residual Norm
        axes[0, 1].semilogy(history['residual_norm'], 'r-o')
        axes[0, 1].set_title('Residual Norm ||R||')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].grid(True)
        
        # Gradient Norm
        axes[1, 0].semilogy(history['grad_theta_norm'], 'g-o')
        axes[1, 0].set_title('Gradient Theta Norm')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].grid(True)
        
        # Trajectories
        n_plot = min(3, w_opt.shape[1])
        for i in range(n_plot):
            axes[1, 1].plot(t_ref, w_ref[:, i], 'k-', label=f'True x{i}')
            axes[1, 1].plot(t_ref, w_opt[:, i], 'r--', label=f'Opt x{i}')
        axes[1, 1].set_title('Trajectories (Top 3)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        print("\nPlots generated.")
        
    except ImportError:
        print("\nMatplotlib not available.")

if __name__ == "__main__":
    example_augmented_lagrangian_v4(_config)
