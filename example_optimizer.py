"""
Example demonstrating the DAE optimizer for parameter identification.

This example shows how to:
1. Generate a reference trajectory with known (true) parameters
2. Select which parameters to optimize (e.g., only capacitors)
3. Perturb ONLY the selected parameters for initial guess (keep others at true values)
4. Use the optimizer to recover the true parameters for selected parameters only
5. Validate that fixed parameters remain unchanged
"""

# Load config and set JAX platform BEFORE importing JAX modules
import os
import argparse
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_jax_device(config: dict):
    """Set JAX platform from config. Must be called before importing JAX modules."""
    device = config.get('optimizer', {}).get('device', 'cpu')
    os.environ['JAX_PLATFORM_NAME'] = device
    return device


# Parse args and configure JAX before other imports
def _init():
    parser = argparse.ArgumentParser(description="DAE Parameter Identification")
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

# Now import JAX-dependent modules
import numpy as np
import json
from src.dae_solver import DAESolver
from src.dae_jacobian import DAEOptimizer


def example_parameter_identification(config: dict):
    """Example: Identify DAE parameters from output trajectory."""

    print("=" * 80)
    print("DAE Parameter Identification Example")
    print("=" * 80)

    # Extract config sections
    solver_cfg = config['dae_solver']
    opt_cfg = config['optimizer']

    # Load DAE specification
    json_path = solver_cfg['dae_specification_file']

    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    print(f"\nLoaded DAE from: {json_path}")
    print(f"  Differential states: {len(dae_data['states'])}")
    print(f"  Algebraic variables: {len(dae_data['alg_vars'])}")
    print(f"  Parameters: {len(dae_data['parameters'])}")

    # Step 1: Generate reference trajectory with TRUE parameters
    print("\n" + "=" * 80)
    print("Step 1: Generate Reference Trajectory")
    print("=" * 80)

    # Store true parameter values
    p_true = np.array([p['value'] for p in dae_data['parameters']])
    param_names = [p['name'] for p in dae_data['parameters']]

    print("\nTrue parameters:")
    for name, val in zip(param_names, p_true):
        print(f"  {name:20s} = {val:.6f}")

    # Solve DAE with true parameters
    solver_true = DAESolver(dae_data)
    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    ncp = solver_cfg['ncp']
    rtol = solver_cfg['rtol']
    atol = solver_cfg['atol']

    print(f"\nSolving DAE with true parameters...")
    print(f"  Time span: {t_span}")
    print(f"  Output points: {ncp}")

    result_true = solver_true.solve(t_span=t_span, ncp=ncp, rtol=rtol, atol=atol)

    t_ref = result_true['t']
    y_ref = result_true['y']  # Reference output trajectory

    print(f"\nReference trajectory generated:")
    print(f"  Time points: {len(t_ref)}")
    print(f"  Output dimension: {y_ref.shape[0]}")
    print(f"  Final time: {t_ref[-1]:.2f}")

    # Step 2: Define which parameters to optimize
    print("\n" + "=" * 80)
    print("Step 2: Select Parameters to Optimize")
    print("=" * 80)

    # Get parameters to optimize from config
    opt_params = opt_cfg['opt_params']

    if not opt_params:
        print("Warning: No optimization parameters specified. Optimizing all parameters.")
        opt_params = None
        optimize_indices = list(range(len(param_names)))
    else:
        print(f"\nParameters to optimize: {opt_params}")
        optimize_indices = [param_names.index(name) for name in opt_params]

    # Step 3: Perturb ONLY the parameters that will be optimized
    print("\n" + "=" * 80)
    print("Step 3: Create Initial Parameter Guess")
    print("=" * 80)

    # Start with true parameters
    p_init = p_true.copy()

    # Perturb only the parameters that will be optimized
    np.random.seed(42)
    perturbation = 0.4
    for idx in optimize_indices:
        p_init[idx] = p_true[idx] * (1 + perturbation * (2 * np.random.rand() - 1))

    print("\nInitial parameter guess:")
    for i, (name, val_true, val_init) in enumerate(zip(param_names, p_true, p_init)):
        error = abs(val_init - val_true) / abs(val_true) * 100
        status = 'Will optimize' if i in optimize_indices else 'Fixed (true value)'
        print(f"  {name:20s} = {val_init:.6f}  (true: {val_true:.6f}, error: {error:>6.1f}%, {status})")

    # Create modified DAE data with initial parameters
    dae_data_init = dae_data.copy()
    for i, p_dict in enumerate(dae_data_init['parameters']):
        p_dict['value'] = float(p_init[i])

    # Step 4: Optimize parameters
    print("\n" + "=" * 80)
    print("Step 4: Optimize Parameters")
    print("=" * 80)

    # Create optimizer with selected parameters
    optimizer = DAEOptimizer(dae_data_init, optimize_params=opt_params)

    # Run optimization
    # Only pass initial values for parameters being optimized
    if opt_params:
        p_init_opt = np.array([p_init[param_names.index(name)] for name in opt_params])
    else:
        p_init_opt = p_init

    result_opt = optimizer.optimize(
        t_array=t_ref,
        y_target=y_ref.T,  # Transpose to (n_time, n_outputs)
        p_init=p_init_opt,
        n_iterations=opt_cfg['max_iterations'],
        step_size=opt_cfg['step_size'],
        tol=opt_cfg['tol'],
        verbose=True
    )

    p_opt_all = result_opt['p_all']  # All parameters (optimized + fixed)

    # Step 5: Compare results
    print("\n" + "=" * 80)
    print("Step 5: Results Comparison")
    print("=" * 80)

    print("\nParameter comparison:")
    print(f"{'Parameter':<20} {'True':>12} {'Initial':>12} {'Optimized':>12} {'Error (%)':>12} {'Status':>12}")
    print("-" * 100)

    for i, (name, val_true, val_init) in enumerate(zip(param_names, p_true, p_init)):
        val_opt = p_opt_all[i]
        error_init = abs(val_init - val_true) / abs(val_true) * 100
        error_opt = abs(val_opt - val_true) / abs(val_true) * 100
        status = 'Optimized' if name in optimizer.optimize_params else 'Fixed'
        print(f"{name:<20} {val_true:>12.6f} {val_init:>12.6f} {val_opt:>12.6f} {error_opt:>11.2f}% {status:>12}")

    print(f"\nInitial loss: {result_opt['history']['loss'][0]:.6e}")
    print(f"Final loss:   {result_opt['loss_final']:.6e}")
    print(f"Reduction:    {result_opt['history']['loss'][0] / result_opt['loss_final']:.1f}x")

    print(f"\nConverged: {result_opt['converged']}")
    print(f"Iterations: {result_opt['n_iterations']}")

    # Step 6: Validate optimized parameters
    print("\n" + "=" * 80)
    print("Step 6: Validate Optimized Parameters")
    print("=" * 80)

    # Solve DAE with optimized parameters
    dae_data_opt = dae_data.copy()
    for i, p_dict in enumerate(dae_data_opt['parameters']):
        p_dict['value'] = float(p_opt_all[i])

    solver_opt = DAESolver(dae_data_opt)
    result_opt_traj = solver_opt.solve(t_span=t_span, ncp=ncp, rtol=rtol, atol=atol)

    y_opt = result_opt_traj['y']
    t_opt = result_opt_traj['t']

    print(f"\nOptimized trajectory:")
    print(f"  Time points: {len(t_opt)}")
    print(f"  Output shape: {y_opt.shape}")
    print(f"  Reference time points: {len(t_ref)}")
    print(f"  Reference output shape: {y_ref.shape}")

    # Compute trajectory error
    traj_error = np.linalg.norm(y_opt - y_ref) / np.linalg.norm(y_ref)
    print(f"\nTrajectory relative error: {traj_error:.6e}")
    print(f"(||y_opt - y_ref|| / ||y_ref||)")

    # Plot optimization history
    print("\n" + "=" * 80)
    print("Plotting Results")
    print("=" * 80)

    try:
        import matplotlib.pyplot as plt

        # Plot 1: Optimization history
        optimizer.plot_optimization_history()

        # Plot 2: Trajectory comparison
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot trajectories
        ax = axes[0]
        n_outputs = y_ref.shape[0]
        for i in range(min(5, n_outputs)):
            ax.plot(t_ref, y_ref[i, :], 'k-', linewidth=2, label=f'True (output {i})')
            ax.plot(t_ref, y_opt[i, :], 'r--', linewidth=2, label=f'Optimized (output {i})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Output')
        ax.set_title('Trajectory Comparison: True vs Optimized')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot trajectory errors
        ax = axes[1]
        for i in range(min(5, n_outputs)):
            error = np.abs(y_opt[i, :] - y_ref[i, :])
            ax.semilogy(t_ref, error, linewidth=2, label=f'Output {i}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Trajectory Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not available, skipping plots")

    print("\n" + "=" * 80)
    print("Example Complete!")
    print("=" * 80)

    return result_opt


if __name__ == "__main__":
    # Config already loaded at module init (for JAX device setup)
    print(f"\nUsing device: {_device}")
    print("\n")
    print("RUNNING FULL PARAMETER IDENTIFICATION EXAMPLE")
    example_parameter_identification(_config)
