"""
Example demonstrating TRUE BDF adjoint with companion matrix.

This uses the correct discrete adjoint for BDF methods (not an approximation).

Key features:
1. TRUE BDF adjoint equations (properly accounts for multi-step coupling)
2. Memory-efficient parallel Jacobian computation
3. Companion matrix for O(log N) parallel scan

Supported methods:
- backward_euler / bdf1: First-order
- trapezoidal: Second-order
- bdf2-bdf6: Higher-order with TRUE adjoint
"""

import os
import argparse
import yaml

VALID_METHODS = ['backward_euler', 'bdf1', 'trapezoidal', 'bdf2', 'bdf3', 'bdf4', 'bdf5', 'bdf6']


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
    parser = argparse.ArgumentParser(description="DAE Parameter Identification (TRUE BDF Adjoint)")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config_cauer.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--method', '-m',
        type=str,
        default=None,
        choices=VALID_METHODS,
        help=f'Discretization method. Choices: {VALID_METHODS}. Default: trapezoidal (or from config)'
    )
    args, _ = parser.parse_known_args()
    config = load_config(args.config)
    device = setup_jax_device(config)
    method = args.method
    return config, device, method


_config, _device, _method = _init()

# Import JAX-dependent modules
import numpy as np
import json
from src.discrete_adjoint.dae_solver import DAESolver
from src.discrete_adjoint.dae_optimizer_parallel_v2_true_bdf import DAEOptimizerParallelV2TrueBDF


def example_true_bdf_adjoint(config: dict, method: str = None):
    """
    Example: Parameter identification with TRUE BDF adjoint.

    Demonstrates:
    - Correct discrete adjoint for BDF methods
    - Multi-step coupling in adjoint equations
    - O(log N) parallel scan via companion matrix
    """
    if method is None:
        method = config.get('optimizer', {}).get('method', 'trapezoidal')

    print("=" * 80)
    print("DAE Parameter Identification with TRUE BDF Adjoint")
    print("=" * 80)
    print(f"Discretization method: {method}")
    print("Using: DAEOptimizerParallelV2TrueBDF")
    print("  - TRUE BDF adjoint (not approximation)")
    print("  - Memory-efficient Jacobian computation")
    print("  - O(log N) parallel scan with companion matrix")

    # Extract config
    solver_cfg = config['dae_solver']
    opt_cfg = config['optimizer']

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
    for name, val in zip(param_names[:10], p_true[:10]):  # Show first 10
        print(f"  {name:20s} = {val:.6f}")
    if len(param_names) > 10:
        print(f"  ... and {len(param_names) - 10} more")

    solver_true = DAESolver(dae_data)
    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    ncp = solver_cfg['ncp']
    rtol = solver_cfg['rtol']
    atol = solver_cfg['atol']

    print(f"\nSolving DAE...")
    print(f"  Time span: {t_span}, Output points: {ncp}")

    result_true = solver_true.solve(t_span=t_span, ncp=ncp, rtol=rtol, atol=atol)
    t_ref = result_true['t']
    y_ref = result_true['y']

    print(f"  Generated {len(t_ref)} time points")

    # Step 2: Select parameters
    print("\n" + "=" * 80)
    print("Step 2: Select Parameters to Optimize")
    print("=" * 80)

    opt_params = opt_cfg['opt_params']
    if not opt_params:
        opt_params = None
        optimize_indices = list(range(len(param_names)))
    else:
        print(f"Parameters to optimize: {opt_params}")
        optimize_indices = [param_names.index(name) for name in opt_params]

    # Step 3: Perturb parameters
    print("\n" + "=" * 80)
    print("Step 3: Create Initial Guess")
    print("=" * 80)

    p_init = p_true.copy()
    np.random.seed(42)
    perturbation = 0.4
    for idx in optimize_indices:
        p_init[idx] = p_true[idx] * (1 + perturbation * (2 * np.random.rand() - 1))

    print(f"\nPerturbed {len(optimize_indices)} parameters")

    # Create DAE with perturbed parameters
    dae_data_init = dae_data.copy()
    for i, p_dict in enumerate(dae_data_init['parameters']):
        p_dict['value'] = float(p_init[i])

    # Step 4: Optimize with TRUE BDF adjoint
    print("\n" + "=" * 80)
    print(f"Step 4: Optimize with TRUE BDF Adjoint ({method})")
    print("=" * 80)

    optimizer = DAEOptimizerParallelV2TrueBDF(
        dae_data_init, 
        optimize_params=opt_params, 
        method=method,
        sequential_fallback_config=opt_cfg.get('sequential_fallback')
    )

    if opt_params:
        p_init_opt = np.array([p_init[param_names.index(name)] for name in opt_params])
    else:
        p_init_opt = p_init

    result_opt = optimizer.optimize(
        t_array=t_ref,
        y_target=y_ref.T,
        p_init=p_init_opt,
        n_iterations=opt_cfg['max_iterations'],
        step_size=opt_cfg['step_size'],
        tol=opt_cfg['tol'],
        verbose=True
    )

    p_opt_all = result_opt['p_all']

    # Step 5: Results
    print("\n" + "=" * 80)
    print("Step 5: Results")
    print("=" * 80)

    print("\nParameter comparison:")
    print(f"{'Parameter':<20} {'True':>12} {'Initial':>12} {'Optimized':>12} {'Error (%)':>12} {'Status':>12}")
    print("-" * 100)

    for i, (name, val_true, val_init) in enumerate(zip(param_names, p_true, p_init)):
        val_opt = p_opt_all[i]
        error_opt = abs(val_opt - val_true) / abs(val_true) * 100 if val_true != 0 else 0
        status = 'Optimized' if name in optimizer.optimize_params else 'Fixed'
        print(f"{name:<20} {val_true:>12.6f} {val_init:>12.6f} {val_opt:>12.6f} {error_opt:>11.2f}% {status:>12}")

    print(f"\nInitial loss: {result_opt['history']['loss'][0]:.6e}")
    print(f"Final loss:   {result_opt['loss_final']:.6e}")
    print(f"Reduction:    {result_opt['history']['loss'][0] / result_opt['loss_final']:.1f}x")

    print(f"\nConverged: {result_opt['converged']}")
    print(f"Iterations: {result_opt['n_iterations']}")

    # Step 6: Validate
    print("\n" + "=" * 80)
    print("Step 6: Validate")
    print("=" * 80)

    dae_data_opt = dae_data.copy()
    for i, p_dict in enumerate(dae_data_opt['parameters']):
        p_dict['value'] = float(p_opt_all[i])

    solver_opt = DAESolver(dae_data_opt)
    result_opt_traj = solver_opt.solve(t_span=t_span, ncp=ncp, rtol=rtol, atol=atol)

    y_opt = result_opt_traj['y']
    traj_error = np.linalg.norm(y_opt - y_ref) / np.linalg.norm(y_ref)
    print(f"\nTrajectory relative error: {traj_error:.6e}")

    # Plot if available
    print("\n" + "=" * 80)
    print("Plotting Results")
    print("=" * 80)

    try:
        import matplotlib.pyplot as plt

        optimizer.plot_optimization_history()

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        ax = axes[0]
        n_outputs = min(5, y_ref.shape[0])
        for i in range(n_outputs):
            ax.plot(t_ref, y_ref[i, :], 'k-', linewidth=2, label=f'True {i}')
            ax.plot(t_ref, y_opt[i, :], 'r--', linewidth=2, label=f'Opt {i}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Output')
        ax.set_title(f'TRUE BDF Adjoint ({method})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for i in range(n_outputs):
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
        print("Matplotlib not available")

    print("\n" + "=" * 80)
    print("Example Complete!")
    print("=" * 80)
    print(f"\n✓ TRUE BDF adjoint successfully used for {method}")
    print(f"✓ Multi-step coupling properly handled")
    print(f"✓ O(log N) parallel scan via companion matrix")

    return result_opt


if __name__ == "__main__":
    print(f"\nDevice: {_device}")
    print(f"Method: {_method if _method else 'from config or default (trapezoidal)'}\n")
    print("=" * 80)
    print("RUNNING TRUE BDF ADJOINT EXAMPLE")
    print("=" * 80)
    print("\nFeatures:")
    print("  ✓ TRUE discrete adjoint (not approximation)")
    print("  ✓ Proper multi-step coupling for BDF")
    print("  ✓ Memory-efficient Jacobian computation")
    print("  ✓ O(log N) parallel scan\n")

    example_true_bdf_adjoint(_config, method=_method)
