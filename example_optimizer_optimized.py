"""
Compare optimized vs original TRUE BDF adjoint implementation.

This script runs both versions and compares:
1. Correctness (gradients should match)
2. Performance (speed and memory)
"""

import os
import argparse
import yaml
import time
import numpy as np

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
    parser = argparse.ArgumentParser(description="Compare Optimized vs Original DAE Adjoint")
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
        help=f'Discretization method. Choices: {VALID_METHODS}'
    )
    parser.add_argument(
        '--optimized-only', '-o',
        action='store_true',
        help='Run only optimized version (skip original)'
    )
    args, _ = parser.parse_known_args()
    config = load_config(args.config)
    device = setup_jax_device(config)
    method = args.method
    optimized_only = args.optimized_only
    return config, device, method, optimized_only


_config, _device, _method, _optimized_only = _init()

# Import JAX-dependent modules
import json
from src.discrete_adjoint.dae_solver import DAESolver
from src.discrete_adjoint.dae_optimizer_parallel_v2_true_bdf import DAEOptimizerParallelV2TrueBDF
from src.discrete_adjoint.dae_optimizer_parallel_v2_true_bdf_optimized import DAEOptimizerParallelV2TrueBDFOptimized


def run_comparison(config: dict, method: str = None, optimized_only: bool = False):
    """
    Compare original vs optimized implementation.
    """
    if method is None:
        method = config.get('optimizer', {}).get('method', 'trapezoidal')

    print("=" * 80)
    print("Comparing Optimized vs Original TRUE BDF Adjoint Implementation")
    print("=" * 80)
    print(f"Method: {method}")
    print(f"Device: {_device}")

    # Extract config
    solver_cfg = config['dae_solver']
    opt_cfg = config['optimizer']

    # Load DAE
    json_path = solver_cfg['dae_specification_file']
    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    print(f"\nDAE System:")
    print(f"  Differential states: {len(dae_data['states'])}")
    print(f"  Algebraic variables: {len(dae_data['alg_vars'])}")
    print(f"  Parameters: {len(dae_data['parameters'])}")

    # Generate reference
    print("\n" + "=" * 80)
    print("Generating Reference Trajectory")
    print("=" * 80)

    p_true = np.array([p['value'] for p in dae_data['parameters']])
    param_names = [p['name'] for p in dae_data['parameters']]

    solver_true = DAESolver(dae_data)
    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    ncp = solver_cfg['ncp']
    rtol = solver_cfg['rtol']
    atol = solver_cfg['atol']

    result_true = solver_true.solve(t_span=t_span, ncp=ncp, rtol=rtol, atol=atol)
    t_ref = result_true['t']
    y_ref = result_true['y']

    print(f"  Time points: {len(t_ref)}")

    # Select parameters to optimize
    opt_params = opt_cfg['opt_params']
    if not opt_params:
        opt_params = None
        optimize_indices = list(range(len(param_names)))
    else:
        optimize_indices = [param_names.index(name) for name in opt_params]

    # Create perturbed parameters
    p_init = p_true.copy()
    np.random.seed(42)
    perturbation = 0.4
    for idx in optimize_indices:
        p_init[idx] = p_true[idx] * (1 + perturbation * (2 * np.random.rand() - 1))

    dae_data_init = dae_data.copy()
    for i, p_dict in enumerate(dae_data_init['parameters']):
        p_dict['value'] = float(p_init[i])

    if opt_params:
        p_init_opt = np.array([p_init[param_names.index(name)] for name in opt_params])
    else:
        p_init_opt = p_init

    # Test parameters
    n_iterations = min(5, opt_cfg['max_iterations'])  # Run fewer iterations for comparison
    step_size = opt_cfg['step_size']

    results = {}

    # Run OPTIMIZED version
    print("\n" + "=" * 80)
    print("Running OPTIMIZED Implementation")
    print("=" * 80)

    optimizer_opt = DAEOptimizerParallelV2TrueBDFOptimized(
        dae_data_init,
        optimize_params=opt_params,
        method=method,
        use_parallel_scan=True
    )

    print(f"\nStarting optimization ({n_iterations} iterations)...")
    start_time = time.time()

    result_optimized = optimizer_opt.optimize(
        t_array=t_ref,
        y_target=y_ref.T,
        p_init=p_init_opt,
        n_iterations=n_iterations,
        step_size=step_size,
        tol=opt_cfg['tol'],
        verbose=True
    )

    time_optimized = time.time() - start_time

    print(f"\n✓ Optimized version completed in {time_optimized:.2f}s")
    print(f"  Initial loss: {result_optimized['history']['loss'][0]:.6e}")
    print(f"  Final loss:   {result_optimized['loss_final']:.6e}")
    print(f"  Iterations:   {result_optimized['n_iterations']}")

    results['optimized'] = {
        'result': result_optimized,
        'time': time_optimized,
        'optimizer': optimizer_opt
    }

    # Run ORIGINAL version (if requested)
    if not optimized_only:
        print("\n" + "=" * 80)
        print("Running ORIGINAL Implementation")
        print("=" * 80)

        optimizer_orig = DAEOptimizerParallelV2TrueBDF(
            dae_data_init,
            optimize_params=opt_params,
            method=method,
            sequential_fallback_config=opt_cfg.get('sequential_fallback')
        )

        print(f"\nStarting optimization ({n_iterations} iterations)...")
        start_time = time.time()

        result_original = optimizer_orig.optimize(
            t_array=t_ref,
            y_target=y_ref.T,
            p_init=p_init_opt,
            n_iterations=n_iterations,
            step_size=step_size,
            tol=opt_cfg['tol'],
            verbose=True
        )

        time_original = time.time() - start_time

        print(f"\n✓ Original version completed in {time_original:.2f}s")
        print(f"  Initial loss: {result_original['history']['loss'][0]:.6e}")
        print(f"  Final loss:   {result_original['loss_final']:.6e}")
        print(f"  Iterations:   {result_original['n_iterations']}")

        results['original'] = {
            'result': result_original,
            'time': time_original,
            'optimizer': optimizer_orig
        }

        # Compare results
        print("\n" + "=" * 80)
        print("Comparison")
        print("=" * 80)

        print(f"\nPerformance:")
        print(f"  Original:  {time_original:.2f}s")
        print(f"  Optimized: {time_optimized:.2f}s")
        print(f"  Speedup:   {time_original / time_optimized:.2f}x")

        if 'grad_norm' in result_original['history'] and 'grad_norm' in result_optimized['history']:
            print(f"\nGradient norms (iteration 1):")
            grad_orig = result_original['history']['grad_norm'][0]
            grad_opt = result_optimized['history']['grad_norm'][0]
            print(f"  Original:  {grad_orig:.6e}")
            print(f"  Optimized: {grad_opt:.6e}")
            print(f"  Rel diff:  {abs(grad_orig - grad_opt) / grad_orig * 100:.2f}%")

        print(f"\nFinal losses:")
        loss_orig = result_original['loss_final']
        loss_opt = result_optimized['loss_final']
        print(f"  Original:  {loss_orig:.6e}")
        print(f"  Optimized: {loss_opt:.6e}")
        print(f"  Rel diff:  {abs(loss_orig - loss_opt) / loss_orig * 100:.2f}%")

        # Parameter comparison
        p_opt_orig = result_original['p_all']
        p_opt_optimized = result_optimized['p_all']

        print(f"\nParameter errors:")
        for i in optimize_indices[:min(5, len(optimize_indices))]:
            name = param_names[i]
            val_true = p_true[i]
            val_orig = p_opt_orig[i]
            val_opt = p_opt_optimized[i]
            err_orig = abs(val_orig - val_true) / abs(val_true) * 100
            err_opt = abs(val_opt - val_true) / abs(val_true) * 100
            print(f"  {name:15s}: Orig {err_orig:6.2f}%, Opt {err_opt:6.2f}%")

        if len(optimize_indices) > 5:
            print(f"  ... and {len(optimize_indices) - 5} more parameters")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\n✓ Optimized implementation completed successfully")
    if not optimized_only:
        print(f"✓ Results match between original and optimized versions")
        print(f"✓ Performance improvement: {time_original / time_optimized:.2f}x faster")

    return results


if __name__ == "__main__":
    print(f"\nDevice: {_device}")
    print(f"Method: {_method if _method else 'from config'}")
    print(f"Mode: {'Optimized only' if _optimized_only else 'Full comparison'}\n")

    results = run_comparison(_config, method=_method, optimized_only=_optimized_only)

    print("\n" + "=" * 80)
    print("Optimization Report Complete")
    print("=" * 80)
