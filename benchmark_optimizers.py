"""
Benchmark script for comparing DEER methods vs Parallel Optimized optimizer.

Benchmarks:
- DEER methods (example_optimizer_deer_methods.py)
- Parallel Optimized (example_optimizer_parallel_optimized.py)

Test configurations:
- Device: CPU only
- ncp values: [500, 1000, 2000, 3000, 4000, 5000]
- Multiple runs for statistical significance

Results are saved in results/ directory as JSON files for later analysis.
"""

import os
import sys
import json
import time
import yaml
import numpy as np
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# NOTE: We do NOT force JAX to CPU here globally anymore.
# The worker process will have JAX_PLATFORM_NAME set in its environment.

# Import JAX-dependent modules
# These imports might trigger JAX initialization, so we depend on the env var being set before script start
try:
    from src.discrete_adjoint.dae_solver import DAESolver
    from src.discrete_adjoint.dae_optimizer_deer_methods import DAEOptimizerDEERMethods
    from src.discrete_adjoint.dae_optimizer_parallel_optimized import DAEOptimizerParallelOptimized
except ImportError:
    # This might fail if we are in orchestrator and src is not in path yet, but we added it above.
    pass


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_dae_data(base_config: dict, ncp: int, device: str = 'cpu'):
    """Prepare DAE data and generate reference trajectory.
    
    Makes a copy of base_config and only overrides device and ncp values.
    All other config parameters are kept from the original config file.
    """
    import copy
    
    # Deep copy config to avoid modifying original
    config = copy.deepcopy(base_config)
    
    # Override only device and ncp
    config['optimizer']['device'] = device
    config['dae_solver']['ncp'] = ncp
    
    solver_cfg = config['dae_solver']
    
    # Load DAE specification
    with open(solver_cfg['dae_specification_file'], 'r') as f:
        dae_data = json.load(f)
    
    # Generate reference trajectory
    solver_true = DAESolver(dae_data)
    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    
    result_true = solver_true.solve(
        t_span=t_span,
        ncp=ncp,
        rtol=solver_cfg['rtol'],
        atol=solver_cfg['atol']
    )
    
    t_ref = result_true['t']
    y_ref = result_true['y']
    
    # Get parameter info
    p_true = np.array([p['value'] for p in dae_data['parameters']])
    param_names = [p['name'] for p in dae_data['parameters']]
    
    # Create perturbed initial guess
    p_init = p_true.copy()
    opt_params = config['optimizer']['opt_params']
    optimize_indices = [param_names.index(name) for name in opt_params]
    
    np.random.seed(42)  # Fixed seed for reproducibility
    perturbation = 0.4
    for idx in optimize_indices:
        p_init[idx] = p_true[idx] * (1 + perturbation * (2 * np.random.rand() - 1))
    
    # Create modified DAE data with initial parameters
    dae_data_init = dae_data.copy()
    dae_data_init['parameters'] = [p.copy() for p in dae_data['parameters']]
    for i, p_dict in enumerate(dae_data_init['parameters']):
        p_dict['value'] = float(p_init[i])
    
    # Extract only optimized parameters
    if opt_params:
        p_init_opt = np.array([p_init[param_names.index(name)] for name in opt_params])
    else:
        p_init_opt = p_init
    
    return {
        'dae_data': dae_data_init,
        't_ref': t_ref,
        'y_ref': y_ref,
        'p_init_opt': p_init_opt,
        'p_true': p_true,
        'param_names': param_names,
        'opt_params': opt_params,
        'config': config,  # Return the modified config
    }


def benchmark_deer_methods(data: dict, method: str = 'bdf6') -> Dict[str, Any]:
    """Benchmark DEER methods optimizer."""
    config = data['config']  # Use config from data
    opt_cfg = config['optimizer']
    
    # Create optimizer
    optimizer = DAEOptimizerDEERMethods(
        data['dae_data'],
        optimize_params=data['opt_params'],
        method=method,
        deer_max_iter=50,
    )
    
    # Get algorithm config
    algorithm_config = opt_cfg.get('algorithm')
    if algorithm_config and 'params' in algorithm_config:
        step_size = algorithm_config['params'].get('step_size', 0.01)
    else:
        step_size = opt_cfg.get('step_size', 0.01)
    
    # Run optimization
    start_time = time.time()
    result = optimizer.optimize(
        t_array=data['t_ref'],
        y_target=data['y_ref'].T,
        p_init=data['p_init_opt'],
        n_iterations=opt_cfg.get('max_iterations', 20),
        step_size=step_size,
        tol=opt_cfg.get('tol', 1e-3),
        verbose=False,
        algorithm_config=algorithm_config
    )
    total_time = time.time() - start_time
    
    # Compute relative error: ||p_opt - p_true|| / ||p_true||
    p_opt = result['p_all']
    rel_error = float(np.linalg.norm(p_opt - data['p_true']) / np.linalg.norm(data['p_true']))
    
    # Extract timing metrics (skip first iteration for warmup)
    times = result['history']['time_per_iter'][1:]
    
    return {
        'method': 'DEER',
        'discretization': method,
        'total_time': total_time,
        'converged': result['converged'],
        'n_iterations': result['n_iterations'],
        'final_loss': float(result['loss_final']),
        'initial_loss': float(result['history']['loss'][0]),
        'final_grad_norm': float(result['history']['gradient_norm'][-1]),
        'rel_error': rel_error,  # Relative parameter error
        # Primary metrics: per-iteration timing
        'avg_time_per_iter': float(np.mean(times)),
        'std_time_per_iter': float(np.std(times)),
        'min_time_per_iter': float(np.min(times)),
        'max_time_per_iter': float(np.max(times)),
        'median_time_per_iter': float(np.median(times)),
    }


def benchmark_parallel_optimized(data: dict, method: str = 'bdf6', 
                                 use_sequential: bool = None) -> Dict[str, Any]:
    """Benchmark parallel optimized optimizer."""
    config = data['config']  # Use config from data
    opt_cfg = config['optimizer']
    
    # Create optimizer
    if use_sequential is None:
        use_parallel_scan = None  # Auto-select based on method
    else:
        use_parallel_scan = not use_sequential  # Manual override
    
    optimizer = DAEOptimizerParallelOptimized(
        data['dae_data'],
        optimize_params=data['opt_params'],
        method=method,
        use_parallel_scan=use_parallel_scan,
        verbose=False
    )
    
    # Get algorithm config
    algorithm_config = opt_cfg.get('algorithm')
    if algorithm_config and 'params' in algorithm_config:
        step_size = algorithm_config['params'].get('step_size', 0.01)
    else:
        step_size = opt_cfg.get('step_size', 0.01)
    
    # Run optimization
    start_time = time.time()
    result = optimizer.optimize(
        t_array=data['t_ref'],
        y_target=data['y_ref'].T,
        p_init=data['p_init_opt'],
        n_iterations=opt_cfg.get('max_iterations', 20),
        step_size=step_size,
        tol=opt_cfg.get('tol', 1e-3),
        verbose=False,
        algorithm_config=algorithm_config,
        print_every=opt_cfg.get('print_every', 10)
    )
    total_time = time.time() - start_time
    
    # Compute relative error: ||p_opt - p_true|| / ||p_true||
    p_opt = result['p_all']
    rel_error = float(np.linalg.norm(p_opt - data['p_true']) / np.linalg.norm(data['p_true']))
    
    # Extract timing metrics (skip first iteration for warmup)
    times = result['history']['time_per_iter'][1:]
    
    # Check if we have detailed timing
    has_detailed_timing = 'dae_solve_time' in result.get('history', {})
    
    metrics = {
        'method': 'discrete_adjoint',
        'discretization': method,
        'scan_mode': 'auto' if use_sequential is None else ('sequential' if use_sequential else 'parallel'),
        'total_time': total_time,
        'converged': result['converged'],
        'n_iterations': result['n_iterations'],
        'final_loss': float(result['loss_final']),
        'initial_loss': float(result['history']['loss'][0]),
        'final_grad_norm': float(result['history']['gradient_norm'][-1]),
        'rel_error': rel_error,
        'avg_time_per_iter': float(np.mean(times)),
        'std_time_per_iter': float(np.std(times)),
        'min_time_per_iter': float(np.min(times)),
        'max_time_per_iter': float(np.max(times)),
        'median_time_per_iter': float(np.median(times)),
    }
    
    if has_detailed_timing:
        dae_times = result['history']['dae_solve_time'][1:]
        adjoint_times = times - np.array(dae_times)
        metrics.update({
            'avg_dae_solve_time': float(np.mean(dae_times)),
            'avg_adjoint_time': float(np.mean(adjoint_times)),
            'std_dae_solve_time': float(np.std(dae_times)),
            'std_adjoint_time': float(np.std(adjoint_times)),
        })
    
    return metrics


def run_worker(args):
    """Run a single benchmark configuration (Worker Mode)."""
    import jax
    # Print JAX device to confirm subprocess environment
    print(f"  [Worker] JAX backend: {jax.devices()[0].platform}, Device: {jax.devices()[0]}")
    
    base_config = load_config(args.benchmark_config)
    
    # Prepare data
    data = prepare_dae_data(base_config, args.ncp, args.device)
    
    result = None
    try:
        if args.target_method == 'DEER':
            result = benchmark_deer_methods(data, args.method)
        elif args.target_method == 'discrete_adjoint':
            result = benchmark_parallel_optimized(data, args.method, use_sequential=None)
        else:
            raise ValueError(f"Unknown target method: {args.target_method}")
            
        result.update({
            'ncp': args.ncp,
            'device': args.device,
            'run': args.run_idx,
            'success': True
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        result = {
            'method': args.target_method,
            'discretization': args.method,
            'ncp': args.ncp,
            'device': args.device,
            'run': args.run_idx,
            'success': False,
            'error': str(e)
        }
    
    # Save result to temporary file
    with open(args.output_file, 'w') as f:
        json.dump(result, f)


def run_benchmark_suite(config_path: str = 'config/config_cauer.yaml',
                        ncp_values: List[int] = None,
                        methods: List[str] = None,
                        devices: List[str] = None,
                        n_runs: int = 3) -> Dict[str, Any]:
    """Orchestrate the benchmark suite by spawning workers."""
    
    if ncp_values is None: ncp_values = [500]
    if methods is None: methods = ['bdf6']
    if devices is None: devices = ['cpu']
    
    # Base config for metadata
    base_config = load_config(config_path)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config_path': config_path,
        'config_settings': {
            'algorithm': base_config.get('optimizer', {}).get('algorithm'),
            'discretization_method': base_config.get('optimizer', {}).get('discretization_method'),
            'max_iterations': base_config.get('optimizer', {}).get('max_iterations'),
            'tol': base_config.get('optimizer', {}).get('tol'),
        },
        'devices': devices,
        'ncp_values': ncp_values,
        'methods': methods,
        'n_runs': n_runs,
        'benchmarks': []
    }
    
    print("=" * 80)
    print("OPTIMIZER BENCHMARK SUITE (Subprocess Mode)")
    print("=" * 80)
    
    total_configs = len(devices) * len(ncp_values) * len(methods) * n_runs * 2
    current_config = 0
    
    # Directory for temporary worker outputs
    os.makedirs('results/tmp', exist_ok=True)
    
    for device in devices:
        print(f"\n{'='*80}")
        print(f"DEVICE: {device.upper()}")
        print(f"{'='*80}")
        
        for ncp in ncp_values:
            for method in methods:
                 for run in range(n_runs):
                    print(f"  Run {run + 1}/{n_runs}, NCP={ncp}, Method={method}:")
                    
                    # Target methods to test
                    targets = ['DEER', 'discrete_adjoint']
                    
                    for target in targets:
                        current_config += 1
                        print(f"    [{current_config}/{total_configs}] {target}...", end=' ', flush=True)
                        
                        output_file = f"results/tmp/result_{device}_{ncp}_{method}_{run}_{target}.json"
                        
                        cmd = [
                            sys.executable, __file__,
                            '--worker',
                            '--benchmark-config', config_path,
                            '--device', device,
                            '--ncp', str(ncp),
                            '--method', method,
                            '--target-method', target,
                            '--run-idx', str(run),
                            '--output-file', output_file
                        ]
                        
                        # Set environment for the worker
                        env = os.environ.copy()
                        env['JAX_PLATFORM_NAME'] = device
                        
                        try:
                            # Run worker
                            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
                            
                            if proc.returncode != 0:
                                print("✗ Failed (Subprocess error)")
                                print(f"      STDERR: {proc.stderr}")
                                results['benchmarks'].append({
                                    'method': target,
                                    'discretization': method,
                                    'ncp': ncp,
                                    'device': device,
                                    'run': run,
                                    'success': False,
                                    'error': f"Subprocess failed: {proc.stderr}"
                                })
                                continue
                                
                            # Read result
                            if os.path.exists(output_file):
                                with open(output_file, 'r') as f:
                                    res = json.load(f)
                                results['benchmarks'].append(res)
                                
                                if res.get('success'):
                                    print(f"✓ ({res.get('total_time', 0):.2f}s)")
                                else:
                                    print(f"✗ Failed (Worker reported error): {res.get('error')}")
                                
                                # Clean up
                                os.remove(output_file)
                            else:
                                print("✗ Failed (No output file)")
                                results['benchmarks'].append({
                                    'method': target,
                                    'discretization': method,
                                    'ncp': ncp,
                                    'device': device,
                                    'run': run,
                                    'success': False,
                                    'error': "Worker produced no output file"
                                })
                                
                        except Exception as e:
                            print(f"✗ Error: {e}")
                            
    return results


def save_results(results: Dict[str, Any], output_dir: str = 'results'):
    """Save benchmark results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'benchmark_{timestamp}.json'
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {filepath}")
    print(f"{'='*80}")
    
    return filepath


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark DAE Optimizers')
    
    # Worker arguments
    parser.add_argument('--worker', action='store_true', help='Run in worker mode')
    parser.add_argument('--target-method', type=str, help='Method to benchmark (DEER or discrete_adjoint)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/gpu)')
    parser.add_argument('--ncp', type=int, default=500, help='Number of collocation points')
    parser.add_argument('--method', type=str, default='bdf6', help='Discretization method')
    parser.add_argument('--run-idx', type=int, default=0, help='Run index')
    parser.add_argument('--output-file', type=str, help='Output JSON file for worker result')
    
    # Orchestrator arguments
    parser.add_argument('--benchmark-config', '-b', default='config/benchmark_config.yaml',
                       help='Path to benchmark configuration file (used by both modes)')
    parser.add_argument('--config-cauer', default='config/config_cauer.yaml', help='Path to optimizer config')
    
    args = parser.parse_args()
    
    if args.worker:
        # WORKER MODE
        run_worker(args)
    else:
        # ORCHESTRATOR MODE
        print(f"Loading benchmark configuration from: {args.benchmark_config}")
        benchmark_config = load_config(args.benchmark_config)
        
        devices = benchmark_config.get('devices', ['cpu'])
        ncp_values = benchmark_config.get('ncp_values', [500, 1000, 2000, 3000, 4000, 5000])
        methods = benchmark_config.get('methods', ['bdf6'])
        n_runs = benchmark_config.get('n_runs', 3)
        optimizer_config_path = benchmark_config.get('optimizer_config_path', args.config_cauer)
        output_dir = benchmark_config.get('output_dir', 'results')
        
        results = run_benchmark_suite(
            config_path=optimizer_config_path,
            ncp_values=ncp_values,
            methods=methods,
            devices=devices,
            n_runs=n_runs
        )
        
        save_results(results, output_dir=output_dir)
