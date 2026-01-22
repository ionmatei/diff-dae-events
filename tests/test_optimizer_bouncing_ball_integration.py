"""
Integration test for event-aware DAE optimization.

Tests the bouncing ball optimization with the implemented event-aware optimizer.
"""

import os
import sys
import yaml

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configuration file path
CONFIG_PATH = 'config/config_bouncing_ball.yaml'


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_jax_device(config: dict):
    """Set JAX platform from config. Must be called BEFORE importing JAX."""
    device = config.get('optimizer', {}).get('device', 'cpu')
    os.environ['JAX_PLATFORM_NAME'] = device

    if device == 'gpu':
        gpu_mem_fraction = config.get('optimizer', {}).get('gpu_memory_fraction')
        if gpu_mem_fraction is not None:
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(gpu_mem_fraction)

    return device


# Load config and setup device BEFORE importing JAX-dependent modules
config = load_config(CONFIG_PATH)
device = setup_jax_device(config)

# Now import JAX-dependent modules
import json
import numpy as np
from src.discrete_adjoint.dae_optimizer_parallel_optimized import DAEOptimizerParallelOptimized
from src.discrete_adjoint.dae_solver import DAESolver

print("="*80)
print("Event-Aware DAE Optimization - Bouncing Ball Test")
print("="*80)
print(f"  Device: {device}")
print(f"  Config: {CONFIG_PATH}")

# Extract config values
solver_cfg = config.get('dae_solver', {})
opt_cfg = config.get('optimizer', {})

# Solver config
dae_spec_file = solver_cfg.get('dae_specification_file',
                                'dae_examples/dae_specification_bouncing_ball.json')
t_start = solver_cfg.get('start_time', 0.0)
t_stop = solver_cfg.get('stop_time', 3.0)
ncp = solver_cfg.get('ncp', 300)
rtol = solver_cfg.get('rtol', 1e-4)
atol = solver_cfg.get('atol', 1e-4)

# Optimizer config
method = opt_cfg.get('discretization_method', 'bdf6')
loss_type = opt_cfg.get('loss_type', 'mean')
opt_params = opt_cfg.get('opt_params', ['g', 'e'])
max_iterations = opt_cfg.get('max_iterations', 100)
tol = opt_cfg.get('tol', 1e-5)
print_every = opt_cfg.get('print_every', 10)

# Algorithm config
algorithm_config = opt_cfg.get('algorithm')
if algorithm_config:
    step_size = algorithm_config.get('params', {}).get('step_size', 0.01)
else:
    step_size = 0.01
    algorithm_config = {'type': 'SGD', 'params': {'step_size': step_size}}

# Sequential fallback config
seq_cfg = opt_cfg.get('sequential_fallback', {})
use_sequential = seq_cfg.get('enable', False)

# Load bouncing ball DAE
print("\nLoading bouncing ball DAE specification...")
with open(dae_spec_file, 'r') as f:
    dae_data = json.load(f)

param_names = [p['name'] for p in dae_data['parameters']]
print(f"  DAE file: {dae_spec_file}")
print(f"  Parameters: {param_names}")
print(f"  Events: {len(dae_data.get('when', []))}")

# Store true parameter values
p_true = np.array([p['value'] for p in dae_data['parameters']])

# Generate reference data with true parameters
print("\n" + "="*80)
print("Generating Reference Data")
print("="*80)

print("\nTrue parameters:")
for name, val in zip(param_names, p_true):
    print(f"  {name:20s} = {val:.6f}")

solver_ref = DAESolver(dae_data, verbose=False)
t_span = (t_start, t_stop)

print(f"\nSolving DAE with events...")
print(f"  Time span: {t_span}, Output points: {ncp}")

result_ref = solver_ref.solve_with_events(
    t_span=t_span,
    ncp=ncp,
    rtol=rtol,
    atol=atol,
    min_event_delta=0.01,
    verbose=False
)

print(f"  Reference simulation completed")
print(f"  Time points: {len(result_ref['t'])}")
print(f"  Events detected: {len(result_ref['event_times'])}")

# Extract target data (use height as output)
t_target = result_ref['t']
y_target = result_ref['x'][0:1, :].T  # Height only, shape (n_time, 1)

print(f"  Target data shape: {y_target.shape}")

# Add small noise to target data for realism
np.random.seed(42)
noise_level = 0.01
y_target_noisy = y_target + noise_level * np.random.randn(*y_target.shape)

# Create perturbed initial guess
print("\n" + "="*80)
print("Creating Initial Guess")
print("="*80)

# Get indices of parameters to optimize
if not opt_params:
    opt_params = None
    optimize_indices = list(range(len(param_names)))
else:
    print(f"Parameters to optimize: {opt_params}")
    optimize_indices = [param_names.index(name) for name in opt_params]

# Perturb selected parameters
p_init_all = p_true.copy()
perturbation = 0.2
for idx in optimize_indices:
    p_init_all[idx] = p_true[idx] * (1 + perturbation * (2 * np.random.rand() - 1))

print(f"\nPerturbed {len(optimize_indices)} parameters by up to {perturbation*100:.0f}%:")
for idx in optimize_indices:
    name = param_names[idx]
    print(f"  {name:20s}: {p_true[idx]:.6f} -> {p_init_all[idx]:.6f}")

# Create DAE with perturbed parameters for optimizer
dae_data_init = dae_data.copy()
dae_data_init['parameters'] = [p.copy() for p in dae_data['parameters']]
for i, p_dict in enumerate(dae_data_init['parameters']):
    p_dict['value'] = float(p_init_all[i])

# Create optimizer
print("\n" + "="*80)
print(f"Running Optimization ({method})")
print("="*80)

print(f"\nOptimizer settings:")
print(f"  Method: {method}")
print(f"  Algorithm: {algorithm_config.get('type', 'SGD')}")
print(f"  Step size: {step_size}")
print(f"  Max iterations: {max_iterations}")
print(f"  Tolerance: {tol}")
print(f"  Use sequential: {use_sequential}")

solver = DAESolver(dae_data_init, verbose=False)
optimizer = DAEOptimizerParallelOptimized(
    dae_data=dae_data_init,
    dae_solver=solver,
    optimize_params=opt_params,
    loss_type=loss_type,
    method=method,
    use_parallel_scan=not use_sequential,
    rtol=rtol,
    atol=atol,
    verbose=True
)

# Extract initial values for parameters to optimize
if opt_params:
    p_init_opt = np.array([p_init_all[param_names.index(name)] for name in opt_params])
else:
    p_init_opt = p_init_all

# Run optimization
try:
    result_opt = optimizer.optimize(
        t_array=t_target,
        y_target=y_target_noisy,
        p_init=p_init_opt,
        n_iterations=max_iterations,
        step_size=step_size,
        tol=tol,
        verbose=True,
        algorithm_config=algorithm_config,
        print_every=print_every,
        min_event_delta=0.01
    )

    print("\n" + "="*80)
    print("Optimization Results")
    print("="*80)

    p_opt = result_opt['p_opt']

    print("\nParameter comparison:")
    print(f"  {'Parameter':<20} {'True':>12} {'Initial':>12} {'Optimized':>12} {'Error %':>10}")
    print("  " + "-"*70)

    total_init_error = 0.0
    total_final_error = 0.0

    for i, name in enumerate(opt_params if opt_params else param_names):
        idx = param_names.index(name) if opt_params else i
        true_val = p_true[idx]
        init_val = p_init_all[idx]
        opt_val = p_opt[i]
        error_pct = abs(opt_val - true_val) / abs(true_val) * 100 if true_val != 0 else 0

        total_init_error += (init_val - true_val)**2
        total_final_error += (opt_val - true_val)**2

        print(f"  {name:<20} {true_val:>12.6f} {init_val:>12.6f} {opt_val:>12.6f} {error_pct:>10.2f}%")

    init_error = np.sqrt(total_init_error)
    final_error = np.sqrt(total_final_error)

    print(f"\nOptimization stats:")
    print(f"  Converged: {result_opt['converged']}")
    print(f"  Final loss: {result_opt['loss_final']:.6e}")
    print(f"  Iterations: {len(result_opt['history']['loss'])}")

    if len(result_opt['history']['n_events']) > 0:
        print(f"  Avg events per iteration: {np.mean(result_opt['history']['n_events']):.1f}")
        print(f"  Early terminations: {sum(result_opt['history']['early_termination'])}")

    # Timing statistics
    if 'time_forward' in result_opt['history'] and len(result_opt['history']['time_forward']) > 1:
        t_fwd = result_opt['history']['time_forward']
        t_adj = result_opt['history']['time_adjoint']
        t_total = result_opt['history']['time_per_iter']
        print(f"\nTiming (excluding first iteration):")
        print(f"  Forward simulation: {np.mean(t_fwd[1:])*1000:.1f} ms")
        print(f"  Adjoint solve:      {np.mean(t_adj[1:])*1000:.1f} ms")
        print(f"  Total per iter:     {np.mean(t_total[1:])*1000:.1f} ms")

    print(f"\nParameter error (L2 norm):")
    print(f"  Initial: {init_error:.6f}")
    print(f"  Final:   {final_error:.6f}")
    print(f"  Improvement: {(1 - final_error/init_error)*100:.1f}%")

    if final_error < init_error:
        print("\n✓ SUCCESS: Optimization improved parameters!")
    else:
        print("\n✗ WARNING: Optimization did not improve parameters")

except Exception as e:
    print(f"\n✗ ERROR during optimization: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Test Complete")
print("="*80)
