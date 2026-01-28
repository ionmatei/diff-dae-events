"""
Test the Matrix-Based Discrete Adjoint Optimizer on the Bouncing Ball example.

The bouncing ball has:
- States: h (height), v (velocity)
- Parameters: g (gravity), e (restitution coefficient)
- Event: when h < 0, reinit v = -e * prev(v)

We optimize parameters to match observed trajectory using full matrix gradients.
Configuration is loaded from YAML file.
"""

import os
import argparse
import yaml
import numpy as np
import json
import time

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
    
    # Verify
    import jax
    print(f"JAX Platform: {jax.default_backend()}")
    return device


def run_bouncing_ball_test(config: dict):
    print("=" * 80)
    print("Bouncing Ball - Matrix-Based Discrete Adjoint Test")
    print("=" * 80)
    
    # Delayed import to ensure JAX finds the correct platform env var
    from src.discrete_adjoint.dae_solver import DAESolver
    from src.discrete_adjoint.dae_optimizer_matrix import DAEOptimizerMatrix

    # Extract config sections
    solver_cfg = config['dae_solver']
    opt_cfg = config['optimizer']

    # Load DAE specification
    json_path = solver_cfg['dae_specification_file']
    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    print(f"\nLoaded DAE from: {json_path}")
    print(f"  States: {[s['name'] for s in dae_data['states']]}")
    print(f"  Parameters: {[p['name'] for p in dae_data['parameters']]}")
    print(f"  Events: {len(dae_data.get('when', []))}")

    # True parameters
    p_true = {p['name']: p['value'] for p in dae_data['parameters']}
    print(f"\nTrue parameters: g={p_true['g']}, e={p_true['e']}")

    # =========================================================================
    # Step 1: Generate reference trajectory with true parameters
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 1: Generate Reference Trajectory")
    print("-" * 40)

    solver_true = DAESolver(dae_data, verbose=False)
    
    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    ncp = solver_cfg.get('ncp', 10)
    
    aug_sol_true = solver_true.solve_augmented(t_span=t_span, ncp=ncp)

    print(f"  Simulation time: {t_span}")
    print(f"  Number of segments: {len(aug_sol_true.segments)}")
    print(f"  Number of events: {len(aug_sol_true.events)}")

    # Extract reference data by concatenating segments
    t_target_list = []
    y_target_list = []
    
    for seg in aug_sol_true.segments:
        t_target_list.append(seg.t)
        y_target_list.append(seg.x)
        
    t_target = np.concatenate(t_target_list)
    y_target = np.concatenate(y_target_list)
    
    n_targets = len(t_target)

    # Add small noise to ground truth
    noise_std = 0.0
    np.random.seed(42)
    y_target += np.random.normal(0, noise_std, y_target.shape)
    print(f"  Added Gaussian noise (std={noise_std}) to targets")

    print(f"  Target times: {n_targets} points from {t_target[0]:.2f} to {t_target[-1]:.2f}")
    print(f"  Event times: {[ev.t_event for ev in aug_sol_true.events]}")

    # =========================================================================
    # Step 2: Create initial guess with perturbed parameter
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 2: Create Perturbed Initial Guess")
    print("-" * 40)

    # Perturb parameters
    dae_data_init = json.loads(json.dumps(dae_data))  # Deep copy
    
    optimize_params = opt_cfg['opt_params']
    
    for p in dae_data_init['parameters']:
        p_name = p['name']
        if p_name in optimize_params:
            if p_name == 'g':
                p['value'] = p_true['g'] * 0.9  # 10% perturbation
            elif p_name == 'e':
                p['value'] = p_true['e'] * 0.9  # 10% perturbation
                
    # Print status
    for p_name in ['g', 'e']:
        val_true = p_true[p_name]
        val_init = next(p['value'] for p in dae_data_init['parameters'] if p['name'] == p_name)
        is_opt = p_name in optimize_params
        status = "PERTURBED" if is_opt else "FIXED"
        pct_diff = 100 * (val_init / val_true - 1)
        print(f"  {p_name}: True={val_true}, Init={val_init:.4f} ({pct_diff:+.0f}%) -> {status}")

    # =========================================================================
    # Step 3: Create optimizer and run optimization
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 3: Run Optimization (Matrix-Based Adjoint)")
    print("-" * 40)

    optimizer = DAEOptimizerMatrix(
        dae_data=dae_data_init,
        solver_config=solver_cfg,
        verbose=True
    )

    result = optimizer.optimize(
        target_times=t_target,
        target_data=y_target,
        optimize_params=optimize_params,
        learning_rate=opt_cfg.get('learning_rate', 0.01),
        max_iterations=opt_cfg['max_iterations'],
        tol=opt_cfg['tol'],
        ncp=ncp
    )

    # =========================================================================
    # Step 4: Print results
    # =========================================================================
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)

    print(f"\nConverged: {result['success']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final Loss: {result['history']['loss'][-1]:.6e}")
    print(f"Final Gradient Norm: {result['history']['grad_norm'][-1]:.6e}")

    print("\nParameter Estimates:")
    for i, name in enumerate(result['param_names']):
        val_est = result['params'][i]
        val_true = p_true[name]
        error = abs(val_est - val_true)
        pct_error = 100 * error / val_true
        print(f"  {name}: Estimated={val_est:.6f}, True={val_true:.6f}, Error={error:.6e} ({pct_error:.2f}%)")

    # Print convergence history summary
    print("\nConvergence History:")
    print(f"  Iteration 0: Loss={result['history']['loss'][0]:.6e}")
    if len(result['history']['loss']) > 1:
        mid_idx = len(result['history']['loss']) // 2
        print(f"  Iteration {mid_idx}: Loss={result['history']['loss'][mid_idx]:.6e}")
    print(f"  Final: Loss={result['history']['loss'][-1]:.6e}")

    # Optional: Save results
    if opt_cfg.get('save_results', False):
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/matrix_optimizer_results.npz"
        np.savez(
            output_file,
            params=result['params'],
            param_names=result['param_names'],
            loss_history=result['history']['loss'],
            grad_norm_history=result['history']['grad_norm'],
            time_per_iter=result['history']['time_per_iter']
        )
        print(f"\nResults saved to: {output_file}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Matrix-Based Optimizer Test")
    parser.add_argument('--config', type=str, default='config/config_bouncing_ball.yaml',
                        help='Path to configuration YAML file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup device
    setup_jax_device(config)

    # Run test
    result = run_bouncing_ball_test(config)

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
