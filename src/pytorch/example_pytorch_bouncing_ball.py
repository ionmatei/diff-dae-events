"""
Test the PyTorch DAE Optimizer on the Bouncing Ball example.

The bouncing ball has:
- States: h (height), v (velocity)
- Parameters: g (gravity), e (restitution coefficient)
- Event: when h < 0, reinit v = -e * prev(v)

We optimize the restitution coefficient e and gravity g to match observed trajectory.
Configuration is loaded from YAML file.
"""

import os
import argparse
import yaml
import numpy as np
import json
import time
import torch

torch.set_default_dtype(torch.float64)

from dae_optimizer_pytorch import BouncingBallModel, DAEOptimizerPyTorch


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_bouncing_ball_test(config: dict):
    print("=" * 80)
    print("Bouncing Ball - PyTorch Optimizer Test")
    print("=" * 80)

    # Extract config sections
    solver_cfg = config['dae_solver']
    opt_cfg = config['optimizer']
    algo_cfg = opt_cfg.get('algorithm', {})

    # Load DAE specification for true parameter values
    json_path = solver_cfg['dae_specification_file']
    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    print(f"\nLoaded DAE from: {json_path}")
    print(f"  States: {[s['name'] for s in dae_data['states']]}")
    print(f"  Parameters: {[p['name'] for p in dae_data['parameters']]}")

    # True parameters
    p_true = {p['name']: p['value'] for p in dae_data['parameters']}
    g_true = p_true['g']
    e_true = p_true['e']
    print(f"\nTrue parameters: g={g_true}, e={e_true}")

    # Initial conditions from DAE spec
    h0 = dae_data['states'][0]['start']  # height
    v0 = dae_data['states'][1]['start']  # velocity

    # =========================================================================
    # Step 1: Generate reference trajectory with true parameters
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 1: Generate Reference Trajectory")
    print("-" * 40)

    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    nbounces = opt_cfg.get('max_segments', 10)
    ncp = solver_cfg.get('ncp', 150)  # Number of collocation points per segment

    # Create model with true parameters
    model_true = BouncingBallModel(g=g_true, e=e_true, h0=h0, v0=v0, ncp=ncp)

    with torch.no_grad():
        times_true, h_true, v_true, event_times_true = model_true.simulate(
            t_span[1], nbounces=nbounces
        )

    print(f"  Simulation time: {t_span}")
    print(f"  Number of bounces: {len(event_times_true)}")
    if event_times_true:
        print(f"  Event times: {[float(et) for et in event_times_true]}")

    # Extract reference data at uniform times
    t_duration = t_span[1] - t_span[0]
    n_targets = max(300, int(300 * t_duration))
    t_target = np.linspace(t_span[0] + 0.1, t_span[1] - 0.1, n_targets)

    # Interpolate true trajectory at target times
    times_np = times_true.numpy()
    h_np = h_true.numpy()
    y_target = np.interp(t_target, times_np, h_np)

    # Add small noise to ground truth
    noise_std = 0.0
    np.random.seed(42)
    y_target += np.random.normal(0, noise_std, y_target.shape)
    print(f"  Added Gaussian noise (std={noise_std}) to targets")
    print(f"  Target times: {n_targets} points from {t_target[0]:.2f} to {t_target[-1]:.2f}")

    # =========================================================================
    # Step 2: Create initial guess with perturbed parameter
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 2: Create Perturbed Initial Guess")
    print("-" * 40)

    # Perturb parameters
    g_init = g_true * 0.8  # 20% perturbation
    e_init = e_true * 0.8  # 20% perturbation

    optimize_params = opt_cfg['opt_params']

    for p_name in ['g', 'e']:
        val_true = p_true[p_name]
        val_init = g_init if p_name == 'g' else e_init
        is_opt = p_name in optimize_params
        status = "PERTURBED" if is_opt else "FIXED"
        pct_diff = 100 * (val_init / val_true - 1)
        print(f"  {p_name}: True={val_true}, Init={val_init:.4f} ({pct_diff:+.0f}%) -> {status}")

    # =========================================================================
    # Step 3: Create optimizer and run optimization
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 3: Run Optimization")
    print("-" * 40)

    # Create model with perturbed parameters
    model_opt = BouncingBallModel(g=g_init, e=e_init, h0=h0, v0=v0, ncp=ncp)

    # Create optimizer
    optimizer = DAEOptimizerPyTorch(
        model=model_opt,
        optimize_params=optimize_params,
        verbose=True,
        nbounces=nbounces
    )

    # Optimization settings from config
    algo_params = algo_cfg.get('params', {})
    step_size = algo_params.get('step_size', 0.05)
    algorithm_type = algo_cfg.get('type', 'adam').lower()
    beta1 = algo_params.get('beta1', 0.9)
    beta2 = algo_params.get('beta2', 0.999)
    epsilon = algo_params.get('epsilon', 1e-8)

    result = optimizer.optimize(
        t_span=t_span,
        target_times=t_target,
        target_outputs=y_target,
        max_iterations=opt_cfg['max_iterations'],
        step_size=step_size,
        tol=opt_cfg['tol'],
        print_every=opt_cfg.get('print_every', 10),
        algorithm=algorithm_type,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    )

    # =========================================================================
    # Step 4: Results
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 4: Results")
    print("-" * 40)

    # Extract optimized values
    params_final = result['params']

    # Map back to names
    p_opt_dict = {}
    for i, name in enumerate(optimize_params):
        p_opt_dict[name] = params_final[i]

    if 'g' in p_opt_dict and 'e' in p_opt_dict:
        g_opt = p_opt_dict['g']
        e_opt = p_opt_dict['e']
        g_error_pct = 100 * abs(g_opt - g_true) / g_true
        e_error_pct = 100 * abs(e_opt - e_true) / e_true

        print(f"\n  Parameter Recovery:")
        print(f"    True g:      {g_true:.6f}, True e:      {e_true:.6f}")
        print(f"    Initial g:   {g_init:.6f}, Initial e:   {e_init:.6f}")
        print(f"    Optimized g: {g_opt:.6f}, Optimized e: {e_opt:.6f}")
        print(f"    Error g:     {g_error_pct:.2f}%, Error e:     {e_error_pct:.2f}%")

    print(f"\n  Optimization Stats:")
    print(f"    Initial loss: {result['history']['loss'][0]:.6e}")
    print(f"    Final loss:   {result['history']['loss'][-1]:.6e}")
    print(f"    Converged:    {result['converged']}")
    print(f"    Time:         {result['elapsed_time']:.2f}s")

    # =========================================================================
    # Step 5: Validate by re-simulating
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 5: Validation")
    print("-" * 40)

    # Simulate with optimized parameters
    with torch.no_grad():
        times_opt, h_opt, v_opt, event_times_opt = model_opt.simulate(
            t_span[1], nbounces=nbounces
        )

    # Interpolate at target times
    times_opt_np = times_opt.numpy()
    h_opt_np = h_opt.numpy()
    y_opt = np.interp(t_target, times_opt_np, h_opt_np)

    traj_error = np.linalg.norm(y_opt - y_target) / np.linalg.norm(y_target)
    print(f"  Trajectory relative error: {traj_error:.6e}")

    # =========================================================================
    # Step 6: Plot results
    # =========================================================================
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot height trajectories
        ax = axes[0, 0]
        ax.plot(times_np, h_np, 'b-', linewidth=2, label='True')
        ax.plot(times_opt_np, h_opt_np, 'r--', linewidth=2, label='Optimized')
        # ax.scatter(t_target, y_target, c='k', s=20, zorder=5, label='Target points')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Height h [m]')
        ax.set_title('Height Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot velocity trajectories
        ax = axes[0, 1]
        v_true_np = v_true.numpy()
        v_opt_np = v_opt.numpy()
        ax.plot(times_np, v_true_np, 'b-', linewidth=2, label='True')
        ax.plot(times_opt_np, v_opt_np, 'r--', linewidth=2, label='Optimized')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity v [m/s]')
        ax.set_title('Velocity Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Loss history
        ax = axes[1, 0]
        ax.semilogy(result['history']['loss'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Loss History')
        ax.grid(True, alpha=0.3)

        # Gradient norm history
        ax = axes[1, 1]
        ax.semilogy(result['history']['gradient_norm'], 'r-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm History')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('bouncing_ball_pytorch_result.png', dpi=150)
        print("\n  Plot saved to: bouncing_ball_pytorch_result.png")
        plt.show()

    except ImportError:
        print("\n  Matplotlib not available - skipping plots")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bouncing Ball PyTorch Optimizer Test")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config_bouncing_ball.yaml',
        help='Path to configuration YAML file'
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_bouncing_ball_test(config)
