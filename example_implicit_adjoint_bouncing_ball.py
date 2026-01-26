"""
Test the Explicit Discrete Adjoint Optimizer on the Bouncing Ball example.

The bouncing ball has:
- States: h (height), v (velocity)
- Parameters: g (gravity), e (restitution coefficient)
- Event: when h < 0, reinit v = -e * prev(v)

We optimize the restitution coefficient e to match observed trajectory.
Configuration is loaded from YAML file.
"""

import os
import argparse
import yaml

# Set device before importing JAX
# Removed hardcoded 'cpu' setting to allow config-driven device selection

import numpy as np
import json
import time

# Imports of DAESolver and DAEOptimizerExplicitAdjoint are moved inside run_bouncing_ball_test 
# to ensure JAX environment variables are set from config BEFORE JAX is imported/initialized.


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
    print("Bouncing Ball - Explicit Discrete Adjoint Test (Config Driven)")
    print("=" * 80)
    
    # Delayed import to ensure JAX finds the correct platform env var
    from src.discrete_adjoint.dae_solver import DAESolver
    from src.discrete_adjoint.dae_optimizer_implicit_adjoint import DAEOptimizerImplicitAdjoint

    # Extract config sections
    solver_cfg = config['dae_solver']
    opt_cfg = config['optimizer']
    algo_cfg = opt_cfg.get('algorithm', {})

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
    ncp = solver_cfg['ncp']
    
    aug_sol_true = solver_true.solve_augmented(t_span=t_span, ncp=ncp)

    print(f"  Simulation time: {t_span}")
    print(f"  Number of segments: {len(aug_sol_true.segments)}")
    print(f"  Number of events: {len(aug_sol_true.events)}")

    # Extract reference data by concatenating segments directly
    # This avoids interpolation and uses the exact points from the solver
    t_target_list = []
    y_target_list = []
    
    # Identify output indices (assuming standard state indices for now)
    # The bouncing ball 'h' and 'v' are states 0 and 1.
    
    for seg in aug_sol_true.segments:
        t_target_list.append(seg.t)
        # seg.x is (n_points, n_states), output is usually just states here
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
    
    # We apply the same perturbation logic as before for consistency in this example,
    # or we could move this to config. For now, we stick to the 15% / 20% perturbation
    # to maintain the integrity of the specific test case unless specified otherwise.
    g_true_val = p_true['g']
    e_true_val = p_true['e']
    g_init = g_true_val * 0.8  # 10% perturbation
    e_init = e_true_val * 0.8   # 10% perturbation

    # Only perturb parameters that are in the optimization list
    optimize_params = opt_cfg['opt_params']
    
    for p in dae_data_init['parameters']:
        p_name = p['name']
        if p_name in optimize_params:
            if p_name == 'g':
                p['value'] = g_true_val * 0.8  # 10% perturbation
            elif p_name == 'e':
                p['value'] = e_true_val * 0.8   # 10% perturbation
                
    # Print status
    for p_name in ['g', 'e']:
        val_true = p_true[p_name]
        # Find initialized value
        val_init = next(p['value'] for p in dae_data_init['parameters'] if p['name'] == p_name)
        is_opt = p_name in optimize_params
        status = "PERTURBED" if is_opt else "FIXED (Default)"
        pct_diff = 100 * (val_init / val_true - 1)
        print(f"  {p_name}: True={val_true}, Init={val_init:.4f} ({pct_diff:+.0f}%) -> {status}")



    # =========================================================================
    # Step 3: Create optimizer and run optimization
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 3: Run Optimization")
    print("-" * 40)

    optimizer = DAEOptimizerImplicitAdjoint(
        dae_data=dae_data_init,
        optimize_params=opt_cfg['opt_params'],
        blend_sharpness=opt_cfg.get('blend_sharpness', 100.0),
        max_segments=opt_cfg.get('max_segments', 20),
        max_points_per_seg=opt_cfg.get('max_points_per_segment', 500),
        prediction_method=opt_cfg.get('prediction_method', 'sigmoid'),
        verbose=True
    )

    # Determine step size logic
    step_size = algo_cfg.get('params', {}).get('step_size', 0.05)
    algorithm_type = algo_cfg.get('type', 'adam').lower()

    result = optimizer.optimize(
        t_span=t_span,
        target_times=t_target,
        target_outputs=y_target,
        max_iterations=opt_cfg['max_iterations'],
        step_size=step_size,
        tol=opt_cfg['tol'],
        ncp=ncp,
        print_every=opt_cfg.get('print_every', 10),
        algorithm=algorithm_type,
        blend_sharpness=opt_cfg.get('blend_sharpness', 100.0),
        max_segments=opt_cfg.get('max_segments', 20),
        max_points_per_seg=opt_cfg.get('max_points_per_segment', 500)
    )

    # =========================================================================
    # Step 4: Results
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 4: Results")
    print("-" * 40)

    # Extract optimized values assuming order matches opt_params ['g', 'e']
    # DAEOptimizerExplicitAdjoint stores params in order of optimize_params list
    params_final = result['params']
    
    # Map back to names for display
    # Assuming opt_params are ['g', 'e']
    p_opt_dict = {}
    for i, name in enumerate(opt_cfg['opt_params']):
        p_opt_dict[name] = params_final[i]
    
    if 'g' in p_opt_dict and 'e' in p_opt_dict:
        g_opt = p_opt_dict['g']
        e_opt = p_opt_dict['e']
        g_error_pct = 100 * abs(g_opt - g_true_val) / g_true_val
        e_error_pct = 100 * abs(e_opt - e_true_val) / e_true_val

        print(f"\n  Parameter Recovery:")
        print(f"    True g:      {g_true_val:.6f}, True e:      {e_true_val:.6f}")
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
    dae_data_opt = json.loads(json.dumps(dae_data))
    for p in dae_data_opt['parameters']:
        p_name = p['name']
        if p_name in p_opt_dict:
            p['value'] = float(p_opt_dict[p_name])

    solver_opt = DAESolver(dae_data_opt, verbose=False)
    aug_sol_opt = solver_opt.solve_augmented(t_span=t_span, ncp=ncp)

    # Use same prediction method as optimizer for consistent comparison
    optimizer_val = DAEOptimizerImplicitAdjoint(
        dae_data=dae_data_opt,
        optimize_params=opt_cfg['opt_params'],
        verbose=False,
        blend_sharpness=opt_cfg.get('blend_sharpness', 100.0),
        max_segments=opt_cfg.get('max_segments', 20),
        max_points_per_seg=opt_cfg.get('max_points_per_segment', 500),
        prediction_method=opt_cfg.get('prediction_method', 'sigmoid')
    )
    y_opt = optimizer_val.predict_outputs(aug_sol_opt, t_target)
    traj_error = np.linalg.norm(y_opt - y_target) / np.linalg.norm(y_target)

    print(f"  Trajectory relative error: {traj_error:.6e}")

    # =========================================================================
    # Step 6: Plot results
    # =========================================================================
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot trajectories
        ax = axes[0, 0]
        t_true_all, h_true_all = extract_state_trajectory(aug_sol_true, 0)
        t_opt_all, h_opt_all = extract_state_trajectory(aug_sol_opt, 0)
        ax.plot(t_true_all, h_true_all, 'b-', linewidth=2, label='True')
        ax.plot(t_opt_all, h_opt_all, 'r--', linewidth=2, label='Optimized')
        # Note: y_target uses sigmoid blending for smooth optimization gradients,
        # so we interpolate directly for visualization to match the trajectory lines
        h_target_interp = np.interp(t_target, t_true_all, h_true_all)
        # ax.scatter(t_target, h_target_interp, c='k', s=20, zorder=5, label='Target times')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Height h [m]')
        ax.set_title('Height Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot velocity
        ax = axes[0, 1]
        _, v_true_all = extract_state_trajectory(aug_sol_true, 1)
        _, v_opt_all = extract_state_trajectory(aug_sol_opt, 1)
        ax.plot(t_true_all, v_true_all, 'b-', linewidth=2, label='True')
        ax.plot(t_opt_all, v_opt_all, 'r--', linewidth=2, label='Optimized')
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
        plt.savefig('bouncing_ball_explicit_adjoint_result.png', dpi=150)
        print("\n  Plot saved to: bouncing_ball_explicit_adjoint_result.png")
        plt.show() # Disabled for headless run

    except ImportError:
        print("\n  Matplotlib not available - skipping plots")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

    return result


def extract_state_trajectory(aug_sol, state_idx):
    """Extract full state trajectory from augmented solution."""
    t_all = []
    x_all = []

    for seg in aug_sol.segments:
        t_all.extend(seg.t.tolist())
        x_all.extend(seg.x[:, state_idx].tolist())

    return np.array(t_all), np.array(x_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bouncing Ball Explicit Adjoint Test")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config_bouncing_ball.yaml',
        help='Path to configuration YAML file'
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_jax_device(config)
    
    run_bouncing_ball_test(config)
