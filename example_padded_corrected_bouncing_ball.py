"""
Test the Corrected Explicit Discrete Adjoint Optimizer on the Bouncing Ball example.

Uses DAEOptimizerPaddedCorrected which implements:
1. Total Jacobian for Coupled DAEs
2. Time-dependent reset map sensitivity
3. Consistency Adjoint

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
    
    import jax
    print(f"JAX Platform: {jax.default_backend()}")
    return device

def run_bouncing_ball_test(config: dict):
    print("=" * 80)
    print("Bouncing Ball - CORRECTED Adjoint Test")
    print("=" * 80)
    
    from src.discrete_adjoint.dae_solver import DAESolver
    from src.discrete_adjoint.dae_optimizer_padded_corrected import DAEOptimizerPaddedCorrected

    # Extract config sections
    solver_cfg = config['dae_solver']
    opt_cfg = config['optimizer']
    algo_cfg = opt_cfg.get('algorithm', {})

    # Load DAE specification
    json_path = solver_cfg['dae_specification_file']
    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    print(f"\\nLoaded DAE from: {json_path}")
    print(f"  States: {[s['name'] for s in dae_data['states']]}")
    print(f"  Parameters: {[p['name'] for p in dae_data['parameters']]}")
    print(f"  Events: {len(dae_data.get('when', []))}")

    # True parameters
    p_true = {p['name']: p['value'] for p in dae_data['parameters']}
    print(f"\\nTrue parameters: g={p_true['g']}, e={p_true['e']}")

    # =========================================================================
    # Step 1: Generate reference trajectory with true parameters
    # =========================================================================
    print("\\n" + "-" * 40)
    print("Step 1: Generate Reference Trajectory")
    print("-" * 40)

    solver_true = DAESolver(dae_data, verbose=False)
    
    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    ncp = solver_cfg['ncp']
    
    aug_sol_true = solver_true.solve_augmented(t_span=t_span, ncp=ncp)

    print(f"  Simulation time: {t_span}")
    print(f"  Number of segments: {len(aug_sol_true.segments)}")
    print(f"  Number of events: {len(aug_sol_true.events)}")

    t_target_list = []
    y_target_list = []
    
    for seg in aug_sol_true.segments:
        t_target_list.append(seg.t)
        # For bouncing ball example, seg.x contains [h, v]
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

    # =========================================================================
    # Step 2: Create initial guess with perturbed parameter
    # =========================================================================
    print("\\n" + "-" * 40)
    print("Step 2: Create Perturbed Initial Guess")
    print("-" * 40)

    dae_data_init = json.loads(json.dumps(dae_data))  # Deep copy
    
    g_true_val = p_true['g']
    e_true_val = p_true['e']
    g_init = g_true_val * 0.8  # 10% perturbation
    e_init = e_true_val * 0.8   # 10% perturbation

    optimize_params = opt_cfg['opt_params']
    
    for p in dae_data_init['parameters']:
        p_name = p['name']
        if p_name in optimize_params:
            if p_name == 'g':
                p['value'] = g_true_val * 0.8
            elif p_name == 'e':
                p['value'] = e_true_val * 0.8
                
    for p_name in ['g', 'e']:
        val_true = p_true[p_name]
        val_init = next(p['value'] for p in dae_data_init['parameters'] if p['name'] == p_name)
        is_opt = p_name in optimize_params
        status = "PERTURBED" if is_opt else "FIXED (Default)"
        pct_diff = 100 * (val_init / val_true - 1)
        print(f"  {p_name}: True={val_true}, Init={val_init:.4f} ({pct_diff:+.0f}%) -> {status}")

    # =========================================================================
    # Step 3: Create optimizer and run optimization
    # =========================================================================
    print("\\n" + "-" * 40)
    print("Step 3: Run Optimization (Corrected Algorithm)")
    print("-" * 40)

    optimizer = DAEOptimizerPaddedCorrected(
        dae_data=dae_data_init,
        optimize_params=opt_cfg['opt_params'],
        blend_sharpness=opt_cfg.get('blend_sharpness', 100.0),
        max_segments=opt_cfg.get('max_segments', 20),
        ncp=ncp,
        safety_buffer_pct=opt_cfg.get('safety_buffer_pct', 1.2),
        prediction_method='linear', # Only linear supported by corrected opt for now
        verbose=True
    )

    step_size = algo_cfg.get('params', {}).get('step_size', 0.05)
    algorithm_type = algo_cfg.get('type', 'adam').lower()

    result = optimizer.optimize(
        t_span=t_span,
        target_times=t_target,
        target_outputs=y_target,
        max_iterations=opt_cfg['max_iterations'],
        step_size=step_size,
        tol=opt_cfg['tol'],
        ncp=ncp
    )

    # =========================================================================
    # Step 4: Results
    # =========================================================================
    print("\\n" + "-" * 40)
    print("Step 4: Results")
    print("-" * 40)

    params_final = result['params']
    p_opt_dict = {}
    for i, name in enumerate(opt_cfg['opt_params']):
        p_opt_dict[name] = params_final[i]
    
    if 'g' in p_opt_dict and 'e' in p_opt_dict:
        g_opt = p_opt_dict['g']
        e_opt = p_opt_dict['e']
        g_error_pct = 100 * abs(g_opt - g_true_val) / g_true_val
        e_error_pct = 100 * abs(e_opt - e_true_val) / e_true_val

        print(f"\\n  Parameter Recovery:")
        print(f"    True g:      {g_true_val:.6f}, True e:      {e_true_val:.6f}")
        print(f"    Initial g:   {g_init:.6f}, Initial e:   {e_init:.6f}")
        print(f"    Optimized g: {g_opt:.6f}, Optimized e: {e_opt:.6f}")
        print(f"    Error g:     {g_error_pct:.2f}%, Error e:     {e_error_pct:.2f}%")

    print(f"\\n  Optimization Stats:")
    print(f"    Initial loss: {result['history']['loss'][0]:.6e}")
    print(f"    Final loss:   {result['history']['loss'][-1]:.6e}")
    print(f"    Converged:    {result['converged']}")

    # =========================================================================
    # Step 5: Validate by re-simulating
    # =========================================================================
    print("\\n" + "-" * 40)
    print("Step 5: Validation")
    print("-" * 40)

    dae_data_opt = json.loads(json.dumps(dae_data))
    for p in dae_data_opt['parameters']:
        p_name = p['name']
        if p_name in p_opt_dict:
            p['value'] = float(p_opt_dict[p_name])

    solver_opt = DAESolver(dae_data_opt, verbose=False)
    aug_sol_opt = solver_opt.solve_augmented(t_span=t_span, ncp=ncp)
    
    # Simple relative error check
    y_opt = []
    for seg in aug_sol_opt.segments:
        y_opt.append(seg.x)
    y_opt = np.concatenate(y_opt) if y_opt else np.array([])
    
    # Truncate to matching length to be safe (though normally they match if ncp same)
    min_len = min(len(y_opt), len(y_target))
    traj_error = np.linalg.norm(y_opt[:min_len] - y_target[:min_len]) / np.linalg.norm(y_target[:min_len])

    print(f"  Trajectory relative error: {traj_error:.6e}")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Trajectory
        ax = axes[0, 0]
        t_true, x_true = extract_state_trajectory(aug_sol_true, 0) # h
        t_opt, x_opt = extract_state_trajectory(aug_sol_opt, 0)
        ax.plot(t_true, x_true, 'b-', label='True')
        ax.plot(t_opt, x_opt, 'r--', label='Optimized')
        ax.set_title('Height Trajectory')
        ax.legend()
        
        # Loss
        ax = axes[1, 0]
        ax.semilogy(result['history']['loss'])
        ax.set_title('Loss History')
        
        plt.tight_layout()
        plt.savefig('bouncing_ball_corrected_result.png')
        print("\\n  Plot saved to: bouncing_ball_corrected_result.png")
    except ImportError:
        pass

    print("\\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
    return result

def extract_state_trajectory(aug_sol, state_idx):
    t_all = []
    x_all = []
    for seg in aug_sol.segments:
        t_all.extend(seg.t.tolist())
        x_all.extend(seg.x[:, state_idx].tolist())
    return np.array(t_all), np.array(x_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bouncing Ball Corrected Adjoint Test")
    parser.add_argument('--config', '-c', type=str, default='config/config_bouncing_ball.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_jax_device(config)
    run_bouncing_ball_test(config)
