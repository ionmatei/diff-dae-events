"""
Test Adam optimization on the bouncing ball DAE.

Generates ground-truth data from the default parameters, biases the
optimized parameters, then runs the Adam optimizer to recover them.
"""

import numpy as np
import yaml
import json
import os
import sys

import jax
import jax.numpy as jnp
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_platform_name", "cpu")

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from src.discrete_adjoint.dae_solver import DAESolver
from src.discrete_adjoint.dae_padded_gradient import DAEPaddedGradient
import torch
from src.pytorch.dae_optimizer_pytorch import BouncingBallModel


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    solver_cfg = cfg['dae_solver']
    opt_cfg = cfg['optimizer']
    dae_spec_path = solver_cfg['dae_specification_file']
    with open(dae_spec_path, 'r') as f:
        dae_data = json.load(f)
    return dae_data, solver_cfg, opt_cfg


def prepare_loss_targets(sol, n_x, t_start, t_end):
    """Extract interior target times/data from solution."""
    all_t = []
    all_x = []
    
    for seg in sol.segments:
        if len(seg.t) > 0:
            all_t.append(seg.t)
            all_x.append(seg.x)

    if not all_t:
        return jnp.array([]), jnp.array([])

    target_times = jnp.concatenate([jnp.array(t[:-1]) for t in all_t])
    target_data = jnp.concatenate([jnp.array(x[:-1]) for x in all_x])
    return target_times, target_data


def run_optimization_test():
    print("=" * 70)
    print("TEST: Adam Optimization on Bouncing Ball DAE")
    print("=" * 70)

    # --- 1. Load config ---
    config_path = os.path.join(root_dir, 'config/config_bouncing_ball.yaml')
    dae_data, solver_cfg, opt_cfg = load_config(config_path)

    t_start = solver_cfg['start_time']
    t_stop = solver_cfg['stop_time']
    ncp = solver_cfg['ncp']
    t_span = (t_start, t_stop)

    # Optimizer settings
    adam_params = opt_cfg['algorithm']['params']
    step_size = adam_params['step_size']
    beta1 = adam_params['beta1']
    beta2 = adam_params['beta2']
    epsilon = adam_params['epsilon']
    blend_sharpness = opt_cfg['blend_sharpness']
    max_iter = opt_cfg['max_iterations']
    tol = opt_cfg['tol']
    print_every = opt_cfg['print_every']
    max_blocks = opt_cfg['max_blocks']
    max_pts = opt_cfg['max_points_per_segment']
    max_targets = opt_cfg['max_targets']
    downsample_segments = opt_cfg.get('downsample_segments', False)
    all_segments = opt_cfg.get('all_segments', False)
    sim_top_time = opt_cfg.get('sim_top_time', False)

    # Resolve optimized parameter indices
    param_names = [p['name'] for p in dae_data['parameters']]
    opt_param_names = opt_cfg['opt_params']
    opt_param_indices = [param_names.index(n) for n in opt_param_names]
    print(f"Optimized parameters: {opt_param_names} (indices {opt_param_indices})")
    print(f"Adaptive Horizon (sim_top_time): {sim_top_time}")

    # --- 2. Ground truth ---
    true_p = [p['value'] for p in dae_data['parameters']]
    print(f"True parameters: {dict(zip(param_names, true_p))}")

    solver = DAESolver(dae_data, verbose=False)
    solver.update_parameters(true_p)
    sol_true = solver.solve_augmented(t_span, ncp=ncp)

    n_x = len(dae_data['states'])
    target_times, target_data = prepare_loss_targets(sol_true, n_x, t_start, t_stop)
    delta_t = target_times[1:] - target_times[:-1]
    print(f"Delta t: {jnp.min(delta_t)}")



    
    print(f"Target data points: {len(target_times)}")

    # --- 3. Biased initial guess ---
    bias = {
        'g': -2.0,   # 9.81 -> ~10.61
        'e': -0.15,  # 0.8  -> 0.65
    }
    p_init = list(true_p)
    for name in opt_param_names:
        idx = param_names.index(name)
        p_init[idx] += bias.get(name, 0.0)
    p_init = jnp.array(p_init)
    print(f"Initial (biased) parameters: {dict(zip(param_names, np.asarray(p_init)))}")

    # --- 4. Build gradient computer and run optimizer ---
    grad_computer = DAEPaddedGradient(
        dae_data, max_blocks=max_blocks, max_pts=max_pts, max_targets=max_targets,
        downsample_segments=downsample_segments, all_segments=all_segments
    )

    result = grad_computer.optimize_adam(
        solver=solver,
        p_init=p_init,
        opt_param_indices=opt_param_indices,
        target_times=target_times,
        target_data=target_data,
        t_span=t_span,
        ncp=ncp,
        max_iter=max_iter,
        tol=tol,
        step_size=step_size,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        blend_sharpness=blend_sharpness,
        print_every=print_every,
        adaptive_horizon=sim_top_time
    )

    # --- 5. Report ---
    p_opt = result['p_opt']
    print("\n" + "=" * 70)
    print("Optimization Result")
    print("=" * 70)
    print(f"True params:      {dict(zip(param_names, true_p))}")
    print(f"Initial params:   {dict(zip(param_names, np.asarray(p_init)))}")
    print(f"Optimized params: {dict(zip(param_names, np.asarray(p_opt)))}")
    print(f"Iterations:       {result['n_iter']}")
    print(f"Converged:        {result['converged']}")
    print(f"Final loss:       {result['loss_history'][-1]:.6e}")
    print(f"Final |grad|:     {result['grad_norm_history'][-1]:.6e}")
    if 'avg_iter_time' in result:
        print(f"Avg iter time:    {result['avg_iter_time']:.2f} ms")

    # Per-parameter error
    for name in opt_param_names:
        idx = param_names.index(name)
        err = abs(float(p_opt[idx]) - true_p[idx])
        print(f"  {name}: true={true_p[idx]:.4f}  opt={float(p_opt[idx]):.4f}  err={err:.6e}")

    # --- 6. Plotting ---
    print("\nGenerating plots...")
    import matplotlib.pyplot as plt

    # Run simulation with PyTorch model using optimized parameters
    print("Validating with PyTorch simulator...")
    
    # 1. Instantiate PyTorch model with optimized parameters
    p_opt_dict = dict(zip(param_names, np.asarray(p_opt)))
    
    # Extract params (defaults from dae_data/config if not optimized)
    # Note: BouncingBallModel (singular) uses 'e'
    g_val = float(p_opt_dict.get('g', 9.81))
    e_val = float(p_opt_dict.get('e', 0.8))
    
    # Initial state from DAE data
    h0_val = dae_data['states'][0]['start']
    v0_val = dae_data['states'][1]['start']
    
    pt_model = BouncingBallModel(
        g=g_val, e=e_val, h0=h0_val, v0=v0_val, ncp=ncp
    )
    
    # 2. Compute predictions at exact target times (without interpolation)
    target_times_t = torch.tensor(np.array(target_times), dtype=torch.float64)
    with torch.no_grad():
        y_pred_pt = pt_model.simulate_at_targets(target_times_t)
        
    y_pred_np = y_pred_pt.numpy()
    
    # Calculate and print validation error
    target_data_np = np.array(target_data)
    # y_pred_pt is (N, 2) [h, v], target_data is (N, 2)
    # Compute loss on height only (index 0) to match PyTorch optimizer logic/avoid velocity discontinuities
    val_mse = np.mean((y_pred_np[:, 0] - target_data_np[:, 0])**2)
    print(f"Final Loss (PyTorch, Height only): {val_mse:.6e}")
    
    # 3. Generate dense trajectory for plotting (using PyTorch)
    # Use max_blocks as approximate nbounces
    with torch.no_grad():
        times_sim, h_sim, v_sim, _ = pt_model.simulate(t_stop, nbounces=max_blocks)
        
    sim_t = times_sim.numpy()
    sim_x = np.stack([h_sim.numpy(), v_sim.numpy()], axis=1) # Stack to (N, 2)

    # Prepare figure
    fig = plt.figure(figsize=(14, 10))
    # Grid layout: 2 rows, 2 columns.
    # Top row: State 0 (Height), State 1 (Velocity)
    # Bottom row: Loss, Gradient Norm
    gs = fig.add_gridspec(2, 2)

    # --- Plot State 0 (Height) ---
    ax_h = fig.add_subplot(gs[0, 0])
    i = 0
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    label = dae_data['states'][i]['name']
    color = colors[i % len(colors)]
    
    # 1. Sim
    ax_h.plot(sim_t, sim_x[:, i], color=color, alpha=0.3, label=f'{label} (Sim)')
    # 2. Target
    ax_h.plot(target_times, target_data[:, i], 'x', color=color, markersize=6, alpha=0.5, label=f'{label} (Target)')
    # 3. Interp
    ax_h.plot(target_times, y_pred_np[:, i], '.', color=color, markersize=4, label=f'{label} (Interp)')
    
    ax_h.set_xlabel('Time')
    ax_h.set_ylabel(f'{label}')
    ax_h.set_title(f'State: {label}')
    ax_h.legend()
    ax_h.grid(True, alpha=0.3)

    # --- Plot State 1 (Velocity) ---
    ax_v = fig.add_subplot(gs[0, 1])
    i = 1 
    # Check if state 1 exists (robustness)
    if n_x > 1:
        label = dae_data['states'][i]['name']
        color = colors[i % len(colors)]
        
        # 1. Sim
        ax_v.plot(sim_t, sim_x[:, i], color=color, alpha=0.3, label=f'{label} (Sim)')
        # 2. Target
        ax_v.plot(target_times, target_data[:, i], 'x', color=color, markersize=6, alpha=0.5, label=f'{label} (Target)')
        # 3. Interp
        ax_v.plot(target_times, y_pred_np[:, i], '.', color=color, markersize=4, label=f'{label} (Interp)')
        
        ax_v.set_xlabel('Time')
        ax_v.set_ylabel(f'{label}')
        ax_v.set_title(f'State: {label}')
        ax_v.legend()
        ax_v.grid(True, alpha=0.3)
    else:
        ax_v.text(0.5, 0.5, "No second state", ha='center', va='center')

    # Main Title with Loss
    final_loss = result['loss_history'][-1]
    fig.suptitle(f'Trajectory Optimization Results\nFinal Loss: {final_loss:.6e}', fontsize=16)

    # Subplot 3: Loss History
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(result['loss_history'], 'b-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss History')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Subplot 4: Gradient Norm History
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(result['grad_norm_history'], 'r-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_title('Gradient Norm History')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plot_path = os.path.join(root_dir, 'results', 'optimization_result_jax_bouncing_ball_block.png')
    # Return benchmark results
    benchmark_results = {
        'method': 'jax_block',
        'ncp': ncp,
        'avg_iter_time': result.get('avg_iter_time', 0.0),
        'p_opt': dict(zip(param_names, np.asarray(p_opt))),
        'p_true': dict(zip(param_names, true_p)),
        'final_validation_loss': float(val_mse),
        'iterations': result['n_iter'],
        'converged': bool(result['converged'])
    }
    return benchmark_results

if __name__ == "__main__":
    run_optimization_test()
