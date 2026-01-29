
"""
Test Adam optimization using the Direct Matrix Gradient formulation.
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
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from src.discrete_adjoint.dae_solver import DAESolver
from debug.dae_matrix_gradient import DAEMatrixGradient

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    solver_cfg = cfg['dae_solver']
    opt_cfg = cfg['optimizer']
    dae_spec_path = solver_cfg['dae_specification_file']
    with open(dae_spec_path, 'r') as f:
        dae_data = json.load(f)
    return dae_data, solver_cfg, opt_cfg

def prepare_loss_targets(sol):
    """Simple extraction of all points."""
    all_t = []
    all_x = []
    for seg in sol.segments:
        if len(seg.t) > 0:
            all_t.append(seg.t)
            all_x.append(seg.x)
    if not all_t:
        return jnp.array([]), jnp.array([])
    return jnp.concatenate([jnp.array(t) for t in all_t]), jnp.concatenate([jnp.array(x) for x in all_x])

def run_optimization_test():
    print("=" * 70)
    print("TEST: Adam Optimization using Matrix Gradient (Direct Struct)")
    print("=" * 70)

    # Load
    config_path = os.path.join(root_dir, 'config/config_bouncing_ball.yaml')
    dae_data, solver_cfg, opt_cfg = load_config(config_path)

    # Optimizer settings
    adam_params = opt_cfg['algorithm']['params']
    ncp = solver_cfg['ncp']
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



    # Setup
    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    
    # Init Solver
    solver = DAESolver(dae_data, verbose=False)
    
    # Ground Truth
    true_p = [p['value'] for p in dae_data['parameters']]
    solver.update_parameters(true_p)
    sol_true = solver.solve_augmented(t_span, ncp=ncp)
    target_times, target_data = prepare_loss_targets(sol_true)
    print(f"Target points: {len(target_times)}")

    # Init Guess
    param_names = [p['name'] for p in dae_data['parameters']]
    opt_param_names = opt_cfg['opt_params']
    opt_indices = [param_names.index(n) for n in opt_param_names]
    
    p_init = list(true_p)
    # Bias
    bias = {'g': 1.0, 'e': 0.1}
    for n, b in bias.items():
        if n in opt_param_names:
            idx = param_names.index(n)
            p_init[idx] += b
            
    print(f"True Params: {dict(zip(param_names, true_p))}")
    print(f"Init Params: {dict(zip(param_names, p_init))}")
    
    # Gradient Computer
    grad_computer = DAEMatrixGradient(dae_data)
    
    # Optimize
    result = grad_computer.optimize_adam(
        solver=solver,
        p_init=p_init,
        opt_param_indices=opt_indices,
        target_times=target_times,
        target_data=target_data,
        t_span=t_span,
        ncp=ncp,
        max_iter=max_iter,
        step_size=step_size,
        blend_sharpness=blend_sharpness,
        tol=tol,
        print_every=print_every,
        soft_interp=True
    )
    
    # Report
    p_opt = result['p_opt']
    print("\nOptimization Finished")
    print(f"Final Params: {dict(zip(param_names, np.asarray(p_opt)))}")
    
    # Plot
    import matplotlib.pyplot as plt
    solver.update_parameters(p_opt)
    sol_opt = solver.solve_augmented(t_span, ncp=ncp)
    
    ts_opt, xs_opt = prepare_loss_targets(sol_opt)
    
    plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)
    
    # Subplot 1: Trajectory
    ax1 = plt.figure(figsize=(15, 10)).add_subplot(gs[0, :])
    # Re-get figure to avoid creating new one
    fig = plt.gcf()

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for i in range(xs_opt.shape[1]):
        color = colors[i % len(colors)]
        label = dae_data['states'][i]['name']
        ax1.plot(ts_opt, xs_opt[:, i], '-', color=color, linewidth=2, label=f'{label} (Opt)')
        ax1.plot(target_times, target_data[:, i], 'x', color=color, markersize=6, alpha=0.7, label=f'{label} (Target)')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('State Value')
    ax1.set_title('Trajectory: Target Data vs Optimized Simulation (Matrix)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Loss History
    ax2 = fig.add_subplot(gs[1, 0])
    # Check if loss history has values
    loss_hist = result['loss_history']
    if any(loss_hist):
        ax2.plot(loss_hist, 'b-', linewidth=2)
        ax2.set_yscale('log')
    else:
        ax2.plot(loss_hist, 'b-', linewidth=2, label='(Not computed)')
        ax2.legend()
        
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss History')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Gradient Norm History
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(result['grad_norm_history'], 'r-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_title('Gradient Norm History')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(current_dir, 'optimization_result_matrix.png')
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    run_optimization_test()
