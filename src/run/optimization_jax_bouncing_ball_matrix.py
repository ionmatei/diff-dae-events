
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
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from src.discrete_adjoint.dae_solver import DAESolver
from src.discrete_adjoint.dae_matrix_gradient import DAEMatrixGradient
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
    return jnp.concatenate([jnp.array(t[:-1]) for t in all_t]), jnp.concatenate([jnp.array(x[:-1]) for x in all_x])

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
    downsample_segments = opt_cfg.get('downsample_segments', False)
    all_segments = opt_cfg.get('all_segments', False)
    sim_top_time = opt_cfg.get('sim_top_time', False)

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
    print(f"Adaptive Horizon (sim_top_time): {sim_top_time}")
    
    p_init = list(true_p)
    # Bias
    bias = {'g': -2.0, 'e': -0.15}
    for n, b in bias.items():
        if n in opt_param_names:
            idx = param_names.index(n)
            p_init[idx] += b
            
    print(f"True Params: {dict(zip(param_names, true_p))}")
    print(f"Init Params: {dict(zip(param_names, p_init))}")
    
    # Gradient Computer
    grad_computer = DAEMatrixGradient(dae_data, max_pts=max_pts, downsample_segments=downsample_segments, all_segments=all_segments)
    
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
        adaptive_horizon=sim_top_time,
    )
    
    # Report
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
    
    # Plot
    print("\nGenerating plots...")
    import matplotlib.pyplot as plt
    # Run simulation with PyTorch model using optimized parameters
    print("Validating with PyTorch simulator...")
    
    # 1. Instantiate PyTorch model with optimized parameters
    p_opt_dict = dict(zip(param_names, np.asarray(p_opt)))
    
    # Extract params (defaults from dae_data/config if not optimized)
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
    # Compute loss on height only (index 0) to match PyTorch optimizer logic
    val_mse = np.mean((y_pred_np[:, 0] - target_data_np[:, 0])**2)
    print(f"Final Loss (PyTorch, Height only): {val_mse:.6e}")
    
    # 3. Generate dense trajectory for plotting (using PyTorch)
    # Use max_blocks as approximate nbounces
    stop_time = solver_cfg['stop_time']
    with torch.no_grad():
        times_sim, h_sim, v_sim, _ = pt_model.simulate(stop_time, nbounces=max_blocks)
        
    # Note: Matrix plot code doesn't use sim_t/sim_x directly for lines in existing code?
    # Ah, wait. The validatin/plot code in matrix script:
    # "solver.update_parameters(p_opt) ... sol_opt = solver.solve_augmented..."
    # "y_pred = grad_computer.predict_trajectory..."
    # "ax1.plot(target_times, y_pred_np[:, idx], 'b-', label='Opt (Interp)')"
    # It PLOTS the INTERPOLATED points as a LINE ('b-'). 
    # This is slightly different from block script which plots sim_t/sim_x as line AND y_pred as dots.
    # The user instruction was "The final simulation ... should use the Pytorch version".
    # Since matrix script only plots the output at target times (interpolated), 
    # substituting y_pred_np from PyTorch (exact at targets) works perfectly.
    # We DO NOT need the dense simulation here if we just plot y_pred_np against target_times.
    # However, if target_times are sparse, a line plot 'b-' might look jagged.
    # But existing code does: "ax1.plot(target_times, y_pred_np[:, idx], 'b-', ...)"
    # So I will stick to what I have: y_pred_np IS the data to plot.
    pass
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    final_loss = result['loss_history'][-1] if result['loss_history'] else 0.0
    state_names = [s['name'] for s in dae_data['states']]
    n_x = len(state_names)
    
    # Subplot 1: Height (State 0)
    ax1 = fig.add_subplot(gs[0, 0])
    idx = 0
    if idx < n_x:
        ax1.plot(target_times, target_data[:, idx], 'kx', label='Target', alpha=0.6)
        ax1.plot(target_times, y_pred_np[:, idx], 'b-', label='Opt (Interp)')
        ax1.set_title(f"State: {state_names[idx].capitalize()}", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    # Subplot 2: Velocity (State 1)
    ax2 = fig.add_subplot(gs[0, 1])
    idx = 1
    if idx < n_x:
        ax2.plot(target_times, target_data[:, idx], 'kx', label='Target', alpha=0.6)
        ax2.plot(target_times, y_pred_np[:, idx], 'r-', label='Opt (Interp)')
        ax2.set_title(f"State: {state_names[idx].capitalize()}", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
    # Subplot 3: Loss History
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(result['loss_history'], 'b-', linewidth=1.5)
    ax3.set_yscale('log')
    ax3.set_title(f"Loss History (Final: {final_loss:.2e})", fontsize=12)
    ax3.set_xlabel('Iteration')
    ax3.grid(True, alpha=0.3, which='both')

    # Subplot 4: Gradient Norm History
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(result['grad_norm_history'], 'r-', linewidth=1.5)
    ax4.set_yscale('log')
    ax4.set_title("Gradient Norm History", fontsize=12)
    ax4.set_xlabel('Iteration')
    ax4.grid(True, alpha=0.3, which='both')

    plt.suptitle(f"Matrix Optimization Results (Final Loss: {final_loss:.2e})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_path = os.path.join(root_dir, 'results', 'optimization_result_jax_bouncing_ball_matrix.png')
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    run_optimization_test()
