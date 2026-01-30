"""
Test Adam optimization on the bouncing balls DAE (3 balls, multiple event sources).

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
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from src.discrete_adjoint.dae_solver import DAESolver
from dae_padded_gradient import DAEPaddedGradient


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
    print("TEST: Adam Optimization on Bouncing Balls DAE (3 balls)")
    print("=" * 70)

    # --- 1. Load config ---
    config_path = os.path.join(root_dir, 'config/config_bouncing_balls.yaml')
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

    # Resolve optimized parameter indices
    param_names = [p['name'] for p in dae_data['parameters']]
    opt_param_names = opt_cfg['opt_params']
    opt_param_indices = [param_names.index(n) for n in opt_param_names]
    print(f"Optimized parameters: {opt_param_names} (indices {opt_param_indices})")

    # --- 2. Ground truth ---
    true_p = [p['value'] for p in dae_data['parameters']]
    print(f"True parameters: {dict(zip(param_names, true_p))}")

    solver = DAESolver(dae_data, verbose=False)
    solver.update_parameters(true_p)
    sol_true = solver.solve_augmented(t_span, ncp=ncp)

    n_x = len(dae_data['states'])
    target_times, target_data = prepare_loss_targets(sol_true, n_x, t_start, t_stop)
    delta_t = target_times[1:] - target_times[:-1]
    print(f"Delta t min: {jnp.min(delta_t)}")
    print(f"Number of segments: {len(sol_true.segments)}")
    print(f"Target data points: {len(target_times)}")

    # --- 3. Biased initial guess ---
    bias = {
        'g': -0.00,
        'e_g': 0.1,  # 0.8 -> 0.65
        'e_b': 0.1,  # 0.9 -> 0.75
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
        downsample_segments=downsample_segments
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
        adaptive_horizon=False
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

    # Per-parameter error
    for name in opt_param_names:
        idx = param_names.index(name)
        err = abs(float(p_opt[idx]) - true_p[idx])
        print(f"  {name}: true={true_p[idx]:.4f}  opt={float(p_opt[idx]):.4f}  err={err:.6e}")

    # --- 6. Plotting ---
    print("\nGenerating plots...")
    import matplotlib.pyplot as plt

    # Run solver with optimized parameters
    solver.update_parameters(p_opt)
    sol_opt = solver.solve_augmented(t_span, ncp=ncp)

    # Flatten simulated data for plotting (concatenating segments)
    sim_t = []
    sim_x = []
    for seg in sol_opt.segments:
        sim_t.append(seg.t)
        sim_x.append(seg.x)
    sim_t = np.concatenate(sim_t)
    sim_x = np.concatenate(sim_x)

    # Interpolated predictions
    print(f"Interpolating optimized solution at {len(target_times)} target points...")
    y_pred = grad_computer.predict_trajectory(
        sol_opt, target_times, blend_sharpness=blend_sharpness
    )
    y_pred_np = np.asarray(y_pred)

    # State name -> index mapping
    state_names = [s['name'] for s in dae_data['states']]

    # Ball definitions: (label, x_idx, y_idx)
    balls = [
        ('Ball 1', state_names.index('x1'), state_names.index('y1')),
        ('Ball 2', state_names.index('x2'), state_names.index('y2')),
        ('Ball 3', state_names.index('x3'), state_names.index('y3')),
    ]
    ball_colors = ['b', 'r', 'g']

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 3)

    # --- Top row: x-y trajectory for each ball ---
    for col, (ball_label, xi, yi) in enumerate(balls):
        ax = fig.add_subplot(gs[0, col])
        c = ball_colors[col]

        # Simulated trajectory
        ax.plot(sim_x[:, xi], sim_x[:, yi], '-', color=c, alpha=0.3, label='Sim (opt)')
        # Target data
        ax.plot(np.asarray(target_data[:, xi]), np.asarray(target_data[:, yi]),
                'x', color=c, markersize=4, alpha=0.5, label='Target')
        # Interpolated prediction
        ax.plot(y_pred_np[:, xi], y_pred_np[:, yi],
                '.', color=c, markersize=2, label='Interp')

        ax.set_xlabel(state_names[xi])
        ax.set_ylabel(state_names[yi])
        ax.set_title(ball_label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')

    # --- Bottom left: Loss History ---
    ax_loss = fig.add_subplot(gs[1, 0])
    ax_loss.plot(result['loss_history'], 'b-', linewidth=2)
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Loss History')
    ax_loss.set_yscale('log')
    ax_loss.grid(True, alpha=0.3)

    # --- Bottom center: Gradient Norm History ---
    ax_grad = fig.add_subplot(gs[1, 1])
    ax_grad.plot(result['grad_norm_history'], 'r-', linewidth=2)
    ax_grad.set_xlabel('Iteration')
    ax_grad.set_ylabel('Gradient Norm')
    ax_grad.set_title('Gradient Norm History')
    ax_grad.set_yscale('log')
    ax_grad.grid(True, alpha=0.3)

    # --- Bottom right: y-position time series (all balls) ---
    ax_yt = fig.add_subplot(gs[1, 2])
    for (ball_label, xi, yi), c in zip(balls, ball_colors):
        ax_yt.plot(sim_t, sim_x[:, yi], '-', color=c, alpha=0.3, label=f'{ball_label} (Sim)')
        ax_yt.plot(np.asarray(target_times), np.asarray(target_data[:, yi]),
                   'x', color=c, markersize=3, alpha=0.5, label=f'{ball_label} (Target)')
    ax_yt.set_xlabel('Time')
    ax_yt.set_ylabel('y position')
    ax_yt.set_title('Height vs Time')
    ax_yt.legend(fontsize=7)
    ax_yt.grid(True, alpha=0.3)

    final_loss = result['loss_history'][-1]
    fig.suptitle(
        f'Bouncing Balls Optimization (3 balls, {len(sol_true.segments)} segments)\n'
        f'Final Loss: {final_loss:.6e}',
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plot_path = os.path.join(current_dir, 'optimization_result_balls.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    # plt.show()


if __name__ == "__main__":
    run_optimization_test()
