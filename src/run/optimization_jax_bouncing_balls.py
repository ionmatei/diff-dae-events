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
# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from src.discrete_adjoint.dae_solver import DAESolver
from src.discrete_adjoint.dae_padded_gradient import DAEPaddedGradient
import torch
from src.pytorch.bouncing_balls import BouncingBallsModel


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



def create_animation(times, traj_opt, traj_true, x_min, x_max, y_min, y_max, filename='bouncing_balls_animation.mp4'):
    """Create a 2D animation of the bouncing balls."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("Matplotlib not available, skipping animation.")
        return

    n_balls = 3
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Static background: True trajectories and initial positions
    for i in range(n_balls):
        idx = i * 4
        # True trajectory (faint)
        if traj_true.shape[0] > 0:
            ax.plot(traj_true[:, idx], traj_true[:, idx+1], 'b-', alpha=0.2, linewidth=1, label='Target Trajectory' if i==0 else None)
            # Initial position
            ax.plot(traj_true[0, idx], traj_true[0, idx+1], 'bx', markersize=8, alpha=0.6, label='Initial Target' if i==0 else None)

    # Dynamic elements: Optimized balls
    balls = []
    trails = []
    # Distinct colors for the 3 optimized balls
    colors = ['#FF5733', '#33FF57', '#3357FF'] 
    
    for i in range(n_balls):
        ball, = ax.plot([], [], 'o', color=colors[i], markersize=28, markeredgecolor='k', label=f'Optimized Ball {i+1}')
        trail, = ax.plot([], [], '-', color=colors[i], alpha=0.5, linewidth=1.5)
        balls.append(ball)
        trails.append(trail)

    ax.set_xlim(x_min-1, x_max+1)
    ax.set_ylim(y_min-1, y_max+1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Bouncing Balls Optimization: Validation Animation')
    ax.legend(loc='upper right')
    
    # Animation update function
    def update(frame):
        artists = []
        for i in range(n_balls):
            idx = i * 4
            x = traj_opt[frame, idx]
            y = traj_opt[frame, idx+1]
            balls[i].set_data([x], [y])
            
            # Trail (last 50 frames)
            start_frame = max(0, frame - 50)
            trail_x = traj_opt[start_frame:frame+1, idx]
            trail_y = traj_opt[start_frame:frame+1, idx+1]
            trails[i].set_data(trail_x, trail_y)
            artists.append(balls[i])
            artists.append(trails[i])
        return artists
    
    # Downsample if too many frames to keep file size reasonable
    n_frames = len(times)
    # Target ~400 frames max
    step = max(1, n_frames // 400) 
    frames = range(0, n_frames, step)
    
    print(f"Creating animation ({len(frames)} frames)...")
    anim = FuncAnimation(fig, update, frames=frames, interval=30, blit=True)
    
    # Save
    try:
        # Try MP4 first (requires ffmpeg)
        anim.save(filename, writer='ffmpeg', fps=30)
        print(f"  Animation saved to: {filename}")
    except Exception as e:
        print(f"  Could not save MP4 (ffmpeg might be missing): {e}")
        try:
            # Fallback to GIF (requires pillow)
            gif_filename = filename.replace('.mp4', '.gif')
            anim.save(gif_filename, writer='pillow', fps=30)
            print(f"  Animation saved to: {gif_filename}")
        except Exception as e2:
             print(f"  Could not save GIF either: {e2}")

    plt.close(fig)


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
    
    g_val = float(p_opt_dict.get('g', 9.81))
    e_g_val = float(p_opt_dict.get('e_g', 0.8))
    e_b_val = float(p_opt_dict.get('e_b', 0.9))
    d_sq_val = float(p_opt_dict.get('d_sq', 0.1))
    
    # Box params usually in dae_data params or fixed
    # Checking dae_data['parameters'] logic
    x_min_val = float(p_opt_dict.get('x_min', 0.0))
    x_max_val = float(p_opt_dict.get('x_max', 3.0))
    y_min_val = float(p_opt_dict.get('y_min', 0.0))
    y_max_val = float(p_opt_dict.get('y_max', 3.0))
    
    # Initial state
    initial_state_val = [s['start'] for s in dae_data['states']]
    
    pt_model = BouncingBallsModel(
        g=g_val, e_g=e_g_val, e_b=e_b_val, d_sq=d_sq_val,
        x_min=x_min_val, x_max=x_max_val, y_min=y_min_val, y_max=y_max_val,
        initial_state=initial_state_val, ncp=ncp
    )

    # 2. Compute predictions at exact target times (without interpolation)
    target_times_t = torch.tensor(np.array(target_times), dtype=torch.float64)
    with torch.no_grad():
        y_pred_pt = pt_model.simulate_at_targets(target_times_t)
    
    y_pred_np = y_pred_pt.numpy()
    
    # Calculate and print validation error
    target_data_np = np.array(target_data)
    
    # Compute loss on positions only (x, y) to match PyTorch simulator loss logic
    # Indices: 0(x1), 1(y1), 4(x2), 5(y2), 8(x3), 9(y3)
    pos_idx = [0, 1, 4, 5, 8, 9]
    y_pred_pos = y_pred_np[:, pos_idx]
    target_pos = target_data_np[:, pos_idx]
    
    val_mse = np.mean((y_pred_pos - target_pos)**2)
    print(f"Final Loss (PyTorch, Positions only): {val_mse:.6e}")

    # 3. Generate dense trajectory for plotting/animation (using PyTorch)
    # Using simulate_fixed_grid for dense output (similar to reference PyTorch script)
    stop_time = float(solver_cfg['stop_time'])
    with torch.no_grad():
        times_sim, traj_sim = pt_model.simulate_fixed_grid(stop_time, n_points=ncp)
        
    sim_t = times_sim.numpy()
    sim_x = traj_sim.numpy()

    # Flatten simulated data for plotting (concatenating segments)
    # NOTE: PyTorch simulate_fixed_grid already returns concatenated (Time, States) tensors.
    # So we don't need the loop over segments like in JAX solver.
    # Just need to ensure variable names match what plotting expects (sim_t, sim_x).
    
    # Interpolated predictions (Using PyTorch results as 'Prediction')
    y_pred = y_pred_pt # For compatibility with plotting name used below if any
    
    # Just confirming shape match for plotting code below...
    # plotting starts at line 298 using sim_x[:, xi], etc.
    # y_pred_np is used for 'Interp' dots.
    pass

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
    plot_path = os.path.join(root_dir, 'results', 'optimization_result_jax_bouncing_balls.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    
    # --- 7. Animation ---
    print("\nGenerating animation...")
    
    # Reconstruction of true trajectory for animation
    true_sim_t = []
    true_sim_x = []
    for seg in sol_true.segments:
        if len(seg.t) > 0:
            true_sim_t.append(seg.t)
            true_sim_x.append(seg.x)
    
    if true_sim_t:
        traj_true = np.concatenate(true_sim_x)
    else:
        traj_true = np.zeros((0, n_x))

    # Get bounds
    p_dict = dict(zip(param_names, true_p))
    x_min, x_max = p_dict['x_min'], p_dict['x_max']
    y_min, y_max = p_dict['y_min'], p_dict['y_max']

    create_animation(
        sim_t, sim_x, traj_true,
        x_min, x_max, y_min, y_max,
        filename=os.path.join(root_dir, 'results', 'animation_jax_balls.mp4')
    )
    # plt.show()


if __name__ == "__main__":
    run_optimization_test()
