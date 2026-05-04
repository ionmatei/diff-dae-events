"""
Adam optimization on the N-ball bouncing-balls DAE.

Generalization of `optimization_jax_bouncing_balls.py` (3 balls hard-coded)
to an arbitrary N defined by the DAE specification. Generates ground truth
with the IDA-based `DAESolver` (compiled-residual fast path), biases the
optimized parameters, runs Adam via `DAEPaddedGradient`, and validates by
re-simulating with the PyTorch `BouncingBallsNModel`.
"""

import os
import sys
import argparse

import numpy as np
import yaml

import jax
import jax.numpy as jnp
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
# Default to CPU unless the caller already set JAX_PLATFORMS (e.g.
# "cuda" / "gpu"). This lets a benchmark wrapper opt into GPU without
# the runner overriding the choice at import time.
if not os.environ.get("JAX_PLATFORMS"):
    jax_config.update("jax_platform_name", "cpu")

# Persistent JIT compilation cache: the gradient kernel's first compile
# can take tens of seconds for N>=15. With this cache enabled, subsequent
# runs that share the same (max_blocks, max_pts, max_targets, n_x, n_z, n_p)
# reuse the compiled XLA module from disk and start instantly. Override
# the cache directory via the JAX_COMPILATION_CACHE_DIR env var.
_jax_cache_dir = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "jax_dae_optim"),
)
os.makedirs(_jax_cache_dir, exist_ok=True)
jax_config.update("jax_compilation_cache_dir", _jax_cache_dir)
jax_config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax_config.update("jax_persistent_cache_min_compile_time_secs", 0)

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from src.discrete_adjoint.dae_solver import DAESolver  # noqa: E402
from src.discrete_adjoint.dae_padded_gradient import DAEPaddedGradient  # noqa: E402
import torch  # noqa: E402
from src.pytorch.bouncing_balls_n import BouncingBallsNModel  # noqa: E402


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    solver_cfg = cfg['dae_solver']
    opt_cfg = cfg['optimizer']
    dae_spec_path = solver_cfg['dae_specification_file']
    if not os.path.isabs(dae_spec_path):
        dae_spec_path = os.path.join(root_dir, dae_spec_path)
    generate_animation = cfg.get('generate_animation', False)
    # Spec may be YAML or JSON; yaml.safe_load handles both.
    with open(dae_spec_path, 'r') as f:
        dae_data = yaml.safe_load(f)
    return dae_data, solver_cfg, opt_cfg, generate_animation


def count_balls(dae_data: dict) -> int:
    """Return the number of balls inferred from the state list (one ball
    per `xK` state with K an integer suffix)."""
    state_names = [s['name'] for s in dae_data['states']]
    n_balls = sum(1 for n in state_names if n.startswith('x') and n[1:].isdigit())
    if 4 * n_balls != len(state_names):
        raise ValueError(
            f"Spec layout mismatch: counted {n_balls} balls but {len(state_names)} states "
            f"(expected 4N = {4 * n_balls})."
        )
    return n_balls


def position_indices(dae_data: dict, n_balls: int):
    """Return the flat index list of (x_i, y_i) positions in the layout."""
    state_names = [s['name'] for s in dae_data['states']]
    pos_idx = []
    for i in range(1, n_balls + 1):
        pos_idx.append(state_names.index(f'x{i}'))
        pos_idx.append(state_names.index(f'y{i}'))
    return pos_idx


def prepare_loss_targets(sol):
    """Stack interior target times/data from each segment (drops the
    last point of each segment to keep targets strictly inside)."""
    all_t, all_x = [], []
    for seg in sol.segments:
        if len(seg.t) > 0:
            all_t.append(seg.t)
            all_x.append(seg.x)
    if not all_t:
        return jnp.array([]), jnp.array([])
    target_times = jnp.concatenate([jnp.array(t[:-1]) for t in all_t])
    target_data = jnp.concatenate([jnp.array(x[:-1]) for x in all_x])
    return target_times, target_data


def create_animation(times, traj_opt, traj_true, x_min, x_max, y_min, y_max,
                     n_balls: int, filename='bouncing_balls_animation_N.mp4'):
    """2D animation of the N optimized balls overlaid on the true trajectories."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("Matplotlib not available, skipping animation.")
        return

    fig, ax = plt.subplots(figsize=(9, 9))
    cmap = colormaps.get_cmap('hsv').resampled(n_balls)

    for i in range(n_balls):
        idx = i * 4
        if traj_true.shape[0] > 0:
            ax.plot(traj_true[:, idx], traj_true[:, idx + 1],
                    'b-', alpha=0.15, linewidth=1,
                    label='Target Trajectory' if i == 0 else None)
            ax.plot(traj_true[0, idx], traj_true[0, idx + 1],
                    'bx', markersize=6, alpha=0.5,
                    label='Initial Target' if i == 0 else None)

    balls, trails = [], []
    for i in range(n_balls):
        c = cmap(i)
        ball, = ax.plot([], [], 'o', color=c, markersize=10,
                        markeredgecolor='k')
        trail, = ax.plot([], [], '-', color=c, alpha=0.5, linewidth=1.0)
        balls.append(ball)
        trails.append(trail)

    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Bouncing Balls Optimization ({n_balls} balls): Validation')
    ax.legend(loc='upper right', fontsize=8)

    def update(frame):
        artists = []
        for i in range(n_balls):
            idx = i * 4
            x = traj_opt[frame, idx]
            y = traj_opt[frame, idx + 1]
            balls[i].set_data([x], [y])
            start_frame = max(0, frame - 50)
            trail_x = traj_opt[start_frame:frame + 1, idx]
            trail_y = traj_opt[start_frame:frame + 1, idx + 1]
            trails[i].set_data(trail_x, trail_y)
            artists.append(balls[i])
            artists.append(trails[i])
        return artists

    n_frames = len(times)
    step = max(1, n_frames // 400)
    frames = range(0, n_frames, step)

    print(f"Creating animation ({len(frames)} frames)...")
    anim = FuncAnimation(fig, update, frames=frames, interval=30, blit=True)

    try:
        anim.save(filename, writer='ffmpeg', fps=30)
        print(f"  Animation saved to: {filename}")
    except Exception as e:
        print(f"  Could not save MP4 (ffmpeg might be missing): {e}")
        try:
            gif_filename = filename.replace('.mp4', '.gif')
            anim.save(gif_filename, writer='pillow', fps=30)
            print(f"  Animation saved to: {gif_filename}")
        except Exception as e2:
            print(f"  Could not save GIF either: {e2}")
    plt.close(fig)


def run_optimization_test(config_path: str):
    print("=" * 70)
    print("Adam Optimization on Bouncing Balls DAE (N balls)")
    print("=" * 70)

    # --- 1. Load config ---
    dae_data, solver_cfg, opt_cfg, generate_animation = load_config(config_path)
    n_balls = count_balls(dae_data)
    pos_idx = position_indices(dae_data, n_balls)
    print(f"Detected N = {n_balls} balls, {len(dae_data['states'])} states.")

    t_start = solver_cfg['start_time']
    t_stop = solver_cfg['stop_time']
    ncp = solver_cfg['ncp']
    t_span = (t_start, t_stop)

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

    param_names = [p['name'] for p in dae_data['parameters']]
    opt_param_names = opt_cfg['opt_params']
    opt_param_indices = [param_names.index(n) for n in opt_param_names]
    print(f"Optimized parameters: {opt_param_names} (indices {opt_param_indices})")

    # --- 2. Ground truth ---
    true_p = [p['value'] for p in dae_data['parameters']]
    print(f"True parameters: {dict(zip(param_names, true_p))}")

    solver = DAESolver(dae_data, verbose=False, use_compiled_residual=True)
    solver.update_parameters(true_p)
    sol_true = solver.solve_augmented(t_span, ncp=ncp)

    target_times, target_data = prepare_loss_targets(sol_true)
    delta_t = target_times[1:] - target_times[:-1]
    print(f"Delta t min: {jnp.min(delta_t)}")
    print(f"Number of segments: {len(sol_true.segments)}")
    print(f"Target data points: {len(target_times)}")

    # --- 3. Biased initial guess ---
    bias = {'g': -1.00, 'e_g': 0.1, 'e_b': 0.1}
    p_init = list(true_p)
    for name in opt_param_names:
        idx = param_names.index(name)
        p_init[idx] += bias.get(name, 0.0)
    p_init = jnp.array(p_init)
    print(f"Initial (biased) parameters: {dict(zip(param_names, np.asarray(p_init)))}")

    # --- 4. Build gradient computer and run Adam ---
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
        adaptive_horizon=False,
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

    for name in opt_param_names:
        idx = param_names.index(name)
        err = abs(float(p_opt[idx]) - true_p[idx])
        print(f"  {name}: true={true_p[idx]:.4f}  opt={float(p_opt[idx]):.4f}  err={err:.6e}")

    # --- 6. PyTorch validation ---
    print("\nValidating with PyTorch simulator...")
    p_opt_dict = dict(zip(param_names, np.asarray(p_opt)))
    g_val = float(p_opt_dict.get('g', 9.81))
    e_g_val = float(p_opt_dict.get('e_g', 0.8))
    e_b_val = float(p_opt_dict.get('e_b', 0.9))
    d_sq_val = float(p_opt_dict.get('d_sq', 0.25))
    p_dict_true = dict(zip(param_names, true_p))
    x_min_val = float(p_dict_true.get('x_min', 0.0))
    x_max_val = float(p_dict_true.get('x_max', 10.0))
    y_min_val = float(p_dict_true.get('y_min', 0.0))
    y_max_val = float(p_dict_true.get('y_max', 10.0))

    initial_state_val = [s['start'] for s in dae_data['states']]

    pt_model = BouncingBallsNModel(
        N=n_balls,
        g=g_val, e_g=e_g_val, e_b=e_b_val, d_sq=d_sq_val,
        x_min=x_min_val, x_max=x_max_val, y_min=y_min_val, y_max=y_max_val,
        initial_state=initial_state_val, adjoint=False,
    )

    target_times_t = torch.tensor(np.array(target_times), dtype=torch.float64)
    with torch.no_grad():
        y_pred_pt = pt_model.simulate_at_targets(target_times_t, max_events=400)
    y_pred_np = y_pred_pt.numpy()

    target_data_np = np.array(target_data)
    val_mse = float(np.mean((y_pred_np[:, pos_idx] - target_data_np[:, pos_idx]) ** 2))
    print(f"Final Loss (PyTorch, positions only, {len(pos_idx)} entries): {val_mse:.6e}")

    # Dense PyTorch trajectory for plotting / animation.
    stop_time = float(t_stop)
    with torch.no_grad():
        times_sim, traj_sim, _ = pt_model.simulate_fixed_grid(
            stop_time - t_start, n_points=ncp, max_events=400
        )
    sim_t = times_sim.numpy() + t_start
    sim_x = traj_sim.numpy()

    # --- 7. Plotting ---
    print("\nGenerating plots...")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    cmap = colormaps.get_cmap('hsv').resampled(n_balls)
    state_names = [s['name'] for s in dae_data['states']]

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 3)

    # Top-left: x-y trajectories overlay (all balls).
    ax_xy = fig.add_subplot(gs[0, 0])
    for i in range(1, n_balls + 1):
        xi = state_names.index(f'x{i}')
        yi = state_names.index(f'y{i}')
        c = cmap(i - 1)
        ax_xy.plot(sim_x[:, xi], sim_x[:, yi], '-', color=c, alpha=0.5, linewidth=0.8)
    ax_xy.plot([x_min_val, x_max_val, x_max_val, x_min_val, x_min_val],
               [y_min_val, y_min_val, y_max_val, y_max_val, y_min_val],
               'k--', alpha=0.4, linewidth=1)
    ax_xy.set_xlabel('x position')
    ax_xy.set_ylabel('y position')
    ax_xy.set_title(f'{n_balls}-ball optimized trajectories (PyTorch)')
    ax_xy.set_aspect('equal', adjustable='datalim')
    ax_xy.grid(True, alpha=0.3)

    # Top-center: y(t) for all balls, optimized vs target samples.
    ax_yt = fig.add_subplot(gs[0, 1])
    target_times_np = np.asarray(target_times)
    for i in range(1, n_balls + 1):
        yi = state_names.index(f'y{i}')
        c = cmap(i - 1)
        ax_yt.plot(sim_t, sim_x[:, yi], '-', color=c, alpha=0.6, linewidth=0.8)
        ax_yt.plot(target_times_np, target_data_np[:, yi],
                   'x', color=c, markersize=2, alpha=0.4)
    ax_yt.set_xlabel('time [s]')
    ax_yt.set_ylabel('y position')
    ax_yt.set_title('y(t): optimized (line) vs targets (x)')
    ax_yt.grid(True, alpha=0.3)

    # Top-right: x(t) for all balls.
    ax_xt = fig.add_subplot(gs[0, 2])
    for i in range(1, n_balls + 1):
        xi = state_names.index(f'x{i}')
        c = cmap(i - 1)
        ax_xt.plot(sim_t, sim_x[:, xi], '-', color=c, alpha=0.6, linewidth=0.8)
        ax_xt.plot(target_times_np, target_data_np[:, xi],
                   'x', color=c, markersize=2, alpha=0.4)
    ax_xt.set_xlabel('time [s]')
    ax_xt.set_ylabel('x position')
    ax_xt.set_title('x(t): optimized (line) vs targets (x)')
    ax_xt.grid(True, alpha=0.3)

    # Bottom-left: loss history.
    ax_loss = fig.add_subplot(gs[1, 0])
    ax_loss.plot(result['loss_history'], 'b-', linewidth=2)
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Loss History')
    ax_loss.set_yscale('log')
    ax_loss.grid(True, alpha=0.3)

    # Bottom-center: grad-norm history.
    ax_grad = fig.add_subplot(gs[1, 1])
    ax_grad.plot(result['grad_norm_history'], 'r-', linewidth=2)
    ax_grad.set_xlabel('Iteration')
    ax_grad.set_ylabel('Gradient Norm')
    ax_grad.set_title('Gradient Norm History')
    ax_grad.set_yscale('log')
    ax_grad.grid(True, alpha=0.3)

    # Bottom-right: per-parameter true / initial / optimized bars.
    ax_p = fig.add_subplot(gs[1, 2])
    bar_pos = np.arange(len(opt_param_names))
    width = 0.27
    true_vals = [true_p[opt_param_indices[j]] for j in range(len(opt_param_names))]
    init_vals = [float(p_init[opt_param_indices[j]]) for j in range(len(opt_param_names))]
    opt_vals = [float(p_opt[opt_param_indices[j]]) for j in range(len(opt_param_names))]
    ax_p.bar(bar_pos - width, true_vals, width, label='True', color='tab:blue')
    ax_p.bar(bar_pos, init_vals, width, label='Initial', color='tab:gray')
    ax_p.bar(bar_pos + width, opt_vals, width, label='Optimized', color='tab:orange')
    ax_p.set_xticks(bar_pos)
    ax_p.set_xticklabels(opt_param_names)
    ax_p.set_ylabel('Parameter value')
    ax_p.set_title('Parameter recovery')
    ax_p.legend(fontsize=8)
    ax_p.grid(True, alpha=0.3, axis='y')

    final_loss = result['loss_history'][-1]
    fig.suptitle(
        f'Bouncing Balls Optimization ({n_balls} balls, {len(sol_true.segments)} segments)\n'
        f'Final Loss: {final_loss:.6e}',
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plot_path = os.path.join(root_dir, 'results', f'optimization_result_jax_bouncing_balls_N{n_balls}.png')
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")

    # --- 8. Animation ---
    if generate_animation:
        print("\nGenerating animation...")
        true_sim_x = []
        for seg in sol_true.segments:
            if len(seg.t) > 0:
                true_sim_x.append(np.asarray(seg.x))
        traj_true = np.concatenate(true_sim_x) if true_sim_x else np.zeros((0, len(dae_data['states'])))

        create_animation(
            sim_t, sim_x, traj_true,
            x_min_val, x_max_val, y_min_val, y_max_val,
            n_balls=n_balls,
            filename=os.path.join(root_dir, 'results', f'animation_jax_balls_N{n_balls}.mp4'),
        )

    # Prediction-error series: ‖p_iter − p_true‖ on the optimized
    # parameters only, evaluated at every iteration. Cheap to compute
    # (3-element vector norm per iter) and comparable across optimizers.
    p_true_opt = np.array([true_p[i] for i in opt_param_indices], dtype=float)
    prediction_error_history = [
        float(np.linalg.norm(np.asarray(p_iter, dtype=float) - p_true_opt))
        for p_iter in result.get('p_history', [])
    ]

    benchmark_results = {
        'method': 'jax_multi_N',
        'n_balls': n_balls,
        'ncp': ncp,
        'avg_iter_time': result.get('avg_iter_time', 0.0),
        'p_opt': dict(zip(param_names, np.asarray(p_opt))),
        'p_true': dict(zip(param_names, true_p)),
        'opt_param_names': list(opt_param_names),
        'iterations': result['n_iter'],
        'converged': bool(result['converged']),
        'final_validation_loss': float(val_mse),
        'prediction_error_history': prediction_error_history,
    }
    return benchmark_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-ball JAX optimization test")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=os.path.join('config', 'config_bouncing_balls_N3.yaml'),
        help='Path to configuration YAML file (relative to project root or absolute).',
    )
    args = parser.parse_args()
    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(root_dir, cfg_path)
    run_optimization_test(cfg_path)
