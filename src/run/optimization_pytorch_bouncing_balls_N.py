"""
PyTorch optimization for the N-ball bouncing-balls DAE (N set by the
loaded spec). Uses `BouncingBallsNModel` for the differentiable forward
simulation and `DAESolver` (compiled-residual fast path) to generate
ground truth.
"""

from __future__ import annotations

import os
import sys
import time
import argparse

import numpy as np
import torch
import yaml

torch.set_default_dtype(torch.float64)

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from src.pytorch.bouncing_balls_n import BouncingBallsNModel  # noqa: E402
from src.discrete_adjoint.dae_solver import DAESolver  # noqa: E402


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_dae_spec(spec_path: str) -> dict:
    with open(spec_path, 'r') as f:
        return yaml.safe_load(f)  # YAML accepts JSON


def count_balls(dae_data: dict) -> int:
    """Count balls from state names (one ball per `xK` state)."""
    state_names = [s['name'] for s in dae_data['states']]
    n = sum(1 for s in state_names if s.startswith('x') and s[1:].isdigit())
    if 4 * n != len(state_names):
        raise ValueError(
            f"Spec layout mismatch: counted {n} balls but {len(state_names)} states."
        )
    return n


def position_indices(dae_data: dict, n_balls: int):
    state_names = [s['name'] for s in dae_data['states']]
    pos_idx = []
    for i in range(1, n_balls + 1):
        pos_idx.append(state_names.index(f'x{i}'))
        pos_idx.append(state_names.index(f'y{i}'))
    return pos_idx


def prepare_loss_targets(sol):
    all_t, all_x = [], []
    for seg in sol.segments:
        if len(seg.t) > 0:
            all_t.append(seg.t)
            all_x.append(seg.x)
    if not all_t:
        return np.array([]), np.array([])
    target_times = np.concatenate([np.array(t[:-1]) for t in all_t])
    target_data = np.concatenate([np.array(x[:-1]) for x in all_x])
    return target_times, target_data


class DAEOptimizerPyTorchMultiEventN:
    """PyTorch-based Adam/SGD optimizer for the N-ball bouncing-balls model."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimize_params: list,
        pos_idx: list,
        verbose: bool = True,
        max_events: int = 400,
    ):
        self.model = model
        self.optimize_params = optimize_params
        self.pos_idx_t = torch.tensor(pos_idx, dtype=torch.long)
        self.verbose = verbose
        self.max_events = max_events

        self.param_dict = {name: p for name, p in model.named_parameters()}
        self.opt_params = [self.param_dict[n] for n in optimize_params]

    def _compute_loss(self, target_times: torch.Tensor, target_data: torch.Tensor) -> torch.Tensor:
        """Loss = MSE on positions only (continuous across events)."""
        y_pred = self.model.simulate_at_targets(target_times, max_events=self.max_events)
        return torch.mean((y_pred[:, self.pos_idx_t] - target_data[:, self.pos_idx_t]) ** 2)

    def optimize(
        self,
        target_times: np.ndarray,
        target_data: np.ndarray,
        max_iterations: int = 100,
        step_size: float = 0.001,
        tol: float = 1e-6,
        print_every: int = 10,
        algorithm: str = 'adam',
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> dict:
        if algorithm.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.opt_params, lr=step_size, betas=(beta1, beta2), eps=epsilon
            )
        else:
            optimizer = torch.optim.SGD(self.opt_params, lr=step_size)

        history = {'loss': [], 'gradient_norm': [], 'params': []}

        p_init = np.concatenate([p.detach().numpy().flatten() for p in self.opt_params])
        if self.verbose:
            print(f"\nStarting PyTorch optimization (N-ball)")
            print(f"  Algorithm: {algorithm}")
            print(f"  Max iterations: {max_iterations}")
            print(f"  Step size: {step_size}")
            print(f"  Parameters: {self.optimize_params}")
            print(f"  Initial values: {p_init}")
            print()

        target_times_t = torch.tensor(target_times, dtype=torch.float64)
        target_data_t = torch.tensor(target_data, dtype=torch.float64)

        start_time = time.time()
        converged = False
        iter_times = []

        for it in range(max_iterations):
            iter_start = time.time()

            optimizer.zero_grad()
            loss = self._compute_loss(target_times_t, target_data_t)
            loss.backward()

            grad_norm = 0.0
            for p in self.opt_params:
                if p.grad is not None:
                    grad_norm += float(torch.sum(p.grad ** 2))
            grad_norm = float(np.sqrt(grad_norm))

            loss_val = float(loss.detach())
            history['loss'].append(loss_val)
            history['gradient_norm'].append(grad_norm)
            history['params'].append(
                np.concatenate([p.detach().numpy().flatten() for p in self.opt_params]).copy()
            )

            iter_time = (time.time() - iter_start) * 1000.0
            iter_times.append(iter_time)

            if it % print_every == 0 or it == 0:
                param_str = " ".join([f"{p.item():.4f}" for p in self.opt_params])
                print(f"  Iter {it:4d} | loss={loss_val:.6e} | |grad|={grad_norm:.6e} | "
                      f"p=[{param_str}] | {iter_time:.1f} ms ")
                if it == 0:
                    for name, param in zip(self.optimize_params, self.opt_params):
                        if param.grad is not None:
                            print(f"    grad[{name}] = {param.grad.item():.6e}")

            optimizer.step()

            if grad_norm < tol and it > 0:
                print(f"\nConverged at iteration {it}")
                converged = True
                break

            with torch.no_grad():
                if hasattr(self.model, 'e_g'):
                    self.model.e_g.data.clamp_(0.001, 5.0)
                if hasattr(self.model, 'e_b'):
                    self.model.e_b.data.clamp_(0.001, 5.0)
                if hasattr(self.model, 'g'):
                    self.model.g.data.clamp_(0.001, 100.0)

        elapsed = time.time() - start_time
        p_final = np.concatenate([p.detach().numpy().flatten() for p in self.opt_params])
        avg_iter_time = (sum(iter_times[1:]) / max(1, len(iter_times) - 1)
                         if len(iter_times) > 1 else 0.0)

        if self.verbose:
            print(f"\nOptimization complete in {elapsed:.2f}s")
            print(f"  Final loss: {history['loss'][-1]:.6e}")
            print(f"  Final params: {p_final}")

        return {
            'params': p_final,
            'history': history,
            'elapsed_time': elapsed,
            'converged': converged,
            'n_iter': len(history['loss']),
            'avg_iter_time': avg_iter_time,
        }


def create_animation(times, traj_opt, traj_true, x_min, x_max, y_min, y_max,
                     n_balls: int, filename='bouncing_balls_pytorch_animation_N.mp4'):
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
        ax.plot(traj_true[:, idx], traj_true[:, idx + 1],
                'b-', alpha=0.15, linewidth=1,
                label='Target Trajectory' if i == 0 else None)
        ax.plot(traj_true[0, idx], traj_true[0, idx + 1],
                'bx', markersize=6, alpha=0.5,
                label='Initial Target' if i == 0 else None)

    balls, trails = [], []
    for i in range(n_balls):
        c = cmap(i)
        ball, = ax.plot([], [], 'o', color=c, markersize=10, markeredgecolor='k')
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
            artists.append(balls[i]); artists.append(trails[i])
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


def run_bouncing_balls_test(config: dict, config_dir: str):
    print("=" * 80)
    print("Bouncing Balls (N balls) - PyTorch Optimizer Test")
    print("=" * 80)

    solver_cfg = config['dae_solver']
    opt_cfg = config['optimizer']
    algo_cfg = opt_cfg.get('algorithm', {})
    generate_animation = config.get('generate_animation', False)

    spec_path = solver_cfg['dae_specification_file']
    if not os.path.isabs(spec_path):
        spec_path = os.path.join(root_dir, spec_path)
    dae_data = load_dae_spec(spec_path)

    n_balls = count_balls(dae_data)
    pos_idx = position_indices(dae_data, n_balls)
    state_names = [s['name'] for s in dae_data['states']]

    print(f"\nLoaded DAE from: {spec_path}")
    print(f"  Detected N = {n_balls} balls, {len(state_names)} states.")
    print(f"  Parameters: {[p['name'] for p in dae_data['parameters']]}")
    if dae_data.get('when'):
        print(f"  Events: {len(dae_data['when'])} event sources")

    p_true = {p['name']: p['value'] for p in dae_data['parameters']}
    g_true = float(p_true['g'])
    e_g_true = float(p_true['e_g'])
    e_b_true = float(p_true['e_b'])
    d_sq_true = float(p_true['d_sq'])
    x_min = float(p_true['x_min']); x_max = float(p_true['x_max'])
    y_min = float(p_true['y_min']); y_max = float(p_true['y_max'])
    initial_state = [float(s['start']) for s in dae_data['states']]

    print(f"\nTrue parameters:")
    print(f"  g={g_true}, e_g={e_g_true}, e_b={e_b_true}")
    print(f"  Box: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], d_sq={d_sq_true}")

    # ---------------------------------------------------------------------
    # Step 1: Generate reference trajectory with the IDA solver (true params)
    # ---------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Step 1: Generate Reference Trajectory (DAESolver, compiled fast path)")
    print("-" * 40)

    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    ncp = solver_cfg.get('ncp', 600)

    true_p = [p['value'] for p in dae_data['parameters']]
    solver = DAESolver(dae_data, verbose=False, use_compiled_residual=True)
    solver.update_parameters(true_p)

    print(f"  Simulating with DAESolver (true parameters)...")
    sol_true = solver.solve_augmented(t_span, ncp=ncp)
    t_target, y_target = prepare_loss_targets(sol_true)

    delta_t = t_target[1:] - t_target[:-1]
    print(f"  Simulation time: {t_span}")
    print(f"  Number of segments: {len(sol_true.segments)}")
    print(f"  Target data points: {len(t_target)}")
    if len(delta_t):
        print(f"  Delta t min: {np.min(delta_t):.6e}")

    # ---------------------------------------------------------------------
    # Step 2: Perturbed initial guess
    # ---------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Step 2: Create Perturbed Initial Guess")
    print("-" * 40)

    g_init = g_true - 1.0
    e_g_init = e_g_true + 0.1
    e_b_init = e_b_true + 0.1

    optimize_params = opt_cfg['opt_params']
    init_lookup = {'g': g_init, 'e_g': e_g_init, 'e_b': e_b_init}
    for p_name in ('g', 'e_g', 'e_b'):
        val_true = p_true[p_name]
        val_init = init_lookup[p_name]
        is_opt = p_name in optimize_params
        status = "OPTIMIZED" if is_opt else "FIXED"
        diff = val_init - val_true
        print(f"  {p_name}: True={val_true:.4f}, Init={val_init:.4f} (diff={diff:+.4f}) -> {status}")

    # ---------------------------------------------------------------------
    # Step 3: Build model + optimizer, run Adam
    # ---------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Step 3: Run Optimization")
    print("-" * 40)

    model_opt = BouncingBallsNModel(
        N=n_balls,
        g=g_init, e_g=e_g_init, e_b=e_b_init, d_sq=d_sq_true,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        initial_state=initial_state, adjoint=False,
    )

    max_events = max(opt_cfg.get('max_segments', 50) * 8, 200)
    optimizer = DAEOptimizerPyTorchMultiEventN(
        model=model_opt,
        optimize_params=optimize_params,
        pos_idx=pos_idx,
        verbose=True,
        max_events=max_events,
    )

    algo_params = algo_cfg.get('params', {})
    step_size = algo_params.get('step_size', 0.001)
    algorithm_type = algo_cfg.get('type', 'adam').lower()
    beta1 = algo_params.get('beta1', 0.9)
    beta2 = algo_params.get('beta2', 0.999)
    epsilon = algo_params.get('epsilon', 1e-8)

    result = optimizer.optimize(
        target_times=t_target,
        target_data=y_target,
        max_iterations=opt_cfg['max_iterations'],
        step_size=step_size,
        tol=opt_cfg['tol'],
        print_every=opt_cfg.get('print_every', 10),
        algorithm=algorithm_type,
        beta1=beta1, beta2=beta2, epsilon=epsilon,
    )

    # ---------------------------------------------------------------------
    # Step 4: Report
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Optimization Result")
    print("=" * 70)

    p_true_subset = {k: p_true[k] for k in optimize_params}
    p_init_subset = {k: init_lookup[k] for k in optimize_params}
    p_opt_subset = {k: float(result['params'][i]) for i, k in enumerate(optimize_params)}

    print(f"True params:      {p_true_subset}")
    print(f"Initial params:   {p_init_subset}")
    print(f"Optimized params: {p_opt_subset}")
    print(f"Iterations:       {result['n_iter']}")
    print(f"Converged:        {result['converged']}")
    print(f"Final loss:       {result['history']['loss'][-1]:.6e}")
    print(f"Final |grad|:     {result['history']['gradient_norm'][-1]:.6e}")
    print(f"Avg iter time:    {result['avg_iter_time']:.2f} ms")

    for name in optimize_params:
        true_val = p_true[name]
        opt_val = p_opt_subset[name]
        err = abs(opt_val - true_val)
        print(f"  {name}: true={true_val:.4f}  opt={opt_val:.4f}  err={err:.6e}")

    # ---------------------------------------------------------------------
    # Step 5: Validation re-simulation
    # ---------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Step 5: Validation")
    print("-" * 40)

    target_times_t = torch.tensor(t_target, dtype=torch.float64)
    with torch.no_grad():
        y_pred = model_opt.simulate_at_targets(target_times_t, max_events=max_events)
    y_pred_np = y_pred.numpy()

    val_mse = float(np.mean((y_pred_np[:, pos_idx] - y_target[:, pos_idx]) ** 2))
    print(f"Final Loss (PyTorch, positions only, {len(pos_idx)} entries): {val_mse:.6e}")

    t_end_val = float(t_target[-1]) + 1e-6
    with torch.no_grad():
        times_opt, traj_opt, _ = model_opt.simulate_fixed_grid(
            t_end_val, n_points=ncp, max_events=max_events
        )
    times_opt_np = times_opt.numpy()
    traj_opt_np = traj_opt.numpy()

    y_opt_pos = y_pred_np[:, pos_idx]
    y_tgt_pos = y_target[:, pos_idx]
    denom = np.linalg.norm(y_tgt_pos)
    traj_error = float(np.linalg.norm(y_opt_pos - y_tgt_pos) / denom) if denom > 0 else 0.0
    print(f"  Trajectory relative error (positions): {traj_error:.6e}")

    all_t_true, all_x_true = [], []
    for seg in sol_true.segments:
        if len(seg.t) > 0:
            all_t_true.append(np.asarray(seg.t))
            all_x_true.append(np.asarray(seg.x))
    times_true_np = np.concatenate(all_t_true) if all_t_true else np.zeros((0,))
    traj_true_np = np.concatenate(all_x_true) if all_x_true else np.zeros((0, 4 * n_balls))

    # ---------------------------------------------------------------------
    # Step 6: Plot
    # ---------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colormaps

        cmap = colormaps.get_cmap('hsv').resampled(n_balls)
        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(2, 3)

        # Top-left: x-y trajectories overlay (true vs optimized).
        ax_xy = fig.add_subplot(gs[0, 0])
        for i in range(1, n_balls + 1):
            xi = state_names.index(f'x{i}')
            yi = state_names.index(f'y{i}')
            c = cmap(i - 1)
            ax_xy.plot(traj_true_np[:, xi], traj_true_np[:, yi],
                       '-', color=c, alpha=0.35, linewidth=0.8)
            ax_xy.plot(traj_opt_np[:, xi], traj_opt_np[:, yi],
                       '--', color=c, alpha=0.7, linewidth=0.8)
        ax_xy.plot([x_min, x_max, x_max, x_min, x_min],
                   [y_min, y_min, y_max, y_max, y_min],
                   'k--', alpha=0.4, linewidth=1)
        ax_xy.set_xlabel('x')
        ax_xy.set_ylabel('y')
        ax_xy.set_title(f'{n_balls}-ball trajectories: true (-) vs optimized (--)')
        ax_xy.set_aspect('equal', adjustable='datalim')
        ax_xy.grid(True, alpha=0.3)

        # Top-center: y(t).
        ax_yt = fig.add_subplot(gs[0, 1])
        for i in range(1, n_balls + 1):
            yi = state_names.index(f'y{i}')
            c = cmap(i - 1)
            ax_yt.plot(times_true_np, traj_true_np[:, yi], '-', color=c, alpha=0.35, linewidth=0.8)
            ax_yt.plot(times_opt_np, traj_opt_np[:, yi], '--', color=c, alpha=0.7, linewidth=0.8)
        ax_yt.set_xlabel('time [s]')
        ax_yt.set_ylabel('y')
        ax_yt.set_title('y(t): true (-) vs optimized (--)')
        ax_yt.grid(True, alpha=0.3)

        # Top-right: x(t).
        ax_xt = fig.add_subplot(gs[0, 2])
        for i in range(1, n_balls + 1):
            xi = state_names.index(f'x{i}')
            c = cmap(i - 1)
            ax_xt.plot(times_true_np, traj_true_np[:, xi], '-', color=c, alpha=0.35, linewidth=0.8)
            ax_xt.plot(times_opt_np, traj_opt_np[:, xi], '--', color=c, alpha=0.7, linewidth=0.8)
        ax_xt.set_xlabel('time [s]')
        ax_xt.set_ylabel('x')
        ax_xt.set_title('x(t): true (-) vs optimized (--)')
        ax_xt.grid(True, alpha=0.3)

        # Bottom-left: loss.
        ax = fig.add_subplot(gs[1, 0])
        ax.semilogy(result['history']['loss'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Loss History')
        ax.grid(True, alpha=0.3)

        # Bottom-center: gradient norm.
        ax = fig.add_subplot(gs[1, 1])
        ax.semilogy(result['history']['gradient_norm'], 'r-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm History')
        ax.grid(True, alpha=0.3)

        # Bottom-right: parameter history.
        ax = fig.add_subplot(gs[1, 2])
        param_hist = np.array(result['history']['params'])
        for i, name in enumerate(optimize_params):
            true_val = p_true[name]
            ax.plot(param_hist[:, i], linewidth=2, label=name)
            ax.axhline(true_val, linestyle='--', alpha=0.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Value')
        ax.set_title('Parameter History')
        ax.legend()
        ax.grid(True, alpha=0.3)

        final_loss = result['history']['loss'][-1]
        fig.suptitle(
            f'Bouncing Balls Optimization ({n_balls} balls, {len(sol_true.segments)} segments)\n'
            f'Final Loss: {final_loss:.6e}',
            fontsize=14
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])

        output_path = os.path.join(root_dir, 'results', f'optimization_result_pytorch_bouncing_balls_N{n_balls}.png')
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"\n  Plot saved to: {output_path}")

        if generate_animation:
            print("\n  Generating animation...")
            create_animation(
                times_opt_np, traj_opt_np, traj_true_np,
                x_min, x_max, y_min, y_max,
                n_balls=n_balls,
                filename=os.path.join(root_dir, 'results', f'animation_pytorch_balls_N{n_balls}.mp4'),
            )
    except ImportError:
        print("\n  Matplotlib not available - skipping plots")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

    # Prediction-error series: ‖p_iter − p_true‖ on the optimized
    # parameters only, evaluated at every iteration. result['history']
    # ['params'] already stores the optimized-param vector per iter
    # (in `optimize_params` order).
    p_true_opt = np.array([p_true[name] for name in optimize_params], dtype=float)
    prediction_error_history = [
        float(np.linalg.norm(np.asarray(p_iter, dtype=float) - p_true_opt))
        for p_iter in result['history'].get('params', [])
    ]

    return {
        'method': 'pytorch_multi_N',
        'n_balls': n_balls,
        'ncp': ncp,
        'avg_iter_time': result['avg_iter_time'],
        'p_opt': p_opt_subset,
        'p_true': p_true_subset,
        'opt_param_names': list(optimize_params),
        'final_validation_loss': float(val_mse),
        'iterations': result['n_iter'],
        'converged': result['converged'],
        'prediction_error_history': prediction_error_history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-ball PyTorch optimizer test")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=os.path.join('config', 'config_bouncing_balls_N15.yaml'),
        help='Path to configuration YAML file (relative to project root or absolute).',
    )
    args = parser.parse_args()
    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(root_dir, cfg_path)
    cfg = load_config(cfg_path)
    run_bouncing_balls_test(cfg, os.path.dirname(cfg_path))
