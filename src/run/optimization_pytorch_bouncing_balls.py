"""
PyTorch optimization for 3 bouncing balls with multiple event sources.

Uses the bouncing_balls.py model and dae_optimizer_pytorch.py infrastructure.
Optimizes e_g (wall/ground restitution) and e_b (ball-ball restitution).
"""

import os
import sys
import argparse
import yaml
import numpy as np
import json
import torch

torch.set_default_dtype(torch.float64)

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from src.pytorch.bouncing_balls import BouncingBallsModel
from src.discrete_adjoint.dae_solver import DAESolver


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_loss_targets(sol, n_x):
    """Extract interior target times/data from solution (matches JAX version)."""
    all_t = []
    all_x = []

    for seg in sol.segments:
        if len(seg.t) > 0:
            all_t.append(seg.t)
            all_x.append(seg.x)

    if not all_t:
        return np.array([]), np.array([])

    # Concatenate all segments except last point of each
    target_times = np.concatenate([np.array(t[:-1]) for t in all_t])
    target_data = np.concatenate([np.array(x[:-1]) for x in all_x])
    return target_times, target_data


class DAEOptimizerPyTorchMultiEvent:
    """
    PyTorch-based optimizer for DAEs with multiple event sources.
    Simplified version focusing on bouncing balls.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimize_params: list,
        verbose: bool = True,
        max_events: int = 50
    ):
        """
        Args:
            model: PyTorch model with simulate() method
            optimize_params: List of parameter names to optimize
            verbose: Print progress
            max_events: Maximum number of events per simulation
        """
        self.model = model
        self.optimize_params = optimize_params
        self.verbose = verbose
        self.max_events = max_events

        # Get optimizable parameters
        self.param_dict = {name: param for name, param in model.named_parameters()}
        self.opt_params = [self.param_dict[name] for name in optimize_params]

    def _differentiable_interp(self, x: torch.Tensor, y: torch.Tensor,
                                x_new: torch.Tensor) -> torch.Tensor:
        """Differentiable linear interpolation."""
        # Sort if needed
        sorted_indices = torch.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        # Find indices for interpolation
        indices = torch.searchsorted(x_sorted, x_new, right=True) - 1
        indices = torch.clamp(indices, 0, len(x_sorted) - 2)

        # Linear interpolation
        x0 = x_sorted[indices]
        x1 = x_sorted[indices + 1]
        y0 = y_sorted[sorted_indices[indices]]
        y1 = y_sorted[sorted_indices[indices + 1]]

        # Avoid division by zero
        dx = x1 - x0
        dx = torch.where(dx.abs() < 1e-12, torch.ones_like(dx) * 1e-12, dx)

        t = (x_new - x0) / dx
        t = torch.clamp(t, 0.0, 1.0)

        return y0 + t * (y1 - y0)

    def _compute_loss(
        self,
        target_times: torch.Tensor,
        target_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss by simulating directly at target times (no interpolation).
        Uses simulate_at_targets which follows the torchdiffeq reference pattern.

        Only matches position states (x, y) which are continuous across events.
        Velocity states are discontinuous at event boundaries and cause spurious
        errors when PyTorch and DAE event times don't exactly match.
        """
        y_pred = self.model.simulate_at_targets(target_times)
        # Position indices: x1=0, y1=1, x2=4, y2=5, x3=8, y3=9
        pos_idx = [0, 1, 4, 5, 8, 9]
        loss = torch.mean((y_pred[:, pos_idx] - target_data[:, pos_idx]) ** 2)
        return loss

    def optimize(
        self,
        t_span: tuple,
        target_times: np.ndarray,
        target_data: np.ndarray,
        max_iterations: int = 100,
        step_size: float = 0.001,
        tol: float = 1e-6,
        print_every: int = 10,
        algorithm: str = 'adam',
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ) -> dict:
        """
        Run optimization loop.

        Args:
            t_span: (t_start, t_end)
            target_times: Target measurement times (N,)
            target_data: Target states at those times (N, 12)
            max_iterations: Max iterations
            step_size: Learning rate
            tol: Gradient norm tolerance
            print_every: Print interval
            algorithm: 'sgd' or 'adam'
            beta1, beta2, epsilon: Adam parameters

        Returns:
            Dictionary with results
        """
        import time

        # Setup optimizer
        if algorithm.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.opt_params, lr=step_size, betas=(beta1, beta2), eps=epsilon
            )
        else:
            optimizer = torch.optim.SGD(self.opt_params, lr=step_size)

        history = {'loss': [], 'gradient_norm': [], 'params': [], 'n_segments': []}

        # Get initial param values
        p_init = np.concatenate([p.detach().numpy().flatten() for p in self.opt_params])

        if self.verbose:
            print(f"\nStarting PyTorch optimization")
            print(f"  Algorithm: {algorithm}")
            print(f"  Max iterations: {max_iterations}")
            print(f"  Step size: {step_size}")
            print(f"  Parameters: {self.optimize_params}")
            print(f"  Initial values: {p_init}")
            print()

        start_time = time.time()
        converged = False

        # Convert targets to tensors (once)
        target_times_t = torch.tensor(target_times, dtype=torch.float64)
        target_data_t = torch.tensor(target_data, dtype=torch.float64)

        iter_times = []

        for it in range(max_iterations):
            iter_start = time.time()

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            loss = self._compute_loss(target_times_t, target_data_t)

            # Backward pass
            loss.backward()

            # Get gradient norm
            grad_norm = 0.0
            for param in self.opt_params:
                if param.grad is not None:
                    grad_norm += float(torch.sum(param.grad ** 2))
            grad_norm = np.sqrt(grad_norm)

            # # Count segments (for diagnostics)
            # with torch.no_grad():
            #     _, _, event_times, _ = self.model.simulate(t_end, self.max_events)
            #     n_segments = len(event_times) + 1

            # Record history
            loss_val = float(loss.detach())
            history['loss'].append(loss_val)
            history['gradient_norm'].append(grad_norm)
            # history['n_segments'].append(n_segments)

            p_current = np.concatenate([p.detach().numpy().flatten() for p in self.opt_params])
            history['params'].append(p_current.copy())

            iter_time = (time.time() - iter_start) * 1000.0  # ms
            iter_times.append(iter_time)

            if it % print_every == 0 or it == 0:
                param_str = " ".join([f"{p.item():.4f}" for p in self.opt_params])
                print(f"  Iter {it:4d} | loss={loss_val:.6e} | |grad|={grad_norm:.6e} | "
                      f"p=[{param_str}] | {iter_time:.1f} ms ")

                # Debug: print individual gradients
                if it == 0:
                    for name, param in zip(self.optimize_params, self.opt_params):
                        if param.grad is not None:
                            print(f"    grad[{name}] = {param.grad.item():.6e}")

            # Update parameters first
            optimizer.step()

            # Check convergence after update
            if grad_norm < tol and it > 0:
                print(f"\nConverged at iteration {it}")
                converged = True
                break

            # # Clamp parameters to valid ranges
            with torch.no_grad():
                # Ensure restitution coefficients are in (0, 1)
                if hasattr(self.model, 'e_g'):
                    self.model.e_g.data.clamp_(0.001, 5.0)
                if hasattr(self.model, 'e_b'):
                    self.model.e_b.data.clamp_(0.001, 5.0)
                # Ensure g is positive
                if hasattr(self.model, 'g'):
                    self.model.g.data.clamp_(0.001, 100.0)

        elapsed = time.time() - start_time

        # Final parameters
        p_final = np.concatenate([p.detach().numpy().flatten() for p in self.opt_params])
        
        # Calculate average iteration time (excluding first)
        if len(iter_times) > 1:
            avg_iter_time = sum(iter_times[1:]) / (len(iter_times) - 1)
        else:
            avg_iter_time = 0.0

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
            'avg_iter_time': avg_iter_time
        }



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


def run_bouncing_balls_test(config: dict):
    print("=" * 80)
    print("Bouncing Balls (3 balls) - PyTorch Optimizer Test")
    print("=" * 80)

    # Extract config sections
    solver_cfg = config['dae_solver']
    opt_cfg = config['optimizer']
    algo_cfg = opt_cfg.get('algorithm', {})
    generate_animation = config.get('generate_animation', False)

    # Load DAE specification for true parameter values
    json_path = solver_cfg['dae_specification_file']
    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    print(f"\nLoaded DAE from: {json_path}")
    print(f"  States: {len(dae_data['states'])} (12 states: 3 balls × 4 vars each)")
    print(f"  Parameters: {[p['name'] for p in dae_data['parameters']]}")
    print(f"  Events: {len(dae_data['when'])} event sources")

    # True parameters
    p_true = {p['name']: p['value'] for p in dae_data['parameters']}
    g_true = p_true['g']
    e_g_true = p_true['e_g']
    e_b_true = p_true['e_b']
    d_sq_true = p_true['d_sq']
    x_min, x_max = p_true['x_min'], p_true['x_max']
    y_min, y_max = p_true['y_min'], p_true['y_max']

    print(f"\nTrue parameters:")
    print(f"  g={g_true}, e_g={e_g_true}, e_b={e_b_true}")
    print(f"  Box: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], d_sq={d_sq_true}")

    # Initial conditions from DAE spec
    initial_state = [s['start'] for s in dae_data['states']]
    print(f"  Initial state: {initial_state}")

    # =========================================================================
    # Step 1: Generate reference trajectory with true parameters using DAESolver
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 1: Generate Reference Trajectory (using DAESolver)")
    print("-" * 40)

    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    ncp = solver_cfg.get('ncp', 600)
    max_events = opt_cfg.get('max_segments', 50) * 2  # conservative

    # Use DAESolver for ground truth (matches JAX implementation)
    true_p = [p['value'] for p in dae_data['parameters']]
    solver = DAESolver(dae_data, verbose=False)
    solver.update_parameters(true_p)

    print(f"  Simulating with DAESolver (true parameters)...")
    sol_true = solver.solve_augmented(t_span, ncp=ncp)

    n_x = len(dae_data['states'])
    t_target, y_target = prepare_loss_targets(sol_true, n_x)

    delta_t = t_target[1:] - t_target[:-1]
    print(f"  Simulation time: {t_span}")
    print(f"  Number of segments: {len(sol_true.segments)}")
    print(f"  Target data points: {len(t_target)}")
    print(f"  Delta t min: {np.min(delta_t):.6e}")

    # =========================================================================
    # Step 2: Create initial guess with perturbed parameters
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 2: Create Perturbed Initial Guess")
    print("-" * 40)

    # Perturb parameters
    g_init = g_true + 1.0  # Keep g fixed
    e_g_init = e_g_true + 0.1  # Bias e_g
    e_b_init = e_b_true + 0.1  # Bias e_b

    optimize_params = opt_cfg['opt_params']

    for p_name in ['g', 'e_g', 'e_b']:
        val_true = p_true[p_name]
        if p_name == 'g':
            val_init = g_init
        elif p_name == 'e_g':
            val_init = e_g_init
        else:
            val_init = e_b_init

        is_opt = p_name in optimize_params
        status = "OPTIMIZED" if is_opt else "FIXED"
        diff = val_init - val_true
        print(f"  {p_name}: True={val_true:.4f}, Init={val_init:.4f} (diff={diff:+.4f}) -> {status}")

    # =========================================================================
    # Step 3: Create optimizer and run optimization
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 3: Run Optimization")
    print("-" * 40)

    # Create model with perturbed parameters
    model_opt = BouncingBallsModel(
        g=g_init, e_g=e_g_init, e_b=e_b_init, d_sq=d_sq_true,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        initial_state=initial_state, ncp=ncp
    )

    # Create optimizer
    optimizer = DAEOptimizerPyTorchMultiEvent(
        model=model_opt,
        optimize_params=optimize_params,
        verbose=True,
        max_events=max_events
    )

    # Optimization settings from config
    algo_params = algo_cfg.get('params', {})
    step_size = algo_params.get('step_size', 0.001)
    algorithm_type = algo_cfg.get('type', 'adam').lower()
    beta1 = algo_params.get('beta1', 0.9)
    beta2 = algo_params.get('beta2', 0.999)
    epsilon = algo_params.get('epsilon', 1e-8)

    result = optimizer.optimize(
        t_span=t_span,
        target_times=t_target,
        target_data=y_target,
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
    print("\n" + "=" * 70)
    print("Optimization Result")
    print("=" * 70)
    
    # Construct mappings
    p_true_subset = {k: p_true[k] for k in optimize_params}
    
    # Initial values map
    p_init_subset = {}
    for name in optimize_params:
         if name == 'g': p_init_subset[name] = g_init
         elif name == 'e_g': p_init_subset[name] = e_g_init
         else: p_init_subset[name] = e_b_init
            
    # Optimized values map
    # params_final is flat list/array from optimizer result
    p_opt_subset = {}
    for i, name in enumerate(optimize_params):
        p_opt_subset[name] = float(result['params'][i])

    print(f"True params:      {p_true_subset}")
    print(f"Initial params:   {p_init_subset}")
    print(f"Optimized params: {p_opt_subset}")
    print(f"Iterations:       {result['n_iter']}")
    print(f"Converged:        {result['converged']}")
    print(f"Final loss:       {result['history']['loss'][-1]:.6e}")
    # Gradient norm is tracked in history
    print(f"Final |grad|:     {result['history']['gradient_norm'][-1]:.6e}")
    if 'avg_iter_time' in result:
        print(f"Avg iter time:    {result['avg_iter_time']:.2f} ms")

    # Per-parameter error
    for name in optimize_params:
        true_val = p_true[name]
        opt_val = p_opt_subset[name]
        err = abs(opt_val - true_val)
        print(f"  {name}: true={true_val:.4f}  opt={opt_val:.4f}  err={err:.6e}")

    # =========================================================================
    # Step 5: Validate by re-simulating with PyTorch model
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 5: Validation")
    print("-" * 40)

    # Simulate with optimized parameters using simulate_at_targets
    target_times_t = torch.tensor(t_target, dtype=torch.float64)
    with torch.no_grad():
        y_pred = model_opt.simulate_at_targets(target_times_t)
        
    y_pred_np = y_pred.numpy()
    
    # Calculate MSE on positions (indices 0, 1, 4, 5, 8, 9)
    # y_target contains all states (N, 12) per prepare_loss_targets in this file
    pos_idx = [0, 1, 4, 5, 8, 9]
    val_mse = np.mean((y_pred_np[:, pos_idx] - y_target[:, pos_idx])**2)
    print(f"Final Loss (PyTorch, Positions only): {val_mse:.6e}")
    
    # Simulate densly for plot
    t_end_val = float(t_target[-1]) + 1e-6
    with torch.no_grad():
        times_opt, traj_opt = model_opt.simulate_fixed_grid(t_end_val, n_points=ncp)

    times_opt_np = times_opt.numpy()
    traj_opt_np = traj_opt.numpy()

    # Trajectory relative error calculation (keep existing metric too if desired, or skip)
    # The existing code did interpolation for y_opt_pos. We now have y_pred_np exact.
    y_opt_pos = y_pred_np[:, pos_idx]
    y_tgt_pos = y_target[:, pos_idx]
    traj_error = np.linalg.norm(y_opt_pos - y_tgt_pos) / np.linalg.norm(y_tgt_pos)
    print(f"  Trajectory relative error (positions): {traj_error:.6e}")

    # Extract true trajectory from DAESolver segments for plotting
    all_t_true = []
    all_x_true = []
    for seg in sol_true.segments:
        if len(seg.t) > 0:
            all_t_true.append(seg.t)
            all_x_true.append(seg.x)
    times_true_np = np.concatenate([np.array(t) for t in all_t_true])
    traj_true_np = np.concatenate([np.array(x) for x in all_x_true])

    # =========================================================================
    # Step 6: Plot results
    # =========================================================================
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))

        # Plot x-y trajectories for each ball
        for ball_idx in range(3):
            ax = axes[ball_idx, 0]
            base = ball_idx * 4
            x_true = traj_true_np[:, base]
            y_true = traj_true_np[:, base + 1]
            x_opt = traj_opt_np[:, base]
            y_opt = traj_opt_np[:, base + 1]

            ax.plot(x_true, y_true, 'b-', linewidth=2, label='True')
            ax.plot(x_opt, y_opt, 'r--', linewidth=2, label='Optimized')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_title(f'Ball {ball_idx + 1} Trajectory')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])

        # Plot y-position vs time for each ball
        for ball_idx in range(3):
            ax = axes[ball_idx, 1]
            base = ball_idx * 4
            y_true = traj_true_np[:, base + 1]
            y_opt = traj_opt_np[:, base + 1]

            ax.plot(times_true_np, y_true, 'b-', linewidth=2, label='True')
            ax.plot(times_opt_np, y_opt, 'r--', linewidth=2, label='Optimized')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('y [m]')
            ax.set_title(f'Ball {ball_idx + 1} Height vs Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Loss history
        ax = axes[0, 2]
        ax.semilogy(result['history']['loss'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Loss History')
        ax.grid(True, alpha=0.3)

        # Gradient norm history
        ax = axes[1, 2]
        ax.semilogy(result['history']['gradient_norm'], 'r-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm History')
        ax.grid(True, alpha=0.3)

        # Parameter history
        ax = axes[2, 2]
        param_hist = np.array(result['history']['params'])
        for i, name in enumerate(optimize_params):
            true_val = p_true[name]
            ax.plot(param_hist[:, i], linewidth=2, label=f'{name}')
            ax.axhline(true_val, linestyle='--', alpha=0.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Value')
        ax.set_title('Parameter History')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(root_dir, 'results', 'optimization_result_pytorch_bouncing_balls.png')
        plt.savefig(output_path, dpi=150)
        print(f"\n  Plot saved to: {output_path}")
        
        if generate_animation:
            # Create Animation
            print("\n  Generating animation...")
            create_animation(
                times_opt_np, traj_opt_np, traj_true_np,
                x_min, x_max, y_min, y_max,
            filename=os.path.join(root_dir, 'results', 'animation_pytorch_balls.mp4')
        )

    except ImportError:
        print("\n  Matplotlib not available - skipping plots")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

    # Return benchmark metrics
    avg_iter_time = result.get('avg_iter_time', 0.0) 
    # The optimization_pytorch_bouncing_balls.py uses DAEOptimizerPyTorchMultiEvent locally defined (checking imports...)
    # Wait, Step 4 view file shows DAEOptimizerPyTorchMultiEvent defined LOCALLY in the file.
    # Lines 53-282. It computes result['avg_iter_time'] at line 266-267!
    # "if len(iter_times) > 1: avg_iter_time = sum(iter_times[1:]) ... else 0.0"
    # So I don't need to manually compute it here. It is already in result.
    
    benchmark_results = {
        'method': 'pytorch_multi',
        'ncp': ncp,
        'avg_iter_time': avg_iter_time, 
        'p_opt': p_opt_subset,
        'p_true': p_true_subset,
        'final_validation_loss': float(val_mse),
        'iterations': result['n_iter'],
        'converged': result['converged']
    }
    return benchmark_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bouncing Balls PyTorch Optimizer Test")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config_bouncing_balls.yaml',
        help='Path to configuration YAML file'
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_bouncing_balls_test(config)
