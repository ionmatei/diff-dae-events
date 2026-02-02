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

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

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
        t_end: float,
        target_times: torch.Tensor,
        target_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss by simulating and interpolating (like single ball example).
        """
        # Simulate with events
        times, trajectory = self.model.simulate_fixed_grid(t_end, n_points=500)

        # Interpolate predictions to target times (for all 12 states)
        n_states = 12
        y_pred = torch.zeros(len(target_times), n_states, dtype=torch.float64)

        for i in range(n_states):
            y_pred[:, i] = self._differentiable_interp(
                times, trajectory[:, i], target_times
            )

        # MSE loss
        loss = torch.mean((y_pred - target_data) ** 2)
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
        t_end = t_span[1]

        for it in range(max_iterations):
            iter_start = time.time()

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            loss = self._compute_loss(t_end, target_times_t, target_data_t)

            # Backward pass
            loss.backward()

            # Get gradient norm
            grad_norm = 0.0
            for param in self.opt_params:
                if param.grad is not None:
                    grad_norm += float(torch.sum(param.grad ** 2))
            grad_norm = np.sqrt(grad_norm)

            # Count segments (for diagnostics)
            with torch.no_grad():
                _, _, event_times, _ = self.model.simulate(t_end, self.max_events)
                n_segments = len(event_times) + 1

            # Record history
            loss_val = float(loss.detach())
            history['loss'].append(loss_val)
            history['gradient_norm'].append(grad_norm)
            history['n_segments'].append(n_segments)

            p_current = np.concatenate([p.detach().numpy().flatten() for p in self.opt_params])
            history['params'].append(p_current.copy())

            iter_time = (time.time() - iter_start) * 1000.0  # ms

            if it % print_every == 0 or it == 0:
                param_str = " ".join([f"{p.item():.4f}" for p in self.opt_params])
                print(f"  Iter {it:4d} | loss={loss_val:.6e} | |grad|={grad_norm:.6e} | "
                      f"p=[{param_str}] | {iter_time:.1f} ms | segs={n_segments}")

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

            # Clamp parameters to valid ranges
            with torch.no_grad():
                # Ensure restitution coefficients are in (0, 1)
                if hasattr(self.model, 'e_g'):
                    self.model.e_g.data.clamp_(0.01, 0.99)
                if hasattr(self.model, 'e_b'):
                    self.model.e_b.data.clamp_(0.01, 0.99)
                # Ensure g is positive
                if hasattr(self.model, 'g'):
                    self.model.g.data.clamp_(0.1, 100.0)

        elapsed = time.time() - start_time

        # Final parameters
        p_final = np.concatenate([p.detach().numpy().flatten() for p in self.opt_params])

        if self.verbose:
            print(f"\nOptimization complete in {elapsed:.2f}s")
            print(f"  Final loss: {history['loss'][-1]:.6e}")
            print(f"  Final params: {p_final}")

        return {
            'params': p_final,
            'history': history,
            'elapsed_time': elapsed,
            'converged': converged,
            'n_iter': len(history['loss'])
        }


def run_bouncing_balls_test(config: dict):
    print("=" * 80)
    print("Bouncing Balls (3 balls) - PyTorch Optimizer Test")
    print("=" * 80)

    # Extract config sections
    solver_cfg = config['dae_solver']
    opt_cfg = config['optimizer']
    algo_cfg = opt_cfg.get('algorithm', {})

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
    g_init = g_true * 1.0  # Keep g fixed
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
    print("\n" + "-" * 40)
    print("Step 4: Results")
    print("-" * 40)

    # Extract optimized values
    params_final = result['params']

    # Map back to names
    p_opt_dict = {}
    for i, name in enumerate(optimize_params):
        p_opt_dict[name] = params_final[i]

    print(f"\n  Parameter Recovery:")
    for name in optimize_params:
        true_val = p_true[name]
        opt_val = p_opt_dict[name]
        if name == 'g':
            init_val = g_init
        elif name == 'e_g':
            init_val = e_g_init
        else:
            init_val = e_b_init
        error = abs(opt_val - true_val)
        print(f"    {name:4s}: True={true_val:.6f}, Init={init_val:.6f}, Opt={opt_val:.6f}, Error={error:.6e}")

    print(f"\n  Optimization Stats:")
    print(f"    Initial loss: {result['history']['loss'][0]:.6e}")
    print(f"    Final loss:   {result['history']['loss'][-1]:.6e}")
    print(f"    Converged:    {result['converged']}")
    print(f"    Iterations:   {result['n_iter']}")
    print(f"    Time:         {result['elapsed_time']:.2f}s")

    # =========================================================================
    # Step 5: Validate by re-simulating with PyTorch model
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 5: Validation")
    print("-" * 40)

    # Simulate with optimized parameters using PyTorch model
    with torch.no_grad():
        times_opt, traj_opt, event_times_opt, event_indices_opt = model_opt.simulate(
            t_span[1], max_events=max_events
        )

    # Interpolate at target times (all 12 states)
    times_opt_np = times_opt.numpy()
    traj_opt_np = traj_opt.numpy()
    y_opt = np.zeros((len(t_target), 12))

    for i in range(12):
        y_opt[:, i] = np.interp(t_target, times_opt_np, traj_opt_np[:, i])

    traj_error = np.linalg.norm(y_opt - y_target) / np.linalg.norm(y_target)
    print(f"  Trajectory relative error: {traj_error:.6e}")

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

        # Segment count history
        ax = axes[2, 2]
        ax.plot(result['history']['n_segments'], 'g-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of Segments')
        ax.set_title('Segment Count History')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = 'bouncing_balls_pytorch_result.png'
        plt.savefig(output_path, dpi=150)
        print(f"\n  Plot saved to: {output_path}")

    except ImportError:
        print("\n  Matplotlib not available - skipping plots")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

    return result


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
