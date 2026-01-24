"""
PyTorch-based DAE Optimizer with Events.

Uses forward-mode automatic differentiation through PyTorch's autograd
to compute gradients of the loss with respect to parameters.

The optimizer uses torchdiffeq's odeint_event for handling discontinuities.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from torchdiffeq import odeint, odeint_event


class BouncingBallModel(nn.Module):
    """
    Bouncing ball model for PyTorch-based optimization.

    States: h (height), v (velocity)
    Parameters: g (gravity), e (restitution coefficient)
    Event: when h < 0, reinit v = -e * prev(v)
    """

    def __init__(self, g: float = 9.81, e: float = 0.8, h0: float = 1.0, v0: float = 0.0,
                 ncp: int = 150):
        """
        Args:
            g: Gravity constant
            e: Restitution coefficient
            h0: Initial height
            v0: Initial velocity
            ncp: Number of collocation points per segment (between events)
        """
        super().__init__()
        # Optimizable parameters
        self.g = nn.Parameter(torch.tensor([g], dtype=torch.float64))
        self.e = nn.Parameter(torch.tensor([e], dtype=torch.float64))

        # Initial conditions (can also be made optimizable)
        self.h0 = h0
        self.v0 = v0

        # Number of collocation points per segment
        self.ncp = ncp

    def forward(self, t, state):
        """ODE right-hand side: dh/dt = v, dv/dt = -g"""
        h, v = state
        dh = v
        dv = -self.g
        return dh, dv

    def event_fn(self, t, state):
        """Event function: triggers when h crosses zero from above."""
        h, v = state
        return h

    def state_update(self, state):
        """State update at event: v_new = -e * v_old"""
        h, v = state
        # Small epsilon to avoid immediate re-trigger
        h_new = h + 1e-7
        v_new = -self.e * v
        return (h_new, v_new)

    def get_initial_state(self):
        """Return initial time and state."""
        t0 = torch.tensor([0.0], dtype=torch.float64)
        h0 = torch.tensor([self.h0], dtype=torch.float64)
        v0 = torch.tensor([self.v0], dtype=torch.float64)
        return t0, (h0, v0)

    def simulate(self, t_end: float, nbounces: int = 10):
        """
        Simulate the bouncing ball with events.

        Args:
            t_end: End time for simulation
            nbounces: Maximum number of bounces to simulate

        Returns:
            times: Tensor of time points
            trajectory_h: Height trajectory
            trajectory_v: Velocity trajectory
            event_times: List of event (bounce) times
        """
        t0, state = self.get_initial_state()

        all_times = [t0.reshape(-1)]
        all_h = [state[0].reshape(-1)]
        all_v = [state[1].reshape(-1)]
        event_times = []

        current_t = t0

        for i in range(nbounces):
            if float(current_t) >= t_end:
                break

            try:
                event_t, solution = odeint_event(
                    self,
                    state,
                    current_t,
                    event_fn=self.event_fn,
                    reverse_time=False,
                    atol=1e-8,
                    rtol=1e-8,
                )

                # Check if event occurred before t_end
                if event_t.detach().item() > t_end:
                    # Integrate to t_end instead
                    tt = torch.linspace(float(current_t), t_end, self.ncp, dtype=torch.float64)
                    if len(tt) > 1:
                        sol = odeint(self, state, tt, method='midpoint')
                        all_times.append(tt[1:])
                        all_h.append(sol[0][1:].reshape(-1))
                        all_v.append(sol[1][1:].reshape(-1))
                    break

                event_times.append(event_t)

                # Dense output between current_t and event_t
                event_t_val = event_t.detach().item()
                current_t_val = current_t.detach().item() if hasattr(current_t, 'detach') else float(current_t)
                tt = torch.linspace(current_t_val, event_t_val, self.ncp, dtype=torch.float64)

                if len(tt) > 1:
                    sol = odeint(self, state, tt, method='midpoint')
                    all_times.append(tt[1:])
                    all_h.append(sol[0][1:].reshape(-1))
                    all_v.append(sol[1][1:].reshape(-1))

                # Update state at event
                state = self.state_update(tuple(s[-1] for s in solution))
                current_t = event_t

            except Exception as ex:
                # No more events, integrate to end
                current_t_val = current_t.detach().item() if hasattr(current_t, 'detach') else float(current_t)
                if current_t_val < t_end:
                    tt = torch.linspace(float(current_t), t_end, self.ncp, dtype=torch.float64)
                    if len(tt) > 1:
                        sol = odeint(self, state, tt, method='midpoint')
                        all_times.append(tt[1:])
                        all_h.append(sol[0][1:].reshape(-1))
                        all_v.append(sol[1][1:].reshape(-1))
                break

        # Final segment if needed
        current_t_val = current_t.detach().item() if hasattr(current_t, 'detach') else float(current_t)
        if current_t_val < t_end and len(event_times) > 0:
            tt = torch.linspace(float(current_t), t_end, self.ncp, dtype=torch.float64)
            if len(tt) > 1:
                sol = odeint(self, state, tt, method='midpoint')
                all_times.append(tt[1:])
                all_h.append(sol[0][1:].reshape(-1))
                all_v.append(sol[1][1:].reshape(-1))

        times = torch.cat(all_times)
        trajectory_h = torch.cat(all_h)
        trajectory_v = torch.cat(all_v)

        return times, trajectory_h, trajectory_v, event_times


class DAEOptimizerPyTorch:
    """
    PyTorch-based optimizer for DAEs with events.

    Uses forward simulation with autograd to compute gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        optimize_params: List[str],
        verbose: bool = True,
        nbounces: int = 10
    ):
        """
        Args:
            model: PyTorch model with simulate() method
            optimize_params: List of parameter names to optimize
            verbose: Print progress
            nbounces: Maximum number of event segments
        """
        self.model = model
        self.optimize_params = optimize_params
        self.verbose = verbose
        self.nbounces = nbounces

        # Get optimizable parameters
        self.param_dict = {name: param for name, param in model.named_parameters()}
        self.opt_params = [self.param_dict[name] for name in optimize_params]

    def predict_outputs(self, t_end: float, target_times: np.ndarray) -> np.ndarray:
        """
        Predict outputs at target times.

        Args:
            t_end: End time for simulation
            target_times: Times at which to evaluate outputs

        Returns:
            Predicted outputs (height values) at target times
        """
        with torch.no_grad():
            times, h_traj, v_traj, _ = self.model.simulate(t_end, self.nbounces)

            # Interpolate to target times
            times_np = times.numpy()
            h_np = h_traj.numpy()

            h_interp = np.interp(target_times, times_np, h_np)

        return h_interp

    def _compute_loss(self, t_end: float, target_times: torch.Tensor,
                      target_outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute loss by simulating and comparing to targets.
        """
        times, h_traj, v_traj, _ = self.model.simulate(t_end, self.nbounces)

        # Interpolate predictions to target times
        # Use differentiable interpolation
        y_pred = self._differentiable_interp(times, h_traj, target_times)

        # MSE loss
        loss = torch.mean((y_pred - target_outputs) ** 2)
        return loss

    def _differentiable_interp(self, x: torch.Tensor, y: torch.Tensor,
                                x_new: torch.Tensor) -> torch.Tensor:
        """
        Differentiable linear interpolation.
        """
        # Sort if needed (should already be sorted)
        sorted_indices = torch.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        # Find indices for interpolation
        indices = torch.searchsorted(x_sorted, x_new, right=True) - 1
        indices = torch.clamp(indices, 0, len(x_sorted) - 2)

        # Linear interpolation
        x0 = x_sorted[indices]
        x1 = x_sorted[indices + 1]
        y0 = y_sorted[indices]
        y1 = y_sorted[indices + 1]

        # Avoid division by zero
        dx = x1 - x0
        dx = torch.where(dx.abs() < 1e-12, torch.ones_like(dx) * 1e-12, dx)

        t = (x_new - x0) / dx
        t = torch.clamp(t, 0.0, 1.0)

        return y0 + t * (y1 - y0)

    def optimization_step(
        self,
        t_span: Tuple[float, float],
        target_times: np.ndarray,
        target_outputs: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Single optimization step: forward pass + gradient computation.

        Returns:
            loss: Scalar loss value
            grad: Gradient w.r.t. optimized parameters
        """
        t_end = t_span[1]

        # Convert to tensors
        target_times_t = torch.tensor(target_times, dtype=torch.float64)
        target_outputs_t = torch.tensor(target_outputs, dtype=torch.float64)

        # Zero gradients
        for param in self.opt_params:
            if param.grad is not None:
                param.grad.zero_()

        # Forward pass and loss
        loss = self._compute_loss(t_end, target_times_t, target_outputs_t)

        # Backward pass
        loss.backward()

        # Extract gradients
        grads = []
        for param in self.opt_params:
            if param.grad is not None:
                grads.append(param.grad.detach().numpy().flatten())
            else:
                grads.append(np.zeros(param.shape).flatten())

        grad = np.concatenate(grads)

        return float(loss.detach()), grad

    def optimize(
        self,
        t_span: Tuple[float, float],
        target_times: np.ndarray,
        target_outputs: np.ndarray,
        max_iterations: int = 100,
        step_size: float = 0.01,
        tol: float = 1e-6,
        print_every: int = 10,
        algorithm: str = 'adam',
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ) -> Dict:
        """
        Run optimization loop.

        Args:
            t_span: (t_start, t_end)
            target_times: Target measurement times
            target_outputs: Target values at those times
            max_iterations: Max iterations
            step_size: Learning rate
            tol: Gradient norm tolerance
            print_every: Print interval
            algorithm: 'sgd' or 'adam'
            beta1: Adam beta1 parameter (default 0.9)
            beta2: Adam beta2 parameter (default 0.999)
            epsilon: Adam epsilon parameter (default 1e-8)

        Returns:
            Dictionary with results
        """
        # Setup optimizer with explicit parameters to match JAX implementation
        if algorithm.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.opt_params, lr=step_size, betas=(beta1, beta2), eps=epsilon
            )
        else:
            optimizer = torch.optim.SGD(self.opt_params, lr=step_size)

        history = {'loss': [], 'gradient_norm': [], 'params': []}

        # Get initial param values
        p_init = np.concatenate([p.detach().numpy().flatten() for p in self.opt_params])

        if self.verbose:
            print(f"\nStarting optimization")
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
        target_outputs_t = torch.tensor(target_outputs, dtype=torch.float64)
        t_end = t_span[1]

        for it in range(max_iterations):
            iter_start = time.time()

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            loss = self._compute_loss(t_end, target_times_t, target_outputs_t)

            # Backward pass
            loss.backward()

            # Get gradient norm
            grad_norm = 0.0
            for param in self.opt_params:
                if param.grad is not None:
                    grad_norm += float(torch.sum(param.grad ** 2))
            grad_norm = np.sqrt(grad_norm)

            # Record history
            loss_val = float(loss.detach())
            history['loss'].append(loss_val)
            history['gradient_norm'].append(grad_norm)

            p_current = np.concatenate([p.detach().numpy().flatten() for p in self.opt_params])
            history['params'].append(p_current.copy())

            iter_time = time.time() - iter_start

            if it % print_every == 0 or it == max_iterations - 1:
                param_str = ", ".join([f"{name}={p.item():.6f}"
                                       for name, p in zip(self.optimize_params, self.opt_params)])
                print(f"  Iter {it:4d}: Loss = {loss_val:.6e}, |grad| = {grad_norm:.6e}, "
                      f"t_iter = {iter_time:.3f}s, {param_str}")

            if grad_norm < tol:
                print(f"\nConverged at iteration {it}")
                converged = True
                break

            # Update parameters
            optimizer.step()

            # Clamp parameters to valid ranges
            with torch.no_grad():
                # Ensure e is in (0, 1) for physical restitution
                if hasattr(self.model, 'e'):
                    self.model.e.data.clamp_(0.01, 0.99)
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
            'converged': converged
        }
