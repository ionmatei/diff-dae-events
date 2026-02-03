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
        """
        Event function: triggers when h crosses zero from above.
        Returns tensor to match general pattern.
        """
        h, v = state
        return torch.stack([h])  # Return as 1-element tensor

    def state_update(self, state, event_idx=0):
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

    def _bisect_event(self, current_state, t_start, t_hi, ev_idx=0, n_bisect=50):
        """
        Bisection fallback when odeint_event fails.
        Finds the time when event ev_idx crosses zero in [t_start, t_hi].
        """
        t_start_val = float(t_start)
        t_lo_val = t_start_val
        t_hi_val = float(t_hi)

        for _ in range(n_bisect):
            t_mid = (t_lo_val + t_hi_val) / 2.0
            tt = torch.tensor([t_start_val, t_mid], dtype=torch.float64)
            
            # Pack state for odeint
            y0 = current_state
            sol = odeint(self, y0, tt, atol=1e-8, rtol=1e-8)
            
            # Evaluate event
            state_mid = (sol[0][-1], sol[1][-1])
            ev_val = self.event_fn(tt[1], state_mid)[ev_idx]

            if ev_val.item() > 0:
                t_lo_val = t_mid
            else:
                t_hi_val = t_mid

            if (t_hi_val - t_lo_val) < 1e-10:
                break

        # Final integration
        event_t = torch.tensor([t_hi_val], dtype=torch.float64)
        tt_final = torch.tensor([t_start_val, t_hi_val], dtype=torch.float64)
        sol_final = odeint(self, current_state, tt_final, atol=1e-8, rtol=1e-8)
        state_final = (sol_final[0][-1], sol_final[1][-1])
        return event_t, state_final

    def _find_next_event(self, current_state, current_t, t_end):
        """
        Find the next event using probe + odeint_event/bisection.
        ADAPTED from bouncing_balls.py for scalar/single event model.
        """
        current_t_val = current_t.detach().item() if hasattr(current_t, 'detach') else float(current_t)
        if current_t_val >= t_end:
            return None, None, None

        # Step 1: Probe integration
        n_probe = 30
        tt_probe = torch.linspace(current_t_val, t_end, n_probe, dtype=torch.float64)
        with torch.no_grad():
             sol_probe = odeint(self, current_state, tt_probe, atol=1e-6, rtol=1e-6)
             # sol_probe is tuple (h, v), each (N, 1)

        # Check for zero crossing
        candidates = [] # (t_lo, t_hi)
        
        # event_fn returns [h]
        # evaluating manually along probe
        h_vals = sol_probe[0]
        
        for i in range(1, n_probe):
            h_prev = h_vals[i-1]
            h_curr = h_vals[i]
            # Event: h > 0 -> h <= 0
            if h_prev > 0 and h_curr <= 0:
                candidates.append((tt_probe[i-1].item(), tt_probe[i].item()))
        
        if not candidates:
            return None, None, None
            
        # For this simple model, only one event type (0)
        # Process first candidate
        t_lo, t_hi = candidates[0]
        ev_idx = 0
        
        try:
            # Try odeint_event
            def single_event_fn(t, s):
                h, v = s
                return h # Scalar for odeint_event
                
            event_t, solution = odeint_event(
                self,
                current_state,
                current_t,
                event_fn=single_event_fn,
                reverse_time=False,
                atol=1e-8,
                rtol=1e-8
            )
            
            # solution is list of tensors, select last
            state_at_event = (solution[0][-1], solution[1][-1])
            return event_t, ev_idx, state_at_event
            
        except Exception:
            # Fallback to bisection
            try:
                event_t, state_at_event = self._bisect_event(
                    current_state, current_t_val, t_hi, ev_idx
                )
                return event_t, ev_idx, state_at_event
            except:
                return None, None, None

    def simulate_at_targets(self, target_times: torch.Tensor):
        """
        Simulate and evaluate at specific target times (differentiable).
        Right-continuous at events.
        """
        target_np = target_times.detach().numpy()
        if len(target_np) == 0:
             return torch.zeros((0, 2), dtype=torch.float64)
             
        t_end = float(target_np[-1]) + 1e-6
        n_targets = len(target_np)
        eps = 1e-9

        # Step 1: Detect events
        t0, state0 = self.get_initial_state()
        current_t = t0
        current_state = state0
        events = [] # (event_t, event_idx)
        
        for _ in range(20):
            event_t, event_idx, state_at_event = self._find_next_event(
                current_state, current_t, t_end
            )
            if event_t is None:
                break
            events.append((event_t, event_idx))
            current_state = self.state_update(state_at_event, event_idx)
            current_t = event_t

        # Step 2: Partition targets
        event_vals = [et.detach().item() for et, _ in events]
        seg_targets = [[] for _ in range(len(events) + 1)]
        ptr = 0
        for seg_idx, et_val in enumerate(event_vals):
            while ptr < n_targets and target_np[ptr] < et_val - eps:
                seg_targets[seg_idx].append(ptr)
                ptr += 1
        while ptr < n_targets:
            seg_targets[len(events)].append(ptr)
            ptr += 1

        # Step 3: Evaluate per segment
        current_t = t0
        current_state = state0
        indexed_results = [] # (idx, state_tensor_2d)
        
        for seg_idx in range(len(events) + 1):
            ct_val = current_t.detach().item()
            targets = seg_targets[seg_idx]
            
            at_start = []
            after_start = []
            for tidx in targets:
                if target_np[tidx] <= ct_val + eps:
                    at_start.append(tidx)
                else:
                    after_start.append(tidx)
            
            # At start (or immediately post-event)
            # pack current_state to tensor
            cs_tensor = torch.cat([current_state[0], current_state[1]])
            for tidx in at_start:
                indexed_results.append((tidx, cs_tensor))
                
            if seg_idx < len(events):
                event_t, event_idx = events[seg_idx]
                
                # Time grid: [current, targets..., event]
                tt_list = [current_t.reshape(-1)]
                for tidx in after_start:
                    tt_list.append(torch.tensor([target_np[tidx]], dtype=torch.float64))
                tt_list.append(event_t.reshape(-1))
                tt = torch.cat(tt_list)
                
                # odeint returns tuple(h, v) each (N, ...)
                sol = odeint(self, current_state, tt, atol=1e-8, rtol=1e-8)
                
                # Interior points (indices 1 to N-2)
                for i, tidx in enumerate(after_start):
                    # sol[0][i+1], sol[1][i+1]
                    s_h = sol[0][i+1]
                    s_v = sol[1][i+1]
                    indexed_results.append((tidx, torch.cat([s_h, s_v])))
                    
                # Update
                state_end = (sol[0][-1], sol[1][-1])
                current_state = self.state_update(state_end, event_idx)
                current_t = event_t
                
            else:
                # Last segment
                if after_start:
                    tt_list = [current_t.reshape(-1)]
                    for tidx in after_start:
                        tt_list.append(torch.tensor([target_np[tidx]], dtype=torch.float64))
                    tt = torch.cat(tt_list)
                    
                    sol = odeint(self, current_state, tt, atol=1e-8, rtol=1e-8)
                    for i, tidx in enumerate(after_start):
                        s_h = sol[0][i+1]
                        s_v = sol[1][i+1]
                        indexed_results.append((tidx, torch.cat([s_h, s_v])))

        indexed_results.sort(key=lambda x: x[0])
        if not indexed_results:
             return torch.zeros((0, 2), dtype=torch.float64)
             
        # Stack
        return torch.stack([s for _, s in indexed_results])    

    def simulate(self, t_end: float, nbounces: int = 10):
        """
        Simulate for visualization (backward compatible output format).
        """
        # We can implement a simple version or keep logic similar to old one 
        # but using the new _find_next_event for consistency.
        
        t0, state = self.get_initial_state()
        current_t = t0
        current_state = state
        
        all_times = [t0.reshape(-1)]
        all_h = [state[0].reshape(-1)]
        all_v = [state[1].reshape(-1)]
        event_times = []
        
        for i in range(nbounces):
            if float(current_t) >= t_end:
                 break
                 
            event_t, event_idx, state_at_event = self._find_next_event(
                current_state, current_t, t_end
            )
            
            sim_until = event_t if event_t is not None else torch.tensor([t_end], dtype=torch.float64)
            sim_until_val = sim_until.detach().item()
            current_t_val = current_t.detach().item()

            # Dense integration
            n_pts = max(3, int((sim_until_val - current_t_val) * 100)) # e.g. 100 Hz
            tt = torch.linspace(current_t_val, sim_until_val, n_pts, dtype=torch.float64)
            if len(tt) > 1:
                sol = odeint(self, current_state, tt, atol=1e-5, rtol=1e-5)
                # Skip first point to avoid duplicate
                all_times.append(tt[1:])
                all_h.append(sol[0][1:].reshape(-1))
                all_v.append(sol[1][1:].reshape(-1))
            
            if event_t is None:
                break
                
            event_times.append(event_t)
            current_state = self.state_update(state_at_event, event_idx)
            current_t = event_t
            
        times = torch.cat(all_times)
        trajectory_h = torch.cat(all_h)
        trajectory_v = torch.cat(all_v)
        
        return times, trajectory_h, trajectory_v, event_times


class DAEOptimizerPyTorch:
    """
    PyTorch-based optimizer for DAEs with events.
    Now uses simulate_at_targets for correct gradient propagation.
    """

    def __init__(
        self,
        model: nn.Module,
        optimize_params: List[str],
        verbose: bool = True,
        nbounces: int = 10
    ):
        self.model = model
        self.optimize_params = optimize_params
        self.verbose = verbose
        self.nbounces = nbounces # Unused in new logic but kept for compat

        self.param_dict = {name: param for name, param in model.named_parameters()}
        self.opt_params = [self.param_dict[name] for name in optimize_params]

    def _compute_loss(self, t_end: float, target_times: torch.Tensor,
                      target_outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute loss by simulating directly at target times using simulate_at_targets.
        """
        # Get states at target times [N, 2] (h, v)
        y_pred = self.model.simulate_at_targets(target_times)
        
        # Assume target_outputs contains only height (N,)
        # y_pred columns: 0->h, 1->v
        h_pred = y_pred[:, 0]
        
        # MSE loss
        loss = torch.mean((h_pred - target_outputs) ** 2)
        return loss

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
        if algorithm.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.opt_params, lr=step_size, betas=(beta1, beta2), eps=epsilon
            )
        else:
            optimizer = torch.optim.SGD(self.opt_params, lr=step_size)

        history = {'loss': [], 'gradient_norm': [], 'params': []}

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

        target_times_t = torch.tensor(target_times, dtype=torch.float64)
        target_outputs_t = torch.tensor(target_outputs, dtype=torch.float64)
        t_end = t_span[1]

        for it in range(max_iterations):
            iter_start = time.time()
            optimizer.zero_grad()
            
            loss = self._compute_loss(t_end, target_times_t, target_outputs_t)
            loss.backward()

            grad_norm = 0.0
            for param in self.opt_params:
                if param.grad is not None:
                    grad_norm += float(torch.sum(param.grad ** 2))
            grad_norm = np.sqrt(grad_norm)

            loss_val = float(loss.detach())
            history['loss'].append(loss_val)
            history['gradient_norm'].append(grad_norm)

            p_current = np.concatenate([p.detach().numpy().flatten() for p in self.opt_params])
            history['params'].append(p_current.copy())

            iter_time = time.time() - iter_start

            if it % print_every == 0 or it == 0 or it == max_iterations - 1:
                param_str = ", ".join([f"{name}={p.item():.6f}"
                                       for name, p in zip(self.optimize_params, self.opt_params)])
                print(f"  Iter {it:4d}: Loss = {loss_val:.6e}, |grad| = {grad_norm:.6e}, "
                      f"t_iter = {1000*iter_time:.2f} ms, {param_str}")

            if grad_norm < tol:
                print(f"\nConverged at iteration {it}")
                converged = True
                break

            optimizer.step()

            with torch.no_grad():
                if hasattr(self.model, 'e'):
                    self.model.e.data.clamp_(0.01, 2.0)
                if hasattr(self.model, 'g'):
                    self.model.g.data.clamp_(0.1, 100.0)

        elapsed = time.time() - start_time
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
