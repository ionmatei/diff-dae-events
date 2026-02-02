#!/usr/bin/env python3
"""
PyTorch model for 3 bouncing balls with multiple event sources.

States: x1,y1,vx1,vy1,x2,y2,vx2,vy2,x3,y3,vx3,vy3 (12 states)
Parameters: g, e_g (wall/ground restitution), e_b (ball-ball restitution)
Events:
  - Ground/ceiling/wall collisions (12 events, single reinit each)
  - Ball-ball collisions (3 events, 4 reinits each)
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint, odeint_event
from typing import Tuple, List, Optional

torch.set_default_dtype(torch.float64)


class BouncingBallsModel(nn.Module):
    """
    Three bouncing balls in a 2D box with gravity.

    Features:
    - Wall/ceiling/floor collisions (12 events)
    - Ball-ball collisions (3 events with compound reinit)
    """

    def __init__(
        self,
        g: float = 9.81,
        e_g: float = 0.8,  # ground/wall restitution
        e_b: float = 0.9,  # ball-ball restitution
        d_sq: float = 0.1,  # collision distance squared
        x_min: float = 0.0,
        x_max: float = 3.0,
        y_min: float = 0.0,
        y_max: float = 3.0,
        initial_state: Optional[List[float]] = None,
        ncp: int = 50,
        adjoint: bool = True  # Use adjoint method for gradients
    ):
        """
        Args:
            g: Gravity constant
            e_g: Wall/ground restitution coefficient
            e_b: Ball-ball restitution coefficient
            d_sq: Squared collision distance for ball-ball
            x_min, x_max, y_min, y_max: Box boundaries
            initial_state: Initial [x1,y1,vx1,vy1,x2,y2,vx2,vy2,x3,y3,vx3,vy3]
            ncp: Number of collocation points per segment
        """
        super().__init__()

        # Optimizable parameters
        self.g = nn.Parameter(torch.tensor([g], dtype=torch.float64))
        self.e_g = nn.Parameter(torch.tensor([e_g], dtype=torch.float64))
        self.e_b = nn.Parameter(torch.tensor([e_b], dtype=torch.float64))

        # Box geometry (not optimized)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.d_sq = d_sq

        # Initial state
        if initial_state is None:
            # Default from JSON spec
            initial_state = [
                0.5, 2.0, 0.5, 0.3,    # ball 1
                1.5, 1.5, -0.5, 0.4,   # ball 2
                1.0, 0.5, 0.25, -0.5   # ball 3
            ]
        self.initial_state = torch.tensor(initial_state, dtype=torch.float64)

        self.ncp = ncp
        self.odeint = odeint_adjoint if adjoint else odeint

    def forward(self, t, state):
        """
        ODE right-hand side:
        dx/dt = vx, dy/dt = vy, dvx/dt = 0, dvy/dt = -g (for each ball)
        """
        # state is a 12-element tensor: [x1,y1,vx1,vy1,x2,y2,vx2,vy2,x3,y3,vx3,vy3]
        dstate = torch.zeros_like(state)

        for i in range(3):
            base = i * 4
            # x' = vx, y' = vy
            dstate[base] = state[base + 2]      # dx/dt = vx
            dstate[base + 1] = state[base + 3]  # dy/dt = vy
            # vx' = 0, vy' = -g
            dstate[base + 2] = 0.0
            dstate[base + 3] = -self.g

        return dstate

    def event_fn(self, t, state):
        """
        Aggregate event function for all 15 events.
        Returns a 15-element tensor.

        Event ordering (matches JSON spec):
        0-2:   Floor collisions (y1,y2,y3)
        3-5:   Ceiling collisions (y1,y2,y3)
        6-8:   Left wall collisions (x1,x2,x3)
        9-11:  Right wall collisions (x1,x2,x3)
        12:    Ball 1-2 collision
        13:    Ball 1-3 collision
        14:    Ball 2-3 collision
        """
        x1, y1, vx1, vy1 = state[0:4]
        x2, y2, vx2, vy2 = state[4:8]
        x3, y3, vx3, vy3 = state[8:12]

        events = torch.zeros(15, dtype=torch.float64)

        # Floor collisions (0-2)
        events[0] = y1 - self.y_min
        events[1] = y2 - self.y_min
        events[2] = y3 - self.y_min

        # Ceiling collisions (3-5)
        events[3] = self.y_max - y1
        events[4] = self.y_max - y2
        events[5] = self.y_max - y3

        # Left wall collisions (6-8)
        events[6] = x1 - self.x_min
        events[7] = x2 - self.x_min
        events[8] = x3 - self.x_min

        # Right wall collisions (9-11)
        events[9] = self.x_max - x1
        events[10] = self.x_max - x2
        events[11] = self.x_max - x3

        # Ball-ball collisions (12-14)
        events[12] = (x1 - x2) ** 2 + (y1 - y2) ** 2 - self.d_sq
        events[13] = (x1 - x3) ** 2 + (y1 - y3) ** 2 - self.d_sq
        events[14] = (x2 - x3) ** 2 + (y2 - y3) ** 2 - self.d_sq

        return events

    def state_update(self, state, event_idx: int):
        """
        Apply state update for the given event.

        Args:
            state: Current state (12-element tensor)
            event_idx: Which event fired (0-14)

        Returns:
            Updated state
        """
        new_state = state.clone()
        eps = 1e-7  # Small epsilon to avoid immediate re-trigger

        # Floor collisions (0-2): reverse vy with e_g
        if event_idx == 0:  # Ball 1 floor
            new_state[1] += eps
            new_state[3] = -self.e_g * state[3]
        elif event_idx == 1:  # Ball 2 floor
            new_state[5] += eps
            new_state[7] = -self.e_g * state[7]
        elif event_idx == 2:  # Ball 3 floor
            new_state[9] += eps
            new_state[11] = -self.e_g * state[11]

        # Ceiling collisions (3-5): reverse vy with e_g
        elif event_idx == 3:  # Ball 1 ceiling
            new_state[1] -= eps
            new_state[3] = -self.e_g * state[3]
        elif event_idx == 4:  # Ball 2 ceiling
            new_state[5] -= eps
            new_state[7] = -self.e_g * state[7]
        elif event_idx == 5:  # Ball 3 ceiling
            new_state[9] -= eps
            new_state[11] = -self.e_g * state[11]

        # Left wall collisions (6-8): reverse vx with e_g
        elif event_idx == 6:  # Ball 1 left wall
            new_state[0] += eps
            new_state[2] = -self.e_g * state[2]
        elif event_idx == 7:  # Ball 2 left wall
            new_state[4] += eps
            new_state[6] = -self.e_g * state[6]
        elif event_idx == 8:  # Ball 3 left wall
            new_state[8] += eps
            new_state[10] = -self.e_g * state[10]

        # Right wall collisions (9-11): reverse vx with e_g
        elif event_idx == 9:  # Ball 1 right wall
            new_state[0] -= eps
            new_state[2] = -self.e_g * state[2]
        elif event_idx == 10:  # Ball 2 right wall
            new_state[4] -= eps
            new_state[6] = -self.e_g * state[6]
        elif event_idx == 11:  # Ball 3 right wall
            new_state[8] -= eps
            new_state[10] = -self.e_g * state[10]

        # Ball-ball collisions (12-14): reverse all velocities with e_b
        elif event_idx == 12:  # Ball 1-2 collision
            new_state[2] = -self.e_b * state[2]   # vx1
            new_state[3] = -self.e_b * state[3]   # vy1
            new_state[6] = -self.e_b * state[6]   # vx2
            new_state[7] = -self.e_b * state[7]   # vy2
        elif event_idx == 13:  # Ball 1-3 collision
            new_state[2] = -self.e_b * state[2]   # vx1
            new_state[3] = -self.e_b * state[3]   # vy1
            new_state[10] = -self.e_b * state[10] # vx3
            new_state[11] = -self.e_b * state[11] # vy3
        elif event_idx == 14:  # Ball 2-3 collision
            new_state[6] = -self.e_b * state[6]   # vx2
            new_state[7] = -self.e_b * state[7]   # vy2
            new_state[10] = -self.e_b * state[10] # vx3
            new_state[11] = -self.e_b * state[11] # vy3

        return new_state

    def get_initial_state(self):
        """Return initial time and state."""
        t0 = torch.tensor([0.0], dtype=torch.float64)
        return t0, self.initial_state.clone()

    def simulate_fixed_grid(self, t_end: float, n_points: int = 500):
        """
        Simulate with events using adjoint method (differentiable).

        Similar to single ball example, uses odeint_event with adjoint.

        Args:
            t_end: End time
            n_points: Approximate number of points (actual depends on events)

        Returns:
            times: tensor of time points
            trajectory: (N, 12) tensor
        """
        # Get event times using odeint_event
        event_times, event_indices = self.get_event_times(t_end, max_events=20)

        # Build dense trajectory
        t0, state = self.get_initial_state()
        all_times = [t0.reshape(-1)]
        all_states = [state.reshape(1, -1)]

        current_t = t0
        current_state = state

        for event_t, event_idx in zip(event_times, event_indices):
            # Integrate to event
            current_t_val = current_t.detach().item() if hasattr(current_t, 'detach') else float(current_t)
            event_t_val = event_t.detach().item() if hasattr(event_t, 'detach') else float(event_t)

            tt = torch.linspace(
                current_t_val, event_t_val,
                max(2, int((event_t_val - current_t_val) * self.ncp))
            )[1:-1]
            tt = torch.cat([current_t.reshape(-1), tt, event_t.reshape(-1)])

            if len(tt) > 1:
                sol = self.odeint(self, current_state, tt, atol=1e-8, rtol=1e-8)
                all_times.append(tt[1:])
                all_states.append(sol[1:].reshape(-1, 12))

            # Apply state update
            current_state = self.state_update(sol[-1] if len(tt) > 1 else current_state, event_idx)
            current_t = event_t

        # Final segment if needed
        current_t_val = current_t.detach().item() if hasattr(current_t, 'detach') else float(current_t)
        if current_t_val < t_end:
            tt = torch.linspace(current_t_val, t_end, self.ncp)
            if len(tt) > 1:
                sol = self.odeint(self, current_state, tt, atol=1e-8, rtol=1e-8)
                all_times.append(tt[1:])
                all_states.append(sol[1:].reshape(-1, 12))

        times = torch.cat(all_times)
        trajectory = torch.cat(all_states, dim=0)

        return times, trajectory

    def simulate_at_times(self, target_times: torch.Tensor):
        """
        Simulate with events and evaluate at specific target times (differentiable).
        Integrates directly at target times between events, avoiding interpolation.
        """
        t_end = target_times[-1].item()
        event_times, event_indices = self.get_event_times(t_end, max_events=20)

        t0, state = self.get_initial_state()
        current_t = t0
        current_state = state
        all_states = []
        target_idx = 0

        for event_t, event_idx in zip(event_times, event_indices):
            event_t_val = event_t.detach().item()

            # Find target times in this segment (before event)
            current_t_val = current_t.detach().item()
            segment_targets = []
            while target_idx < len(target_times) and target_times[target_idx].item() <= event_t_val:
                t_val = target_times[target_idx].item()
                # Only include if strictly after current time
                if t_val > current_t_val + 1e-10:
                    segment_targets.append(target_times[target_idx])
                target_idx += 1

            if segment_targets:
                tt = torch.cat([torch.tensor([current_t_val], dtype=torch.float64), torch.stack(segment_targets)])
                sol = self.odeint(self, current_state, tt, atol=1e-8, rtol=1e-8)
                all_states.append(sol[1:].reshape(-1, 12))

            # Integrate to event and apply state update
            current_t_val = current_t.detach().item()
            tt_event = torch.tensor([current_t_val, event_t_val], dtype=torch.float64)
            sol_event = self.odeint(self, current_state, tt_event, atol=1e-8, rtol=1e-8)
            current_state = self.state_update(sol_event[-1], event_idx)
            current_t = event_t

        # Remaining target times after last event
        if target_idx < len(target_times):
            current_t_val = current_t.detach().item()
            # Filter out any targets too close to current time
            segment_targets = [t for t in target_times[target_idx:] if t.item() > current_t_val + 1e-10]

            if segment_targets:
                tt = torch.cat([torch.tensor([current_t_val], dtype=torch.float64), torch.stack(segment_targets)])
                sol = self.odeint(self, current_state, tt, atol=1e-8, rtol=1e-8)
                all_states.append(sol[1:].reshape(-1, 12))

        return torch.cat(all_states, dim=0) if all_states else torch.zeros(0, 12, dtype=torch.float64)

    def get_event_times(self, t_end: float, max_events: int = 20):
        """Get event times and indices using odeint_event."""
        event_times = []
        event_indices = []

        t0, state = self.get_initial_state()
        current_t = t0
        current_state = state

        for i in range(max_events):
            current_t_val = current_t.detach().item() if hasattr(current_t, 'detach') else float(current_t)
            if current_t_val >= t_end:
                break

            # Find next event - try each event function
            next_event_t = None
            next_event_idx = None

            for ev_idx in range(15):
                try:
                    # Create single-event function for this event
                    def single_event_fn(t, s):
                        events = self.event_fn(t, s)
                        return events[ev_idx]

                    event_t, solution = odeint_event(
                        self,
                        current_state,
                        current_t,
                        event_fn=single_event_fn,
                        reverse_time=False,
                        atol=1e-8,
                        rtol=1e-8,
                        odeint_interface=self.odeint,
                    )

                    # Check if this event is sooner (use .item() to avoid gradient warnings)
                    event_t_val = event_t.detach().item() if hasattr(event_t, 'detach') else float(event_t)
                    if event_t_val < t_end:
                        next_event_t_val = next_event_t.detach().item() if (next_event_t is not None and hasattr(next_event_t, 'detach')) else (float(next_event_t) if next_event_t is not None else None)
                        if next_event_t is None or event_t_val < next_event_t_val:
                            next_event_t = event_t
                            next_event_idx = ev_idx

                except:
                    pass

            if next_event_t is None:
                break

            event_times.append(next_event_t)
            event_indices.append(next_event_idx)

            # Update state at event
            # Integrate to event first
            current_t_val = current_t.detach().item() if hasattr(current_t, 'detach') else float(current_t)
            next_event_t_val = next_event_t.detach().item() if hasattr(next_event_t, 'detach') else float(next_event_t)
            tt = torch.linspace(current_t_val, next_event_t_val, 10)
            if len(tt) > 1:
                sol = self.odeint(self, current_state, tt, atol=1e-8, rtol=1e-8)
                current_state = self.state_update(sol[-1], next_event_idx)
            else:
                current_state = self.state_update(current_state, next_event_idx)

            current_t = next_event_t

        return event_times, event_indices

    def simulate(self, t_end: float, max_events: int = 50):
        """
        Simulate the bouncing balls with events (non-differentiable).

        Args:
            t_end: End time for simulation
            max_events: Maximum number of events to handle

        Returns:
            times: Tensor of time points
            trajectory: State trajectory (N x 12)
            event_times: List of event times
            event_indices: List of which event fired at each event time
        """
        t0, state = self.get_initial_state()

        all_times = [t0.reshape(-1)]
        all_states = [state.reshape(1, -1)]
        event_times = []
        event_indices = []

        current_t = t0
        current_state = state

        for event_num in range(max_events):
            if float(current_t) >= t_end:
                break

            try:
                # Try to find next event using odeint_event
                # Note: odeint_event only handles scalar event functions
                # We need to detect which event fires by checking all of them

                # Find next event by integrating with small steps and checking
                # This is a workaround since odeint_event expects scalar event_fn
                event_detected = False
                event_t = None
                event_idx = None

                # Integrate forward with fine resolution to detect events
                t_probe_end = min(float(current_t) + 0.1, t_end)
                n_probe = 50
                tt_probe = torch.linspace(float(current_t), t_probe_end, n_probe, dtype=torch.float64)

                if len(tt_probe) > 1:
                    sol_probe = odeint(self, current_state, tt_probe, atol=1e-8, rtol=1e-8)

                    # Check event functions along trajectory
                    for i in range(1, len(tt_probe)):
                        events_i = self.event_fn(tt_probe[i], sol_probe[i])

                        # Check for zero crossing (event fires when event_fn crosses zero from + to -)
                        if i > 0:
                            events_prev = self.event_fn(tt_probe[i-1], sol_probe[i-1])

                            for ev_idx in range(15):
                                if events_prev[ev_idx] > 0 and events_i[ev_idx] <= 0:
                                    # Event detected
                                    event_detected = True
                                    event_t = tt_probe[i]
                                    event_idx = ev_idx

                                    # Integrate to event time
                                    tt_seg = torch.linspace(float(current_t), float(event_t), self.ncp, dtype=torch.float64)
                                    if len(tt_seg) > 1:
                                        sol_seg = odeint(self, current_state, tt_seg, atol=1e-8, rtol=1e-8)
                                        all_times.append(tt_seg[1:])
                                        all_states.append(sol_seg[1:])

                                    # Apply state update
                                    current_state = self.state_update(sol_probe[i], event_idx)
                                    current_t = event_t
                                    event_times.append(event_t)
                                    event_indices.append(event_idx)
                                    break

                            if event_detected:
                                break

                if not event_detected:
                    # No event in this probe window, continue to end
                    tt_final = torch.linspace(float(current_t), t_end, self.ncp, dtype=torch.float64)
                    if len(tt_final) > 1:
                        sol_final = odeint(self, current_state, tt_final, atol=1e-8, rtol=1e-8)
                        all_times.append(tt_final[1:])
                        all_states.append(sol_final[1:])
                    break

            except Exception as ex:
                # Integration failed, stop
                break

        # Final segment if needed
        if float(current_t) < t_end:
            tt_final = torch.linspace(float(current_t), t_end, self.ncp, dtype=torch.float64)
            if len(tt_final) > 1:
                sol_final = odeint(self, current_state, tt_final, atol=1e-8, rtol=1e-8)
                all_times.append(tt_final[1:])
                all_states.append(sol_final[1:])

        times = torch.cat(all_times)
        trajectory = torch.cat(all_states, dim=0)

        return times, trajectory, event_times, event_indices
