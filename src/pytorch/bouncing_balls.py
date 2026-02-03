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

    def _bisect_event(self, current_state, t_start, t_hi, ev_idx, n_bisect=50):
        """
        Bisection fallback when odeint_event fails.
        Finds the time when event ev_idx crosses zero in [t_start, t_hi].

        Always integrates from t_start with current_state to evaluate the
        event function at the midpoint.

        Returns:
            (event_t, state_at_event) as tensors with grad tracking
        """
        t_start_val = float(t_start)
        t_lo_val = t_start_val
        t_hi_val = float(t_hi)

        for _ in range(n_bisect):
            t_mid = (t_lo_val + t_hi_val) / 2.0
            # Always integrate from the original start to t_mid
            tt = torch.tensor([t_start_val, t_mid], dtype=torch.float64)
            sol = self.odeint(self, current_state, tt, atol=1e-8, rtol=1e-8)
            ev_val = self.event_fn(tt[1], sol[-1])[ev_idx]

            if ev_val.item() > 0:
                t_lo_val = t_mid
            else:
                t_hi_val = t_mid

            if (t_hi_val - t_lo_val) < 1e-10:
                break

        # Final integration to the event time (with grad tracking)
        event_t = torch.tensor([t_hi_val], dtype=torch.float64)
        tt_final = torch.tensor([t_start_val, t_hi_val], dtype=torch.float64)
        sol_final = self.odeint(self, current_state, tt_final, atol=1e-8, rtol=1e-8)
        return event_t, sol_final[-1]

    def _find_next_event(self, current_state, current_t, t_end):
        """
        Find the next event using a probe integration to filter candidates,
        then odeint_event only on candidates that actually cross zero.
        Falls back to bisection when odeint_event fails.

        Returns:
            (event_t, event_idx, state_at_event) or (None, None, None)
        """
        current_t_val = current_t.detach().item() if hasattr(current_t, 'detach') else float(current_t)
        if current_t_val >= t_end:
            return None, None, None

        # Step 1: Probe integration to find which events cross zero
        n_probe = 30
        tt_probe = torch.linspace(current_t_val, t_end, n_probe, dtype=torch.float64)
        with torch.no_grad():
            sol_probe = odeint(self, current_state.detach(), tt_probe, atol=1e-6, rtol=1e-6)

        # Evaluate all event functions along probe trajectory
        # Record the first bracket [t_lo, t_hi] for each candidate
        candidates = {}  # ev_idx -> (t_lo, t_hi)
        events_prev = self.event_fn(tt_probe[0], sol_probe[0])
        for i in range(1, n_probe):
            events_i = self.event_fn(tt_probe[i], sol_probe[i])
            for ev_idx in range(15):
                if ev_idx not in candidates and events_prev[ev_idx] > 0 and events_i[ev_idx] <= 0:
                    candidates[ev_idx] = (tt_probe[i - 1].item(), tt_probe[i].item())
            events_prev = events_i

        if not candidates:
            return None, None, None

        # Step 2: Call odeint_event for each candidate; bisect on failure
        best_event_t = None
        best_event_idx = None
        best_state_at_event = None

        for ev_idx, (t_lo, t_hi) in candidates.items():
            event_t_val = None
            state_at_event = None

            # Try odeint_event first
            try:
                _ev_idx = ev_idx

                def single_event_fn(t, s, _idx=_ev_idx):
                    events = self.event_fn(t, s)
                    return events[_idx]

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
                event_t_val = event_t.detach().item()
                state_at_event = solution[-1]
            except:
                # Fallback: bisection using probe bracket
                try:
                    event_t, state_at_event = self._bisect_event(
                        current_state, current_t_val, t_hi, ev_idx
                    )
                    event_t_val = event_t.detach().item()
                except:
                    continue

            if event_t_val is not None and event_t_val < t_end:
                best_t_val = best_event_t.detach().item() if best_event_t is not None else None
                if best_event_t is None or event_t_val < best_t_val:
                    best_event_t = event_t if isinstance(event_t, torch.Tensor) else torch.tensor([event_t_val], dtype=torch.float64)
                    best_event_idx = ev_idx
                    best_state_at_event = state_at_event

        if best_event_t is None:
            return None, None, None

        return best_event_t, best_event_idx, best_state_at_event

    def simulate_fixed_grid(self, t_end: float, n_points: int = 500):
        """
        Single-pass simulation: detect events and collect dense output together.

        Args:
            t_end: End time
            n_points: Total number of simulation points to distribute across all segments

        Returns:
            times: tensor of time points
            trajectory: (N, 12) tensor
        """
        max_events = 20

        # --- Pass 1 (cheap): detect all events ---
        t0, state = self.get_initial_state()
        current_t = t0
        current_state = state
        event_list = []  # (event_t, event_idx)

        for _ in range(max_events):
            event_t, event_idx, state_at_event = self._find_next_event(
                current_state, current_t, t_end
            )
            if event_t is None:
                break
            event_list.append((event_t, event_idx))
            current_state = self.state_update(state_at_event, event_idx)
            current_t = event_t

        # --- Compute points per segment ---
        t0_val = t0.detach().item()
        total_duration = t_end - t0_val
        segment_bounds = []
        prev_t = t0_val
        for event_t, _ in event_list:
            et_val = event_t.detach().item()
            segment_bounds.append((prev_t, et_val))
            prev_t = et_val
        if prev_t < t_end:
            segment_bounds.append((prev_t, t_end))

        points_per_segment = []
        for t_start, t_stop in segment_bounds:
            duration = t_stop - t_start
            n_seg = max(2, int(n_points * duration / total_duration))
            points_per_segment.append(n_seg)

        # --- Pass 2: dense integration per segment ---
        current_t = t0
        current_state = state
        all_times = [t0.reshape(-1)]
        all_states = [state.reshape(1, -1)]
        seg_idx = 0

        for event_t, event_idx in event_list:
            current_t_val = current_t.detach().item()
            event_t_val = event_t.detach().item()

            n_seg = points_per_segment[seg_idx]
            tt = torch.linspace(current_t_val, event_t_val, n_seg)[1:-1]
            tt = torch.cat([current_t.reshape(-1), tt, event_t.reshape(-1)])

            if len(tt) > 1:
                sol = self.odeint(self, current_state, tt, atol=1e-8, rtol=1e-8)
                all_times.append(tt[1:])
                all_states.append(sol[1:].reshape(-1, 12))

            current_state = self.state_update(sol[-1] if len(tt) > 1 else current_state, event_idx)
            current_t = event_t
            seg_idx += 1

        # Final segment
        if seg_idx < len(points_per_segment):
            current_t_val = current_t.detach().item()
            n_seg = points_per_segment[seg_idx]
            tt = torch.linspace(current_t_val, t_end, n_seg)
            if len(tt) > 1:
                sol = self.odeint(self, current_state, tt, atol=1e-8, rtol=1e-8)
                all_times.append(tt[1:])
                all_states.append(sol[1:].reshape(-1, 12))

        times = torch.cat(all_times)
        trajectory = torch.cat(all_states, dim=0)

        return times, trajectory

    def simulate_at_targets(self, target_times: torch.Tensor):
        """
        Simulate and evaluate at specific target times (differentiable).

        Follows the torchdiffeq bouncing ball reference pattern:
        - Step 1: Detect events using odeint_event (differentiable event times)
        - Step 2: Re-integrate at target times using plain odeint, with event_t
          tensors in the time grid to maintain gradient flow through event times.

        Right-continuous at events: targets at event times get post-event state.

        Args:
            target_times: 1D tensor of sorted evaluation times

        Returns:
            (n_targets, 12) tensor of states at target times
        """
        target_np = target_times.detach().numpy()
        t_end = float(target_np[-1]) + 1e-6
        n_targets = len(target_np)
        eps = 1e-9

        # --- Step 1: Detect events (like reference get_collision_times) ---
        t0, state0 = self.get_initial_state()
        current_t = t0
        current_state = state0
        events = []  # (event_t tensor, event_idx int)

        for _ in range(20):
            event_t, event_idx, state_at_event = self._find_next_event(
                current_state, current_t, t_end
            )
            if event_t is None:
                break
            events.append((event_t, event_idx))
            current_state = self.state_update(state_at_event, event_idx)
            current_t = event_t

        # --- Step 2: Partition targets into segments ---
        # Target at event boundary goes to the NEXT segment (post-event)
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

        # --- Step 3: Evaluate per segment (like reference simulate) ---
        # Uses plain odeint with event_t in time grid for gradient flow
        current_t = t0
        current_state = state0
        indexed_results = []  # (target_idx, state_tensor)

        for seg_idx in range(len(events) + 1):
            ct_val = current_t.detach().item()
            targets = seg_targets[seg_idx]

            # Separate targets at segment start vs strictly after
            at_start = []
            after_start = []
            for tidx in targets:
                if target_np[tidx] <= ct_val + eps:
                    at_start.append(tidx)
                else:
                    after_start.append(tidx)

            # Targets at segment start get current_state (post-event for seg > 0)
            for tidx in at_start:
                indexed_results.append((tidx, current_state))

            if seg_idx < len(events):
                event_t, event_idx = events[seg_idx]

                # Build time grid: [current_t, target_times..., event_t]
                tt_list = [current_t.reshape(-1)]
                for tidx in after_start:
                    tt_list.append(torch.tensor([target_np[tidx]], dtype=torch.float64))
                tt_list.append(event_t.reshape(-1))
                tt = torch.cat(tt_list)

                sol = odeint(self, current_state, tt, atol=1e-8, rtol=1e-8)

                # Record states at interior target times
                for i, tidx in enumerate(after_start):
                    indexed_results.append((tidx, sol[1 + i]))

                # State update using state at event_t (like reference)
                current_state = self.state_update(sol[-1], event_idx)
                current_t = event_t
            else:
                # Last segment (no event at end)
                if after_start:
                    tt_list = [current_t.reshape(-1)]
                    for tidx in after_start:
                        tt_list.append(torch.tensor([target_np[tidx]], dtype=torch.float64))
                    tt = torch.cat(tt_list)
                    sol = odeint(self, current_state, tt, atol=1e-8, rtol=1e-8)
                    for i, tidx in enumerate(after_start):
                        indexed_results.append((tidx, sol[1 + i]))

        # Sort by target index and stack
        indexed_results.sort(key=lambda x: x[0])

        if not indexed_results:
            return torch.zeros(0, 12, dtype=torch.float64)

        return torch.stack([s for _, s in indexed_results]).reshape(-1, 12)

    def get_event_times(self, t_end: float, max_events: int = 20):
        """Get event times and indices using optimized event detection."""
        event_times = []
        event_indices = []

        t0, state = self.get_initial_state()
        current_t = t0
        current_state = state

        for i in range(max_events):
            event_t, event_idx, state_at_event = self._find_next_event(
                current_state, current_t, t_end
            )
            if event_t is None:
                break

            event_times.append(event_t)
            event_indices.append(event_idx)

            current_state = self.state_update(state_at_event, event_idx)
            current_t = event_t

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
