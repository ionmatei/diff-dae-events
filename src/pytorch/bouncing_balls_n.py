"""
PyTorch model for N bouncing balls in a 2D box.

Generalization of `bouncing_balls.BouncingBallsModel` (which is hard-coded
to 3 balls / 12 states / 15 events) to an arbitrary N. Event ordering and
reinit semantics mirror the YAML/JSON spec produced by
`src.run.modelica_balls_to_yaml` so that PyTorch and the IDA-based
DAESolver simulate the *same* physics:

    Per ball i (0-based), states are laid out as
        state[4i + 0] = x_i
        state[4i + 1] = y_i
        state[4i + 2] = vx_i
        state[4i + 3] = vy_i

    Wall events (4 per ball, total 4N), in the order:
        4i + 0 : Floor    (y_i - y_min  -> 0)
        4i + 1 : Ceiling  (y_max - y_i  -> 0)
        4i + 2 : Left     (x_i - x_min  -> 0)
        4i + 3 : Right    (x_max - x_i  -> 0)
    Reinit on each wall event: position clamped to the wall, velocity
    component flipped with restitution e_g.

    Ball-ball events (N*(N-1)/2 total) at indices 4N .. 4N + N(N-1)/2 - 1
    in lexicographic (i, j) order with i < j:
        condition: (xi-xj)**2 + (yi-yj)**2 - d_sq -> 0
    Reinit: vx_i, vy_i, vx_j, vy_j flipped with restitution e_b.
"""

from __future__ import annotations

import itertools
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint, odeint_event

torch.set_default_dtype(torch.float64)


class BouncingBallsNModel(nn.Module):
    """N bouncing balls in a 2D box with gravity, walls, and ball-ball collisions."""

    def __init__(
        self,
        N: int,
        g: float = 9.81,
        e_g: float = 0.8,
        e_b: float = 0.9,
        d_sq: float = 0.25,
        x_min: float = 0.0,
        x_max: float = 10.0,
        y_min: float = 0.0,
        y_max: float = 10.0,
        initial_state: Optional[List[float]] = None,
        adjoint: bool = False,
    ):
        super().__init__()

        if N < 1:
            raise ValueError("N must be >= 1")
        self.N = int(N)
        self.n_state = 4 * self.N

        # Optimizable scalar parameters
        self.g = nn.Parameter(torch.tensor([g], dtype=torch.float64))
        self.e_g = nn.Parameter(torch.tensor([e_g], dtype=torch.float64))
        self.e_b = nn.Parameter(torch.tensor([e_b], dtype=torch.float64))

        # Box geometry / collision threshold (fixed)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.d_sq = float(d_sq)

        # Initial state
        if initial_state is None:
            initial_state = [0.0] * self.n_state
        if len(initial_state) != self.n_state:
            raise ValueError(
                f"initial_state has length {len(initial_state)}, expected {self.n_state}"
            )
        self.initial_state = torch.tensor(initial_state, dtype=torch.float64)

        # Pair table for ball-ball events: pair_idx -> (i, j) with i<j (0-based)
        self.pairs: List[Tuple[int, int]] = list(
            itertools.combinations(range(self.N), 2)
        )
        self.n_wall_events = 4 * self.N
        self.n_pair_events = len(self.pairs)
        self.n_events = self.n_wall_events + self.n_pair_events

        # Pair index buffers for vectorized event_fn. Stored once at init
        # so the per-step pair-distance computation is one tensor op
        # rather than N*(N-1)/2 Python iterations.
        if self.n_pair_events > 0:
            pair_arr = torch.tensor(self.pairs, dtype=torch.long)
            self.register_buffer("_pair_i", pair_arr[:, 0], persistent=False)
            self.register_buffer("_pair_j", pair_arr[:, 1], persistent=False)
        else:
            self.register_buffer("_pair_i", torch.empty(0, dtype=torch.long),
                                 persistent=False)
            self.register_buffer("_pair_j", torch.empty(0, dtype=torch.long),
                                 persistent=False)

        self.odeint = odeint_adjoint if adjoint else odeint

    # ------------------------------------------------------------------ #
    # ODE rhs
    # ------------------------------------------------------------------ #
    def forward(self, t, state):
        dstate = torch.zeros_like(state)
        N = self.N
        # x' = vx, y' = vy, vx' = 0, vy' = -g  (broadcast)
        # state shape (4N,); use slicing for speed.
        dstate[0::4] = state[2::4]      # dx/dt = vx
        dstate[1::4] = state[3::4]      # dy/dt = vy
        # vx' is already zero, vy' = -g
        dstate[3::4] = -self.g
        return dstate

    # ------------------------------------------------------------------ #
    # Event functions
    # ------------------------------------------------------------------ #
    def event_fn(self, t, state):
        """Return the (n_events,) event vector. Sign convention: a value
        crossing from + to - signals the event firing (matches the YAML's
        `expr < 0` triggers).
        """
        events = torch.empty(self.n_events, dtype=state.dtype)

        # Wall events: 4 per ball, interleaved (only the first 4N entries)
        # Per ball i: Floor, Ceiling, Left, Right
        x = state[0::4]
        y = state[1::4]
        nw = self.n_wall_events
        events[0:nw:4] = y - self.y_min      # Floor
        events[1:nw:4] = self.y_max - y      # Ceiling
        events[2:nw:4] = x - self.x_min      # Left
        events[3:nw:4] = self.x_max - x      # Right

        # Ball-ball pair events: single tensor op via precomputed index buffers
        if self.n_pair_events > 0:
            dx = x.index_select(0, self._pair_i) - x.index_select(0, self._pair_j)
            dy = y.index_select(0, self._pair_i) - y.index_select(0, self._pair_j)
            events[nw:] = dx * dx + dy * dy - self.d_sq
        return events

    # ------------------------------------------------------------------ #
    # State update at events
    # ------------------------------------------------------------------ #
    def state_update(self, state, event_idx: int):
        """Apply the reinit equations associated with `event_idx`. Mirrors
        the YAML spec: walls clamp position + flip the appropriate velocity
        component; ball-ball pairs flip both balls' velocity components.
        """
        new_state = state.clone()

        if event_idx < self.n_wall_events:
            ball_i = event_idx // 4
            kind = event_idx % 4
            base = 4 * ball_i
            if kind == 0:  # Floor
                new_state[base + 1] = self.y_min
                new_state[base + 3] = -self.e_g * state[base + 3]
            elif kind == 1:  # Ceiling
                new_state[base + 1] = self.y_max
                new_state[base + 3] = -self.e_g * state[base + 3]
            elif kind == 2:  # Left wall
                new_state[base + 0] = self.x_min
                new_state[base + 2] = -self.e_g * state[base + 2]
            else:           # Right wall
                new_state[base + 0] = self.x_max
                new_state[base + 2] = -self.e_g * state[base + 2]
        else:
            pair_k = event_idx - self.n_wall_events
            i, j = self.pairs[pair_k]
            bi, bj = 4 * i, 4 * j
            new_state[bi + 2] = -self.e_b * state[bi + 2]
            new_state[bi + 3] = -self.e_b * state[bi + 3]
            new_state[bj + 2] = -self.e_b * state[bj + 2]
            new_state[bj + 3] = -self.e_b * state[bj + 3]
        return new_state

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def get_initial_state(self):
        t0 = torch.tensor([0.0], dtype=torch.float64)
        return t0, self.initial_state.clone()

    # ------------------------------------------------------------------ #
    # Event finding (probe + odeint_event with bisection fallback)
    # ------------------------------------------------------------------ #
    def _bisect_event(self, current_state, t_start, t_hi, ev_idx, n_bisect=50):
        t_start_val = float(t_start)
        t_lo_val = t_start_val
        t_hi_val = float(t_hi)
        for _ in range(n_bisect):
            t_mid = 0.5 * (t_lo_val + t_hi_val)
            tt = torch.tensor([t_start_val, t_mid], dtype=torch.float64)
            sol = self.odeint(self, current_state, tt, atol=1e-6, rtol=1e-6)
            ev_val = self.event_fn(tt[1], sol[-1])[ev_idx]
            if ev_val.item() > 0:
                t_lo_val = t_mid
            else:
                t_hi_val = t_mid
            if (t_hi_val - t_lo_val) < 1e-10:
                break
        event_t = torch.tensor([t_hi_val], dtype=torch.float64)
        tt_final = torch.tensor([t_start_val, t_hi_val], dtype=torch.float64)
        sol_final = self.odeint(self, current_state, tt_final, atol=1e-6, rtol=1e-6)
        return event_t, sol_final[-1]

    def _find_next_event(self, current_state, current_t, t_end,
                         probe_lookahead: float = 0.05,
                         n_probe: int = 50):
        """Locate the next root within `[current_t, t_end]`.

        The probe runs over a bounded lookahead window
        `[current_t, current_t + probe_lookahead]`, sampled at `n_probe`
        evenly-spaced points (~1 ms per step with the defaults). If no
        candidate is found within the window, we advance to the end of it
        and probe again. This avoids the historical bug of using a probe
        that spans the full remaining horizon: with a 0.9 s lookahead and
        only 30 samples, repeat ball-ball collisions that resolved within
        ~30 ms slipped between probe points and were silently skipped.
        """
        current_t_val = (
            current_t.detach().item() if hasattr(current_t, "detach") else float(current_t)
        )
        if current_t_val >= t_end:
            return None, None, None

        window_start_state = current_state
        window_start_t = current_t_val

        while window_start_t < t_end:
            window_end_t = min(window_start_t + probe_lookahead, t_end)
            tt_probe = torch.linspace(
                window_start_t, window_end_t, n_probe, dtype=torch.float64
            )
            with torch.no_grad():
                sol_probe = odeint(
                    self, window_start_state.detach(), tt_probe,
                    atol=1e-6, rtol=1e-6,
                )
            result = self._scan_window_for_event(
                sol_probe, tt_probe, window_start_state, current_t, t_end
            )
            if result is not None:
                return result
            # No candidate in this window -- advance to its end and continue.
            window_start_t = float(tt_probe[-1])
            window_start_state = sol_probe[-1]

        return None, None, None

    def _scan_window_for_event(self, sol_probe, tt_probe,
                               window_start_state, current_t, t_end):
        """Scan a single probe window for events. Returns (event_t,
        event_idx, state_at_event) or None if no candidate is found.

        odeint_event refines from `window_start_state` at the window's
        start time, not from the simulator's outer `current_t`. The dense
        re-integration in `simulate_fixed_grid` then runs from the true
        outer state, so event-time accuracy is preserved end-to-end.
        """
        n_probe = int(tt_probe.shape[0])
        EPS_POS = 1e-9
        candidates = {}
        events_at_start = self.event_fn(tt_probe[0], sol_probe[0])
        events_prev = events_at_start
        for i in range(1, n_probe):
            events_i = self.event_fn(tt_probe[i], sol_probe[i])
            for ev_idx in range(self.n_events):
                if (
                    ev_idx not in candidates
                    and events_at_start[ev_idx] > EPS_POS
                    and events_prev[ev_idx] > 0
                    and events_i[ev_idx] <= 0
                ):
                    candidates[ev_idx] = (tt_probe[i - 1].item(), tt_probe[i].item())
            events_prev = events_i

        if not candidates:
            return None

        window_start_t = float(tt_probe[0])
        window_start_t_tensor = tt_probe[0:1]

        # Refine candidates in ascending probe-bracket order. As soon as
        # a refined event time is <= the next candidate's bracket start,
        # no later candidate can produce an earlier event, so we stop.
        # Worst case (many overlapping brackets) we still refine all of
        # them; common case we refine 1, sometimes 2.
        sorted_candidates = sorted(candidates.items(),
                                   key=lambda kv: (kv[1][0], kv[1][1]))

        best_event_t = None
        best_event_idx = None
        best_state_at_event = None
        best_t_val = None

        for ev_idx, (t_lo, t_hi) in sorted_candidates:
            if best_t_val is not None and t_lo >= best_t_val:
                break

            event_t_val = None
            state_at_event = None
            try:
                _ev_idx = ev_idx

                def single_event_fn(t, s, _idx=_ev_idx):
                    return self.event_fn(t, s)[_idx]

                event_t, solution = odeint_event(
                    self,
                    window_start_state,
                    window_start_t_tensor,
                    event_fn=single_event_fn,
                    reverse_time=False,
                    atol=1e-6,
                    rtol=1e-6,
                    odeint_interface=self.odeint,
                )
                event_t_val = event_t.detach().item()
                state_at_event = solution[-1]
            except Exception:
                try:
                    event_t, state_at_event = self._bisect_event(
                        window_start_state, window_start_t, t_hi, ev_idx
                    )
                    event_t_val = event_t.detach().item()
                except Exception:
                    continue

            if event_t_val is not None and event_t_val < t_end:
                if best_t_val is None or event_t_val < best_t_val:
                    best_event_t = (
                        event_t
                        if isinstance(event_t, torch.Tensor)
                        else torch.tensor([event_t_val], dtype=torch.float64)
                    )
                    best_event_idx = ev_idx
                    best_state_at_event = state_at_event
                    best_t_val = event_t_val

        if best_event_t is None:
            return None
        return best_event_t, best_event_idx, best_state_at_event

    # ------------------------------------------------------------------ #
    # Dense simulation
    # ------------------------------------------------------------------ #
    def simulate_fixed_grid(self, t_end: float, n_points: int = 500,
                            max_events: int = 200,
                            zeno_dt_threshold: float = 1.0e-9,
                            verbose: bool = False):
        """Two-pass forward simulation: detect events, then integrate
        densely on each segment.

        Returns:
            times       : (T,) tensor
            trajectory  : (T, 4N) tensor
            events_log  : list of (event_t, event_idx) for diagnostics
        """
        t0, state = self.get_initial_state()
        current_t = t0
        current_state = state
        event_list: List[Tuple[torch.Tensor, int]] = []
        last_event_t = None

        # --- Pass 1: detect events ------------------------------------
        for _ in range(max_events):
            event_t, event_idx, state_at_event = self._find_next_event(
                current_state, current_t, t_end
            )
            if event_t is None:
                break
            et_val = event_t.detach().item()
            if last_event_t is not None and (et_val - last_event_t) < zeno_dt_threshold:
                if verbose:
                    print(f"  [pytorch] Zeno guard at t={et_val:.6f} "
                          f"(dt={et_val - last_event_t:.2e}); stopping simulation.")
                break
            event_list.append((event_t, event_idx))
            current_state = self.state_update(state_at_event, event_idx)
            current_t = event_t
            last_event_t = et_val

        # --- Allocate per-segment grids --------------------------------
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
            duration = max(t_stop - t_start, 0.0)
            n_seg = max(2, int(n_points * duration / total_duration)) if total_duration > 0 else 2
            points_per_segment.append(n_seg)

        # --- Pass 2: dense integration ---------------------------------
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

            sol = current_state.unsqueeze(0)
            if len(tt) > 1:
                sol = self.odeint(self, current_state, tt, atol=1e-6, rtol=1e-6)
                all_times.append(tt[1:])
                all_states.append(sol[1:].reshape(-1, self.n_state))

            current_state = self.state_update(sol[-1] if len(tt) > 1 else current_state,
                                              event_idx)
            current_t = event_t
            seg_idx += 1

        if seg_idx < len(points_per_segment):
            current_t_val = current_t.detach().item()
            n_seg = points_per_segment[seg_idx]
            tt = torch.linspace(current_t_val, t_end, n_seg)
            if len(tt) > 1:
                sol = self.odeint(self, current_state, tt, atol=1e-6, rtol=1e-6)
                all_times.append(tt[1:])
                all_states.append(sol[1:].reshape(-1, self.n_state))

        times = torch.cat(all_times)
        trajectory = torch.cat(all_states, dim=0)
        return times, trajectory, event_list

    # ------------------------------------------------------------------ #
    # Differentiable evaluation at specific target times
    # ------------------------------------------------------------------ #
    def simulate_at_targets(self, target_times: torch.Tensor,
                            max_events: int = 400,
                            zeno_dt_threshold: float = 1.0e-9):
        """
        Simulate and evaluate at specific target times (differentiable).

        Generalization of `bouncing_balls.BouncingBallsModel.simulate_at_targets`
        to arbitrary N. Detects events with `_find_next_event` (differentiable
        via `odeint_event`), then re-integrates each segment with plain
        `odeint`, including the segment-bounding `event_t` tensors in the
        time grid so gradients flow through event times.

        Right-continuous at events: a target sample at an event time gets
        the post-event state.

        Args:
            target_times: (n_targets,) tensor of sorted evaluation times.
            max_events:   safety bound on the number of events detected.
            zeno_dt_threshold: minimum gap between consecutive events;
                               below this we treat the simulation as Zeno
                               and stop detecting further events.

        Returns:
            (n_targets, 4*N) tensor of states at the requested times.
        """
        target_np = target_times.detach().numpy()
        n_targets = len(target_np)
        if n_targets == 0:
            return torch.zeros(0, self.n_state, dtype=torch.float64)
        t_end = float(target_np[-1]) + 1e-6
        eps = 1e-9

        # --- Step 1: Detect events ---
        t0, state0 = self.get_initial_state()
        current_t = t0
        current_state = state0
        events: List[Tuple[torch.Tensor, int]] = []
        last_event_t = None

        for _ in range(max_events):
            event_t, event_idx, state_at_event = self._find_next_event(
                current_state, current_t, t_end
            )
            if event_t is None:
                break
            et_val = event_t.detach().item()
            if last_event_t is not None and (et_val - last_event_t) < zeno_dt_threshold:
                break
            events.append((event_t, event_idx))
            current_state = self.state_update(state_at_event, event_idx)
            current_t = event_t
            last_event_t = et_val

        # --- Step 2: Partition targets into per-segment buckets ---
        # A target at an event boundary goes to the NEXT segment (post-event).
        event_vals = [et.detach().item() for et, _ in events]

        seg_targets: List[List[int]] = [[] for _ in range(len(events) + 1)]
        ptr = 0
        for seg_idx, et_val in enumerate(event_vals):
            while ptr < n_targets and target_np[ptr] < et_val - eps:
                seg_targets[seg_idx].append(ptr)
                ptr += 1
        while ptr < n_targets:
            seg_targets[len(events)].append(ptr)
            ptr += 1

        # --- Step 3: Evaluate per segment ---
        # Each segment's integration includes the bounding event_t in the
        # time grid so gradients can flow through event times (matches the
        # torchdiffeq bouncing-ball reference pattern).
        current_t = t0
        current_state = state0
        indexed_results: List[Tuple[int, torch.Tensor]] = []

        for seg_idx in range(len(events) + 1):
            ct_val = current_t.detach().item()
            targets = seg_targets[seg_idx]

            at_start, after_start = [], []
            for tidx in targets:
                if target_np[tidx] <= ct_val + eps:
                    at_start.append(tidx)
                else:
                    after_start.append(tidx)

            for tidx in at_start:
                indexed_results.append((tidx, current_state))

            if seg_idx < len(events):
                event_t, event_idx = events[seg_idx]

                tt_list = [current_t.reshape(-1)]
                for tidx in after_start:
                    tt_list.append(torch.tensor([target_np[tidx]], dtype=torch.float64))
                tt_list.append(event_t.reshape(-1))
                tt = torch.cat(tt_list)

                sol = odeint(self, current_state, tt, atol=1e-6, rtol=1e-6)

                for i, tidx in enumerate(after_start):
                    indexed_results.append((tidx, sol[1 + i]))

                current_state = self.state_update(sol[-1], event_idx)
                current_t = event_t
            else:
                if after_start:
                    tt_list = [current_t.reshape(-1)]
                    for tidx in after_start:
                        tt_list.append(torch.tensor([target_np[tidx]], dtype=torch.float64))
                    tt = torch.cat(tt_list)
                    sol = odeint(self, current_state, tt, atol=1e-6, rtol=1e-6)
                    for i, tidx in enumerate(after_start):
                        indexed_results.append((tidx, sol[1 + i]))

        indexed_results.sort(key=lambda x: x[0])
        if not indexed_results:
            return torch.zeros(0, self.n_state, dtype=torch.float64)
        return torch.stack([s for _, s in indexed_results]).reshape(-1, self.n_state)
