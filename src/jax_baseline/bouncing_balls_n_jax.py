"""
JAX/diffrax model for N bouncing balls in a 2D box.

This is a JAX equivalent of `src/pytorch/bouncing_balls_n.py` (which uses
PyTorch + torchdiffeq). The same forward dynamics, event ordering, and
reinit semantics are reproduced so that both baselines can be compared
apples-to-apples for AD-through-simulation parameter learning.

State layout (matches the PyTorch model and the YAML/JSON spec):
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

Ball-ball events at indices [4N, 4N + N(N-1)/2) in lex (i,j), i<j:
    condition: (xi-xj)^2 + (yi-yj)^2 - d_sq -> 0
Reinit: vx_i, vy_i, vx_j, vy_j flipped with restitution e_b.
"""

from __future__ import annotations

import functools
import itertools
from typing import List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import diffrax
import optimistix as optx

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)


class BouncingBallsParams(NamedTuple):
    """Optimizable parameters as a JAX-friendly pytree leaf."""
    g: jnp.ndarray
    e_g: jnp.ndarray
    e_b: jnp.ndarray


class BouncingBallsNModelJAX:
    """N bouncing balls in a 2D box with gravity, walls, and ball-ball
    collisions. Forward pass and event handling are differentiable via
    JAX, with diffrax handling the ODE integration and the implicit
    differentiation of event times.
    """

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
        rtol: float = 1.0e-6,
        atol: float = 1.0e-6,
        max_segments: int = 64,
        max_pts_per_seg: int = 600,
    ):
        if N < 1:
            raise ValueError("N must be >= 1")
        self.N = int(N)
        self.n_state = 4 * self.N

        # Box geometry / collision threshold (fixed, non-optimized)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.d_sq = float(d_sq)

        # Initial conditions
        if initial_state is None:
            initial_state = [0.0] * self.n_state
        if len(initial_state) != self.n_state:
            raise ValueError(
                f"initial_state has length {len(initial_state)}, expected {self.n_state}"
            )
        self.initial_state = jnp.asarray(initial_state, dtype=jnp.float64)

        # Pair table for ball-ball events
        pairs: List[Tuple[int, int]] = list(
            itertools.combinations(range(self.N), 2)
        )
        self.pairs = pairs
        self.n_wall_events = 4 * self.N
        self.n_pair_events = len(pairs)
        self.n_events = self.n_wall_events + self.n_pair_events

        if self.n_pair_events > 0:
            arr = jnp.asarray(pairs, dtype=jnp.int32)
            self._pair_i = arr[:, 0]
            self._pair_j = arr[:, 1]
        else:
            self._pair_i = jnp.empty(0, dtype=jnp.int32)
            self._pair_j = jnp.empty(0, dtype=jnp.int32)

        # Initial param values (caller may use these to build the optimization
        # state). Kept as a NamedTuple so jax.grad/jax.tree_map work cleanly.
        self.init_params = BouncingBallsParams(
            g=jnp.asarray(g, dtype=jnp.float64),
            e_g=jnp.asarray(e_g, dtype=jnp.float64),
            e_b=jnp.asarray(e_b, dtype=jnp.float64),
        )

        # Solver / controller (matches PyTorch baseline tolerances).
        # Args convention everywhere: a 2-tuple (params, ev_idx). ev_idx is
        # a placeholder int for non-event integrations; for event refinement
        # calls it picks which entry of `event_fn` is the cond_fn root. Using
        # a single fixed args structure (instead of per-event lambda
        # closures) means diffrax's internal JIT cache is hit across every
        # gradient call — one compile per shape, not one per event index.
        self.rtol = float(rtol)
        self.atol = float(atol)
        self._term = diffrax.ODETerm(
            lambda t, y, args: self.vector_field(t, y, args[0])
        )
        # Dopri5: 5(4)-order Dormand-Prince adaptive Runge-Kutta. Same
        # method that torchdiffeq.odeint defaults to in the PyTorch
        # baseline, so head-to-head comparison reflects algorithmic
        # equivalence rather than solver-method differences.
        self._solver = diffrax.Dopri5()
        # Composite event for the forward simulation: stops at the FIRST
        # event by tracking min(event_fn) restricted to currently-active
        # events. The active mask handles the post-reinit boundary case
        # (e.g. just after a wall hit, position == wall, so the event that
        # fired is at zero and Bisection can't bracket it). Mirrors the
        # PyTorch baseline's `events_at_start > EPS_POS` filter.
        #
        # Args convention everywhere: (params, active_mask). active_mask is
        # shape (n_events,) bool — True for events currently positive
        # (eligible to fire). Non-event integrations pass an all-True mask;
        # vector_field ignores args[1].
        _LARGE = jnp.asarray(1.0e30, dtype=jnp.float64)

        def _composite_cond_fn(t, y, args, **kwargs):
            params_, active_mask_ = args
            events = self.event_fn(t, y, params_)
            return jnp.min(jnp.where(active_mask_, events, _LARGE))

        self._composite_cond_fn = _composite_cond_fn
        # Real-valued cond_fn => MUST pass a root_finder, otherwise diffrax
        # treats the output as a bool (any non-zero → triggered) and the
        # returned event time is unreliable. Bisection is the right choice:
        # cond_fn = min(events) is non-smooth at the argmin, so a Newton
        # solver would be brittle. Bisection only needs sign change in a
        # bracket, which diffrax establishes per step.
        self._composite_event = diffrax.Event(
            cond_fn=self._composite_cond_fn,
            root_finder=optx.Bisection(rtol=self.rtol, atol=self.atol),
        )
        self._all_active_mask = jnp.ones(self.n_events, dtype=jnp.bool_)

        # Padding budgets to keep diffrax JIT cache stable across iters:
        #   max_segments     - fixed number of segment integrations per
        #                      simulate_at_targets call (real segments
        #                      count up to actual n_events+1; the rest are
        #                      no-op segments at t=t_end).
        #   max_pts_per_seg  - fixed shape of the per-segment SaveAt(ts)
        #                      buffer. Set >= max # targets in any one
        #                      segment. Padded entries are at the segment
        #                      endpoint; their saved state is discarded.
        # With these bounds, every diffeqsolve call has the same input
        # shapes across iters, so the JIT cache hits even when the actual
        # segment topology changes during optimization.
        self.max_segments = int(max_segments)
        self.max_pts_per_seg = int(max_pts_per_seg)

    # ------------------------------------------------------------------ #
    # ODE rhs
    # ------------------------------------------------------------------ #
    def vector_field(self, t, state, params):
        """ODE rhs: x' = vx, y' = vy, vx' = 0, vy' = -g."""
        g = params.g
        # build dstate via segmented sets (fast on CPU/GPU, no scatter loop)
        dstate = jnp.zeros_like(state)
        dstate = dstate.at[0::4].set(state[2::4])     # dx/dt = vx
        dstate = dstate.at[1::4].set(state[3::4])     # dy/dt = vy
        # vx' is already zero from zeros_like
        dstate = dstate.at[3::4].set(-g)              # dvy/dt = -g
        return dstate

    # ------------------------------------------------------------------ #
    # Event functions
    # ------------------------------------------------------------------ #
    def event_fn(self, t, state, params):
        """Return the (n_events,) event vector. Sign convention: a value
        crossing from + to - signals the event firing (mirrors the PyTorch
        baseline and the YAML's `expr < 0` triggers).
        """
        x = state[0::4]
        y = state[1::4]
        nw = self.n_wall_events
        events = jnp.zeros(self.n_events, dtype=state.dtype)
        events = events.at[0:nw:4].set(y - self.y_min)      # Floor
        events = events.at[1:nw:4].set(self.y_max - y)      # Ceiling
        events = events.at[2:nw:4].set(x - self.x_min)      # Left
        events = events.at[3:nw:4].set(self.x_max - x)      # Right
        if self.n_pair_events > 0:
            dx = x[self._pair_i] - x[self._pair_j]
            dy = y[self._pair_i] - y[self._pair_j]
            events = events.at[nw:].set(dx * dx + dy * dy - self.d_sq)
        return events

    # ------------------------------------------------------------------ #
    # State update at events (JIT-friendly via lax.cond / lax.switch)
    # ------------------------------------------------------------------ #
    def state_update(self, state, event_idx, params):
        """Apply the reinit equations associated with `event_idx`.

        Mirrors the PyTorch model's `state_update`:
          - Wall event: clamp the relevant position component, flip the
            corresponding velocity with -e_g.
          - Pair event: flip both balls' (vx, vy) with -e_b. No position
            update for pair events.

        `event_idx` may be a Python int or a 0-d traced int array; both
        paths use lax.cond / lax.switch / dynamic_index_in_dim so the
        whole call is jit-able and differentiable.
        """
        e_g = params.e_g
        e_b = params.e_b
        nw = self.n_wall_events

        ev_idx = jnp.asarray(event_idx, dtype=jnp.int32)

        def _wall_update(state):
            ball_i = ev_idx // 4
            kind = ev_idx % 4
            base = 4 * ball_i

            def floor(s):
                s = jax.lax.dynamic_update_index_in_dim(s, jnp.asarray(self.y_min, dtype=s.dtype), base + 1, 0)
                v_y = jax.lax.dynamic_index_in_dim(s, base + 3, 0, keepdims=False)
                s = jax.lax.dynamic_update_index_in_dim(s, -e_g * v_y, base + 3, 0)
                return s

            def ceiling(s):
                s = jax.lax.dynamic_update_index_in_dim(s, jnp.asarray(self.y_max, dtype=s.dtype), base + 1, 0)
                v_y = jax.lax.dynamic_index_in_dim(s, base + 3, 0, keepdims=False)
                s = jax.lax.dynamic_update_index_in_dim(s, -e_g * v_y, base + 3, 0)
                return s

            def left(s):
                s = jax.lax.dynamic_update_index_in_dim(s, jnp.asarray(self.x_min, dtype=s.dtype), base + 0, 0)
                v_x = jax.lax.dynamic_index_in_dim(s, base + 2, 0, keepdims=False)
                s = jax.lax.dynamic_update_index_in_dim(s, -e_g * v_x, base + 2, 0)
                return s

            def right(s):
                s = jax.lax.dynamic_update_index_in_dim(s, jnp.asarray(self.x_max, dtype=s.dtype), base + 0, 0)
                v_x = jax.lax.dynamic_index_in_dim(s, base + 2, 0, keepdims=False)
                s = jax.lax.dynamic_update_index_in_dim(s, -e_g * v_x, base + 2, 0)
                return s

            return jax.lax.switch(kind, [floor, ceiling, left, right], state)

        def _pair_update(state):
            pair_k = ev_idx - nw
            i = self._pair_i[pair_k]
            j = self._pair_j[pair_k]
            bi = 4 * i
            bj = 4 * j
            s = state
            for offset in (2, 3):  # vx, vy of ball i
                v_i = jax.lax.dynamic_index_in_dim(s, bi + offset, 0, keepdims=False)
                s = jax.lax.dynamic_update_index_in_dim(s, -e_b * v_i, bi + offset, 0)
            for offset in (2, 3):  # vx, vy of ball j
                v_j = jax.lax.dynamic_index_in_dim(s, bj + offset, 0, keepdims=False)
                s = jax.lax.dynamic_update_index_in_dim(s, -e_b * v_j, bj + offset, 0)
            return s

        is_wall = ev_idx < nw
        return jax.lax.cond(is_wall, _wall_update, _pair_update, state)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def get_initial_state(self):
        return jnp.asarray(0.0, dtype=jnp.float64), self.initial_state

    def _controller(self, rtol=None, atol=None):
        # `dtmax` is critical for event detection on this ODE. Free-fall
        # dynamics (y'' = -g) have essentially zero step error for any
        # decent RK method, so the PID controller would otherwise pick
        # huge steps. Diffrax's Event mechanism only checks `cond_fn` at
        # step boundaries, so a step larger than the event-transit window
        # steps right OVER a transient ball-ball collision (where
        # distance^2 dips below d_sq for ~30 ms then returns positive).
        # `dtmax = 5 ms` ensures cond_fn is sampled at least every 5 ms,
        # catching any transit through `d_sq` reliably.
        return diffrax.PIDController(
            rtol=self.rtol if rtol is None else rtol,
            atol=self.atol if atol is None else atol,
            dtmax=5.0e-3,
        )

    # ------------------------------------------------------------------ #
    # Differentiable evaluation at specific target times — fully JIT'd
    # ------------------------------------------------------------------ #
    def simulate_at_targets(
        self,
        params,
        target_times,
        max_events: int = 400,        # kept for API compat; not used here
        zeno_dt_threshold: float = 1.0e-9,
        max_steps: int = 4096,
    ):
        """Differentiable forward simulation at requested target times.

        Fully JIT'd: the entire forward pass (event-detection loop +
        per-segment integration + result aggregation) compiles to one XLA
        computation. Eliminates the ~100-250 ms of per-call diffrax
        dispatch overhead that dominated the previous Python-loop version
        on small problems.

        Algorithm (all inside the outer JIT):
          1. `lax.scan` over `max_segments` iterations, each step running
             one `diffeqsolve` with composite-event termination. Found
             events apply `state_update`; "padding" iterations (after the
             real events end) integrate trivially over [t_end, t_end].
          2. `lax.scan` over the same `max_segments`, integrating each
             segment with `SaveAt(ts=clip(target_times, t_start, t_end))`.
             Targets clamped outside their segment get garbage values
             (state at the boundary), to be masked out.
          3. Per-target mask: pick the LATEST real segment containing each
             target time. Right-continuous: a target equal to an event
             time gets the post-event state.

        Right-continuous at events matches the PyTorch baseline.
        """
        target_times_j = jnp.asarray(target_times, dtype=jnp.float64)
        n_targets = target_times_j.shape[0]
        if n_targets == 0:
            return jnp.zeros((0, self.n_state), dtype=jnp.float64)
        return self._simulate_at_targets_jit(params, target_times_j)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _simulate_at_targets_jit(self, params, target_times):
        """JIT'd implementation.

        `self` is treated as a static (hashable-by-identity) arg, so all
        Python-side constants (n_events, n_state, max_segments,
        max_pts_per_seg, the diffrax solver / term / event objects, etc.)
        are baked into the trace. `params` and `target_times` are traced.
        """
        n_targets = target_times.shape[0]
        max_seg = self.max_segments
        n_state = self.n_state

        t0 = jnp.asarray(0.0, dtype=jnp.float64)
        # Add a tiny epsilon so target_times[-1] is strictly inside [0, t_end_total].
        t_end_total = target_times[-1] + jnp.asarray(1e-6, dtype=jnp.float64)

        # ----- Step 1: event-detection scan over max_segments iters -----
        # Carry: (current_t, current_state) — state evolves across events.
        # Output per iter: (t_start, t_end, state_start, is_real) describing
        # the segment that this iter "claimed". Padding iters (after all
        # events) emit a degenerate segment with t_start == t_end.
        EPS_ACTIVE = 1e-9

        def detect_step(carry, _k):
            current_t, current_state = carry

            # Active-mask: events currently strictly positive at start, so
            # Bisection has a sign change to bracket. Mirrors the PyTorch
            # baseline's `events_at_start > EPS_POS` filter.
            events_at_start = self.event_fn(current_t, current_state, params)
            active_mask = jax.lax.stop_gradient(events_at_start) > EPS_ACTIVE

            # If current_t already reached t_end_total (we're in padding
            # territory after detecting all real events), nudge t1 to keep
            # diffrax happy with t1 > t0; the result is unused.
            t1 = jnp.maximum(t_end_total, current_t + jnp.asarray(1e-12, dtype=jnp.float64))
            dt0 = jnp.maximum((t1 - current_t) * 1e-3,
                              jnp.asarray(1e-12, dtype=jnp.float64))

            sol = diffrax.diffeqsolve(
                self._term,
                self._solver,
                t0=current_t,
                t1=t1,
                dt0=dt0,
                y0=current_state,
                args=(params, active_mask),
                event=self._composite_event,
                stepsize_controller=self._controller(),
                max_steps=4096,
            )

            event_fired = sol.event_mask
            et_jnp = sol.ts[-1]
            state_at_event = sol.ys[-1]

            # "Real" event: fired AND we hadn't already finished AND not
            # at the final-time boundary (which would be a spurious event
            # from a cond_fn quirk at t1).
            already_done = current_t >= (t_end_total - jnp.asarray(1e-12, dtype=jnp.float64))
            at_boundary = et_jnp >= (t_end_total - jnp.asarray(1e-12, dtype=jnp.float64))
            is_real_event = event_fired & ~already_done & ~at_boundary

            # Determine event_idx by argmin of event values at sol.ys[-1].
            # Mask inactive events to a large value so argmin picks an
            # active one.
            ev_vals = self.event_fn(et_jnp, state_at_event, params)
            ev_vals_sg = jax.lax.stop_gradient(
                jnp.where(active_mask, ev_vals, jnp.asarray(1e30, dtype=ev_vals.dtype))
            )
            ev_idx = jnp.argmin(ev_vals_sg).astype(jnp.int32)

            # Apply state_update if real event; otherwise carry state through.
            state_after = jax.lax.cond(
                is_real_event,
                lambda: self.state_update(state_at_event, ev_idx, params),
                lambda: state_at_event,
            )

            # Advance current_t: event_t if real, else t_end_total so the
            # next iter's segment is degenerate (t_start == t_end).
            new_t = jnp.where(is_real_event, et_jnp, t_end_total)

            seg_out = {
                't_start': current_t,
                't_end': jnp.where(is_real_event, et_jnp, t_end_total),
                'state_start': current_state,
                # is_real: this iter actually integrated (t_start < t_end_total).
                'is_real': ~already_done,
            }
            return (new_t, state_after), seg_out

        (_, _), segments = jax.lax.scan(
            detect_step,
            (t0, self.initial_state),
            xs=jnp.arange(max_seg),
        )
        # segments: dict of (max_seg, ...) arrays.

        # ----- Step 2: per-segment integration scan -----
        # For each segment k: integrate from t_start[k] to t_end[k] with
        # SaveAt at target_times clamped to [t_start[k], t_end[k]]. Targets
        # outside the segment yield garbage values that we mask out below.
        def integrate_step(_carry, seg_data):
            t_start, t_end, state_start, is_real = seg_data
            # Pad t_end so dt0 > 0 even for degenerate segments.
            t_end_safe = jnp.maximum(
                t_end, t_start + jnp.asarray(1e-12, dtype=jnp.float64)
            )
            save_ts = jnp.clip(target_times, t_start, t_end_safe)
            dt0 = jnp.maximum(
                (t_end_safe - t_start) * 1e-3,
                jnp.asarray(1e-12, dtype=jnp.float64),
            )
            sol = diffrax.diffeqsolve(
                self._term,
                self._solver,
                t0=t_start,
                t1=t_end_safe,
                dt0=dt0,
                y0=state_start,
                args=(params, self._all_active_mask),
                saveat=diffrax.SaveAt(ts=save_ts),
                stepsize_controller=self._controller(),
                max_steps=4096,
            )
            return None, sol.ys  # (n_targets, n_state)

        _, all_ys = jax.lax.scan(
            integrate_step,
            None,
            (segments['t_start'], segments['t_end'],
             segments['state_start'], segments['is_real']),
        )
        # all_ys: (max_seg, n_targets, n_state)

        # ----- Step 3: combine via right-continuous segment selection -----
        # For each target i, pick the LATEST real segment k whose closed
        # range [t_start[k], t_end[k]] contains target_times[i]. Targets at
        # event boundaries (= t_start of segment k+1 = t_end of segment k)
        # get the post-event value (segment k+1) — right-continuous.
        TOL = jnp.asarray(1e-12, dtype=jnp.float64)
        t_starts = segments['t_start']    # (max_seg,)
        t_ends = segments['t_end']        # (max_seg,)
        is_real = segments['is_real']     # (max_seg,)

        in_range = (
            (target_times[None, :] >= t_starts[:, None] - TOL)
            & (target_times[None, :] <= t_ends[:, None] + TOL)
            & is_real[:, None]
        )  # (max_seg, n_targets)

        # Score = (k + 1) if in_range else 0; argmax picks largest k with True.
        seg_indices = jnp.arange(max_seg)
        weighted = in_range.astype(jnp.int32) * (seg_indices[:, None] + 1)
        chosen_seg = jnp.argmax(weighted, axis=0)  # (n_targets,)

        # Gather: result[i] = all_ys[chosen_seg[i], i, :]
        return all_ys[chosen_seg, jnp.arange(n_targets)]

