"""
JAX/diffrax DAE optimizer with state-dependent events (`when` clauses).

Combines:
  * the segmented integration scheme used by
    `src/jax_baseline/bouncing_balls_n_jax.py` (composite event +
    `lax.scan` over `max_segments` + per-segment `SaveAt` + right-
    continuous segment selection), and
  * the implicit-function-theorem algebraic solve from
    `src/dae/dae_optimizer.py` (`custom_jvp` + chord Newton).

Spec format additions over `dae_optimizer.py`:
    "when": [
        {"condition": "C3_v > 0.5", "reinit": "C3_v = 0"},
        ...
    ]

Each clause becomes one row of the event vector; `condition` is rewritten
to a scalar that is positive before the trigger, zero at the trigger, and
negative after. `reinit` is a single `name = expr` assignment to a
differential state. Algebraic variables are re-solved at every segment
boundary via the same custom_jvp solver used during integration.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import diffrax
import optimistix as optx
from jax import jit, jacfwd, value_and_grad, custom_jvp, vmap
from jax.scipy.linalg import lu_factor, lu_solve

from src.dae.dae_optimizer import (
    _normalize_expr, _build_fused_function, _MATH_FUNCS,
)


# ---------------------------------------------------------------------- #
# Spec parsing helpers
# ---------------------------------------------------------------------- #

def _parse_condition_to_event_expr(cond_str: str) -> str:
    """Convert 'lhs OP rhs' into a scalar expression that is positive
    before the trigger, zero at the trigger, negative after.

    'C3_v > 0.5'  -> '(0.5) - (C3_v)'
    'x < 1.0'     -> '(x) - (1.0)'
    `>=` / `<=` are treated identically to their strict counterparts —
    measure-zero crossings, fine for ODE event detection.
    """
    s = cond_str.strip()
    m = re.match(r'^(.+?)\s*(>=|<=|>|<)\s*(.+)$', s)
    if not m:
        raise ValueError(f"Cannot parse event condition: {cond_str!r}")
    lhs, op, rhs = m.group(1).strip(), m.group(2), m.group(3).strip()
    if op in ('>', '>='):
        return f"({rhs}) - ({lhs})"
    return f"({lhs}) - ({rhs})"


def _parse_reinit(reinit_str: str) -> Tuple[str, str]:
    """Parse a single `name = expr` reinit assignment."""
    m = re.match(r'^\s*(\w+)\s*=\s*(.+)$', reinit_str.strip())
    if not m:
        raise ValueError(f"Cannot parse reinit: {reinit_str!r}")
    return m.group(1), m.group(2).strip()


def _build_reinit_branch(
    state_names: List[str],
    alg_var_names: List[str],
    param_names: List[str],
    target_idx: int,
    rhs_expr: str,
):
    """Codegen one branch function `(state, t, z, p) -> updated state`."""
    src_lines = ["def _reinit_branch(state, t, z, p):"]
    for i, n in enumerate(state_names):
        src_lines.append(f"    {n} = state[{i}]")
    for i, n in enumerate(alg_var_names):
        src_lines.append(f"    {n} = z[{i}]")
    for i, n in enumerate(param_names):
        src_lines.append(f"    {n} = p[{i}]")
    src_lines.append(f"    return state.at[{target_idx}].set(({rhs_expr}))")
    src = "\n".join(src_lines) + "\n"
    ns = {'jnp': jnp, **_MATH_FUNCS}
    exec(compile(src, "<_reinit_branch>", "exec"), ns)
    return ns["_reinit_branch"]


# ---------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------- #

class DAEOptimizerJaxADEvents:
    """Adaptive-diffrax DAE optimizer with state-dependent events.

    The dynamics are integrated with diffrax adaptive RK methods; the
    algebraic constraint `g(t, x, z, p) = 0` is solved at every step
    via the custom_jvp + chord-Newton path; events are detected by a
    Bisection root finder on a composite `min(event_values)`; reinit
    equations are applied at each event boundary.

    Args:
        dae_data: parsed DAE spec (states / alg_vars / parameters /
            f / g / h / when).
        optimize_params: subset of parameter names to optimize (defaults
            to all).
        loss_type: 'sum' or 'mean'.
        diffrax_solver: 'Tsit5', 'Dopri5', 'Dopri8', 'Heun'.
        rtol, atol: tolerances for both the PID controller and the
            Bisection event root finder.
        newton_max_iter: chord iterations per algebraic solve.
        max_segments: static upper bound on # event-bounded segments
            per simulation. Padding segments after the last real event
            integrate trivially over [t_end, t_end].
        dtmax: optional max integration step for the PID controller —
            useful if the event window is much shorter than the natural
            adaptive step (otherwise diffrax can step right over the
            transit).
        diffrax_max_steps: per-segment internal step budget for diffrax.
    """

    def __init__(
        self,
        dae_data: Dict[str, Any],
        optimize_params: Optional[List[str]] = None,
        loss_type: str = 'sum',
        diffrax_solver: str = 'Tsit5',
        rtol: float = 1.0e-6,
        atol: float = 1.0e-6,
        newton_max_iter: int = 10,
        max_segments: int = 16,
        dtmax: Optional[float] = None,
        diffrax_max_steps: int = 4096,
        blend_sharpness: Optional[float] = None,
    ):
        if loss_type not in ('sum', 'mean'):
            raise ValueError(f"loss_type {loss_type!r} not in ('sum', 'mean')")

        self.dae_data = dae_data
        self.loss_type = loss_type
        self.diffrax_solver_name = diffrax_solver
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.newton_max_iter = int(newton_max_iter)
        self.max_segments = int(max_segments)
        self.dtmax = dtmax
        self.diffrax_max_steps = int(diffrax_max_steps)
        self.blend_sharpness = (None if blend_sharpness is None
                                else float(blend_sharpness))

        # Variable / parameter layout
        self.state_names = [s['name'] for s in dae_data.get('states', [])]
        self.alg_var_names = [a['name'] for a in dae_data.get('alg_vars', [])]
        self.param_names = [p['name'] for p in dae_data.get('parameters', [])]
        self.n_states = len(self.state_names)
        self.n_alg = len(self.alg_var_names)
        self.n_params_total = len(self.param_names)

        self.x0 = jnp.array(
            [s['start'] for s in dae_data['states']], dtype=jnp.float64
        )
        self.z0 = jnp.array(
            [a.get('start', 0.0) for a in dae_data['alg_vars']],
            dtype=jnp.float64,
        )
        self.p_all = jnp.array(
            [p['value'] for p in dae_data['parameters']], dtype=jnp.float64
        )

        if optimize_params is None:
            self.optimize_params = list(self.param_names)
            self.optimize_indices = list(range(self.n_params_total))
        else:
            self.optimize_params = list(optimize_params)
            self.optimize_indices = [
                self.param_names.index(n) for n in self.optimize_params
            ]
        self.n_params_opt = len(self.optimize_indices)
        self.optimize_indices_jax = jnp.array(
            self.optimize_indices, dtype=jnp.int32
        )

        # Equations
        self.f_eqs = self._parse_f_eqs()
        self.g_eqs = self._parse_g_eqs()
        self.h_eqs = self._parse_h_eqs()
        self.when_clauses = dae_data.get('when', None) or []
        self.n_events = len(self.when_clauses)
        self.n_outputs = len(self.h_eqs) if self.h_eqs else self.n_states

        if self.n_events == 0:
            raise ValueError(
                "No `when` clauses found in spec. Use "
                "`DAEOptimizerJaxAD` with `solver_method='diffrax'` "
                "for event-free DAEs."
            )

        self._build_jit_functions()

        print(f"\nDAEOptimizerJaxADEvents initialized:")
        print(f"  Differential states: {self.n_states}")
        print(f"  Algebraic variables: {self.n_alg}")
        print(f"  Parameters to optimize: {self.n_params_opt} of {self.n_params_total}")
        print(f"  Outputs: {self.n_outputs}")
        print(f"  Event clauses: {self.n_events}")
        print(f"  Diffrax solver: {self.diffrax_solver_name}  "
              f"rtol={self.rtol:g}  atol={self.atol:g}")
        print(f"  max_segments: {self.max_segments}  "
              f"newton_max_iter: {self.newton_max_iter}")
        if self.blend_sharpness is not None:
            print(f"  blend_sharpness: {self.blend_sharpness:g}  "
                  f"(soft-segment blending enabled)")

    # -- spec parsing ------------------------------------------------- #
    def _parse_f_eqs(self):
        eqs = []
        for s in (self.dae_data.get('f') or []):
            m = re.match(r'der\((\w+)\)\s*=\s*(.+)', s.strip())
            if not m:
                raise ValueError(f"Cannot parse f equation: {s!r}")
            eqs.append((m.group(1), m.group(2).strip()))
        return eqs

    def _parse_g_eqs(self):
        eqs = []
        for s in (self.dae_data.get('g') or []):
            m = re.match(r'0(?:\.0*)?\s*=\s*(.+)', s.strip())
            if not m:
                raise ValueError(f"Cannot parse g equation: {s!r}")
            eqs.append(m.group(1).strip())
        return eqs

    def _parse_h_eqs(self):
        eqs = []
        for s in (self.dae_data.get('h') or []) or []:
            if s.strip() in self.state_names + self.alg_var_names:
                eqs.append(('output', s.strip()))
                continue
            m = re.match(r'(\w+)\s*=\s*(.+)', s.strip())
            if m:
                eqs.append((m.group(1), m.group(2).strip()))
            else:
                eqs.append(('output', s.strip()))
        return eqs

    # -- core build --------------------------------------------------- #
    def _build_jit_functions(self):
        state_names = self.state_names
        alg_var_names = self.alg_var_names
        param_names = self.param_names
        f_eqs = self.f_eqs
        g_eqs = self.g_eqs
        h_eqs = self.h_eqs
        n_states = self.n_states
        n_alg = self.n_alg
        n_events = self.n_events
        newton_max_iter = self.newton_max_iter

        # 1. Fused codegen for f, g, h, event-vector --------------------- #
        f_rhs_by_state = dict(f_eqs)
        f_rhs_ordered = [f_rhs_by_state.get(n, '0.0') for n in state_names]
        eval_f = _build_fused_function(
            "_fused_f", state_names, alg_var_names, param_names, f_rhs_ordered
        )
        eval_g = _build_fused_function(
            "_fused_g", state_names, alg_var_names, param_names, list(g_eqs)
        )
        if eval_f is None or eval_g is None:
            raise ValueError(
                "Spec name collision with protected names; the events "
                "optimizer requires the codegen path."
            )

        if h_eqs:
            h_rhs = [expr for _, expr in h_eqs]
            eval_h = _build_fused_function(
                "_fused_h", state_names, alg_var_names, param_names, h_rhs
            )
            if eval_h is None:
                raise ValueError("Spec name collision in `h` block.")
        else:
            def eval_h(t, x, z, p):
                return x

        # Event vector: each row from the codegen of one `condition`.
        ev_exprs = [
            _parse_condition_to_event_expr(c['condition'])
            for c in self.when_clauses
        ]
        event_eval_xz = _build_fused_function(
            "_fused_events", state_names, alg_var_names, param_names, ev_exprs
        )
        if event_eval_xz is None:
            raise ValueError("Spec name collision in event conditions.")

        self._eval_f = eval_f
        self._eval_g = eval_g
        self._eval_h = eval_h
        self._event_eval_xz = event_eval_xz

        # 2. Algebraic solver: chord Newton + custom_jvp via IFT --------- #
        _dg_dz_fn = jacfwd(eval_g, argnums=2)
        eye_alg = jnp.eye(n_alg) if n_alg > 0 else None

        def solve_algebraic_newton_fwd(t, x, p, z_init):
            if n_alg == 0:
                return jnp.zeros(0, dtype=jnp.float64)
            jac = _dg_dz_fn(t, x, z_init, p) + 1e-12 * eye_alg
            lu_piv = lu_factor(jac)

            def chord_step(z, _):
                gv = eval_g(t, x, z, p)
                dz = lu_solve(lu_piv, -gv)
                return z + dz, None

            z_final, _ = jax.lax.scan(
                chord_step, z_init, None, length=newton_max_iter
            )
            return z_final

        @custom_jvp
        def solve_algebraic(t, x, p, z_init):
            return solve_algebraic_newton_fwd(t, x, p, z_init)

        @solve_algebraic.defjvp
        def solve_algebraic_jvp(primals, tangents):
            t, x, p, z_init = primals
            dt, dx, dp, _ = tangents
            z = solve_algebraic_newton_fwd(t, x, p, z_init)
            if n_alg == 0:
                return z, jnp.zeros_like(z)
            dg_dz = _dg_dz_fn(t, x, z, p) + 1e-12 * eye_alg
            _, dg_dot = jax.jvp(
                lambda tt, xx, pp: eval_g(tt, xx, z, pp),
                (t, x, p),
                (dt, dx, dp),
            )
            dz = -jnp.linalg.solve(dg_dz, dg_dot)
            return z, dz

        self._solve_algebraic = solve_algebraic

        # 3. Event vector / state-update wrappers ----------------------- #
        z0_const = self.z0

        def event_fn(t, x, p):
            """(n_events,) event vector. Algebraic vars are re-solved
            against current x so conditions referencing `z`/`DER_*`
            stay correct, but typically conditions reference only
            differential states + parameters."""
            z = solve_algebraic(t, x, p, z0_const)
            return event_eval_xz(t, x, z, p)

        self._event_fn = event_fn

        # Build a branch fn per clause; lax.switch dispatches by ev_idx.
        branch_fns = []
        for clause in self.when_clauses:
            target, rhs = _parse_reinit(clause['reinit'])
            if target not in state_names:
                raise ValueError(
                    f"reinit target {target!r} is not a differential "
                    f"state. (Algebraic-var reinit not supported.)"
                )
            idx = state_names.index(target)
            rhs_expr = _normalize_expr(rhs)
            branch_fns.append(_build_reinit_branch(
                state_names, alg_var_names, param_names, idx, rhs_expr
            ))

        def state_update(state, ev_idx, t, z, p):
            wrapped = [
                (lambda f: lambda s: f(s, t, z, p))(b) for b in branch_fns
            ]
            return jax.lax.switch(ev_idx, wrapped, state)

        self._state_update = state_update

        # 4. Diffrax solver / controller / composite event -------------- #
        solver_map = {
            'Tsit5': diffrax.Tsit5,
            'Dopri5': diffrax.Dopri5,
            'Dopri8': diffrax.Dopri8,
            'Heun': diffrax.Heun,
        }
        if self.diffrax_solver_name not in solver_map:
            raise ValueError(
                f"Unknown diffrax_solver {self.diffrax_solver_name!r}"
            )
        diff_solver = solver_map[self.diffrax_solver_name]()

        pid_kwargs = dict(rtol=self.rtol, atol=self.atol)
        if self.dtmax is not None:
            pid_kwargs['dtmax'] = float(self.dtmax)
        controller = diffrax.PIDController(**pid_kwargs)

        # Vector field: dx/dt = f(t, x, z(t,x,p), p)
        def vector_field(t, x, args):
            p_, _active = args
            z = solve_algebraic(t, x, p_, z0_const)
            return eval_f(t, x, z, p_)

        term = diffrax.ODETerm(vector_field)

        # Composite cond_fn: min over events, masked to currently-active
        # ones (positive at segment start). Mirrors bouncing_balls_n_jax.
        _LARGE = jnp.asarray(1.0e30, dtype=jnp.float64)
        EPS_ACTIVE = 1.0e-9

        # Note: diffrax 0.7+ calls `cond_fn` with all-keyword args, so the
        # second parameter MUST be named `y` (not `x`) for the keyword
        # call to bind correctly.
        def composite_cond_fn(t, y, args, **kwargs):
            p_, active = args
            ev = event_fn(t, y, p_)
            return jnp.min(jnp.where(active, ev, _LARGE))

        composite_event = diffrax.Event(
            cond_fn=composite_cond_fn,
            root_finder=optx.Bisection(rtol=self.rtol, atol=self.atol),
        )

        max_segments = self.max_segments
        max_steps = self.diffrax_max_steps
        blend_sharpness = self.blend_sharpness

        # 5. simulate_at_targets — three-stage segmented scheme --------- #
        def simulate_at_targets(p, target_times, x0):
            """Return state trajectory at `target_times` with events
            applied. Padding to a fixed `max_segments` keeps the JIT
            cache stable across optimization iters."""
            t_end_total = (
                target_times[-1] + jnp.asarray(1.0e-6, dtype=jnp.float64)
            )

            # ----- Stage 1: event detection ----------------------------
            def detect_step(carry, _k):
                cur_t, cur_x = carry

                ev_at_start = event_fn(cur_t, cur_x, p)
                active = jax.lax.stop_gradient(ev_at_start) > EPS_ACTIVE

                t1 = jnp.maximum(
                    t_end_total, cur_t + jnp.asarray(1e-12, dtype=jnp.float64)
                )
                dt0 = jnp.maximum(
                    (t1 - cur_t) * 1e-3,
                    jnp.asarray(1e-12, dtype=jnp.float64),
                )

                sol = diffrax.diffeqsolve(
                    term, diff_solver,
                    t0=cur_t, t1=t1, dt0=dt0,
                    y0=cur_x,
                    args=(p, active),
                    event=composite_event,
                    stepsize_controller=controller,
                    max_steps=max_steps,
                )

                event_fired = sol.event_mask
                et = sol.ts[-1]
                state_at = sol.ys[-1]

                already_done = cur_t >= (
                    t_end_total - jnp.asarray(1e-12, dtype=jnp.float64)
                )
                at_boundary = et >= (
                    t_end_total - jnp.asarray(1e-12, dtype=jnp.float64)
                )
                is_real = event_fired & ~already_done & ~at_boundary

                ev_vals = event_fn(et, state_at, p)
                ev_vals_sg = jax.lax.stop_gradient(
                    jnp.where(active, ev_vals, _LARGE)
                )
                ev_idx = jnp.argmin(ev_vals_sg).astype(jnp.int32)

                z_at = solve_algebraic(et, state_at, p, z0_const)
                state_after = jax.lax.cond(
                    is_real,
                    lambda: state_update(state_at, ev_idx, et, z_at, p),
                    lambda: state_at,
                )

                new_t = jnp.where(is_real, et, t_end_total)
                seg = {
                    't_start': cur_t,
                    't_end': jnp.where(is_real, et, t_end_total),
                    'state_start': cur_x,
                    'is_real': ~already_done,
                }
                return (new_t, state_after), seg

            (final_t, _), segments = jax.lax.scan(
                detect_step,
                (target_times[0], x0),
                xs=jnp.arange(max_segments),
            )
            # `current_t` advances to `t_end_total` only when an iter's
            # integration made it to the right endpoint without hitting
            # an event. If the scan exits with `final_t < t_end_total`
            # we ran out of segment slots before the final time — at
            # least one event past the cap was silently missed and the
            # trajectory beyond `final_t` is bogus. Surface this so the
            # public `simulate()` can raise on it.
            scan_saturated = final_t < (
                t_end_total - jnp.asarray(1e-9, dtype=jnp.float64)
            )

            # ----- Stage 2: per-segment integration with SaveAt --------
            all_active_post = jnp.ones(n_events, dtype=jnp.bool_)

            def integrate_step(_carry, seg):
                t_start, t_end_, state_start, _is_real_ = seg
                t_end_safe = jnp.maximum(
                    t_end_, t_start + jnp.asarray(1e-12, dtype=jnp.float64)
                )
                save_ts = jnp.clip(target_times, t_start, t_end_safe)
                dt0 = jnp.maximum(
                    (t_end_safe - t_start) * 1e-3,
                    jnp.asarray(1e-12, dtype=jnp.float64),
                )
                sol = diffrax.diffeqsolve(
                    term, diff_solver,
                    t0=t_start, t1=t_end_safe, dt0=dt0,
                    y0=state_start,
                    args=(p, all_active_post),
                    saveat=diffrax.SaveAt(ts=save_ts),
                    stepsize_controller=controller,
                    max_steps=max_steps,
                )
                return None, sol.ys

            _, all_ys = jax.lax.scan(
                integrate_step, None,
                (segments['t_start'], segments['t_end'],
                 segments['state_start'], segments['is_real']),
            )
            # all_ys: (max_segments, n_targets, n_states)

            # ----- Stage 3: right-continuous segment selection ---------
            TOL = jnp.asarray(1e-12, dtype=jnp.float64)
            t_starts = segments['t_start']
            t_ends = segments['t_end']
            is_real_arr = segments['is_real']

            in_range = (
                (target_times[None, :] >= t_starts[:, None] - TOL)
                & (target_times[None, :] <= t_ends[:, None] + TOL)
                & is_real_arr[:, None]
            )
            n_targets = target_times.shape[0]

            if blend_sharpness is None:
                # Hard right-continuous segment selection (default).
                seg_indices = jnp.arange(max_segments)
                weighted = (
                    in_range.astype(jnp.int32) * (seg_indices[:, None] + 1)
                )
                chosen = jnp.argmax(weighted, axis=0)
                x_traj = all_ys[chosen, jnp.arange(n_targets)]
            else:
                # Sigmoid-blended segment selection. For each (segment k,
                # target i) we build a smooth window
                #   w_{k,i} = σ(β·(t_i − t_start_k)) · σ(β·(t_end_k − t_i))
                # which is ≈1 when `t_i` is well inside segment k and
                # ≈0 outside. Multiplying by `is_real` zeroes out the
                # padding-segment contributions. Normalizing across
                # segments yields a partition-of-unity in the limit
                # β→∞, recovering the hard argmax. For finite β,
                # contributions from neighboring segments smear across
                # the event boundary — widening the basin of attraction
                # for gradient-based optimization (matches what the
                # discrete-adjoint runner does for its loss). The
                # trajectory is the weighted average of each segment's
                # locally-integrated value at `t_i`.
                #
                # Cost: one extra (max_segments × n_targets) sigmoid
                # evaluation; the per-segment integrations were already
                # done in stage 2.
                beta = jnp.asarray(blend_sharpness, dtype=target_times.dtype)
                left = jax.nn.sigmoid(
                    beta * (target_times[None, :] - t_starts[:, None])
                )
                right = jax.nn.sigmoid(
                    beta * (t_ends[:, None] - target_times[None, :])
                )
                w = left * right * is_real_arr[:, None].astype(left.dtype)
                w_norm = w / (jnp.sum(w, axis=0, keepdims=True) + 1e-30)
                # all_ys: (max_segments, n_targets, n_states)
                # w_norm: (max_segments, n_targets)
                x_traj = jnp.sum(w_norm[:, :, None] * all_ys, axis=0)

            n_real_segs = jnp.sum(is_real_arr.astype(jnp.int32))
            # NOTE: when `scan_saturated`, the loss-level sat_factor in
            # `compute_loss` turns BOTH loss and gradient to NaN. We do
            # NOT inject NaN per-target here — that adds an extra op to
            # the AD tape and can affect the gradient even when targets
            # all match.
            return x_traj, n_real_segs, scan_saturated

        self._simulate_at_targets = simulate_at_targets

        @jit
        def simulate_at_targets_jit(p, target_times, x0):
            return simulate_at_targets(p, target_times, x0)

        self._simulate_at_targets_jit = simulate_at_targets_jit

        # 6. Loss and gradient ------------------------------------------- #
        optimize_indices_jax = self.optimize_indices_jax
        p_all = self.p_all
        h_is_identity = not h_eqs

        def compute_loss(p_opt, target_times, y_target, x0):
            p_full = p_all.at[optimize_indices_jax].set(p_opt)
            x_traj, _, scan_saturated = simulate_at_targets(
                p_full, target_times, x0
            )
            if h_is_identity:
                y_traj = x_traj
            else:
                def reconstruct(t, x):
                    z = solve_algebraic(t, x, p_full, z0_const)
                    return eval_h(t, x, z, p_full)
                y_traj = vmap(reconstruct)(target_times, x_traj)
            err = y_traj - y_target
            if self.loss_type == 'mean':
                loss = jnp.mean(err ** 2)
            else:
                loss = jnp.sum(err ** 2)

            # Saturation gate. Only kicks in when `scan_saturated` is
            # True; in every other case `sat_factor == 1.0` and this is
            # a numerical no-op (`d loss / dp` unchanged). When True,
            # both `loss` and `d loss / dp` become NaN — multiplication
            # by NaN propagates through forward AND backward, unlike
            # `jnp.where(saturated, NaN, loss)` which would only NaN
            # the forward (its "unselected branch" gradient is zero).
            sat_factor = jnp.where(
                scan_saturated,
                jnp.asarray(jnp.nan, dtype=loss.dtype),
                jnp.asarray(1.0, dtype=loss.dtype),
            )
            return loss * sat_factor

        self._compute_loss = compute_loss
        self._loss_and_grad = jit(value_and_grad(compute_loss))

    # -- public API --------------------------------------------------- #
    def simulate(self, target_times, p=None) -> Dict[str, np.ndarray]:
        """Simulate at the given target times. Returns dict with
        keys 't', 'x', 'z', 'y'. Only `x` is integrated; `z` and `y`
        are reconstructed at the saved points.

        Raises `RuntimeError` if event detection saturated `max_segments`
        (i.e. an event past the cap was missed and the trajectory after
        that point is unusable). Bump `max_segments` and re-run.
        """
        target_times_j = jnp.asarray(target_times, dtype=jnp.float64)
        p_j = self.p_all if p is None else jnp.asarray(p, dtype=jnp.float64)
        x_traj, n_real_segs, scan_saturated = self._simulate_at_targets_jit(
            p_j, target_times_j, self.x0
        )

        if bool(scan_saturated):
            raise RuntimeError(
                f"Event detection saturated `max_segments="
                f"{self.max_segments}` — at least one event past the "
                f"cap was missed and the trajectory beyond that point "
                f"is unusable. Real segments detected: "
                f"{int(n_real_segs)}. Increase `max_segments` (in the "
                f"YAML or constructor) and re-run."
            )
        if bool(jnp.any(jnp.isnan(x_traj))):
            n_nan = int(jnp.sum(jnp.any(jnp.isnan(x_traj), axis=1)))
            raise RuntimeError(
                f"`simulate` produced {n_nan} NaN target points (no "
                f"real segment covers them). This typically means "
                f"`max_segments` is too small even though saturation "
                f"didn't trip — bump it and re-run."
            )

        if self.h_eqs:
            def recon(t, x):
                z = self._solve_algebraic(t, x, p_j, self.z0)
                y = self._eval_h(t, x, z, p_j)
                return z, y
            z_traj, y_traj = vmap(recon)(target_times_j, x_traj)
        else:
            n_t = x_traj.shape[0]
            z_traj = jnp.zeros((n_t, self.n_alg), dtype=x_traj.dtype)
            y_traj = x_traj

        return {
            't': np.asarray(target_times_j),
            'x': np.asarray(x_traj).T,
            'z': np.asarray(z_traj).T,
            'y': np.asarray(y_traj).T,
        }
