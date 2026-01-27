"""
Explicit Discrete Adjoint Optimizer for DAEs with Events.

Uses the discretization viewpoint where the entire hybrid trajectory is treated
as a solution to a large nonlinear algebraic system R(U, θ) = 0.

The unknown vector U contains:
- All state values at grid points: {Y_{i,k}} for segment i, time point k
- All event times: {τ_i} for i = 1..M

The residual R includes:
- Trapezoidal timestep residuals within each segment
- Event guard constraints (ψ = 0 at event surface)
- Event reset constraints (state jump equations)

Gradient computation via discrete adjoint:
    R_U^T λ = J_U^T
    ∇_θ J = J_θ - R_θ^T λ
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from typing import Tuple, Dict, List, NamedTuple, Optional
import numpy as np
import time
from functools import partial
import re

jax.config.update("jax_enable_x64", True)

from .dae_solver import DAESolver, AugmentedSolution

# =============================================================================
# 1. Equation Compiler
# =============================================================================

def compile_equations_to_jax(eqn_strings, state_names, alg_names, param_names, extra_args=None):
    """Parses string equations into a single JAX-jittable function."""
    if not eqn_strings:
        return lambda t, x, z, p: jnp.array([])

    subs = []
    if extra_args:
        for name, repl in extra_args.items(): subs.append((name, repl))
    for i, name in enumerate(state_names): subs.append((name, f"x[{i}]"))
    for i, name in enumerate(alg_names):   subs.append((name, f"z[{i}]"))
    for i, name in enumerate(param_names): subs.append((name, f"p[{i}]"))
    subs.append(('time', 't'))
    subs.append(('t', 't'))
    subs.sort(key=lambda item: len(item[0]), reverse=True)

    processed_eqns = []
    math_funcs = {'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'exp', 'log', 'sqrt', 'abs', 'pow', 'min', 'max'}
    
    for eq in eqn_strings:
        processed_eq = eq
        for name, repl in subs:
            if name in math_funcs: continue
            pattern = r'(?<!\.)\b' + re.escape(name) + r'\b'
            processed_eq = re.sub(pattern, repl, processed_eq)
        processed_eqns.append(processed_eq)

    return_expr = "[" + ", ".join(processed_eqns) + "]"
    
    def create_closure():
        exec_ns = {'jnp': jnp}
        for f in math_funcs:
            if hasattr(jnp, f): exec_ns[f] = getattr(jnp, f)
        exec_ns['power'] = jnp.power 
        source = f"def compiled_fn(t, x, z, p): return jnp.array({return_expr})"
        exec(source, exec_ns)
        return exec_ns['compiled_fn']

    return create_closure()

# =============================================================================
# 2. Standalone JIT Kernels
# =============================================================================



@partial(jit, static_argnames=['f_fn', 'n_states', 'n_alg', 'optimize_indices'])
def backward_scan_kernel_implicit_masked(
    lambda_init, grad_p_init,
    t_kp1_seq, t_k_seq, y_kp1_seq, y_k_seq, dL_k_seq, mask_seq,
    p_all_default, optimize_indices, n_states, n_alg, f_fn
):
    """
    Masked version of backward scan kernel for padded arrays.
    mask_seq: (n_steps,) array of 1.0 for valid steps, 0.0 for padding.
    When mask=0, the step is an identity operation (lambda passes through, grad unchanged).
    """
    p_opt_initial = p_all_default[jnp.array(optimize_indices)]

    def scan_step(carry, inputs):
        lambda_kp1, grad_p_acc = carry
        t_kp1, t_k, y_kp1, y_k, dL_k, mask = inputs
        h = t_kp1 - t_k

        x_k = y_k[:n_states]
        x_kp1 = y_kp1[:n_states]
        z_k = y_k[n_states:n_states + n_alg] if n_alg > 0 else jnp.array([])

        # 1. Compute Jacobians df/dx (State)
        def f_wrapper_x(x, t_val): return f_fn(t_val, x, z_k, p_all_default)
        J_k = jax.jacfwd(f_wrapper_x)(x_k, t_k)
        J_kp1 = jax.jacfwd(f_wrapper_x)(x_kp1, t_kp1)

        # 2. Implicit Adjoint Solve
        I = jnp.eye(n_states)
        factor = 0.5 * h
        A_matrix = I - factor * J_kp1.T

        # Safe solve: For padded steps (h≈0), A_matrix≈I, so solve is stable
        lambda_mid = jnp.linalg.solve(A_matrix, lambda_kp1)

        # 3. Update Adjoint
        lambda_k_computed = (I + factor * J_k.T) @ lambda_mid + dL_k[:n_states]

        # 4. Parameter Gradient
        def f_wrapper_p(p_subset):
            p_full = p_all_default.at[jnp.array(optimize_indices)].set(p_subset)
            f_k = f_fn(t_k, x_k, z_k, p_full)
            f_kp1 = f_fn(t_kp1, x_kp1, z_k, p_full)
            return -(h / 2.0) * (f_k + f_kp1)

        dr_dp = jax.jacfwd(f_wrapper_p)(p_opt_initial)
        grad_p_computed = grad_p_acc - (dr_dp.T @ lambda_mid)

        # Apply mask: use computed values if valid, pass-through if padded
        lambda_k = jnp.where(mask > 0.5, lambda_k_computed, lambda_kp1)
        grad_p_new = jnp.where(mask > 0.5, grad_p_computed, grad_p_acc)

        return (lambda_k, grad_p_new), lambda_k

    (lambda_final, grad_p_final), _ = lax.scan(
        scan_step,
        (lambda_init, grad_p_init),
        (t_kp1_seq, t_k_seq, y_kp1_seq, y_k_seq, dL_k_seq, mask_seq)
    )
    return lambda_final, grad_p_final

# =============================================================================
# NOTE: Residual Separation Compliance
# The Flow Residuals (backward_scan_kernel) operate strictly on segment-interior
# points (including boundaries that approach the event).
# The Event Residuals (backward_event_kernel) operate strictly on the jump
# condition at the event time, connecting the end of one segment to the start
# of the next. Strict separation is maintained.
# =============================================================================


@partial(jit, static_argnames=['f_fn', 'jump_fn', 'zc_fn', 'n_states', 'event_idx', 'optimize_indices'])
def backward_event_kernel(lambda_post, x_pre, z_pre, x_post, z_post, tau, p_all, optimize_indices, event_idx, n_states, f_fn, jump_fn, zc_fn, dL_dtau):
    lambda_post_x = lambda_post[:n_states]
    p_opt = p_all[jnp.array(optimize_indices)]

    def jump_x_wrapper(x): return jump_fn(x, z_pre, tau, p_all, event_idx)
    J_jump_x = jax.jacfwd(jump_x_wrapper)(x_pre)
    
    def zc_x_wrapper(x): return zc_fn(tau, x, z_pre, p_all)[event_idx]
    def zc_t_wrapper(t): return zc_fn(t, x_pre, z_pre, p_all)[event_idx]
    
    grad_psi_x = jax.grad(zc_x_wrapper)(x_pre)
    grad_psi_t = jax.grad(zc_t_wrapper)(tau)

    f_pre = f_fn(tau, x_pre, z_pre, p_all)
    f_post = f_fn(tau, x_post, z_post, p_all)
    lambda_tilde = J_jump_x.T @ lambda_post_x
    
    H_post = jnp.dot(lambda_post_x, f_post)
    H_pre = jnp.dot(lambda_tilde, f_pre)
    dpsi_dt_total = grad_psi_t + jnp.dot(grad_psi_x, f_pre)
    safe_denom = jnp.where(jnp.abs(dpsi_dt_total) < 1e-8, 1e-8 * jnp.sign(dpsi_dt_total + 1e-12), dpsi_dt_total)
    
    gamma = (H_post - H_pre + dL_dtau) / safe_denom
    gamma = jnp.clip(gamma, -1e6, 1e6)

    lambda_pre = lambda_tilde - gamma * grad_psi_x

    def jump_p_wrapper(p_sub):
        p_full = p_all.at[jnp.array(optimize_indices)].set(p_sub)
        return jump_fn(x_pre, z_pre, tau, p_full, event_idx)
    def zc_p_wrapper(p_sub):
        p_full = p_all.at[jnp.array(optimize_indices)].set(p_sub)
        return zc_fn(tau, x_pre, z_pre, p_full)[event_idx]

    J_jump_p = jax.jacfwd(jump_p_wrapper)(p_opt)
    grad_psi_p = jax.grad(zc_p_wrapper)(p_opt)
    grad_p_event = J_jump_p.T @ lambda_post_x + gamma * grad_psi_p
    return lambda_pre, grad_p_event

# --- Unified Prediction Kernel (Optimization + Inference) ---
@partial(jit, static_argnames=['eval_h_fn', 'n_outputs'])
def predict_trajectory_kernel(segments_t, segments_x, segments_z, events_tau, target_times, p_all, eval_h_fn, n_outputs, blend_sharpness=100.0):
    """
    Unified JIT kernel for predicting trajectory outputs.
    Used by both the optimization loss and the public predict_outputs method.
    """
    def predict_single_time(t_q):
        y_accum = jnp.zeros(n_outputs)
        w_accum = 0.0

        # Iterate over tuple of segments (unrolled by JAX)
        for i in range(len(segments_t)):
            t_seg = segments_t[i]
            x_seg = segments_x[i]
            z_seg = segments_z[i]

            t_start, t_end = t_seg[0], t_seg[-1]
            lower = t_start if i == 0 else events_tau[i-1]
            upper = t_end if i == len(segments_t)-1 else events_tau[i]

            # Sigmoid Masking
            mask = jax.nn.sigmoid(blend_sharpness * (t_q - lower)) * jax.nn.sigmoid(blend_sharpness * (upper - t_q))

            # Interpolation
            t_clip = jnp.clip(t_q, t_start, t_end)
            idx = jnp.searchsorted(t_seg, t_clip, side='right') - 1
            idx = jnp.clip(idx, 0, len(t_seg)-2)
            t0, t1 = t_seg[idx], t_seg[idx+1]
            s = jnp.clip((t_clip - t0) / (t1 - t0 + 1e-12), 0.0, 1.0)

            x_i = x_seg[idx] * (1-s) + x_seg[idx+1] * s
            z_i = z_seg[idx] * (1-s) + z_seg[idx+1] * s if z_seg.size > 0 else jnp.array([])

            h_val = eval_h_fn(t_clip, x_i, z_i, p_all)
            y_accum += mask * h_val
            w_accum += mask

        return y_accum / (w_accum + 1e-8)

    return vmap(predict_single_time)(target_times)


# --- Linear Interpolation Prediction Kernel (PyTorch-style) ---
@partial(jit, static_argnames=['eval_h_fn', 'n_outputs'])
def predict_trajectory_kernel_linear(segments_t, segments_x, segments_z, target_times, p_all, eval_h_fn, n_outputs):
    """
    Linear interpolation prediction kernel (matches PyTorch implementation).
    No sigmoid blending - just concatenates all segments and interpolates.
    """
    # Concatenate all segments into single arrays
    all_t = jnp.concatenate(segments_t)
    all_x = jnp.concatenate(segments_x)

    # Handle algebraic variables
    if segments_z[0].size > 0:
        all_z = jnp.concatenate(segments_z)
    else:
        all_z = jnp.zeros((len(all_t), 0))

    def predict_single_time(t_q):
        # Find interpolation index
        idx = jnp.searchsorted(all_t, t_q, side='right') - 1
        idx = jnp.clip(idx, 0, len(all_t) - 2)

        t0, t1 = all_t[idx], all_t[idx + 1]
        dt = t1 - t0
        dt_safe = jnp.where(jnp.abs(dt) < 1e-12, 1e-12, dt)
        s = jnp.clip((t_q - t0) / dt_safe, 0.0, 1.0)

        x_i = all_x[idx] * (1 - s) + all_x[idx + 1] * s
        z_i = all_z[idx] * (1 - s) + all_z[idx + 1] * s if all_z.shape[1] > 0 else jnp.array([])

        return eval_h_fn(t_q, x_i, z_i, p_all)

    return vmap(predict_single_time)(target_times)


# --- Padded Linear Interpolation Prediction Kernel (PyTorch-style) ---
@partial(jit, static_argnames=['eval_h_fn', 'n_outputs', 'max_segments', 'max_points', 'n_alg'])
def predict_trajectory_kernel_padded_linear(
    t_start, pad_tau, pad_x, pad_z, seg_mask,
    target_times, p_all, eval_h_fn, n_outputs,
    max_segments, max_points, n_alg
):
    """
    JIT-stable linear interpolation prediction kernel (matches PyTorch implementation).
    Time grid is constructed dynamically from pad_tau to ensure differentiability w.r.t. event times.

    Args:
        t_start: Start time of simulation
        pad_tau: (max_segments,) - padded event times (ends of segments)
        pad_x: (max_segments, max_points, n_states) - padded state arrays
        pad_z: (max_segments, max_points, n_alg_padded) - padded algebraic arrays
        seg_mask: (max_segments,) - 1.0 for valid segments, 0.0 for padding
        target_times: (n_targets,) - query times
        p_all: parameter vector
        eval_h_fn: output function
        n_outputs: number of outputs
        max_segments: static max number of segments
        max_points: static max points per segment
        n_alg: number of algebraic variables
    """
    n_states = pad_x.shape[2]

    # Dynamically construct time grid to preserve gradients w.r.t pad_tau
    # Seg 0: t_start -> pad_tau[0]
    # Seg i: pad_tau[i-1] -> pad_tau[i]
    starts = jnp.concatenate([jnp.array([t_start]), pad_tau[:-1]])
    ends = pad_tau
    
    # Grid 0..1 per segment: shape (max_points,)
    unit_grid = jnp.linspace(0.0, 1.0, max_points)
    
    # Broadcast to (max_segments, max_points)
    # dynamic_t[i, k] = start[i] + unit_grid[k] * (end[i] - start[i])
    dynamic_t = starts[:, None] + (ends - starts)[:, None] * unit_grid[None, :]

    # Build concatenated timeline from all valid segments
    # Each segment contributes max_points entries
    flat_t = dynamic_t.reshape(-1)  # (max_segments * max_points,)
    flat_x = pad_x.reshape(-1, n_states)
    
    if n_alg > 0:
        flat_z = pad_z.reshape(-1, pad_z.shape[2])
    else:
        # Dummy flat_z if n_alg is 0 (to allow indexing)
        flat_z = jnp.zeros((flat_t.shape[0], 0))

    # Create validity mask per point (expanded from segment mask)
    point_mask = jnp.repeat(seg_mask, max_points)

    def predict_single_time(t_q):
        # Find which segment contains this time
        # We iterate through segments and pick the LAST one that contains t_q
        # This ensures we use POST-event values at boundaries (matching PyTorch behavior)

        def find_in_segment(carry, seg_data):
            result = carry
            t_seg, x_seg, z_seg, is_valid = seg_data

            t_start_seg, t_end_seg = t_seg[0], t_seg[-1]

            # Check if t_q is in this segment's range
            in_range = (t_q >= t_start_seg) & (t_q <= t_end_seg) & (is_valid > 0.5)

            # Interpolation within segment
            # Using searchsorted on the dynamic grid segment
            idx = jnp.searchsorted(t_seg, t_q, side='right') - 1
            idx = jnp.clip(idx, 0, max_points - 2)
            t0, t1 = t_seg[idx], t_seg[idx + 1]
            dt = t1 - t0
            dt_safe = jnp.where(jnp.abs(dt) < 1e-12, 1e-12, dt)
            s = jnp.clip((t_q - t0) / dt_safe, 0.0, 1.0)

            x_i = x_seg[idx] * (1 - s) + x_seg[idx + 1] * s
            z_i = z_seg[idx] * (1 - s) + z_seg[idx + 1] * s if n_alg > 0 else jnp.array([])
            z_i = z_i[:n_alg] if n_alg > 0 else jnp.array([])

            h_val = eval_h_fn(t_q, x_i, z_i, p_all)

            # Update result if this segment contains the query time
            # (overwrites previous matches, so we get the LAST matching segment)
            new_result = jnp.where(in_range, h_val, result)

            return new_result, None

        init = jnp.zeros(n_outputs)
        # Stack segment data for scan
        seg_data = (dynamic_t, pad_x, pad_z, seg_mask)

        result, _ = lax.scan(find_in_segment, init, seg_data)

        return result

    return vmap(predict_single_time)(target_times)


# --- Padded Prediction Kernel for JIT stability ---
@partial(jit, static_argnames=['eval_h_fn', 'n_outputs', 'max_segments', 'max_points', 'n_alg'])
def predict_trajectory_kernel_padded(
    pad_t, pad_x, pad_z, events_tau, seg_mask,
    target_times, p_all, eval_h_fn, n_outputs,
    max_segments, max_points, n_alg, blend_sharpness=100.0
):
    """
    JIT-stable prediction kernel with fixed-shape padded arrays.

    Args:
        pad_t: (max_segments, max_points) - padded time arrays
        pad_x: (max_segments, max_points, n_states) - padded state arrays
        pad_z: (max_segments, max_points, n_alg_padded) - padded algebraic arrays
        events_tau: (max_segments,) - event times (padded)
        seg_mask: (max_segments,) - 1.0 for valid segments, 0.0 for padding
        target_times: (n_targets,) - query times
        p_all: parameter vector
        eval_h_fn: output function
        n_outputs: number of outputs
        max_segments: static max number of segments
        max_points: static max points per segment
        n_alg: number of algebraic variables (0 means use zeros)
        blend_sharpness: sigmoid blending sharpness
    """
    def predict_single_time(t_q):
        def segment_contribution(i, carry):
            y_accum, w_accum = carry

            t_seg = pad_t[i]
            x_seg = pad_x[i]
            z_seg = pad_z[i]
            is_valid = seg_mask[i]

            t_start, t_end = t_seg[0], t_seg[-1]

            # Determine boundaries using event times
            lower = jnp.where(i == 0, t_start, events_tau[i - 1])
            upper = jnp.where(i == max_segments - 1, t_end, events_tau[i])

            # Sigmoid masking for smooth blending between segments
            mask = (jax.nn.sigmoid(blend_sharpness * (t_q - lower)) *
                    jax.nn.sigmoid(blend_sharpness * (upper - t_q)))
            mask = mask * is_valid  # Zero out invalid (padded) segments

            # Interpolation within segment
            t_clip = jnp.clip(t_q, t_start, t_end)
            idx = jnp.searchsorted(t_seg, t_clip, side='right') - 1
            idx = jnp.clip(idx, 0, max_points - 2)
            t0, t1 = t_seg[idx], t_seg[idx + 1]
            s = jnp.clip((t_clip - t0) / (t1 - t0 + 1e-12), 0.0, 1.0)

            x_i = x_seg[idx] * (1 - s) + x_seg[idx + 1] * s
            # For algebraic vars: use interpolated value or empty array
            z_i = jnp.where(n_alg > 0,
                           z_seg[idx] * (1 - s) + z_seg[idx + 1] * s,
                           z_seg[idx])[:n_alg] if n_alg > 0 else jnp.array([])

            h_val = eval_h_fn(t_clip, x_i, z_i, p_all)
            y_accum = y_accum + mask * h_val
            w_accum = w_accum + mask

            return (y_accum, w_accum)

        init = (jnp.zeros(n_outputs), 0.0)
        y_final, w_final = lax.fori_loop(0, max_segments, segment_contribution, init)
        return y_final / (w_final + 1e-8)

    return vmap(predict_single_time)(target_times)

# =============================================================================
# 3. Main Optimizer Class
# =============================================================================

class DAEOptimizerPadded:
    def __init__(self, dae_data: Dict, optimize_params: List[str], solver=None, verbose: bool = True,
                 blend_sharpness: float = 100.0, max_segments: int = 20, 
                 ncp: int = 200, safety_buffer_pct: float = 1.2,
                 prediction_method: str = 'sigmoid'):
        """
        Args:
            dae_data: DAE specification dictionary
            optimize_params: List of parameter names to optimize
            solver: Optional DAESolver instance
            verbose: Print progress
            blend_sharpness: Sigmoid blending sharpness (only used if prediction_method='sigmoid')
            max_segments: Maximum number of segments for JIT stability
            ncp: Target number of collocation points (used to size the buffer)
            safety_buffer_pct: Buffer multiplier (e.g. 1.2 for 20% safety margin)
            prediction_method: 'sigmoid' or 'linear'
        """
        self.dae_data = dae_data
        self.optimize_params = optimize_params
        self.verbose = verbose
        self.blend_sharpness = blend_sharpness
        self.prediction_method = prediction_method.lower()

        self.solver = solver if solver else DAESolver(dae_data, verbose=verbose)
        self.param_names = [p['name'] for p in dae_data['parameters']]
        self.p_all = np.array([p['value'] for p in dae_data['parameters']])
        self.optimize_indices = tuple([self.param_names.index(p) for p in optimize_params])
        self.n_opt_params = len(optimize_params)
        self.p_opt = np.array([self.p_all[i] for i in self.optimize_indices])

        self.n_states = len(dae_data['states'])
        self.n_alg = len(dae_data.get('alg_vars', []))
        self.n_total = self.n_states + self.n_alg

        # Padding limits for JIT stability
        self.max_segments = max_segments
        self.max_points_per_seg = int(ncp * safety_buffer_pct)
        if self.verbose:
            print(f"  Max segments: {self.max_segments}")
            print(f"  Max points per seg: {self.max_points_per_seg} (ncp={ncp}, buffer={safety_buffer_pct})")

        if self.verbose:
            print(f"  Prediction method: {self.prediction_method}")

        self._compile_jax_functions()

    def _compile_jax_functions(self):
        s, a, p = self.solver.state_names, self.solver.alg_names, self.param_names
        self._eval_f = compile_equations_to_jax(self.solver.f_funcs, s, a, p)
        self._eval_g = compile_equations_to_jax(self.solver.g_funcs, s, a, p)
        self._eval_zc = compile_equations_to_jax(self.solver.zc_funcs if self.solver.zc_funcs else [], s, a, p)
        
        h_eqs = self.dae_data.get('h', None)
        if h_eqs:
            clean_h = [eq.split('=', 1)[1].strip() if '=' in eq else eq for eq in h_eqs]
            self._eval_h = compile_equations_to_jax(clean_h, s, a, p)
        else:
            self._eval_h = lambda t,x,z,p: x
        self.n_outputs = len(h_eqs) if h_eqs else self.n_states

        self.jump_residual_funcs = []
        if hasattr(self.solver, 'event_reinit_exprs'):
            for i, reinit_expr in enumerate(self.solver.event_reinit_exprs):
                var_name = self.solver.event_reinit_var_names[i]
                reinit_modified = re.sub(r'prev\(\s*(\w+)\s*\)', r'prev_\1', reinit_expr)
                subs = [(f"prev_{n}", f"x[{j}]") for j, n in enumerate(s)] + \
                       [(f"prev_{n}", f"z[{j}]") for j, n in enumerate(a)] + \
                       [(n, f"p[{j}]") for j, n in enumerate(p)] + \
                       [(var_name, "val_new"), ('time', 't'), ('t', 't')]
                subs.sort(key=lambda item: len(item[0]), reverse=True)
                
                processed_eq = reinit_modified
                for name, repl in subs:
                    processed_eq = re.sub(r'(?<!\.)\b' + re.escape(name) + r'\b', repl, processed_eq)
                
                exec_ns = {'jnp': jnp}
                exec(f"def jump_res(t, x, z, p, val_new): return jnp.array([{processed_eq}])[0]", exec_ns)
                self.jump_residual_funcs.append(exec_ns['jump_res'])

    def _eval_jump(self, x_pre, z_pre, tau, p_all, event_idx):
        x_post = x_pre.copy()
        if not self.jump_residual_funcs: return x_post
        var_type, var_idx = self.solver.event_reinit_vars[event_idx]
        res_fn = self.jump_residual_funcs[event_idx]
        val_at_0 = res_fn(tau, x_pre, z_pre, p_all, 0.0)
        val_at_1 = res_fn(tau, x_pre, z_pre, p_all, 1.0)
        coeff = val_at_1 - val_at_0
        new_val = -val_at_0 / jnp.where(jnp.abs(coeff) < 1e-12, 1.0, coeff)
        if var_type == 'state': x_post = x_post.at[var_idx].set(new_val)
        return x_post

    def _pad_trajectory(self, aug_sol: AugmentedSolution):
        """
        Pad trajectory segments to fixed shapes for JAX JIT stability.
        
        Logic:
        - If segment > max_points: Truncate (drop end points).
        - If segment < max_points: Pad (duplicate last point).
        - Mask: 1.0 for valid points, 0.0 for padded points.
        """
        n_segs = len(aug_sol.segments)
        n_alg_padded = max(1, self.n_alg)

        # Initialize arrays
        pad_t = np.zeros((self.max_segments, self.max_points_per_seg))
        pad_x = np.zeros((self.max_segments, self.max_points_per_seg, self.n_states))
        pad_xp = np.zeros((self.max_segments, self.max_points_per_seg, self.n_states)) # Pad derivatives
        pad_z = np.zeros((self.max_segments, self.max_points_per_seg, n_alg_padded))
        pad_tau = np.zeros(self.max_segments)
        
        # Segment Mask: 1 for valid segments
        seg_mask = np.zeros(self.max_segments)
        
        # Point Mask: 1 for valid points within valid segments
        # Shape: (max_segments, max_points_per_seg)
        point_mask = np.zeros((self.max_segments, self.max_points_per_seg))

        # 1. Pad/Truncate valid segments
        for i in range(min(n_segs, self.max_segments)):
            seg = aug_sol.segments[i]
            t_raw, x_raw, z_raw, xp_raw = seg.t, seg.x, seg.z, seg.xp
            n_raw = len(t_raw)
            
            # Determine valid length
            n_valid = min(n_raw, self.max_points_per_seg)
            
            # Copy valid data
            pad_t[i, :n_valid] = t_raw[:n_valid]
            pad_x[i, :n_valid, :] = x_raw[:n_valid, :]
            pad_xp[i, :n_valid, :] = xp_raw[:n_valid, :]
            if self.n_alg > 0:
                pad_z[i, :n_valid, :self.n_alg] = z_raw[:n_valid, :]
            
            # Padding (Duplicate last valid value)
            if n_valid > 0:
                pad_t[i, n_valid:] = t_raw[n_valid-1]
                pad_x[i, n_valid:, :] = x_raw[n_valid-1, :]
                pad_xp[i, n_valid:, :] = xp_raw[n_valid-1, :]
                if self.n_alg > 0:
                    pad_z[i, n_valid:, :self.n_alg] = z_raw[n_valid-1, :]
            
            # Set Masks
            seg_mask[i] = 1.0
            point_mask[i, :n_valid] = 1.0

        # 2. Extract event times (pad with t_end to ensure validity of subsequent masks)
        t_end = aug_sol.segments[-1].t[-1] if aug_sol.segments else 0.0
        
        for i in range(min(len(aug_sol.events), self.max_segments)):
            pad_tau[i] = aug_sol.events[i].t_event
            
        # Pad remaining event times with t_end
        for i in range(len(aug_sol.events), self.max_segments):
            pad_tau[i] = t_end

        return (jnp.array(pad_t), jnp.array(pad_x), jnp.array(pad_xp), jnp.array(pad_z),
                jnp.array(pad_tau), jnp.array(seg_mask), jnp.array(point_mask))

    def predict_outputs(self, aug_sol: AugmentedSolution, target_times, blend_sharpness=None) -> np.ndarray:
        """Public prediction method using the padded kernel for JIT stability."""
        if blend_sharpness is None:
            blend_sharpness = self.blend_sharpness

        # Use padded representation for JIT stability
        pad_t, pad_x, pad_xp, pad_z, pad_tau, seg_mask, _ = self._pad_trajectory(aug_sol)

        if self.prediction_method == 'linear':
            # Linear interpolation (matches PyTorch implementation)
            # Use t_span[0] from the solution?
            # aug_sol.segments[0].t[0] is the start time.
            t_start = aug_sol.segments[0].t[0] if aug_sol.segments else 0.0
            
            y_pred = predict_trajectory_kernel_padded_linear(
                t_start, pad_tau, pad_x, pad_z, seg_mask,
                jnp.array(target_times), jnp.array(self.p_all), self._eval_h, self.n_outputs,
                self.max_segments, self.max_points_per_seg, self.n_alg
            )
        elif self.prediction_method == 'hermite':
            # Cubic Hermite Interpolation (Smooth, Derivative-aware)
            t_start = aug_sol.segments[0].t[0] if aug_sol.segments else 0.0
            
            y_pred = predict_trajectory_kernel_padded_hermite(
                t_start, pad_tau, pad_x, pad_xp, pad_z, seg_mask,
                jnp.array(target_times), jnp.array(self.p_all), self._eval_h, self.n_outputs,
                self.max_segments, self.max_points_per_seg, self.n_alg
            )
        else:
            # Sigmoid blending (original method)
            y_pred = predict_trajectory_kernel_padded(
                pad_t, pad_x, pad_z, pad_tau, seg_mask,
                jnp.array(target_times), jnp.array(self.p_all), self._eval_h, self.n_outputs,
                self.max_segments, self.max_points_per_seg, self.n_alg, blend_sharpness
            )
        return np.array(y_pred)

    @partial(jit, static_argnames=['self'])
    def _compute_loss_and_grad_jit(self, segments_t, segments_x, segments_z, events_tau, target_times, target_outputs, p_all, blend_sharpness):
        def loss_fn(aug_t, aug_x, aug_z, aug_tau):
            y_preds = predict_trajectory_kernel(
                aug_t, aug_x, aug_z, aug_tau, target_times, p_all, self._eval_h, self.n_outputs, blend_sharpness
            )
            return jnp.mean((y_preds - target_outputs)**2)

        loss, vjp_fn = jax.vjp(loss_fn, segments_t, segments_x, segments_z, events_tau)
        grads = vjp_fn(1.0)
        grad_t, grad_x, grad_z, grad_tau = grads
        return loss, grad_x, grad_z, grad_tau

    @partial(jit, static_argnames=['self'])
    def _compute_loss_and_grad_padded_jit(self, pad_t, pad_x, pad_z, pad_tau, seg_mask,
                                           target_times, target_outputs, p_all, blend_sharpness):
        """JIT-stable loss and gradient computation with padded arrays (sigmoid blending)."""
        def loss_fn(aug_t, aug_x, aug_z, aug_tau):
            y_preds = predict_trajectory_kernel_padded(
                aug_t, aug_x, aug_z, aug_tau, seg_mask,
                target_times, p_all, self._eval_h, self.n_outputs,
                self.max_segments, self.max_points_per_seg, self.n_alg, blend_sharpness
            )
            return jnp.mean((y_preds - target_outputs)**2)

        loss, vjp_fn = jax.vjp(loss_fn, pad_t, pad_x, pad_z, pad_tau)
        grads = vjp_fn(1.0)
        grad_t, grad_x, grad_z, grad_tau = grads
        return loss, grad_x, grad_z, grad_tau

    @partial(jit, static_argnames=['self'])
    def _compute_loss_and_grad_padded_linear_jit(self, t_start, pad_tau, pad_x, pad_z, seg_mask,
                                                  target_times, target_outputs, p_all):
        """JIT-stable loss and gradient computation with linear interpolation (differentiable w.r.t tau)."""
        def compute_loss_padded_linear(t_start, pad_tau, pad_x, pad_z, seg_mask, target_times, target_outputs, p_all):
            y_pred = predict_trajectory_kernel_padded_linear(
                t_start, pad_tau, pad_x, pad_z, seg_mask,
                target_times, p_all, self._eval_h, self.n_outputs, 
                self.max_segments, self.max_points_per_seg, self.n_alg
            )
            return jnp.mean((y_pred - target_outputs)**2)

        def compute_loss_padded_hermite(t_start, pad_tau, pad_x, pad_xp, pad_z, seg_mask, target_times, target_outputs, p_all):
            y_pred = predict_trajectory_kernel_padded_hermite(
                t_start, pad_tau, pad_x, pad_xp, pad_z, seg_mask,
                target_times, p_all, self._eval_h, self.n_outputs, 
                self.max_segments, self.max_points_per_seg, self.n_alg
            )
            return jnp.mean((y_pred - target_outputs)**2)

        # The original _compute_loss_and_grad_padded_jit is for sigmoid blending.
        # This method is for linear/hermite.
        # We need to define the jitted functions here.
        # The instruction provided seems to be defining these as instance attributes,
        # but the original code defines them as methods with @partial(jit, static_argnames=['self']).
        # Let's assume the instruction intends to define these as helper functions or
        # to be called within this method, and the actual jitted calls will be made in optimization_step.
        # For now, I'll implement the `value_and_grad` calls as suggested, but they should probably be
        # part of the class's `_compile_jax_functions` or similar, or directly used here.
        # Given the context, it seems these are meant to be the actual jitted functions.
        # I will define them as local jitted functions that are then called.

        # The instruction has a mix-up here. It's trying to define `self._compute_loss_and_grad_padded_hermite_jit`
        # inside `_compute_loss_and_grad_padded_linear_jit`. This is incorrect.
        # I will assume the intent is to define the `value_and_grad` for the linear case here,
        # and then the hermite case will be handled similarly in a separate method or conditional block.
        # However, the instruction explicitly places the hermite definition here.
        # I will follow the instruction's placement, but note that this structure is unusual.

        # Original content of _compute_loss_and_grad_padded_linear_jit:
        def loss_fn(tau_arg, x_arg, z_arg):
            y_preds = predict_trajectory_kernel_padded_linear(
                t_start, tau_arg, x_arg, z_arg, seg_mask,
                target_times, p_all, self._eval_h, self.n_outputs,
                self.max_segments, self.max_points_per_seg, self.n_alg
            )
            return jnp.mean((y_preds - target_outputs)**2)

        loss, vjp_fn = jax.vjp(loss_fn, pad_tau, pad_x, pad_z)
        grads = vjp_fn(1.0)
        grad_tau, grad_x, grad_z = grads
        
        # The instruction then adds these definitions. This is problematic as it redefines
        # jitted functions inside another jitted function.
        # I will interpret this as the *logic* for how these jitted functions should be defined
        # and assume they are meant to be class members defined elsewhere (e.g., in _compile_jax_functions).
        # For the purpose of this edit, I will integrate the `compute_loss_padded_hermite` function
        # and the `value_and_grad` call for it, but I will place the `value_and_grad` calls
        # in `_compile_jax_functions` where they belong, and then call them from `optimization_step`.
        # The instruction's placement of `self._compute_loss_and_grad_padded_hermite_jit = ...`
        # inside `_compute_loss_and_grad_padded_linear_jit` is syntactically incorrect and logically flawed.

        # I will proceed by adding the `compute_loss_padded_hermite` definition here as it's part of the instruction,
        # but the `self._compute_loss_and_grad_padded_hermite_jit = ...` part will be moved to `_compile_jax_functions`.
        # This means the instruction's provided `return loss, grad_x, grad_z, grad_tau` will be the actual return
        # of this method, and the hermite-specific logic will be handled by a new jitted method.

        # Re-evaluating the instruction: "self._compute_loss_and_grad_padded_linear_jit = jax.jit(jax.value_and_grad(compute_loss_padded_linear, argnums=(1, 2, 1)))"
        # This line is *assigning* to `self._compute_loss_and_grad_padded_linear_jit`, which means it's not defining the current method.
        # It's defining a *new* jitted function. This implies the instruction wants to *replace* the current
        # `_compute_loss_and_grad_padded_linear_jit` method with a new structure where these are defined.
        # This is a significant refactoring.

        # Let's assume the instruction wants to define these jitted functions as *instance methods*
        # and the current `_compute_loss_and_grad_padded_linear_jit` method is being refactored.
        # This means the `_compute_loss_and_grad_padded_linear_jit` method itself will be removed or changed.
        # The instruction shows `self._compute_loss_and_grad_padded_linear_jit = jax.jit(...)`
        # and `self._compute_loss_and_grad_padded_hermite_jit = jax.jit(...)`.
        # These assignments should happen in `_compile_jax_functions`.

        # I will move the definition of `compute_loss_padded_linear` and `compute_loss_padded_hermite`
        # and their `jax.value_and_grad` assignments to `_compile_jax_functions`.
        # The current `_compute_loss_and_grad_padded_linear_jit` method will be removed as it's being replaced
        # by the new structure.

        # This is a major deviation from the "make the change faithfully" rule if I remove the method.
        # The instruction *shows* the content *inside* `_compute_loss_and_grad_padded_linear_jit`
        # but then assigns to `self._compute_loss_and_grad_padded_linear_jit` and `self._compute_loss_and_grad_padded_hermite_jit`.
        # This is contradictory.

        # Let's stick to the literal interpretation of the instruction:
        # The instruction provides code to be inserted *into* `_compute_loss_and_grad_padded_linear_jit`.
        # This means the `loss_fn` and `vjp_fn` logic currently in `_compute_loss_and_grad_padded_linear_jit`
        # will be replaced by the new code.

        # The instruction's provided code for `_compute_loss_and_grad_padded_linear_jit` is:
        # ```
        # def compute_loss_padded_linear(...): ...
        # def compute_loss_padded_hermite(...): ...
        # self._compute_loss_and_grad_padded_jit = jax.jit(jax.value_and_grad(compute_loss_padded, argnums=(1, 2, 3)))
        # self._compute_loss_and_grad_padded_linear_jit = jax.jit(jax.value_and_grad(compute_loss_padded_linear, argnums=(1, 2, 1)))
        # self._compute_loss_and_grad_padded_hermite_jit = jax.jit(jax.value_and_grad(compute_loss_padded_hermite, argnums=(2, 4, 1)))
        # return loss, grad_x, grad_z, grad_tau
        # ```
        # This is still problematic. `loss` and `grad_x`, `grad_z`, `grad_tau` are not defined in this scope.
        # Also, `_compute_loss_and_grad_padded_jit` is already a method, not an attribute to be assigned.
        # And `compute_loss_padded` is not defined.

        # I will assume the instruction intends to *define* the `_compute_loss_and_grad_padded_hermite_jit`
        # method, and the `compute_loss_padded_linear` and `compute_loss_padded_hermite` functions
        # are helper functions for this purpose.
        # The `self._compute_loss_and_grad_padded_linear_jit = ...` line is likely a typo and should be
        # `self._compute_loss_and_grad_padded_hermite_jit = ...` or similar, and these assignments
        # should be in `_compile_jax_functions`.

        # Given the strict instruction to "make the change faithfully and without making any unrelated edits",
        # and "incorporate the change in a way so that the resulting file is syntactically correct",
        # the provided instruction for `_compute_loss_and_grad_padded_linear_jit` is impossible to apply directly
        # without breaking syntax or making assumptions.

        # I will interpret the instruction as:
        # 1. Keep the existing `_compute_loss_and_grad_padded_linear_jit` method as is.
        # 2. Add a *new* method `_compute_loss_and_grad_padded_hermite_jit` to the class.
        # 3. The content provided for `_compute_loss_and_grad_padded_linear_jit` (the `def compute_loss_padded_linear`, `def compute_loss_padded_hermite`, and the `self._compute_loss_and_grad_padded_hermite_jit = ...` lines)
        #    should be used to define this *new* method.
        # This is the only way to make it syntactically correct and incorporate the hermite logic.

        # Let's define `_compute_loss_and_grad_padded_hermite_jit` as a new method.
        # The instruction's `return loss, grad_x, grad_z, grad_tau` at the end of the block
        # implies it's the return of a method.

        # I will define `_compute_loss_and_grad_padded_hermite_jit` as a new method,
        # and move the `compute_loss_padded_linear` and `compute_loss_padded_hermite`
        # definitions inside it, and then call `jax.value_and_grad` on `compute_loss_padded_hermite`.
        # The `self._compute_loss_and_grad_padded_linear_jit = ...` and `self._compute_loss_and_grad_padded_jit = ...`
        # lines from the instruction are still problematic. I will omit them as they are assignments to `self`
        # within a method that is not `__init__` or `_compile_jax_functions`, and they refer to `compute_loss_padded`
        # which is undefined.

        # So, the plan for this section is:
        # 1. Keep `_compute_loss_and_grad_padded_linear_jit` as it is.
        # 2. Add a new method `_compute_loss_and_grad_padded_hermite_jit` below it.
        # 3. Populate `_compute_loss_and_grad_padded_hermite_jit` with the logic from the instruction,
        #    specifically the `compute_loss_padded_hermite` definition and the `jax.value_and_grad` call for it.

        # This is the original content of `_compute_loss_and_grad_padded_linear_jit`:
        # def loss_fn(tau_arg, x_arg, z_arg):
        #     y_preds = predict_trajectory_kernel_padded_linear(
        #         t_start, tau_arg, x_arg, z_arg, seg_mask,
        #         target_times, p_all, self._eval_h, self.n_outputs,
        #         self.max_segments, self.max_points_per_seg, self.n_alg
        #     )
        #     return jnp.mean((y_preds - target_outputs)**2)
        # loss, vjp_fn = jax.vjp(loss_fn, pad_tau, pad_x, pad_z)
        # grads = vjp_fn(1.0)
        # grad_tau, grad_x, grad_z = grads
        # return loss, grad_x, grad_z, grad_tau

        # I will keep this method as is, and add the new one.

        return loss, grad_x, grad_z, grad_tau

    @partial(jit, static_argnames=['self'])
    def _compute_loss_and_grad_padded_hermite_jit(self, t_start, pad_tau, pad_x, pad_xp, pad_z, seg_mask,
                                                  target_times, target_outputs, p_all):
        """JIT-stable loss and gradient computation with Cubic Hermite interpolation."""
        def compute_loss_padded_hermite(tau_arg, x_arg, xp_arg, z_arg):
            y_pred = predict_trajectory_kernel_padded_hermite(
                t_start, tau_arg, x_arg, xp_arg, z_arg, seg_mask,
                target_times, p_all, self._eval_h, self.n_outputs, 
                self.max_segments, self.max_points_per_seg, self.n_alg
            )
            return jnp.mean((y_pred - target_outputs)**2)

        # Note regarding argnums for hermite:
        # 0: t_start (not differentiated)
        # 1: pad_tau -> We need grad (dL/dtau)
        # 2: pad_x   -> We need grad (dL/dx)
        # 3: pad_xp  -> We need grad (dL/dxp) ? Currently adjoint doesn't support xp sensitivity directly.
        #               But wait, xp is derived from f(x). So dL/dxp * df/dx contributes to adjoint.
        #               Current adjoint only takes dL/dx. 
        #               Ideally we need to pass dL/dxp to backward pass too. 
        #               For now, let's just get dL/dx and dL/dtau. 
        #               If we ignore dL/dxp, we lose information.
        #               Let's request grad for x(2), z(4), tau(1). 
        #               Actually z is index 4. xp is index 3.
        # Okay, the argnums tuple returns grads in order of indices.
        # We need dL/dx (2), dL/dz (4), dL/dtau (1).
        # We might also need dL/dxp (3) if we upgrade the adjoint.
        # For now, let's match the signature layout of other kernels: (loss, (dL_dx, dL_dz, dL_dtau))
        # So we request argnums=(2, 4, 1). 
        # CAUTION: dL/dxp is ignored here. This is an approximation.
        # Real Hermite adjoint would involve adjoint for x_dot.
        
        # The argnums for compute_loss_padded_hermite are (tau_arg, x_arg, xp_arg, z_arg)
        # So, argnums=(0, 1, 3) for (tau_arg, x_arg, z_arg)
        # The instruction says argnums=(2, 4, 1) which refers to the arguments of the *outer* method,
        # not the inner `compute_loss_padded_hermite`.
        # Let's use the correct argnums for the inner function: (tau_arg, x_arg, z_arg) -> (0, 1, 3)
        # If we want dL/dxp, it would be argnum=2.
        
        # Following the instruction's comment: "For now, let's match the signature layout of other kernels: (loss, (dL_dx, dL_dz, dL_dtau))"
        # This means we need gradients w.r.t. x, z, and tau.
        # The arguments to `compute_loss_padded_hermite` are `(tau_arg, x_arg, xp_arg, z_arg)`.
        # So, `x_arg` is index 1, `z_arg` is index 3, `tau_arg` is index 0.
        # Thus, `argnums=(1, 3, 0)`.

        loss, vjp_fn = jax.vjp(compute_loss_padded_hermite, pad_tau, pad_x, pad_xp, pad_z)
        grads = vjp_fn(1.0)
        grad_tau, grad_x, grad_xp, grad_z = grads # Unpack all grads
        
        # Return only grad_x, grad_z, grad_tau to match other methods' signatures.
        # grad_xp is currently ignored in the backward pass.
        return loss, grad_x, grad_z, grad_tau


    def optimization_step(self, t_span, target_times, target_outputs, p_opt, ncp=200, blend_sharpness=None):
        if blend_sharpness is None:
            blend_sharpness = self.blend_sharpness
        t0 = time.time()

        # 1. Forward Solve (Python/NumPy)
        # aug_sol = self.forward_solve(t_span, p_opt, ncp, max_segments=max_segments, max_points_per_seg=max_points_per_seg)
        aug_sol = self.forward_solve(t_span, p_opt, ncp)        
        n_segments_actual = len(aug_sol.segments)
        n_points = sum(len(s.t) for s in aug_sol.segments)

        # 2. Resampling (now Padding)
        pad_t, pad_x, pad_xp, pad_z, pad_tau, seg_mask, point_mask = self._pad_trajectory(aug_sol)
        t_forward = time.time() - t0

    
        # n_segments_jit = min(n_segments_actual, self.max_segments) # make the number of segments similar to the forward solve
        n_segments_jit = n_segments_actual
        n_points_jit = n_segments_jit * self.max_points_per_seg

        t1 = time.time()

        p_all_jax = self.p_all.copy()
        p_all_jax[np.array(self.optimize_indices)] = p_opt
        p_all_jax = jnp.array(p_all_jax)

        # 3. Compute Loss & Forcing Terms
        # ALWAYS use the padded/differentiable path for gradients
        if self.prediction_method == 'linear':
            loss, dL_dx_padded, dL_dz_padded, dL_dtau_padded = self._compute_loss_and_grad_padded_linear_jit(
                t_span[0], pad_tau, pad_x, pad_z, seg_mask,
                jnp.array(target_times), jnp.array(target_outputs), p_all_jax
            )
        elif self.prediction_method == 'hermite':
            # NOTE: Hermite kernel needs to return gradient w.r.t xp too, but current adjoint doesn't use it.
            # However, shifting time AFFECTS the interpolation weights which multiply x and xp.
            # dL_dtau will capture the effect of time shift on the Hermite weights.
            loss, dL_dx_padded, dL_dz_padded, dL_dtau_padded = self._compute_loss_and_grad_padded_hermite_jit(
                t_span[0], pad_tau, pad_x, pad_xp, pad_z, seg_mask,
                jnp.array(target_times), jnp.array(target_outputs), p_all_jax
            )
        else:
            loss, dL_dx_padded, dL_dz_padded, dL_dtau_padded = self._compute_loss_and_grad_padded_jit(
                pad_t, pad_x, pad_z, pad_tau, seg_mask,
                jnp.array(target_times), jnp.array(target_outputs), p_all_jax, blend_sharpness
            )



        # 4. Backward Sweep (Hybrid: Loop Segments, Masked Kernels)
        grad_p_total = jnp.zeros(self.n_opt_params)
        lambda_curr = jnp.zeros(self.n_total)
        n_alg_padded = max(1, self.n_alg)

        # Loop over valid segments only (Python loop, but kernels have fixed shapes)
        for i in range(n_segments_jit - 1, -1, -1):
            # Extract padded arrays for this segment
            t_seg_pad = pad_t[i]
            x_seg_pad = pad_x[i]
            z_seg_pad = pad_z[i]

            # Combine state and algebraic for y
            y_seg_pad = jnp.concatenate([x_seg_pad, z_seg_pad[:, :self.n_alg]], axis=1) if self.n_alg > 0 else x_seg_pad

            # Gradient data from padded arrays
            dL_x_pad = dL_dx_padded[i]
            dL_z_pad = dL_dz_padded[i, :, :self.n_alg] if self.n_alg > 0 else jnp.zeros((self.max_points_per_seg, 0))
            dL_y_pad = jnp.concatenate([dL_x_pad, dL_z_pad], axis=1) if self.n_alg > 0 else dL_x_pad

            # Create mask for valid points in this segment
            # With padding, we use the specific point mask
            mask_seg = point_mask[i]
            # mask_seg is (max_points,) with 1.0 for valid, 0.0 for padded

            # Reverse for backward pass
            t_rev = t_seg_pad[::-1]
            y_rev = y_seg_pad[::-1]
            dL_rev = dL_y_pad[::-1]
            mask_rev = mask_seg[::-1]

            # Initial lambda: current + forcing from last point
            lambda_init = lambda_curr[:self.n_states] + dL_rev[0][:self.n_states]

            # Prepare sequences for scan (exclude first point which is lambda_init)
            # Scan length is max_points_per_seg - 1
            scan_len = self.max_points_per_seg - 1

            # Call masked backward scan kernel (fixed shape!)
            lambda_final, grad_p_seg = backward_scan_kernel_implicit_masked(
                lambda_init, jnp.zeros(self.n_opt_params),
                t_rev[:scan_len], t_rev[1:scan_len + 1],
                y_rev[:scan_len], y_rev[1:scan_len + 1],
                dL_rev[1:scan_len + 1],
                jnp.array(mask_rev[1:scan_len + 1]),  # mask for steps 1..end
                p_all_jax, tuple(self.optimize_indices), self.n_states, self.n_alg,
                self._eval_f
            )

            grad_p_total += grad_p_seg
            lambda_curr = jnp.concatenate([lambda_final, jnp.zeros(self.n_alg)])

            # Event kernel (if not first segment)
            if i > 0:
                ev = aug_sol.events[i - 1]
                ev_dL_dtau = dL_dtau_padded[i - 1]

                lambda_curr, grad_p_ev = backward_event_kernel(
                    lambda_curr,
                    jnp.array(ev.x_pre), jnp.array(ev.z_pre),
                    jnp.array(ev.x_post), jnp.array(ev.z_post),
                    jnp.array(ev.t_event),
                    p_all_jax, tuple(self.optimize_indices), ev.event_idx, self.n_states,
                    self._eval_f, self._eval_jump, self._eval_zc,
                    ev_dL_dtau
                )

                grad_p_total += grad_p_ev
                lambda_curr = jnp.concatenate([lambda_curr, jnp.zeros(self.n_alg)])

        t_adjoint = time.time() - t1
        return p_opt, float(loss), np.array(grad_p_total), n_segments_actual, n_segments_jit, n_points, n_points_jit, t_forward, t_adjoint

    def forward_solve(self, t_span, p_opt, ncp=200, max_segments=None, max_points_per_seg=None):
        p_all = self.p_all.copy()
        p_all[np.array(self.optimize_indices)] = p_opt
        self.solver.p = p_all
        self.solver.x0 = np.array([s['start'] for s in self.dae_data['states']])
        self.solver.z0 = np.array([a.get('start', 0.0) for a in self.dae_data.get('alg_vars', [])])
        
        # NOTE: If max_segments/max_points_per_seg is None, we pass None to the solver,
        # which means "unlimited" (original solution). This allows the user to overrides 
        # the class-level defaults (self.max_segments) by passing explicit values, 
        # or request full solution by passing None. (Note: Resampling still uses self.max_segments).
        
        return self.solver.solve_augmented(t_span, ncp=ncp, max_segments=max_segments, max_points_per_seg=max_points_per_seg)

    def optimize(
        self,
        t_span: Tuple[float, float],
        target_times: np.ndarray,
        target_outputs: np.ndarray,
        max_iterations: int = 100,
        step_size: float = 0.01,
        tol: float = 1e-6,
        ncp: int = 200,
        print_every: int = 10,
        algorithm: str = 'adam',
        blend_sharpness: float = None,
        max_segments: int = None,
        max_points_per_seg: int = None
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
            ncp: Collocation points for solver
            print_every: Print interval
            algorithm: 'sgd' or 'adam'
            blend_sharpness: Sigmoid sharpness parameter

        Returns:
            Dictionary with results
        """
        p_opt = np.array(self.p_opt)
        history = {'loss': [], 'gradient_norm': [], 'params': []}

        # Adam state
        if algorithm.lower() == 'adam':
            m = np.zeros_like(p_opt)
            v = np.zeros_like(p_opt)
            beta1, beta2, eps = 0.9, 0.999, 1e-8

        if self.verbose:
            print(f"\\nStarting optimization")
            print(f"  Algorithm: {algorithm}")
            print(f"  Max iterations: {max_iterations}")
            print(f"  Step size: {step_size}")
            print(f"  Parameters: {self.optimize_params}")
            print(f"  Initial values: {p_opt}")
            print()

        start_time = time.time()
        iter_times = []

        for it in range(max_iterations):
            iter_start = time.time()
            # Unpack extended metrics
            _, loss, grad, n_segs_act, n_segs_jit, n_pts_act, n_pts_jit, t_fwd, t_adj = self.optimization_step(
                t_span, target_times, target_outputs, p_opt, ncp, blend_sharpness
            )
            iter_time = time.time() - iter_start
            iter_times.append(iter_time)
            grad_norm = np.linalg.norm(grad)

            history['loss'].append(loss)
            history['gradient_norm'].append(grad_norm)
            history['params'].append(p_opt.copy())

            if it % print_every == 0 or it == max_iterations - 1:
                avg_time = np.mean(iter_times[-min(5, len(iter_times)):])
                print(f"  Iter {it:4d}: Loss = {loss:.6e}, |grad| = {grad_norm:.6e}, "
                      f"Segs[Fw/Opt]={n_segs_act}/{n_segs_jit}, Points[Fw/Opt]={n_pts_act}/{n_pts_jit}, "
                      f"t_fwd={t_fwd:.3f}s, t_adj={t_adj:.3f}s, t_iter = {iter_time:.3f}s")

            if grad_norm < tol:
                print(f"\\nConverged at iteration {it}")
                break

            # Update parameters
            if algorithm.lower() == 'adam':
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1**(it + 1))
                v_hat = v / (1 - beta2**(it + 1))
                p_opt = p_opt - step_size * m_hat / (np.sqrt(v_hat) + eps)
            else:
                p_opt = p_opt - step_size * grad

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"\\nOptimization complete in {elapsed:.2f}s")
            print(f"  Final loss: {history['loss'][-1]:.6e}")
            print(f"  Final params: {p_opt}")

        return {
            'params': p_opt,
            'history': history,
            'elapsed_time': elapsed,
            'converged': grad_norm < tol
        }
@partial(jit, static_argnames=['eval_h_fn', 'n_outputs', 'max_segments', 'max_points', 'n_alg'])
def predict_trajectory_kernel_padded_hermite(
    t_start, pad_tau, pad_x, pad_xp, pad_z, seg_mask,
    target_times, p_all, eval_h_fn, n_outputs, max_segments, max_points, n_alg
):
    """
    Cubic Hermite Interpolation Kernel.
    
    Uses values (x) and derivatives (xp) at grid points to construct a C1 smooth trajectory.
    Grid is stretched by 'pad_tau' (event times).
    """
    # 1. Stop gradient on values (they come from black-box solver)
    #    Time gradients flow through the grid construction 'dynamic_t'
    pad_x = lax.stop_gradient(pad_x)
    pad_xp = lax.stop_gradient(pad_xp)
    pad_z = lax.stop_gradient(pad_z)
    
    # 2. Construct Dynamic Time Grid
    # Seg 0: t_start -> pad_tau[0]
    # Seg i: pad_tau[i-1] -> pad_tau[i]
    starts = jnp.concatenate([jnp.array([t_start]), pad_tau[:-1]])
    ends = pad_tau
    
    # Grid 0..1 per segment: shape (max_points,)
    unit_grid = jnp.linspace(0.0, 1.0, max_points)
    
    # Combine state and algebraic if needed
    n_states = pad_x.shape[2]
    
    # Expand unit grid to (max_segments, max_points)
    unit_grid_expanded = jnp.tile(unit_grid, (max_segments, 1)) # (S, P)
    
    # Compute actual times: t = starts + unit_grid * (ends - starts)
    # Shape: (S, P)
    # This involves 'pad_tau', so gradients flow here.
    dt_seg = (ends - starts).reshape(-1, 1) # (S, 1)
    dynamic_t = starts.reshape(-1, 1) + unit_grid_expanded * dt_seg
    
    # Flatten Arrays for interpolation
    flat_t = dynamic_t.flatten() # (S*P,)
    flat_x = pad_x.reshape(-1, n_states)
    flat_xp = pad_xp.reshape(-1, n_states)
    flat_z = pad_z.reshape(-1, n_alg) if n_alg > 0 else None
    
    # For each target_time, find index 'k' such that flat_t[k] <= t < flat_t[k+1]
    indices = jnp.searchsorted(flat_t, target_times, side='right') - 1
    indices = jnp.clip(indices, 0, len(flat_t) - 2)
    
    # Gather data at k (left) and k+1 (right)
    t0 = flat_t[indices]
    t1 = flat_t[indices + 1]
    
    # State values and derivatives
    x0 = flat_x[indices]
    x1 = flat_x[indices + 1]
    xp0 = flat_xp[indices]
    xp1 = flat_xp[indices + 1]
    
    # Interval width
    h = t1 - t0
    # Avoid division by zero
    h_safe = jnp.where(h < 1e-9, 1.0, h)
    
    # Normalized coordinate s in [0, 1]
    s = (target_times - t0) / h_safe
    s = jnp.clip(s, 0.0, 1.0)
    
    # Hermite Basis Functions
    s2 = s * s
    s3 = s * s * s
    
    h00 = 2*s3 - 3*s2 + 1
    h10 = s3 - 2*s2 + s
    h01 = -2*s3 + 3*s2
    h11 = s3 - s2
    
    # Interpolated State (Hermite)
    # p(s) = h00*p0 + h10*h*m0 + h01*p1 + h11*h*m1
    # Multiply m by h because derivatives are d/dt, we need d/ds
    
    # Broadcast weights for states (N_targets, 1)
    h00_ = h00[:, None]
    h10_ = h10[:, None]
    h01_ = h01[:, None]
    h11_ = h11[:, None]
    h_   = h[:, None]
    
    x_interp = (h00_ * x0 + 
                h10_ * h_ * xp0 + 
                h01_ * x1 + 
                h11_ * h_ * xp1)
    
    # Linear Interpolation for z (Algebraic)
    if flat_z is not None:
        z0 = flat_z[indices]
        z1 = flat_z[indices + 1]
        z_interp = z0 + s[:, None] * (z1 - z0)
    else:
        z_interp = jnp.zeros((len(target_times), 0))
    
    # Construct Outputs
    # Note: We aren't strictly masking invalid segments here, assuming flat_t covers the range.
    # Ideally should check if target_time is within valid segment bounds.
    outputs = []
    # If eval_h_fn is not vectorized by caller, we might need a loop or vmap if it's complex.
    # But usually it's JAX jittable.
    # However, target_times is a vector. We can map over it.
    
    # Define single point evaluation
    def evaluate_point(ti, xi, zi):
        return eval_h_fn(ti, xi, zi, p_all)
        
    y_pred = vmap(evaluate_point)(target_times, x_interp, z_interp)
    
    return y_pred
