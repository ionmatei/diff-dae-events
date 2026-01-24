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
def backward_scan_kernel_implicit(
    lambda_init, grad_p_init, 
    t_kp1_seq, t_k_seq, y_kp1_seq, y_k_seq, dL_k_seq,
    p_all_default, optimize_indices, n_states, n_alg, f_fn
):
    p_opt_initial = p_all_default[jnp.array(optimize_indices)]

    def scan_step(carry, inputs):
        lambda_kp1, grad_p_acc = carry
        t_kp1, t_k, y_kp1, y_k, dL_k = inputs
        h = t_kp1 - t_k
        
        x_k = y_k[:n_states]
        x_kp1 = y_kp1[:n_states]
        z_k = y_k[n_states:] if n_alg > 0 else jnp.array([])
        
        # 1. Compute Jacobians df/dx (State)
        def f_wrapper_x(x, t_val): return f_fn(t_val, x, z_k, p_all_default)
        J_k = jax.jacfwd(f_wrapper_x)(x_k, t_k)
        J_kp1 = jax.jacfwd(f_wrapper_x)(x_kp1, t_kp1)
        
        # 2. Implicit Adjoint Solve
        # We need to compute: lambda_mid = (I - h/2 * J_{k+1}^T)^(-1) * lambda_{k+1}
        # This projects the adjoint through the implicit part of the trapezoidal step.
        
        I = jnp.eye(n_states)
        factor = 0.5 * h
        
        # Matrix A = (I - h/2 * J_{k+1})^T = I - h/2 * J_{k+1}^T
        A_matrix = I - factor * J_kp1.T
        
        # Solve A * lambda_mid = lambda_kp1
        # For small h, this is well-conditioned. For stiff systems, this is crucial.
        lambda_mid = jnp.linalg.solve(A_matrix, lambda_kp1)
        
        # 3. Update Adjoint to current step
        # lambda_k = (I + h/2 * J_k)^T * lambda_mid + forcing
        lambda_k = (I + factor * J_k.T) @ lambda_mid + dL_k[:n_states]

        # 4. Parameter Gradient (dL/dp)
        # CRITICAL FIX: Use lambda_mid, NOT lambda_kp1. 
        # The parameter sensitivity passes through the same implicit inversion.
        def f_wrapper_p(p_subset):
            p_full = p_all_default.at[jnp.array(optimize_indices)].set(p_subset)
            f_k = f_fn(t_k, x_k, z_k, p_full)
            f_kp1 = f_fn(t_kp1, x_kp1, z_k, p_full)
            # Residual sensitivity dR/dp = -h/2 * (df_k/dp + df_kp1/dp)
            return -(h / 2.0) * (f_k + f_kp1)

        dr_dp = jax.jacfwd(f_wrapper_p)(p_opt_initial)
        
        # Accumulate: grad += lambda_mid^T @ (dR/dp)
        # Note the sign: Adjoint rule typically is -lambda^T * R_p
        # Since dr_dp above includes the negative sign from the residual definition,
        # we subtract: grad - (lambda @ dr_dp).
        # Wait, dr_dp above is defined as negative. 
        # Residual: x_new - x - step = 0. dR/dp = -dstep/dp.
        # Gradient: - lambda^T * dR/dp = - lambda^T * (-dstep/dp) = lambda^T * dstep/dp.
        # My dr_dp function returns -dstep/dp. 
        # So we subtract: grad_acc - (lambda_mid @ dr_dp)
        
        grad_p_new = grad_p_acc - (dr_dp.T @ lambda_mid)

        return (lambda_k, grad_p_new), lambda_k

    (lambda_final, grad_p_final), _ = lax.scan(
        scan_step,
        (lambda_init, grad_p_init),
        (t_kp1_seq, t_k_seq, y_kp1_seq, y_k_seq, dL_k_seq)
    )
    return lambda_final, grad_p_final


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

class DAEOptimizerExplicitAdjoint:
    def __init__(self, dae_data: Dict, optimize_params: List[str], solver=None, verbose: bool = True,
                 blend_sharpness: float = 100.0, max_segments: int = 20, max_points_per_seg: int = 500):
        self.dae_data = dae_data
        self.optimize_params = optimize_params
        self.verbose = verbose
        self.blend_sharpness = blend_sharpness
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
        self.max_points_per_seg = max_points_per_seg

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
        Pad trajectory data to fixed shapes for JIT stability.

        Returns:
            pad_t: (max_segments, max_points_per_seg) - time points
            pad_x: (max_segments, max_points_per_seg, n_states) - state values
            pad_z: (max_segments, max_points_per_seg, n_alg_padded) - algebraic values
            pad_tau: (max_segments,) - event times
            seg_mask: (max_segments,) - 1 if segment is valid, 0 otherwise
            seg_lengths: (max_segments,) - actual length of each segment
        """
        n_segs = len(aug_sol.segments)
        n_alg_padded = max(1, self.n_alg)  # At least 1 to avoid shape issues

        # Initialize padded arrays
        pad_t = np.zeros((self.max_segments, self.max_points_per_seg))
        pad_x = np.zeros((self.max_segments, self.max_points_per_seg, self.n_states))
        pad_z = np.zeros((self.max_segments, self.max_points_per_seg, n_alg_padded))

        seg_mask = np.zeros(self.max_segments)
        seg_lengths = np.zeros(self.max_segments, dtype=np.int32)
        pad_tau = np.zeros(self.max_segments)

        # Fill valid segments
        for i in range(min(n_segs, self.max_segments)):
            seg = aug_sol.segments[i]
            n_pts = min(len(seg.t), self.max_points_per_seg)

            # Copy valid data
            pad_t[i, :n_pts] = seg.t[:n_pts]
            pad_x[i, :n_pts, :] = seg.x[:n_pts, :]
            if self.n_alg > 0:
                pad_z[i, :n_pts, :] = seg.z[:n_pts, :]

            # Pad remaining points with last valid value (avoids interpolation issues)
            if n_pts < self.max_points_per_seg:
                pad_t[i, n_pts:] = seg.t[n_pts - 1]
                pad_x[i, n_pts:, :] = seg.x[n_pts - 1, :]
                if self.n_alg > 0:
                    pad_z[i, n_pts:, :] = seg.z[n_pts - 1, :]

            seg_mask[i] = 1.0
            seg_lengths[i] = n_pts

        # Extract event times
        for i in range(min(len(aug_sol.events), self.max_segments)):
            pad_tau[i] = aug_sol.events[i].t_event

        # Pad remaining event times with last valid or final time
        if len(aug_sol.events) > 0:
            last_tau = aug_sol.events[-1].t_event
            for i in range(len(aug_sol.events), self.max_segments):
                pad_tau[i] = last_tau

        return (jnp.array(pad_t), jnp.array(pad_x), jnp.array(pad_z),
                jnp.array(pad_tau), jnp.array(seg_mask), seg_lengths)

    def predict_outputs(self, aug_sol: AugmentedSolution, target_times, blend_sharpness=None) -> np.ndarray:
        """Public prediction method using the padded kernel for JIT stability."""
        if blend_sharpness is None:
            blend_sharpness = self.blend_sharpness

        # Use padded representation for JIT stability
        pad_t, pad_x, pad_z, pad_tau, seg_mask, _ = self._pad_trajectory(aug_sol)

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
        """JIT-stable loss and gradient computation with padded arrays."""
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

    def optimization_step(self, t_span, target_times, target_outputs, p_opt, ncp=200, blend_sharpness=None):
        if blend_sharpness is None:
            blend_sharpness = self.blend_sharpness
        t0 = time.time()

        # 1. Forward Solve (Python/NumPy)
        aug_sol = self.forward_solve(t_span, p_opt, ncp)
        t_forward = time.time() - t0

        n_segments_actual = len(aug_sol.segments)
        n_points = sum(len(s.t) for s in aug_sol.segments)

        # Check for truncation and warn
        if n_segments_actual > self.max_segments:
            print(f"Warning: Simulation has {n_segments_actual} segments but max_segments={self.max_segments}. "
                  f"Increase max_segments in config.")
        max_pts_in_seg = max(len(s.t) for s in aug_sol.segments)
        if max_pts_in_seg > self.max_points_per_seg:
            print(f"Warning: Segment has {max_pts_in_seg} points but max_points_per_segment={self.max_points_per_seg}. "
                  f"Increase max_points_per_segment in config.")

        # Clamp to max values for processing
        n_segments = min(n_segments_actual, self.max_segments)

        t1 = time.time()

        # 2. Data Transfer with Padding for JIT stability
        pad_t, pad_x, pad_z, pad_tau, seg_mask, seg_lengths = self._pad_trajectory(aug_sol)

        p_all_jax = self.p_all.copy()
        p_all_jax[np.array(self.optimize_indices)] = p_opt
        p_all_jax = jnp.array(p_all_jax)

        # 3. Compute Loss & Forcing Terms (JIT Compiled with padding)
        loss, dL_dx_padded, dL_dz_padded, dL_dtau_padded = self._compute_loss_and_grad_padded_jit(
            pad_t, pad_x, pad_z, pad_tau, seg_mask,
            jnp.array(target_times), jnp.array(target_outputs), p_all_jax, blend_sharpness
        )

        # 4. Backward Sweep (Hybrid: Loop Segments, Masked Kernels)
        grad_p_total = jnp.zeros(self.n_opt_params)
        lambda_curr = jnp.zeros(self.n_total)
        n_alg_padded = max(1, self.n_alg)

        # Loop over valid segments only (Python loop, but kernels have fixed shapes)
        for i in range(n_segments - 1, -1, -1):
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
            # Valid points are 0..seg_lengths[i]-1, padded points are beyond
            valid_len = int(seg_lengths[i])
            point_mask = np.zeros(self.max_points_per_seg)
            point_mask[:valid_len] = 1.0

            # Reverse for backward pass
            t_rev = t_seg_pad[::-1]
            y_rev = y_seg_pad[::-1]
            dL_rev = dL_y_pad[::-1]
            mask_rev = point_mask[::-1]

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
        return p_opt, float(loss), np.array(grad_p_total), n_segments_actual, n_points, t_forward, t_adjoint

    def forward_solve(self, t_span, p_opt, ncp=200):
        p_all = self.p_all.copy()
        p_all[np.array(self.optimize_indices)] = p_opt
        self.solver.p = p_all
        self.solver.x0 = np.array([s['start'] for s in self.dae_data['states']])
        self.solver.z0 = np.array([a.get('start', 0.0) for a in self.dae_data.get('alg_vars', [])])
        return self.solver.solve_augmented(t_span, ncp=ncp)

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
        blend_sharpness: float = None
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
            _, loss, grad, n_segs, n_pts, t_fwd, t_adj = self.optimization_step(
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
                print(f"  Iter {it:4d}: Loss = {loss:.6e}, |grad| = {grad_norm:.6e}, Segs={n_segs}, Points={n_pts}, t_fwd={t_fwd:.3f}s, t_adj={t_adj:.3f}s, t_iter = {iter_time:.3f}s")

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
