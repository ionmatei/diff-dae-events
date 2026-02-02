"""
Explicit Discrete Adjoint Optimizer for DAEs with Events (Corrected).

Addresses analysis findings:
1. Implicit Jacobian term (dz/dx) for coupled DAEs.
2. Time-dependent reset map sensitivity in Gamma.
3. Consistency adjoint (mu_cons) at events.
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

# Import DAESolver from the package
# Assuming relative import works if placed in the same directory
from .dae_solver import DAESolver, AugmentedSolution

# =============================================================================
# 1. Equation Compiler (Reused)
# =============================================================================

def compile_equations_to_jax(eqn_strings, state_names, alg_names, param_names, extra_args=None):
    """Parses string equations into a single JAX-jittable function."""
    if not eqn_strings:
        # Return zeros matching the expected output logic? 
        # For f: returns [n_states]. For g: returns [n_alg].
        # It's better if the caller handles empty lists by not calling or expecting empty.
        # But to be safe return empty array.
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
# 2. Standalone JIT Kernels (Corrected)
# =============================================================================

@partial(jit, static_argnames=['f_fn', 'g_fn', 'n_states', 'n_alg', 'optimize_indices'])
def backward_scan_kernel_implicit_corrected(
    lambda_init, grad_p_init,
    t_kp1_seq, t_k_seq, y_kp1_seq, y_k_seq, dL_k_seq, mask_seq,
    p_all_default, optimize_indices, n_states, n_alg, f_fn, g_fn
):
    """
    Corrected backward scan kernel.
    INCLUDES: Implicit Jacobian correction dz/dx = -(dg/dz)^-1 * (dg/dx).
    """
    p_opt_initial = p_all_default[jnp.array(optimize_indices)]

    def scan_step(carry, inputs):
        lambda_kp1, grad_p_acc = carry
        t_kp1, t_k, y_kp1, y_k, dL_k, mask = inputs
        h = t_kp1 - t_k

        x_k = y_k[:n_states]
        x_kp1 = y_kp1[:n_states]
        z_k = y_k[n_states:n_states + n_alg] if n_alg > 0 else jnp.array([])

        # --- FIX 1: Implicit Jacobian Correction ---
        # Jacobian df/dx_total = df/dx_partial + df/dz * dz/dx
        # Where dz/dx = -(dg/dz)^-1 * (dg/dx)
        
        # Helper to get all partials
        def f_full_wrapper(x, z, t, p): return f_fn(t, x, z, p)
        def g_full_wrapper(x, z, t, p): return g_fn(t, x, z, p)
        
        # WE JUST NEED TO CORRECT J_k and J_kp1 to be TOTAL derivatives.
        
        def compute_total_jacobian(x_curr, z_curr, t_curr):
            # df/dx, df/dz
            jac_f_x = jax.jacfwd(lambda x: f_fn(t_curr, x, z_curr, p_all_default))(x_curr)
            
            if n_alg > 0:
                jac_f_z = jax.jacfwd(lambda z: f_fn(t_curr, x_curr, z, p_all_default))(z_curr)
                
                # dg/dx, dg/dz
                jac_g_x = jax.jacfwd(lambda x: g_fn(t_curr, x, z_curr, p_all_default))(x_curr)
                jac_g_z = jax.jacfwd(lambda z: g_fn(t_curr, x_curr, z, p_all_default))(z_curr)
                
                # dz/dx = - (dg/dz)^-1 (dg/dx)
                # We want J_total^T v = (df/dx + df/dz dz/dx)^T v
                # = df/dx^T v + dz/dx^T df/dz^T v
                # = df/dx^T v - dg/dx^T (dg/dz^{-T} df/dz^T v)
                
                # Instead of inverting explicitly, we can construct the operator or matrix.
                # Since dimension is small, matrix is fine.
                # dz_dx = - solve(jac_g_z, jac_g_x)
                
                # Robust solve for singular dg/dz? 
                # DAE index-1 assumption implies dg/dz is non-singular.
                dz_dx = -jnp.linalg.solve(jac_g_z, jac_g_x)
                
                J_total = jac_f_x + jac_f_z @ dz_dx
                return J_total
            else:
                return jac_f_x

        J_k = compute_total_jacobian(x_k, z_k, t_k)
        
        # In padded code, z is stored per point.
        # But y_kp1 contains (x_kp1, z_kp1).
        # We should use z_kp1.
        z_kp1 = y_kp1[n_states:n_states+n_alg] if n_alg > 0 else jnp.array([])
        J_kp1_corr = compute_total_jacobian(x_kp1, z_kp1, t_kp1)
        
        # 2. Implicit Adjoint Solve
        I = jnp.eye(n_states)
        factor = 0.5 * h
        A_matrix = I - factor * J_kp1_corr.T

        # Safe solve
        lambda_mid = jnp.linalg.solve(A_matrix, lambda_kp1)

        # 3. Update Adjoint
        lambda_k_computed = (I + factor * J_k.T) @ lambda_mid + dL_k[:n_states]
        # Note: If loss depends on z, dL_k contains dL/dz.
        # The total derivative of loss wrt x is dL/dx + dL/dz * dz/dx.
        # The input dL_k_seq usually assumes partials. 
        # If we want total adjoint, we should add dL/dz contribution here too.
        if n_alg > 0:
            jac_g_x_k = jax.jacfwd(lambda x: g_fn(t_k, x, z_k, p_all_default))(x_k)
            jac_g_z_k = jax.jacfwd(lambda z: g_fn(t_k, x_k, z, p_all_default))(z_k)
            dz_dx_k = -jnp.linalg.solve(jac_g_z_k, jac_g_x_k)
            
            dL_dx_k = dL_k[:n_states]
            dL_dz_k = dL_k[n_states:n_states+n_alg]
            
            dL_total_x = dL_dx_k + dz_dx_k.T @ dL_dz_k
            
            # Recalculate lambda_k with total loss sensitivity
            lambda_k_computed = (I + factor * J_k.T) @ lambda_mid + dL_total_x
        
        # 4. Parameter Gradient
        def f_wrapper_p(p_subset):
            p_full = p_all_default.at[jnp.array(optimize_indices)].set(p_subset)
            f_k = f_fn(t_k, x_k, z_k, p_full)
            f_kp1 = f_fn(t_kp1, x_kp1, z_kp1, p_full) # Use z_kp1
            return -(h / 2.0) * (f_k + f_kp1)

        dr_dp = jax.jacfwd(f_wrapper_p)(p_opt_initial)
        grad_p_computed = grad_p_acc - (dr_dp.T @ lambda_mid)

        # Apply mask
        lambda_k = jnp.where(mask > 0.5, lambda_k_computed, lambda_kp1)
        grad_p_new = jnp.where(mask > 0.5, grad_p_computed, grad_p_acc)

        return (lambda_k, grad_p_new), lambda_k

    (lambda_final, grad_p_final), _ = lax.scan(
        scan_step,
        (lambda_init, grad_p_init),
        (t_kp1_seq, t_k_seq, y_kp1_seq, y_k_seq, dL_k_seq, mask_seq)
    )
    return lambda_final, grad_p_final


@partial(jit, static_argnames=['f_fn', 'g_fn', 'jump_fn', 'zc_fn', 'n_states', 'n_alg', 'event_idx', 'optimize_indices'])
def backward_event_kernel_corrected(
    lambda_post, x_pre, z_pre, x_post, z_post, tau, p_all, optimize_indices, 
    event_idx, n_states, n_alg, f_fn, g_fn, jump_fn, zc_fn, dL_dtau
):
    """
    Corrected event adjoint kernel.
    INCLUDES:
    1. Consistency Adjoint (mu_cons).
    2. Time-dependent reset map sensitivity in Gamma.
    """
    
    # Let's assume lambda_post has shape (n_states + n_alg).
    lambda_post_x = lambda_post[:n_states]
    lambda_post_z = lambda_post[n_states:n_states+n_alg] if n_alg > 0 else jnp.array([])
    
    p_opt = p_all[jnp.array(optimize_indices)]

    # --- FIX 3: Consistency Adjoint (mu_cons) ---
    
    dt_cons = 0.0
    lambda_x_post_consistent = lambda_post_x
    
    if n_alg > 0:
        jac_g_z = jax.jacfwd(lambda z: g_fn(tau, x_post, z, p_all))(z_post)
        jac_g_x = jax.jacfwd(lambda x: g_fn(tau, x, z_post, p_all))(x_post)
        jac_g_t = jax.jacfwd(lambda t: g_fn(t, x_post, z_post, p_all))(tau)
        
        # Solve for mu_cons
        # (dg/dz)^T mu_cons = - lambda_post_z
        mu_cons = -jnp.linalg.solve(jac_g_z.T, lambda_post_z)
        
        # Update state adjoint
        lambda_x_post_consistent = lambda_post_x + jac_g_x.T @ mu_cons
        
        # Time contribution
        dt_cons = jac_g_t.T @ mu_cons
    
    # --- FIX 2: Gamma Calculation with Time Dependent Reset ---
    
    def jump_x_wrapper(x): return jump_fn(x, z_pre, tau, p_all, event_idx)
    def jump_t_wrapper(t): return jump_fn(x_pre, z_pre, t, p_all, event_idx)
    
    J_jump_x = jax.jacfwd(jump_x_wrapper)(x_pre)
    J_jump_t = jax.jacfwd(jump_t_wrapper)(tau) # Shape (n_states,) since h returns x
    
    # lambda_tilde = (dh/dx)^T lambda_x_post_consistent
    lambda_tilde = J_jump_x.T @ lambda_x_post_consistent
    
    # Coupling term: lambda_x_post_consistent @ dh/dt
    term_jump_t = jnp.dot(lambda_x_post_consistent, J_jump_t)
    
    # Terms for Gamma
    f_pre = f_fn(tau, x_pre, z_pre, p_all)
    f_post = f_fn(tau, x_post, z_post, p_all)
    
    H_post = jnp.dot(lambda_x_post_consistent, f_post)
    H_pre = jnp.dot(lambda_tilde, f_pre)
    
    # Guard derivatives
    def zc_x_wrapper(x): return zc_fn(tau, x, z_pre, p_all)[event_idx]
    def zc_t_wrapper(t): return zc_fn(t, x_pre, z_pre, p_all)[event_idx]
    grad_psi_x = jax.grad(zc_x_wrapper)(x_pre)
    grad_psi_t = jax.grad(zc_t_wrapper)(tau)
    
    dpsi_dt_total = grad_psi_t + jnp.dot(grad_psi_x, f_pre)
    safe_denom = jnp.where(jnp.abs(dpsi_dt_total) < 1e-8, 1e-8 * jnp.sign(dpsi_dt_total + 1e-12), dpsi_dt_total)
    
    # Gamma formula
    gamma = (H_post - H_pre + term_jump_t + dt_cons + dL_dtau) / safe_denom
    gamma = jnp.clip(gamma, -1e6, 1e6)

    lambda_pre = lambda_tilde - gamma * grad_psi_x

    # Parameter Gradients
    def jump_p_wrapper(p_sub):
        p_full = p_all.at[jnp.array(optimize_indices)].set(p_sub)
        return jump_fn(x_pre, z_pre, tau, p_full, event_idx)
    def zc_p_wrapper(p_sub):
        p_full = p_all.at[jnp.array(optimize_indices)].set(p_sub)
        return zc_fn(tau, x_pre, z_pre, p_full)[event_idx]

    J_jump_p = jax.jacfwd(jump_p_wrapper)(p_opt)
    grad_psi_p = jax.grad(zc_p_wrapper)(p_opt)
    
    grad_p_event = J_jump_p.T @ lambda_x_post_consistent + gamma * grad_psi_p
    
    return lambda_pre, grad_p_event


# --- Reusing Prediction Kernels ---

@partial(jit, static_argnames=['eval_h_fn', 'n_outputs'])
def predict_trajectory_kernel_linear(segments_t, segments_x, segments_z, target_times, p_all, eval_h_fn, n_outputs):
    all_t = jnp.concatenate(segments_t)
    all_x = jnp.concatenate(segments_x)
    if segments_z[0].size > 0:
        all_z = jnp.concatenate(segments_z)
    else:
        all_z = jnp.zeros((len(all_t), 0))
    def predict_single_time(t_q):
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

@partial(jit, static_argnames=['eval_h_fn', 'n_outputs', 'max_segments', 'max_points', 'n_alg'])
def predict_trajectory_kernel_padded_linear(
    t_start, pad_tau, pad_x, pad_z, seg_mask,
    target_times, p_all, eval_h_fn, n_outputs,
    max_segments, max_points, n_alg
):
    n_states = pad_x.shape[2]
    starts = jnp.concatenate([jnp.array([t_start]), pad_tau[:-1]])
    ends = pad_tau
    unit_grid = jnp.linspace(0.0, 1.0, max_points)
    dynamic_t = starts[:, None] + (ends - starts)[:, None] * unit_grid[None, :]
    flat_t = dynamic_t.reshape(-1)
    flat_x = pad_x.reshape(-1, n_states)
    flat_z = pad_z.reshape(-1, n_alg) if n_alg > 0 else jnp.zeros((len(flat_t), 0))
    point_mask = jnp.repeat(seg_mask, max_points)

    def predict_single_time(t_q):
        def find_in_segment(carry, seg_data):
            result = carry
            t_seg, x_seg, z_seg, is_valid = seg_data
            t_start_seg, t_end_seg = t_seg[0], t_seg[-1]
            in_range = (t_q >= t_start_seg) & (t_q <= t_end_seg) & (is_valid > 0.5)
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
            new_result = jnp.where(in_range, h_val, result)
            return new_result, None
        init = jnp.zeros(n_outputs)
        seg_data = (dynamic_t, pad_x, pad_z, seg_mask)
        result, _ = lax.scan(find_in_segment, init, seg_data)
        return result
    return vmap(predict_single_time)(target_times)

# =============================================================================
# 3. Main Optimizer Class
# =============================================================================

class DAEOptimizerPaddedCorrected:
    def __init__(self, dae_data: Dict, optimize_params: List[str], solver=None, verbose: bool = True,
                 blend_sharpness: float = 100.0, max_segments: int = 20, 
                 ncp: int = 200, safety_buffer_pct: float = 1.2,
                 prediction_method: str = 'linear'):
        
        self.dae_data = dae_data
        self.optimize_params = optimize_params
        self.verbose = verbose
        # Only supporting linear prediction for now
        self.prediction_method = 'linear' 
        
        self.solver = solver if solver else DAESolver(dae_data, verbose=verbose)
        self.param_names = [p['name'] for p in dae_data['parameters']]
        self.p_all = np.array([p['value'] for p in dae_data['parameters']])
        self.optimize_indices = tuple([self.param_names.index(p) for p in optimize_params])
        self.n_opt_params = len(optimize_params)
        self.p_opt = np.array([self.p_all[i] for i in self.optimize_indices])

        self.n_states = len(dae_data['states'])
        self.n_alg = len(dae_data.get('alg_vars', []))
        self.n_total = self.n_states + self.n_alg

        # Padding limits
        self.max_segments = max_segments
        self.max_points_per_seg = int(ncp * safety_buffer_pct)
        
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
        n_segs = len(aug_sol.segments)
        n_alg_padded = max(1, self.n_alg)

        pad_t = np.zeros((self.max_segments, self.max_points_per_seg))
        pad_x = np.zeros((self.max_segments, self.max_points_per_seg, self.n_states))
        pad_z = np.zeros((self.max_segments, self.max_points_per_seg, n_alg_padded))
        pad_tau = np.zeros(self.max_segments)
        seg_mask = np.zeros(self.max_segments)
        point_mask = np.zeros((self.max_segments, self.max_points_per_seg))

        for i in range(min(n_segs, self.max_segments)):
            seg = aug_sol.segments[i]
            t_raw, x_raw, z_raw = seg.t, seg.x, seg.z
            n_raw = len(t_raw)
            n_valid = min(n_raw, self.max_points_per_seg)
            
            pad_t[i, :n_valid] = t_raw[:n_valid]
            pad_x[i, :n_valid, :] = x_raw[:n_valid, :]
            if self.n_alg > 0:
                pad_z[i, :n_valid, :self.n_alg] = z_raw[:n_valid, :]
            
            if n_valid > 0:
                pad_t[i, n_valid:] = t_raw[n_valid-1]
                pad_x[i, n_valid:, :] = x_raw[n_valid-1, :]
                if self.n_alg > 0:
                    pad_z[i, n_valid:, :self.n_alg] = z_raw[n_valid-1, :]
            
            seg_mask[i] = 1.0
            point_mask[i, :n_valid] = 1.0

        t_end = aug_sol.segments[-1].t[-1] if aug_sol.segments else 0.0
        for i in range(min(len(aug_sol.events), self.max_segments)):
            pad_tau[i] = aug_sol.events[i].t_event
        for i in range(len(aug_sol.events), self.max_segments):
            pad_tau[i] = t_end

        return (jnp.array(pad_t), jnp.array(pad_x), jnp.array(pad_z),
                jnp.array(pad_tau), jnp.array(seg_mask), jnp.array(point_mask))

    @partial(jit, static_argnames=['self'])
    def _compute_loss_and_grad_padded_linear_jit(self, t_start, pad_tau, pad_x, pad_z, seg_mask,
                                                  target_times, target_outputs, p_all):
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
        return loss, grad_x, grad_z, grad_tau

    def optimization_step(self, t_span, target_times, target_outputs, p_opt, ncp=200):
        t0 = time.time()
        
        # 1. Forward Solve
        aug_sol = self.forward_solve(t_span, p_opt, ncp)
        n_segments_actual = len(aug_sol.segments)
        
        # 2. Resampling/Padding
        pad_t, pad_x, pad_z, pad_tau, seg_mask, point_mask = self._pad_trajectory(aug_sol)
        t_forward = time.time() - t0
        
        t1 = time.time()
        p_all_jax = self.p_all.copy()
        p_all_jax[np.array(self.optimize_indices)] = p_opt
        p_all_jax = jnp.array(p_all_jax)

        # 3. Loss Gradients (Backprop through prediction)
        loss, dL_dx_padded, dL_dz_padded, dL_dtau_padded = self._compute_loss_and_grad_padded_linear_jit(
            t_span[0], pad_tau, pad_x, pad_z, seg_mask,
            jnp.array(target_times), jnp.array(target_outputs), p_all_jax
        )

        # 4. Backward Sweep
        grad_p_total = jnp.zeros(self.n_opt_params)
        lambda_curr = jnp.zeros(self.n_total) # Stores [lambda_x, lambda_z]
        
        # Loop backwards
        n_segments_jit = n_segments_actual # Use actual for loop count
        
        for i in range(n_segments_jit - 1, -1, -1):
            t_seg_pad = pad_t[i]
            x_seg_pad = pad_x[i]
            z_seg_pad = pad_z[i]
            y_seg_pad = jnp.concatenate([x_seg_pad, z_seg_pad[:, :self.n_alg]], axis=1) if self.n_alg > 0 else x_seg_pad
            
            dL_x_pad = dL_dx_padded[i]
            dL_z_pad = dL_dz_padded[i, :, :self.n_alg] if self.n_alg > 0 else jnp.zeros((self.max_points_per_seg, 0))
            dL_y_pad = jnp.concatenate([dL_x_pad, dL_z_pad], axis=1) if self.n_alg > 0 else dL_x_pad # Combined dL
            
            mask_seg = point_mask[i]
            
            # Reverse
            t_rev = t_seg_pad[::-1]
            y_rev = y_seg_pad[::-1]
            dL_rev = dL_y_pad[::-1]
            mask_rev = mask_seg[::-1]
            
            lambda_init = lambda_curr[:self.n_states] + dL_rev[0][:self.n_states]
            scan_len = self.max_points_per_seg - 1
            
            # Call Corrected Scan
            lambda_final, grad_p_seg = backward_scan_kernel_implicit_corrected(
                lambda_init, jnp.zeros(self.n_opt_params),
                t_rev[:scan_len], t_rev[1:scan_len + 1],
                y_rev[:scan_len], y_rev[1:scan_len + 1],
                dL_rev[1:scan_len + 1],
                jnp.array(mask_rev[1:scan_len + 1]),
                p_all_jax, tuple(self.optimize_indices), self.n_states, self.n_alg,
                self._eval_f, self._eval_g
            )
            
            grad_p_total += grad_p_seg
            
            # Handle Event at i-1
            if i > 0:
                ev = aug_sol.events[i - 1]
                ev_dL_dtau = dL_dtau_padded[i - 1]
                
                # Reconstruct lambda_z from dL at the restart point
                dL_z_at_start = dL_y_pad[0][self.n_states:] # dL/dz at t_start
                lambda_curr = jnp.concatenate([lambda_final, dL_z_at_start if self.n_alg > 0 else jnp.array([])])
                
                lambda_curr, grad_p_ev = backward_event_kernel_corrected(
                    lambda_curr,
                    jnp.array(ev.x_pre), jnp.array(ev.z_pre),
                    jnp.array(ev.x_post), jnp.array(ev.z_post),
                    jnp.array(ev.t_event),
                    p_all_jax, tuple(self.optimize_indices), ev.event_idx, 
                    self.n_states, self.n_alg,
                    self._eval_f, self._eval_g, self._eval_jump, self._eval_zc,
                    ev_dL_dtau
                )
                grad_p_total += grad_p_ev
            else:
                # Last segment (first in time), just update lambda_curr
                lambda_curr = jnp.concatenate([lambda_final, jnp.zeros(0)])

        t_adjoint = time.time() - t1
        return p_opt, float(loss), np.array(grad_p_total), t_forward, t_adjoint

    def forward_solve(self, t_span, p_opt, ncp=200):
        p_all = self.p_all.copy()
        p_all[np.array(self.optimize_indices)] = p_opt
        self.solver.p = p_all
        self.solver.x0 = np.array([s['start'] for s in self.dae_data['states']])
        self.solver.z0 = np.array([a.get('start', 0.0) for a in self.dae_data.get('alg_vars', [])])
        return self.solver.solve_augmented(t_span, ncp=ncp)

    def optimize(self, t_span, target_times, target_outputs, max_iterations=100, step_size=0.01, tol=1e-6, ncp=200):
        p_opt = np.array(self.p_opt)
        history = {'loss': [], 'gradient_norm': [], 'params': []}
        
        m = np.zeros_like(p_opt)
        v = np.zeros_like(p_opt)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for it in range(max_iterations):
            _, loss, grad, t_f, t_a = self.optimization_step(t_span, target_times, target_outputs, p_opt, ncp)
            grad_norm = np.linalg.norm(grad)
            history['loss'].append(loss)
            history['gradient_norm'].append(grad_norm)
            history['params'].append(p_opt.copy())

            if self.verbose:
                print(f"Iter {it}: Loss={loss:.6e}, |g|={grad_norm:.6e}, Tf={t_f:.2f}s, Ta={t_a:.2f}s")
            
            if grad_norm < tol:
                if self.verbose: print("Converged.")
                break
            
            # Adam
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**(it + 1))
            v_hat = v / (1 - beta2**(it + 1))
            p_opt = p_opt - step_size * m_hat / (np.sqrt(v_hat) + eps)
            
        return {'params': p_opt, 'history': history, 'converged': grad_norm < tol}
