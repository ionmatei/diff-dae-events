"""
Explicit Discrete Adjoint Optimizer for DAEs with Events (Padded JIT Version).
Uses verified JIT kernels for fast gradient computation with variable-length trajectories.
"""

import jax
import jax.numpy as jnp
from jax import jit, grad
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Callable
from functools import partial
import re

from .dae_solver import DAESolver, AugmentedSolution
from .dae_direct_padded import compute_adjoint_sweep_padded, pad_problem_data

# Use double precision
jax.config.update("jax_enable_x64", True)

# =============================================================================
# Helper: Equation Compiler (Adapted)
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
    
    # Debug: Print first equation's compiled form if verbose?
    # Cannot easily pass verbose flag here.
    # But usually this is robust.
    
    def create_closure():
        exec_ns = {'jnp': jnp}
        for f in math_funcs:
            if hasattr(jnp, f): exec_ns[f] = getattr(jnp, f)
        exec_ns['power'] = jnp.power 
        source = f"def compiled_fn(t, x, z, p): return jnp.array({return_expr})"
        print(f"Compiled Source: {source}") # Uncomment for deeper debug
        exec(source, exec_ns)
        return exec_ns['compiled_fn']

    return create_closure()

# =============================================================================
# Helper: Prediction & Loss Kernels (JIT)
# =============================================================================

@partial(jit, static_argnames=['eval_h_fn', 'n_outputs', 'max_blocks', 'max_pts', 'dims'])
def predict_trajectory_padded(
    W_padded, TS_padded, b_types, b_indices, b_param,
    target_times, p_all, eval_h_fn, n_outputs,
    max_blocks, max_pts, dims, blend_sharpness=100.0
):
    """
    Predicts outputs from Padded Trajectory using Sigmoid Blending.
    Iterates over target_times (vmap). For each tq, scans blocks to find active segment.
    """
    n_x, n_z, n_p = dims
    
    def predict_single(t_q):
        def scan_body_pred(carry, idx):
            # idx is block index
            y_acc, w_acc = carry
            
            b_type = b_types[idx] # 1=Seg
            t_seg = TS_padded[idx]
            w_seg = W_padded[idx] 
            
            # Validity Check
            is_seg = (b_type == 1)
            
            # Start/End times
            n_valid = b_indices[idx, 1] 
            # Note: For padding, n_valid is at least 0. 
            # If b_type=0, n_valid might be 0.
            
            # We need safe access
            t_start = t_seg[0]
            # Use where to avoid indexing -1 if n_valid=0
            safe_end_idx = jnp.maximum(0, n_valid - 1)
            t_end_val = t_seg[safe_end_idx]
            
            # Masking Logic
            # Assuming simple containment for now: t_start <= tq <= t_end
            # With sigmoid smoothing at boundaries.
            lower = t_start
            upper = t_end_val
            
            # Sigmoid mask
            mask = (jax.nn.sigmoid(blend_sharpness * (t_q - lower)) * 
                    jax.nn.sigmoid(blend_sharpness * (upper - t_q)))
            
            # Zero out if not segment or invalid
            # Mask requires n_valid >= 2 typically for interpolation
            valid_len = (n_valid >= 2)
            mask = mask * is_seg * valid_len
            
            # Interpolation
            idx_t = jnp.searchsorted(t_seg, t_q, side='right') - 1
            idx_t = jnp.clip(idx_t, 0, max_pts - 2)
            
            t0, t1 = t_seg[idx_t], t_seg[idx_t+1]
            dt = t1 - t0
            # Safe dt
            dt = jnp.where(jnp.abs(dt) < 1e-12, 1e-12, dt)
            s = jnp.clip((t_q - t0)/dt, 0.0, 1.0)
            
            w0 = w_seg[idx_t]
            w1 = w_seg[idx_t+1]
            w_i = w0*(1-s) + w1*s
            
            x_i = w_i[:n_x]
            z_i = w_i[n_x:] if n_z > 0 else jnp.array([])
            
            val = eval_h_fn(t_q, x_i, z_i, p_all)
            
            # Only accumulate if mask > cutoff? 
            # Strict accumulation
            return (y_acc + mask*val, w_acc + mask), None

        init = (jnp.zeros(n_outputs), 0.0)
        (y_final, w_final), _ = jax.lax.scan(scan_body_pred, init, jnp.arange(max_blocks))
        
        return y_final / (w_final + 1e-9)

    return jax.vmap(predict_single)(target_times)

# Standard Mean Squared Error Loss
def mse_loss(y_pred, y_true):
    return jnp.mean((y_pred - y_true)**2)

# =============================================================================
# Optimizer Class
# =============================================================================

class DAEOptimizerPadded:
    def __init__(self, dae_data: Dict, optimize_params: List[str], verbose: bool = True,
                 max_blocks: int = 50, max_pts: int = 200, ncp: int = 50):
        self.dae_data = dae_data
        self.verbose = verbose
        self.max_blocks = max_blocks
        self.max_pts_per_seg = max_pts
        self.ncp = ncp
        
        # 1. Initialize Solver
        self.solver = DAESolver(dae_data, verbose=False)
        
        # 2. Dimensions and Params
        self.param_names = [p['name'] for p in dae_data['parameters']]
        self.p_all_init = np.array([p['value'] for p in dae_data['parameters']])
        self.opt_indices = [self.param_names.index(p) for p in optimize_params]
        self.n_p_opt = len(self.opt_indices)
        self.p_opt_indices_jax = jnp.array(self.opt_indices)
        
        self.n_x = len(dae_data['states'])
        self.n_z = len(dae_data.get('alg_vars', []))
        self.n_p = len(self.param_names)
        self.dims = (self.n_x, self.n_z, self.n_p)
        
        # 3. Compile JAX Functions
        self._compile_functions()
        
        # 4. Compile Adjoint Kernel (Bind static args)
        # Using imported Verified Kernel
        # def compute_adjoint_sweep_padded(W_padded, TS_padded, p_opt, ..., funcs, dims, max_blocks)
        
        self.adjoint_kernel_jit = partial(compute_adjoint_sweep_padded,
                                          funcs=self.jax_funcs,
                                          dims=self.dims,
                                          max_blocks=self.max_blocks)
                                          
        # 5. Compile Loss Gradient Logic
        # We need dL/dW_padded.
        # Loss = MSE( Predict(W_padded) - Targets )
        
        # Define Loss Function Closure
        def loss_fn(W_padded, TS_padded, b_types, b_indices, b_param, 
                   target_times, target_outputs, p_curr_all):
            
            y_pred = predict_trajectory_padded(
                W_padded, TS_padded, b_types, b_indices, b_param,
                target_times, p_curr_all, self._eval_h, self.solver.n_outputs,
                self.max_blocks, self.max_pts_per_seg, self.dims
            )
            return mse_loss(y_pred, target_outputs)
            
        # Compile Gradient of Loss w.r.t W_padded
        # argnums=0 is W_padded
        self.loss_grad_jit = jit(grad(loss_fn, argnums=0))
        # Compile Gradient of Loss w.r.t P (Direct contribution)
        self.loss_grad_p_jit = jit(grad(loss_fn, argnums=7))

        # 6. Warmup / Pre-compile
        if self.verbose:
            print("Compiling JIT kernels (Warmup)...")
        self._warmup()
        if self.verbose:
            print("Compilation Complete.")

    @staticmethod
    def create_jax_functions(dae_data):
        import re
        state_names = [s['name'] for s in dae_data['states']]
        alg_names = [a['name'] for a in dae_data.get('alg_vars', [])]
        param_names = [p['name'] for p in dae_data['parameters']]
        
        # Compile f
        f_eqs = dae_data['f']
        f_exprs = [eq.split('=', 1)[1].strip() if '=' in eq else eq for eq in f_eqs]
                
        # Compile g
        g_eqs = dae_data.get('g', [])
        g_exprs = []
        for eq in g_eqs:
            if '=' in eq:
                lhs, rhs = eq.split('=', 1)
                g_exprs.append(f"({lhs.strip()}) - ({rhs.strip()})")
            else:
                g_exprs.append(eq)
                
        # Compile Guard & Reinit
        when_clauses = dae_data.get('when', [])
        guard_exprs = []
        reinit_exprs = [] 
        reinit_vars = [] 
        
        for wc in when_clauses:
            cond = wc['condition']
            if '<' in cond: lhs, rhs = cond.split('<', 1)
            elif '>' in cond: lhs, rhs = cond.split('>', 1)
            else: lhs, rhs = cond.split('=', 1)
            guard_exprs.append(f"({lhs}) - ({rhs})")
            
            reinit_str = wc['reinit']
            if '=' in reinit_str:
                lhs, rhs = reinit_str.split('=', 1)
                raw_expr = f"({lhs}) - ({rhs})"
                lhs_clean = lhs
                for i, name in enumerate(state_names):
                     if re.search(r'\b' + re.escape(name) + r'\b', lhs_clean):
                         reinit_vars.append(('state', i))
                         break
            else:
                raw_expr = reinit_str 
            reinit_exprs.append(raw_expr)
        
        h_exprs = dae_data.get('h', [])
        use_default_h = (len(h_exprs) == 0)
        
        def compile_to_jax(expr_list, is_reinit=False):
            if not expr_list:
                if is_reinit: return lambda t, xp, zp, x, z, p: jnp.array([])
                else: return lambda t, x, z, p: jnp.array([])
                
            subs = []
            for i, n in enumerate(state_names): 
                target = f"x_post[{i}]" if is_reinit else f"x[{i}]"
                subs.append((n, target))
            for i, n in enumerate(alg_names):
                target = f"z_post[{i}]" if is_reinit else f"z[{i}]"
                subs.append((n, target))
            for i, n in enumerate(param_names): subs.append((n, f"p[{i}]"))
            subs.append(('time', 't'))
            subs.sort(key=lambda x: len(x[0]), reverse=True)
            
            jax_exprs = []
            for e in expr_list:
                final_e = e
                if is_reinit:
                    def replace_prev(match):
                        var = match.group(1)
                        if var in state_names: return f"x_pre[{state_names.index(var)}]"
                        if var in alg_names: return f"z_pre[{alg_names.index(var)}]"
                        return f"prev_{var}"
                    final_e = re.sub(r'prev\s*\(\s*(\w+)\s*\)', replace_prev, final_e)
                
                for name, repl in subs:
                    pattern = r'(?<!\.)\b' + re.escape(name) + r'\b'
                    final_e = re.sub(pattern, repl, final_e)
                jax_exprs.append(final_e)
                
            args = "t, x_post, z_post, x_pre, z_pre, p" if is_reinit else "t, x, z, p"
            code = f"def func({args}): return jnp.array([{', '.join(jax_exprs)}])"
            
            local_scope = {'jnp': jnp}
            # Include standard math
            math_funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'tanh']
            for f in math_funcs:
                if hasattr(jnp, f): local_scope[f] = getattr(jnp, f)
            local_scope['power'] = jnp.power

            exec(code, local_scope)
            return local_scope['func']

        f_fn = compile_to_jax(f_exprs, False)
        g_fn = compile_to_jax(g_exprs, False)
        guard_fn = compile_to_jax(guard_exprs, False)
        reinit_res_fn = compile_to_jax(reinit_exprs, True)
        h_fn = lambda t, x, z, p: x if use_default_h else compile_to_jax(h_exprs, False)(t, x, z, p)

        return f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, tuple(reinit_vars), (len(state_names), len(alg_names), len(param_names))

    def _compile_functions(self):
        """
        Uses static helper verified in debug scripts.
        """
        if self.verbose: 
            print("[DAEOptimizerPadded] Compiling functions from DAE specification (Static Method)...")
            
        (self._eval_f, self._eval_g, self._eval_h, 
         self._eval_guard, self._eval_reinit, 
         self.reinit_vars, dims) = DAEOptimizerPadded.create_jax_functions(self.dae_data)
        
        # Infer n_outputs from h
        h_eqs = self.dae_data.get('h', [])
        self.solver.n_outputs = len(h_eqs) if h_eqs else self.n_x

        # Bundle for Solver
        self.jax_funcs = (
            self._eval_f,
            self._eval_g,
            self._eval_h,
            self._eval_guard,
            self._eval_reinit,
            self.reinit_vars,
            self.dims
        )

    def _warmup(self):
        """Runs a dummy call to force JIT compilation."""
        # Dummy Padded Data
        n_w = self.n_x + self.n_z
        W_dummy = jnp.zeros((self.max_blocks, self.max_pts_per_seg, n_w))
        TS_dummy = jnp.linspace(0, 1, self.max_pts_per_seg)[None, :].repeat(self.max_blocks, axis=0)
        b_types = jnp.zeros(self.max_blocks, dtype=int)
        b_indices = jnp.zeros((self.max_blocks, 2), dtype=int)
        b_param = jnp.zeros(self.max_blocks)
        dL_padded = jnp.zeros_like(W_dummy)
        dL_dp = jnp.zeros(self.n_p)
        
        target_times = jnp.array([0.0, 1.0])
        target_outputs = jnp.zeros((2, self.solver.n_outputs))
        
        # 1. Warmup Loss Grad
        _ = self.loss_grad_jit(W_dummy, TS_dummy, b_types, b_indices, b_param, target_times, target_outputs, self.p_all_init)
        
        # 2. Warmup Adjoint
        _ = self.adjoint_kernel_jit(
            W_dummy, TS_dummy, self.p_all_init,
            b_types, b_indices, b_param,
            dL_padded, dL_dp
        )

    def compute_gradient(self, p_new_values: np.ndarray, 
                         target_times: Optional[np.ndarray] = None, 
                         target_outputs: Optional[np.ndarray] = None):
        """
        Computes the gradient of the loss with respect to parameters.
        """
        # 1. Update Parameters (Numpy)
        for i, idx in enumerate(self.opt_indices):
            self.dae_data['parameters'][idx]['value'] = float(p_new_values[i])
        
        # 2. Forward Simulation (Solver)
        sol = self.solver.solve_augmented((0.0, 2.0), ncp=self.ncp) # TODO: Configurable T_span?
        
        # 3. Data Prep (Padding)
        # Assuming targets provided, OR use internal?
        # If None, must be set previously.
        if target_times is None:
             raise ValueError("Target times/outputs required for gradient computation")
             
        ts_all = [s.t for s in sol.segments]
        ys_all = [jnp.concatenate([s.x, s.z], axis=1) for s in sol.segments]
        event_infos = [(e.t_event, 0) for e in sol.events] # Assuming single event type (idx=0)
        
        W_p, TS_p, b_types, b_indices, b_param, _ = pad_problem_data(
            ts_all, ys_all, event_infos, self.max_blocks, self.max_pts_per_seg, self.dims
        )
        
        # Convert to JAX arrays
        W_p_jax = jnp.array(W_p)
        TS_p_jax = jnp.array(TS_p)
        b_types_jax = jnp.array(b_types)
        b_indices_jax = jnp.array(b_indices)
        b_param_jax = jnp.array(b_param)
        
        p_curr_jax = jnp.array([p['value'] for p in self.dae_data['parameters']])
        target_times_jax = jnp.array(target_times)
        target_outputs_jax = jnp.array(target_outputs)
        
        # 4. Compute Loss Gradients (AD phase)
        # dL/dW_padded
        dL_dW_p = self.loss_grad_jit(
             W_p_jax, TS_p_jax, b_types_jax, b_indices_jax, b_param_jax,
             target_times_jax, target_outputs_jax, p_curr_jax
        )
        if self.verbose: 
             print(f"dL_dW norm: {jnp.linalg.norm(dL_dW_p)}")
             print(f"dL_dW nonzero: {jnp.count_nonzero(dL_dW_p)}")
             print(f"W_p stats: max={jnp.max(W_p_jax)}, min={jnp.min(W_p_jax)}")
             print(f"Block Types: {b_types_jax}")
             print(f"Block Indices: {b_indices_jax}")
        
        # dL/dp (Explicit dependency)
        dL_dp = self.loss_grad_p_jit(
             W_p_jax, TS_p_jax, b_types_jax, b_indices_jax, b_param_jax,
             target_times_jax, target_outputs_jax, p_curr_jax
        )
        
        # 5. Run Adjoint (JIT)
        grad_p_total = self.adjoint_kernel_jit(
            W_p_jax, TS_p_jax, p_curr_jax,
            b_types_jax, b_indices_jax, b_param_jax,
            dL_dW_p, dL_dp
        )
        if self.verbose:
             print(f"Full Gradient Vector: {grad_p_total}")
        
        # 6. Extract Optimized Params Gradient
        grad_opt = grad_p_total[self.p_opt_indices_jax]
        
        return grad_opt
