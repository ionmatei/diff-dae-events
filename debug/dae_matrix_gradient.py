
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import time

# Import helper to create functions
import re

class DAEMatrixGradient:
    def __init__(self, dae_data, max_pts=500, downsample_segments=False, all_segments=False):
        """
        Initialize the Matrix Gradient Computer.
        Args:
            dae_data: Dictionary containing DAE system specification
            max_pts: Maximum points per segment (used when downsample_segments=True)
            downsample_segments: If True, downsample segments exceeding max_pts
            all_segments: If True (and downsample_segments=True), resample ALL segments to max_pts
        """
        self.dae_data = dae_data
        self.max_pts = max_pts
        self.downsample_segments = downsample_segments
        self.all_segments = all_segments

        # Create JAX functions
        self.funcs = self._create_jax_functions(dae_data)
        self.f_fn, self.g_fn, self.h_fn, self.guard_fn, self.reinit_res_fn, self.reinit_vars, self.dims = self.funcs
        self.n_x, self.n_z, self.n_p = self.dims
        self.n_w = self.n_x + self.n_z
        
        # Initial state (fixed for now, or could be parameter)
        self.x0_start = jnp.array([s['start'] for s in dae_data['states']])

        # Cache for JIT-compiled gradient functions: structure_hash -> jit_fn
        self._kernel_cache = {}

    def _create_jax_functions(self, dae_data):
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

            # Handle list-valued reinit
            reinit = wc['reinit']
            reinit_list = reinit if isinstance(reinit, list) else [reinit]

            for reinit_str in reinit_list:
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
            exec(code, local_scope)
            return local_scope['func']

        f_fn = compile_to_jax(f_exprs, False)
        g_fn = compile_to_jax(g_exprs, False)
        guard_fn = compile_to_jax(guard_exprs, False)
        reinit_res_fn = compile_to_jax(reinit_exprs, True)
        h_fn = lambda t, x, z, p: x if use_default_h else compile_to_jax(h_exprs, False)(t, x, z, p)

        return f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, tuple(reinit_vars), (len(state_names), len(alg_names), len(param_names))

    @staticmethod
    def _downsample_segment(t_seg, y_seg, max_pts):
        """Downsample a segment to max_pts points via linear interpolation.

        Preserves first and last time instants. Interior points are
        uniformly spaced. State values are linearly interpolated.
        """
        t_new = np.linspace(t_seg[0], t_seg[-1], max_pts)
        y_new = np.empty((max_pts, y_seg.shape[1]), dtype=y_seg.dtype)
        for col in range(y_seg.shape[1]):
            y_new[:, col] = np.interp(t_new, t_seg, y_seg[:, col])
        return t_new, y_new

    def pack_solution(self, sol):
        """
        Packs AugmentedSolution into flat W and structure description.
        Returns: W_flat, structure, grid_taus
        """
        w_list = []
        structure = []
        grid_taus = []

        num_seg = len(sol.segments)
        num_events = len(sol.events)

        # Pre-extract and optionally downsample segment data (numpy)
        seg_ts = [np.asarray(s.t) for s in sol.segments]
        seg_ys = [np.concatenate([s.x, s.z], axis=1) for s in sol.segments]
        if self.downsample_segments:
            for i in range(num_seg):
                should_downsample = self.all_segments or (seg_ts[i].shape[0] > self.max_pts)
                if should_downsample:
                    seg_ts[i], seg_ys[i] = self._downsample_segment(
                        seg_ts[i], seg_ys[i], self.max_pts
                    )

        for i in range(num_seg):
            t_arr = seg_ts[i]
            y_arr = seg_ys[i]
            n_points = t_arr.shape[0]
            
            # Calculate Tau Grid (Normalized)
            t_start = t_arr[0]
            t_end = t_arr[-1]
            denom = t_end - t_start
            if denom < 1e-12: denom = 1.0
            tau = (t_arr - t_start) / denom
            grid_taus.append(tau)

            seg_start_idx = len(w_list)

            for k in range(n_points):
                w_list.extend(y_arr[k])
                
            seg_len = len(w_list) - seg_start_idx
            structure.append(('segment', n_points, seg_len))
            
            if i < num_events:
                ev = sol.events[i]
                w_list.append(ev.t_event)
                structure.append(('event_time', 1))
                
        return jnp.array(w_list), structure, grid_taus

    def _get_gradient_kernel(self, structure):
        """
        Returns a JIT-compiled function that computes total gradients for the given structure.
        The function signature is:
           compute_grads(W_flat, p_val, grid_taus, target_times, target_data)
        """
        # Create a hashable representation of structure
        struct_key = tuple(structure)
        
        if struct_key in self._kernel_cache:
            return self._kernel_cache[struct_key]

        print(f"Compiling Matrix Gradient Kernel for structure: {len(structure)} blocks...")
        
        # Capture constants
        dims = self.dims
        funcs = self.funcs
        x0_start = self.x0_start
        n_x, n_z, n_p = dims
        n_w = n_x + n_z

        def residual_fn(W_flat, p, grid_taus, t_final_val):
            # Reconstruct and compute residual vector R(W, p) = 0
            residuals = []
            
            # Identify indices
            event_indices_in_W = []
            idx_scan = 0
            for kind, count, *extra in structure:
                if kind == 'event_time':
                    event_indices_in_W.append(idx_scan)
                length = extra[0] if kind == 'segment' else count
                idx_scan += length
                
            event_counter = 0
            seg_counter = 0
            idx_scan = 0
            t_start_seg = 0.0 # Will be updated
            
            last_x = None
            last_z = None

            for i, (kind, count, *extra) in enumerate(structure):
                if kind == 'segment':
                    n_pts = count
                    length = extra[0]
                    segment_data = W_flat[idx_scan : idx_scan + length].reshape((n_pts, n_w))
                    idx_scan += length
                    
                    xs = segment_data[:, :n_x]
                    zs = segment_data[:, n_x:]
                    
                    # Determine te for this segment
                    if event_counter < len(event_indices_in_W):
                        te_idx = event_indices_in_W[event_counter]
                        te = W_flat[te_idx]
                    else:
                        te = t_final_val
                        
                    current_tau = grid_taus[seg_counter]
                    t0 = t_start_seg
                    ts = t0 + current_tau * (te - t0)
                    
                    # 1. Initial Condition / Continuity (First Segment)
                    if i == 0:
                        residuals.extend(xs[0] - x0_start)
                        
                    # 2. Flow Residuals
                    # Vectorized across points 0..N-2
                    if n_pts > 1:
                        # Slice arrays
                        ts_curr = ts[:-1]
                        ts_next = ts[1:]
                        xs_curr = xs[:-1]
                        xs_next = xs[1:]
                        zs_curr = zs[:-1] if n_z > 0 else None
                        zs_next = zs[1:] if n_z > 0 else None
                        
                        dt = ts_next - ts_curr
                        
                        # vmap eval
                        # f_fn(t, x, z, p)
                        def call_f(t, x, z): return funcs[0](t, x, z, p)
                        def call_g(t, x, z): return funcs[1](t, x, z, p)
                        
                        f_curr = jax.vmap(call_f)(ts_curr, xs_curr, zs_curr)
                        f_next = jax.vmap(call_f)(ts_next, xs_next, zs_next)
                        
                        # Trapezoidal / Implicit Euler
                        # Res = -x_next + x_curr + 0.5*h*(f_curr + f_next)
                        res_flow = -xs_next + xs_curr + 0.5 * dt[:, None] * (f_curr + f_next)
                        residuals.append(res_flow.flatten())
                        
                        if n_z > 0:
                            g_curr = jax.vmap(call_g)(ts_curr, xs_curr, zs_curr)
                            residuals.append(g_curr.flatten())

                    # G at last point
                    if n_z > 0:
                         residuals.append(funcs[1](ts[-1], xs[-1], zs[-1], p).flatten())
                         
                    last_x = xs[-1]
                    last_z = zs[-1]
                    t_start_seg = te
                    seg_counter += 1
                    
                elif kind == 'event_time':
                    idx_scan += 1
                    te = W_flat[idx_scan - 1]
                    
                    # Event Logic: Guard and Reset
                    # Need next segment data for reset
                    if i + 1 < len(structure):
                        next_kind, next_count, *next_extra = structure[i+1]
                        next_len = next_extra[0]
                        next_seg_data = W_flat[idx_scan : idx_scan + next_len].reshape((next_count, n_w))
                        
                        x_post = next_seg_data[0, :n_x]
                        z_post = next_seg_data[0, n_x:]
                        x_pre = last_x
                        z_pre = last_z if n_z > 0 else jnp.array([])
                        
                        # Guard
                        val_guard = funcs[3](te, x_pre, z_pre, p)
                        residuals.append(val_guard.reshape(-1))
                        
                        # Reset
                        val_reset = funcs[4](te, x_post, z_post, x_pre, z_pre, p)
                        residuals.append(val_reset.reshape(-1))
                        
                        # Continuity for non-reset vars
                        # Check each state
                        # Note: reinit_vars is static list of (type, index)
                        # We can likely vectorize this check or unroll it
                        # For compilation, normal python loop is fine
                        diffs = []
                        for k in range(n_x):
                            is_reinit = False
                            for (rtype, ridx) in funcs[5]: # reinit_vars
                                if rtype == 'state' and ridx == k:
                                    is_reinit = True
                                    break
                            if not is_reinit:
                                diffs.append(x_post[k] - x_pre[k])
                        if diffs:
                            residuals.append(jnp.stack(diffs))
                            
                        # G constraint at post-event
                        if n_z > 0:
                            residuals.append(funcs[1](te, x_post, z_post, p).flatten())
                            
                    event_counter += 1

            return jnp.concatenate([r.flatten() for r in residuals])

        def loss_fn(W_flat, p, grid_taus, target_times, target_data, t_final_val, blend_sharpness=150.0):
            # Unpack W into trajectory for prediction
            # Logic similar to residual but just extracting (t, y)

            # Identify event indices
            idx_scan = 0
            event_vals = []
            for kind, count, *extra in structure:
                length = extra[0] if kind == 'segment' else count
                if kind == 'event_time':
                    event_vals.append(W_flat[idx_scan])
                idx_scan += length

            event_counter = 0
            seg_counter = 0
            idx_scan = 0
            t_start_seg = 0.0

            segments_t = []
            segments_x = []

            for kind, count, *extra in structure:
                if kind == 'segment':
                    length = extra[0]
                    chunk = W_flat[idx_scan : idx_scan + length].reshape((count, n_w))
                    idx_scan += length

                    xs = chunk[:, :n_x]

                    if event_counter < len(event_vals):
                        te = event_vals[event_counter]
                    else:
                        te = t_final_val

                    current_tau = grid_taus[seg_counter]
                    t0 = t_start_seg
                    ts = t0 + current_tau * (te - t0)

                    segments_t.append(ts)
                    segments_x.append(xs)

                    t_start_seg = te
                    seg_counter += 1

                elif kind == 'event_time':
                    idx_scan += 1
                    event_counter += 1

            # Predict using soft (Gaussian kernel) interpolation

            def predict_at_t(t_q):
                y_accum = jnp.zeros(n_x)
                w_accum = 0.0

                for i in range(len(segments_t)):
                    ts = segments_t[i]
                    xs = segments_x[i]
                    t_start = ts[0]
                    t_end = ts[-1]

                    # Window
                    lower = t_start if i == 0 else event_vals[i-1]
                    upper = t_end if i == len(segments_t)-1 else event_vals[i]

                    mask = jax.nn.sigmoid(blend_sharpness * (t_q - lower)) * jax.nn.sigmoid(blend_sharpness * (upper - t_q))

                    # O(1) piecewise-linear interpolation via soft fractional index
                    t_clip = jnp.clip(t_q, t_start, t_end)
                    n_pts = ts.shape[0]
                    span = jnp.maximum(t_end - t_start, 1e-12)
                    frac = jnp.clip((t_clip - t_start) / span, 0.0, 1.0)
                    frac_idx = frac * (n_pts - 1)
                    idx_lo = jnp.floor(frac_idx).astype(jnp.int32)
                    idx_lo = jnp.clip(idx_lo, 0, n_pts - 2)
                    a = frac_idx - idx_lo.astype(jnp.float64)
                    val = xs[idx_lo] * (1.0 - a) + xs[idx_lo + 1] * a

                    y_accum += mask * val
                    w_accum += mask

                return y_accum / (w_accum + 1e-8)

            predictions = jax.vmap(predict_at_t)(target_times)
            
            # Loss: MSE
            diff = predictions - target_data
            return jnp.mean(jnp.square(diff))

        def compute_grads(W_flat, p_val, grid_taus, target_times, target_data, t_final_val, blend_sharpness):
            # 1. Residual Jacobians
            J_W = jax.jacfwd(residual_fn, argnums=0)(W_flat, p_val, grid_taus, t_final_val)
            J_p = jax.jacfwd(residual_fn, argnums=1)(W_flat, p_val, grid_taus, t_final_val)

            # 2. Loss Gradients
            loss_val, dL_dW = jax.value_and_grad(loss_fn, argnums=0)(W_flat, p_val, grid_taus, target_times, target_data, t_final_val, blend_sharpness)
            dL_dp = jax.grad(loss_fn, argnums=1)(W_flat, p_val, grid_taus, target_times, target_data, t_final_val, blend_sharpness)

            # 3. Adjoint Solve
            lam = jnp.linalg.solve(J_W.T, -dL_dW)

            # 4. Total Gradient
            total_grad = dL_dp + jnp.dot(lam, J_p)
            return loss_val, total_grad

        # Compile
        jit_kernel = jax.jit(compute_grads)
        self._kernel_cache[struct_key] = jit_kernel
        return jit_kernel

    def _get_prediction_kernel(self, structure):
        """Returns a JIT kernel that computes predicted trajectory."""
        struct_key = ('predict',) + tuple(structure)
        if struct_key in self._kernel_cache:
            return self._kernel_cache[struct_key]
        
        n_x, n_z, n_p = self.dims
        n_w = n_x + n_z

        def predict_fn(W_flat, grid_taus, target_times, t_final_val, blend_sharpness):
            # Same unpacking logic as loss_fn
            idx_scan = 0
            event_vals = []
            for kind, count, *extra in structure:
                length = extra[0] if kind == 'segment' else count
                if kind == 'event_time':
                    event_vals.append(W_flat[idx_scan])
                idx_scan += length
            
            event_counter = 0
            seg_counter = 0
            idx_scan = 0
            t_start_seg = 0.0
            segments_t = []
            segments_x = []
            
            for kind, count, *extra in structure:
                if kind == 'segment':
                    length = extra[0]
                    chunk = W_flat[idx_scan : idx_scan + length].reshape((count, n_w))
                    idx_scan += length
                    xs = chunk[:, :n_x]
                    if event_counter < len(event_vals):
                        te = event_vals[event_counter]
                    else:
                        te = t_final_val
                    current_tau = grid_taus[seg_counter]
                    t0 = t_start_seg
                    ts = t0 + current_tau * (te - t0)
                    segments_t.append(ts)
                    segments_x.append(xs)
                    t_start_seg = te
                    seg_counter += 1
                elif kind == 'event_time':
                    idx_scan += 1
                    event_counter += 1
            
            def predict_at_t(t_q):
                y_accum = jnp.zeros(n_x)
                w_accum = 0.0
                for i in range(len(segments_t)):
                    ts = segments_t[i]
                    xs = segments_x[i]
                    t_start = ts[0]
                    t_end = ts[-1]
                    lower = t_start if i == 0 else event_vals[i-1]
                    upper = t_end if i == len(segments_t)-1 else event_vals[i]
                    mask = jax.nn.sigmoid(blend_sharpness * (t_q - lower)) * jax.nn.sigmoid(blend_sharpness * (upper - t_q))
                    t_clip = jnp.clip(t_q, t_start, t_end)
                    n_pts = ts.shape[0]
                    span = jnp.maximum(t_end - t_start, 1e-12)
                    frac = jnp.clip((t_clip - t_start) / span, 0.0, 1.0)
                    frac_idx = frac * (n_pts - 1)
                    idx_lo = jnp.floor(frac_idx).astype(jnp.int32)
                    idx_lo = jnp.clip(idx_lo, 0, n_pts - 2)
                    a = frac_idx - idx_lo.astype(jnp.float64)
                    val = xs[idx_lo] * (1.0 - a) + xs[idx_lo + 1] * a
                    y_accum += mask * val
                    w_accum += mask
                return y_accum / (w_accum + 1e-8)
            
            return jax.vmap(predict_at_t)(target_times)

        jit_kernel = jax.jit(predict_fn)
        self._kernel_cache[struct_key] = jit_kernel
        return jit_kernel

    def _get_loss_grad_kernel(self, structure):
        """Returns a JIT kernel that computes (loss_val, dL_dW_flat)."""
        struct_key = ('loss_grad',) + tuple(structure)
        if struct_key in self._kernel_cache:
            return self._kernel_cache[struct_key]

        n_x, n_z, n_p = self.dims
        n_w = n_x + n_z

        def loss_fn(W_flat, p, grid_taus, target_times, target_data, t_final_val, blend_sharpness=150.0):
            idx_scan = 0
            event_vals = []
            for kind, count, *extra in structure:
                length = extra[0] if kind == 'segment' else count
                if kind == 'event_time':
                    event_vals.append(W_flat[idx_scan])
                idx_scan += length

            event_counter = 0
            seg_counter = 0
            idx_scan = 0
            t_start_seg = 0.0

            segments_t = []
            segments_x = []

            for kind, count, *extra in structure:
                if kind == 'segment':
                    length = extra[0]
                    chunk = W_flat[idx_scan : idx_scan + length].reshape((count, n_w))
                    idx_scan += length
                    xs = chunk[:, :n_x]
                    if event_counter < len(event_vals):
                        te = event_vals[event_counter]
                    else:
                        te = t_final_val
                    current_tau = grid_taus[seg_counter]
                    t0 = t_start_seg
                    ts = t0 + current_tau * (te - t0)
                    segments_t.append(ts)
                    segments_x.append(xs)
                    t_start_seg = te
                    seg_counter += 1
                elif kind == 'event_time':
                    idx_scan += 1
                    event_counter += 1

            def predict_at_t(t_q):
                y_accum = jnp.zeros(n_x)
                w_accum = 0.0
                for i in range(len(segments_t)):
                    ts = segments_t[i]
                    xs = segments_x[i]
                    t_start = ts[0]
                    t_end = ts[-1]
                    lower = t_start if i == 0 else event_vals[i-1]
                    upper = t_end if i == len(segments_t)-1 else event_vals[i]
                    mask = jax.nn.sigmoid(blend_sharpness * (t_q - lower)) * jax.nn.sigmoid(blend_sharpness * (upper - t_q))
                    t_clip = jnp.clip(t_q, t_start, t_end)
                    n_pts = ts.shape[0]
                    span = jnp.maximum(t_end - t_start, 1e-12)
                    frac = jnp.clip((t_clip - t_start) / span, 0.0, 1.0)
                    frac_idx = frac * (n_pts - 1)
                    idx_lo = jnp.floor(frac_idx).astype(jnp.int32)
                    idx_lo = jnp.clip(idx_lo, 0, n_pts - 2)
                    a = frac_idx - idx_lo.astype(jnp.float64)
                    val = xs[idx_lo] * (1.0 - a) + xs[idx_lo + 1] * a
                    y_accum += mask * val
                    w_accum += mask
                return y_accum / (w_accum + 1e-8)

            predictions = jax.vmap(predict_at_t)(target_times)
            diff = predictions - target_data
            return jnp.mean(jnp.square(diff))

        def loss_and_grad(W_flat, p_val, grid_taus, target_times, target_data, t_final_val, blend_sharpness):
            return jax.value_and_grad(loss_fn, argnums=0)(
                W_flat, p_val, grid_taus, target_times, target_data, t_final_val, blend_sharpness
            )

        jit_kernel = jax.jit(loss_and_grad)
        self._kernel_cache[struct_key] = jit_kernel
        return jit_kernel

    def compute_loss_and_loss_grad_W(self, sol, p_val, target_times, target_data, blend_sharpness=150.0):
        """Returns (loss_val, dL_dW_flat, W_flat, structure)."""
        t_final = sol.segments[-1].t[-1] if sol.segments else 2.0
        W_flat, structure, grid_taus_list = self.pack_solution(sol)
        grid_taus_tuple = tuple(grid_taus_list)
        kernel = self._get_loss_grad_kernel(structure)
        loss_val, dL_dW = kernel(W_flat, p_val, grid_taus_tuple, target_times, target_data, t_final, blend_sharpness)
        return loss_val, dL_dW, W_flat, structure

    def _get_full_adjoint_kernel(self, structure):
        """Returns a JIT kernel that computes (loss, dL_dW, dL_dp, lam, total_grad)."""
        struct_key = ('full_adjoint',) + tuple(structure)
        if struct_key in self._kernel_cache:
            return self._kernel_cache[struct_key]

        # Reuse the residual_fn and loss_fn from _get_gradient_kernel
        # by building them inline (same closure over structure).
        n_x, n_z, n_p = self.dims
        n_w = n_x + n_z
        funcs = self.funcs
        x0_start = self.x0_start

        def residual_fn(W_flat, p, grid_taus, t_final_val):
            residuals = []
            event_indices_in_W = []
            idx_scan = 0
            for kind, count, *extra in structure:
                if kind == 'event_time':
                    event_indices_in_W.append(idx_scan)
                length = extra[0] if kind == 'segment' else count
                idx_scan += length

            event_counter = 0
            seg_counter = 0
            idx_scan = 0
            t_start_seg = 0.0
            last_x = None
            last_z = None

            for i, (kind, count, *extra) in enumerate(structure):
                if kind == 'segment':
                    n_pts = count
                    length = extra[0]
                    segment_data = W_flat[idx_scan : idx_scan + length].reshape((n_pts, n_w))
                    idx_scan += length
                    xs = segment_data[:, :n_x]
                    zs = segment_data[:, n_x:]
                    if event_counter < len(event_indices_in_W):
                        te = W_flat[event_indices_in_W[event_counter]]
                    else:
                        te = t_final_val
                    current_tau = grid_taus[seg_counter]
                    t0 = t_start_seg
                    ts = t0 + current_tau * (te - t0)
                    if i == 0:
                        residuals.extend(xs[0] - x0_start)
                    if n_pts > 1:
                        ts_curr, ts_next = ts[:-1], ts[1:]
                        xs_curr, xs_next = xs[:-1], xs[1:]
                        zs_curr = zs[:-1] if n_z > 0 else None
                        zs_next = zs[1:] if n_z > 0 else None
                        dt = ts_next - ts_curr
                        def call_f(t, x, z): return funcs[0](t, x, z, p)
                        def call_g(t, x, z): return funcs[1](t, x, z, p)
                        f_curr = jax.vmap(call_f)(ts_curr, xs_curr, zs_curr)
                        f_next = jax.vmap(call_f)(ts_next, xs_next, zs_next)
                        res_flow = -xs_next + xs_curr + 0.5 * dt[:, None] * (f_curr + f_next)
                        residuals.append(res_flow.flatten())
                        if n_z > 0:
                            g_curr = jax.vmap(call_g)(ts_curr, xs_curr, zs_curr)
                            residuals.append(g_curr.flatten())
                    if n_z > 0:
                        residuals.append(funcs[1](ts[-1], xs[-1], zs[-1], p).flatten())
                    last_x = xs[-1]
                    last_z = zs[-1]
                    t_start_seg = te
                    seg_counter += 1
                elif kind == 'event_time':
                    idx_scan += 1
                    te = W_flat[idx_scan - 1]
                    if i + 1 < len(structure):
                        next_kind, next_count, *next_extra = structure[i+1]
                        next_len = next_extra[0]
                        next_seg_data = W_flat[idx_scan : idx_scan + next_len].reshape((next_count, n_w))
                        x_post = next_seg_data[0, :n_x]
                        z_post = next_seg_data[0, n_x:]
                        x_pre = last_x
                        z_pre = last_z if n_z > 0 else jnp.array([])
                        residuals.append(funcs[3](te, x_pre, z_pre, p).reshape(-1))
                        residuals.append(funcs[4](te, x_post, z_post, x_pre, z_pre, p).reshape(-1))
                        diffs = []
                        for k in range(n_x):
                            is_reinit = False
                            for (rtype, ridx) in funcs[5]:
                                if rtype == 'state' and ridx == k:
                                    is_reinit = True
                                    break
                            if not is_reinit:
                                diffs.append(x_post[k] - x_pre[k])
                        if diffs:
                            residuals.append(jnp.stack(diffs))
                        if n_z > 0:
                            residuals.append(funcs[1](te, x_post, z_post, p).flatten())
                    event_counter += 1
            return jnp.concatenate([r.flatten() for r in residuals])

        def loss_fn(W_flat, p, grid_taus, target_times, target_data, t_final_val, blend_sharpness=150.0):
            idx_scan = 0
            event_vals = []
            for kind, count, *extra in structure:
                length = extra[0] if kind == 'segment' else count
                if kind == 'event_time':
                    event_vals.append(W_flat[idx_scan])
                idx_scan += length
            event_counter = 0
            seg_counter = 0
            idx_scan = 0
            t_start_seg = 0.0
            segments_t = []
            segments_x = []
            for kind, count, *extra in structure:
                if kind == 'segment':
                    length = extra[0]
                    chunk = W_flat[idx_scan : idx_scan + length].reshape((count, n_w))
                    idx_scan += length
                    xs = chunk[:, :n_x]
                    if event_counter < len(event_vals):
                        te = event_vals[event_counter]
                    else:
                        te = t_final_val
                    current_tau = grid_taus[seg_counter]
                    t0 = t_start_seg
                    ts = t0 + current_tau * (te - t0)
                    segments_t.append(ts)
                    segments_x.append(xs)
                    t_start_seg = te
                    seg_counter += 1
                elif kind == 'event_time':
                    idx_scan += 1
                    event_counter += 1

            def predict_at_t(t_q):
                y_accum = jnp.zeros(n_x)
                w_accum = 0.0
                for i in range(len(segments_t)):
                    ts = segments_t[i]
                    xs = segments_x[i]
                    t_start = ts[0]
                    t_end = ts[-1]
                    lower = t_start if i == 0 else event_vals[i-1]
                    upper = t_end if i == len(segments_t)-1 else event_vals[i]
                    mask = jax.nn.sigmoid(blend_sharpness * (t_q - lower)) * jax.nn.sigmoid(blend_sharpness * (upper - t_q))
                    t_clip = jnp.clip(t_q, t_start, t_end)
                    n_pts = ts.shape[0]
                    span = jnp.maximum(t_end - t_start, 1e-12)
                    frac = jnp.clip((t_clip - t_start) / span, 0.0, 1.0)
                    frac_idx = frac * (n_pts - 1)
                    idx_lo = jnp.floor(frac_idx).astype(jnp.int32)
                    idx_lo = jnp.clip(idx_lo, 0, n_pts - 2)
                    a = frac_idx - idx_lo.astype(jnp.float64)
                    val = xs[idx_lo] * (1.0 - a) + xs[idx_lo + 1] * a
                    y_accum += mask * val
                    w_accum += mask
                return y_accum / (w_accum + 1e-8)
            predictions = jax.vmap(predict_at_t)(target_times)
            diff = predictions - target_data
            return jnp.mean(jnp.square(diff))

        def full_adjoint(W_flat, p_val, grid_taus, target_times, target_data, t_final_val, blend_sharpness):
            J_W = jax.jacfwd(residual_fn, argnums=0)(W_flat, p_val, grid_taus, t_final_val)
            J_p = jax.jacfwd(residual_fn, argnums=1)(W_flat, p_val, grid_taus, t_final_val)
            loss_val, dL_dW = jax.value_and_grad(loss_fn, argnums=0)(
                W_flat, p_val, grid_taus, target_times, target_data, t_final_val, blend_sharpness
            )
            dL_dp = jax.grad(loss_fn, argnums=1)(
                W_flat, p_val, grid_taus, target_times, target_data, t_final_val, blend_sharpness
            )
            lam = jnp.linalg.solve(J_W.T, -dL_dW)
            total_grad = dL_dp + jnp.dot(lam, J_p)
            return loss_val, dL_dW, dL_dp, lam, total_grad

        jit_kernel = jax.jit(full_adjoint)
        self._kernel_cache[struct_key] = jit_kernel
        return jit_kernel

    def compute_full_adjoint(self, sol, p_val, target_times, target_data, blend_sharpness=150.0):
        """Returns (loss, dL_dW, dL_dp, lam, total_grad, W_flat, structure)."""
        t_final = sol.segments[-1].t[-1] if sol.segments else 2.0
        W_flat, structure, grid_taus_list = self.pack_solution(sol)
        grid_taus_tuple = tuple(grid_taus_list)
        kernel = self._get_full_adjoint_kernel(structure)
        loss_val, dL_dW, dL_dp, lam, total_grad = kernel(
            W_flat, p_val, grid_taus_tuple, target_times, target_data, t_final, blend_sharpness
        )
        return loss_val, dL_dW, dL_dp, lam, total_grad, W_flat, structure

    def compute_total_gradient(self, sol, p_val, target_times, target_data, blend_sharpness=150.0):
        # 1. Pack
        t_final = sol.segments[-1].t[-1] if sol.segments else 2.0

        W_flat, structure, grid_taus_list = self.pack_solution(sol)
        grid_taus_tuple = tuple(grid_taus_list)

        # 2. Get Kernel
        kernel = self._get_gradient_kernel(structure)

        # 3. Compute
        loss_val, grad_p = kernel(W_flat, p_val, grid_taus_tuple, target_times, target_data, t_final, blend_sharpness)

        return loss_val, grad_p

    def predict_trajectory(self, sol, target_times, blend_sharpness=150.0):
        """
        Predict trajectory values at target times using the JIT compiled kernel.
        """
        t_final = sol.segments[-1].t[-1] if sol.segments else 2.0
        W_flat, structure, grid_taus_list = self.pack_solution(sol)
        grid_taus_tuple = tuple(grid_taus_list)
        
        kernel = self._get_prediction_kernel(structure)
        
        y_pred = kernel(W_flat, grid_taus_tuple, target_times, t_final, blend_sharpness)
        return y_pred

    def optimize_adam(self, solver, p_init, opt_param_indices, target_times, target_data,
                      t_span, ncp, max_iter=150, tol=1e-8, step_size=0.01,
                      beta1=0.9, beta2=0.999, epsilon=1e-8,
                      blend_sharpness=150.0, print_every=10):
        
        n_opt = len(opt_param_indices)
        p_current = jnp.array(p_init, dtype=jnp.float64)

        m = jnp.zeros(n_opt)
        v = jnp.zeros(n_opt)

        loss_history = []
        grad_norm_history = []
        converged = False

        print(f"Adam optimization (Matrix Direct): {n_opt} parameters, max_iter={max_iter}, lr={step_size}")

        for it in range(1, max_iter + 1):
            t0 = time.perf_counter()

            # Forward
            solver.update_parameters(np.asarray(p_current))
            try:
                # Use fewer points to keep matrix size manageable if needed
                # But we use configured NCP
                sol = solver.solve_augmented(t_span, ncp=ncp)
            except Exception as e:
                print(f"  Iter {it}: solver failed — {e}")
                break
                
            # Compute Gradient
            loss_val_d, total_grad = self.compute_total_gradient(sol, p_current, target_times, target_data, blend_sharpness)
            total_grad.block_until_ready()
            
            grad_opt = total_grad[jnp.array(opt_param_indices)]
            grad_norm = float(jnp.linalg.norm(grad_opt))
            
            # Extract scalar loss
            loss_val = float(loss_val_d)
            
            loss_history.append(loss_val)
            grad_norm_history.append(grad_norm)

            # Update
            m = beta1 * m + (1.0 - beta1) * grad_opt
            v = beta2 * v + (1.0 - beta2) * grad_opt ** 2
            m_hat = m / (1.0 - beta1 ** it)
            v_hat = v / (1.0 - beta2 ** it)
            step = step_size * m_hat / (jnp.sqrt(v_hat) + epsilon)

            p_current = p_current.at[jnp.array(opt_param_indices)].add(-step)

            elapsed = (time.perf_counter() - t0) * 1000.0

            if it % print_every == 0 or it == 1:
                print(f"  Iter {it:4d} | loss={loss_val:.6e} | |grad|={grad_norm:.6e} | p={np.asarray(p_current[jnp.array(opt_param_indices)])} | {elapsed:.1f} ms")

            if grad_norm < tol:
                converged = True
                break
                
        return {
            'p_opt': p_current,
            'n_iter': it,
            'converged': converged,
            'grad_norm_history': grad_norm_history,
            'loss_history': loss_history
        }
