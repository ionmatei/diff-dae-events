
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import time

# Import helper to create functions
import debug.verify_residual_gmres as gmres_impl

class DAEMatrixGradient:
    def __init__(self, dae_data):
        """
        Initialize the Matrix Gradient Computer.
        Args:
            dae_data: Dictionary containing DAE system specification
        """
        self.dae_data = dae_data

        # Create JAX functions
        self.funcs = gmres_impl.create_jax_functions(dae_data)
        self.f_fn, self.g_fn, self.h_fn, self.guard_fn, self.reinit_res_fn, self.reinit_vars, self.dims = self.funcs
        self.n_x, self.n_z, self.n_p = self.dims
        self.n_w = self.n_x + self.n_z
        
        # Initial state (fixed for now, or could be parameter)
        self.x0_start = jnp.array([s['start'] for s in dae_data['states']])

        # Cache for JIT-compiled gradient functions: structure_hash -> jit_fn
        self._kernel_cache = {}

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
        
        for i in range(num_seg):
            seg = sol.segments[i]
            n_points = len(seg.t)
            
            # Calculate Tau Grid (Normalized)
            t_start = seg.t[0]
            t_end = seg.t[-1]
            denom = t_end - t_start
            if denom < 1e-12: denom = 1.0
            tau = (seg.t - t_start) / denom
            grid_taus.append(tau)
            
            seg_start_idx = len(w_list)
            
            for k in range(n_points):
                w_list.extend(seg.x[k])
                w_list.extend(seg.z[k] if len(seg.z) > 0 else [])
                
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

        def loss_fn(W_flat, p, grid_taus, target_times, target_data, t_final_val, blend_sharpness=150.0, soft_interp=False):
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
            
            # Predict
            # Vectorized prediction over target_times
            
            def predict_at_t(t_q):
                y_accum = jnp.zeros(n_x)
                w_accum = 0.0
                
                # Iterate all segments
                for i in range(len(segments_t)):
                    ts = segments_t[i]
                    xs = segments_x[i]
                    t_start = ts[0]
                    t_end = ts[-1]
                    
                    # Window
                    lower = t_start if i == 0 else event_vals[i-1]
                    upper = t_end if i == len(segments_t)-1 else event_vals[i]
                    
                    mask = jax.nn.sigmoid(blend_sharpness * (t_q - lower)) * jax.nn.sigmoid(blend_sharpness * (upper - t_q))
                    
                    # Interpolate
                    t_clip = jnp.clip(t_q, t_start, t_end)
                    
                    if soft_interp:
                         # Implement a softer lookup if requested? 
                         # For now, standard linear interpolation is differentiable w.r.t grid values.
                         pass
                         
                    idx = jnp.searchsorted(ts, t_clip, side='right') - 1
                    idx = jnp.clip(idx, 0, len(ts)-2)
                    
                    t0_g, t1_g = ts[idx], ts[idx+1]
                    denom = t1_g - t0_g
                    denom = jnp.where(jnp.abs(denom) < 1e-12, 1e-12, denom)
                    s = (t_clip - t0_g) / denom
                    
                    val = xs[idx] * (1.0 - s) + xs[idx+1] * s
                    
                    y_accum += mask * val
                    w_accum += mask
                    
                return y_accum / (w_accum + 1e-8)
                
            predictions = jax.vmap(predict_at_t)(target_times)
            
            # Loss: MSE
            diff = predictions - target_data
            return jnp.mean(jnp.square(diff))

        def compute_grads(W_flat, p_val, grid_taus, target_times, target_data, t_final_val, blend_sharpness, soft_interp):
            # 1. Residual Jacobians
            J_W = jax.jacfwd(residual_fn, argnums=0)(W_flat, p_val, grid_taus, t_final_val)
            J_p = jax.jacfwd(residual_fn, argnums=1)(W_flat, p_val, grid_taus, t_final_val)
            
            # 2. Loss Gradients
            dL_dW = jax.grad(loss_fn, argnums=0)(W_flat, p_val, grid_taus, target_times, target_data, t_final_val, blend_sharpness, soft_interp)
            dL_dp = jax.grad(loss_fn, argnums=1)(W_flat, p_val, grid_taus, target_times, target_data, t_final_val, blend_sharpness, soft_interp)
            
            # 3. Adjoint Solve
            lam = jnp.linalg.solve(J_W.T, -dL_dW)
            
            # 4. Total Gradient
            total_grad = dL_dp + jnp.dot(lam, J_p)
            return total_grad

        # Compile
        jit_kernel = jax.jit(compute_grads, static_argnames=['soft_interp'])
        self._kernel_cache[struct_key] = jit_kernel
        return jit_kernel

    def compute_total_gradient(self, sol, p_val, target_times, target_data, blend_sharpness=150.0, soft_interp=False):
        # 1. Pack
        t_final = sol.segments[-1].t[-1] if sol.segments else 2.0 
        
        W_flat, structure, grid_taus_list = self.pack_solution(sol)
        grid_taus_tuple = tuple(grid_taus_list)
        
        # 2. Get Kernel
        kernel = self._get_gradient_kernel(structure)
        
        # 3. Compute
        grad_p = kernel(W_flat, p_val, grid_taus_tuple, target_times, target_data, t_final, blend_sharpness, soft_interp)
        
        return grad_p

    def optimize_adam(self, solver, p_init, opt_param_indices, target_times, target_data,
                      t_span, ncp, max_iter=150, tol=1e-8, step_size=0.01,
                      beta1=0.9, beta2=0.999, epsilon=1e-8,
                      blend_sharpness=150.0, print_every=10, soft_interp=False):
        
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
            total_grad = self.compute_total_gradient(sol, p_current, target_times, target_data, blend_sharpness, soft_interp=soft_interp)
            total_grad.block_until_ready()
            
            grad_opt = total_grad[jnp.array(opt_param_indices)]
            grad_norm = float(jnp.linalg.norm(grad_opt))
            
            # Loss (Cheap way: just extract from grad calculation? No, we need separate call or return it)
            # For now, just log gradient.
            loss_val = 0.0 # Placeholder or re-run prediction
            
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
                print(f"  Iter {it:4d} | |grad|={grad_norm:.6e} | p={np.asarray(p_current[jnp.array(opt_param_indices)])} | {elapsed:.1f} ms")

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
