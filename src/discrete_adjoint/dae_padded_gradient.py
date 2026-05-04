import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import numpy as np

# Import the padded sweep function
import re
from functools import partial

class DAEPaddedGradient:
    def __init__(self, dae_data, max_blocks=50, max_pts=500, max_targets=200,
                 downsample_segments=False, all_segments=False,
                 warmup_kernels=('total_grad',)):
        """
        Initialize the Padded Gradient Computer.

        Args:
            dae_data: Dictionary containing DAE system specification
            max_blocks: Maximum number of blocks (segments + events)
            max_pts: Maximum points per segment
            max_targets: Maximum number of target time-points (avoids JIT recompilation)
            downsample_segments: If True, downsample segments exceeding max_pts
            all_segments: If True (and downsample_segments=True), resample ALL segments to max_pts
            warmup_kernels: Which JIT kernels to compile eagerly during construction.
                Subset of {'sweep', 'loss_grad', 'total_grad'}. The default
                ('total_grad',) compiles only the kernel used by `optimize_adam`,
                which subsumes the work of the other two — compiling all three
                duplicates XLA effort. Pass () to skip warmup entirely (each
                kernel will compile lazily on first call). Pass 'all' or
                ('sweep', 'loss_grad', 'total_grad') for the legacy behavior.
        """
        self.dae_data = dae_data
        self.max_blocks = max_blocks
        self.max_pts = max_pts
        self.max_targets = max_targets
        self.downsample_segments = downsample_segments
        self.all_segments = all_segments
        if warmup_kernels == 'all':
            warmup_kernels = ('sweep', 'loss_grad', 'total_grad')
        self.warmup_kernels = tuple(warmup_kernels) if warmup_kernels else ()

        # Create JAX functions
        self.funcs = self._create_jax_functions(dae_data)
        self.f_fn, self.g_fn, self.h_fn, self.guard_fn, self.reinit_res_fn, self.reinit_vars, self.reinit_event_owner, self.reinit_state_target, self.dims = self.funcs

        # Pre-allocate persistent host-side buffers to avoid repeated np.zeros()
        n_x, n_z, n_p = self.dims
        n_w = n_x + n_z
        self._host_W = np.zeros((max_blocks, max_pts, n_w), dtype=np.float64)
        self._host_TS = np.zeros((max_blocks, max_pts), dtype=np.float64)
        self._host_block_types = np.zeros(max_blocks, dtype=np.int32)
        self._host_block_indices = np.zeros((max_blocks, 2), dtype=np.int32)
        self._host_block_param = np.zeros(max_blocks, dtype=np.float64)
        self._host_dL = np.zeros((max_blocks, max_pts, n_w), dtype=np.float64)
        self._host_target_times = np.empty(max_targets, dtype=np.float64)
        self._host_target_data = np.empty((max_targets, n_x), dtype=np.float64)

        # JIT Compile the sweep function — close over funcs/dims/max_blocks
        # so they become compile-time constants (not static_argnames lookups)
        print(f"JAX Device for Gradient Computation: {jax.devices()[0]}")
        print(f"Compiling Padded JIT solver (Max Blocks: {max_blocks}, Max Pts: {max_pts})...")
        _funcs = self.funcs
        _dims = self.dims
        _max_blocks = max_blocks

        def _sweep_closed(W, TS, p, bt, bi, bp, J_wn, J_wc, J_p, dR_dtc, dR_dtn, dW, dp):
            return DAEPaddedGradient._compute_adjoint_sweep_padded(
                W, TS, p, bt, bi, bp, J_wn, J_wc, J_p, dR_dtc, dR_dtn, dW, dp,
                _funcs, _dims, _max_blocks
            )
        self.jit_sweep = jax.jit(_sweep_closed)

        # JIT Compile the Loss Gradient (for individual use if needed)
        self.jit_loss_grad = jax.jit(
            jax.grad(self._loss_fn_padded, argnums=[0, 1, 5]),
            static_argnames=['n_x', 'adaptive_horizon']
        )

        # JIT Compile the Unified Total Gradient Kernel
        print("Compiling Unified Gradient JIT kernel...")
        def _total_grad_closed(W_p, TS_p, p_val, b_types, b_indices, b_param,
                               target_times, target_data, n_targets, t_final, blend_sharpness,
                               adaptive_horizon=False):
            return DAEPaddedGradient._total_gradient_kernel(
                W_p, TS_p, p_val, b_types, b_indices, b_param,
                target_times, target_data, n_targets, t_final, blend_sharpness,
                _funcs, _dims, _max_blocks, adaptive_horizon=adaptive_horizon
            )
        self.jit_total_grad = jax.jit(_total_grad_closed, static_argnames=['adaptive_horizon'])
        
        # Trigger compilation with dummy data
        # (Optional, but ensures first call is fast)
        self._warmup()

    def _warmup(self):
        """Trigger JIT compilation for the kernels listed in self.warmup_kernels.

        `optimize_adam` only calls `jit_total_grad`, which already subsumes
        the value_and_grad and adjoint-sweep computations — so compiling
        `jit_loss_grad` and `jit_sweep` separately duplicates XLA work.
        Default: warm up only `jit_total_grad`. Set warmup_kernels='all' or
        () in the constructor to override.
        """
        import time as _time

        if not self.warmup_kernels:
            print("Compilation deferred (warmup_kernels=()) - first call will compile.")
            return

        n_x, n_z, n_p = self.dims
        n_w = n_x + n_z

        # Dummy inputs (only allocate what's actually needed)
        W_p = jnp.zeros((self.max_blocks, self.max_pts, n_w))
        TS_p = jnp.zeros((self.max_blocks, self.max_pts))
        b_types = jnp.zeros(self.max_blocks, dtype=jnp.int32)
        b_indices = jnp.zeros((self.max_blocks, 2), dtype=jnp.int32)
        b_param = jnp.zeros(self.max_blocks)
        p_val = jnp.zeros(n_p)
        target_times = jnp.zeros(self.max_targets)
        target_data = jnp.zeros((self.max_targets, n_x))
        n_targets = jnp.int32(10)

        if 'sweep' in self.warmup_kernels:
            t0 = _time.perf_counter()
            dL_p = jnp.zeros((self.max_blocks, self.max_pts, n_w))
            dL_dp = jnp.zeros(n_p)
            J_wn = jnp.zeros((self.max_blocks, self.max_pts - 1, n_w, n_w))
            J_wc = jnp.zeros((self.max_blocks, self.max_pts - 1, n_w, n_w))
            J_p_jac = jnp.zeros((self.max_blocks, self.max_pts - 1, n_w, n_p))
            dR_dtc = jnp.zeros((self.max_blocks, self.max_pts - 1, n_w))
            dR_dtn = jnp.zeros((self.max_blocks, self.max_pts - 1, n_w))
            _ = self.jit_sweep(
                W_p, TS_p, p_val,
                b_types, b_indices, b_param,
                J_wn, J_wc, J_p_jac, dR_dtc, dR_dtn, dL_p, dL_dp,
            )
            print(f"  jit_sweep compiled in {_time.perf_counter() - t0:.2f}s")

        if 'loss_grad' in self.warmup_kernels:
            t0 = _time.perf_counter()
            _ = self.jit_loss_grad(
                W_p, p_val, TS_p, b_types, b_indices, b_param,
                target_times, target_data, n_targets, 1.0, 150.0, n_x,
            )
            print(f"  jit_loss_grad compiled in {_time.perf_counter() - t0:.2f}s")

        if 'total_grad' in self.warmup_kernels:
            t0 = _time.perf_counter()
            _ = self.jit_total_grad(
                W_p, TS_p, p_val, b_types, b_indices, b_param,
                target_times, target_data, n_targets, 1.0, 150.0,
            )
            print(f"  jit_total_grad compiled in {_time.perf_counter() - t0:.2f}s")

        print("Compilation complete.")

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
                
        # Compile Guard & Reinit (supports list-valued reinit per event)
        when_clauses = dae_data.get('when', [])
        guard_exprs = []
        reinit_exprs = []        # flat list of all reinit expressions
        reinit_event_owner = []  # reinit_event_owner[k] = event index owning expression k
        reinit_state_target = [] # reinit_state_target[k] = state index for expression k
        reinit_vars = []         # kept for legacy (flat list of (type, idx))

        for ev_i, wc in enumerate(when_clauses):
            cond = wc['condition']
            if '<' in cond: lhs, rhs = cond.split('<', 1)
            elif '>' in cond: lhs, rhs = cond.split('>', 1)
            else: lhs, rhs = cond.split('=', 1)
            guard_exprs.append(f"({lhs}) - ({rhs})")

            reinit_raw = wc['reinit']
            reinit_list = reinit_raw if isinstance(reinit_raw, list) else [reinit_raw]

            for reinit_str in reinit_list:
                if '=' in reinit_str:
                    rl, rr = reinit_str.split('=', 1)
                    raw_expr = f"({rl}) - ({rr})"
                    for i, name in enumerate(state_names):
                        if re.search(r'\b' + re.escape(name) + r'\b', rl):
                            reinit_vars.append(('state', i))
                            reinit_state_target.append(i)
                            break
                else:
                    raw_expr = reinit_str
                reinit_exprs.append(raw_expr)
                reinit_event_owner.append(ev_i)
        
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
            # Bare math-function references in spec expressions
            # (e.g. `tanh(...)`, `exp(...)`) must resolve to jax ops at
            # eval time. Without this map, a Cauer-style g equation like
            # `... + tanh(50.0 * (time - t0)) * ...` raises NameError.
            local_scope = {
                'jnp': jnp,
                'exp': jnp.exp, 'log': jnp.log, 'log10': jnp.log10,
                'sqrt': jnp.sqrt, 'abs': jnp.abs,
                'sin': jnp.sin, 'cos': jnp.cos, 'tan': jnp.tan,
                'asin': jnp.arcsin, 'acos': jnp.arccos, 'atan': jnp.arctan,
                'sinh': jnp.sinh, 'cosh': jnp.cosh, 'tanh': jnp.tanh,
                'sign': jnp.sign, 'floor': jnp.floor, 'ceil': jnp.ceil,
                'min': jnp.minimum, 'max': jnp.maximum,
                'sigmoid': jax.nn.sigmoid,
            }
            exec(code, local_scope)
            return local_scope['func']

        f_fn = compile_to_jax(f_exprs, False)
        g_fn = compile_to_jax(g_exprs, False)
        guard_fn = compile_to_jax(guard_exprs, False)
        reinit_res_fn = compile_to_jax(reinit_exprs, True)
        h_fn = lambda t, x, z, p: x if use_default_h else compile_to_jax(h_exprs, False)(t, x, z, p)

        return f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, tuple(reinit_vars), tuple(reinit_event_owner), tuple(reinit_state_target), (len(state_names), len(alg_names), len(param_names))

    @staticmethod
    def _downsample_segment(t_seg, y_seg, max_pts):
        """Downsample a segment to max_pts points via linear interpolation.

        Preserves first and last time instants. Interior points are
        uniformly spaced. State values are linearly interpolated.

        Only called when t_seg.shape[0] > max_pts.
        """
        t_new = np.linspace(t_seg[0], t_seg[-1], max_pts)
        y_new = np.empty((max_pts, y_seg.shape[1]), dtype=y_seg.dtype)
        for col in range(y_seg.shape[1]):
            y_new[:, col] = np.interp(t_new, t_seg, y_seg[:, col])
        return t_new, y_new

    def _pad_problem_data(self, ts_all, ys_all, event_infos):
        """
        Converts dynamic trajectory data into padded fixed-size arrays.
        Reuses pre-allocated host buffers to avoid repeated allocations.
        """
        W_padded = self._host_W
        TS_padded = self._host_TS
        block_types = self._host_block_types
        block_indices = self._host_block_indices
        block_param = self._host_block_param
        dL_padded = self._host_dL

        curr_blk = 0
        n_segs = len(ts_all)
        n_evs = len(event_infos)
        n_used = n_segs + n_evs

        # Small 1-D arrays: zero fully (cheap, max_blocks elements)
        block_types.fill(0)
        block_indices.fill(0)
        block_param.fill(0.0)
        # Large 3-D buffers: zero only the rows that will be written.
        # Stale data beyond n_used is harmless — block_types==0 masks it out.
        W_padded[:n_used] = 0.0
        TS_padded[:n_used] = 0.0
        # dL_padded: always zero — initialized at construction, never written in the loop
        
        for i in range(n_segs):
            # 1. Segment
            if curr_blk >= self.max_blocks: 
                raise ValueError(f"Exceeded max_blocks ({self.max_blocks})")
            
            n_pts = ts_all[i].shape[0]
            if n_pts > self.max_pts: 
                raise ValueError(f"Segment {i} points {n_pts} > max_pts {self.max_pts}")
                
            W_padded[curr_blk, :n_pts, :] = ys_all[i]
            # Store normalized tau (0..1) instead of absolute times.
            # Actual times are reconstructed inside JIT kernels from
            # tau and the event-time bounds in b_param, so that
            # gradients propagate through event times.
            t0_seg = ts_all[i][0]
            t1_seg = ts_all[i][-1]
            denom = t1_seg - t0_seg
            if abs(denom) < 1e-12:
                denom = 1.0
            TS_padded[curr_blk, :n_pts] = (ts_all[i] - t0_seg) / denom
            block_types[curr_blk] = 1 # Segment
            block_indices[curr_blk] = [0, n_pts]
            curr_blk += 1

            # 2. Event
            if i < n_evs:
                if curr_blk >= self.max_blocks:
                    raise ValueError(f"Exceeded max_blocks ({self.max_blocks}) during event")

                ev_t, ev_idx = event_infos[i]
                block_types[curr_blk] = 2 # Event
                block_indices[curr_blk] = [ev_idx, 1]
                block_param[curr_blk] = ev_t
                TS_padded[curr_blk, 0] = 0.0  # not used for events
                curr_blk += 1
                
        return W_padded, TS_padded, block_types, block_indices, block_param, dL_padded

    def _reconstruct_abs_times(self, TS_tau, b_types, b_param, sol):
        """Reconstruct absolute times from normalized tau grid (numpy, Python-side).

        Used before passing to the adjoint sweep which expects actual times.
        """
        TS_abs = np.array(TS_tau, copy=True)
        t_final = sol.segments[-1].t[-1] if sol.segments else 0.0
        for i in range(self.max_blocks):
            if b_types[i] == 0:
                break
            if b_types[i] == 1:  # segment
                lower = b_param[i - 1] if (i > 0 and b_types[i - 1] == 2) else 0.0
                upper = b_param[i + 1] if (i + 1 < self.max_blocks and b_types[i + 1] == 2) else t_final
                n_pts = int(np.sum(TS_tau[i] > 0)) + 1  # tau[0]==0 is valid
                # Actually use block_indices for accurate count — but we don't
                # have it here.  Simpler: reconstruct the whole row.
                TS_abs[i] = lower + TS_tau[i] * (upper - lower)
        return TS_abs

    def _pad_targets(self, target_times, target_data):
        """Pad target arrays to fixed (max_targets,) shape for stable JIT compilation."""
        n_targets = target_times.shape[0]
        if n_targets > self.max_targets:
            raise ValueError(f"Number of targets {n_targets} exceeds max_targets ({self.max_targets})")
        tt = self._host_target_times; tt.fill(0.0)
        td = self._host_target_data;  td.fill(0.0)
        tt[:n_targets] = np.asarray(target_times)
        td[:n_targets] = np.asarray(target_data)
        return tt, td, n_targets

    @staticmethod
    @partial(jax.jit, static_argnames=['funcs', 'dims'])
    def _solve_event_system_affine_jit(t_event, w_prev_end, w_post_start, adjoint_state, dL_t, mesh_sens_prev, funcs, dims, p_opt, event_idx):
        """Computes Affine coefficients for Event Multipliers."""
        f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, reinit_event_owner, reinit_state_target, _ = funcs
        n_x, n_z, n_p = dims
        reinit_event_owner_arr = jnp.array(reinit_event_owner, dtype=jnp.int32)
        reinit_state_target_arr = jnp.array(reinit_state_target, dtype=jnp.int32)
        n_total_reinits = len(reinit_event_owner)

        def event_res_fn(t, xp, xpr, p):
            x_p, z_p = xp[:n_x], xp[n_x:]
            x_r, z_r = xpr[:n_x], xpr[n_x:]
            # Guard for the specific event that fired
            r_g = jax.lax.dynamic_slice(guard_fn(t, x_r, z_r, p), (event_idx,), (1,))
            # All reinit residuals
            all_reinits = reinit_res_fn(t, x_p, z_p, x_r, z_r, p)
            # Start with continuity for all states
            continuity = x_p[:n_x] - x_r[:n_x]

            # Vectorized scatter of reinit residuals into their target state
            # positions for the currently firing event. Replaces a Python
            # `for k in range(n_total_reinits)` loop that was being fully
            # unrolled into the JIT trace — for large N the unrolled trace
            # was the dominant compile-time cost (3.9× more reinits at
            # N=15 vs N=7, plus jacrev cotangent multiplier).
            #
            # Semantics: per (firing) event, each reinit k targets one
            # state index `reinit_state_target_arr[k]`, and within one
            # event a given state index is targeted at most once. So the
            # scatter is one-to-one for the matching `is_mine` mask.
            is_mine = (reinit_event_owner_arr == event_idx).astype(all_reinits.dtype)
            overrides = jnp.zeros(n_x, dtype=all_reinits.dtype).at[
                reinit_state_target_arr
            ].add(is_mine * all_reinits)
            override_count = jnp.zeros(n_x, dtype=all_reinits.dtype).at[
                reinit_state_target_arr
            ].add(is_mine)
            state_res = jnp.where(override_count > 0, overrides, continuity)

            r_a = g_fn(t, x_p, z_p, p) if n_z > 0 else jnp.array([])
            return jnp.concatenate([r_g, state_res, r_a])

        # Combined Jacobian computation (Single linearization)
        # Using jacrev allows efficient computation wrt multiple args in n_res passes
        (J_t_partial, J_xp, J_xpr) = jax.jacrev(
            lambda t, xp, xpr: event_res_fn(t, xp, xpr, p_opt),
            argnums=(0, 1, 2)
        )(t_event, w_post_start, w_prev_end)
        
        J_te_total = J_t_partial.reshape((-1, 1))

        # --- Linear Algebra Optimization (SVD) ---
        # Solve J_xp.T @ mu = -adjoint_state
        # and find left null vector of J_xp (null space of J_xp.T)
        
        # J_xp.T is shape (n_vars, n_res)
        # We need full_matrices=True to access the null-space vector in Vt
        U, S, Vt = jnp.linalg.svd(J_xp.T, full_matrices=True)
        
        # 1. Null Vector: Last row of Vt corresponds to the null space of J_xp.T
        v_null = Vt[-1, :]
        
        # 2. Particular Solution mu_0 (Minimum Norm) via Pseudo-Inverse
        # mu_0 = Vt.T @ inv(S) @ U.T @ b
        n_vars = S.shape[0]
        inv_S = jnp.where(S > 1e-12, 1.0 / S, 0.0)
        
        ut_b = jnp.dot(U.T, -adjoint_state)         # (n_vars,)
        scaled_ut_b = ut_b * inv_S                  # (n_vars,)
        mu_0 = jnp.dot(Vt[:n_vars, :].T, scaled_ut_b) # (n_res,)
        
        t_e_rhs_base = dL_t + mesh_sens_prev + jnp.dot(mu_0, J_te_total.flatten())
        t_e_slope    = jnp.dot(v_null, J_te_total.flatten())
        
        load_prev_mu0 = J_xpr.T @ mu_0
        load_prev_v   = J_xpr.T @ v_null
        
        _, vjp_p = jax.vjp(lambda p: event_res_fn(t_event, w_post_start, w_prev_end, p), p_opt)
        gp_mu0 = vjp_p(mu_0)[0]
        gp_v   = vjp_p(v_null)[0]
        
        return (load_prev_mu0, load_prev_v, t_e_rhs_base, t_e_slope, gp_mu0, gp_v, mu_0, v_null)

    @staticmethod
    def _run_segment_backward_sweep(ts, dL_nodes, load_at_end, n_pts_valid,
                                    J_wn_nodes, J_wc_nodes, J_p_nodes,
                                    dR_dtc_nodes, dR_dtn_nodes):
        """Backward sweep using precomputed Jacobians — no AD in the loop."""
        dL_terminal = dL_nodes[n_pts_valid - 1]
        init_load = dL_terminal + load_at_end

        scan_ts_c = ts[:-1][::-1]
        scan_ts_n = ts[1:][::-1]
        scan_dLs = dL_nodes[:-1][::-1]
        scan_J_wn = J_wn_nodes[::-1]
        scan_J_wc = J_wc_nodes[::-1]
        scan_J_p = J_p_nodes[::-1]
        scan_dR_dtc = dR_dtc_nodes[::-1]
        scan_dR_dtn = dR_dtn_nodes[::-1]

        max_steps = ts.shape[0] - 1
        indices_reversed = jnp.arange(max_steps)
        mask_valid = indices_reversed >= (max_steps + 1 - n_pts_valid)

        t0 = ts[0]
        tf = ts[n_pts_valid - 1]
        duration = tf - t0
        dur_inv = jnp.where(jnp.abs(duration) < 1e-12, 0.0, 1.0 / duration)

        n_p = J_p_nodes.shape[-1]
        init_carry = (init_load, 0.0, 0.0, 0.0)

        def scan_body_seg(carry, inputs):
            (dL_at_n, J_wn_step, J_wc_step, J_p_step,
             dR_dtc_step, dR_dtn_step, t_c, t_n, is_valid) = inputs
            load_n, _, gTs, gTe = carry

            def branch_valid(_):
                lam = jnp.linalg.solve(J_wn_step.T, -load_n)
                partial_load = J_wc_step.T @ lam
                grad_p_step = J_p_step.T @ lam
                next_load = partial_load + dL_at_n

                d_tc = jnp.dot(dR_dtc_step, lam)
                d_tn = jnp.dot(dR_dtn_step, lam)
                tau_c = (t_c - t0) * dur_inv
                tau_n = (t_n - t0) * dur_inv
                inc_start = d_tc * (1.0 - tau_c) + d_tn * (1.0 - tau_n)
                inc_end = d_tc * tau_c + d_tn * tau_n

                return (next_load, 0.0, gTs + inc_start, gTe + inc_end), grad_p_step

            def branch_invalid(_):
                return (load_n, 0.0, gTs, gTe), jnp.zeros(n_p)

            return jax.lax.cond(is_valid, branch_valid, branch_invalid, operand=None)

        inputs = (scan_dLs, scan_J_wn, scan_J_wc, scan_J_p,
                  scan_dR_dtc, scan_dR_dtn, scan_ts_c, scan_ts_n, mask_valid)
        final_carry, grads_p = jax.lax.scan(scan_body_seg, init_carry, inputs)

        sens_w0 = final_carry[0]
        total_gp = jnp.sum(grads_p, axis=0)
        total_gTs = final_carry[2]
        total_gTe = final_carry[3]

        return sens_w0, total_gp, total_gTs, total_gTe

    @staticmethod
    def _run_segment_backward_sweep_dual(ts, dL_nodes_pair, load_at_end_pair, n_pts_valid,
                                         J_wn_nodes, J_wc_nodes, J_p_nodes,
                                         dR_dtc_nodes, dR_dtn_nodes):
        """Dual backward sweep — processes 2 load vectors in one pass, sharing LU factorization.

        Instead of vmapping the single sweep over 2 load/dL pairs (which runs the
        scan twice), this fuses both into one scan with batched linear solves.

        Args:
            dL_nodes_pair: (2, max_pts, n_w)
            load_at_end_pair: (2, n_w)
            Others: same as _run_segment_backward_sweep.

        Returns:
            sens_w0_pair (2, n_w), total_gp_pair (2, n_p),
            total_gTs_pair (2,), total_gTe_pair (2,).
        """
        dL_terminal_pair = dL_nodes_pair[:, n_pts_valid - 1, :]   # (2, n_w)
        init_load_pair = dL_terminal_pair + load_at_end_pair       # (2, n_w)

        scan_ts_c = ts[:-1][::-1]
        scan_ts_n = ts[1:][::-1]
        scan_dLs_pair = dL_nodes_pair[:, :-1, :][:, ::-1, :]      # (2, max_steps, n_w)
        scan_dLs_pair_t = jnp.transpose(scan_dLs_pair, (1, 0, 2)) # (max_steps, 2, n_w)
        scan_J_wn = J_wn_nodes[::-1]
        scan_J_wc = J_wc_nodes[::-1]
        scan_J_p = J_p_nodes[::-1]
        scan_dR_dtc = dR_dtc_nodes[::-1]
        scan_dR_dtn = dR_dtn_nodes[::-1]

        max_steps = ts.shape[0] - 1
        indices_reversed = jnp.arange(max_steps)
        mask_valid = indices_reversed >= (max_steps + 1 - n_pts_valid)

        t0 = ts[0]
        tf = ts[n_pts_valid - 1]
        duration = tf - t0
        dur_inv = jnp.where(jnp.abs(duration) < 1e-12, 0.0, 1.0 / duration)

        n_p = J_p_nodes.shape[-1]
        init_carry = (init_load_pair, 0.0, jnp.zeros(2), jnp.zeros(2))

        def scan_body_seg(carry, inputs):
            (dL_at_n_pair, J_wn_step, J_wc_step, J_p_step,
             dR_dtc_step, dR_dtn_step, t_c, t_n, is_valid) = inputs
            load_n_pair, _, gTs_pair, gTe_pair = carry

            def branch_valid(_):
                # Single factorization, 2 back-substitutions
                lam_pair = jnp.linalg.solve(J_wn_step.T, -load_n_pair.T)  # (n_w, 2)

                partial_load_pair = (J_wc_step.T @ lam_pair).T   # (2, n_w)
                grad_p_pair = (J_p_step.T @ lam_pair).T          # (2, n_p)
                next_load_pair = partial_load_pair + dL_at_n_pair # (2, n_w)

                d_tc_pair = dR_dtc_step @ lam_pair  # (2,)
                d_tn_pair = dR_dtn_step @ lam_pair  # (2,)
                tau_c = (t_c - t0) * dur_inv
                tau_n = (t_n - t0) * dur_inv
                inc_start_pair = d_tc_pair * (1.0 - tau_c) + d_tn_pair * (1.0 - tau_n)
                inc_end_pair = d_tc_pair * tau_c + d_tn_pair * tau_n

                return (next_load_pair, 0.0, gTs_pair + inc_start_pair, gTe_pair + inc_end_pair), grad_p_pair

            def branch_invalid(_):
                return (load_n_pair, 0.0, gTs_pair, gTe_pair), jnp.zeros((2, n_p))

            return jax.lax.cond(is_valid, branch_valid, branch_invalid, operand=None)

        inputs = (scan_dLs_pair_t, scan_J_wn, scan_J_wc, scan_J_p,
                  scan_dR_dtc, scan_dR_dtn, scan_ts_c, scan_ts_n, mask_valid)
        final_carry, grads_p_pair = jax.lax.scan(scan_body_seg, init_carry, inputs)

        sens_w0_pair = final_carry[0]                     # (2, n_w)
        total_gp_pair = jnp.sum(grads_p_pair, axis=0)     # (2, n_p)
        total_gTs_pair = final_carry[2]                    # (2,)
        total_gTe_pair = final_carry[3]                    # (2,)

        return sens_w0_pair, total_gp_pair, total_gTs_pair, total_gTe_pair

    @staticmethod
    @partial(jax.jit, static_argnames=['funcs', 'dims', 'max_blocks'])
    def _compute_adjoint_sweep_padded(
        W_padded, TS_padded, p_opt,
        block_types, block_indices, block_param,
        J_wn_padded, J_wc_padded, J_p_padded, dR_dtc_padded, dR_dtn_padded,
        dL_padded, dL_dp,
        funcs, dims, max_blocks,
        dL_db_param=None
    ):
        f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, reinit_event_owner, reinit_state_target, dims = funcs
        n_x, n_z, n_p = dims

        init_adjoint = jnp.zeros(n_x + n_z)
        init_mesh_sens = 0.0
        init_grad_p = dL_dp
        n_mu_max = n_x + n_z + 1

        init_pending = (
            0.0, jnp.zeros(n_mu_max), jnp.zeros(n_mu_max),
            jnp.zeros(n_x + n_z), jnp.zeros(n_x + n_z),
            0.0, 0.0, jnp.zeros(n_p), jnp.zeros(n_p)
        )
        init_carry = (init_adjoint, init_mesh_sens, init_grad_p, init_pending)
        scan_indices = jnp.arange(max_blocks)[::-1]

        def scan_body(carry, block_idx):
            adj_load, mesh_sens, grad_p, pending = carry
            b_type = block_types[block_idx]
            T_block = TS_padded[block_idx]
            J_wn_block = J_wn_padded[block_idx]
            J_wc_block = J_wc_padded[block_idx]
            J_p_block = J_p_padded[block_idx]
            dR_dtc_block = dR_dtc_padded[block_idx]
            dR_dtn_block = dR_dtn_padded[block_idx]
            dL_block = dL_padded[block_idx]
            n_pts_valid = block_indices[block_idx, 1]

            def case_pad(c_args):
                return c_args[0], c_args[1], c_args[2], c_args[3]

            def case_segment(c_args):
                curr_adj, curr_mesh, curr_gp, curr_pend = c_args
                pending_active = curr_pend[0]

                def branch_joint_solve(args):
                    p_adj, p_mesh, p_gp, p_pend = args
                    _, mu_0, v_null, lp_mu0, lp_v, te_rhs, te_slope, gp_mu0, gp_v = p_pend
                    dL_zeros = jnp.zeros_like(dL_block)
                    dL_stack = jnp.stack([dL_block, dL_zeros])   # (2, max_pts, n_w)
                    load_stack = jnp.stack([lp_mu0, lp_v])       # (2, n_w)
                    sens_w0_pair, grad_p_pair, grad_ts_pair, grad_te_pair = \
                        DAEPaddedGradient._run_segment_backward_sweep_dual(
                            T_block, dL_stack, load_stack, n_pts_valid,
                            J_wn_block, J_wc_block, J_p_block, dR_dtc_block, dR_dtn_block
                        )
                    grad_te_mu0 = grad_te_pair[0]
                    grad_te_v   = grad_te_pair[1]
                    denom = te_slope + grad_te_v
                    c_val = -(te_rhs + grad_te_mu0) / (denom + 1e-12)
                    total_adj = sens_w0_pair[0] + c_val * sens_w0_pair[1]
                    total_gp_seg = grad_p_pair[0] + c_val * grad_p_pair[1]
                    inc_gp = gp_mu0 + c_val * gp_v + total_gp_seg
                    new_mesh = grad_ts_pair[0] + c_val * grad_ts_pair[1]
                    new_pend = init_pending
                    return total_adj, new_mesh, p_gp + inc_gp, new_pend

                def branch_standard_solve(args):
                    p_adj, p_mesh, p_gp, p_pend = args
                    sens_w0, grad_p_seg, grad_ts, grad_te = DAEPaddedGradient._run_segment_backward_sweep(
                        T_block, dL_block, p_adj, n_pts_valid,
                        J_wn_block, J_wc_block, J_p_block, dR_dtc_block, dR_dtn_block
                    )
                    return sens_w0, grad_ts, p_gp + grad_p_seg, p_pend

                return jax.lax.cond(pending_active > 0.5, branch_joint_solve, branch_standard_solve, c_args)

            def case_event(c_args):
                 curr_adj, curr_mesh, curr_gp, curr_pend = c_args
                 prev_block_idx = block_idx - 1
                 next_block_idx = block_idx + 1
                 w_pre_seg = W_padded[prev_block_idx]
                 n_pts_pre = block_indices[prev_block_idx, 1]
                 w_prev_end = w_pre_seg[n_pts_pre - 1]
                 w_post_seg = W_padded[next_block_idx]
                 w_post_start = w_post_seg[0]
                 t_event = block_param[block_idx]
                 if dL_db_param is not None:
                     dL_t = dL_db_param[block_idx]
                 else:
                     dL_t = dL_block[0, 0]

                 ev_idx = block_indices[block_idx, 0]
                 tuple_res = DAEPaddedGradient._solve_event_system_affine_jit(
                     t_event, w_prev_end, w_post_start, curr_adj, dL_t, curr_mesh, funcs, dims, p_opt, ev_idx
                 )
                 lp_mu0, lp_v, rhs, slope, gp_mu0, gp_v, mu_0, v_null = tuple_res
                 new_pending = (1.0, mu_0, v_null, lp_mu0, lp_v, rhs, slope, gp_mu0, gp_v)
                 zeros_adj = jnp.zeros_like(curr_adj)
                 return zeros_adj, curr_mesh, curr_gp, new_pending

            new_carry = jax.lax.switch(b_type, [case_pad, case_segment, case_event], carry)
            return new_carry, None

        final_carry, _ = jax.lax.scan(scan_body, init_carry, scan_indices)
        return final_carry[2]

    @staticmethod
    def _total_gradient_kernel(W_p, TS_p, p_val, b_types, b_indices, b_param, target_times, target_data, n_targets, t_final, blend_sharpness, funcs, dims, max_blocks, adaptive_horizon=False):
        """Unified JIT kernel: Loss Differentiation + Adjoint Sweep."""
        n_x, n_z, n_p = dims

        # 1. Compute Loss and Gradients simultaneously
        #    TS_p contains normalized tau; _loss_fn_padded → _predict_trajectory_padded_kernel
        #    reconstructs actual times from tau + b_param inside the differentiable graph.
        (loss_val, (dL_dW_p, dL_dp, dL_db_param)) = jax.value_and_grad(
            DAEPaddedGradient._loss_fn_padded,
            argnums=[0, 1, 5]
        )(
            W_p, p_val, TS_p, b_types, b_indices, b_param,
            target_times, target_data, n_targets, t_final, blend_sharpness, n_x,
            adaptive_horizon=adaptive_horizon
        )

        # 2. Reconstruct absolute times for adjoint sweep (not differentiated)
        _blk = jnp.arange(max_blocks)
        _prev = jnp.maximum(_blk - 1, 0)
        _next = jnp.minimum(_blk + 1, max_blocks - 1)
        _lower = jnp.where(
            jnp.logical_and(_blk > 0, b_types[_prev] == 2),
            b_param[_prev], 0.0)
        _upper = jnp.where(
            b_types[_next] == 2, b_param[_next], t_final)
        TS_abs = _lower[:, None] + TS_p * (_upper - _lower)[:, None]

        # 3. Precompute all Jacobians (dense) outside the scan
        J_wn, J_wc, J_p_jac, dR_dtc, dR_dtn = DAEPaddedGradient._precompute_jacobians(
            W_p, TS_abs, b_types, b_indices, p_val, funcs, dims
        )

        # 4. Call Adjoint Sweep
        total_grad = DAEPaddedGradient._compute_adjoint_sweep_padded(
            W_p, TS_abs, p_val,
            b_types, b_indices, b_param,
            J_wn, J_wc, J_p_jac, dR_dtc, dR_dtn,
            dL_dW_p, dL_dp,
            funcs, dims, max_blocks,
            dL_db_param=dL_db_param
        )
        return loss_val, total_grad

    @staticmethod
    def _predict_trajectory_padded_kernel(W_p, TS_p, b_types, b_indices, b_param, target_times, t_final, blend_sharpness, n_x):
        """JITable version of trajectory prediction using padded arrays.

        TS_p contains normalized tau values (0..1).  Actual times are
        reconstructed as  lower + tau * (upper - lower)  so that gradients
        flow through the event times stored in b_param.
        """
        max_blocks = W_p.shape[0]
        block_idx = jnp.arange(max_blocks)

        # Precompute per-block constants (vectorized, not per-target)
        is_seg = (b_types == 1).astype(jnp.float64)                   # (max_blocks,)
        prev_idx = jnp.maximum(block_idx - 1, 0)
        next_idx = jnp.minimum(block_idx + 1, max_blocks - 1)
        lower = jnp.where(
            jnp.logical_and(block_idx > 0, b_types[prev_idx] == 2),
            b_param[prev_idx], 0.0)
        upper = jnp.where(
            b_types[next_idx] == 2,
            b_param[next_idx], t_final)

        # Reconstruct actual times from tau and event-time bounds
        seg_span = (upper - lower)[:, None]                            # (max_blocks, 1)
        TS_actual = lower[:, None] + TS_p * seg_span                   # (max_blocks, max_pts)

        n_pts = b_indices[:, 1]                                        # (max_blocks,)
        n_pts_safe = jnp.maximum(n_pts, 2)                             # avoid 0-index issues
        t_starts = TS_actual[:, 0]                                     # (max_blocks,)
        t_ends = TS_actual[block_idx, n_pts_safe - 1]                  # (max_blocks,)
        xs_all = W_p[:, :, :n_x]                                       # (max_blocks, max_pts, n_x)

        def _interp_block(ts_row, xs_row, t_c, np_safe):
            """O(1) piecewise-linear interpolation via soft fractional index.

            Computes a continuous index from time span, then linearly
            interpolates between the two bracketing nodes.  Differentiable
            w.r.t. t_c and xs_row (gradients through knot times flow via
            the outer sigmoid mask on event-time bounds).
            """
            t_first = ts_row[0]
            t_last = ts_row[jnp.maximum(np_safe - 1, 0)]
            span = jnp.maximum(t_last - t_first, 1e-12)
            frac = jnp.clip((t_c - t_first) / span, 0.0, 1.0)
            frac_idx = frac * (np_safe - 1)
            idx_lo = jnp.floor(frac_idx).astype(jnp.int32)
            idx_lo = jnp.clip(idx_lo, 0, ts_row.shape[0] - 2)
            alpha = frac_idx - idx_lo.astype(jnp.float64)
            return xs_row[idx_lo] * (1.0 - alpha) + xs_row[idx_lo + 1] * alpha

        def predict_single(t_q):
            # Masks for all blocks at once
            mask = (jax.nn.sigmoid(blend_sharpness * (t_q - lower)) *
                    jax.nn.sigmoid(blend_sharpness * (upper - t_q)) *
                    is_seg)                                             # (max_blocks,)

            t_clip = jnp.clip(t_q, t_starts, t_ends)                   # (max_blocks,)

            # vmap interpolation over blocks
            vals = jax.vmap(_interp_block)(
                TS_actual, xs_all, t_clip, n_pts_safe
            )                                                           # (max_blocks, n_x)

            y_accum = jnp.sum(mask[:, None] * vals, axis=0)            # (n_x,)
            w_accum = jnp.sum(mask)
            return y_accum / (w_accum + 1e-8)

        return jax.vmap(predict_single)(target_times)

    @staticmethod
    def _step_jacobian_full(w_c, w_n, t_c, t_n, p, funcs, dims):
        """Computes all Jacobians of the step residual in one jacrev pass.

        Returns (J_wn, J_wc, J_p, dR_dtc, dR_dtn) — same cost as computing
        J_wn alone since jacrev reuses the same n_w reverse passes.
        """
        f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, reinit_event_owner, reinit_state_target, dims = funcs
        n_x, n_z, n_p = dims

        def step_res(wc, wn, tc, tn, p_var):
            xc, zc = wc[:n_x], wc[n_x:]
            xn, zn = wn[:n_x], wn[n_x:]
            h = tn - tc
            fc = f_fn(tc, xc, zc, p_var)
            fn = f_fn(tn, xn, zn, p_var)
            r_flow = -xn + xc + (h / 2.0) * (fc + fn)
            r_alg = g_fn(tn, xn, zn, p_var)
            return jnp.concatenate([r_flow, r_alg])

        J_wc, J_wn, J_tc, J_tn, J_p = jax.jacrev(
            step_res, argnums=(0, 1, 2, 3, 4)
        )(w_c, w_n, t_c, t_n, p)
        return J_wn, J_wc, J_p, J_tc, J_tn

    @staticmethod
    def _precompute_jacobians(W_p, TS_abs, b_types, b_indices, p_val, funcs, dims):
        """Precomputes all Jacobians (J_wn, J_wc, J_p, dR_dtc, dR_dtn) for all steps."""
        w_c = W_p[:, :-1, :]
        w_n = W_p[:, 1:, :]
        t_c = TS_abs[:, :-1]
        t_n = TS_abs[:, 1:]

        def compute_block_jacobians(wc_seq, wn_seq, tc_seq, tn_seq, b_type):
            def compute_single_step(wc, wn, tc, tn):
                return DAEPaddedGradient._step_jacobian_full(wc, wn, tc, tn, p_val, funcs, dims)

            J_wn, J_wc, J_p, J_tc, J_tn = jax.vmap(compute_single_step)(
                wc_seq, wn_seq, tc_seq, tn_seq
            )
            is_seg = (b_type == 1)
            return J_wn * is_seg, J_wc * is_seg, J_p * is_seg, J_tc * is_seg, J_tn * is_seg

        return jax.vmap(compute_block_jacobians)(w_c, w_n, t_c, t_n, b_types)

    @staticmethod
    def _loss_fn_padded(W_p, p, TS_p, b_types, b_indices, b_param, target_times, target_data, 
    n_targets, t_final, blend_sharpness, n_x, adaptive_horizon=False):
        """JITable loss function with target masking for fixed-shape compilation.

        If adaptive_horizon=True, targets beyond t_final are excluded from the
        loss.  This prevents spurious contributions when the simulation is
        shorter than the reference (e.g. Zeno-like early termination).
        """
        y_pred = DAEPaddedGradient._predict_trajectory_padded_kernel(
            W_p, TS_p, b_types, b_indices, b_param, target_times, t_final, blend_sharpness, n_x
        )
        diff_sq = (y_pred - target_data) ** 2
        mask = (jnp.arange(target_times.shape[0]) < n_targets).astype(jnp.float64)
        if adaptive_horizon:
            mask = mask * (target_times <= t_final).astype(jnp.float64)
        return jnp.sum(mask[:, None] * diff_sq) / (jnp.sum(mask) * n_x + 1e-12)


    def unpack_solution_structure(self, W_flat, structure, grid_taus, t_final=2.0):
        n_x, n_z, n_p = self.dims
        n_w = n_x + n_z
        segments_t = []
        segments_x = []
        segments_z = []
        events_tau = []
        
        # Extract event times first
        event_times_vals = []
        temp_idx = 0
        for kind, count, *extra in structure:
            length = extra[0] if kind == 'segment' else count
            if kind == 'event_time':
                event_times_vals.append(W_flat[temp_idx])
            temp_idx += length
    
        idx_scan = 0
        t_curr = 0.0
        # t_final provided as arg
        ev_ptr = 0
        seg_ctr = 0
    
        for kind, count, *extra in structure:
            length = extra[0] if kind == 'segment' else count
            data = W_flat[idx_scan : idx_scan + length]
            
            if kind == 'segment':
                nodes = data.reshape((count, n_w))
                t_end = event_times_vals[ev_ptr] if ev_ptr < len(event_times_vals) else t_final
                
                tau = grid_taus[seg_ctr]
                ts = t_curr + tau * (t_end - t_curr)
                
                segments_t.append(ts)
                segments_x.append(nodes[:, :n_x])
                segments_z.append(nodes[:, n_x:])
                
                t_curr = t_end
                seg_ctr += 1
            elif kind == 'event_time':
                events_tau.append(data[0])
                ev_ptr += 1
            
            idx_scan += length
            
        return segments_t, segments_x, segments_z, jnp.array(events_tau)

    def pack_solution(self, sol):
        """
        Packs the DAESolver solution into a flat vector W and extracts structure.
        Optimized with pre-allocation to avoid list growth overhead.
        """
        num_seg = len(sol.segments)
        num_events = len(sol.events)
        structure = []
        grid_taus = []
        
        # Pre-extract and optionally downsample
        seg_ts = [np.asarray(s.t) for s in sol.segments]
        seg_ys = [np.concatenate([s.x, s.z], axis=1) for s in sol.segments]
        
        if self.downsample_segments:
            for i in range(num_seg):
                should_downsample = self.all_segments or (seg_ts[i].shape[0] > self.max_pts)
                if should_downsample:
                    seg_ts[i], seg_ys[i] = self._downsample_segment(
                        seg_ts[i], seg_ys[i], self.max_pts
                    )

        # 1. First pass: Calculate total size and build structure
        total_len = 0
        seg_meta = [] 

        for i in range(num_seg):
            t_data = seg_ts[i]
            y_data = seg_ys[i]
            n_points = t_data.shape[0]
            
            # Calculate tau (grid_taus logic)
            t_start, t_end = t_data[0], t_data[-1]
            denom = t_end - t_start if abs(t_end - t_start) > 1e-12 else 1.0
            grid_taus.append((t_data - t_start) / denom)
            
            # Calc sizes
            seg_len = y_data.size # Flattened size
            
            structure.append(('segment', n_points, seg_len))
            seg_meta.append(y_data)
            total_len += seg_len
            
            if i < num_events:
                total_len += 1 # Event time
                structure.append(('event_time', 1))

        # 2. Allocate ONCE
        W_flat = np.empty(total_len, dtype=np.float64)
        
        # 3. Fill via slicing (Vectorized copy)
        cursor = 0
        for i in range(num_seg):
            # Segment
            y_data = seg_meta[i]
            
            chunk_size = y_data.size
            W_flat[cursor : cursor + chunk_size] = y_data.ravel()
            cursor += chunk_size
            
            # Event
            if i < num_events:
                W_flat[cursor] = sol.events[i].t_event
                cursor += 1
                
        return jnp.array(W_flat), structure, grid_taus

    def compute_loss_gradients(self, sol, p_opt, target_times, target_data, blend_sharpness=150.0, adaptive_horizon=False):
        """
        Compute gradients of the loss function w.r.t states (W) and parameters (p).
        Self-sufficient and JIT-compiled.

        Returns:
            dL_dW: Gradient w.r.t flat state vector
            dL_dp: Gradient w.r.t parameters
            structure: Structure of the packed solution (needed for compute_gradient)
        """
        # 1. Pack Solution and Pad Data (Python side)
        W_flat, structure, grid_taus = self.pack_solution(sol)
        
        ts_all = [np.asarray(s.t) for s in sol.segments]
        ys_all = [np.concatenate([s.x, s.z], axis=1) for s in sol.segments]
        event_infos = [(e.t_event, e.event_idx) for e in sol.events]

        if self.downsample_segments:
             for i in range(len(ts_all)):
                should_downsample = self.all_segments or (ts_all[i].shape[0] > self.max_pts)
                if should_downsample:
                    ts_all[i], ys_all[i] = self._downsample_segment(
                        ts_all[i], ys_all[i], self.max_pts
                    )
        
        W_p, TS_p, b_types, b_indices, b_param, _ = self._pad_problem_data(
            ts_all, ys_all, event_infos
        )
        
        # 2. Extract metadata
        if hasattr(sol, 'segments') and len(sol.segments) > 0:
            actual_t_final = sol.segments[-1].t[-1]
        else:
            actual_t_final = 0.0
            
        n_x, n_z, n_p = self.dims
        
        # 3. Pad targets to fixed shape
        tt_padded, td_padded, n_tgt = self._pad_targets(target_times, target_data)

        # 4. Transfer all padded arrays to device in one call
        W_p, TS_p, b_types, b_indices, b_param, tt_padded, td_padded = jax.device_put(
            (W_p, TS_p, b_types, b_indices, b_param, tt_padded, td_padded)
        )
        n_tgt = jnp.int32(n_tgt)

        # 5. Call JITed Gradient
        dL_p, dL_dp, dL_db_param = self.jit_loss_grad(
            W_p, p_opt, TS_p,
            b_types, b_indices, b_param,
            tt_padded, td_padded, n_tgt, actual_t_final, blend_sharpness, n_x,
            adaptive_horizon=adaptive_horizon
        )
        
        # 4. Map dL_p and dL_db_param back to dL_dW (flat)
        # To maintain the interface, we flatten the relevant parts of dL_p
        n_w = n_x + n_z
        dL_list = []
        blk_ptr = 0
        for kind, count, *extra in structure:
            if kind == 'segment':
                # Slices from dL_p[blk_ptr, :count, :]
                chunk = dL_p[blk_ptr, :count, :].flatten()
                dL_list.append(chunk)
                blk_ptr += 1
            elif kind == 'event_time':
                # The event time parameter gradient is specifically in dL_db_param
                dL_list.append(dL_db_param[blk_ptr:blk_ptr+1])
                blk_ptr += 1
                
        dL_dW = jnp.concatenate(dL_list)
        
        return dL_dW, dL_dp, structure

    def compute_total_gradient(self, sol, p_val, target_times, target_data, n_targets=None, blend_sharpness=150.0, adaptive_horizon=False):
        """
        Compute total parameter gradients in a single unified JIT call.
        This is the most efficient high-level API.
        
        Args:
            sol: AugmentedSolution from DAESolver
            p_val: Parameter vector
            target_times: Target times (or padded array if n_targets provided)
            target_data: Target data (or padded array if n_targets provided)
            n_targets: Number of valid targets (if None, targets are padded internally)
            ...
        """
        # 1. Pad problem data (Sol -> Padded Arrays)
        # This MUST be done every iteration as solution changes structure/values
        ts_all = [np.asarray(s.t) for s in sol.segments]
        ys_all = [np.concatenate([s.x, s.z], axis=1) for s in sol.segments]
        event_infos = [(e.t_event, e.event_idx) for e in sol.events]

        # Optional downsampling: reduce segments exceeding max_pts or if all_segments is True
        if self.downsample_segments:
            for i in range(len(ts_all)):
                should_downsample = self.all_segments or (ts_all[i].shape[0] > self.max_pts)
                if should_downsample:
                    ts_all[i], ys_all[i] = self._downsample_segment(
                        ts_all[i], ys_all[i], self.max_pts
                    )

        W_p, TS_p, b_types, b_indices, b_param, _ = self._pad_problem_data(
            ts_all, ys_all, event_infos
        )
        actual_t_final = sol.segments[-1].t[-1] if sol.segments else 0.0

        if n_targets is None:
            # Slow path: Pad and transfer targets every time
            tt_padded, td_padded, n_tgt = self._pad_targets(target_times, target_data)
            n_tgt = jnp.int32(n_tgt)
            
            # Transfer all padded arrays to device
            W_p, TS_p, b_types, b_indices, b_param, tt_padded, td_padded = jax.device_put(
                (W_p, TS_p, b_types, b_indices, b_param, tt_padded, td_padded)
            )
        else:
            # Fast path: Targets already padded and on device
            tt_padded = target_times
            td_padded = target_data
            n_tgt = n_targets
            
            # Transfer only solution-dependent arrays
            W_p, TS_p, b_types, b_indices, b_param = jax.device_put(
                (W_p, TS_p, b_types, b_indices, b_param)
            )

        # 5. Unified JIT call
        loss_val, total_grad = self.jit_total_grad(
            W_p, TS_p, p_val,
            b_types, b_indices, b_param,
            tt_padded, td_padded, n_tgt, actual_t_final, blend_sharpness,
            adaptive_horizon=adaptive_horizon
        )
        
        return loss_val, total_grad

    def predict_trajectory(self, sol, target_times, blend_sharpness=150.0):
        """
        Predict trajectory values at target times using the same kernel as the optimizer.

        Args:
            sol: AugmentedSolution from DAESolver
            target_times: array of time points to evaluate at
            blend_sharpness: sharpness parameter for sigmoid blending

        Returns:
            JAX array of shape (len(target_times), n_x) containing predicted state values
        """
        # 1. Pad problem data (Sol -> Padded Arrays)
        ts_all = [np.asarray(s.t) for s in sol.segments]
        ys_all = [np.concatenate([s.x, s.z], axis=1) for s in sol.segments]
        event_infos = [(e.t_event, e.event_idx) for e in sol.events]

        # Optional downsampling
        if self.downsample_segments:
            for i in range(len(ts_all)):
                if ts_all[i].shape[0] > self.max_pts:
                    ts_all[i], ys_all[i] = self._downsample_segment(
                        ts_all[i], ys_all[i], self.max_pts
                    )

        W_p, TS_p, b_types, b_indices, b_param, _ = self._pad_problem_data(
            ts_all, ys_all, event_infos
        )

        n_x, n_z, n_p = self.dims
        t_final = sol.segments[-1].t[-1] if sol.segments else 0.0

        # Transfer to device
        W_p, TS_p, b_types, b_indices, b_param, t_target_dev = jax.device_put(
            (W_p, TS_p, b_types, b_indices, b_param, target_times)
        )

        # Call the JIT-compiled kernel directly
        y_pred = DAEPaddedGradient._predict_trajectory_padded_kernel(
            W_p, TS_p, b_types, b_indices, b_param,
            t_target_dev, t_final, blend_sharpness, n_x
        )
        
        return y_pred

    def compute_gradient(self, sol, p_val, dL_dW, dL_dp, structure):
        """
        Compute gradients using the Padded Adjoint method.
        
        Args:
            sol: AugmentedSolution object from DAESolver
            p_val: Current parameter values (array)
            dL_dW: Gradient of loss w.r.t flattened states
            dL_dp: Gradient of loss w.r.t parameters
            structure: Structure list from pack_solution
            
        Returns:
            grad_total: Total gradient w.r.t parameters
        """
        # 1. Extract and Pad Data
        ts_all = [np.asarray(s.t) for s in sol.segments]
        ys_all = [np.concatenate([s.x, s.z], axis=1) for s in sol.segments]
        event_infos = [(e.t_event, e.event_idx) for e in sol.events]

        # Optional downsampling: reduce segments exceeding max_pts
        if self.downsample_segments:
            for i in range(len(ts_all)):
                if ts_all[i].shape[0] > self.max_pts:
                    ts_all[i], ys_all[i] = self._downsample_segment(
                        ts_all[i], ys_all[i], self.max_pts
                    )

        W_p, TS_p, b_types, b_indices, b_param, dL_p = self._pad_problem_data(
            ts_all, ys_all, event_infos
        )
        
        # 2. Map dL_dW (flat) to dL_p (padded) using block metadata
        n_x, n_z, n_p = self.dims
        n_w = n_x + n_z

        # Single bulk JAX→numpy transfer
        dL_dW_np = np.asarray(dL_dW)

        # Vectorized offset computation from b_types / b_indices
        flat_lens = np.where(b_types == 1, b_indices[:, 1] * n_w,
                    np.where(b_types == 2, 1, 0))
        offsets = np.cumsum(flat_lens) - flat_lens

        # Fixed-bound loop over max_blocks (integer comparisons only)
        for i in range(self.max_blocks):
            bt = b_types[i]
            if bt == 0:
                break
            if bt == 1:  # segment
                count = b_indices[i, 1]
                start = offsets[i]
                dL_p[i, :count, :] = dL_dW_np[start:start + count * n_w].reshape(count, n_w)
            elif bt == 2:  # event
                dL_p[i, 0, 0] = dL_dW_np[offsets[i]]

        # 3. Reconstruct absolute times from tau for the adjoint sweep
        TS_abs = self._reconstruct_abs_times(TS_p, b_types, b_param, sol)

        # 4. Transfer all padded arrays to device in one call
        W_p, TS_abs, b_types, b_indices, b_param, dL_p = jax.device_put(
            (W_p, TS_abs, b_types, b_indices, b_param, dL_p)
        )

        # 5. Precompute Jacobians
        J_wn, J_wc, J_p_jac, dR_dtc, dR_dtn = self._precompute_jacobians(
             W_p, TS_abs, b_types, b_indices, p_val, self.funcs, self.dims
        )

        # 6. Execute JIT Kernel
        grad_total = self.jit_sweep(
            W_p, TS_abs, p_val,
            b_types, b_indices, b_param,
            J_wn, J_wc, J_p_jac, dR_dtc, dR_dtn, dL_p, dL_dp
        )

        return grad_total

    def optimize_adam(self, solver, p_init, opt_param_indices, target_times, target_data,
                      t_span, ncp, max_iter=150, tol=1e-8, step_size=0.01,
                      beta1=0.9, beta2=0.999, epsilon=1e-8,
                      blend_sharpness=150.0, print_every=10,
                      adaptive_horizon=False):
        """
        Run Adam optimization over the selected DAE parameters.

        Args:
            solver: DAESolver instance (used for forward simulation)
            p_init: Full parameter vector (jnp array), initial values
            opt_param_indices: List of indices into p_init that are being optimized
            target_times: Precomputed target time instants (jnp array)
            target_data: Precomputed target data at those instants (jnp array)
            t_span: (t_start, t_end) for the DAE simulation
            ncp: Number of communication points for the solver
            max_iter: Maximum number of Adam iterations
            tol: Gradient norm tolerance for early stopping
            step_size: Adam learning rate (alpha)
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            blend_sharpness: Sharpness for sigmoid blending in loss
            print_every: Print progress every N iterations

        Returns:
            dict with keys:
                'p_opt': optimized full parameter vector
                'loss_history': list of loss values per iteration
                'grad_norm_history': list of gradient norms per iteration
                'p_history': list of np.ndarray, per-iter values of the
                    optimized parameters (opt_param_indices order)
                'n_iter': number of iterations performed
                'converged': whether gradient norm tolerance was reached
        """
        import time

        n_opt = len(opt_param_indices)
        p_current = jnp.array(p_init, dtype=jnp.float64)

        # Adam state (only for optimized parameters)
        m = jnp.zeros(n_opt)
        v = jnp.zeros(n_opt)

        loss_history = []
        grad_norm_history = []
        # Per-iteration values of the optimized parameters (in opt_param_indices
        # order). Cheap to record (a small host transfer of n_opt floats per
        # iter) and lets callers compute, e.g., a prediction-error series
        # ‖p_iter − p_true‖ comparable across optimizers.
        p_history = []
        opt_idx_array = jnp.array(opt_param_indices)
        converged = False

        print(f"Adam optimization: {len(opt_param_indices)} parameters, max_iter={max_iter}, lr={step_size}")
        
        # 1. Pad and transfer targets ONCE (Static Data Persistence optimization)
        tt_padded, td_padded, n_tgt = self._pad_targets(target_times, target_data)
        tt_device, td_device = jax.device_put((tt_padded, td_padded))
        n_tgt_device = jax.device_put(jnp.int32(n_tgt))

        iter_times = []

        for it in range(1, max_iter + 1):
            t0 = time.perf_counter()

            # --- Forward simulation ---
            solver.update_parameters(np.asarray(p_current))
            max_segs = (self.max_blocks + 1) // 2
            try:
                sol = solver.solve_augmented(t_span, ncp=ncp, max_segments=max_segs)
            except Exception as e:
                print(f"  Iter {it}: solver failed — {e}")
                break

            # --- Compute total gradient AND loss (unified JIT kernel) ---
            loss_val_d, total_grad = self.compute_total_gradient(
                sol, p_current, 
                tt_device, td_device, n_targets=n_tgt_device, # Pass pre-padded device arrays
                blend_sharpness=blend_sharpness,
                adaptive_horizon=adaptive_horizon
            )
            total_grad.block_until_ready()
            
            # Extract scalar loss
            loss_val = float(loss_val_d)

            # Extract gradient for optimized parameters only
            grad_opt = total_grad[jnp.array(opt_param_indices)]
            grad_norm = float(jnp.linalg.norm(grad_opt))

            loss_history.append(loss_val)
            grad_norm_history.append(grad_norm)
            p_history.append(np.asarray(p_current[opt_idx_array]).copy())

            # --- Adam update ---
            m = beta1 * m + (1.0 - beta1) * grad_opt
            v = beta2 * v + (1.0 - beta2) * grad_opt ** 2
            m_hat = m / (1.0 - beta1 ** it)
            v_hat = v / (1.0 - beta2 ** it)
            step = step_size * m_hat / (jnp.sqrt(v_hat) + epsilon)

            p_current = p_current.at[opt_idx_array].add(-step)

            elapsed = (time.perf_counter() - t0) * 1000.0
            iter_times.append(elapsed)
            n_segments = len(sol.segments)

            if it % print_every == 0 or it == 1:
                print(f"  Iter {it:4d} | loss={loss_val:.6e} | |grad|={grad_norm:.6e} | "
                      f"p={np.asarray(p_current[jnp.array(opt_param_indices)])} | {elapsed:.1f} ms | segs={n_segments}")

            # --- Convergence check ---
            if grad_norm < tol:
                print(f"  Converged at iter {it}: |grad|={grad_norm:.6e} < tol={tol:.1e}")
                converged = True
                break

        # Calculate average iteration time (excluding first)
        if len(iter_times) > 1:
            avg_iter_time = sum(iter_times[1:]) / (len(iter_times) - 1)
        else:
            avg_iter_time = 0.0

        print(f"Adam finished: {it} iterations, final loss={loss_history[-1]:.6e}")
        return {
            'p_opt': p_current,
            'loss_history': loss_history,
            'grad_norm_history': grad_norm_history,
            'p_history': p_history,
            'n_iter': it,
            'converged': converged,
            'avg_iter_time': avg_iter_time
        }
