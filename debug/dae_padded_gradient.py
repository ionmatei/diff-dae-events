import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import numpy as np

# Import the padded sweep function
from debug.verify_residual_direct_padded import compute_adjoint_sweep_padded
import debug.verify_residual_gmres as gmres_impl

class DAEPaddedGradient:
    def __init__(self, dae_data, max_blocks=50, max_pts=500, max_targets=200):
        """
        Initialize the Padded Gradient Computer.

        Args:
            dae_data: Dictionary containing DAE system specification
            max_blocks: Maximum number of blocks (segments + events)
            max_pts: Maximum points per segment
            max_targets: Maximum number of target time-points (avoids JIT recompilation)
        """
        self.dae_data = dae_data
        self.max_blocks = max_blocks
        self.max_pts = max_pts
        self.max_targets = max_targets

        # Create JAX functions
        self.funcs = gmres_impl.create_jax_functions(dae_data)
        self.f_fn, self.g_fn, self.h_fn, self.guard_fn, self.reinit_res_fn, self.reinit_vars, self.dims = self.funcs

        # Pre-allocate persistent host-side buffers to avoid repeated np.zeros()
        n_x, n_z, n_p = self.dims
        n_w = n_x + n_z
        self._host_W = np.empty((max_blocks, max_pts, n_w), dtype=np.float64)
        self._host_TS = np.empty((max_blocks, max_pts), dtype=np.float64)
        self._host_block_types = np.empty(max_blocks, dtype=np.int32)
        self._host_block_indices = np.empty((max_blocks, 2), dtype=np.int32)
        self._host_block_param = np.empty(max_blocks, dtype=np.float64)
        self._host_dL = np.empty((max_blocks, max_pts, n_w), dtype=np.float64)
        self._host_target_times = np.empty(max_targets, dtype=np.float64)
        self._host_target_data = np.empty((max_targets, n_x), dtype=np.float64)

        # JIT Compile the sweep function — close over funcs/dims/max_blocks
        # so they become compile-time constants (not static_argnames lookups)
        print(f"JAX Device for Gradient Computation: {jax.devices()[0]}")
        print(f"Compiling Padded JIT solver (Max Blocks: {max_blocks}, Max Pts: {max_pts})...")
        _funcs = self.funcs
        _dims = self.dims
        _max_blocks = max_blocks

        def _sweep_closed(W, TS, p, bt, bi, bp, dW, dp):
            return compute_adjoint_sweep_padded(
                W, TS, p, bt, bi, bp, dW, dp, _funcs, _dims, _max_blocks
            )
        self.jit_sweep = jax.jit(_sweep_closed)

        # JIT Compile the Loss Gradient (for individual use if needed)
        self.jit_loss_grad = jax.jit(
            jax.grad(self._loss_fn_padded, argnums=[0, 1, 5]),
            static_argnames=['n_x', 'adaptive_horizon', 'soft_interp']
        )

        # JIT Compile the Unified Total Gradient Kernel
        print("Compiling Unified Gradient JIT kernel...")
        def _total_grad_closed(W_p, TS_p, p_val, b_types, b_indices, b_param,
                               target_times, target_data, n_targets, t_final, blend_sharpness,
                               adaptive_horizon=False, soft_interp=False):
            return DAEPaddedGradient._total_gradient_kernel(
                W_p, TS_p, p_val, b_types, b_indices, b_param,
                target_times, target_data, n_targets, t_final, blend_sharpness,
                _funcs, _dims, _max_blocks, adaptive_horizon=adaptive_horizon,
                soft_interp=soft_interp
            )
        self.jit_total_grad = jax.jit(_total_grad_closed, static_argnames=['adaptive_horizon', 'soft_interp'])
        
        # Trigger compilation with dummy data
        # (Optional, but ensures first call is fast)
        self._warmup()

    def _warmup(self):
        """Perform a dummy call to trigger JIT compilation."""
        n_x, n_z, n_p = self.dims
        n_w = n_x + n_z
        
        # Create dummy inputs
        W_p = jnp.zeros((self.max_blocks, self.max_pts, n_w))
        TS_p = jnp.zeros((self.max_blocks, self.max_pts))
        b_types = jnp.zeros(self.max_blocks, dtype=jnp.int32)
        b_indices = jnp.zeros((self.max_blocks, 2), dtype=jnp.int32)
        b_param = jnp.zeros(self.max_blocks)
        dL_p = jnp.zeros((self.max_blocks, self.max_pts, n_w))
        dL_dp = jnp.zeros(n_p)
        p_val = jnp.zeros(n_p)
        
        # Dummy targets (padded to max_targets for stable compilation)
        target_times = jnp.zeros(self.max_targets)
        target_data = jnp.zeros((self.max_targets, n_x))
        n_targets = jnp.int32(10)

        # Helper to simulate compilation without blocking too long
        _ = self.jit_sweep(
            W_p, TS_p, p_val,
            b_types, b_indices, b_param,
            dL_p, dL_dp
        )

        _ = self.jit_loss_grad(
            W_p, p_val, TS_p, b_types, b_indices, b_param,
            target_times, target_data, n_targets, 1.0, 150.0, n_x
        )

        _ = self.jit_total_grad(
            W_p, TS_p, p_val, b_types, b_indices, b_param,
            target_times, target_data, n_targets, 1.0, 150.0
        )
        print("Compilation complete.")

    def _pad_problem_data(self, ts_all, ys_all, event_infos):
        """
        Converts dynamic trajectory data into padded fixed-size arrays.
        Reuses pre-allocated host buffers to avoid repeated allocations.
        """
        W_padded = self._host_W;   W_padded.fill(0.0)
        TS_padded = self._host_TS; TS_padded.fill(0.0)
        block_types = self._host_block_types;   block_types.fill(0)
        block_indices = self._host_block_indices; block_indices.fill(0)
        block_param = self._host_block_param;   block_param.fill(0.0)
        dL_padded = self._host_dL; dL_padded.fill(0.0)
        
        curr_blk = 0
        n_segs = len(ts_all)
        n_evs = len(event_infos)
        
        for i in range(n_segs):
            # 1. Segment
            if curr_blk >= self.max_blocks: 
                raise ValueError(f"Exceeded max_blocks ({self.max_blocks})")
            
            n_pts = ts_all[i].shape[0]
            if n_pts > self.max_pts: 
                raise ValueError(f"Segment {i} points {n_pts} > max_pts {self.max_pts}")
                
            W_padded[curr_blk, :n_pts, :] = ys_all[i]
            TS_padded[curr_blk, :n_pts] = ts_all[i]
            block_types[curr_blk] = 1 # Segment
            block_indices[curr_blk] = [0, n_pts] 
            curr_blk += 1
            
            # 2. Event
            if i < n_evs:
                if curr_blk >= self.max_blocks: 
                    raise ValueError(f"Exceeded max_blocks ({self.max_blocks}) during event")
                
                ev_t = event_infos[i]
                block_types[curr_blk] = 2 # Event
                block_indices[curr_blk] = [0, 1] 
                block_param[curr_blk] = ev_t
                TS_padded[curr_blk, 0] = ev_t
                curr_blk += 1
                
        return W_padded, TS_padded, block_types, block_indices, block_param, dL_padded

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
    def _total_gradient_kernel(W_p, TS_p, p_val, b_types, b_indices, b_param, target_times, target_data, n_targets, t_final, blend_sharpness, funcs, dims, max_blocks, adaptive_horizon=False, soft_interp=False):
        """Unified JIT kernel: Loss Differentiation + Adjoint Sweep."""
        n_x, n_z, n_p = dims

        # 1. Compute Loss and Gradients simultaneously
        (loss_val, (dL_dW_p, dL_dp, dL_db_param)) = jax.value_and_grad(
            DAEPaddedGradient._loss_fn_padded, 
            argnums=[0, 1, 5]
        )(
            W_p, p_val, TS_p, b_types, b_indices, b_param,
            target_times, target_data, n_targets, t_final, blend_sharpness, n_x,
            adaptive_horizon=adaptive_horizon, soft_interp=soft_interp
        )
        
        # 2. Call Adjoint Sweep (uses module-level import)
        total_grad = compute_adjoint_sweep_padded(
            W_p, TS_p, p_val,
            b_types, b_indices, b_param,
            dL_dW_p, dL_dp,
            funcs, dims, max_blocks,
            dL_db_param=dL_db_param
        )
        return loss_val, total_grad

    @staticmethod
    def _predict_trajectory_padded_kernel(W_p, TS_p, b_types, b_indices, b_param, target_times, t_final, blend_sharpness, n_x, soft_interp=False):
        """JITable version of trajectory prediction using padded arrays."""
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

        n_pts = b_indices[:, 1]                                        # (max_blocks,)
        n_pts_safe = jnp.maximum(n_pts, 2)                             # avoid 0-index issues
        t_starts = TS_p[:, 0]                                          # (max_blocks,)
        t_ends = TS_p[block_idx, n_pts_safe - 1]                       # (max_blocks,)
        xs_all = W_p[:, :, :n_x]                                       # (max_blocks, max_pts, n_x)

        def _interp_block(ts_row, xs_row, t_c, np_safe):
            """Interpolate one block at one clipped query time (piecewise-linear, searchsorted)."""
            mask_padded = (jnp.arange(ts_row.shape[0]) >= np_safe)
            ts_search = jnp.where(mask_padded, 1e10, ts_row)

            idx = jnp.searchsorted(ts_search, t_c, side='right') - 1
            idx = jnp.clip(idx, 0, np_safe - 2)
            t0, t1 = ts_row[idx], ts_row[idx + 1]
            denom = jnp.where(jnp.abs(t1 - t0) < 1e-12, 1e-12, t1 - t0)
            s = jnp.clip((t_c - t0) / denom, 0.0, 1.0)
            return xs_row[idx] * (1.0 - s) + xs_row[idx + 1] * s

        def _interp_block_soft(ts_row, xs_row, t_c, np_safe):
            """Soft-weight interpolation: fully differentiable w.r.t. knot times.

            Uses Gaussian kernel weights instead of searchsorted so that
            gradients propagate through knot locations and event times.
            """
            i = jnp.arange(ts_row.shape[0])
            valid = (i < np_safe)

            # Adaptive bandwidth from block time span
            t_first = ts_row[0]
            t_last = ts_row[jnp.maximum(np_safe - 1, 0)]
            dt_typ = jnp.maximum((t_last - t_first) / jnp.maximum(np_safe - 1, 1), 1e-12)
            sigma = 2.0 * dt_typ
            alpha = 0.5 / (sigma * sigma)

            # Gaussian weights
            d = ts_row - t_c
            w = jnp.exp(-alpha * d * d)
            w = jnp.where(valid, w, 0.0)

            # Normalize
            w_sum = jnp.sum(w)
            w = w / (w_sum + 1e-12)

            return jnp.sum(xs_row * w[:, None], axis=0)

        _interp_fn = _interp_block_soft if soft_interp else _interp_block

        def predict_single(t_q):
            # Masks for all blocks at once
            mask = (jax.nn.sigmoid(blend_sharpness * (t_q - lower)) *
                    jax.nn.sigmoid(blend_sharpness * (upper - t_q)) *
                    is_seg)                                             # (max_blocks,)

            t_clip = jnp.clip(t_q, t_starts, t_ends)                   # (max_blocks,)

            # vmap interpolation over blocks
            vals = jax.vmap(_interp_fn)(
                TS_p, xs_all, t_clip, n_pts_safe
            )                                                           # (max_blocks, n_x)

            y_accum = jnp.sum(mask[:, None] * vals, axis=0)            # (n_x,)
            w_accum = jnp.sum(mask)
            return y_accum / (w_accum + 1e-8)

        return jax.vmap(predict_single)(target_times)

    @staticmethod
    def _loss_fn_padded(W_p, p, TS_p, b_types, b_indices, b_param, target_times, target_data, 
    n_targets, t_final, blend_sharpness, n_x, adaptive_horizon=False, soft_interp=False):
        """JITable loss function with target masking for fixed-shape compilation.

        If adaptive_horizon=True, targets beyond t_final are excluded from the
        loss.  This prevents spurious contributions when the simulation is
        shorter than the reference (e.g. Zeno-like early termination).
        """
        y_pred = DAEPaddedGradient._predict_trajectory_padded_kernel(
            W_p, TS_p, b_types, b_indices, b_param, target_times, t_final, blend_sharpness, n_x,
            soft_interp=soft_interp
        )
        diff_sq = (y_pred - target_data) ** 2
        mask = (jnp.arange(target_times.shape[0]) < n_targets).astype(jnp.float64)
        if adaptive_horizon:
            mask = mask * (target_times <= t_final).astype(jnp.float64)
        return jnp.sum(mask[:, None] * diff_sq) / (jnp.sum(mask) * n_x + 1e-12)

    @staticmethod
    def predict_trajectory_sigmoid(segments_t, segments_x, segments_z, events_tau, target_times, blend_sharpness=300.0):
        n_outputs = segments_x[0].shape[1]
        
        def predict_single(t_q):
            y_accum = jnp.zeros(n_outputs)
            w_accum = 0.0
            
            for i in range(len(segments_t)):
                ts = segments_t[i]
                xs = segments_x[i]
                t_start, t_end = ts[0], ts[-1]
                
                lower = t_start if i == 0 else events_tau[i-1]
                upper = t_end if i == len(segments_t)-1 else events_tau[i]
                
                mask = jax.nn.sigmoid(blend_sharpness * (t_q - lower)) * jax.nn.sigmoid(blend_sharpness * (upper - t_q))
                t_clip = jnp.clip(t_q, t_start, t_end)
                
                idx = jnp.searchsorted(ts, t_clip, side='right') - 1
                idx = jnp.clip(idx, 0, len(ts)-2)
                
                t0_g, t1_g = ts[idx], ts[idx+1]
                denom = jnp.where(jnp.abs(t1_g - t0_g) < 1e-12, 1e-12, t1_g - t0_g)
                s = jnp.clip((t_clip - t0_g) / denom, 0.0, 1.0)
                
                val = xs[idx] * (1.0 - s) + xs[idx+1] * s
                y_accum += mask * val
                w_accum += mask
                
            return y_accum / (w_accum + 1e-8)

        return jax.vmap(predict_single)(target_times)

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
        
        # 1. First pass: Calculate total size and build structure
        total_len = 0
        seg_meta = [] # Store (n_points, n_vars) for filling later

        for i in range(num_seg):
            seg = sol.segments[i]
            n_points = len(seg.t)
            
            # Calculate tau (grid_taus logic)
            t_start, t_end = seg.t[0], seg.t[-1]
            denom = t_end - t_start if abs(t_end - t_start) > 1e-12 else 1.0
            grid_taus.append((seg.t - t_start) / denom)
            
            # Calc sizes
            # Assuming x and z are shapes (n_points, dim)
            n_x_vars = seg.x.shape[1]
            n_z_vars = seg.z.shape[1] if len(seg.z) > 0 else 0
            seg_len = n_points * (n_x_vars + n_z_vars)
            
            structure.append(('segment', n_points, seg_len))
            seg_meta.append((seg.x, seg.z))
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
            xs, zs = seg_meta[i]
            
            # Flatten segment data: Concatenate X and Z horizontally, then flatten.
            if zs is not None and zs.shape[1] > 0:
                merged = np.hstack([xs, zs])
            else:
                merged = xs
                
            chunk_size = merged.size
            W_flat[cursor : cursor + chunk_size] = merged.ravel()
            cursor += chunk_size
            
            # Event
            if i < num_events:
                W_flat[cursor] = sol.events[i].t_event
                cursor += 1
                
        return jnp.array(W_flat), structure, grid_taus

    def compute_loss_gradients(self, sol, p_opt, target_times, target_data, blend_sharpness=150.0, adaptive_horizon=False, soft_interp=False):
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
        
        ts_all = [s.t for s in sol.segments]
        ys_all = [jnp.concatenate([s.x, s.z], axis=1) for s in sol.segments]
        event_infos = [e.t_event for e in sol.events]
        
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
            adaptive_horizon=adaptive_horizon, soft_interp=soft_interp
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

    def compute_total_gradient(self, sol, p_val, target_times, target_data, n_targets=None, blend_sharpness=150.0, adaptive_horizon=False, soft_interp=False):
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
        ts_all = [s.t for s in sol.segments]
        ys_all = [jnp.concatenate([s.x, s.z], axis=1) for s in sol.segments]
        event_infos = [e.t_event for e in sol.events]
        
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
            adaptive_horizon=adaptive_horizon, soft_interp=soft_interp
        )
        
        return loss_val, total_grad

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
        ts_all = [s.t for s in sol.segments]
        ys_all = [jnp.concatenate([s.x, s.z], axis=1) for s in sol.segments]
        event_infos = [e.t_event for e in sol.events]
        
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

        # 3. Transfer all padded arrays to device in one call
        W_p, TS_p, b_types, b_indices, b_param, dL_p = jax.device_put(
            (W_p, TS_p, b_types, b_indices, b_param, dL_p)
        )

        # 4. Execute JIT Kernel
        grad_total = self.jit_sweep(
            W_p, TS_p, p_val,
            b_types, b_indices, b_param,
            dL_p, dL_dp
        )
        
        return grad_total

    def optimize_adam(self, solver, p_init, opt_param_indices, target_times, target_data,
                      t_span, ncp, max_iter=150, tol=1e-8, step_size=0.01,
                      beta1=0.9, beta2=0.999, epsilon=1e-8,
                      blend_sharpness=150.0, print_every=10,
                      adaptive_horizon=False, soft_interp=False):
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
        converged = False

        print(f"Adam optimization: {len(opt_param_indices)} parameters, max_iter={max_iter}, lr={step_size}")
        
        # 1. Pad and transfer targets ONCE (Static Data Persistence optimization)
        tt_padded, td_padded, n_tgt = self._pad_targets(target_times, target_data)
        tt_device, td_device = jax.device_put((tt_padded, td_padded))
        n_tgt_device = jax.device_put(jnp.int32(n_tgt))

        for it in range(1, max_iter + 1):
            t0 = time.perf_counter()

            # --- Forward simulation ---
            solver.update_parameters(np.asarray(p_current))
            try:
                sol = solver.solve_augmented(t_span, ncp=ncp)
            except Exception as e:
                print(f"  Iter {it}: solver failed — {e}")
                break

            # --- Compute total gradient AND loss (unified JIT kernel) ---
            loss_val_d, total_grad = self.compute_total_gradient(
                sol, p_current, 
                tt_device, td_device, n_targets=n_tgt_device, # Pass pre-padded device arrays
                blend_sharpness=blend_sharpness,
                adaptive_horizon=adaptive_horizon,
                soft_interp=soft_interp
            )
            total_grad.block_until_ready()
            
            # Extract scalar loss
            loss_val = float(loss_val_d)

            # Extract gradient for optimized parameters only
            grad_opt = total_grad[jnp.array(opt_param_indices)]
            grad_norm = float(jnp.linalg.norm(grad_opt))

            loss_history.append(loss_val)
            grad_norm_history.append(grad_norm)

            # --- Adam update ---
            m = beta1 * m + (1.0 - beta1) * grad_opt
            v = beta2 * v + (1.0 - beta2) * grad_opt ** 2
            m_hat = m / (1.0 - beta1 ** it)
            v_hat = v / (1.0 - beta2 ** it)
            step = step_size * m_hat / (jnp.sqrt(v_hat) + epsilon)

            p_current = p_current.at[jnp.array(opt_param_indices)].add(-step)

            elapsed = (time.perf_counter() - t0) * 1000.0

            if it % print_every == 0 or it == 1:
                print(f"  Iter {it:4d} | loss={loss_val:.6e} | |grad|={grad_norm:.6e} | "
                      f"p={np.asarray(p_current[jnp.array(opt_param_indices)])} | {elapsed:.1f} ms")

            # --- Convergence check ---
            if grad_norm < tol:
                print(f"  Converged at iter {it}: |grad|={grad_norm:.6e} < tol={tol:.1e}")
                converged = True
                break

        print(f"Adam finished: {it} iterations, final loss={loss_history[-1]:.6e}")
        return {
            'p_opt': p_current,
            'loss_history': loss_history,
            'grad_norm_history': grad_norm_history,
            'n_iter': it,
            'converged': converged,
        }
