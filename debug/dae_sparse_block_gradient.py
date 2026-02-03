
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import time

# Import helper to create functions
import debug.verify_residual_gmres as gmres_impl

def block_tridiag_solve(A_blocks, B_blocks, C_blocks, rhs):
    """
    Solves a Block-Tridiagonal system M x = rhs.
    M has structure:
    [ B0  C0         ] [ x0 ]   [ r0 ]
    [ A1  B1  C1     ] [ x1 ] = [ r1 ]
    [     ...        ] [ .. ]   [ .. ]
    [     AN-1 BN-1  ] [ xN-1]  [ rN-1]

    Note: The nomenclature here matches standard Thomas Algorithm:
    - B_i: Main Diagonal Block (NOT off-diagonal B in DAE context)
    - A_i: Sub-Diagonal Block (Left)
    - C_i: Super-Diagonal Block (Right)
    
    Arguments:
        A_blocks: Shape (N-1, d, d)  (Lower diagonal, starting at row 1)
        B_blocks: Shape (N, d, d)    (Main diagonal)
        C_blocks: Shape (N-1, d, d)  (Upper diagonal, starting at row 0)
        rhs:      Shape (N, d)
        
    Returns:
        x: Shape (N, d)
    """
    N = B_blocks.shape[0]
    
    # 1. Forward Elimination (Modify C' and d')
    # C'_0 = B_0^-1 * C_0
    # d'_0 = B_0^-1 * r_0
    
    # C'_i = (B_i - A_i * C'_{i-1})^-1 * C_i
    # d'_i = (B_i - A_i * C'_{i-1})^-1 * (r_i - A_i * d'_{i-1})
    
    # We use lax.scan for this sequential process.
    
    # Initial step (row 0)
    B0_inv = jnp.linalg.inv(B_blocks[0])
    # Handle C0 if N > 1
    if N > 1:
        C0_prime = B0_inv @ C_blocks[0]
    else:
        C0_prime = jnp.zeros_like(B_blocks[0]) # Dummy
        
    d0_prime = B0_inv @ rhs[0]
    
    def forward_scan(carrier, i):
        C_prev, d_prev = carrier
        
        # Current blocks
        # A is at i-1 (since A starts at row 1)
        # B is at i
        # C is at i
        A_curr = A_blocks[i-1]
        B_curr = B_blocks[i]
        r_curr = rhs[i]
        
        # Temp = (B_i - A_i * C'_{i-1})
        temp = B_curr - A_curr @ C_prev
        temp_inv = jnp.linalg.inv(temp)
        
        d_curr = temp_inv @ (r_curr - A_curr @ d_prev)
        
        # Calculate C_next only if i < N-1
        # To keep shapes static, we compute it but mask or just compute safely.
        # C_blocks has length N-1. Index i is valid for i < N-1.
        # If i == N-1 (last row), there is no C.
        
        def compute_c():
            return temp_inv @ C_blocks[i]
            
        # If i == N-1, return zeros (or garbage, won't be used)
        # But we need consistent shape.
        C_curr = jax.lax.cond(i < N-1, compute_c, lambda: jnp.zeros_like(B_curr))
        
        return (C_curr, d_curr), (C_curr, d_curr)

    if N > 1:
        indices = jnp.arange(1, N)
        _, (C_primes_rest, d_primes_rest) = jax.lax.scan(forward_scan, (C0_prime, d0_prime), indices)
        
        # Concatenate results
        # C_primes: [C'0, C'1, ..., C'N-2, (dummy C'N-1)]
        # We need to reconstruct full arrays for backward pass
        
        # d_primes: [d'0, d'1, ..., d'N-1]
        all_d_primes = jnp.concatenate([d0_prime[None, :], d_primes_rest], axis=0)
        all_C_primes = jnp.concatenate([C0_prime[None, :, :], C_primes_rest], axis=0)
    else:
        all_d_primes = d0_prime[None, :]
        all_C_primes = jnp.zeros((1, *B0_inv.shape)) # Dummy
        
    # 2. Backward Substitution
    # x_N-1 = d'_N-1
    # x_i = d'_i - C'_i * x_{i+1}
    
    x_last = all_d_primes[-1]
    
    def backward_scan(x_next, i):
        # i goes from N-2 down to 0
        d_curr = all_d_primes[i]
        C_curr = all_C_primes[i] # This is C'_i
        
        x_curr = d_curr - C_curr @ x_next
        return x_curr, x_curr
        
    if N > 1:
        rev_indices = jnp.arange(N-2, -1, -1)
        _, x_rest = jax.lax.scan(backward_scan, x_last, rev_indices)
        # x_rest is [x_N-2, ..., x_0] (reverse order of creation, scan puts output in order of iteration?)
        # Jax scan output matches iteration order. So x_rest[0] is x_N-2.
        
        # We need to reverse x_rest to get [x_0, ..., x_N-2]
        x_rest_ordered = x_rest[::-1]
        
        x_final = jnp.concatenate([x_rest_ordered, x_last[None, :]], axis=0)
    else:
        x_final = x_last[None, :]
        
    return x_final


class DAESparseBlockGradient:
    def __init__(self, dae_data):
        self.dae_data = dae_data
        self.funcs = gmres_impl.create_jax_functions(dae_data)
        self.f_fn, self.g_fn, self.h_fn, self.guard_fn, self.reinit_res_fn, self.reinit_vars, self.dims = self.funcs
        self.n_x, self.n_z, self.n_p = self.dims
        self.n_w = self.n_x + self.n_z
        self.x0_start = jnp.array([s['start'] for s in dae_data['states']])
        self._kernel_cache = {}

    def pack_solution(self, sol):
        # Same as DAEMatrixGradient
        w_list = []
        structure = []
        grid_taus = [] 
        num_seg = len(sol.segments)
        num_events = len(sol.events)
        for i in range(num_seg):
            seg = sol.segments[i]
            n_points = len(seg.t)
            t_start, t_end = seg.t[0], seg.t[-1]
            denom = t_end - t_start if abs(t_end-t_start) > 1e-12 else 1.0
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
        struct_key = tuple(structure)
        if struct_key in self._kernel_cache:
            return self._kernel_cache[struct_key]

        print(f"Compiling Sparse Block Gradient Kernel for structure: {len(structure)} blocks...")
        
        dims = self.dims
        funcs = self.funcs
        x0_start = self.x0_start
        n_x, n_z, n_p = dims
        n_w = n_x + n_z

        # Identify dimensions
        # Total Unknowns N = len(W) / n_w? No, W includes event times.
        # W has mixed types: block of n_w vars (point), then 1 var (event time).
        # To use block solver, we need uniform blocks or handle the mix.
        #
        # BUT: The Block-Tridiagonal solver assumes uniform block size d.
        # The DAE system is NOT uniform:
        # - Most blocks are size n_w (Collocation points)
        # - Event blocks are size 1 (Event time)
        #
        # STRATEGY:
        # We can implement a "General Block Sparse Solver" that handles the specific structure of this problem without full N^3.
        # OR: We can "pad" the event blocks to size n_w so everything is uniform size D=n_w.
        # Padding is easier for JAX scan.
        # The equation for event time is scalar: Guard(te) = 0.
        # We can add n_w - 1 dummy equations (e.g. Identity) to fill the block.
        #
        # Let d_max = max(n_w, 1) = n_w.
        # We treat the system as a sequence of blocks of size n_w.
        # Event time variable `te` will be stored in the first slot of a size-n_w block. Other slots unused.
        
        # Loss Function Definition
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

        d_block = n_w

        # Segment-Level Block Strategy
        # Block i = [Segment i Points (Flattened), Event i Time (or Dummy)]
        # This restores Tridiagonal structure:
        # - Block i depends on Block i (Self)
        # - Block i depends on Block i-1 (Continuity from previous event/segment)
        # - Block i depends on Block i+1 (Reset from next segment if event occurs? No, Reset depends on POST state)
        #   Actually, Event i (te) determines T_end for Block i and T_start for Block i+1.
        #   So Block i+1 depends on Block i.
        #   Reset Logic: x_post (Block i+1 start) = Reset(x_pre (Block i end)).
        #   So Block i+1 depends on Block i.
        #   This is purely Lower Triangular / Block Bidiagonal if we order correctly?
        #   Wait, Boundary Value Problems (Collocation) couple points within the block.
        #   But between blocks, it is causal (forward in time).
        #   UNLESS we have boundary conditions at T_final? No, Initial Value Problem.
        #
        #   If the problem is an IVP (Initial Value Problem), the Jacobian is Lower Block Triangular.
        #   So we can solve it with Forward Substitution alone! (No Need for Thomas/Backward pass).
        #   
        #   Let's verify:
        #   Res_0 (Seg 0) depends on x0 (fixed) and variables in Seg 0.
        #   Res_1 (Seg 1) depends on Seg 0 end (continuity) and variables in Seg 1.
        #   
        #   So J is:
        #   [ J_00   0    0 ]
        #   [ J_10  J_11  0 ]
        #   [  0    J_21 J_22 ]
        #
        #   Yes! For IVPs, the system is Lower Block Triangular.
        #   We just need to invert the diagonal blocks J_ii and forward propagate.
        #   
        #   J_ii is the Jacobian of the Segment Collocation System w.r.t itself.
        #   This internal block is dense (or banded). We will treat it as dense for now.
        
        # Max dimension for padding
        # We need uniform block size.
        # Size = max_pts * n_w + 1 (event)
        # We'll need to define a 'max_pts' based on structure or config.
        # For this implementation, we take the max from the structure passed in.
        
        max_pts_in_struct = 0
        for kind, count, *extra in structure:
            if kind == 'segment':
                max_pts_in_struct = max(max_pts_in_struct, count)
        
        D_block = max_pts_in_struct * n_w + 1
        
        def compute_grads(W_flat, p_val, grid_taus, target_times, target_data, t_final_val, blend_sharpness, soft_interp):
            
            # 1. Reconstruction & Padding
            # We transform W_flat into a stack of blocks: (N_blocks, D_block)
            
            blocks_list = []
            
            w_cursor = 0
            ev_cursor = 0
            
            # We also need to construct "Segment Objects" for the residual function
            # Since JAX needs static shapes, we iterate and pad.
            
            # Map structure to logical segment blocks
            # Each 'segment' in structure is a block.
            # Associated event time is usually the *next* thing in W after segment, 
            # OR the end of the segment.
            # Let's say Block k contains Seg k and Event k (te).
            
            # Iterate structure to bundle (Seg, Event)
            # Structure: Seg0, Ev0, Seg1, Ev1, ...
            # Last segment might not have event.
            
            seg_defs = [] # (w_slice, count, ev_slice)
            
            temp_cursor = 0
            i = 0
            while i < len(structure):
                kind, count, *extra = structure[i]
                length = extra[0]
                
                seg_slice = slice(temp_cursor, temp_cursor + length)
                temp_cursor += length
                i += 1
                
                ev_slice = None
                if i < len(structure):
                    next_kind = structure[i][0]
                    if next_kind == 'event_time':
                        ev_slice = slice(temp_cursor, temp_cursor + 1)
                        temp_cursor += 1
                        i += 1
                
                seg_defs.append({'seg': seg_slice, 'count': count, 'ev': ev_slice})

            N_blocks = len(seg_defs)
            
            # Function to extract padded block k
            def get_block_k(k, W):
                # This must be JIT-compatible.
                # Since 'k' is iterated in Python for J_blocks construction, we can use static slicing logic?
                # No, we want to construct the J_blocks.
                #
                # Actually, simpler:
                # Just defining the Residual function `Res_k(Block_k, Block_{k-1}, p)`
                # is enough.
                #
                # We define a function `make_residual_k(k)` that returns a function `rk(b_curr, b_prev, p)`.
                pass

            # 2. Compute Jacobian Blocks
            # Since it's Lower Block Triangular:
            # - Diagonal J_kk = dRes_k / dBlock_k
            # - Off-Diag J_k,k-1 = dRes_k / dBlock_{k-1}
            #
            # We can loop k from 0 to N_blocks-1
            
            J_diags = [] # J_kk inverted? Or just J_kk.
            J_lowers = [] # J_k,k-1
            
            # Data preparation for all blocks (for Residual eval)
            # We need to reconstruct full trajectory to get T_start, etc.
            # Actually, `Res_k` only needs relative time or T_start passed in.
            # T_start of Block k depends on T_end of Block k-1 (which is in Block k-1).
            # So `Res_k` is purely varying with (Block_k, Block_{k-1}).
            
            # Exception: Target Data residuals (Loss)
            # The total gradient includes dL/dW.
            # L is sum over blocks?
            # L = Sum(L_k).
            # dL/dW is vector dL/dW_k.
            
            # Let's compute dL/dW first.
            dL_dW = jax.grad(loss_fn, argnums=0)(W_flat, p_val, grid_taus, target_times, target_data, t_final_val, blend_sharpness, soft_interp)
            dL_dp = jax.grad(loss_fn, argnums=1)(W_flat, p_val, grid_taus, target_times, target_data, t_final_val, blend_sharpness, soft_interp)
            
            # Now we solve J^T lambda = -dL_dW
            # J is Lower Block Triangular.
            # J^T is Upper Block Triangular.
            # [ J00^T  J10^T  ... ] [ lam0 ]   [ -dL0 ]
            # [   0    J11^T  J21^T ] [ lam1 ] = [ -dL1 ]
            # ...
            # Backward Substitution!
            # equation for lam_last (N-1):
            # J_{N-1, N-1}^T * lam_{N-1} = -dL_{N-1}
            # lam_{N-1} = J_{N-1, N-1}^-T * (-dL_{N-1})
            #
            # equation for lam_k:
            # J_{k,k}^T * lam_k + J_{k+1,k}^T * lam_{k+1} = -dL_k
            # J_{k,k}^T * lam_k = -dL_k - J_{k+1,k}^T * lam_{k+1}
            
            def compute_block_jacobians(k):
                # Returns (J_kk, J_k_prev, J_p_k)
                
                # Def local residual
                # inputs: b_curr_flat, b_prev_flat, p
                # We need to unflatten to (Points, Event)
                
                curr_def = seg_defs[k]
                prev_def = seg_defs[k-1] if k > 0 else None
                
                n_pts_curr = curr_def['count']
                tau_curr = grid_taus[k] # This is tied to structure
                
                def local_res_fn(b_curr, b_prev, p_arg):
                    # Unpack b_curr
                    # segment part: first n_pts_curr * n_w
                    seg_len = n_pts_curr * n_w
                    seg_data = b_curr[:seg_len].reshape((n_pts_curr, n_w))
                    xs = seg_data[:, :n_x]
                    zs = seg_data[:, n_x:]
                    
                    # te is last element? Or we just assume b_curr HAS it if defined.
                    # In our packing, te is in W.
                    # We passed slices of W. 
                    # b_curr passed here is the Slice of W?
                    # Yes.
                    
                    # If current block has event, it's at end.
                    has_ev = (curr_def['ev'] is not None)
                    if has_ev:
                        te = b_curr[-1]
                    else:
                        te = t_final_val # Last segment uses fixed horizon

                    # Previous state/time
                    if k == 0:
                        x0_prev = x0_start
                        t_prev_end = 0.0 # Start time
                        # Continuity: xs[0] - x0_prev
                        pass
                    else:
                        # Unpack prev
                        n_pts_prev = prev_def['count']
                        seg_len_prev = n_pts_prev * n_w
                        seg_data_prev = b_prev[:seg_len_prev].reshape((n_pts_prev, n_w))
                        x_last_prev = seg_data_prev[-1, :n_x]
                        z_last_prev = seg_data_prev[-1, n_x:]
                        
                        # Has prev event? Should.
                        if prev_def['ev'] is not None:
                            te_prev = b_prev[-1]
                        else:
                            te_prev = 0.0 # Should not happen if k>0
                        
                        t_prev_end = te_prev
                        
                        # Apply Guard/Reset logic to get x0_prev expected
                        # Wait, Reset happens at te_prev.
                        # Transition from x_last_prev (at te_prev-) to xs[0] (at te_prev+)
                        pass

                    # --- Compute Residuals ---
                    res_list = []
                    
                    # 1. Continuity / Reset at Start
                    if k == 0:
                         res_list.append(xs[0] - x0_start)
                    else:
                        # Reset depends on x_last_prev, te_prev
                        # And xs[0] (x_post)
                        # Constraint: Reinit(te_prev, xs[0], zs[0], x_last_prev, z_last_prev) = 0
                        # And continuity for non-reinit vars
                        
                        val_reset = funcs[4](t_prev_end, xs[0], zs[0], x_last_prev, z_last_prev, p_arg)
                        res_list.append(val_reset.reshape(-1))
                        
                        # Continuity
                        # Manual unrolled check
                        diffs = []
                        for idx_state in range(n_x):
                             is_reinit = False
                             for (rtype, ridx) in funcs[5]:
                                 if rtype == 'state' and ridx == idx_state:
                                     is_reinit = True
                                     break
                             if not is_reinit:
                                 diffs.append(xs[0, idx_state] - x_last_prev[idx_state])
                        if diffs:
                            res_list.append(jnp.stack(diffs))

                        # G constraint at post-event
                        if n_z > 0:
                            res_list.append(funcs[1](t_prev_end, xs[0], zs[0], p_arg).flatten())

                    # 2. Seg Flow
                    if n_pts_curr > 1:
                        # Needs t_start = t_prev_end
                        # Needs te = te (current)
                        
                        t0 = t_prev_end
                        denom = te - t0
                        ts = t0 + tau_curr * denom
                        
                        ts_c = ts[:-1]
                        ts_n = ts[1:]
                        dt = ts_n - ts_c
                        
                        # vmap functions
                        def call_f(t, x, z): return funcs[0](t, x, z, p_arg)
                        def call_g(t, x, z): return funcs[1](t, x, z, p_arg)
                        
                        f_c = jax.vmap(call_f)(ts_c, xs[:-1], zs[:-1])
                        f_n = jax.vmap(call_f)(ts_n, xs[1:], zs[1:])
                        
                        res_flow = -xs[1:] + xs[:-1] + 0.5 * dt[:, None] * (f_c + f_n)
                        res_list.append(res_flow.flatten())
                        
                        if n_z > 0:
                            g_c = jax.vmap(call_g)(ts_c, xs[:-1], zs[:-1])
                            res_list.append(g_c.flatten())
                            
                    # 3. G at last point
                    t_end_curr = te # or if no event, t_final
                    if n_z > 0:
                         res_list.append(funcs[1](t_end_curr, xs[-1], zs[-1], p_arg).flatten())

                    # 4. Guard Condition (if event)
                    if has_ev:
                        # Guard(te, x_last, z_last) = 0
                        val_guard = funcs[3](te, xs[-1], zs[-1], p_arg)
                        res_list.append(val_guard.reshape(-1))
                        
                    return jnp.concatenate([r.flatten() for r in res_list])

                # Get Slices values
                b_curr_val = W_flat[curr_def['seg']]
                if curr_def['ev']:
                    b_curr_val = jnp.concatenate([b_curr_val, W_flat[curr_def['ev']]])
                
                if k > 0:
                    b_prev_val = W_flat[prev_def['seg']]
                    if prev_def['ev']:
                        b_prev_val = jnp.concatenate([b_prev_val, W_flat[prev_def['ev']]])
                else:
                    b_prev_val = jnp.array([]) # Dummy

                # Jacobians
                J_kk = jax.jacfwd(local_res_fn, argnums=0)(b_curr_val, b_prev_val, p_val)
                J_p_k = jax.jacfwd(local_res_fn, argnums=2)(b_curr_val, b_prev_val, p_val)
                
                J_k_prev = None
                if k > 0:
                    J_k_prev = jax.jacfwd(local_res_fn, argnums=1)(b_curr_val, b_prev_val, p_val)
                
                return J_kk, J_k_prev, J_p_k

            # Build all blocks
            J_blocks_diag = [] # [J_00, J_11, ...]
            J_blocks_prev = [] # [None, J_10, J_21, ...]
            J_p_total = jnp.zeros_like(p_val) # Accumulator? No, J_p is tall matrix. 
            # Actually J_p is part of the total J. J_p_blocks.
            J_p_blocks = []
            
            for k in range(N_blocks):
                d, p_prev, p_param = compute_block_jacobians(k)
                J_blocks_diag.append(d)
                J_blocks_prev.append(p_prev)
                J_p_blocks.append(p_param)
            
            # Solve Adjoint: Backward Pass
            # J_{k,k}^T * lam_k = -dL_k - J_{k+1,k}^T * lam_{k+1}
            
            # Map global dL_dW to blocks
            lambdas = [None] * N_blocks
            
            # Backward loop
            for k in reversed(range(N_blocks)):
                # Extract dL_k
                # Construct slice for this block in W
                sl_seg = curr_def = seg_defs[k]['seg']
                sl_ev = seg_defs[k]['ev']
                
                dL_k_seg = dL_dW[sl_seg]
                if sl_ev:
                    dL_k = jnp.concatenate([dL_k_seg, dL_dW[sl_ev]])
                else:
                    dL_k = dL_k_seg
                
                # RHS term
                rhs = -dL_k
                if k < N_blocks - 1:
                    # Add -J_{k+1, k}^T * lam_{k+1}
                    # J_{k+1, k} is J_blocks_prev[k+1]
                    term = J_blocks_prev[k+1].T @ lambdas[k+1]
                    rhs = rhs - term
                
                # Solve using dense solve for the block
                # J_kk.T * lam_k = rhs
                lam_k = jnp.linalg.solve(J_blocks_diag[k].T, rhs)
                lambdas[k] = lam_k
                
            # Total Gradient w.r.t p
            # grad = dL/dp + sum(lambda_k^T * dRes_k/dp)
            #      = dL/dp + sum(lambda_k . J_p_k)
            
            term_p = jnp.zeros_like(dL_dp)
            for k in range(N_blocks):
                term_p = term_p + lambdas[k] @ J_p_blocks[k]
                
            return dL_dp + term_p
            
        return jax.jit(compute_grads, static_argnames=['soft_interp'])


    def compute_total_gradient(self, sol, p_val, target_times, target_data, blend_sharpness=150.0, soft_interp=False):
        flat_W, structure, grid_taus = self.pack_solution(sol)
        # Convert lists to tuples for hashing if needed, or pass structure as is
        # structure is list of tuples.
        
        kernel = self._get_gradient_kernel(structure)
        
        # Determine t_final
        # t_final is the end time of the last segment
        t_final = sol.segments[-1].t[-1]
        
        # grid_taus is list of arrays. Need to handle carefully in JIT?
        # The kernel 'grid_taus' argument is expected to be consistent.
        # But here grid_taus length changes with structure.
        # The JIT kernel is specialized for structure, so it knows the length.
        
        # Pass data to kernel
        # We need to ensure JAX sees these as static or consistent?
        # arrays in grid_taus vary in size.
        # JIT expects fixed args.
        # Structure is baked in.
        # We can pass grid_taus as a list (pytree).
        
        # Convert target_data to JAX array if not already
        target_data = jnp.array(target_data)
        target_times = jnp.array(target_times)
        
        grad_p = kernel(flat_W, p_val, grid_taus, target_times, target_data, t_final, blend_sharpness, soft_interp)
        return grad_p
        
    def optimize_adam(self, solver, p_init, opt_param_indices, target_times, target_data,
                      t_span, ncp, max_iter=150, tol=1e-8, step_size=0.01,
                      beta1=0.9, beta2=0.999, epsilon=1e-8,
                      blend_sharpness=150.0, print_every=10, soft_interp=False):
        """
        Runs Adam optimization using the Sparse Block-Gradient method.
        """
        p_current = jnp.array(p_init)
        m = jnp.zeros_like(p_current)
        v = jnp.zeros_like(p_current)
        
        converged = False
        grad_norm_history = []
        loss_history = []
        
        print(f"Adam optimization (Sparse Block): {len(p_init)} parameters, max_iter={max_iter}, lr={step_size}")
        
        iter_times = []

        for it in range(1, max_iter + 1):
            iter_start_time = time.time()
            
            # 1. Forward Simulation
            # Update solver parameters
            solver.update_parameters(np.asarray(p_current))
            
            sol = solver.solve_augmented(t_span, ncp=ncp)
            
            # 2. Compute Gradient
            total_grad = self.compute_total_gradient(sol, p_current, target_times, target_data, blend_sharpness, soft_interp=soft_interp)
            total_grad.block_until_ready()
            
            # Mask gradient for non-optimized parameters
            grad_active = jnp.zeros_like(p_current)
            # This logic assumes we want to zero out gradients for fixed params?
            # Or does p_current include all params? 
            # The interface implies p_init matches full params or subset?
            # Typically p_init is full params.
            # We fix non-opt params by zeroing their gradient.
            
            # Create a mask
            mask = np.zeros(len(p_current))
            mask[opt_param_indices] = 1.0
            mask = jnp.array(mask)
            
            final_grad = total_grad * mask
            
            grad_norm = jnp.linalg.norm(final_grad)
            grad_norm_history.append(float(grad_norm))
            loss_history.append(0.0) # Placeholder as loss calculation is inside JIT
            
            iter_time = time.time() - iter_start_time
            iter_times.append(iter_time * 1000.0)
            
            p_print = p_current[jnp.array(opt_param_indices)]
            if it % print_every == 0 or it == 1:
                print(f"  Iter {it:4d} | |grad|={grad_norm:.6e} | p={p_print} | {iter_time*1000:.1f} ms")
            
            if grad_norm < tol:
                print(f"Converged at iteration {it}")
                converged = True
                break
                
            # 3. Adam Update
            m = beta1 * m + (1 - beta1) * final_grad
            v = beta2 * v + (1 - beta2) * (final_grad ** 2)
            
            m_hat = m / (1 - beta1 ** it)
            v_hat = v / (1 - beta2 ** it)
            
            p_current = p_current - step_size * m_hat / (jnp.sqrt(v_hat) + epsilon)
            
        # Calculate average iteration time (excluding first)
        if len(iter_times) > 1:
            avg_iter_time = sum(iter_times[1:]) / (len(iter_times) - 1)
        else:
            avg_iter_time = 0.0
            
        return {
            'p_opt': p_current,
            'n_iter': it,
            'converged': converged,
            'grad_norm_history': grad_norm_history,
            'loss_history': loss_history,
            'avg_iter_time': avg_iter_time
        }
