"""
Fast Adjoint System Solver using Parallel Associative Scan

This is an improved version of adjoint_solver.py that uses the parallel
associative scan technique from DEER for O(log N) parallel depth instead
of O(N) sequential operations.

The adjoint system:
    J_curr[k].T @ λ[k] + J_prev[k+1].T @ λ[k+1] = b[k]  for k=0..N-2
    J_curr[N-1].T @ λ[N-1] = b[N-1]

Can be rewritten as a backward recurrence:
    λ[k] = A[k] @ λ[k+1] + c[k]

where:
    A[k] = -(J_curr[k].T)^{-1} @ J_prev[k+1].T
    c[k] = (J_curr[k].T)^{-1} @ b[k]

This recurrence can be solved in O(log N) parallel steps using associative scan.

Performance improvements over adjoint_solver.py:
1. Parallel associative scan: O(log N) depth vs O(N) sequential
2. Batch matrix inversions using vmap
3. Vectorized parameter sensitivity computation
4. Pre-transposed Jacobians to avoid repeated transposes
"""

import jax
import jax.numpy as jnp
from jax import vmap
from typing import List, Tuple, Union
from functools import partial
import numpy as np


# ============================================================================
# Associative Scan Infrastructure (adapted from DEER)
# ============================================================================

def _scan_binop_adjoint(elem_i: Tuple[jnp.ndarray, jnp.ndarray],
                        elem_j: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative binary operator for adjoint recurrence.

    For recurrence: y[i] = A[i] @ y[i+1] + c[i]

    Combining (A_i, c_i) with (A_j, c_j) where j = i+1:
        y[i] = A_i @ y[i+1] + c_i
             = A_i @ (A_j @ y[i+2] + c_j) + c_i
             = (A_i @ A_j) @ y[i+2] + (A_i @ c_j + c_i)

    So the combined element is (A_i @ A_j, A_i @ c_j + c_i)
    """
    A_i, c_i = elem_i
    A_j, c_j = elem_j
    A_combined = A_i @ A_j
    c_combined = A_i @ c_j + c_i
    return A_combined, c_combined


def _interleave(a, b, axis):
    """Interleave two tensors along given axis."""
    from jax._src.lax import lax
    assert a.shape[axis] == b.shape[axis] or a.shape[axis] == b.shape[axis] + 1
    a_pad = [(0, 0, 0)] * a.ndim
    b_pad = [(0, 0, 0)] * b.ndim
    a_pad[axis] = (0, 1 if a.shape[axis] == b.shape[axis] else 0, 1)
    b_pad[axis] = (1, 0 if a.shape[axis] == b.shape[axis] else 1, 1)
    op = jax.lax.bitwise_or if a.dtype == jnp.bool_ else jax.lax.add
    return op(jax.lax.pad(a, lax._const(a, 0), a_pad),
              jax.lax.pad(b, lax._const(b, 0), b_pad))


def associative_scan_reverse(fn, elems, axis: int = 0):
    """
    Associative scan in reverse order (for backward recurrences).

    This is a simplified version that handles the backward adjoint solve.
    Uses direct indexing to avoid JAX slice_in_dim bugs.
    """
    from jax._src import util

    elems_flat, tree = jax.tree_util.tree_flatten(elems)

    # Reverse inputs
    elems_flat = [jax.lax.rev(elem, [axis]) for elem in elems_flat]

    def combine(a_flat, b_flat):
        a = jax.tree_util.tree_unflatten(tree, a_flat)
        b = jax.tree_util.tree_unflatten(tree, b_flat)
        c = fn(a, b)
        c_flat, _ = jax.tree_util.tree_flatten(c)
        return c_flat

    num_elems = int(elems_flat[0].shape[axis])

    def get_idxs(elem, slc):
        lst = [slice(None, None, None) for _ in range(len(elem.shape))]
        lst[axis] = slc
        return tuple(lst)

    def _scan(elems):
        num_elems = elems[0].shape[axis]

        if num_elems < 2:
            return elems

        # Combine adjacent pairs
        reduced_elems = combine(
            [elem[get_idxs(elem, slice(0, -1, 2))] for elem in elems],
            [elem[get_idxs(elem, slice(1, None, 2))] for elem in elems])

        # Recursively scan
        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = combine(
                [e[get_idxs(e, slice(0, -1, None))] for e in odd_elems],
                [e[get_idxs(e, slice(2, None, 2))] for e in elems])
        else:
            even_elems = combine(
                odd_elems,
                [e[get_idxs(e, slice(2, None, 2))] for e in elems])

        even_elems = [
            jax.lax.concatenate([elem[get_idxs(elem, slice(0, 1, None))], result],
                                dimension=axis)
            for (elem, result) in zip(elems, even_elems)]

        return list(util.safe_map(partial(_interleave, axis=axis), even_elems, odd_elems))

    scans = _scan(elems_flat)

    # Reverse outputs
    scans = [jax.lax.rev(scanned, [axis]) for scanned in scans]

    return jax.tree_util.tree_unflatten(tree, scans)


# ============================================================================
# Fast Adjoint Solver
# ============================================================================

def solve_adjoint_system_fast(
    J_prev_list: Union[List[np.ndarray], jnp.ndarray],
    J_curr_list: Union[List[np.ndarray], jnp.ndarray],
    b: jnp.ndarray
) -> jnp.ndarray:
    """
    Solve adjoint system using parallel associative scan.

    This is O(log N) parallel depth vs O(N) sequential in the original.

    The adjoint system:
        J_curr[k].T @ λ[k] + J_prev[k+1].T @ λ[k+1] = b[k]  for k=0..N-2
        J_curr[N-1].T @ λ[N-1] = b[N-1]

    Args:
        J_prev_list: Jacobian blocks dr_{k+1}/dy_k, shape (N, m, m)
        J_curr_list: Jacobian blocks dr_{k+1}/dy_{k+1}, shape (N, m, m)
        b: Right-hand side vectors, shape (N, m)

    Returns:
        λ: Adjoint solution, shape (N, m)
    """
    # Convert to arrays
    if isinstance(J_prev_list, list):
        J_prev = jnp.array(J_prev_list)
    else:
        J_prev = J_prev_list

    if isinstance(J_curr_list, list):
        J_curr = jnp.array(J_curr_list)
    else:
        J_curr = J_curr_list

    N, m, _ = J_curr.shape

    # Pre-transpose all Jacobians (batch operation)
    J_curr_T = jnp.swapaxes(J_curr, -2, -1)  # (N, m, m)
    J_prev_T = jnp.swapaxes(J_prev, -2, -1)  # (N, m, m)

    # Compute inverses of J_curr_T using batched solve
    # J_curr_T_inv[k] = (J_curr[k].T)^{-1}
    eye_batch = jnp.broadcast_to(jnp.eye(m), (N, m, m))
    J_curr_T_inv = vmap(jnp.linalg.solve)(J_curr_T, eye_batch)  # (N, m, m)

    # Compute recurrence coefficients:
    # A[k] = -(J_curr[k].T)^{-1} @ J_prev[k+1].T  for k=0..N-2
    # c[k] = (J_curr[k].T)^{-1} @ b[k]  for k=0..N-1

    # c[k] for all k
    c = vmap(lambda inv, bi: inv @ bi)(J_curr_T_inv, b)  # (N, m)

    # A[k] for k=0..N-2: need J_curr_T_inv[k] and J_prev_T[k+1]
    # A has shape (N-1, m, m)
    A = -vmap(lambda inv, jp: inv @ jp)(J_curr_T_inv[:-1], J_prev_T[1:])  # (N-1, m, m)

    # Terminal condition: λ[N-1] = c[N-1]
    lam_N = c[-1]

    # For k=0..N-2, we have recurrence: λ[k] = A[k] @ λ[k+1] + c[k]
    # Use associative scan to solve this in parallel

    if N == 1:
        return lam_N[None, :]

    # Prepare elements for associative scan
    # We need to compute the cumulative effect from each position to the end
    # Element k represents (A[k], c[k])

    # Add identity at the end for terminal condition
    eye = jnp.eye(m)[None, :]  # (1, m, m)
    A_extended = jnp.concatenate([A, jnp.zeros((1, m, m))], axis=0)  # (N, m, m)
    # The last element should be identity matrix and lam_N
    A_extended = A_extended.at[-1].set(jnp.eye(m))

    c_extended = c.at[-1].set(lam_N)  # Replace c[N-1] with the actual terminal value

    # Run reverse associative scan
    # After scan, each position k will have the cumulative transformation from k to N-1
    elems = (A_extended, c_extended)
    _, lam = associative_scan_reverse(_scan_binop_adjoint, elems, axis=0)

    return lam


def solve_adjoint_system_scan(
    J_prev_list: Union[List[np.ndarray], jnp.ndarray],
    J_curr_list: Union[List[np.ndarray], jnp.ndarray],
    b: jnp.ndarray
) -> jnp.ndarray:
    """
    Solve adjoint system using optimized sequential scan.

    This version pre-computes all matrix operations in batch before the scan,
    making the scan itself much lighter.

    Args:
        J_prev_list: Jacobian blocks dr_{k+1}/dy_k, shape (N, m, m)
        J_curr_list: Jacobian blocks dr_{k+1}/dy_{k+1}, shape (N, m, m)
        b: Right-hand side vectors, shape (N, m)

    Returns:
        λ: Adjoint solution, shape (N, m)
    """
    # Convert to arrays
    if isinstance(J_prev_list, list):
        J_prev = jnp.array(J_prev_list)
    else:
        J_prev = J_prev_list

    if isinstance(J_curr_list, list):
        J_curr = jnp.array(J_curr_list)
    else:
        J_curr = J_curr_list

    N, m, _ = J_curr.shape

    # Pre-transpose all Jacobians
    J_curr_T = jnp.swapaxes(J_curr, -2, -1)
    J_prev_T = jnp.swapaxes(J_prev, -2, -1)

    # Batch compute all LU factorizations for J_curr_T
    # We'll use solve directly but batch the setup

    # Pre-compute: c[k] = (J_curr[k].T)^{-1} @ b[k]
    c = vmap(jnp.linalg.solve)(J_curr_T, b)  # (N, m)

    # Pre-compute: M[k] = -(J_curr[k].T)^{-1} @ J_prev[k+1].T for k=0..N-2
    # This is the matrix that multiplies λ[k+1] to contribute to λ[k]
    def compute_M(J_curr_T_k, J_prev_T_kp1):
        return -jnp.linalg.solve(J_curr_T_k, J_prev_T_kp1)

    M = vmap(compute_M)(J_curr_T[:-1], J_prev_T[1:])  # (N-1, m, m)

    # Terminal: λ[N-1] = c[N-1]
    lam_N = c[-1]

    # Backward scan: λ[k] = M[k] @ λ[k+1] + c[k]
    def backward_step(lam_next, inputs):
        M_k, c_k = inputs
        lam_k = M_k @ lam_next + c_k
        return lam_k, lam_k

    # Prepare reversed inputs
    inputs_reversed = (M[::-1], c[:-1][::-1])

    _, lam_reversed = jax.lax.scan(backward_step, lam_N, inputs_reversed)

    # Combine results
    lam = jnp.concatenate([lam_reversed[::-1], lam_N[None, :]], axis=0)

    return lam


def solve_adjoint_system_multiple_rhs_fast(
    J_prev_list: Union[List[np.ndarray], jnp.ndarray],
    J_curr_list: Union[List[np.ndarray], jnp.ndarray],
    B: jnp.ndarray
) -> jnp.ndarray:
    """
    Solve adjoint system for multiple RHS using vectorization.

    Args:
        J_prev_list: Jacobian blocks, shape (N, m, m)
        J_curr_list: Jacobian blocks, shape (N, m, m)
        B: Multiple RHS vectors, shape (N, m, n_rhs)

    Returns:
        Λ: Solutions, shape (N, m, n_rhs)
    """
    # Convert to arrays
    if isinstance(J_prev_list, list):
        J_prev = jnp.array(J_prev_list)
    else:
        J_prev = J_prev_list

    if isinstance(J_curr_list, list):
        J_curr = jnp.array(J_curr_list)
    else:
        J_curr = J_curr_list

    # Vectorize over RHS dimension
    def solve_single(b_single):
        return solve_adjoint_system_scan(J_prev, J_curr, b_single)

    # B: (N, m, n_rhs) -> transpose to (n_rhs, N, m)
    B_transposed = jnp.transpose(B, (2, 0, 1))

    # vmap over first dimension
    LAM_transposed = vmap(solve_single)(B_transposed)  # (n_rhs, N, m)

    # Transpose back to (N, m, n_rhs)
    return jnp.transpose(LAM_transposed, (1, 2, 0))


def compute_parameter_sensitivity_fast(
    J_prev_list: Union[List[np.ndarray], jnp.ndarray],
    J_curr_list: Union[List[np.ndarray], jnp.ndarray],
    J_param_list: Union[List[np.ndarray], jnp.ndarray],
    dL_dy: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute parameter sensitivity using vectorized operations.

    Args:
        J_prev_list: Jacobian blocks dr_{k+1}/dy_k, shape (N, m, m)
        J_curr_list: Jacobian blocks dr_{k+1}/dy_{k+1}, shape (N, m, m)
        J_param_list: Parameter Jacobian dr_{k+1}/dp, shape (N, m, n_params)
        dL_dy: Objective gradient, shape (N, m)

    Returns:
        dL_dp: Parameter sensitivity, shape (n_params,)
    """
    # Convert to arrays
    if isinstance(J_param_list, list):
        J_param = jnp.array(J_param_list)
    else:
        J_param = J_param_list

    # Solve adjoint system
    lam = solve_adjoint_system_scan(J_prev_list, J_curr_list, dL_dy)

    # Vectorized computation: dL/dp = -sum_k λ[k].T @ J_param[k]
    # λ[k] has shape (m,), J_param[k] has shape (m, n_params)
    # λ[k].T @ J_param[k] has shape (n_params,)

    # Use einsum for efficient batch computation
    # lam: (N, m), J_param: (N, m, n_params)
    # Result: sum over k of lam[k, i] * J_param[k, i, j] = (n_params,)
    dL_dp = -jnp.einsum('ki,kij->j', lam, J_param)

    return dL_dp


def verify_adjoint_solution_fast(
    J_prev_list: Union[List[np.ndarray], jnp.ndarray],
    J_curr_list: Union[List[np.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    lam: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Verify adjoint solution using vectorized residual computation.

    Args:
        J_prev_list: Jacobian blocks
        J_curr_list: Jacobian blocks
        b: RHS vectors
        lam: Computed solution

    Returns:
        residuals: Residual at each time step, shape (N, m)
        max_residual: Maximum residual norm
    """
    if isinstance(J_prev_list, list):
        J_prev = jnp.array(J_prev_list)
    else:
        J_prev = J_prev_list

    if isinstance(J_curr_list, list):
        J_curr = jnp.array(J_curr_list)
    else:
        J_curr = J_curr_list

    N, m, _ = J_curr.shape

    # Pre-transpose
    J_curr_T = jnp.swapaxes(J_curr, -2, -1)
    J_prev_T = jnp.swapaxes(J_prev, -2, -1)

    # Compute J_curr[k].T @ lam[k] for all k
    term1 = vmap(lambda J, l: J @ l)(J_curr_T, lam)  # (N, m)

    # Compute J_prev[k+1].T @ lam[k+1] for k=0..N-2
    term2_interior = vmap(lambda J, l: J @ l)(J_prev_T[1:], lam[1:])  # (N-1, m)

    # Interior residuals: term1[k] + term2[k] - b[k] for k=0..N-2
    residuals_interior = term1[:-1] + term2_interior - b[:-1]

    # Terminal residual: term1[N-1] - b[N-1]
    residual_terminal = term1[-1] - b[-1]

    residuals = jnp.concatenate([residuals_interior, residual_terminal[None, :]], axis=0)

    residual_norms = jnp.linalg.norm(residuals, axis=1)
    max_residual = jnp.max(residual_norms)

    return residuals, max_residual


# ============================================================================
# JIT-compiled versions
# ============================================================================

solve_adjoint_system_fast_jit = jax.jit(solve_adjoint_system_fast)
solve_adjoint_system_scan_jit = jax.jit(solve_adjoint_system_scan)
solve_adjoint_system_multiple_rhs_fast_jit = jax.jit(solve_adjoint_system_multiple_rhs_fast)
compute_parameter_sensitivity_fast_jit = jax.jit(compute_parameter_sensitivity_fast)
verify_adjoint_solution_fast_jit = jax.jit(verify_adjoint_solution_fast)


# ============================================================================
# Benchmark and test
# ============================================================================

if __name__ == "__main__":
    import time

    print("=" * 80)
    print("Fast Adjoint Solver - Benchmark")
    print("=" * 80)

    # Import original for comparison
    from adjoint_solver import (
        solve_adjoint_system_jit as solve_original_jit,
        compute_parameter_sensitivity_jit as compute_sens_original_jit
    )

    # Test sizes
    test_cases = [
        (100, 8),
        (500, 8),
        (1000, 8),
        (1000, 20),
        (2000, 10),
    ]

    for N, m in test_cases:
        print(f"\n{'='*60}")
        print(f"N={N}, m={m} (total unknowns: {N*m})")
        print(f"{'='*60}")

        # Generate test data
        key = jax.random.PRNGKey(42)
        key, *subkeys = jax.random.split(key, 4)

        J_curr = jax.random.normal(subkeys[0], (N, m, m)) + jnp.eye(m) * 10.0
        J_prev = jax.random.normal(subkeys[1], (N, m, m)) * 0.5
        b = jax.random.normal(subkeys[2], (N, m))

        # Warm-up JIT
        _ = solve_original_jit(J_prev, J_curr, b).block_until_ready()
        _ = solve_adjoint_system_scan_jit(J_prev, J_curr, b).block_until_ready()

        # Benchmark original
        n_trials = 10
        times_orig = []
        for _ in range(n_trials):
            start = time.time()
            lam_orig = solve_original_jit(J_prev, J_curr, b).block_until_ready()
            times_orig.append(time.time() - start)

        # Benchmark optimized scan
        times_scan = []
        for _ in range(n_trials):
            start = time.time()
            lam_scan = solve_adjoint_system_scan_jit(J_prev, J_curr, b).block_until_ready()
            times_scan.append(time.time() - start)

        # Verify solutions match
        error = jnp.max(jnp.abs(lam_orig - lam_scan))

        avg_orig = np.mean(times_orig) * 1000
        avg_scan = np.mean(times_scan) * 1000
        speedup = avg_orig / avg_scan

        print(f"\nOriginal solver:   {avg_orig:.3f} ms")
        print(f"Optimized scan:    {avg_scan:.3f} ms")
        print(f"Speedup:           {speedup:.2f}x")
        print(f"Max error:         {error:.2e}")

        # Verify solution correctness
        _, max_res = verify_adjoint_solution_fast_jit(J_prev, J_curr, b, lam_scan)
        print(f"Max residual:      {max_res:.2e}")

    # Test parameter sensitivity
    print(f"\n{'='*60}")
    print("Parameter Sensitivity Benchmark")
    print(f"{'='*60}")

    N, m, n_params = 1000, 10, 20

    key = jax.random.PRNGKey(100)
    key, *subkeys = jax.random.split(key, 5)

    J_curr = jax.random.normal(subkeys[0], (N, m, m)) + jnp.eye(m) * 10.0
    J_prev = jax.random.normal(subkeys[1], (N, m, m)) * 0.5
    J_param = jax.random.normal(subkeys[2], (N, m, n_params)) * 0.1
    dL_dy = jnp.zeros((N, m)).at[-1, 0].set(1.0)

    # Warm-up
    _ = compute_sens_original_jit(J_prev, J_curr, J_param, dL_dy).block_until_ready()
    _ = compute_parameter_sensitivity_fast_jit(J_prev, J_curr, J_param, dL_dy).block_until_ready()

    # Benchmark
    n_trials = 10

    times_orig = []
    for _ in range(n_trials):
        start = time.time()
        sens_orig = compute_sens_original_jit(J_prev, J_curr, J_param, dL_dy).block_until_ready()
        times_orig.append(time.time() - start)

    times_fast = []
    for _ in range(n_trials):
        start = time.time()
        sens_fast = compute_parameter_sensitivity_fast_jit(J_prev, J_curr, J_param, dL_dy).block_until_ready()
        times_fast.append(time.time() - start)

    error = jnp.max(jnp.abs(sens_orig - sens_fast))

    print(f"\nN={N}, m={m}, n_params={n_params}")
    print(f"Original:   {np.mean(times_orig)*1000:.3f} ms")
    print(f"Fast:       {np.mean(times_fast)*1000:.3f} ms")
    print(f"Speedup:    {np.mean(times_orig)/np.mean(times_fast):.2f}x")
    print(f"Max error:  {error:.2e}")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)
