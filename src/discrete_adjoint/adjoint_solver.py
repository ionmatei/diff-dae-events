"""
Adjoint System Solver for DAE Sensitivity Analysis

Solves the block-bidiagonal transpose linear system arising from the adjoint equations
of a DAE discretized with the trapezoidal method.

The system structure:
    J_curr[0].T @ λ[0] + J_prev[1].T @ λ[1] = b[0]
    J_curr[1].T @ λ[1] + J_prev[2].T @ λ[2] = b[1]
    ...
    J_curr[N-2].T @ λ[N-2] + J_prev[N-1].T @ λ[N-1] = b[N-2]
    J_curr[N-1].T @ λ[N-1] = b[N-1]

where:
    - J_curr[k] = dr_{k+1}/dy_{k+1} from compute_jacobian_blocks_jit
    - J_prev[k] = dr_{k+1}/dy_k from compute_jacobian_blocks_jit
    - λ[k] are the adjoint variables (Lagrange multipliers)
    - b[k] is the right-hand side vector

Solution method: Block backward substitution using JAX lax.scan for efficiency.

Author: Generated for DAE sensitivity-free optimization
"""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Union
import numpy as np


def solve_adjoint_system(
    J_prev_list: Union[List[np.ndarray], jnp.ndarray],
    J_curr_list: Union[List[np.ndarray], jnp.ndarray],
    b: jnp.ndarray
) -> jnp.ndarray:
    """
    Solve the adjoint system using Jacobian blocks from compute_jacobian_blocks_jit.

    The adjoint system arises from differentiating the discretized DAE residuals:
        r_k(y_{k-1}, y_k, params) = 0  for k=1..N

    Taking transpose of the Jacobian gives the adjoint system:
        J_curr[k-1].T @ λ[k-1] + J_prev[k].T @ λ[k] = b[k-1]  for k=1..N-1
        J_curr[N-1].T @ λ[N-1] = b[N-1]

    Args:
        J_prev_list: List of N Jacobian matrices where J_prev[i] = dr_{i+1}/dy_i
                     from compute_jacobian_blocks_jit (first output)
                     Shape: list of (m, m) arrays or array of shape (N, m, m)

        J_curr_list: List of N Jacobian matrices where J_curr[i] = dr_{i+1}/dy_{i+1}
                     from compute_jacobian_blocks_jit (second output)
                     Shape: list of (m, m) arrays or array of shape (N, m, m)

        b: Right-hand side vectors, shape (N, m)
           Each b[k] typically comes from output sensitivity: ∂L/∂y_k

    Returns:
        λ: Solution vectors (adjoint variables), shape (N, m)

    Example:
        >>> from dae_jacobian import DAEJacobian
        >>> jac = DAEJacobian(dae_data)
        >>> J_prev_list, J_curr_list = jac.compute_jacobian_blocks_jit(t_array, y_array)
        >>>
        >>> # Define RHS (e.g., for output sensitivity)
        >>> b = jnp.zeros((N, m))
        >>> b = b.at[-1].set(dL_dy_final)  # Terminal cost gradient
        >>>
        >>> # Solve adjoint system
        >>> lam = solve_adjoint_system(J_prev_list, J_curr_list, b)

    Notes:
        - Indexing: J_prev[k] and J_curr[k] both correspond to residual r_{k+1}
        - The solve proceeds backward from k=N-1 to k=0
        - Complexity: O(N * m^3) where N is time steps, m is state dimension
        - Uses JAX lax.scan for efficient compilation and execution
    """
    # Convert lists to JAX arrays if needed
    if isinstance(J_prev_list, list):
        J_prev = jnp.array(J_prev_list)
    else:
        J_prev = J_prev_list

    if isinstance(J_curr_list, list):
        J_curr = jnp.array(J_curr_list)
    else:
        J_curr = J_curr_list

    N, m, _ = J_curr.shape

    # Validate inputs
    assert J_prev.shape == (N, m, m), \
        f"J_prev shape mismatch: expected ({N}, {m}, {m}), got {J_prev.shape}"
    assert b.shape == (N, m), \
        f"b shape mismatch: expected ({N}, {m}), got {b.shape}"

    # Terminal solve: J_curr[N-1].T @ λ[N-1] = b[N-1]
    lam_N = jnp.linalg.solve(J_curr[-1].T, b[-1])

    # Backward substitution for k = N-2, N-3, ..., 0
    def backward_step(lam_next, inputs):
        """
        Single backward substitution step.

        Solves: J_curr[k].T @ λ[k] + J_prev[k+1].T @ λ[k+1] = b[k]
                => λ[k] = (J_curr[k].T)^{-1} @ (b[k] - J_prev[k+1].T @ λ[k+1])

        Args:
            lam_next: λ[k+1] from previous iteration (shape: m)
            inputs: Tuple of (J_curr[k], J_prev[k+1], b[k])

        Returns:
            lam_k: Current solution λ[k] (carry for next iteration)
            lam_k: Current solution λ[k] (output to collect)
        """
        J_curr_k, J_prev_kp1, b_k = inputs

        # Compute modified RHS: b[k] - J_prev[k+1].T @ λ[k+1]
        rhs = b_k - J_prev_kp1.T @ lam_next

        # Solve J_curr[k].T @ λ[k] = rhs
        lam_k = jnp.linalg.solve(J_curr_k.T, rhs)

        return lam_k, lam_k

    # Prepare inputs in reverse order (k = N-2 down to 0)
    # For k = N-2, we need J_curr[N-2], J_prev[N-1], b[N-2]
    # For k = N-3, we need J_curr[N-3], J_prev[N-2], b[N-3]
    # ...
    # For k = 0, we need J_curr[0], J_prev[1], b[0]

    inputs_reversed = (
        J_curr[:-1][::-1],      # J_curr[N-2], J_curr[N-3], ..., J_curr[0]
        J_prev[1:][::-1],       # J_prev[N-1], J_prev[N-2], ..., J_prev[1]
        b[:-1][::-1]            # b[N-2], b[N-3], ..., b[0]
    )

    # Run backward scan
    _, lam_reversed = jax.lax.scan(backward_step, lam_N, inputs_reversed)

    # Concatenate: [λ[0], λ[1], ..., λ[N-2], λ[N-1]]
    lam = jnp.concatenate([lam_reversed[::-1], lam_N[None, :]], axis=0)

    return lam


def solve_adjoint_system_multiple_rhs(
    J_prev_list: Union[List[np.ndarray], jnp.ndarray],
    J_curr_list: Union[List[np.ndarray], jnp.ndarray],
    B: jnp.ndarray
) -> jnp.ndarray:
    """
    Solve adjoint system for multiple right-hand sides simultaneously.

    Useful for computing sensitivities with respect to multiple outputs or
    parameters in a single pass using vectorization.

    Args:
        J_prev_list: Jacobian blocks dr_{k+1}/dy_k, shape (N, m, m)
        J_curr_list: Jacobian blocks dr_{k+1}/dy_{k+1}, shape (N, m, m)
        B: Multiple right-hand side vectors, shape (N, m, n_rhs)

    Returns:
        Λ: Solution vectors for each RHS, shape (N, m, n_rhs)

    Example:
        >>> # Solve for sensitivities w.r.t. multiple output components
        >>> n_outputs = 5
        >>> B = jnp.zeros((N, m, n_outputs))
        >>> for i in range(n_outputs):
        >>>     B = B.at[-1, i, i].set(1.0)  # Sensitivity to each output
        >>>
        >>> LAM = solve_adjoint_system_multiple_rhs(J_prev_list, J_curr_list, B)
    """
    # Convert to arrays if needed
    if isinstance(J_prev_list, list):
        J_prev = jnp.array(J_prev_list)
    else:
        J_prev = J_prev_list

    if isinstance(J_curr_list, list):
        J_curr = jnp.array(J_curr_list)
    else:
        J_curr = J_curr_list

    N, m, _ = J_curr.shape
    n_rhs = B.shape[2]

    # Vectorize solve over the last dimension (n_rhs)
    # B has shape (N, m, n_rhs), we want to vmap over n_rhs dimension
    def solve_single_rhs(b_single):
        # b_single has shape (N, m)
        return solve_adjoint_system(J_prev, J_curr, b_single)

    # Transpose B to (n_rhs, N, m) for vmapping
    B_transposed = jnp.transpose(B, (2, 0, 1))

    # vmap over first dimension (n_rhs)
    solve_fn = jax.vmap(solve_single_rhs, in_axes=0, out_axes=0)

    # LAM_transposed has shape (n_rhs, N, m)
    LAM_transposed = solve_fn(B_transposed)

    # Transpose back to (N, m, n_rhs)
    LAM = jnp.transpose(LAM_transposed, (1, 2, 0))

    return LAM


def compute_parameter_sensitivity(
    J_prev_list: Union[List[np.ndarray], jnp.ndarray],
    J_curr_list: Union[List[np.ndarray], jnp.ndarray],
    J_param_list: Union[List[np.ndarray], jnp.ndarray],
    dL_dy: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute sensitivity of objective L w.r.t. parameters using adjoint method.

    Given an objective L(y_1, ..., y_N, p), computes dL/dp using:
        1. Solve adjoint system: J.T @ λ = dL/dy
        2. Compute: dL/dp = ∂L/∂p - λ.T @ J_param

    Args:
        J_prev_list: Jacobian blocks dr_{k+1}/dy_k, shape (N, m, m)
        J_curr_list: Jacobian blocks dr_{k+1}/dy_{k+1}, shape (N, m, m)
        J_param_list: Parameter Jacobian blocks dr_{k+1}/dp, shape (N, m, n_params)
                      from DAEJacobian.compute_parameter_jacobian()
        dL_dy: Gradient of objective w.r.t. states, shape (N, m)
               Typically dL_dy[-1] = ∂L/∂y_final, others zero for terminal cost

    Returns:
        dL_dp: Sensitivity of objective w.r.t. parameters, shape (n_params,)

    Example:
        >>> from dae_jacobian import DAEJacobian
        >>> jac = DAEJacobian(dae_data)
        >>>
        >>> # Compute Jacobians
        >>> J_prev, J_curr = jac.compute_jacobian_blocks_jit(t, y_sol)
        >>> J_param = jac.compute_parameter_jacobian(t, y_sol)
        >>>
        >>> # Define objective gradient (e.g., minimize final state[0])
        >>> dL_dy = jnp.zeros((N, m))
        >>> dL_dy = dL_dy.at[-1, 0].set(1.0)
        >>>
        >>> # Compute parameter sensitivity
        >>> dL_dp = compute_parameter_sensitivity(J_prev, J_curr, J_param, dL_dy)
    """
    # Convert to arrays if needed
    if isinstance(J_param_list, list):
        J_param = jnp.array(J_param_list)
    else:
        J_param = J_param_list

    # Solve adjoint system
    lam = solve_adjoint_system(J_prev_list, J_curr_list, dL_dy)

    # Compute dL/dp = -λ.T @ J_param
    # Sum over all time steps: each J_param[k] contributes to the gradient
    dL_dp = jnp.zeros(J_param.shape[2])  # n_params

    for k in range(len(J_param)):
        dL_dp -= lam[k].T @ J_param[k]

    return dL_dp


def verify_adjoint_solution(
    J_prev_list: Union[List[np.ndarray], jnp.ndarray],
    J_curr_list: Union[List[np.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    lam: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Verify the adjoint solution by computing residual norms.

    Checks: ||J_curr[k].T @ λ[k] + J_prev[k+1].T @ λ[k+1] - b[k]||

    Args:
        J_prev_list: Jacobian blocks dr_{k+1}/dy_k
        J_curr_list: Jacobian blocks dr_{k+1}/dy_{k+1}
        b: Right-hand side vectors
        lam: Computed adjoint solution

    Returns:
        residuals: Residual vector at each time step, shape (N, m)
        max_residual: Maximum residual norm across all time steps

    Note:
        This function is JIT-compatible. For verbose output, use the
        print_verification_results() helper function after calling this.
    """
    # Convert to arrays if needed
    if isinstance(J_prev_list, list):
        J_prev = jnp.array(J_prev_list)
    else:
        J_prev = J_prev_list

    if isinstance(J_curr_list, list):
        J_curr = jnp.array(J_curr_list)
    else:
        J_curr = J_curr_list

    N, m, _ = J_curr.shape

    # Vectorized computation using vmap for better stability
    # For k = 0 to N-2: residual[k] = J_curr[k].T @ lam[k] + J_prev[k+1].T @ lam[k+1] - b[k]
    def compute_residual_interior(k):
        return J_curr[k].T @ lam[k] + J_prev[k + 1].T @ lam[k + 1] - b[k]

    # Compute interior residuals (k = 0 to N-2)
    k_indices = jnp.arange(N - 1)
    residuals_interior = jax.vmap(compute_residual_interior)(k_indices)

    # Terminal residual (k = N-1)
    residual_terminal = J_curr[-1].T @ lam[-1] - b[-1]

    # Combine all residuals
    residuals = jnp.concatenate([residuals_interior, residual_terminal[None, :]], axis=0)

    # Compute norms
    residual_norms = jnp.linalg.norm(residuals, axis=1)
    max_residual = jnp.max(residual_norms)

    return residuals, max_residual


def print_verification_results(residuals: jnp.ndarray, max_residual: float):
    """
    Print verification results (non-JIT helper function).

    Args:
        residuals: Residual vectors from verify_adjoint_solution
        max_residual: Maximum residual norm
    """
    residual_norms = jnp.linalg.norm(residuals, axis=1)
    mean_residual = jnp.mean(residual_norms)

    print(f"Adjoint solution verification:")
    print(f"  Max residual:  {max_residual:.6e}")
    print(f"  Mean residual: {mean_residual:.6e}")

    if max_residual < 1e-10:
        print(f"  ✓ Solution verified successfully!")
    elif max_residual < 1e-6:
        print(f"  ⚠ Solution acceptable (residual < 1e-6)")
    else:
        print(f"  ✗ Solution may be inaccurate (residual > 1e-6)")


# JIT-compiled versions for production use
solve_adjoint_system_jit = jax.jit(solve_adjoint_system)
solve_adjoint_system_multiple_rhs_jit = jax.jit(solve_adjoint_system_multiple_rhs)
compute_parameter_sensitivity_jit = jax.jit(compute_parameter_sensitivity)
verify_adjoint_solution_jit = jax.jit(verify_adjoint_solution)


if __name__ == "__main__":
    """
    Test the adjoint solver with DAE Jacobian blocks.
    """
    import sys
    import os

    # Add parent directory to path to import dae_jacobian
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("=" * 80)
    print("Testing Adjoint System Solver")
    print("=" * 80)

    # Test 1: Simple synthetic example
    print("\n" + "=" * 80)
    print("Test 1: Synthetic Example")
    print("=" * 80)

    N = 10   # Number of time steps
    m = 5    # State dimension

    # Generate random well-conditioned Jacobian blocks
    key = jax.random.PRNGKey(42)
    key, *subkeys = jax.random.split(key, 4)

    J_curr = jax.random.normal(subkeys[0], (N, m, m)) + jnp.eye(m) * 5.0
    J_prev = jax.random.normal(subkeys[1], (N, m, m)) * 0.5
    b = jax.random.normal(subkeys[2], (N, m))

    print(f"\nProblem size:")
    print(f"  N (time steps): {N}")
    print(f"  m (state dimension): {m}")
    print(f"  Total unknowns: {N * m}")

    # Solve adjoint system
    print("\nSolving adjoint system...")
    lam = solve_adjoint_system_jit(J_prev, J_curr, b)

    print(f"Solution computed!")
    print(f"  Solution shape: {lam.shape}")
    print(f"  Solution norm: {jnp.linalg.norm(lam):.6e}")

    # Verify solution
    print("\nVerifying solution...")
    residuals, max_res = verify_adjoint_solution_jit(J_prev, J_curr, b, lam)
    print_verification_results(residuals, max_res)

    # Test 2: Multiple RHS
    print("\n" + "=" * 80)
    print("Test 2: Multiple RHS")
    print("=" * 80)

    n_rhs = 3
    B = jax.random.normal(jax.random.PRNGKey(43), (N, m, n_rhs))

    print(f"\nSolving {n_rhs} adjoint systems simultaneously...")
    LAM = solve_adjoint_system_multiple_rhs_jit(J_prev, J_curr, B)

    print(f"  Solutions shape: {LAM.shape}")
    print(f"  Expected shape: ({N}, {m}, {n_rhs})")

    # Verify each RHS
    for i in range(n_rhs):
        print(f"\nVerifying RHS {i+1}:")
        residuals_i, max_res_i = verify_adjoint_solution_jit(J_prev, J_curr, B[:, :, i], LAM[:, :, i])
        print_verification_results(residuals_i, max_res_i)

    # Test 3: Parameter sensitivity
    print("\n" + "=" * 80)
    print("Test 3: Parameter Sensitivity")
    print("=" * 80)

    n_params = 4
    J_param = jax.random.normal(jax.random.PRNGKey(44), (N, m, n_params)) * 0.1

    # Define terminal cost: L = y_N[0]
    dL_dy = jnp.zeros((N, m))
    dL_dy = dL_dy.at[-1, 0].set(1.0)

    print(f"\nComputing sensitivity to {n_params} parameters...")
    print(f"Objective: L = y_final[0] (minimize first state at final time)")

    dL_dp = compute_parameter_sensitivity_jit(J_prev, J_curr, J_param, dL_dy)

    print(f"\nParameter sensitivities:")
    for i, sens in enumerate(dL_dp):
        print(f"  dL/dp[{i}] = {sens:.6e}")

    # Test 4: Integration with DAE Jacobian (if available)
    print("\n" + "=" * 80)
    print("Test 4: Integration with DAE Jacobian")
    print("=" * 80)

    try:
        from dae_jacobian import DAEJacobian
        from dae_solver import DAESolver
        import json

        # Load a simple DAE example
        json_path = "../dae_examples/dae_specification_smooth.json"

        if os.path.exists(json_path):
            print(f"\nLoading DAE from: {json_path}")

            with open(json_path, 'r') as f:
                dae_data = json.load(f)

            # Create Jacobian object
            jac = DAEJacobian(dae_data)

            # Create solver to get a trajectory
            solver = DAESolver(dae_data)
            result = solver.solve(t_span=(0.0, 1.0), ncp=50)

            t_array = result['t']
            y_array = np.vstack([result['x'], result['z']])  # (m, N+1)

            print(f"Trajectory computed:")
            print(f"  Time points: {len(t_array)}")
            print(f"  State dimension: {y_array.shape[0]}")

            # Compute Jacobian blocks
            print("\nComputing Jacobian blocks...")
            J_prev_list, J_curr_list = jac.compute_jacobian_blocks_jit(t_array, y_array)

            print(f"Jacobian blocks computed:")
            print(f"  Number of blocks: {len(J_curr_list)}")
            print(f"  Block size: {J_curr_list[0].shape}")

            # Define a simple objective: minimize final state
            N_dae = len(J_curr_list)
            m_dae = J_curr_list[0].shape[0]

            dL_dy_dae = jnp.zeros((N_dae, m_dae))
            dL_dy_dae = dL_dy_dae.at[-1, :jac.n_states].set(1.0)  # Gradient w.r.t. differential states

            print(f"\nSolving adjoint system for DAE...")
            lam_dae = solve_adjoint_system_jit(J_prev_list, J_curr_list, dL_dy_dae)

            print(f"Adjoint solution:")
            print(f"  Shape: {lam_dae.shape}")
            print(f"  Norm: {jnp.linalg.norm(lam_dae):.6e}")

            # Verify
            print("\nVerifying DAE adjoint solution...")
            residuals_dae, max_res_dae = verify_adjoint_solution_jit(
                J_prev_list, J_curr_list, dL_dy_dae, lam_dae
            )
            print_verification_results(residuals_dae, max_res_dae)

            # Compute parameter sensitivity
            print("\nComputing parameter sensitivity for DAE...")
            J_param_list = jac.compute_parameter_jacobian(t_array, y_array)

            dL_dp_dae = compute_parameter_sensitivity_jit(
                J_prev_list, J_curr_list, J_param_list, dL_dy_dae
            )

            print(f"\nParameter sensitivities for DAE:")
            param_names = [p['name'] for p in dae_data['parameters']]
            for i, (name, sens) in enumerate(zip(param_names, dL_dp_dae)):
                print(f"  dL/d{name} = {sens:.6e}")

            print("\n✓ DAE integration test completed successfully!")

        else:
            print(f"DAE example file not found: {json_path}")
            print("Skipping DAE integration test.")

    except ImportError as e:
        print(f"Could not import DAE modules: {e}")
        print("Skipping DAE integration test.")
    except Exception as e:
        print(f"Error in DAE integration test: {e}")
        import traceback
        traceback.print_exc()

    # Benchmark
    print("\n" + "=" * 80)
    print("Benchmark: Large-Scale Problem")
    print("=" * 80)

    N_large = 1000
    m_large = 20

    print(f"\nProblem size:")
    print(f"  N = {N_large} time steps")
    print(f"  m = {m_large} state dimension")
    print(f"  Total unknowns: {N_large * m_large} = {N_large * m_large}")

    # Generate large problem
    key = jax.random.PRNGKey(100)
    key, *subkeys = jax.random.split(key, 4)

    J_curr_large = jax.random.normal(subkeys[0], (N_large, m_large, m_large)) + jnp.eye(m_large) * 5.0
    J_prev_large = jax.random.normal(subkeys[1], (N_large, m_large, m_large)) * 0.5
    b_large = jax.random.normal(subkeys[2], (N_large, m_large))

    # Warm-up
    print("\nWarming up JIT compilation...")
    _ = solve_adjoint_system_jit(J_prev_large, J_curr_large, b_large).block_until_ready()

    # Benchmark
    print("Running benchmark...")
    import time
    n_trials = 10
    times = []

    for _ in range(n_trials):
        start = time.time()
        lam_large = solve_adjoint_system_jit(J_prev_large, J_curr_large, b_large).block_until_ready()
        times.append(time.time() - start)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nBenchmark results ({n_trials} trials):")
    print(f"  Average time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  Throughput: {N_large / avg_time:.0f} time steps/second")
    print(f"  Time per block solve: {avg_time/N_large*1e6:.2f} μs")

    # Verify large solution
    print("\nVerifying large-scale solution...")
    residuals_large, max_res_large = verify_adjoint_solution_jit(
        J_prev_large, J_curr_large, b_large, lam_large
    )
    print_verification_results(residuals_large, max_res_large)

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
