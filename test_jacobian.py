#!/usr/bin/env python3
"""
Test file for DAE Jacobian computation.

Verifies that:
1. Block Jacobians are computed correctly
2. Block Jacobians match the full Jacobian when assembled
3. Vectorized computation is efficient
"""

import json
import numpy as np
import sys
sys.path.insert(0, 'src')

from dae_jacobian import DAEJacobian
from dae_solver import DAESolver

try:
    import jax
    import jax.numpy as jnp
    from jax import jacfwd
except ImportError:
    print("JAX is required for this test")
    sys.exit(1)


def compute_full_jacobian_direct(jac_solver, t_array, y_array, p=None):
    """
    Compute full Jacobian directly using JAX (for verification).

    This computes the Jacobian of R(Y) where:
        R = [r_1(y_0, y_1), r_2(y_1, y_2), ..., r_N(y_{N-1}, y_N)]
        Y = [y_1, y_2, ..., y_N]  (excluding y_0 which is fixed)

    Returns:
        J_full: Jacobian matrix dR/dY, shape (N*n_total, N*n_total)
    """
    if p is None:
        p = jac_solver.p
    p = jnp.array(p)

    if y_array.shape[0] == jac_solver.n_total and y_array.shape[1] == len(t_array):
        y_array = y_array.T

    N = len(t_array) - 1
    n = jac_solver.n_total

    # Define residual function for all intervals
    def residual_full(Y_flat):
        """
        Compute residuals for all intervals.

        Args:
            Y_flat: flattened array of [y_1, y_2, ..., y_N], shape (N*n_total,)

        Returns:
            R: stacked residuals [r_1, r_2, ..., r_N], shape (N*n_total,)
        """
        # Reshape to (N, n_total)
        Y = Y_flat.reshape(N, n)

        # y_0 is fixed (initial condition)
        y_0 = jnp.array(y_array[0])

        residuals = []
        for k in range(N):
            if k == 0:
                y_prev = y_0
            else:
                y_prev = Y[k-1]
            y_curr = Y[k]

            t_k = t_array[k]
            t_kp1 = t_array[k+1]

            r = jac_solver.residual_trapezoidal_single(t_k, t_kp1, y_prev, y_curr, p)
            residuals.append(r)

        return jnp.concatenate(residuals)

    # Flatten Y (excluding y_0)
    Y_flat = jnp.array(y_array[1:].flatten())

    # Compute Jacobian
    J_full = jacfwd(residual_full)(Y_flat)

    return np.array(J_full)


def compute_parameter_jacobian_direct(jac_solver, t_array, y_array, p=None):
    """
    Compute parameter Jacobian directly using JAX (for verification).

    This computes the Jacobian of R(p) where:
        R = [r_1(y_0, y_1, p), r_2(y_1, y_2, p), ..., r_N(y_{N-1}, y_N, p)]

    Returns:
        J_param: Jacobian matrix dR/dp, shape (N*n_total, n_params)
    """
    if p is None:
        p = jac_solver.p
    p = jnp.array(p)

    if y_array.shape[0] == jac_solver.n_total and y_array.shape[1] == len(t_array):
        y_array = y_array.T

    N = len(t_array) - 1
    n = jac_solver.n_total

    # Define residual function with respect to parameters
    def residual_full_param(p_var):
        """
        Compute residuals for all intervals as function of parameters.

        Args:
            p_var: parameter vector

        Returns:
            R: stacked residuals [r_1, r_2, ..., r_N], shape (N*n_total,)
        """
        residuals = []
        for k in range(N):
            y_prev = jnp.array(y_array[k])
            y_curr = jnp.array(y_array[k+1])

            t_k = t_array[k]
            t_kp1 = t_array[k+1]

            r = jac_solver.residual_trapezoidal_single(t_k, t_kp1, y_prev, y_curr, p_var)
            residuals.append(r)

        return jnp.concatenate(residuals)

    # Compute Jacobian with respect to parameters
    J_param = jacfwd(residual_full_param)(p)

    return np.array(J_param)


def test_jacobian_blocks():
    """Test Jacobian block computation."""
    print("="*80)
    print("Testing DAE Jacobian Computation")
    print("="*80)

    # Load DAE
    json_path = "dae_examples/dae_specification_smooth.json"
    print(f"\nLoading DAE from: {json_path}")

    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    # Create Jacobian solver
    print("\nInitializing Jacobian solver...")
    jac_solver = DAEJacobian(dae_data)

    # Solve DAE first to get trajectory
    print("\nSolving DAE to get reference trajectory...")
    solver = DAESolver(dae_data)
    result = solver.solve(t_span=(0.0, 30.0), ncp=10, rtol=1e-4, atol=1e-5)

    t_array = result['t']
    x_array = result['x']
    z_array = result['z']
    y_array = np.vstack([x_array, z_array]).T  # shape: (N+1, n_total)

    N = len(t_array) - 1
    n_total = jac_solver.n_total

    print(f"\nTrajectory info:")
    print(f"  Number of intervals: {N}")
    print(f"  State dimension: {n_total} ({jac_solver.n_states} states + {jac_solver.n_alg} algebraic)")
    print(f"  Time points: {len(t_array)}")
    print(f"  t_array shape: {t_array.shape}")
    print(f"  y_array shape before vstack: x={x_array.shape}, z={z_array.shape}")
    print(f"  y_array shape after vstack: {y_array.shape}")

    # Test 1: Compute block Jacobians
    print("\n" + "-"*80)
    print("Test 1: Computing block Jacobians using vmap")
    print("-"*80)

    import time
    start = time.time()
    J_prev_list, J_curr_list = jac_solver.compute_jacobian_blocks(t_array, y_array)
    elapsed = time.time() - start

    print(f"Computed {len(J_prev_list)} Jacobian blocks in {elapsed:.4f} seconds")

    if len(J_prev_list) == 0:
        print("ERROR: No Jacobian blocks computed!")
        print(f"  N intervals: {N}")
        print(f"  t_array shape: {t_array.shape}")
        print(f"  y_array shape: {y_array.shape}")
        return None, None, None, None

    print(f"J_prev[k] shape: {J_prev_list[0].shape} (where J_prev[k] = dr_{{k+1}}/dy_k)")
    print(f"J_curr[k] shape: {J_curr_list[0].shape} (where J_curr[k] = dr_{{k+1}}/dy_{{k+1}})")

    # Test 1b: Compare with JIT-compiled version
    print("\n" + "-"*80)
    print("Test 1b: Computing block Jacobians using JIT-compiled vmap")
    print("-"*80)

    start = time.time()
    J_prev_list_jit, J_curr_list_jit = jac_solver.compute_jacobian_blocks_jit(t_array, y_array)
    elapsed_jit = time.time() - start

    print(f"Computed {len(J_prev_list_jit)} Jacobian blocks in {elapsed_jit:.4f} seconds")

    # Verify JIT results match regular vmap results
    max_diff_prev = np.max([np.max(np.abs(J_prev_list[i] - J_prev_list_jit[i])) for i in range(N)])
    max_diff_curr = np.max([np.max(np.abs(J_curr_list[i] - J_curr_list_jit[i])) for i in range(N)])

    print(f"\nComparison between vmap and JIT+vmap:")
    print(f"  Regular vmap time:     {elapsed:.4f} seconds")
    print(f"  JIT+vmap time:         {elapsed_jit:.4f} seconds")
    print(f"  Speedup:               {elapsed/elapsed_jit:.2f}x")
    print(f"  Max diff in J_prev:    {max_diff_prev:.6e}")
    print(f"  Max diff in J_curr:    {max_diff_curr:.6e}")

    if max_diff_prev < 1e-10 and max_diff_curr < 1e-10:
        print("  ✓ PASSED: JIT results match regular vmap results")
    else:
        print("  ✗ FAILED: JIT results differ from regular vmap")

    # Test multiple calls to show JIT warmup effect
    print("\n  Testing JIT warmup effect (10 repeated calls):")
    times_jit = []
    for i in range(10):
        start = time.time()
        _ = jac_solver.compute_jacobian_blocks_jit(t_array, y_array)
        times_jit.append(time.time() - start)

    print(f"    First call:  {times_jit[0]:.6f}s (includes JIT compilation)")
    print(f"    Second call: {times_jit[1]:.6f}s")
    print(f"    Average (calls 2-10): {np.mean(times_jit[1:]):.6f}s")
    print(f"    Min time: {min(times_jit[1:]):.6f}s")
    print(f"    Speedup after warmup: {elapsed/np.mean(times_jit[1:]):.2f}x vs regular vmap")

    # Test 2: Verify against direct computation
    print("\n" + "-"*80)
    print("Test 2: Verifying against full Jacobian")
    print("-"*80)

    print("Computing full Jacobian directly (for verification)...")
    start = time.time()
    J_full_direct = compute_full_jacobian_direct(jac_solver, t_array, y_array)
    elapsed_direct = time.time() - start
    print(f"Direct computation took {elapsed_direct:.4f} seconds")
    print(f"Full Jacobian shape: {J_full_direct.shape}")

    print("\nAssembling full Jacobian from blocks...")
    J_full_assembled = jac_solver.assemble_full_jacobian(J_prev_list, J_curr_list)
    print(f"Assembled Jacobian shape: {J_full_assembled.shape}")

    # Compare
    max_diff = np.max(np.abs(J_full_direct - J_full_assembled))
    rel_error = max_diff / (np.max(np.abs(J_full_direct)) + 1e-10)

    print(f"\nComparison:")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Relative error: {rel_error:.6e}")

    if rel_error < 1e-10:
        print("  ✓ PASSED: Block Jacobians match full Jacobian")
    else:
        print("  ✗ FAILED: Block Jacobians do not match")

    # Test 3: Verify block structure
    print("\n" + "-"*80)
    print("Test 3: Verifying block-bidiagonal structure")
    print("-"*80)

    # Extract blocks from full Jacobian and compare
    # Full Jacobian: dR/dY where R=[r_1,...,r_N], Y=[y_1,...,y_N] (y_0 is fixed initial condition)
    # Row k of J_full corresponds to residual r_{k+1}
    # Column j of J_full corresponds to variable y_{j+1}

    print("\nDetailed block-by-block comparison (checking ALL blocks):")
    print(f"Full Jacobian shape: {J_full_direct.shape} (should be {N*n_total} x {N*n_total})")
    print(f"Block size: {n_total} x {n_total}")
    print(f"Checking {N}x{N} = {N*N} blocks total")

    all_match = True
    block_tol = 1e-12

    for row in range(N):
        print(f"\n--- Row block {row} (residual r_{row+1}) ---")

        for col in range(N):
            # Extract block from full Jacobian
            block_extracted = J_full_direct[row*n_total:(row+1)*n_total, col*n_total:(col+1)*n_total]
            max_val = np.max(np.abs(block_extracted))

            # Determine what this block should be
            if col == row:
                # Diagonal block: should be J_curr[row]
                expected = J_curr_list[row]
                diff = np.max(np.abs(block_extracted - expected))
                block_name = f"J_curr[{row}]"
                print(f"  Col {col} (diagonal): dr_{row+1}/dy_{row+1}")
                print(f"    Should be: {block_name}")
                print(f"    Max diff: {diff:.6e}")
                if diff > 1e-10:
                    all_match = False
                    print(f"    ✗ MISMATCH!")

            elif col == row - 1 and row > 0:
                # Sub-diagonal block: should be J_prev[row]
                expected = J_prev_list[row]
                diff = np.max(np.abs(block_extracted - expected))
                block_name = f"J_prev[{row}]"
                print(f"  Col {col} (sub-diagonal): dr_{row+1}/dy_{row+1-1}")
                print(f"    Should be: {block_name}")
                print(f"    Max diff: {diff:.6e}")
                if diff > 1e-10:
                    all_match = False
                    print(f"    ✗ MISMATCH!")

            else:
                # Should be zero block
                print(f"  Col {col}: should be ZERO")
                print(f"    Max |value|: {max_val:.6e}")
                if max_val > block_tol:
                    all_match = False
                    print(f"    ✗ NON-ZERO block found!")

    if all_match:
        print("\n  ✓ PASSED: All blocks match")
    else:
        print("\n  ✗ FAILED: Some blocks do not match")

    # Save blocks to files for detailed inspection
    print("\n  Saving blocks to files for inspection...")
    output_dir = "jacobian_blocks_debug"
    import os
    os.makedirs(output_dir, exist_ok=True)

    for row in range(N):
        for col in range(N):
            # Extract block from full Jacobian
            block_extracted = J_full_direct[row*n_total:(row+1)*n_total, col*n_total:(col+1)*n_total]

            filename = f"{output_dir}/block_row{row}_col{col}.txt"
            with open(filename, 'w') as f:
                f.write(f"Block at row {row}, col {col} (residual r_{row+1}, variable y_{col+1})\n")
                f.write("="*80 + "\n\n")

                if col == row:
                    # Diagonal block
                    expected = J_curr_list[row]
                    f.write(f"This should be: J_curr[{row}] = dr_{row+1}/dy_{row+1}\n\n")

                    f.write("Block from FULL JACOBIAN:\n")
                    f.write("-"*80 + "\n")
                    for i in range(n_total):
                        f.write("  ".join(f"{block_extracted[i,j]:12.6e}" for j in range(n_total)) + "\n")

                    f.write("\n\nBlock from J_curr LIST:\n")
                    f.write("-"*80 + "\n")
                    for i in range(n_total):
                        f.write("  ".join(f"{expected[i,j]:12.6e}" for j in range(n_total)) + "\n")

                    diff = block_extracted - expected
                    f.write("\n\nDIFFERENCE (Full - List):\n")
                    f.write("-"*80 + "\n")
                    for i in range(n_total):
                        f.write("  ".join(f"{diff[i,j]:12.6e}" for j in range(n_total)) + "\n")
                    f.write(f"\n\nMax absolute difference: {np.max(np.abs(diff)):.6e}\n")

                elif col == row - 1 and row > 0:
                    # Sub-diagonal block
                    expected = J_prev_list[row]
                    f.write(f"This should be: J_prev[{row}] = dr_{row+1}/dy_{row}\n\n")

                    f.write("Block from FULL JACOBIAN:\n")
                    f.write("-"*80 + "\n")
                    for i in range(n_total):
                        f.write("  ".join(f"{block_extracted[i,j]:12.6e}" for j in range(n_total)) + "\n")

                    f.write("\n\nBlock from J_prev LIST:\n")
                    f.write("-"*80 + "\n")
                    for i in range(n_total):
                        f.write("  ".join(f"{expected[i,j]:12.6e}" for j in range(n_total)) + "\n")

                    diff = block_extracted - expected
                    f.write("\n\nDIFFERENCE (Full - List):\n")
                    f.write("-"*80 + "\n")
                    for i in range(n_total):
                        f.write("  ".join(f"{diff[i,j]:12.6e}" for j in range(n_total)) + "\n")
                    f.write(f"\n\nMax absolute difference: {np.max(np.abs(diff)):.6e}\n")

                else:
                    # Should be zero block
                    f.write("This should be: ZERO BLOCK\n\n")
                    f.write("Block from FULL JACOBIAN:\n")
                    f.write("-"*80 + "\n")
                    for i in range(n_total):
                        f.write("  ".join(f"{block_extracted[i,j]:12.6e}" for j in range(n_total)) + "\n")
                    f.write(f"\n\nMax absolute value: {np.max(np.abs(block_extracted)):.6e}\n")

    print(f"  Saved {N*N} block files to directory: {output_dir}/")
    print(f"  Each file shows: block from full Jacobian, expected block from list, and difference")

    # Test 4: Check sparsity structure
    print("\n" + "-"*80)
    print("Test 4: Checking sparsity structure")
    print("-"*80)

    # Count non-zero blocks
    block_tol = 1e-12
    nnz_blocks = 0
    for i in range(N):
        for j in range(N):
            block = J_full_assembled[i*n_total:(i+1)*n_total, j*n_total:(j+1)*n_total]
            if np.max(np.abs(block)) > block_tol:
                nnz_blocks += 1

    expected_nnz = N + (N-1)  # N diagonal blocks + (N-1) sub-diagonal blocks
    print(f"Non-zero blocks: {nnz_blocks} (expected {expected_nnz})")
    print(f"Sparsity: {(1 - nnz_blocks/(N*N))*100:.1f}%")

    if nnz_blocks == expected_nnz:
        print("  ✓ PASSED: Correct block-bidiagonal structure")
    else:
        print("  ✗ FAILED: Incorrect sparsity structure")

    # Test 5: Parameter Jacobian
    print("\n" + "-"*80)
    print("Test 5: Computing Parameter Jacobian")
    print("-"*80)

    print("\nComputing parameter Jacobian using vmap+jit...")
    start = time.time()
    J_param_full = jac_solver.assemble_full_parameter_jacobian(t_array, y_array)
    elapsed_param = time.time() - start
    print(f"Computed parameter Jacobian in {elapsed_param:.4f} seconds")
    print(f"Parameter Jacobian shape: {J_param_full.shape}")
    print(f"  Expected shape: ({N*n_total}, {len(jac_solver.p)})")

    print("\nComputing parameter Jacobian directly (for verification)...")
    start = time.time()
    J_param_direct = compute_parameter_jacobian_direct(jac_solver, t_array, y_array)
    elapsed_param_direct = time.time() - start
    print(f"Direct computation took {elapsed_param_direct:.4f} seconds")
    print(f"Direct parameter Jacobian shape: {J_param_direct.shape}")

    # Compare
    max_diff_param = np.max(np.abs(J_param_full - J_param_direct))
    rel_error_param = max_diff_param / (np.max(np.abs(J_param_direct)) + 1e-10)

    print(f"\nComparison:")
    print(f"  Max absolute difference: {max_diff_param:.6e}")
    print(f"  Relative error: {rel_error_param:.6e}")
    print(f"  Speedup: {elapsed_param_direct/elapsed_param:.2f}x")

    if rel_error_param < 1e-10:
        print("  ✓ PASSED: Parameter Jacobian matches direct computation")
    else:
        print("  ✗ FAILED: Parameter Jacobian does not match")

    # Test multiple calls to show JIT warmup effect
    print("\n  Testing JIT warmup effect (10 repeated calls):")
    times_param = []
    for i in range(10):
        start = time.time()
        _ = jac_solver.assemble_full_parameter_jacobian(t_array, y_array)
        times_param.append(time.time() - start)

    print(f"    First call:  {times_param[0]:.6f}s")
    print(f"    Second call: {times_param[1]:.6f}s")
    print(f"    Average (calls 2-10): {np.mean(times_param[1:]):.6f}s")
    print(f"    Min time: {min(times_param[1:]):.6f}s")
    print(f"    Speedup after warmup: {elapsed_param_direct/np.mean(times_param[1:]):.2f}x vs direct")

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"✓ Block Jacobians computed successfully")
    print(f"✓ Regular vmap time: {elapsed:.4f}s")
    print(f"✓ JIT+vmap time (after warmup): {np.mean(times_jit[1:]):.4f}s")
    print(f"✓ JIT speedup: {elapsed/np.mean(times_jit[1:]):.2f}x")
    print(f"✓ Speedup vs direct full Jacobian: {elapsed_direct/np.mean(times_jit[1:]):.2f}x (using JIT)")
    print(f"✓ Verification: max relative error = {rel_error:.6e}")
    print(f"\n✓ Parameter Jacobian computed successfully")
    print(f"✓ Parameter Jacobian time (after warmup): {np.mean(times_param[1:]):.6f}s")
    print(f"✓ Speedup vs direct: {elapsed_param_direct/np.mean(times_param[1:]):.2f}x")
    print(f"✓ Verification: max relative error = {rel_error_param:.6e}")

    return J_prev_list, J_curr_list, J_full_direct, J_full_assembled, J_param_full


if __name__ == "__main__":
    J_prev, J_curr, J_direct, J_assembled, J_param = test_jacobian_blocks()
