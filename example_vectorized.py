#!/usr/bin/env python3
"""
Example demonstrating vectorized function evaluation using JAX vmap.

This script shows how to use the eval_f_vectorized, eval_g_vectorized,
and eval_h_vectorized methods to efficiently compute functions over
multiple time points in parallel.

The DAESolver now compiles vmapped functions ONCE during initialization,
so repeated calls to the vectorized methods have minimal overhead.
"""

import json
import numpy as np
import sys
sys.path.insert(0, 'src')

from dae_solver import DAESolver

def main():
    # Load DAE specification
    json_path = "dae_examples/dae_specification_smooth.json"

    print("Loading DAE from:", json_path)
    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    # Create solver - this will compile vmap functions once during initialization
    print("\nInitializing solver...")
    solver = DAESolver(dae_data)
    print("Note: vmap functions are compiled once during initialization for efficiency")

    # Solve the DAE
    print("\nSolving DAE...")
    result = solver.solve(
        t_span=(0.0, 60.0),
        ncp=100,  # Use fewer points for faster demo
        rtol=1e-5,
        atol=1e-5,
    )

    print(f"Solution computed at {len(result['t'])} time points")

    # Extract solution data
    t_array = result['t']
    x_array = result['x']  # shape: (n_states, n_times)
    z_array = result['z']  # shape: (n_alg, n_times)

    # Combine into y = [x, z]
    y_array = np.vstack([x_array, z_array])  # shape: (n_states + n_alg, n_times)

    print(f"\nData shapes:")
    print(f"  t_array: {t_array.shape}")
    print(f"  x_array: {x_array.shape}")
    print(f"  z_array: {z_array.shape}")
    print(f"  y_array: {y_array.shape}")

    # ========================================================================
    # Example 1: Vectorized evaluation of f (derivatives)
    # ========================================================================
    print("\n" + "="*70)
    print("Example 1: Vectorized evaluation of f(t, x, z)")
    print("="*70)

    f_vectorized = solver.eval_f_vectorized(t_array, y_array)
    print(f"Result shape: {f_vectorized.shape}")
    print(f"Expected: ({len(t_array)}, {len(solver.state_names)})")
    print(f"\nSample values at first 3 time points:")
    for i in range(min(3, len(t_array))):
        print(f"  t={t_array[i]:.2f}: f = {f_vectorized[i, :3]}")

    # ========================================================================
    # Example 2: Vectorized evaluation of g (algebraic constraints)
    # ========================================================================
    print("\n" + "="*70)
    print("Example 2: Vectorized evaluation of g(t, x, z)")
    print("="*70)

    g_vectorized = solver.eval_g_vectorized(t_array, y_array)
    print(f"Result shape: {g_vectorized.shape}")
    print(f"Expected: ({len(t_array)}, {len(solver.alg_names)})")
    print(f"\nConstraint satisfaction (should be near zero):")
    print(f"  Max |g|: {np.max(np.abs(g_vectorized)):.6e}")
    print(f"  Mean |g|: {np.mean(np.abs(g_vectorized)):.6e}")
    print(f"\nSample constraint values at first 3 time points:")
    for i in range(min(3, len(t_array))):
        print(f"  t={t_array[i]:.2f}: g = {g_vectorized[i, :3]}")

    # ========================================================================
    # Example 3: Vectorized evaluation of h (outputs)
    # ========================================================================
    print("\n" + "="*70)
    print("Example 3: Vectorized evaluation of h(t, x, z)")
    print("="*70)

    h_vectorized = solver.eval_h_vectorized(t_array, y_array)
    print(f"Result shape: {h_vectorized.shape}")

    if solver.h_funcs:
        print(f"Expected: ({len(t_array)}, {len(solver.output_names)})")
        print(f"Note: Using user-defined output equations")
    else:
        print(f"Expected: ({len(t_array)}, {len(solver.state_names)})")
        print(f"Note: No h defined, returning state vector x (identity mapping)")

    print(f"\nSample output values at first 3 time points:")
    for i in range(min(3, len(t_array))):
        print(f"  t={t_array[i]:.2f}: h = {h_vectorized[i, :min(3, h_vectorized.shape[1])]}")

    # ========================================================================
    # Example 4: Compare with loop-based evaluation
    # ========================================================================
    print("\n" + "="*70)
    print("Example 4: Performance comparison (vectorized vs loops)")
    print("="*70)

    import time

    # Vectorized evaluation (using pre-compiled vmap)
    start = time.time()
    f_vec = solver.eval_f_vectorized(t_array, y_array)
    time_vec = time.time() - start

    # Loop-based evaluation
    start = time.time()
    f_loop = np.zeros((len(t_array), len(solver.state_names)))
    for i in range(len(t_array)):
        x_i = x_array[:, i]
        z_i = z_array[:, i]
        f_loop[i] = solver.eval_f(t_array[i], x_i, z_i)
    time_loop = time.time() - start

    print(f"Vectorized time: {time_vec:.6f} seconds (using pre-compiled vmap)")
    print(f"Loop time:       {time_loop:.6f} seconds")
    print(f"Speedup:         {time_loop/time_vec:.2f}x")
    print(f"Max difference:  {np.max(np.abs(f_vec - f_loop)):.6e}")

    # ========================================================================
    # Example 4b: Multiple calls show no vmap compilation overhead
    # ========================================================================
    print("\n" + "="*70)
    print("Example 4b: Multiple vectorized calls (no vmap overhead)")
    print("="*70)

    print("Calling eval_f_vectorized 10 times...")
    times = []
    for _ in range(10):
        start = time.time()
        _ = solver.eval_f_vectorized(t_array, y_array)
        times.append(time.time() - start)

    print(f"Call times: min={min(times):.6f}s, max={max(times):.6f}s, avg={np.mean(times):.6f}s")
    print(f"Consistency: {(max(times)-min(times))/np.mean(times)*100:.1f}% variation")
    print("Note: All calls use the same pre-compiled vmap function")

    # ========================================================================
    # Example 5: Evaluate on a subset of time points
    # ========================================================================
    print("\n" + "="*70)
    print("Example 5: Evaluate on a custom set of time points")
    print("="*70)

    # Create custom time points (not necessarily from solution)
    t_custom = np.linspace(0, 60, 20)

    # Interpolate solution to these points
    from scipy.interpolate import interp1d
    x_interp = interp1d(t_array, x_array, axis=1, kind='cubic')
    z_interp = interp1d(t_array, z_array, axis=1, kind='cubic')

    x_custom = x_interp(t_custom)
    z_custom = z_interp(t_custom)
    y_custom = np.vstack([x_custom, z_custom])

    print(f"Evaluating at {len(t_custom)} custom time points...")
    f_custom = solver.eval_f_vectorized(t_custom, y_custom)
    g_custom = solver.eval_g_vectorized(t_custom, y_custom)

    print(f"f result shape: {f_custom.shape}")
    print(f"g result shape: {g_custom.shape}")
    print(f"Sample f at t={t_custom[0]:.2f}: {f_custom[0, :3]}")
    print(f"Sample g at t={t_custom[0]:.2f}: {g_custom[0, :3]}")

    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)

    # ========================================================================
    # Summary
    # ========================================================================
    print("\nSUMMARY:")
    print("--------")
    print("The vectorized methods accept:")
    print("  - t_array: array of time points, shape (n_times,)")
    print("  - y_array: combined state [x, z], shape (n_times, n_total)")
    print("            or shape (n_total, n_times) - auto-detected")
    print("\nThe vectorized methods return:")
    print("  - eval_f_vectorized: shape (n_times, n_states)")
    print("  - eval_g_vectorized: shape (n_times, n_alg)")
    print("  - eval_h_vectorized: shape (n_times, n_outputs) if h defined")
    print("                       shape (n_times, n_states) if h not defined (identity)")
    print("\nKey features:")
    print("  1. JAX vmap functions are compiled ONCE during DAESolver initialization")
    print("  2. Subsequent calls have no vmap compilation overhead")
    print("  3. Automatic fallback to numpy loops if JAX is not available")
    print("  4. Flexible input shapes (auto-detects and transposes if needed)")
    print("  5. When h is not defined, eval_h returns state vector x (identity mapping)")


if __name__ == "__main__":
    main()
