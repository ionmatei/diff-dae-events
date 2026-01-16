"""
Test script to compare performance between separate JIT functions vs combined JIT function.

This script demonstrates:
1. Using the original optimization_step (separate JIT functions for each step)
2. Using the new optimization_step_combined (single JIT function for steps 2-7)
3. Comparing timing results between the two approaches
"""

# Configure JAX to use CPU (must be done BEFORE importing modules that use JAX)
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import numpy as np
import json
from src.dae_solver import DAESolver
from src.dae_jacobian import DAEOptimizer


def test_combined_jit():
    """Test and compare the combined JIT vs separate JIT implementations."""

    print("=" * 80)
    print("Testing Combined JIT vs Separate JIT Performance")
    print("=" * 80)

    # Load DAE specification
    json_path = "dae_examples/dae_specification_smooth.json"
    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    print(f"\nLoaded DAE from: {json_path}")

    # Generate reference trajectory
    solver_true = DAESolver(dae_data)
    t_span = (0.0, 10.0)
    ncp = 50  # Smaller for faster testing
    result_true = solver_true.solve(t_span=t_span, ncp=ncp, rtol=1e-4, atol=1e-4)

    t_ref = result_true['t']
    y_ref = result_true['y']

    print(f"Reference trajectory: {len(t_ref)} time points")

    # Create optimizer
    capacitor_params = ['C1_C', 'C2_C']
    optimizer = DAEOptimizer(dae_data, optimize_params=capacitor_params)

    # Initial parameter guess (perturbed)
    param_names = [p['name'] for p in dae_data['parameters']]
    p_true = np.array([p['value'] for p in dae_data['parameters']])
    p_init = p_true.copy()

    np.random.seed(42)
    for name in capacitor_params:
        idx = param_names.index(name)
        p_init[idx] = p_true[idx] * 1.3  # 30% error

    p_init_opt = np.array([p_init[param_names.index(name)] for name in capacitor_params])

    # Test 1: Original optimization_step (separate JIT functions)
    print("\n" + "=" * 80)
    print("Test 1: Original optimization_step (Separate JIT functions)")
    print("=" * 80)
    print("\nRunning 3 iterations to measure timing...")

    p_current = p_init_opt.copy()
    for i in range(3):
        print(f"\n--- Iteration {i+1} ---")
        p_new, loss, grad = optimizer.optimization_step(
            t_array=t_ref,
            y_target=y_ref.T,
            p_opt=p_current,
            step_size=0.001
        )
        p_current = p_new
        print(f"Loss: {loss:.6e}, Gradient norm: {np.linalg.norm(grad):.6e}")

    # Test 2: New optimization_step_combined (single combined JIT function)
    print("\n" + "=" * 80)
    print("Test 2: New optimization_step_combined (Combined JIT function)")
    print("=" * 80)
    print("\nRunning 3 iterations to measure timing...")

    p_current = p_init_opt.copy()
    for i in range(3):
        print(f"\n--- Iteration {i+1} ---")
        p_new, loss, grad = optimizer.optimization_step_combined(
            t_array=t_ref,
            y_target=y_ref.T,
            p_opt=p_current,
            step_size=0.001
        )
        p_current = p_new
        print(f"Loss: {loss:.6e}, Gradient norm: {np.linalg.norm(grad):.6e}")

    print("\n" + "=" * 80)
    print("Performance Comparison Summary")
    print("=" * 80)
    print("\nKey observations:")
    print("1. First iteration may be slower due to JIT compilation")
    print("2. Subsequent iterations show the true performance difference")
    print("3. Combined JIT reduces Python overhead between steps")
    print("4. On CPU, the difference may be small (5-15%)")
    print("5. On GPU, combined JIT would show larger improvements")
    print("\nNote: Check the 'steps_2_7_combined_jit' time vs sum of step_2 through step_7")
    print("=" * 80)


if __name__ == "__main__":
    test_combined_jit()
