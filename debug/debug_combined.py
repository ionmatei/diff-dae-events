"""
Debug script to compare combined vs non-combined optimization step by step.
"""

# Configure JAX to use CPU
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import numpy as np
import json
from src.dae_solver import DAESolver
from src.dae_jacobian import DAEOptimizer


def debug_optimization_step():
    """Compare single step of combined vs non-combined."""

    print("=" * 80)
    print("Debugging Combined vs Non-Combined Optimization")
    print("=" * 80)

    # Load DAE specification
    json_path = "dae_examples/dae_specification_smooth.json"
    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    print(f"\nLoaded DAE from: {json_path}")

    # Generate reference trajectory
    solver_true = DAESolver(dae_data)
    t_span = (0.0, 10.0)
    ncp = 50
    result_true = solver_true.solve(t_span=t_span, ncp=ncp, rtol=1e-4, atol=1e-4)

    t_ref = result_true['t']
    y_ref = result_true['y']

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
        p_init[idx] = p_true[idx] * 1.3

    p_init_opt = np.array([p_init[param_names.index(name)] for name in capacitor_params])

    print(f"\nInitial optimized parameters: {p_init_opt}")

    # Test both methods with same inputs
    print("\n" + "=" * 80)
    print("Test 1: Non-Combined (Original)")
    print("=" * 80)

    p_new_orig, loss_orig, grad_orig = optimizer.optimization_step(
        t_array=t_ref,
        y_target=y_ref.T,
        p_opt=p_init_opt,
        step_size=0.001
    )

    print(f"\nResults (Non-Combined):")
    print(f"  Loss: {loss_orig:.10e}")
    print(f"  Gradient: {grad_orig}")
    print(f"  Gradient norm: {np.linalg.norm(grad_orig):.10e}")
    print(f"  Updated params: {p_new_orig}")

    # Test combined
    print("\n" + "=" * 80)
    print("Test 2: Combined JIT")
    print("=" * 80)

    p_new_comb, loss_comb, grad_comb = optimizer.optimization_step_combined(
        t_array=t_ref,
        y_target=y_ref.T,
        p_opt=p_init_opt,
        step_size=0.001
    )

    print(f"\nResults (Combined):")
    print(f"  Loss: {loss_comb:.10e}")
    print(f"  Gradient: {grad_comb}")
    print(f"  Gradient norm: {np.linalg.norm(grad_comb):.10e}")
    print(f"  Updated params: {p_new_comb}")

    # Compare
    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)

    print(f"\nLoss difference: {abs(loss_orig - loss_comb):.10e}")
    print(f"  Relative: {abs(loss_orig - loss_comb) / abs(loss_orig):.10e}")

    grad_diff = np.linalg.norm(grad_orig - grad_comb)
    print(f"\nGradient difference (norm): {grad_diff:.10e}")
    print(f"  Relative: {grad_diff / np.linalg.norm(grad_orig):.10e}")

    param_diff = np.linalg.norm(p_new_orig - p_new_comb)
    print(f"\nParameter update difference (norm): {param_diff:.10e}")
    print(f"  Relative: {param_diff / np.linalg.norm(p_new_orig):.10e}")

    print("\nElement-wise gradient comparison:")
    for i in range(len(grad_orig)):
        diff = abs(grad_orig[i] - grad_comb[i])
        rel_diff = diff / abs(grad_orig[i]) if abs(grad_orig[i]) > 1e-15 else 0
        print(f"  grad[{i}]: orig={grad_orig[i]:.6e}, comb={grad_comb[i]:.6e}, diff={diff:.6e}, rel={rel_diff:.6e}")

    # Determine if results match
    print("\n" + "=" * 80)
    if grad_diff < 1e-6:
        print("✓ Results MATCH (difference < 1e-6)")
    elif grad_diff < 1e-3:
        print("⚠ Results are CLOSE but not identical (difference < 1e-3)")
    else:
        print("✗ Results DIFFER significantly!")
    print("=" * 80)


if __name__ == "__main__":
    debug_optimization_step()
