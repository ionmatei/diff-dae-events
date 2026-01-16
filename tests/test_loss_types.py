"""
Test script to compare 'sum' and 'mean' loss types.
"""

import numpy as np
import json
from src.dae_solver import DAESolver
from src.dae_jacobian import DAEOptimizer


def test_loss_types():
    """Test both sum and mean loss types."""

    # Load DAE specification
    json_path = "dae_examples/dae_specification_smooth.json"

    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    print("=" * 80)
    print("Testing Loss Types: 'sum' vs 'mean'")
    print("=" * 80)

    # Generate reference trajectory
    p_true = np.array([p['value'] for p in dae_data['parameters']])
    param_names = [p['name'] for p in dae_data['parameters']]

    solver_true = DAESolver(dae_data)
    t_span = (0.0, 5.0)
    ncp = 50

    result_true = solver_true.solve(t_span=t_span, ncp=ncp, rtol=1e-4, atol=1e-4)
    t_ref = result_true['t']
    y_ref = result_true['y']

    print(f"\nReference trajectory: {y_ref.shape}")

    # Perturb parameters
    capacitor_params = ['C1_C', 'C2_C']
    optimize_indices = [param_names.index(name) for name in capacitor_params]

    p_init = p_true.copy()
    np.random.seed(42)
    perturbation = 0.3
    for idx in optimize_indices:
        p_init[idx] = p_true[idx] * (1 + perturbation * (2 * np.random.rand() - 1))

    # Create modified DAE data
    dae_data_init = dae_data.copy()
    for i, p_dict in enumerate(dae_data_init['parameters']):
        p_dict['value'] = float(p_init[i])

    p_init_opt = np.array([p_init[param_names.index(name)] for name in capacitor_params])

    # Test with 'sum' loss
    print("\n" + "=" * 80)
    print("Test 1: Loss type = 'sum'")
    print("=" * 80)

    optimizer_sum = DAEOptimizer(dae_data_init, optimize_params=capacitor_params, loss_type='sum')

    result_sum = optimizer_sum.optimize(
        t_array=t_ref,
        y_target=y_ref.T,
        p_init=p_init_opt.copy(),
        n_iterations=10,
        step_size=0.001,
        tol=1e-6,
        verbose=True
    )

    # Test with 'mean' loss
    print("\n" + "=" * 80)
    print("Test 2: Loss type = 'mean'")
    print("=" * 80)

    optimizer_mean = DAEOptimizer(dae_data_init, optimize_params=capacitor_params, loss_type='mean')

    result_mean = optimizer_mean.optimize(
        t_array=t_ref,
        y_target=y_ref.T,
        p_init=p_init_opt.copy(),
        n_iterations=10,
        step_size=0.001,
        tol=1e-6,
        verbose=True
    )

    # Compare results
    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)

    print(f"\nLoss type 'sum':")
    print(f"  Initial loss: {result_sum['history']['loss'][0]:.6e}")
    print(f"  Final loss:   {result_sum['history']['loss'][-1]:.6e}")
    print(f"  Reduction:    {result_sum['history']['loss'][0] / result_sum['history']['loss'][-1]:.2f}x")

    print(f"\nLoss type 'mean':")
    print(f"  Initial loss: {result_mean['history']['loss'][0]:.6e}")
    print(f"  Final loss:   {result_mean['history']['loss'][-1]:.6e}")
    print(f"  Reduction:    {result_mean['history']['loss'][0] / result_mean['history']['loss'][-1]:.2f}x")

    print(f"\nNote: Mean loss = Sum loss / N, where N = {y_ref.shape[0] * y_ref.shape[1]}")
    print(f"  Ratio: {result_sum['history']['loss'][0] / result_mean['history']['loss'][0]:.1f}")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_loss_types()
