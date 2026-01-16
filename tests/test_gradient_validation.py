"""
Test script to validate adjoint-based gradient computation.

This script compares the gradient computed using the adjoint method
(at line 1307 in src/dae_jacobian.py) with a numerical finite-difference
approximation.
"""

import numpy as np
import json
from src.dae_solver import DAESolver
from src.dae_jacobian import DAEOptimizer


def compute_numerical_gradient(optimizer, t_array, y_target, p_opt, epsilon=1e-5):
    """
    Compute numerical gradient using central finite differences.

    For each parameter p_i, compute:
        dL/dp_i ≈ (L(p + ε*e_i) - L(p - ε*e_i)) / (2ε)

    Parameters
    ----------
    optimizer : DAEOptimizer
        The optimizer instance
    t_array : np.ndarray
        Time points for trajectory
    y_target : np.ndarray
        Target output trajectory (n_time, n_outputs)
    p_opt : np.ndarray
        Current values of optimized parameters
    epsilon : float
        Finite difference step size

    Returns
    -------
    grad_numerical : np.ndarray
        Numerical gradient (n_params_opt,)
    """
    n_params_opt = len(p_opt)
    grad_numerical = np.zeros(n_params_opt)

    print(f"\nComputing numerical gradient with ε = {epsilon:.2e}...")

    for i in range(n_params_opt):
        # Create perturbed parameter vectors
        p_plus = np.array(p_opt)
        p_minus = np.array(p_opt)

        p_plus[i] += epsilon
        p_minus[i] -= epsilon

        # Update full parameter vectors (convert JAX array to NumPy first)
        p_all_plus = np.array(optimizer.p_all)
        p_all_minus = np.array(optimizer.p_all)

        for j, idx in enumerate(optimizer.optimize_indices):
            p_all_plus[idx] = p_plus[j]
            p_all_minus[idx] = p_minus[j]

        # Compute loss with p_plus
        # Create new DAE data with perturbed parameters
        dae_data_plus = optimizer.dae_data.copy()
        for j, p_dict in enumerate(dae_data_plus['parameters']):
            p_dict['value'] = float(p_all_plus[j])

        # Create new solver with perturbed parameters
        from src.dae_solver import DAESolver
        solver_plus = DAESolver(dae_data_plus)
        result_plus = solver_plus.solve(t_span=(t_array[0], t_array[-1]), ncp=len(t_array), rtol=1e-4, atol=1e-4)
        y_pred_plus = result_plus['y']  # shape: (n_outputs, n_time)
        loss_plus = optimizer.compute_loss(y_pred_plus, y_target.T)

        # Compute loss with p_minus
        # Create new DAE data with perturbed parameters
        dae_data_minus = optimizer.dae_data.copy()
        for j, p_dict in enumerate(dae_data_minus['parameters']):
            p_dict['value'] = float(p_all_minus[j])

        # Create new solver with perturbed parameters
        solver_minus = DAESolver(dae_data_minus)
        result_minus = solver_minus.solve(t_span=(t_array[0], t_array[-1]), ncp=len(t_array), rtol=1e-4, atol=1e-4)
        y_pred_minus = result_minus['y']  # shape: (n_outputs, n_time)
        loss_minus = optimizer.compute_loss(y_pred_minus, y_target.T)

        # Central difference
        grad_numerical[i] = (loss_plus - loss_minus) / (2 * epsilon)

        print(f"  Parameter {i+1}/{n_params_opt}: "
              f"L(+ε) = {loss_plus:.6e}, L(-ε) = {loss_minus:.6e}, "
              f"dL/dp = {grad_numerical[i]:.6e}")

    return grad_numerical


def test_gradient_validation():
    """Test adjoint gradient against numerical gradient."""

    print("=" * 80)
    print("Gradient Validation Test")
    print("=" * 80)

    # Load DAE specification
    json_path = "dae_examples/dae_specification_smooth.json"

    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    print(f"\nLoaded DAE from: {json_path}")

    # Generate reference trajectory with true parameters
    p_true = np.array([p['value'] for p in dae_data['parameters']])
    param_names = [p['name'] for p in dae_data['parameters']]

    solver_true = DAESolver(dae_data)
    t_span = (0.0, 30)
    ncp = 300  # Use fewer points for faster numerical gradient computation

    print(f"\nGenerating reference trajectory...")
    print(f"  Time span: {t_span}")
    print(f"  Output points: {ncp}")

    result_true = solver_true.solve(t_span=t_span, ncp=ncp, rtol=1e-4, atol=1e-4)
    t_ref = result_true['t']
    y_ref = result_true['y']

    print(f"  Reference trajectory shape: {y_ref.shape}")

    # Select parameters to optimize
    capacitor_params = ['C1_C', 'C2_C']
    optimize_indices = [param_names.index(name) for name in capacitor_params]

    print(f"\nParameters to optimize: {capacitor_params}")

    # Create initial parameter guess (perturbed)
    p_init = p_true.copy()
    np.random.seed(42)
    perturbation = 0.5
    for idx in optimize_indices:
        p_init[idx] = p_true[idx] * (1 + perturbation * (2 * np.random.rand() - 1))

    print(f"\nInitial parameter values:")
    for i, (name, val_true, val_init) in enumerate(zip(param_names, p_true, p_init)):
        error = abs(val_init - val_true) / abs(val_true) * 100
        status = 'Optimized' if i in optimize_indices else 'Fixed'
        print(f"  {name:20s} = {val_init:.6f}  (true: {val_true:.6f}, error: {error:>6.1f}%, {status})")

    # Create optimizer
    dae_data_init = dae_data.copy()
    for i, p_dict in enumerate(dae_data_init['parameters']):
        p_dict['value'] = float(p_init[i])

    optimizer = DAEOptimizer(dae_data_init, optimize_params=capacitor_params, loss_type='sum')

    p_init_opt = np.array([p_init[param_names.index(name)] for name in capacitor_params])

    # Compute adjoint-based gradient
    print("\n" + "=" * 80)
    print("Computing Adjoint-Based Gradient")
    print("=" * 80)

    p_new, loss_current, grad_adjoint  = optimizer.optimization_step(t_ref, y_ref.T, p_init_opt)

    print(f"\nAdjoint gradient computed:")
    print(f"  Loss value: {loss_current:.6e}")
    print(f"  Gradient: {grad_adjoint}")
    print(f"  Gradient norm: {np.linalg.norm(grad_adjoint):.6e}")

    # Compute numerical gradient
    print("\n" + "=" * 80)
    print("Computing Numerical Gradient (Finite Differences)")
    print("=" * 80)

    grad_numerical = compute_numerical_gradient(
        optimizer,
        t_ref,
        y_ref.T,
        p_init_opt,
        epsilon=1e-6
    )

    print(f"\nNumerical gradient computed:")
    print(f"  Gradient: {grad_numerical}")
    print(f"  Gradient norm: {np.linalg.norm(grad_numerical):.6e}")

    # Compare gradients
    print("\n" + "=" * 80)
    print("Gradient Comparison")
    print("=" * 80)

    print(f"\n{'Parameter':<20} {'Adjoint':>15} {'Numerical':>15} {'Abs Error':>15} {'Rel Error':>15}")
    print("-" * 85)

    for i, name in enumerate(capacitor_params):
        abs_error = abs(grad_adjoint[i] - grad_numerical[i])
        rel_error = abs_error / (abs(grad_numerical[i]) + 1e-16) * 100
        print(f"{name:<20} {grad_adjoint[i]:>15.6e} {grad_numerical[i]:>15.6e} "
              f"{abs_error:>15.6e} {rel_error:>14.2f}%")

    # Overall error metrics
    abs_error_norm = np.linalg.norm(grad_adjoint - grad_numerical)
    rel_error_norm = abs_error_norm / (np.linalg.norm(grad_numerical) + 1e-16) * 100

    print(f"\n{'Overall':20} {'':>15} {'':>15} "
          f"{abs_error_norm:>15.6e} {rel_error_norm:>14.2f}%")

    # Validation status
    print("\n" + "=" * 80)
    print("Validation Result")
    print("=" * 80)

    tolerance = 1.0  # 1% relative error tolerance

    if rel_error_norm < tolerance:
        print(f"\n✓ PASSED: Relative error {rel_error_norm:.4f}% < {tolerance}%")
        print("  Adjoint gradient matches numerical gradient!")
    else:
        print(f"\n✗ FAILED: Relative error {rel_error_norm:.4f}% >= {tolerance}%")
        print("  Adjoint gradient does not match numerical gradient.")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

    return {
        'grad_adjoint': grad_adjoint,
        'grad_numerical': grad_numerical,
        'abs_error_norm': abs_error_norm,
        'rel_error_norm': rel_error_norm,
        'passed': rel_error_norm < tolerance
    }


if __name__ == "__main__":
    result = test_gradient_validation()
