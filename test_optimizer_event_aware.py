"""
Test script for event-aware DAE optimizer.

Tests the discrete adjoint optimization on the bouncing ball problem.
"""

import os
# Set CPU device BEFORE any JAX imports for faster testing
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time

from src.discrete_adjoint.dae_solver import DAESolver, AugmentedSolution
from src.discrete_adjoint.dae_optimizer_event_aware import DAEOptimizerEventAware


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_target_trajectory(solver, t_span, ncp, true_params):
    """Generate target trajectory with known parameters."""
    # Save current params
    original_params = solver.p.copy()
    
    # Set true parameters
    for name, value in true_params.items():
        idx = solver.param_names.index(name)
        solver.p[idx] = value
    
    # Solve
    aug_sol = solver.solve_augmented(t_span=t_span, ncp=ncp)
    
    # Extract target times and outputs  
    # Stitch segments together
    t_all = []
    x_all = []
    for seg in aug_sol.segments:
        t_all.append(seg.t)
        x_all.append(seg.x)
    
    t_target = np.concatenate(t_all)
    x_target = np.concatenate(x_all)
    
    # Restore parameters
    solver.p[:] = original_params
    
    return t_target, x_target, aug_sol


def test_basic_forward_backward():
    """Test basic forward solve and backward pass."""
    print("=" * 60)
    print("TEST: Basic Forward/Backward Pass")
    print("=" * 60)
    
    # Load DAE
    with open("dae_examples/dae_specification_bouncing_ball.json", 'r') as f:
        dae_data = json.load(f)
    
    # Create solver
    solver = DAESolver(dae_data, verbose=True)
    
    # Create optimizer
    optimizer = DAEOptimizerEventAware(
        dae_data=dae_data,
        dae_solver=solver,
        optimize_params=['e', 'g'],
        method='trapezoidal',
        verbose=True
    )
    
    # True parameters (what we're trying to recover)
    true_params = {'g': 9.81, 'e': 0.8}
    
    # Generate target trajectory
    t_span = (0.0, 2.0)
    ncp = 100
    
    print("\nGenerating target trajectory with true params...")
    t_target, x_target, aug_sol_true = generate_target_trajectory(
        solver, t_span, ncp, true_params
    )
    
    print(f"  Target points: {len(t_target)}")
    print(f"  Events: {len(aug_sol_true.events)}")
    
    # Test with perturbed parameters
    perturbed_params = {'g': 8.0, 'e': 0.6}
    
    print(f"\nTesting with perturbed params: g={perturbed_params['g']}, e={perturbed_params['e']}")
    
    # Set perturbed parameters
    for name, value in perturbed_params.items():
        idx = solver.param_names.index(name)
        solver.p[idx] = value
    
    # Optimization step
    p_opt = np.array([perturbed_params['e'], perturbed_params['g']])
    
    try:
        p_new, loss, grad = optimizer.optimization_step_events(
            t_span=t_span,
            target_times=t_target,
            target_outputs=x_target,
            p_opt=p_opt,
            step_size=0.01,
            ncp=ncp
        )
        
        print(f"\n  Initial loss: {loss:.6f}")
        print(f"  Gradient: {grad}")
        print(f"  New params: {p_new}")
        print("\n  PASSED: Forward/backward pass completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_finite_difference_gradient():
    """Verify gradients against finite differences."""
    print("\n" + "=" * 60)
    print("TEST: Finite Difference Gradient Verification")
    print("=" * 60)
    
    # Load DAE
    with open("dae_examples/dae_specification_bouncing_ball.json", 'r') as f:
        dae_data = json.load(f)
    
    solver = DAESolver(dae_data, verbose=False)
    
    optimizer = DAEOptimizerEventAware(
        dae_data=dae_data,
        dae_solver=solver,
        optimize_params=['e'],  # Single param for simpler test
        method='trapezoidal',
        verbose=False
    )
    
    # Simple target
    t_span = (0.0, 1.0)
    ncp = 50
    true_params = {'g': 9.81, 'e': 0.7}
    
    t_target, x_target, _ = generate_target_trajectory(
        solver, t_span, ncp, true_params
    )
    
    # Test point
    p_opt = np.array([0.6])  # e value
    
    # Compute adjoint gradient
    _, loss_center, grad_adjoint = optimizer.optimization_step_events(
        t_span, t_target, x_target, p_opt, step_size=0.0, ncp=ncp
    )
    
    # Finite difference gradient
    eps = 1e-5
    grad_fd = np.zeros_like(p_opt)
    
    for i in range(len(p_opt)):
        p_plus = p_opt.copy()
        p_plus[i] += eps
        
        p_minus = p_opt.copy()
        p_minus[i] -= eps
        
        _, loss_plus, _ = optimizer.optimization_step_events(
            t_span, t_target, x_target, p_plus, step_size=0.0, ncp=ncp
        )
        _, loss_minus, _ = optimizer.optimization_step_events(
            t_span, t_target, x_target, p_minus, step_size=0.0, ncp=ncp
        )
        
        grad_fd[i] = (loss_plus - loss_minus) / (2 * eps)
    
    print(f"\n  Adjoint gradient: {grad_adjoint}")
    print(f"  FD gradient:      {grad_fd}")
    print(f"  Relative error:   {np.abs(grad_adjoint - grad_fd) / (np.abs(grad_fd) + 1e-10)}")
    
    rel_error = np.max(np.abs(grad_adjoint - grad_fd) / (np.abs(grad_fd) + 1e-10))
    
    if rel_error < 0.1:  # 10% tolerance for now
        print("\n  PASSED: Gradients match within tolerance")
        return True
    else:
        print("\n  WARNING: Gradient mismatch (may be expected due to discrete events)")
        return False


def test_optimization_convergence():
    """Test that optimization actually converges."""
    print("\n" + "=" * 60)
    print("TEST: Optimization Convergence")
    print("=" * 60)
    
    # Load DAE
    with open("dae_examples/dae_specification_bouncing_ball.json", 'r') as f:
        dae_data = json.load(f)
    
    solver = DAESolver(dae_data, verbose=False)
    
    optimizer = DAEOptimizerEventAware(
        dae_data=dae_data,
        dae_solver=solver,
        optimize_params=['e'],  # Just optimize restitution
        method='trapezoidal',
        verbose=False
    )
    
    # True parameters
    true_params = {'g': 9.81, 'e': 0.75}
    t_span = (0.0, 1.5)
    ncp = 60
    
    # Generate target
    t_target, x_target, _ = generate_target_trajectory(
        solver, t_span, ncp, true_params
    )
    
    # Initial guess (wrong)
    optimizer.p_current = np.array([0.5])  # e = 0.5 (true = 0.75)
    
    print(f"\n  True e = {true_params['e']}")
    print(f"  Initial guess e = {optimizer.p_current[0]}")
    
    # Run optimization
    result = optimizer.optimize_events(
        t_span=t_span,
        target_times=t_target,
        target_outputs=x_target,
        max_iterations=50,
        step_size=0.1,
        tol=1e-4,
        ncp=ncp,
        print_every=10
    )
    
    final_e = result['params'][0]
    error = abs(final_e - true_params['e'])
    
    print(f"\n  Final e = {final_e:.4f}")
    print(f"  Error = {error:.4f}")
    
    if error < 0.1:
        print("\n  PASSED: Optimization converged to near-true value")
        return True
    else:
        print("\n  PARTIAL: Optimization made progress but didn't fully converge")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EVENT-AWARE DAE OPTIMIZER TESTS")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Basic forward/backward
    results['forward_backward'] = test_basic_forward_backward()
    
    # Test 2: Finite difference verification
    results['finite_diff'] = test_finite_difference_gradient()
    
    # Test 3: Optimization convergence
    results['convergence'] = test_optimization_convergence()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
