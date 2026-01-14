"""
Example: Using SDAG Optimizer for DAE Parameter Identification

This script demonstrates:
1. Loading a DAE from JSON specification
2. Generating target data with "true" parameters
3. Optimizing to recover parameters from observations
4. Visualizing convergence
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
import matplotlib.pyplot as plt
from dae_solver import DAESolver
from sdag_optimizer import SDAGOptimizer


def main():
    print("="*80)
    print("SDAG Optimizer Example: DAE Parameter Identification")
    print("="*80)

    # Load DAE specification
    # Get the repository root directory
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(repo_root, "dae_examples", "dae_specification_smooth.json")
    print(f"\nLoading DAE from: {json_path}")

    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    # Create solver
    solver = DAESolver(dae_data)

    # Set "true" parameters (nominal values from JSON)
    theta_true = solver.p.copy()
    print(f"\nTrue parameters (nominal): {theta_true}")

    # Solve with true parameters to get target trajectory
    print("\nGenerating target trajectory...")
    result_true = solver.solve(
        t_span=(0.0, 60.0),
        ncp=100,
        rtol=1e-4,
        atol=1e-4
    )

    target_trajectory = {
        't': result_true['t'],
        'x': result_true['x'],
        'z': result_true['z'],
    }

    print(f"Target trajectory generated:")
    print(f"  Time points: {len(target_trajectory['t'])}")
    print(f"  States: {target_trajectory['x'].shape}")
    print(f"  Algebraic: {target_trajectory['z'].shape}")

    # Create initial guess for parameters (perturbed from true values)
    theta_init = theta_true.copy()
    # Perturb specific parameters that affect dynamics
    # For this circuit example, perturb resistance and capacitance values
    param_indices_to_perturb = [22, 24, 26]  # c5, c3, c1 indices
    perturbation_factor = 1.2  # 20% error

    for idx in param_indices_to_perturb:
        if idx < len(theta_init):
            theta_init[idx] *= perturbation_factor

    print(f"\nInitial guess (perturbed): {theta_init}")
    print(f"Parameter error norm: {np.linalg.norm(theta_init - theta_true):.6e}")

    # Create SDAG optimizer
    print("\n" + "="*80)
    print("Creating SDAG Optimizer")
    print("="*80)

    optimizer = SDAGOptimizer(
        dae_solver=solver,
        target_trajectory=target_trajectory,
        discretization="trapezoidal",
        alpha=0.001,  # Step size
        gmres_tol=1e-6,
        gmres_maxiter=20  # Reduced for testing
    )

    # Run optimization
    n_iterations = 20
    history = optimizer.optimize(
        theta_init=theta_init,
        n_iterations=n_iterations,
        verbose=True
    )

    # Analyze results
    print("\n" + "="*80)
    print("Optimization Results")
    print("="*80)

    theta_final = history['theta_final']
    param_error = np.linalg.norm(theta_final - theta_true)

    print(f"\nTrue parameters:     {theta_true}")
    print(f"Initial guess:       {theta_init}")
    print(f"Optimized parameters: {theta_final}")
    print(f"\nFinal parameter error: {param_error:.6e}")
    print(f"Relative error: {param_error / np.linalg.norm(theta_true):.6e}")

    # Plot convergence
    plot_convergence(history, theta_true)

    # Compare trajectories
    print("\nComparing trajectories...")
    solver.p = theta_final
    result_final = solver.solve(
        t_span=(0.0, 60.0),
        ncp=100,
        rtol=1e-4,
        atol=1e-4
    )

    plot_trajectory_comparison(result_true, result_final)

    print("\nDone!")


def plot_convergence(history, theta_true):
    """Plot optimization convergence metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    iterations = np.arange(len(history['loss']))

    # Loss
    ax = axes[0, 0]
    ax.semilogy(iterations, history['loss'], 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Loss Function Convergence')
    ax.grid(True, alpha=0.3)

    # Gradient norm
    ax = axes[0, 1]
    ax.semilogy(iterations, history['gradient_norm'], 'r-o', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm Convergence')
    ax.grid(True, alpha=0.3)

    # Parameter error
    ax = axes[1, 0]
    param_errors = [np.linalg.norm(theta - theta_true) for theta in history['theta']]
    ax.semilogy(iterations, param_errors[1:], 'g-o', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Parameter Error ||θ - θ_true||')
    ax.set_title('Parameter Error Convergence')
    ax.grid(True, alpha=0.3)

    # Adjoint residual
    ax = axes[1, 1]
    ax.semilogy(iterations, history['adjoint_residual'], 'm-o', linewidth=2, markersize=4)
    ax.axhline(y=1e-6, color='k', linestyle='--', label='GMRES tolerance')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Adjoint Residual')
    ax.set_title('Adjoint System Residual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sdag_convergence.png', dpi=150)
    print("Saved convergence plot: sdag_convergence.png")
    plt.show()


def plot_trajectory_comparison(result_true, result_final):
    """Compare true and optimized trajectories."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    t_true = result_true['t']
    t_final = result_final['t']

    # Plot first 3 states
    ax = axes[0]
    n_plot = min(3, result_true['x'].shape[0])
    for i in range(n_plot):
        ax.plot(t_true, result_true['x'][i, :], '-', linewidth=2, label=f"{result_true['state_names'][i]} (true)")
        ax.plot(t_final, result_final['x'][i, :], '--', linewidth=2, label=f"{result_final['state_names'][i]} (optimized)")

    ax.set_xlabel('Time')
    ax.set_ylabel('State Values')
    ax.set_title('Differential States: True vs Optimized')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot first 3 algebraic variables
    ax = axes[1]
    n_plot = min(3, result_true['z'].shape[0])
    for i in range(n_plot):
        ax.plot(t_true, result_true['z'][i, :], '-', linewidth=2, label=f"{result_true['alg_names'][i]} (true)")
        ax.plot(t_final, result_final['z'][i, :], '--', linewidth=2, label=f"{result_final['alg_names'][i]} (optimized)")

    ax.set_xlabel('Time')
    ax.set_ylabel('Algebraic Values')
    ax.set_title('Algebraic Variables: True vs Optimized')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sdag_trajectory_comparison.png', dpi=150)
    print("Saved trajectory comparison: sdag_trajectory_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
