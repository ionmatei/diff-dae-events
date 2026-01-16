"""
Simple example demonstrating the adjoint solver for DAE sensitivity analysis.

This example shows how to use the adjoint solver with Jacobian blocks
computed from dae_jacobian.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
from src.adjoint_solver import (
    solve_adjoint_system_jit,
    solve_adjoint_system_multiple_rhs_jit,
    verify_adjoint_solution_jit,
    print_verification_results,
    compute_parameter_sensitivity_jit
)


def example_1_synthetic():
    """Example 1: Synthetic problem to demonstrate basic usage."""
    print("=" * 80)
    print("Example 1: Synthetic Adjoint System")
    print("=" * 80)

    # Problem setup
    N = 20   # Number of time steps
    m = 4    # State dimension

    print(f"\nProblem size:")
    print(f"  N (time steps): {N}")
    print(f"  m (state dimension): {m}")
    print(f"  Total unknowns: {N * m}")

    # Generate random well-conditioned Jacobian blocks
    key = jax.random.PRNGKey(42)
    key, *subkeys = jax.random.split(key, 4)

    # J_curr[k] = dr_{k+1}/dy_{k+1}
    J_curr = jax.random.normal(subkeys[0], (N, m, m)) + jnp.eye(m) * 10.0

    # J_prev[k] = dr_{k+1}/dy_k
    J_prev = jax.random.normal(subkeys[1], (N, m, m)) * 0.3

    # Right-hand side (e.g., from output sensitivity)
    b = jnp.zeros((N, m))
    b = b.at[-1, 0].set(1.0)  # Sensitivity to first state at final time

    print(f"\nRight-hand side:")
    print(f"  Non-zero only at final time: b[-1, 0] = 1.0")

    # Solve adjoint system
    print("\nSolving adjoint system...")
    lam = solve_adjoint_system_jit(J_prev, J_curr, b)

    print(f"\nSolution computed!")
    print(f"  Solution shape: {lam.shape}")
    print(f"  Solution norm: {jnp.linalg.norm(lam):.6e}")
    print(f"  Max |λ|: {jnp.max(jnp.abs(lam)):.6e}")

    # Verify solution
    print("\nVerifying solution...")
    residuals, max_res = verify_adjoint_solution_jit(J_prev, J_curr, b, lam)
    print_verification_results(residuals, max_res)

    return J_prev, J_curr, b, lam


def example_2_multiple_rhs():
    """Example 2: Multiple right-hand sides for batch sensitivity."""
    print("\n" + "=" * 80)
    print("Example 2: Multiple RHS (Batch Sensitivity)")
    print("=" * 80)

    N = 15
    m = 3
    n_outputs = 4  # Compute sensitivity w.r.t. 4 different outputs

    print(f"\nProblem size:")
    print(f"  N = {N}, m = {m}, n_outputs = {n_outputs}")

    # Generate Jacobian blocks
    key = jax.random.PRNGKey(100)
    key, *subkeys = jax.random.split(key, 3)

    J_curr = jax.random.normal(subkeys[0], (N, m, m)) + jnp.eye(m) * 8.0
    J_prev = jax.random.normal(subkeys[1], (N, m, m)) * 0.25

    # Multiple RHS: sensitivity to different output components
    B = jnp.zeros((N, m, n_outputs))
    for i in range(n_outputs):
        # Each output is sensitive to a different state component
        B = B.at[-1, i % m, i].set(1.0)

    print(f"\nRight-hand side shape: {B.shape}")
    print(f"  Each RHS has sensitivity at final time")

    # Solve for all RHS simultaneously
    print(f"\nSolving {n_outputs} adjoint systems simultaneously...")
    LAM = solve_adjoint_system_multiple_rhs_jit(J_prev, J_curr, B)

    print(f"\nSolutions computed!")
    print(f"  Solutions shape: {LAM.shape}")
    print(f"  Expected shape: ({N}, {m}, {n_outputs})")

    # Verify each solution
    print("\nVerifying solutions:")
    for i in range(n_outputs):
        print(f"\n  RHS {i+1}/{n_outputs}:")
        residuals_i, max_res_i = verify_adjoint_solution_jit(
            J_prev, J_curr, B[:, :, i], LAM[:, :, i]
        )
        print(f"    Max residual: {max_res_i:.6e}")

    return J_prev, J_curr, B, LAM


def example_3_parameter_sensitivity():
    """Example 3: Computing parameter sensitivities."""
    print("\n" + "=" * 80)
    print("Example 3: Parameter Sensitivity")
    print("=" * 80)

    N = 10
    m = 3
    n_params = 5

    print(f"\nProblem size:")
    print(f"  N = {N}, m = {m}, n_params = {n_params}")

    # Generate Jacobian blocks
    key = jax.random.PRNGKey(200)
    key, *subkeys = jax.random.split(key, 4)

    J_curr = jax.random.normal(subkeys[0], (N, m, m)) + jnp.eye(m) * 7.0
    J_prev = jax.random.normal(subkeys[1], (N, m, m)) * 0.2
    J_param = jax.random.normal(subkeys[2], (N, m, n_params)) * 0.1

    # Objective: minimize sum of states at final time
    dL_dy = jnp.zeros((N, m))
    dL_dy = dL_dy.at[-1, :].set(1.0)

    print(f"\nObjective gradient:")
    print(f"  dL/dy = [0, 0, ..., ones(m)]")
    print(f"  Objective: L = sum(y_final)")

    # Compute parameter sensitivity
    print(f"\nComputing sensitivity to {n_params} parameters...")
    dL_dp = compute_parameter_sensitivity_jit(J_prev, J_curr, J_param, dL_dy)

    print(f"\nParameter sensitivities:")
    for i, sens in enumerate(dL_dp):
        print(f"  dL/dp[{i}] = {sens:.6e}")

    # Interpretation
    print(f"\nInterpretation:")
    most_sensitive = int(jnp.argmax(jnp.abs(dL_dp)))
    print(f"  Most sensitive parameter: p[{most_sensitive}]")
    print(f"  Sensitivity magnitude: {abs(dL_dp[most_sensitive]):.6e}")

    return dL_dp


def example_4_integration_with_dae():
    """Example 4: Integration with DAE Jacobian computation."""
    print("\n" + "=" * 80)
    print("Example 4: Integration with DAE System")
    print("=" * 80)

    try:
        from src.dae_jacobian import DAEJacobian
        from src.dae_solver import DAESolver
        import json
        import os

        # Check if example file exists
        json_path = "dae_examples/dae_specification_smooth.json"

        if not os.path.exists(json_path):
            print(f"\nDAE example file not found: {json_path}")
            print("Skipping DAE integration example.")
            return None

        print(f"\nLoading DAE from: {json_path}")

        with open(json_path, 'r') as f:
            dae_data = json.load(f)

        # Create Jacobian and solver objects
        jac = DAEJacobian(dae_data)
        solver = DAESolver(dae_data)

        # Solve DAE
        print("\nSolving DAE system...")
        result = solver.solve(t_span=(0.0, 2.0), ncp=30, atol=1e-4, rtol=1e-4)

        t_array = result['t']
        y_array = np.vstack([result['x'], result['z']])

        print(f"DAE solution computed:")
        print(f"  Time points: {len(t_array)}")
        print(f"  State dimension: {y_array.shape[0]}")

        # Compute Jacobian blocks
        print("\nComputing Jacobian blocks...")
        J_prev_list, J_curr_list = jac.compute_jacobian_blocks_jit(t_array, y_array)

        N_dae = len(J_curr_list)
        m_dae = J_curr_list[0].shape[0]

        print(f"Jacobian blocks computed:")
        print(f"  Number of blocks: {N_dae}")
        print(f"  Block size: {m_dae} × {m_dae}")

        # Define objective: minimize first differential state at final time
        dL_dy = jnp.zeros((N_dae, m_dae))
        dL_dy = dL_dy.at[-1, 0].set(1.0)

        print(f"\nObjective: Minimize x[0] at final time")

        # Solve adjoint system
        print("\nSolving adjoint system for DAE...")
        lam = solve_adjoint_system_jit(J_prev_list, J_curr_list, dL_dy)

        print(f"\nAdjoint solution:")
        print(f"  Shape: {lam.shape}")
        print(f"  Norm: {jnp.linalg.norm(lam):.6e}")

        # Verify
        print("\nVerifying adjoint solution...")
        residuals, max_res = verify_adjoint_solution_jit(
            J_prev_list, J_curr_list, dL_dy, lam
        )
        print_verification_results(residuals, max_res)

        # Compute parameter sensitivity
        print("\nComputing parameter sensitivities...")
        J_param_list = jac.compute_parameter_jacobian(t_array, y_array)

        dL_dp = compute_parameter_sensitivity_jit(
            J_prev_list, J_curr_list, J_param_list, dL_dy
        )

        print(f"\nParameter sensitivities:")
        param_names = [p['name'] for p in dae_data['parameters']]
        for i, (name, sens) in enumerate(zip(param_names, dL_dp)):
            print(f"  dL/d{name} = {sens:.6e}")

        print("\n✓ DAE integration example completed successfully!")

        return lam, dL_dp

    except ImportError as e:
        print(f"\nCould not import required modules: {e}")
        print("Please ensure dae_jacobian.py and dae_solver.py are available.")
        return None
    except Exception as e:
        print(f"\nError in DAE integration: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("ADJOINT SOLVER EXAMPLES")
    print("=" * 80)

    # Example 1: Basic usage
    J_prev, J_curr, b, lam = example_1_synthetic()

    # Example 2: Multiple RHS
    example_2_multiple_rhs()

    # Example 3: Parameter sensitivity
    example_3_parameter_sensitivity()

    # Example 4: DAE integration (if available)
    example_4_integration_with_dae()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
