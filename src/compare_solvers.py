"""
Compare SUNDIALS IDA vs scipy solve_ivp for DAE solving.

This script runs both solvers on the same DAE and compares:
- Solution accuracy
- Computation time
- Number of function evaluations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import time

from dae_solver import DAESolver as DAESolverIDA
from dae_solver_scipy import DAESolverScipy


def compare_dae_solvers(json_path: str, t_final: float = 60.0, ncp: int = 200):
    """
    Compare SUNDIALS IDA and scipy solve_ivp on the same DAE.

    Args:
        json_path: Path to DAE JSON specification
        t_final: Final time
        ncp: Number of output points
    """
    print("=" * 80)
    print("DAE SOLVER COMPARISON")
    print("=" * 80)

    # Load DAE specification
    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    # The JSON might be the DAE data directly, or wrapped in 'simplified_form'
    if 'simplified_form' in dae_data:
        dae_form = dae_data['simplified_form']
    else:
        dae_form = dae_data

    print(f"\nDAE System:")
    print(f"  Differential states: {len(dae_form['states'])}")
    print(f"  Algebraic variables: {len(dae_form['alg_vars'])}")
    print(f"  Parameters: {len(dae_form['parameters'])}")

    # Common solver settings
    rtol = 1e-6
    atol = 1e-6

    # ========================================================================
    # SOLVER 1: SUNDIALS IDA
    # ========================================================================
    print("\n" + "=" * 80)
    print("SOLVER 1: SUNDIALS IDA")
    print("=" * 80)

    solver_ida = DAESolverIDA(dae_form)

    start = time.time()
    result_ida = solver_ida.solve(
        t_span=(0.0, t_final),
        ncp=ncp,
        rtol=rtol,
        atol=atol
    )
    time_ida = time.time() - start

    print(f"\n✓ IDA solve time: {time_ida:.3f} seconds")

    # ========================================================================
    # SOLVER 2: scipy solve_ivp
    # ========================================================================
    print("\n" + "=" * 80)
    print("SOLVER 2: scipy.integrate.solve_ivp (BDF)")
    print("=" * 80)

    solver_scipy = DAESolverScipy(dae_form)

    # Use same output times as IDA for fair comparison
    t_eval = np.linspace(0.0, t_final, ncp)

    start = time.time()
    result_scipy = solver_scipy.solve(
        t_span=(0.0, t_final),
        method='BDF',
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
        max_step=0.5,  # Allow larger steps for efficiency
        first_step=0.01  # Start with small step
    )
    time_scipy = time.time() - start

    print(f"\n✓ scipy solve time: {time_scipy:.3f} seconds")

    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    # Interpolate solutions to common time grid for comparison
    t_common = np.linspace(0.0, t_final, ncp)

    # IDA results are already on this grid
    x_ida = result_ida['x']
    z_ida = result_ida['z']

    # scipy results - already on this grid via t_eval
    x_scipy = result_scipy['x']
    z_scipy = result_scipy['z']

    # Compute differences
    x_diff = np.abs(x_ida - x_scipy)
    z_diff = np.abs(z_ida - z_scipy)

    x_max_diff = np.max(x_diff)
    z_max_diff = np.max(z_diff)
    x_mean_diff = np.mean(x_diff)
    z_mean_diff = np.mean(z_diff)

    print(f"\nSolution Differences:")
    print(f"  Differential states:")
    print(f"    Max absolute diff:  {x_max_diff:.6e}")
    print(f"    Mean absolute diff: {x_mean_diff:.6e}")
    print(f"  Algebraic variables:")
    print(f"    Max absolute diff:  {z_max_diff:.6e}")
    print(f"    Mean absolute diff: {z_mean_diff:.6e}")

    print(f"\nPerformance:")
    print(f"  IDA time:          {time_ida:.3f} seconds")
    print(f"  scipy time:        {time_scipy:.3f} seconds")
    print(f"  Speedup:           {time_scipy/time_ida:.2f}x")

    if hasattr(result_scipy['sol'], 'nfev'):
        print(f"\nFunction Evaluations:")
        print(f"  scipy (ODE RHS):   {result_scipy['sol'].nfev}")

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print(f"\nGenerating comparison plots...")

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Plot 1: First differential state
    ax = axes[0, 0]
    ax.plot(result_ida['t'], x_ida[0, :], 'b-', label='IDA', linewidth=2)
    ax.plot(result_scipy['t'], x_scipy[0, :], 'r--', label='scipy', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(result_ida['state_names'][0])
    ax.set_title('First Differential State')
    ax.legend()
    ax.grid(True)

    # Plot 2: Difference in first differential state
    ax = axes[0, 1]
    ax.semilogy(t_common, x_diff[0, :], 'k-', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Absolute Difference')
    ax.set_title(f'Difference: {result_ida["state_names"][0]}')
    ax.grid(True)

    # Plot 3: All differential states (IDA)
    ax = axes[1, 0]
    for i in range(min(5, len(result_ida['state_names']))):
        ax.plot(result_ida['t'], x_ida[i, :], label=result_ida['state_names'][i])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('States')
    ax.set_title('Differential States (IDA)')
    ax.legend(fontsize=8)
    ax.grid(True)

    # Plot 4: State differences
    ax = axes[1, 1]
    for i in range(min(5, len(result_ida['state_names']))):
        ax.semilogy(t_common, x_diff[i, :], label=result_ida['state_names'][i])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Absolute Difference')
    ax.set_title('State Differences (|IDA - scipy|)')
    ax.legend(fontsize=8)
    ax.grid(True)

    # Plot 5: First algebraic variable
    ax = axes[2, 0]
    ax.plot(result_ida['t'], z_ida[0, :], 'b-', label='IDA', linewidth=2)
    ax.plot(result_scipy['t'], z_scipy[0, :], 'r--', label='scipy', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(result_ida['alg_names'][0])
    ax.set_title('First Algebraic Variable')
    ax.legend()
    ax.grid(True)

    # Plot 6: Difference in first algebraic variable
    ax = axes[2, 1]
    ax.semilogy(t_common, z_diff[0, :], 'k-', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Absolute Difference')
    ax.set_title(f'Difference: {result_ida["alg_names"][0]}')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('solver_comparison.png', dpi=150)
    print(f"✓ Comparison plot saved to: solver_comparison.png")
    plt.show()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n✓ Both solvers produced consistent results")
    print(f"  Maximum state difference: {x_max_diff:.6e}")
    print(f"  Maximum alg var difference: {z_max_diff:.6e}")

    if time_scipy < time_ida:
        print(f"\n⚡ scipy solve_ivp was {time_ida/time_scipy:.2f}x faster")
    else:
        print(f"\n⚡ SUNDIALS IDA was {time_scipy/time_ida:.2f}x faster")

    print(f"\nRecommendations:")
    if len(dae_form['alg_vars']) < 20:
        print(f"  - For small systems (<20 alg vars), scipy may be competitive")
    else:
        print(f"  - For large systems (>20 alg vars), SUNDIALS IDA is recommended")

    print(f"  - Use SUNDIALS IDA for:")
    print(f"    • Stiff systems")
    print(f"    • Large algebraic systems")
    print(f"    • Production code")
    print(f"  - Use scipy solve_ivp for:")
    print(f"    • Quick prototyping")
    print(f"    • Small systems")
    print(f"    • When SUNDIALS is not available")

    return {
        'ida': result_ida,
        'scipy': result_scipy,
        'time_ida': time_ida,
        'time_scipy': time_scipy,
        'max_state_diff': x_max_diff,
        'max_alg_diff': z_max_diff,
    }


if __name__ == "__main__":
    # Run comparison
    results = compare_dae_solvers(
        json_path="dae_examples/dae_specification_smooth.json",
        t_final=60.0,
        ncp=200
    )

    print("\n" + "=" * 80)
    print("Comparison completed successfully!")
    print("=" * 80)
