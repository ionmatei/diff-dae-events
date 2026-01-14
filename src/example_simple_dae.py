"""
Simple DAE example using the dae_solver module.

This demonstrates how to use the solver with a simple pendulum DAE.
"""

from dae_solver import DAESolver
import matplotlib.pyplot as plt
import numpy as np

# For this example, we'll use the circuit DAE from the JSON file
# In practice, you would have your own JSON specification

def example_circuit_dae():
    """Example: Solve the circuit DAE from JSON specification."""

    print("=" * 80)
    print("Example: Circuit DAE from JSON")
    print("=" * 80)

    # Load DAE
    solver = DAESolver("dae_examples/dae_specification.json", use_simplified=True)

    print(f"\nSystem size:")
    print(f"  Differential states: {len(solver.state_names)}")
    print(f"  Algebraic variables: {len(solver.alg_names)}")
    print(f"  Parameters: {len(solver.param_names)}")

    # Solve
    result = solver.solve(
        t_span=(0.0, 5.0),
        ncp=200,
        rtol=1e-6,
        atol=1e-8
    )

    # Create custom plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Selected differential states
    ax = axes[0, 0]
    for i in range(min(5, len(solver.state_names))):
        ax.plot(result['t'], result['x'][i, :], label=solver.state_names[i])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Differential States')
    ax.set_title('Differential States (first 5)')
    ax.legend(fontsize=8)
    ax.grid(True)

    # Plot 2: Selected algebraic variables
    ax = axes[0, 1]
    for i in range(min(5, len(solver.alg_names))):
        ax.plot(result['t'], result['z'][i, :], label=solver.alg_names[i])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Algebraic Variables')
    ax.set_title('Algebraic Variables (first 5)')
    ax.legend(fontsize=8)
    ax.grid(True)

    # Plot 3: State trajectory norms
    ax = axes[1, 0]
    x_norms = np.linalg.norm(result['x'], axis=0)
    z_norms = np.linalg.norm(result['z'], axis=0)
    ax.plot(result['t'], x_norms, label='||x|| (states)', linewidth=2)
    ax.plot(result['t'], z_norms, label='||z|| (algebraic)', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('2-Norm')
    ax.set_title('Solution Vector Norms')
    ax.legend()
    ax.grid(True)

    # Plot 4: Phase portrait of first two states
    ax = axes[1, 1]
    if len(solver.state_names) >= 2:
        ax.plot(result['x'][0, :], result['x'][1, :], linewidth=2)
        ax.set_xlabel(solver.state_names[0])
        ax.set_ylabel(solver.state_names[1])
        ax.set_title('Phase Portrait (first 2 states)')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('dae_solution.png', dpi=150)
    print(f"\nPlot saved to: dae_solution.png")
    plt.show()

    # Print statistics
    print("\n" + "=" * 80)
    print("Solution Statistics")
    print("=" * 80)
    print(f"Time span: [{result['t'][0]:.2f}, {result['t'][-1]:.2f}]")
    print(f"Number of time points: {len(result['t'])}")
    print(f"\nDifferential states:")
    print(f"  Max norm: {np.max(x_norms):.6e}")
    print(f"  Final norm: {x_norms[-1]:.6e}")
    print(f"\nAlgebraic variables:")
    print(f"  Max norm: {np.max(z_norms):.6e}")
    print(f"  Final norm: {z_norms[-1]:.6e}")


def demonstrate_custom_usage():
    """Demonstrate how to access and use the solver programmatically."""

    print("\n" + "=" * 80)
    print("Custom Usage Example")
    print("=" * 80)

    # Load solver
    solver = DAESolver("dae_examples/dae_specification.json", use_simplified=True)

    # You can access equation strings
    print(f"\nFirst f equation (differential):")
    print(f"  {solver.f_eqs[0][:80]}...")

    print(f"\nFirst g equation (algebraic):")
    print(f"  {solver.g_eqs[0][:80]}...")

    # You can evaluate equations at specific points
    t = 0.0
    x = solver.x0
    z = solver.z0

    print(f"\nEvaluating equations at t=0:")
    f_val = solver.eval_f(t, x, z)
    g_val = solver.eval_g(t, x, z)

    print(f"  ||f(0, x0, z0)|| = {np.linalg.norm(f_val):.6e}")
    print(f"  ||g(0, x0, z0)|| = {np.linalg.norm(g_val):.6e}")

    # Solve with custom settings
    print(f"\nSolving with custom settings...")
    result = solver.solve(
        t_span=(0.0, 1.0),
        ncp=50,
        rtol=1e-4,
        atol=1e-6
    )

    print(f"  Completed in {len(result['t'])} time points")
    print(f"  Final state norm: {np.linalg.norm(result['x'][:, -1]):.6e}")


if __name__ == "__main__":
    # Run the main example
    example_circuit_dae()

    # Demonstrate custom usage
    demonstrate_custom_usage()

    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)
