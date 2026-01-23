"""
Test the Explicit Discrete Adjoint Optimizer on the Bouncing Ball example.

The bouncing ball has:
- States: h (height), v (velocity)
- Parameters: g (gravity), e (restitution coefficient)
- Event: when h < 0, reinit v = -e * prev(v)

We optimize the restitution coefficient e to match observed trajectory.
"""

import os

# Set device before importing JAX
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import numpy as np
import json
import time

# Import optimizer and solver
from src.discrete_adjoint.dae_solver import DAESolver
from src.discrete_adjoint.dae_optimizer_explicit_adjoint import DAEOptimizerExplicitAdjoint


def run_bouncing_ball_test():
    print("=" * 80)
    print("Bouncing Ball - Explicit Discrete Adjoint Test")
    print("=" * 80)

    # Load DAE specification
    json_path = "dae_examples/dae_specification_bouncing_ball.json"
    with open(json_path, 'r') as f:
        dae_data = json.load(f)

    print(f"\nLoaded DAE from: {json_path}")
    print(f"  States: {[s['name'] for s in dae_data['states']]}")
    print(f"  Parameters: {[p['name'] for p in dae_data['parameters']]}")
    print(f"  Events: {len(dae_data.get('when', []))}")

    # True parameters
    p_true = {p['name']: p['value'] for p in dae_data['parameters']}
    print(f"\nTrue parameters: g={p_true['g']}, e={p_true['e']}")

    # =========================================================================
    # Step 1: Generate reference trajectory with true parameters
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 1: Generate Reference Trajectory")
    print("-" * 40)

    solver_true = DAESolver(dae_data, verbose=False)
    t_span = (0.0, 1.0)  # Shorter time span to avoid Zeno barrier with small e
    ncp = 100  # Fewer collocation points

    aug_sol_true = solver_true.solve_augmented(t_span=t_span, ncp=ncp)

    print(f"  Simulation time: {t_span}")
    print(f"  Number of segments: {len(aug_sol_true.segments)}")
    print(f"  Number of events: {len(aug_sol_true.events)}")

    # Extract reference data at uniform times
    # IMPORTANT: Avoid times very close to events to prevent interpolation issues
    n_targets = 20
    t_target = np.linspace(t_span[0] + 0.1, t_span[1] - 0.1, n_targets)

    # Create a temporary optimizer with true parameters to generate targets
    # This ensures targets use the same interpolation as predictions
    optimizer_true = DAEOptimizerExplicitAdjoint(
        dae_data=dae_data,
        optimize_params=['g', 'e'],
        verbose=False
    )
    y_target = optimizer_true.predict_outputs(aug_sol_true, t_target)

    print(f"  Target times: {n_targets} points from {t_target[0]:.2f} to {t_target[-1]:.2f}")
    print(f"  Event times: {[ev.t_event for ev in aug_sol_true.events]}")

    # =========================================================================
    # Step 2: Create initial guess with perturbed parameter
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 2: Create Perturbed Initial Guess")
    print("-" * 40)

    # Perturb both parameters
    dae_data_init = json.loads(json.dumps(dae_data))  # Deep copy

    g_true = p_true['g']
    e_true = p_true['e']
    g_init = g_true * 0.85  # 15% perturbation
    e_init = e_true * 0.8   # 20% perturbation (smaller to avoid Zeno)

    for p in dae_data_init['parameters']:
        if p['name'] == 'g':
            p['value'] = g_init
        if p['name'] == 'e':
            p['value'] = e_init

    print(f"  True g = {g_true}, True e = {e_true}")
    print(f"  Initial g = {g_init:.4f} ({100*(g_init/g_true - 1):.0f}%), Initial e = {e_init} ({100*(e_init/e_true - 1):.0f}%)")

    # =========================================================================
    # Step 3: Create optimizer and run optimization
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 3: Run Optimization")
    print("-" * 40)

    optimizer = DAEOptimizerExplicitAdjoint(
        dae_data=dae_data_init,
        optimize_params=['g', 'e'],  # Optimize both gravity and restitution
        verbose=True
    )

    result = optimizer.optimize(
        t_span=t_span,
        target_times=t_target,
        target_outputs=y_target,
        max_iterations=50,
        step_size=0.05,
        tol=1e-8,
        ncp=ncp,
        print_every=5,
        algorithm='adam'
    )

    # =========================================================================
    # Step 4: Results
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 4: Results")
    print("-" * 40)

    g_opt = result['params'][0]
    e_opt = result['params'][1]
    g_error_pct = 100 * abs(g_opt - g_true) / g_true
    e_error_pct = 100 * abs(e_opt - e_true) / e_true

    print(f"\n  Parameter Recovery:")
    print(f"    True g:      {g_true:.6f}, True e:      {e_true:.6f}")
    print(f"    Initial g:   {g_init:.6f}, Initial e:   {e_init:.6f}")
    print(f"    Optimized g: {g_opt:.6f}, Optimized e: {e_opt:.6f}")
    print(f"    Error g:     {g_error_pct:.2f}%, Error e:     {e_error_pct:.2f}%")

    print(f"\n  Optimization Stats:")
    print(f"    Initial loss: {result['history']['loss'][0]:.6e}")
    print(f"    Final loss:   {result['history']['loss'][-1]:.6e}")
    print(f"    Converged:    {result['converged']}")
    print(f"    Time:         {result['elapsed_time']:.2f}s")

    # =========================================================================
    # Step 5: Validate by re-simulating
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 5: Validation")
    print("-" * 40)

    # Simulate with optimized parameters
    dae_data_opt = json.loads(json.dumps(dae_data))
    for p in dae_data_opt['parameters']:
        if p['name'] == 'g':
            p['value'] = float(g_opt)
        if p['name'] == 'e':
            p['value'] = float(e_opt)

    solver_opt = DAESolver(dae_data_opt, verbose=False)
    aug_sol_opt = solver_opt.solve_augmented(t_span=t_span, ncp=ncp)

    # Use same prediction method as optimizer for consistent comparison
    optimizer_val = DAEOptimizerExplicitAdjoint(
        dae_data=dae_data_opt,
        optimize_params=['g', 'e'],
        verbose=False
    )
    y_opt = optimizer_val.predict_outputs(aug_sol_opt, t_target)
    traj_error = np.linalg.norm(y_opt - y_target) / np.linalg.norm(y_target)

    print(f"  Trajectory relative error: {traj_error:.6e}")

    # =========================================================================
    # Step 6: Plot results
    # =========================================================================
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot trajectories
        ax = axes[0, 0]
        t_true_all, h_true_all = extract_state_trajectory(aug_sol_true, 0)
        t_opt_all, h_opt_all = extract_state_trajectory(aug_sol_opt, 0)
        ax.plot(t_true_all, h_true_all, 'b-', linewidth=2, label=f'True (g={g_true:.2f}, e={e_true})')
        ax.plot(t_opt_all, h_opt_all, 'r--', linewidth=2, label=f'Optimized (g={g_opt:.2f}, e={e_opt:.3f})')
        ax.scatter(t_target, y_target[:, 0], c='k', s=20, zorder=5, label='Targets')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Height h [m]')
        ax.set_title('Height Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot velocity
        ax = axes[0, 1]
        _, v_true_all = extract_state_trajectory(aug_sol_true, 1)
        _, v_opt_all = extract_state_trajectory(aug_sol_opt, 1)
        ax.plot(t_true_all, v_true_all, 'b-', linewidth=2, label='True')
        ax.plot(t_opt_all, v_opt_all, 'r--', linewidth=2, label='Optimized')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity v [m/s]')
        ax.set_title('Velocity Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Loss history
        ax = axes[1, 0]
        ax.semilogy(result['history']['loss'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Loss History')
        ax.grid(True, alpha=0.3)

        # Gradient norm history
        ax = axes[1, 1]
        ax.semilogy(result['history']['gradient_norm'], 'r-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm History')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('bouncing_ball_explicit_adjoint_result.png', dpi=150)
        print("\n  Plot saved to: bouncing_ball_explicit_adjoint_result.png")
        plt.show()

    except ImportError:
        print("\n  Matplotlib not available - skipping plots")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

    return result


def interpolate_augmented_solution(aug_sol, t_query):
    """Interpolate augmented solution at query times."""
    n_states = aug_sol.segments[0].x.shape[1]
    y_out = np.zeros((len(t_query), n_states))

    for i, t_q in enumerate(t_query):
        # Find which segment contains this time
        for seg in aug_sol.segments:
            if seg.t[0] <= t_q <= seg.t[-1]:
                # Linear interpolation within segment
                idx = np.searchsorted(seg.t, t_q, side='right') - 1
                idx = np.clip(idx, 0, len(seg.t) - 2)

                t0, t1 = seg.t[idx], seg.t[idx + 1]
                x0, x1 = seg.x[idx], seg.x[idx + 1]

                h = t1 - t0
                if h > 1e-12:
                    s = (t_q - t0) / h
                    y_out[i] = x0 * (1 - s) + x1 * s
                else:
                    y_out[i] = x0
                break

    return y_out


def extract_state_trajectory(aug_sol, state_idx):
    """Extract full state trajectory from augmented solution."""
    t_all = []
    x_all = []

    for seg in aug_sol.segments:
        t_all.extend(seg.t.tolist())
        x_all.extend(seg.x[:, state_idx].tolist())

    return np.array(t_all), np.array(x_all)


if __name__ == "__main__":
    run_bouncing_ball_test()
