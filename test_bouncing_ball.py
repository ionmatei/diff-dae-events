"""
Test script for bouncing ball with event handling.

This tests the event handling capabilities of the DAE solver.
"""

import json
import numpy as np
import sys
sys.path.insert(0, 'src/discrete_adjoint')

from dae_solver import DAESolver

# Load bouncing ball DAE
print("Loading bouncing ball DAE...")
with open('dae_examples/dae_specification_bouncing_ball.json', 'r') as f:
    dae_data = json.load(f)

# Create solver
solver = DAESolver(dae_data, verbose=True)

# Solve with events
print("\n" + "="*80)
print("Solving with Events")
print("="*80)

result = solver.solve_with_events(
    t_span=(0, 5),
    ncp=500,
    rtol=1e-6,
    atol=1e-8,
    min_event_delta=0.01,  # Stop if events occur more frequently than 0.01s
    verbose=True
)

print("\n" + "="*80)
print("Results")
print("="*80)

print(f"\nSimulation time: {result['t'][0]:.3f} to {result['t'][-1]:.3f} seconds")
print(f"Number of time points: {len(result['t'])}")
print(f"Number of events: {len(result['event_times'])}")
print(f"Early termination: {result['early_termination']}")

if len(result['event_times']) > 0:
    print(f"\nEvent Details:")
    for i, (t_event, idx, (var_name, old_val, new_val)) in enumerate(zip(
        result['event_times'], result['event_indices'], result['event_vars_changed']
    )):
        print(f"  Event {i+1} at t={t_event:.6f}:")
        print(f"    Condition {idx}: {solver.when_clauses[idx]['condition']}")
        print(f"    Variable '{var_name}': {old_val:.6e} -> {new_val:.6e}")

# Print final state
print(f"\nFinal State:")
print(f"  Height (h): {result['x'][0, -1]:.6f}")
print(f"  Velocity (v): {result['x'][1, -1]:.6f}")

# Plot results
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot height
    ax = axes[0]
    ax.plot(result['t'], result['x'][0, :], 'b-', linewidth=2, label='Height (h)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Mark events
    for t_event in result['event_times']:
        ax.axvline(x=t_event, color='r', linestyle=':', alpha=0.5)
    
    ax.set_ylabel('Height [m]')
    ax.set_title('Bouncing Ball Simulation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot velocity
    ax = axes[1]
    ax.plot(result['t'], result['x'][1, :], 'g-', linewidth=2, label='Velocity (v)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Mark events
    for t_event in result['event_times']:
        ax.axvline(x=t_event, color='r', linestyle=':', alpha=0.5, label='Event' if t_event == result['event_times'][0] else '')
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [m/s]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bouncing_ball_events.png', dpi=150)
    print(f"\nPlot saved to 'bouncing_ball_events.png'")
    plt.show()
    
except ImportError:
    print("\nMatplotlib not available, skipping plot")

print("\n" + "="*80)
print("Test Complete!")
print("="*80)
