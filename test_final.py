"""
Final test - using simpler approach without frequency threshold initially.
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
solver = DAESolver(dae_data, verbose=False)

# Test reinitialization works
print("\nVerifying reinitialization logic:")
x_pre = np.array([0.0, -3.0])  # h=0, v=-3.0 (downward)
z_pre = np.array([])
x_post, z_post = solver._apply_reinit(0, 0.0, x_pre, z_pre, x_pre, z_pre)
print(f"Before: h={x_pre[0]}, v={x_pre[1]}")
print(f"After:  h={x_post[0]}, v={x_post[1]}")
print(f"Expected v = -e*prev(v) = -0.8*(-3.0) = 2.4")
print(f"Match: {np.isclose(x_post[1], 2.4)}")

print("\n" + "="*80)
print("Running simulation with events (NO frequency threshold)")
print("="*80)

result = solver.solve_with_events(
    t_span=(0, 5.0),
    ncp=500,
    rtol=1e-6,
    atol=1e-8,
    min_event_delta=0.01,  # Stop if bounces occur faster than every 10ms
    verbose=False
)

print(f"\nResults:")
print(f"  Simulation time: {result['t'][0]:.3f} to {result['t'][-1]:.3f} seconds")
print(f"  Number of time points: {len(result['t'])}")
print(f"  Number of events: {len(result['event_times'])}")
print(f"  Early termination: {result['early_termination']}")

if len(result['event_times']) > 0:
    print(f"\nFirst few events:")
    for i in range(min(5, len(result['event_times']))):
        t_event, idx, (var_name, old_val, new_val) = (
            result['event_times'][i], 
            result['event_indices'][i], 
            result['event_vars_changed'][i]
        )
        print(f"  Event {i+1} at t={t_event:.4f}: {var_name} changed {old_val:.4f} -> {new_val:.4f}")
    
    # Check velocity is reversing with damping
    print("\nVelocity reversal check (should decrease by factor e=0.8 each bounce):")
    for i in range(min(3, len(result['event_times']))):
        old_val, new_val = result['event_vars_changed'][i][1:]
        ratio = new_val / old_val if old_val != 0 else 0
        print(f"  Bounce {i+1}: ratio = {ratio:.4f} (expected ≈ -0.8)")

print(f"\nFinal state:")
print(f"  Height: {result['x'][0, -1]:.6f}")
print(f"  Velocity: {result['x'][1, -1]:.6f}")

print("\n" + "="*80)
print("SUCCESS!" if len(result['event_times']) > 0 and not result['early_termination'] else "NEEDS DEBUGGING")
print("="*80)
