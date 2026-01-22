"""
Debug test for bouncing ball with event handling.
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

print("\nDAE Specification:")
print(json.dumps(dae_data, indent=2))

# Create solver
solver = DAESolver(dae_data, verbose=True)

# Test event parsing manually
print("\n" + "="*80)
print("Testing Event Parsing")
print("="*80)
print(f"When clause: {solver.when_clauses[0]}")
print(f"Zero-crossing expression: {solver.zc_funcs[0]}")
print(f"Reinit equation: {solver.event_reinit_exprs[0]}")
print(f"Reinit variable: {solver.event_reinit_var_names[0]}")

# Test zero-crossing at initial condition
t0 = 0
x0 = solver.x0
z0 = solver.z0
print(f"\nInitial state: h={x0[0]}, v={x0[1]}")
zc = solver.eval_zc(t0, x0, z0)
print(f"Initial zero-crossing: zc={zc[0]} (condition is {'TRUE' if zc[0] < 0 else 'FALSE'})")

# Test at a time when h < 0
x_test = np.array([-0.1, -2.0])  # h = -0.1, v = -2.0
zc_test = solver.eval_zc(t0, x_test, z0)
print(f"\nTest state (h=-0.1, v=-2.0):")
print(f"  Zero-crossing: zc={zc_test[0]} (condition is {'TRUE' if zc_test[0] < 0 else 'FALSE'})")

# Test reinitialization
print("\n" + "="*80)
print("Testing Reinitialization")
print("="*80)
x_pre = np.array([0.0, -3.0])  # at collision, velocity is negative
z_pre = z0
print(f"Pre-event state: h={x_pre[0]}, v={x_pre[1]}")
x_post, z_post = solver._apply_reinit(0, t0, x_pre, z_pre, x_pre, z_pre)
print(f"Post-event state: h={x_post[0]}, v={x_post[1]}")
print(f"Expected new v (using e=0.8): v_new = -e * prev(v) = -0.8 * {x_pre[1]} = {-0.8 * x_pre[1]}")

print("\n" + "="*80)
print("Solve short simulation with events")
print("="*80)

result = solver.solve_with_events(
    t_span=(0, 1.0),  # Shorter time
    ncp=100,
    rtol=1e-6,
    atol=1e-8,
    min_event_delta=0.01,  # Add threshold
    verbose=True
)

print("\n" + "="*80)
print("Results")
print("="*80)
print(f"Final time: {result['t'][-1]:.6f}")
print(f"Number of events: {len(result['event_times'])}")
print(f"Early termination: {result['early_termination']}")
print(f"Event times: {result['event_times']}")
