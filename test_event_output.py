import json
import sys
sys.path.insert(0, 'src/discrete_adjoint')
from dae_solver import DAESolver

with open('dae_examples/dae_specification_bouncing_ball.json', 'r') as f:
    dae = json.load(f)

solver = DAESolver(dae, verbose=False)
result = solver.solve_with_events(t_span=(0, 2.0), min_event_delta=0.01, verbose=False)

print("Event Variables Changed Format:\n")
print("event_vars_changed is a list of tuples: (var_name, old_val, new_val)\n")

for i, (t, idx, (var_name, old_val, new_val)) in enumerate(
    zip(result['event_times'], result['event_indices'], result['event_vars_changed'])
):
    print(f"Event {i+1} at t={t:.4f}s:")
    print(f"  Variable: '{var_name}'")
    print(f"  Before event: {old_val:.6f}")
    print(f"  After event:  {new_val:.6f}")
    print(f"  Change: {new_val - old_val:.6f}")
    print()

print(f"Total events: {len(result['event_times'])}")
