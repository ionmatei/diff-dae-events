"""
Ultra-verbose debug test.
"""

import json
import numpy as np
import sys
sys.path.insert(0, 'src/discrete_adjoint')

from dae_solver import DAESolver

# Load bouncing ball DAE
with open('dae_examples/dae_specification_bouncing_ball.json', 'r') as f:
    dae_data = json.load(f)

solver = DAESolver(dae_data, verbose=False)

print("Running ultra-verbose simulation...")
result = solver.solve_with_events(
    t_span=(0, 2.0),
    ncp=100,
    rtol=1e-6,
    atol=1e-8,
    min_event_delta=None,
    verbose=True  # VERBOSE!
)

print(f"\n\nFINAL RESULTS:")
print(f"Events: {len(result['event_times'])}")
print(f"Event times: {result['event_times']}")
print(f"Early term: {result['early_termination']}")
