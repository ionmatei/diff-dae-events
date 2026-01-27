
import numpy as np
import json
import os
from src.discrete_adjoint.dae_solver import DAESolver

# Define Bouncing Ball DAE manually to avoid dependency issues
dae_data = {
    "states": [
        {"name": "h", "start": 1.0},
        {"name": "v", "start": 0.0}
    ],
    "parameters": [
        {"name": "g", "value": 9.81},
        {"name": "e", "value": 0.8}
    ],
    "f": [
        "der(h) = v",
        "der(v) = -g"
    ],
    "when": [
        {
            "condition": "h < 0",
            "reinit": "v = -e*prev(v)"
        }
    ]
}

solver = DAESolver(dae_data, verbose=False)
t_span = (0.0, 2.0)
ncp = 20

# Run solve_with_events
result = solver.solve_with_events(t_span, ncp=ncp, rtol=1e-6, atol=1e-8, verbose=True)

# Analyze first segment
t = result['t']
dt = np.diff(t)

print(f"\nNumber of time points: {len(t)}")
print(f"Time points: {t}")
print(f"dt values: {dt}")

# Check variance of dt
dt_mean = np.mean(dt)
dt_std = np.std(dt)
print(f"Mean dt: {dt_mean}")
print(f"Std dt: {dt_std}")

if dt_std < 1e-9:
    print("CONCLUSION: Samples are UNIFORMLY spaced (Interpolated).")
else:
    print("CONCLUSION: Samples are NON-UNIFORM (Likely Raw Steps).")
