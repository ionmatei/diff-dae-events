"""Test first few RHS evaluations"""
from dae_solver import DAESolver
import numpy as np
import time

json_path = "dae_examples/dae_specification.json"

print("Loading DAE...")
solver = DAESolver(json_path, use_simplified=True)

# Initialize
t0 = 0.0
solver.z_current = solver.z0.copy()

print("\nTesting RHS evaluation...")
for i in range(3):
    print(f"\nEvaluation {i+1}:")
    start = time.time()
    try:
        dxdt = solver.ode_rhs(t0, solver.x0)
        elapsed = time.time() - start
        print(f"  Time: {elapsed:.3f} seconds")
        print(f"  dxdt norm: {np.linalg.norm(dxdt):.6e}")
        print(f"  dxdt[:5]: {dxdt[:5]}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"  Failed after {elapsed:.3f} seconds: {e}")
        break
