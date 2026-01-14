"""Quick test to see if we can evaluate the equations"""
import json
import numpy as np
from dae_solver import DAESolver

json_path = "dae_examples/dae_specification.json"

print("Loading DAE...")
solver = DAESolver(json_path, use_simplified=True)

print("\nTrying to evaluate f at t=0...")
t = 0.0
x = solver.x0
z = solver.z0

print(f"x shape: {x.shape}, first 5 values: {x[:5]}")
print(f"z shape: {z.shape}, all zeros: {np.allclose(z, 0)}")

print("\nEvaluating g(t=0, x0, z=0)...")
try:
    g_val = solver.eval_g(t, x, z)
    print(f"g shape: {g_val.shape}")
    print(f"g norm: {np.linalg.norm(g_val):.6e}")
    print(f"g first 5: {g_val[:5]}")
    print(f"g max: {np.max(np.abs(g_val)):.6e}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTrying to solve for z using fsolve...")
from scipy.optimize import fsolve
import time

def residual(z):
    return solver.eval_g(t, x, z)

print("Starting fsolve...")
start = time.time()
try:
    z_sol = fsolve(residual, z, full_output=False, xtol=1e-4, maxfev=100)
    elapsed = time.time() - start
    print(f"fsolve completed in {elapsed:.3f} seconds")
    print(f"Solution norm: {np.linalg.norm(z_sol):.6e}")
    print(f"Residual norm: {np.linalg.norm(residual(z_sol)):.6e}")
except Exception as e:
    elapsed = time.time() - start
    print(f"fsolve failed after {elapsed:.3f} seconds: {e}")
