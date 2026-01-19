
import json
import numpy as np
import time
from src.discrete_adjoint.dae_solver import DAESolver

def debug_solver():
    json_path = "dae_examples/dae_specification_smooth.json"
    with open(json_path, 'r') as f:
        dae_data = json.load(f)
        
    print("Loaded DAE data")
    
    solver = DAESolver(dae_data)
    print("Initialized DAESolver")
    
    try:
        # Use same params as config
        result = solver.solve(
            t_span=(0.0, 150.0),
            ncp=300,
            rtol=1e-4,
            atol=1e-4,
            verbose=True
        )
        print("Solve finished")
        print(f"Time points: {len(result['t'])}")
        print(f"Final time: {result['t'][-1]}")
    except Exception as e:
        print(f"Solver failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_solver()
