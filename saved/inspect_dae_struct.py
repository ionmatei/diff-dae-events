
import sys
import os
sys.path.append(os.getcwd())
import jax.numpy as jnp
from src.discrete_adjoint.dae_solver import DAESolver
import debug.verify_residual_gmres as dense

def inspect():
    dae_data, solver_cfg = dense.load_system('config/config_bouncing_ball.yaml')
    
    print("States:", [s['name'] for s in dae_data['states']])
    print("When Clauses:", dae_data.get('when', []))
    
    funcs = dense.create_jax_functions(dae_data)
    f_fn, g_fn, h_fn, guard_fn, reinit_res_fn, reinit_vars, dims = funcs
    n_x, n_z, n_p = dims
    
    print(f"Dims: n_x={n_x}, n_z={n_z}")
    print(f"Reinit Vars: {reinit_vars}")
    
    # Test reinit_res_fn size
    # Dummy inputs
    t = 0.0
    x = jnp.zeros(n_x)
    z = jnp.zeros(n_z) if n_z > 0 else jnp.array([])
    p = jnp.zeros(n_p)
    
    res = reinit_res_fn(t, x, z, x, z, p)
    print(f"Reinit Residual Shape: {res.shape}")
    print(f"Reinit Residual: {res}")
    
    # Calculate expected n_res
    # Guard (1)
    # Reinit (res.shape[0])
    # Continuity (n_x - len(reinit_vars) ??? NO. reinit_vars logic counts duplicates?)
    
    r_c_count = 0
    for k in range(n_x):
        is_r = False
        for (type_, idx_) in reinit_vars:
             if type_=='state' and idx_==k: is_r=True
        if not is_r: r_c_count += 1
        
    print(f"Continuity Count: {r_c_count}")
    
    total_res = 1 + res.shape[0] + r_c_count + n_z
    print(f"Total Event Residuals: {total_res}")
    print(f"Expected System Matrix Cols: {total_res}")
    print(f"Expected System Matrix Rows: {n_x + n_z + 1}")

if __name__ == "__main__":
    inspect()
