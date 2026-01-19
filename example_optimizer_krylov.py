"""
Example demonstrating Matrix-Free Krylov Adjoint Solver.

This uses the new DAEOptimizerKrylov which solves the adjoint system using
GMRES + VJP, avoiding the construction of any Jacobian matrices.

Key features:
1. Matrix-Free: O(TN) memory complexity
2. Works for any JAX-differentiable residual (Trapezoidal, BDF2-6)
3. Iterative solver (GMRES)
"""

import os
import argparse
import yaml
import time
import numpy as np
import json
import jax

# Import the new optimizer
from src.discrete_adjoint.dae_solver import DAESolver
from src.discrete_adjoint.dae_optimizer_krylov import DAEOptimizerKrylov

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def example_krylov_adjoint(config_path='config/config_cauer.yaml', method='bdf2', solver='gmres'):
    print("=" * 80)
    print("DAE Parameter Identification with Matrix-Free GMRES Adjoint")
    print("=" * 80)
    print(f"Discretization method: {method}")
    print(f"Linear Solver: {solver}")
    
    config = load_config(config_path)
    solver_cfg = config['dae_solver']
    opt_cfg = config['optimizer']
    
    # Load DAE
    json_path = solver_cfg['dae_specification_file']
    with open(json_path, 'r') as f:
        dae_data = json.load(f)
        
    print(f"Loaded DAE from: {json_path}")
    
    # 1. Generate Reference
    solver_true = DAESolver(dae_data)
    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    
    # Adjust ncp for faster testing? Keep as is.
    # Use ncp from config
    ncp = solver_cfg['ncp']
    
    print("Generating reference trajectory...")
    result_true = solver_true.solve(
        t_span=t_span, 
        ncp=ncp,
        rtol=solver_cfg.get('rtol', 1e-6),
        atol=solver_cfg.get('atol', 1e-8)
    )
    t_ref = result_true['t']
    y_ref = result_true['y']
    
    # 2. Setup Optimization
    p_true = np.array([p['value'] for p in dae_data['parameters']])
    param_names = [p['name'] for p in dae_data['parameters']]
    
    # Perturb
    p_init = p_true.copy()
    np.random.seed(42)
    perturbation = 0.2
    
    opt_params = opt_cfg.get('opt_params', None)
    if opt_params:
        optimize_indices = [param_names.index(name) for name in opt_params]
        for idx in optimize_indices:
            p_init[idx] *= (1.0 + perturbation * (2*np.random.rand() - 1))
    else:
        p_init *= (1.0 + perturbation * (2*np.random.rand() - 1))
        
    dae_data_init = dae_data.copy()
    for i, p_dict in enumerate(dae_data_init['parameters']):
        p_dict['value'] = float(p_init[i])
        
    print(f"Starting optimization...")
    
    # 3. Optimize with Krylov
    optimizer = DAEOptimizerKrylov(
        dae_data_init,
        optimize_params=opt_params,
        method=method,
        krylov_tol=opt_cfg.get('tol_krylov', 1e-5),
        krylov_maxiter=opt_cfg.get('max_iterations_krylov', 200),
        solver_type=solver
    )
    
    if opt_params:
        p_init_opt = np.array([p_init[param_names.index(name)] for name in opt_params])
    else:
        p_init_opt = p_init
        
    start_time = time.time()
    result_opt = optimizer.optimize(
        t_array=t_ref,
        y_target=y_ref.T,
        p_init=p_init_opt,
        n_iterations=opt_cfg['max_iterations'],
        step_size=opt_cfg['step_size'],
        tol=opt_cfg['tol'],
        verbose=True
    )
    end_time = time.time()
    
    print("\n" + "="*80)
    print("Optimization Complete")
    print(f"Time taken: {end_time - start_time:.2f} s")
    print(f"Initial Loss: {result_opt['history']['loss'][0]:.6e}")
    print(f"Final Loss:   {result_opt['loss_final']:.6e}")
    print(f"Reduction:    {result_opt['history']['loss'][0]/result_opt['loss_final']:.2f}x")
    
    p_opt_all = result_opt['p_all']
    print("\nParameter Calibration:")
    for i, name in enumerate(param_names):
        print(f"  {name:10s}: True={p_true[i]:.4f}, Init={p_init[i]:.4f}, Opt={p_opt_all[i]:.4f}")
        
    return result_opt

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='trapezoidal')
    parser.add_argument('--solver', default='gmres', help='gmres or bicgstab')
    args = parser.parse_args()
    
    example_krylov_adjoint(method=args.method, solver=args.solver)
