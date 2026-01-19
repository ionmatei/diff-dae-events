
import time
import yaml
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres
from jax import vjp, jit

# Enable 64-bit precision for stability
jax.config.update("jax_enable_x64", True)

from src.discrete_adjoint.dae_solver import DAESolver
from src.discrete_adjoint.dae_optimizer_krylov import DAEOptimizerKrylov

def benchmark_krylov(config_path='config/config_cauer.yaml', method='trapezoidal'):
    print("=" * 80)
    print("Benchmarking Krylov Adjoint Solver Components")
    print("=" * 80)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    solver_cfg = config['dae_solver']
    json_path = solver_cfg['dae_specification_file']
    with open(json_path, 'r') as f:
        dae_data = json.load(f)
        
    # Generate reference trajectory (needed for time steps)
    solver_true = DAESolver(dae_data)
    t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
    ncp = solver_cfg['ncp']
    
    print("Generating reference trajectory for time steps...")
    # Add tolerances to avoid IDA error
    result_true = solver_true.solve(
        t_span=t_span, 
        ncp=ncp,
        rtol=solver_cfg.get('rtol', 1e-6),
        atol=solver_cfg.get('atol', 1e-8)
    )
    t_ref = result_true['t']
    
    
    print(f"Time points: {len(t_ref)}")
    
    # Initialize Optimizer
    opt_params = config['optimizer'].get('opt_params', None)
    optimizer = DAEOptimizerKrylov(
        dae_data,
        optimize_params=opt_params,
        method=method,
        krylov_tol=1e-5,
        krylov_maxiter=100
    )
    
    # Prepare data for benchmark
    p_init = np.array([p['value'] for p in dae_data['parameters']])
    if opt_params:
         param_names = [p['name'] for p in dae_data['parameters']]
         p_init_opt = np.array([p_init[param_names.index(name)] for name in opt_params])
    else:
        p_init_opt = p_init
        
    # Convert to JAX arrays
    t_sol = jnp.array(t_ref)
    p_val_jax = jnp.array(p_init_opt)
    
    # Setup Vectors with correct shapes (using n_total from optimizer)
    N = len(t_ref) - 1
    n_total = optimizer.jac.n_total
    
    print(f"System size: {n_total} variables x {N} time steps = {n_total * N} unknowns")
    
    # Random initial guess for benchmarking
    key = jax.random.PRNGKey(0)
    y_unknowns_flat = jax.random.normal(key, (N * n_total,))
    y0 = jnp.zeros(n_total) # Dummy y0
    rhs_vec = jax.random.normal(key, (N * n_total,))
    
    print("\nPreparing JIT compilation...")
    
    # 1. Define residual function closure
    def residual_fun(y_flat):
        return optimizer._compute_full_residual(y_flat, y0, t_sol, p_val_jax)
        
    # 2. Primal pass and VJP setup
    # Warmup
    print("  Warmup VJP setup...")
    start = time.time()
    _, vjp_fun = vjp(residual_fun, y_unknowns_flat)
    
    @jit
    def matvec_action(v):
        return vjp_fun(v)[0]

    matvec_action(rhs_vec).block_until_ready()
    print(f"  Warmup done: {time.time() - start:.4f}s")

    # Benchmark Matvec
    print("\nBenchmarking Matvec (A^T * v)...")
    N_bench = 100
    
    start = time.time()
    for _ in range(N_bench):
        res = matvec_action(rhs_vec)
        res.block_until_ready()
    end = time.time()
    
    avg_matvec = (end - start) / N_bench
    print(f"  Average Matvec time: {avg_matvec*1000:.4f} ms")
    print(f"  Projected 100 iters: {avg_matvec*100:.4f} s")
    print(f"  Projected 50 iters: {avg_matvec*50:.4f} s")
    
    # Benchmark Full GMRES (restart=100)
    print("\nBenchmarking Full GMRES Call (100 iters, restart=100)...")
    
    def run_gmres_100():
        return gmres(matvec_action, rhs_vec, tol=1e-5, maxiter=100, restart=100)
    
    run_gmres_100_jit = jit(run_gmres_100)
    
    # Warmup
    print("  Warmup GMRES (restart=100)...")
    run_gmres_100_jit()[0].block_until_ready()
    
    start = time.time()
    run_gmres_100_jit()[0].block_until_ready()
    end = time.time()
    
    print(f"  GMRES Time (100 iters, restart=100): {end - start:.4f} s")

    # Benchmark Full GMRES (restart=20) - Used in code
    print("\nBenchmarking Full GMRES Call (100 iters, restart=20)...")
    
    def run_gmres_20():
        return gmres(matvec_action, rhs_vec, tol=1e-5, maxiter=100, restart=20)
    
    run_gmres_20_jit = jit(run_gmres_20)
    
    # Warmup
    print("  Warmup GMRES (restart=20)...")
    run_gmres_20_jit()[0].block_until_ready()
    
    start = time.time()
    run_gmres_20_jit()[0].block_until_ready()
    end = time.time()
    
    print(f"  GMRES Time (100 iters, restart=20): {end - start:.4f} s")

    # Benchmark BiCGSTAB
    print("\nBenchmarking BiCGSTAB (100 iters)...")
    from jax.scipy.sparse.linalg import bicgstab
    
    def run_bicgstab():
        return bicgstab(matvec_action, rhs_vec, tol=1e-5, maxiter=100)
    
    run_bicgstab_jit = jit(run_bicgstab)
    
    # Warmup
    print("  Warmup BiCGSTAB...")
    run_bicgstab_jit()[0].block_until_ready()
    
    start = time.time()
    res, info = run_bicgstab_jit()
    res.block_until_ready()
    end = time.time()
    
    print(f"  BiCGSTAB Time (100 iters?): {end - start:.4f} s")
    print(f"  BiCGSTAB Info: {info}")
    
    # Try with larger maxiter to ensure it runs
    # Also GMRES info
    print("\nChecking GMRES info...")
    res_g, info_g = run_gmres_20_jit()
    print(f"  GMRES Info: {info_g}")


    
if __name__ == "__main__":
    benchmark_krylov()
