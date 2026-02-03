
import os
import sys
import json
import numpy as np
import yaml
import matplotlib.pyplot as plt

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

# Import existing run functions
from src.run.optimization_jax_bouncing_ball_block import run_optimization_test as run_jax_single
from src.run.optimization_pytorch_bouncing_ball import run_bouncing_ball_test as run_pytorch_single, load_config

def run_benchmark():
    print("="*80)
    print("BENCHMARK: Single Bouncing Ball (JAX vs PyTorch)")
    print("="*80)
    
    results_dir = os.path.join(root_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # --- 1. Run JAX Optimization ---
    print("\n>>> Running JAX Optimization...")
    # The JAX script uses internal config loading, but we can rely on it using the default config file
    # for comparable settings if we don't modify it. 
    # run_optimization_test() takes no args and loads 'config/config_bouncing_ball.yaml' internally.
    jax_result = run_jax_single()
    
    # --- 2. Run PyTorch Optimization ---
    print("\n>>> Running PyTorch Optimization...")
    # Load valid config to pass
    config_path = os.path.join(root_dir, 'config/config_bouncing_ball.yaml')
    config = load_config(config_path)
    
    # Ensure settings match what we want if needed, but defaults in yaml should be consistent 
    # between the two scripts as they read the same file (mostly).
    # optimization_jax_bouncing_ball_block.py reads 'config/config_bouncing_ball.yaml'
    # optimization_pytorch_bouncing_ball.py reads whatever is passed.
    
    pt_result = run_pytorch_single(config)
    
    # --- 3. Combine and Save Results ---
    combined_results = {
        'jax': jax_result,
        'pytorch': pt_result
    }
    
    # Helper to convert numpy/tensor types to json serializable
    def default_serializer(obj):
        if isinstance(obj, (np.integer, np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    json_path = os.path.join(results_dir, 'benchmark_single_ball.json')
    with open(json_path, 'w') as f:
        json.dump(combined_results, f, indent=4, default=default_serializer)
        
    print("\n" + "="*80)
    print(f"Benchmark Results Saved to: {json_path}")
    print("="*80)
    
    # Quick Summary Print
    print(f"{'Metric':<25} | {'JAX':<20} | {'PyTorch':<20}")
    print("-" * 70)
    print(f"{'NCP':<25} | {jax_result['ncp']:<20} | {pt_result['ncp']:<20}")
    print(f"{'Avg Iter Time (ms)':<25} | {jax_result['avg_iter_time']:<20.4f} | {pt_result['avg_iter_time']:<20.4f}")
    print(f"{'Final Loss':<25} | {jax_result['final_validation_loss']:<20.6e} | {pt_result['final_validation_loss']:<20.6e}")
    print(f"{'Iterations':<25} | {jax_result['iterations']:<20} | {pt_result['iterations']:<20}")

if __name__ == "__main__":
    run_benchmark()
