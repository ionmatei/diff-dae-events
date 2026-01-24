
import numpy as np
import jax
import jax.numpy as jnp
from src.discrete_adjoint.dae_optimizer_implicit_adjoint import DAEOptimizerImplicitAdjoint
import json

def run_check():
    # 1. Setup minimal DAE (Bouncing Ball)
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
        "g": [],
        "h": [
            "h_obs = h"
        ],
        "when": [
            {
                "condition": "h < 0",
                "reinit": "v = -e * prev(v)"
            }
        ]
    }
    
    print("\n--- Gradient Sanity Check at Optimal Parameters (Self-Consistency) ---")
    
    # Finite Difference settings
    epsilon = 1e-5
    p_true_val = 9.81
    p_check = np.array([9.85]) # Small bias from optimal
    
    methods = ['sigmoid', 'linear']
    target_time = np.array([0.6])
    
    for method in methods:
        print(f"\nMethod: {method}")
        
        # 1. Setup Optimizer & Generate Reference (Self-Consistent)
        opt = DAEOptimizerImplicitAdjoint(
            dae_data, 
            ["g"], 
            verbose=False, 
            prediction_method=method,
            blend_sharpness=100.0, 
            max_segments=5,
            max_points_per_seg=200
        )
        
        # Generate target at true parameters (9.81)
        sol_true = opt.forward_solve((0.0, 1.0), p_opt=np.array([p_true_val]), ncp=100)
        y_target_self = opt.predict_outputs(sol_true, target_time)
        
        print(f"  Target (Self): {y_target_self}")
        
        # 2. Compute Adjoint Gradient at Biased Point (9.85)
        _, loss_adj, grad_adj, _, _, _, _, _, _ = opt.optimization_step(
            (0.0, 1.0),
            target_time,
            y_target_self,
            p_check,
            ncp=100
        )
        
        # 3. Compute Finite Difference Gradient at Optimal Point
        loss_plus = opt.optimization_step(
            (0.0, 1.0), target_time, y_target_self, p_check + epsilon, ncp=100
        )[1]
        
        loss_minus = opt.optimization_step(
            (0.0, 1.0), target_time, y_target_self, p_check - epsilon, ncp=100
        )[1]
        
        grad_fd = (loss_plus - loss_minus) / (2 * epsilon)
        
        print(f"  Loss:     {loss_adj:.8e}")
        print(f"  Adj Grad: {grad_adj}")
        print(f"  FD  Grad: [{grad_fd:.8f}]")
        print(f"  Rel Diff: {np.linalg.norm(grad_adj - grad_fd) / (np.linalg.norm(grad_fd) + 1e-12):.2%}")

    
if __name__ == "__main__":
    run_check()
