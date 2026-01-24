
import numpy as np
import torch
import jax
import jax.numpy as jnp
import json
import yaml
from pathlib import Path

# Import implementations
from src.discrete_adjoint.dae_optimizer_implicit_adjoint_final_fix import DAEOptimizerImplicitAdjoint
from src.pytorch.dae_optimizer_pytorch import BouncingBallModel

def compare_gradients():
    print("=" * 80)
    print("Gradient Comparison: JAX Adjoint vs PyTorch Autograd")
    print("=" * 80)

    # 1. Setup Common Problem
    # -----------------------
    t_span = (0.0, 1.0)
    # Use multiple target points to capture global gradient direction
    target_time = np.linspace(0.1, 0.9, 20)
    
    # "True" parameters for generating target
    g_true = 9.81
    e_true = 0.8
    h0 = 1.0
    v0 = 0.0
    
    # "Perturbed" parameters for gradient evaluation
    g_eval = 8.829 # 10% perturbation (Working Point)
    e_eval = e_true # Keep e fixed to isolate g gradient
    
    # 2. Generate Target (Using PyTorch to ensure physical correctness)
    # -------------------------------------------------------------
    # Note: Linear Interpolation is standard in PyTorch model
    
    model_true = BouncingBallModel(g=g_true, e=e_true, h0=h0, v0=v0)
    with torch.no_grad():
        times_true, h_true, _, _ = model_true.simulate(t_span[1])
        
    # Interpolate at target time
    times_np = times_true.numpy()
    h_np = h_true.numpy()
    y_target_val = np.interp(target_time, times_np, h_np)
    
    print(f"Target Time: {target_time[0]}s")
    print(f"Target Value: {y_target_val[0]}")
    
    # --- PyTorch Autograd ---
    print("\n--- PyTorch Autograd ---")
    
    # Imports for local physics model
    from torchdiffeq import odeint_event, odeint
    
    class SimplePhysics(torch.nn.Module):
        def __init__(self, g_tensor, e_tensor):
            super().__init__()
            self.g = g_tensor
            self.e = e_tensor
            
        def forward(self, t, state):
            h, v = state
            dh = v
            dv = -self.g
            return dh, dv
            
        def event_fn(self, t, state):
            h, v = state
            return h
            
    # Setup Tensors
    g_torch = torch.tensor(g_eval, requires_grad=True, dtype=torch.float64)
    e_torch = torch.tensor(e_eval, requires_grad=False, dtype=torch.float64)
    
    physics = SimplePhysics(g_torch, e_torch)
    
    t0 = torch.tensor([0.0], dtype=torch.float64)
    h0_t = torch.tensor([h0], dtype=torch.float64)
    v0_t = torch.tensor([v0], dtype=torch.float64)
    state = (h0_t, v0_t)
    
    # Simulate logic (simplified from BouncingBallModel)
    current_t = t0
    all_times = [t0]
    all_h = [state[0]]
    nbounces = 5
    t_end = t_span[1]
    ncp = 150
    
    # We must manually implement the loop to accumulate the graph correctly
    for i in range(nbounces):
        if float(current_t) >= t_end:
            break
            
        event_t, solution = odeint_event(
            physics,
            state,
            current_t,
            event_fn=physics.event_fn,
            reverse_time=False,
            atol=1e-8, rtol=1e-8
        )
        
        # Check if event is past end
        if event_t > t_end:
             # Integrate to t_end
             # Differentiable grid: start + (end - start) * unit_grid
             # Note: current_t might be tensor or float.
             unit_grid = torch.linspace(0, 1, ncp, dtype=torch.float64, requires_grad=False)
             if isinstance(current_t, float): current_t = torch.tensor(current_t)
             t_span_diff = t_end - current_t
             tt = current_t + t_span_diff * unit_grid
             
             sol = odeint(physics, state, tt, method='midpoint')
             all_times.append(tt[1:])
             all_h.append(sol[0][1:])
             break
             
        # Event happened
        # Integrate to event
        # Differentiable grid w.r.t event_t
        unit_grid = torch.linspace(0, 1, ncp, dtype=torch.float64, requires_grad=False)
        if isinstance(current_t, float): current_t = torch.tensor(current_t)
        
        t_span_diff = event_t - current_t
        tt = current_t + t_span_diff * unit_grid
        
        if len(tt) > 1:
            sol = odeint(physics, state, tt, method='midpoint')
            all_times.append(tt[1:])
            all_h.append(sol[0][1:])
            
        # State Update (Jump)
        # solution is tuple (h_seq, v_seq). Get last point.
        h_end = solution[0][-1]
        v_end = solution[1][-1]
        
        h_new = h_end + 1e-7 # Lift
        v_new = -e_torch * v_end # Restitution
        
        state = (h_new, v_new)
        current_t = event_t
        
    # Concatenate
    times_pred = torch.cat([t.reshape(-1) for t in all_times])
    h_pred = torch.cat([h.reshape(-1) for h in all_h])
    
    # Interpolation
    def torch_interp(t_eval, t_seq, y_seq):
        idx = torch.bucketize(torch.tensor(t_eval), t_seq) - 1
        idx = torch.clamp(idx, 0, len(t_seq)-2)
        t0 = t_seq[idx]
        t1 = t_seq[idx+1]
        y0 = y_seq[idx]
        y1 = y_seq[idx+1]
        s = (t_eval - t0) / (t1 - t0 + 1e-9)
        return y0 * (1-s) + y1 * s # Linear

    y_pred_torch_list = []
    for t_val in target_time:
        y_val = torch_interp(t_val, times_pred, h_pred)
        y_pred_torch_list.append(y_val)
    y_pred_torch = torch.stack(y_pred_torch_list)
    
    # Loss
    loss_torch = torch.mean((y_pred_torch - torch.tensor(y_target_val))**2)
    
    # Backward
    loss_torch.backward()
    grad_torch = g_torch.grad.item()
    
    print(f"  Loss: {loss_torch.item():.8e}")
    print(f"  Grad (g): {grad_torch:.8f}")
    
    
    # 4. Compute JAX Adjoint Gradients (Sigmoid & Linear)
    # ---------------------------------------------------
    
    dae_data = {
        "states": [{"name": "h", "start": h0}, {"name": "v", "start": v0}],
        "parameters": [{"name": "g", "value": g_eval}, {"name": "e", "value": e_eval}],
        "f": ["der(h) = v", "der(v) = -g"],
        "g": [],
        "h": ["h_obs = h"],
        "when": [{"condition": "h < 0", "reinit": "v = -e * prev(v)"}]
    }
    
    p_init = np.array([g_eval])
    
    methods = ['sigmoid', 'linear']
    
    for method in methods:
        print(f"\n--- JAX Adjoint ({method}) ---")
        
        opt = DAEOptimizerImplicitAdjoint(
            dae_data, 
            ["g"], 
            verbose=True, 
            prediction_method=method,
            blend_sharpness=100.0 if method == 'sigmoid' else 100.0, 
            max_segments=5,
            max_points_per_seg=200
        )
        
        _, loss_adj, grad_adj, _, _, _, _, _, _ = opt.optimization_step(
            t_span,
            target_time,
            y_target_val,
            p_init,
            ncp=100
        )
        
        print(f"  Loss: {loss_adj:.8e}")
        print(f"  Grad (g): {grad_adj[0]:.8f}")
        print(f"  Rel Diff vs PyTorch: {np.abs(grad_adj[0] - grad_torch) / (np.abs(grad_torch) + 1e-9):.2%}")

if __name__ == "__main__":
    compare_gradients()
