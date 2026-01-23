
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Dict

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from discrete_adjoint.dae_solver import AugmentedSolution, TrajectorySegment, EventInfo
from discrete_adjoint.dae_optimizer_event_aware import predict_from_augmented_solution, DAEOptimizerEventAware

def setup_synthetic_data():
    """
    Creates a synthetic AugmentedSolution with 3 segments:
    1. t=[0, 1], x=t, xp=1  (Linear)
    2. t=[1, 2], x=2, xp=0  (Constant, jump at t=1 from 1->2)
    3. t=[2, 3], x=t, xp=1  (Linear, continuous at t=2)
    
    Returns:
        AugmentedSolution
    """
    # Segment 1: t=[0, 1], x=t
    t1 = np.linspace(0, 1, 11)
    x1 = t1[:, None]  # Shape (11, 1)
    xp1 = np.ones_like(x1)
    z1 = np.zeros_like(x1)
    seg1 = TrajectorySegment(t=t1, x=x1, z=z1, xp=xp1)
    
    # Event 1: Jump from x=1 to x=2 at t=1
    evt1 = EventInfo(
        t_event=1.0,
        event_idx=0,
        x_pre=np.array([1.0]),
        z_pre=np.array([0.0]),
        x_post=np.array([2.0]),
        z_post=np.array([0.0])
    )
    
    # Segment 2: t=[1, 2], x=2
    t2 = np.linspace(1, 2, 11)
    x2 = 2.0 * np.ones((11, 1))
    xp2 = np.zeros_like(x2)
    z2 = np.zeros_like(x2)
    seg2 = TrajectorySegment(t=t2, x=x2, z=z2, xp=xp2)
    
    # Event 2: Start ramp again from x=2 at t=2. No jump in value, just slope.
    evt2 = EventInfo(
        t_event=2.0,
        event_idx=1,
        x_pre=np.array([2.0]),
        z_pre=np.array([0.0]),
        x_post=np.array([2.0]),
        z_post=np.array([0.0])
    )
    
    # Segment 3: t=[2, 3], x=t
    t3 = np.linspace(2, 3, 11)
    x3 = t3[:, None]
    xp3 = np.ones_like(x3)
    z3 = np.zeros_like(x3)
    seg3 = TrajectorySegment(t=t3, x=x3, z=z3, xp=xp3)
    
    aug_sol = AugmentedSolution(
        segments=[seg1, seg2, seg3],
        events=[evt1, evt2]
    )
    
    # Convert to JAX-ready dict format expected by optimizer
    aug_sol_jax = {
        'segments': [
            {'t': jnp.array(s.t), 'x': jnp.array(s.x), 'xp': jnp.array(s.xp)} 
            for s in aug_sol.segments
        ],
        'events': [
            {'tau': e.t_event, 'event_idx': e.event_idx} 
            for e in aug_sol.events
        ]
    }
    
    return aug_sol, aug_sol_jax

def test_predict_from_augmented_solution():
    print("\nRunning test_predict_from_augmented_solution...")
    _, aug_sol_jax = setup_synthetic_data()
    
    # Test points
    # 0.5 (Seg 1) -> 0.5
    # 1.0 (Seg 1 end/Event) -> Should be close to 1.0 or blended if blend_sharpness is low
    # 1.5 (Seg 2) -> 2.0
    # 2.5 (Seg 3) -> 2.5
    
    target_times = jnp.array([0.5, 1.5, 2.5])
    expected_values = jnp.array([[0.5], [2.0], [2.5]])
    
    predicted = predict_from_augmented_solution(aug_sol_jax, target_times, blend_sharpness=100.0)
    
    print("Target Times:", target_times)
    print("Predicted:", predicted.reshape(-1))
    print("Expected:", expected_values.reshape(-1))
    
    # Check errors
    errors = jnp.abs(predicted - expected_values)
    max_error = jnp.max(errors)
    print("Max Error:", max_error)
    
    assert max_error < 1e-4, f"Prediction error too high: {max_error}"

if __name__ == "__main__":
    test_predict_from_augmented_solution()
