# DAE Parameter Optimizer

A sensitivity-free optimization algorithm for DAE (Differential-Algebraic Equation) parameters using the adjoint method.

## Overview

The `DAEOptimizer` class provides an efficient method for identifying DAE parameters by minimizing the difference between model outputs and target trajectories. It uses:

1. **Adjoint-based gradients** - Efficient gradient computation without forward sensitivity equations
2. **JAX automatic differentiation** - Fast, JIT-compiled Jacobian computations
3. **Selective parameter optimization** - Optimize only specific parameters while keeping others fixed

## Algorithm

For each iteration:

1. **Solve DAE** with current parameters → get state/algebraic variables
2. **Compute loss** = ||h(y, p) - y_target||²
3. **Compute loss gradient** dL/dy w.r.t. states (this is the "b" vector)
4. **Solve adjoint system** J^T λ = dL/dy to get adjoint variables
5. **Compute parameter gradient** dL/dp = -(dR/dp)^T λ using matrix multiplication
6. **Update parameters** p_new = p - step_size * dL/dp (gradient descent)

## Key Features

### Selective Parameter Optimization

You can specify which parameters to optimize while keeping others fixed:

```python
# Optimize only capacitor parameters
optimizer = DAEOptimizer(
    dae_data,
    optimize_params=['C1', 'C2', 'C3']
)
```

### JIT-Compiled Operations

All Jacobian computations are JIT-compiled for maximum performance:
- `compute_jacobian_blocks_jit()` - State Jacobians
- `compute_residual_jacobian_wrt_params_jit()` - Parameter Jacobians
- `trajectory_loss_gradient_analytical_jit()` - Loss gradients

## Usage Example

```python
from src.dae_jacobian import DAEOptimizer
from src.dae_solver import DAESolver
import numpy as np
import json

# Load DAE specification
with open('dae_examples/dae_specification_smooth.json', 'r') as f:
    dae_data = json.load(f)

# Generate reference trajectory (with true parameters)
solver = DAESolver(dae_data)
result = solver.solve(t_span=(0.0, 5.0), ncp=50)
t_ref = result['t']
y_ref = result['y']  # Target trajectory

# Identify capacitor parameters
capacitor_params = [p['name'] for p in dae_data['parameters']
                    if p['name'].startswith('C')]

# Create optimizer (optimize only capacitors)
# Use loss_type='mean' for scale-invariant loss (optional, default is 'sum')
optimizer = DAEOptimizer(dae_data, optimize_params=capacitor_params, loss_type='sum')

# Set initial guess for parameters being optimized
p_init = np.array([0.5, 0.5, 0.5])  # Initial guesses for C1, C2, C3

# Run optimization
result = optimizer.optimize(
    t_array=t_ref,
    y_target=y_ref.T,  # Shape: (n_time, n_outputs)
    p_init=p_init,
    n_iterations=50,
    step_size=0.001,
    tol=1e-6,
    verbose=True
)

# Extract results
p_opt = result['p_opt']      # Optimized parameters only
p_all = result['p_all']      # All parameters (optimized + fixed)
loss_final = result['loss_final']
converged = result['converged']

# Plot optimization history
optimizer.plot_optimization_history()
```

## Constructor Parameters

```python
DAEOptimizer(
    dae_data: dict,                    # DAE specification
    dae_solver: DAESolver = None,      # Optional solver instance
    optimize_params: List[str] = None, # Parameters to optimize (None = all)
    loss_type: str = 'sum'             # Loss function type: 'sum' or 'mean'
)
```

**Loss Function Types:**
- `'sum'`: Loss = sum((y_pred - y_target)²) - Standard sum of squared errors
- `'mean'`: Loss = mean((y_pred - y_target)²) - Mean squared error (MSE)

The choice affects loss magnitude but not optimization direction. Use `'mean'` for scale-invariant loss values that don't depend on trajectory length.

## Optimize Method Parameters

```python
optimizer.optimize(
    t_array: np.ndarray,      # Time points for trajectory
    y_target: np.ndarray,     # Target output trajectory
    p_init: np.ndarray,       # Initial values for optimized params
    n_iterations: int = 100,  # Maximum iterations
    step_size: float = 0.01,  # Gradient descent step size
    tol: float = 1e-6,        # Convergence tolerance
    verbose: bool = True      # Print progress
)
```

## Return Value

```python
{
    'p_opt': np.ndarray,      # Optimized parameter values
    'p_all': np.ndarray,      # All parameters (optimized + fixed)
    'loss_final': float,      # Final loss value
    'history': dict,          # Optimization history
    'converged': bool,        # Whether converged
    'n_iterations': int       # Number of iterations
}
```

## History Dictionary

```python
history = {
    'loss': List[float],           # Loss at each iteration
    'gradient_norm': List[float],  # Gradient norm at each iteration
    'params': List[np.ndarray],    # Optimized params at each iteration
    'params_all': List[np.ndarray],# All params at each iteration
    'step_size': List[float]       # Step size at each iteration
}
```

## Running the Example

```bash
# Run full parameter identification example
python example_optimizer.py
```

This example:
1. Generates a reference trajectory with known parameters
2. Perturbs parameters by 30% for initial guess
3. Optimizes only capacitor parameters
4. Validates results and plots convergence

## Advantages Over Forward Sensitivity

1. **No sensitivity ODEs** - Avoids solving N_params additional ODEs
2. **Memory efficient** - O(N) memory instead of O(N × N_params)
3. **Fast iterations** - Single adjoint solve per iteration
4. **Selective optimization** - Only compute gradients for selected parameters

## Requirements

- Python 3.8+
- JAX (for automatic differentiation)
- NumPy
- scikits.odes (for DAE solver)
- matplotlib (for plotting)

## References

The adjoint method is a classical technique in optimal control and PDE-constrained optimization, adapted here for DAE parameter identification.
