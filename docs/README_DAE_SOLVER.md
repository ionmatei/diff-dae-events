# DAE Solver Documentation

## Overview

The `dae_solver.py` module provides a solver for Differential-Algebraic Equations (DAEs) specified in JSON format. It uses **SUNDIALS IDA** solver via the `scikits.odes` package, which is highly efficient for large stiff DAE systems.

## Features

- **Reads JSON DAE specifications** with simplified form
- **SUNDIALS IDA integration** - industry-standard DAE solver
- **Handles semi-explicit DAEs**:
  - Differential equations: `dx/dt = f(t, x, z, p)`
  - Algebraic constraints: `0 = g(t, x, z, p)`
  - Output equations: `y = h(t, x, z, p)` (optional)
- **Automatic equation compilation** from string expressions
- **Visualization** of solution trajectories

## Installation

Ensure you have the required packages:

```bash
pip install scikits.odes numpy matplotlib
```

## Usage

### Basic Usage

```python
from dae_solver import DAESolver

# Load DAE from JSON specification
solver = DAESolver("dae_examples/dae_specification.json", use_simplified=True)

# Solve from t=0 to t=10
result = solver.solve(
    t_span=(0.0, 10.0),
    ncp=500,      # number of output points
    rtol=1e-6,    # relative tolerance
    atol=1e-8,    # absolute tolerance
)

# Access results
t = result['t']           # time points
x = result['x']           # differential states (n_states × n_timepoints)
z = result['z']           # algebraic variables (n_alg × n_timepoints)
y = result['y']           # outputs (if defined)
```

### Result Dictionary

The `solve()` method returns a dictionary with:

- `t`: Time points (array of length ncp)
- `x`: Differential states (n_states × ncp array)
- `z`: Algebraic variables (n_alg × ncp array)
- `y`: Outputs (n_outputs × ncp array, if h equations exist)
- `state_names`: List of differential state variable names
- `alg_names`: List of algebraic variable names
- `output_names`: List of output variable names

### Example: Circuit DAE

For the provided circuit example (98 states, 123 algebraic vars):

```python
solver = DAESolver("dae_examples/dae_specification.json")
result = solver.solve(t_span=(0.0, 10.0), ncp=500)

# Solve time: ~24 seconds for 500 time points
# This is significantly faster than solve_ivp with fsolve approach
```

## JSON Specification Format

The solver expects JSON with a `simplified_form` key containing:

```json
{
  "simplified_form": {
    "states": [
      {"name": "x1", "type": "float", "start": 0.0, "orig_name": "..."}
    ],
    "alg_vars": [
      {"name": "z1", "type": "float", "start": null, "orig_name": "..."}
    ],
    "parameters": [
      {"name": "p1", "type": "float", "value": 1.0, "orig_name": "..."}
    ],
    "outputs": [],
    "f": [
      "der(x1) = expression..."
    ],
    "g": [
      "0 = expression..."
    ],
    "h": null
  }
}
```

### Notes on Initial Conditions

- Differential states (`states`) must have valid `start` values
- Algebraic variables (`alg_vars`) can have `null` start values
  - The solver will compute consistent initial conditions automatically
  - Initial guess is set to zero for algebraic variables

## Plotting

Use the included plotting function:

```python
from dae_solver import plot_solution

plot_solution(result, max_vars=10)
```

This creates subplots showing:
1. Differential states over time
2. Algebraic variables over time
3. Outputs over time (if available)

## Performance

**Circuit DAE Example** (98 states + 123 algebraic vars):
- **SUNDIALS IDA**: ~24 seconds for 500 time points
- **Previous approach (solve_ivp + fsolve)**: >2 minutes (timed out)

The SUNDIALS solver is **>5x faster** because:
- Specialized DAE solver (not ODE + algebraic solve)
- No separate nonlinear solve at each step
- Efficient sparse Jacobian handling
- Proven stability for stiff systems

## Advanced Options

### Solver Tolerances

```python
result = solver.solve(
    t_span=(0.0, 10.0),
    rtol=1e-8,   # tighter relative tolerance
    atol=1e-10,  # tighter absolute tolerance
)
```

### Additional IDA Options

Pass additional options to the IDA solver:

```python
result = solver.solve(
    t_span=(0.0, 10.0),
    ncp=1000,
    max_steps=10000,        # maximum number of internal steps
    first_step_size=1e-6,   # initial step size
)
```

## Limitations

1. **Equation evaluation uses `eval()`**: Equations are compiled from strings using Python's `eval()`. This works for most mathematical expressions but has some limitations.

2. **No custom functions**: User-defined functions in equations are not supported unless added to the namespace.

3. **Index-1 DAEs only**: The solver assumes index-1 DAE systems (most physical systems are index-1).

## Troubleshooting

### Solver Flag Warning

If you see "Warning: Solver flag indicates potential issues", the integration may have encountered difficulties but still completed. Check:
- Solution reasonableness
- Try tighter tolerances
- Check for stiff regions in your DAE

### Overflow/Underflow Warnings

Runtime warnings about overflow/underflow in exp/log functions are usually harmless and come from evaluating time-dependent sources or extreme parameter values.

### Slow Performance

If solving is very slow:
- Reduce `ncp` (fewer output points)
- Increase tolerances (less accurate but faster)
- Check for algebraic loops or ill-conditioned equations

## References

- [SUNDIALS IDA Documentation](https://computing.llnl.gov/projects/sundials/ida)
- [scikits.odes Documentation](https://scikits-odes.readthedocs.io/)
