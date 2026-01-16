# Configuring JAX to Use CPU

JAX configuration is **global** - you only need to configure it **once** at the start of your program, and it will affect all modules that import JAX.

## Recommended Methods (in order of preference)

### Method 1: Environment Variable (Most Reliable) ⭐

Set the environment variable **before** running Python:

```bash
JAX_PLATFORM_NAME=cpu python example_optimizer.py
```

Or export it for your session:

```bash
export JAX_PLATFORM_NAME=cpu
python example_optimizer.py
```

### Method 2: Set in Main Script (Before Imports)

At the **very top** of your main script (before importing any modules that use JAX):

```python
# example_optimizer.py
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

# NOW import your modules (JAX will be configured when they import it)
import numpy as np
from src.dae_jacobian import DAEOptimizer
from src.dae_solver import DAESolver

# Rest of your code...
```

### Method 3: Use the configure_jax_device() function

```python
from src.dae_jacobian import configure_jax_device

# Call BEFORE any JAX operations
configure_jax_device(use_cpu=True)

# Now use the modules normally
from src.dae_jacobian import DAEOptimizer
```

**Note:** This method may not work if JAX has already compiled functions.

## Important Points

1. **Configure ONCE**: JAX configuration is global. You only need to set it once at the start of your program.

2. **Affects ALL modules**: Once configured, all files that import JAX will use the same device (CPU or GPU).

3. **Timing**: The configuration must happen **before** JAX performs any operations or compiles any functions.

4. **No need to configure in multiple files**: If you configure JAX to use CPU in your main script, all imported modules (like `dae_jacobian.py`, `adjoint_solver.py`, etc.) will automatically use CPU.

## Checking Current Device

When you import `dae_jacobian`, it will automatically print which device JAX is using:

```
JAX initialized with device: cpu (1 device(s))
```

or

```
JAX initialized with device: gpu (1 device(s))
```

## Alternative: Hide GPUs from JAX

To make JAX not see any GPUs at all:

```bash
CUDA_VISIBLE_DEVICES='' python example_optimizer.py
```

## Example Usage

```python
# example_optimizer.py
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Set BEFORE imports

import numpy as np
import json
from src.dae_solver import DAESolver
from src.dae_jacobian import DAEOptimizer  # Will use CPU

# Your optimization code here...
```

That's it! No need to configure JAX in `dae_jacobian.py`, `adjoint_solver.py`, or any other files.
