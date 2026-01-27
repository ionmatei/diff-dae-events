
import numpy as np
from scikits.odes.dae import dae

def res(t, y, ydot, result):
    # dx/dt = -x => ydot + y = 0
    result[0] = ydot[0] + y[0]

# Add 'one_step_compute': True to force internal stepping mode if supported by wrapper options
solver = dae('ida', res, rtol=1e-6, atol=1e-6, one_step_compute=True)

t0 = 0.0
y0 = np.array([1.0])
yp0 = np.array([-1.0])

print("Initializing step...")
solver.init_step(t0, y0, yp0)

print("Stepping...")
t_out = 1.0
ret = solver.step(t_out)

print(f"Return type: {type(ret)}")
print(f"Return values: {ret}")

if len(ret) >= 2:
    t_ret = ret[1]
    print(f"Returned time: {t_ret}")
    if t_ret < t_out:
        print("Solver took an internal step (GOOD)")
    else:
        print("Solver jumped to t_out (Exact?)")
