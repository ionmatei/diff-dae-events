import numpy as np
from scikits.odes.dae import dae

def res(t, y, ydot, r):
    # y = [x, z], ydot = [xdot, zdot]
    x, z = y
    xdot, zdot = ydot

    # Differential equation: xdot + x - z = 0
    r[0] = xdot + x - z

    # Algebraic equation: x + z - 1 = 0
    r[1] = x + z - 1.0

y0    = np.array([0.0, 1.0])   # consistent with x+z=1
ydot0 = np.array([1.0, 0.0])   # consistent with xdot = -x+z = 1

solver = dae('ida', res, atol=1e-10, rtol=1e-10)
tspan  = np.linspace(0.0, 5.0, 200)

sol = solver.solve(tspan, y0, ydot0)
print(sol.values.y[-1])  # last state