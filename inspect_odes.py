
import numpy as np
from scikits.odes.dae import dae

print("Checking scikits.odes.dae.dae attributes:")
print(dir(dae))

# Create a dummy solver to check instance methods
def res(t, y, ydot, r):
    r[0] = ydot[0] - 1.0

solver = dae('ida', res, rtol=1e-6, atol=1e-6)
print("\nInstance methods:")
print(dir(solver))
