import jax
import jax.numpy as jnp
from jax import lax

# w_k = concat([x_k, z_k]) or a pytree; keeping vector for simplicity
# r_k returns concatenated residual vector for step k (includes both f and g parts).
def step_residual(wk, wk1, theta, h):
    # unpack
    # xk, zk = ...
    # xk1, zk1 = ...
    # implement trapezoid residuals:
    # rf = xk1 - xk - 0.5*h*(f(xk,zk,theta) + f(xk1,zk1,theta))
    # rg = g(xk,zk,theta)
    # return jnp.concatenate([rf, rg], axis=0)
    raise NotImplementedError

def matvec_AT(w, theta, h, lam):
    """
    Compute y = A(w,theta)^T lam without forming A.
    w: (N+1, nw)
    lam: (N, nr)  # one multiplier per step residual r_k
    returns y: (N+1, nw)
    """
    N = lam.shape[0]
    nw = w.shape[1]

    def one_step(carry, inputs):
        wk, wk1, lamk = inputs

        # local VJP wrt (wk, wk1)
        # We want: (∂wk r_k)^T lamk and (∂wk1 r_k)^T lamk
        def rk_args(wk_, wk1_):
            return step_residual(wk_, wk1_, theta, h)

        # vjp returns pullback that maps cotangent -> cotangents for inputs
        (_, pullback) = jax.vjp(rk_args, wk, wk1)
        dk, nk = pullback(lamk)   # dk: nw, nk: nw

        yk = carry + dk
        new_carry = nk
        return new_carry, yk

    c0 = jnp.zeros((nw,), dtype=w.dtype)
    inputs = (w[:-1], w[1:], lam)         # length N
    cN, y_head = lax.scan(one_step, c0, inputs)
    yN = cN[None, :]                      # (1, nw)
    return jnp.concatenate([y_head, yN], axis=0)

# JIT compile the matvec (crucial: keep shapes static)
matvec_AT_jit = jax.jit(matvec_AT, static_argnames=("h",))
