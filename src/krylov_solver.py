import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable, NamedTuple, Optional

# -----------------------------
# Restarted GMRES(m) (JIT-friendly)
# -----------------------------

class GMRESInfo(NamedTuple):
    converged: jnp.ndarray
    iters_total: jnp.ndarray
    restarts_used: jnp.ndarray
    residual_norm: jnp.ndarray
    b_norm: jnp.ndarray

def _safe_norm(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.maximum(jnp.vdot(x, x).real, 0.0))

def _apply_givens(a: jnp.ndarray, b: jnp.ndarray):
    a_abs = jnp.abs(a)
    b_abs = jnp.abs(b)
    r = jnp.sqrt(a_abs * a_abs + b_abs * b_abs)
    c = jnp.where(r > 0, a / r, jnp.array(1.0, dtype=a.dtype))
    s = jnp.where(r > 0, b / r, jnp.array(0.0, dtype=b.dtype))
    return c, s, r

def _back_substitute(R: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    m = g.shape[0]
    y0 = jnp.zeros_like(g)
    # Precompute indices mask outside the loop
    indices = jnp.arange(m)

    def body(i, y):
        k = m - 1 - i
        # Use masking to avoid dynamic slicing issues
        # We want: rhs = g[k] - dot(R[k, k+1:], y[k+1:])
        mask = indices > k
        # Multiply R[k] and y element-wise with mask, then sum
        R_row = R[k]
        rhs = g[k] - jnp.sum(jnp.where(mask, R_row * y, 0.0))
        yk = rhs / R[k, k]
        return y.at[k].set(yk)

    return lax.fori_loop(0, m, body, y0)

def gmres_restarted(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    x0: Optional[jnp.ndarray] = None,
    *,
    m: int,
    max_restarts: int,
    tol: float = 1e-6,
    M_solve: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
):
    if x0 is None:
        x0 = jnp.zeros_like(b)

    if M_solve is None:
        def M_solve(v): return v

    def A_hat(v):
        return M_solve(matvec(v))

    b_hat = M_solve(b)
    b_norm = _safe_norm(b_hat)
    b_norm_safe = jnp.where(b_norm > 0, b_norm, jnp.array(1.0, dtype=b_hat.dtype))

    r0 = b_hat - A_hat(x0)
    beta0 = _safe_norm(r0)

    def is_converged(res_norm):
        return res_norm <= (tol * b_norm_safe)

    class State(NamedTuple):
        x: jnp.ndarray
        res_norm: jnp.ndarray
        iters_total: jnp.ndarray
        restarts_used: jnp.ndarray

    init_state = State(
        x=x0,
        res_norm=beta0,
        iters_total=jnp.array(0, dtype=jnp.int32),
        restarts_used=jnp.array(0, dtype=jnp.int32),
    )

    def one_restart(state: State) -> State:
        x = state.x
        r = b_hat - A_hat(x)
        beta = _safe_norm(r)

        n = b.shape[0]
        V = jnp.zeros((m + 1, n), dtype=b.dtype)
        H = jnp.zeros((m + 1, m), dtype=b.dtype)
        cs = jnp.zeros((m,), dtype=b.dtype)
        sn = jnp.zeros((m,), dtype=b.dtype)
        g = jnp.zeros((m + 1,), dtype=b.dtype)

        v0 = jnp.where(beta > 0, r / beta, jnp.zeros_like(r))
        V = V.at[0].set(v0)
        g = g.at[0].set(beta)

        def arnoldi_step(j, carry):
            V, H, cs, sn, g = carry
            vj = V[j]
            w = A_hat(vj)

            def mgs(i, inner):
                w_cur, H_cur = inner
                hij = jnp.vdot(V[i], w_cur)
                w_new = w_cur - hij * V[i]
                H_new = H_cur.at[i, j].set(hij)
                return (w_new, H_new)

            w, H = lax.fori_loop(0, j + 1, mgs, (w, H))

            h_next = _safe_norm(w)
            H = H.at[j + 1, j].set(h_next)
            v_next = jnp.where(h_next > 0, w / h_next, jnp.zeros_like(w))
            V = V.at[j + 1].set(v_next)

            def apply_prev(i, H_cs_sn):
                H_cur, cs_cur, sn_cur = H_cs_sn
                c = cs_cur[i]
                s = sn_cur[i]
                h_i = H_cur[i, j]
                h_ip1 = H_cur[i + 1, j]
                h_i_new = c * h_i + s * h_ip1
                h_ip1_new = -jnp.conj(s) * h_i + c * h_ip1
                H_cur = H_cur.at[i, j].set(h_i_new)
                H_cur = H_cur.at[i + 1, j].set(h_ip1_new)
                return (H_cur, cs_cur, sn_cur)

            H, cs, sn = lax.fori_loop(0, j, apply_prev, (H, cs, sn))

            c, s, _ = _apply_givens(H[j, j], H[j + 1, j])
            cs = cs.at[j].set(c)
            sn = sn.at[j].set(s)

            h_jj = H[j, j]
            h_j1j = H[j + 1, j]
            H = H.at[j, j].set(c * h_jj + s * h_j1j)
            H = H.at[j + 1, j].set(-jnp.conj(s) * h_jj + c * h_j1j)

            gj = g[j]
            gj1 = g[j + 1]
            g = g.at[j].set(c * gj + s * gj1)
            g = g.at[j + 1].set(-jnp.conj(s) * gj + c * gj1)

            return (V, H, cs, sn, g)

        V, H, cs, sn, g = lax.fori_loop(0, m, arnoldi_step, (V, H, cs, sn, g))

        R = H[:m, :m]
        y = _back_substitute(R, g[:m])

        dx = jnp.einsum("i,in->n", y, V[:m])
        x_new = x + dx
        res_norm_new = jnp.abs(g[m])

        return State(
            x=x_new,
            res_norm=res_norm_new,
            iters_total=state.iters_total + jnp.array(m, dtype=jnp.int32),
            restarts_used=state.restarts_used + jnp.array(1, dtype=jnp.int32),
        )

    def cond_fun(state: State):
        return jnp.logical_and(
            jnp.logical_not(is_converged(state.res_norm)),
            state.restarts_used < jnp.array(max_restarts, dtype=jnp.int32),
        )

    final_state = lax.while_loop(cond_fun, one_restart, init_state)

    info = GMRESInfo(
        converged=is_converged(final_state.res_norm),
        iters_total=final_state.iters_total,
        restarts_used=final_state.restarts_used,
        residual_norm=final_state.res_norm,
        b_norm=b_norm,
    )
    return final_state.x, info


# -----------------------------
# Example: trapezoid discretization of xdot = a*x
# Build A^T matvec via scan + VJP, then solve A^T lambda = g with GMRES
# -----------------------------

def simulate_linear(a: float, x0: float, h: float, N: int):
    # exact solution samples (for demo)
    t = h * jnp.arange(N + 1)
    return x0 * jnp.exp(a * t)  # (N+1,)

def step_residual_trap(xk, xk1, a, h):
    # trapezoid residual: x_{k+1} - x_k - 0.5*h*(a x_k + a x_{k+1})
    return xk1 - xk - 0.5 * h * (a * xk + a * xk1)  # scalar

def matvec_AT_steps(x_traj, a, h, lam_steps):
    """
    Applies A_steps^T to lam_steps, where A_steps is Jacobian of trapezoid residuals only.
    - x_traj: (N+1,) trajectory points
    - lam_steps: (N,) one per step residual
    Returns y: (N+1,)
    """
    N = lam_steps.shape[0]

    def one_step(carry, inputs):
        xk, xk1, lamk = inputs

        def rk(xk_, xk1_):
            return step_residual_trap(xk_, xk1_, a, h)

        (_, pullback) = jax.vjp(rk, xk, xk1)
        dk, nk = pullback(lamk)  # dk contributes to y_k, nk to y_{k+1}

        yk = carry + dk
        carry_next = nk
        return carry_next, yk

    c0 = jnp.array(0.0, dtype=x_traj.dtype)
    inputs = (x_traj[:-1], x_traj[1:], lam_steps)
    cN, y_head = lax.scan(one_step, c0, inputs)
    yN = cN[None]
    return jnp.concatenate([y_head, yN], axis=0)  # (N+1,)

def matvec_AT_full(x_traj, a, h, lam_full):
    """
    Full A^T operator includes an initial condition residual r_init = x0 - x0_given
    so the system is square:
      residuals: [r_init, r_0, ..., r_{N-1}]  => length N+1
      vars: [x_0, ..., x_N]                   => length N+1
    """
    lam_init = lam_full[0]
    lam_steps = lam_full[1:]  # length N
    y = matvec_AT_steps(x_traj, a, h, lam_steps)
    # add init residual contribution to y0: (dr_init/dx0)^T * lam_init = 1 * lam_init
    y = y.at[0].add(lam_init)
    return y


# -----------------------------
# Putting it all together (JIT)
# -----------------------------

def build_solve_adj(m: int, max_restarts: int):
    # JIT compile a single solve with static m/max_restarts
    @jax.jit
    def solve_adj(x_traj, a, h, g, lam0, tol):
        # closure matvec for GMRES
        def matvec(lam):
            return matvec_AT_full(x_traj, a, h, lam)

        lam, info = gmres_restarted(
            matvec, g, lam0,
            m=m, max_restarts=max_restarts, tol=tol
        )
        return lam, info
    return solve_adj


if __name__ == "__main__":
    import time

    # Problem setup
    a = -2.0
    h = 0.05
    N = 2000  # Larger problem to see Krylov advantages
    x0 = 1.0

    x_traj = simulate_linear(a, x0, h, N)  # (N+1,)

    # Suppose Phi(w) = 0.5 * sum_k x_k^2  => grad wrt x is g = x
    g = x_traj  # (N+1,)

    # Solve A^T lam = g
    m = 30
    max_restarts = 10
    tol = 1e-10

    print("=" * 60)
    print("CLASSICAL DIRECT SOLVER (numpy.linalg.solve)")
    print("=" * 60)

    # Build the full matrix A^T explicitly
    def build_AT_matrix(x_traj, a, h):
        n = x_traj.shape[0]
        A_T = jnp.zeros((n, n))

        # The forward Jacobian A has structure:
        # - Row 0: residual r_init = x0 - x0_given, so dr_init/dx = [1, 0, 0, ...]
        # - Row k (k=1..n-1): residual r_{k-1} from step k-1
        #   r_{k-1} = x_k - x_{k-1} - 0.5*h*(a*x_{k-1} + a*x_k)
        #   dr_{k-1}/dx_{k-1} = -1 - 0.5*h*a
        #   dr_{k-1}/dx_k = 1 - 0.5*h*a

        # A^T is the transpose, so:
        # - Column 0 of A^T = Row 0 of A = [1, 0, 0, ...]
        # - For step residuals, we transpose the bidiagonal structure

        # Initial condition contribution: A^T[0,0] = 1
        A_T = A_T.at[0, 0].set(1.0)

        # Step residuals: A has r_{k-1} in row k with derivatives wrt x_{k-1} and x_k
        # So A[k, k-1] = -1 - 0.5*h*a and A[k, k] = 1 - 0.5*h*a
        # Therefore A^T[k-1, k] = -1 - 0.5*h*a and A^T[k, k] = 1 - 0.5*h*a
        dr_dxk_prev = -1.0 - 0.5 * h * a  # derivative wrt x_{k-1}
        dr_dxk = 1.0 - 0.5 * h * a         # derivative wrt x_k

        for k in range(1, n):
            # A^T gets contributions from residual at step k-1
            A_T = A_T.at[k - 1, k].set(dr_dxk_prev)
            A_T = A_T.at[k, k].add(dr_dxk)

        return A_T

    A_T = build_AT_matrix(x_traj, a, h)

    # Convert to numpy for direct solve
    import numpy as np
    A_T_np = np.array(A_T)
    g_np = np.array(g)

    # Time the classical solve
    start = time.perf_counter()
    lam_classical = np.linalg.solve(A_T_np, g_np)
    classical_time = time.perf_counter() - start

    # Check residual
    res_classical = A_T_np @ lam_classical - g_np
    residual_norm_classical = np.linalg.norm(res_classical)

    print(f"Time: {classical_time * 1000:.4f} ms")
    print(f"||A^T lam - g||: {residual_norm_classical:.6e}")
    print(f"||g||: {np.linalg.norm(g_np):.6e}")
    print()

    print("=" * 60)
    print("KRYLOV SOLVER (GMRES)")
    print("=" * 60)

    solve_adj = build_solve_adj(m=m, max_restarts=max_restarts)
    lam0 = jnp.zeros((N + 1,), dtype=x_traj.dtype)

    # Warm up JIT compilation (don't count this time)
    print("Warming up JIT compilation...")
    _ = solve_adj(x_traj, a, h, g, lam0, tol)
    print("JIT compilation complete.")
    print()

    # Time the Krylov solve (after JIT compilation)
    start = time.perf_counter()
    lam_krylov, info = solve_adj(x_traj, a, h, g, lam0, tol)
    krylov_time = time.perf_counter() - start

    # Check residual norm ||A^T lam - g||
    res_krylov = matvec_AT_full(x_traj, a, h, lam_krylov) - g
    residual_norm_krylov = float(_safe_norm(res_krylov))

    print(f"Converged: {bool(info.converged)}")
    print(f"Restarts used: {int(info.restarts_used)}")
    print(f"Iterations total: {int(info.iters_total)}")
    print(f"Time: {krylov_time * 1000:.4f} ms")
    print(f"||A^T lam - g||: {residual_norm_krylov:.6e}")
    print(f"||g||: {float(_safe_norm(g)):.6e}")
    print()

    print("=" * 60)
    print("MATRIX VERIFICATION")
    print("=" * 60)

    # Verify that both methods use the same matrix by comparing matvecs
    test_vec = jnp.ones((N + 1,))

    # Matvec with explicit matrix
    matvec_explicit = A_T_np @ np.array(test_vec)

    # Matvec with implicit VJP-based operator
    matvec_implicit = np.array(matvec_AT_full(x_traj, a, h, test_vec))

    matvec_diff = np.linalg.norm(matvec_explicit - matvec_implicit)
    print(f"||A_explicit * 1 - A_implicit * 1||: {matvec_diff:.6e}")

    if matvec_diff > 1e-10:
        print("WARNING: Matrix construction differs between methods!")
        print(f"Explicit matrix first 5x5 block:\n{A_T_np[:5, :5]}")
        print(f"\nImplicit matvec result (first 5): {matvec_implicit[:5]}")
        print(f"Explicit matvec result (first 5): {matvec_explicit[:5]}")
    else:
        print("Matrix construction is consistent between methods.")
    print()

    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # Compare solutions
    lam_krylov_np = np.array(lam_krylov)
    solution_diff = np.linalg.norm(lam_krylov_np - lam_classical)

    print(f"||lam_krylov - lam_classical||: {solution_diff:.6e}")
    print(f"Speedup: {classical_time / krylov_time:.2f}x")
    print(f"Classical time: {classical_time * 1000:.4f} ms")
    print(f"Krylov time:    {krylov_time * 1000:.4f} ms")

    # Check if Krylov is solving a different problem
    print()
    print("Solution quality check:")
    print(f"Classical: ||A^T lam - g|| / ||g|| = {residual_norm_classical / np.linalg.norm(g_np):.6e}")
    print(f"Krylov:    ||A^T lam - g|| / ||g|| = {residual_norm_krylov / float(_safe_norm(g)):.6e}")
    print()
    print("=" * 60)
    print("NOTES")
    print("=" * 60)
    print(f"Problem size: N = {N} (matrix size {N+1} x {N+1})")
    print()
    print("When to use each method:")
    print("- DIRECT: Small-medium problems (< 10k), dense matrices, need exact solution")
    print("- KRYLOV: Large problems (> 100k), matrix-free operators, only need matvec")
    print()
    print("For this bidiagonal DAE system:")
    print("- Direct solver uses optimized LAPACK routines (very fast)")
    print("- Krylov avoids forming the matrix but has JAX overhead")
    print("- Krylov advantage: matrix-free, works when explicit A is unavailable")
    print("- Krylov becomes faster for very large systems or when matvec is cheap")
