"""
Snapshot Discrete Adjoint Gradient (SDAG) Method for DAE Parameter Optimization

Implements Algorithm from docs/algorithm_2.tex using JAX for:
- Automatic differentiation of residuals
- JIT compilation for performance
- Matrix-free Krylov methods for adjoint solve

Key features:
1. Resets trajectory at each iteration using IDA solver
2. Solves discrete adjoint system using GMRES
3. Computes reduced gradient efficiently
4. Full JAX implementation with JIT compilation
"""

import jax
import jax.numpy as jnp
from jax import grad, jacobian, jit, vmap, custom_vjp
from jax.scipy.sparse.linalg import gmres
import numpy as np
from typing import Callable, Dict, Tuple, Optional
from functools import partial


class SDAGOptimizer:
    """
    Snapshot Discrete Adjoint Gradient optimizer for DAE parameter identification.

    This implements Algorithm 2 (SDAG) from the documentation.
    """

    def __init__(
        self,
        dae_solver,  # DAESolver instance
        target_trajectory: Dict[str, np.ndarray],
        discretization: str = "trapezoidal",
        alpha: float = 0.01,
        gmres_tol: float = 1e-6,
        gmres_maxiter: int = 100,
    ):
        """
        Initialize SDAG optimizer.

        Args:
            dae_solver: DAESolver instance with eval_f and eval_g methods
            target_trajectory: Dictionary with 't', 'x', 'z' containing ground truth
            discretization: "trapezoidal" or "hermite_simpson"
            alpha: Step size for parameter updates
            gmres_tol: Tolerance for GMRES adjoint solve
            gmres_maxiter: Maximum GMRES iterations
        """
        self.dae_solver = dae_solver
        self.target_trajectory = target_trajectory
        self.discretization = discretization
        self.alpha = alpha
        self.gmres_tol = gmres_tol
        self.gmres_maxiter = gmres_maxiter

        # Extract dimensions
        self.n_states = len(dae_solver.state_names)
        self.n_alg = len(dae_solver.alg_names)
        self.n_params = len(dae_solver.param_names)
        self.n_time = len(target_trajectory['t'])

        # Convert target to JAX arrays
        # Keep t_target as list of Python floats to avoid tracing issues
        self.t_target = [float(t) for t in target_trajectory['t']]
        self.x_target = jnp.array(target_trajectory['x'])
        self.z_target = jnp.array(target_trajectory['z'])

        # Flatten target trajectory: w = [x, z] stacked over time
        self.w_target = self._stack_trajectory(self.x_target, self.z_target)

        print(f"SDAG Optimizer initialized:")
        print(f"  States: {self.n_states}, Algebraic: {self.n_alg}, Parameters: {self.n_params}")
        print(f"  Time points: {self.n_time}")
        print(f"  Discretization: {self.discretization}")
        print(f"  GMRES tolerance: {self.gmres_tol}")

        # Compile JAX versions of DAE equations
        self._compile_jax_equations()

        # Build JAX-compiled residual and gradient functions
        self._build_jax_functions()

    def _stack_trajectory(self, x: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """
        Stack state and algebraic trajectories into single vector.

        w[i*(n_x + n_z) : (i+1)*(n_x + n_z)] = [x[:, i], z[:, i]]
        """
        n_w = self.n_states + self.n_alg
        w = jnp.zeros(self.n_time * n_w)

        for i in range(self.n_time):
            idx_start = i * n_w
            w = w.at[idx_start:idx_start + self.n_states].set(x[:, i])
            w = w.at[idx_start + self.n_states:idx_start + n_w].set(z[:, i])

        return w

    def _unstack_trajectory(self, w: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Unstack w into (x, z) trajectories."""
        n_w = self.n_states + self.n_alg
        x = jnp.zeros((self.n_states, self.n_time))
        z = jnp.zeros((self.n_alg, self.n_time))

        for i in range(self.n_time):
            idx_start = i * n_w
            x = x.at[:, i].set(w[idx_start:idx_start + self.n_states])
            z = z.at[:, i].set(w[idx_start + self.n_states:idx_start + n_w])

        return x, z

    def _compile_jax_equations(self):
        """
        Compile DAE equations as JAX-differentiable functions.

        This creates JAX versions of eval_f and eval_g by replacing NumPy
        operations with JAX operations in the equation expressions.
        """
        # Create namespace with JAX math functions
        jax_namespace = {
            'sin': jnp.sin,
            'cos': jnp.cos,
            'tan': jnp.tan,
            'sinh': jnp.sinh,
            'cosh': jnp.cosh,
            'tanh': jnp.tanh,
            'exp': jnp.exp,
            'log': jnp.log,
            'log10': jnp.log10,
            'sqrt': jnp.sqrt,
            'abs': jnp.abs,
            'pow': jnp.power,
            'min': jnp.minimum,
            'max': jnp.maximum,
        }

        # Store equation expressions
        self.f_exprs = self.dae_solver.f_funcs
        self.g_exprs = self.dae_solver.g_funcs
        self.jax_namespace = jax_namespace

        print(f"  JAX-compiled {len(self.f_exprs)} f equations and {len(self.g_exprs)} g equations")

    def _eval_f_jax(self, t: float, x: jnp.ndarray, z: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """
        JAX-compatible evaluation of f(t, x, z, theta).

        Evaluates the DAE right-hand side using JAX operations for automatic differentiation.
        """
        # Create namespace for this evaluation
        ns = self.jax_namespace.copy()
        ns['time'] = t
        ns['t'] = t

        # Add states
        for i, name in enumerate(self.dae_solver.state_names):
            ns[name] = x[i]

        # Add algebraic variables
        for i, name in enumerate(self.dae_solver.alg_names):
            ns[name] = z[i]

        # Add parameters
        for i, name in enumerate(self.dae_solver.param_names):
            ns[name] = theta[i]

        # Evaluate f equations
        f = jnp.zeros(self.n_states)
        for i, expr in enumerate(self.f_exprs):
            f = f.at[i].set(eval(expr, ns))

        return f

    def _eval_g_jax(self, t: float, x: jnp.ndarray, z: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """
        JAX-compatible evaluation of g(t, x, z, theta).

        Evaluates the algebraic constraints using JAX operations for automatic differentiation.
        """
        # Create namespace for this evaluation
        ns = self.jax_namespace.copy()
        ns['time'] = t
        ns['t'] = t

        # Add states
        for i, name in enumerate(self.dae_solver.state_names):
            ns[name] = x[i]

        # Add algebraic variables
        for i, name in enumerate(self.dae_solver.alg_names):
            ns[name] = z[i]

        # Add parameters
        for i, name in enumerate(self.dae_solver.param_names):
            ns[name] = theta[i]

        # Evaluate g equations
        g = jnp.zeros(self.n_alg)
        for i, expr in enumerate(self.g_exprs):
            g = g.at[i].set(eval(expr, ns))

        return g

    def _build_jax_functions(self):
        """Build JAX functions for gradients (JIT applied only to gradient ops)."""

        # Loss function: MSE between trajectory and target
        # JIT this since it's pure JAX computation
        @jit
        def loss_fn(w):
            """Compute MSE loss over full trajectory."""
            return 0.5 * jnp.sum((w - self.w_target) ** 2)

        self.loss_fn = loss_fn
        self.grad_loss = jit(grad(loss_fn))

        print("JAX gradient functions compiled (JIT)")

    def compute_residual(self, w: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Compute full residual vector R_h(w, theta).

        Evaluates how well trajectory w satisfies the discretized DAE equations
        with parameters theta.

        Returns residual of shape n_time * (n_states + n_alg) = same as w

        Residuals include:
        - Initial condition constraints (w_0 - w_0^ref = 0)
        - Trapezoidal discretization for interior points
        - Algebraic constraints at all interior/final points

        This function calls the DAE solver only to evaluate f and g,
        NOT to solve the DAE. The trajectory w is given.

        JAX can automatically differentiate through this function.
        """
        x, z = self._unstack_trajectory(w)

        n_w = self.n_states + self.n_alg
        n_steps = self.n_time - 1
        R_list = []

        # Initial condition residual: w_0 - w_0^ref
        # Get reference initial conditions from DAE solver
        x0_ref = jnp.array(self.dae_solver.x0)
        z0_ref = jnp.array(self.dae_solver.z0)
        r_ic = jnp.concatenate([x[:, 0] - x0_ref, z[:, 0] - z0_ref])
        R_list.append(r_ic)

        for i in range(n_steps):
            # Trapezoidal residual for one time step
            # t_target is a Python list (constant), safe to index
            t_n = float(self.t_target[i])
            t_np1 = float(self.t_target[i+1])
            h = t_np1 - t_n

            x_n = x[:, i]
            z_n = z[:, i]
            x_np1 = x[:, i+1]
            z_np1 = z[:, i+1]

            # Use JAX-compatible evaluation (no NumPy conversions)
            # JAX can automatically differentiate through these calls
            f_n = self._eval_f_jax(t_n, x_n, z_n, theta)
            f_np1 = self._eval_f_jax(t_np1, x_np1, z_np1, theta)
            g_np1 = self._eval_g_jax(t_np1, x_np1, z_np1, theta)

            # Differential residual: x_{n+1} - x_n - h/2 * (f_n + f_{n+1})
            r_diff = x_np1 - x_n - 0.5 * h * (f_n + f_np1)

            # Algebraic residual: g_{n+1}
            r_alg = g_np1

            r_step = jnp.concatenate([r_diff, r_alg])
            R_list.append(r_step)

        # Stack all residuals
        return jnp.concatenate(R_list)

    def solve_forward(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Reset step: Solve DAE with current parameters using IDA.

        Returns: w^k such that R_h(w^k, theta^k) ≈ 0
        """
        # Convert theta to numpy
        theta_np = np.array(theta)

        # Set parameters in solver
        self.dae_solver.p = theta_np

        # Solve DAE using IDA at target time points
        # We need to solve and interpolate to match target times exactly
        t_span = (float(self.t_target[0]), float(self.t_target[-1]))

        # Solve with high resolution, then interpolate
        result = self.dae_solver.solve(
            t_span=t_span,
            ncp=self.n_time,
            rtol=1e-4,
            atol=1e-4
        )

        # Check if we got the right number of time points
        if len(result['t']) != self.n_time:
            print(f"  Warning: IDA returned {len(result['t'])} points instead of {self.n_time}")
            print(f"  Interpolating to target times...")

            # Interpolate to target times
            from scipy.interpolate import interp1d

            x_interp = interp1d(result['t'], result['x'], axis=1, kind='cubic', fill_value='extrapolate')
            z_interp = interp1d(result['t'], result['z'], axis=1, kind='cubic', fill_value='extrapolate')

            x = x_interp(self.t_target)
            z = z_interp(self.t_target)
        else:
            x = result['x']
            z = result['z']

        # Convert to JAX arrays and stack trajectory
        x = jnp.array(x)
        z = jnp.array(z)
        w = self._stack_trajectory(x, z)

        return w

    def solve_adjoint_direct(self, w: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Solve discrete adjoint system: A^T λ = ∇_w Φ

        where A = ∂R_h/∂w is the Jacobian of residuals w.r.t. trajectory.

        Uses direct solve by explicitly forming the Jacobian matrix.
        This is slower than iterative methods but more reliable for debugging.

        With initial condition constraints included, A is square (800 x 800).
        """
        # Right-hand side: gradient of loss w.r.t. trajectory
        g = self.grad_loss(w)

        print(f"   Forming Jacobian A = ∂R/∂w using JAX...")
        print(f"   RHS norm: {float(jnp.linalg.norm(g)):.6e}")

        # Compute Jacobian using JAX
        A = jax.jacobian(lambda w_var: self.compute_residual(w_var, theta))(w)

        print(f"   Jacobian shape: {A.shape}")
        print(f"   Solving A^T λ = g using direct solver...")

        # Solve A^T λ = g
        lam = jnp.linalg.solve(A.T, g)

        # Check residual
        residual = A.T @ lam - g
        res_norm = float(jnp.linalg.norm(residual))
        print(f"   Adjoint solve residual: {res_norm:.6e}")

        return lam

    def compute_reduced_gradient(self, w: jnp.ndarray, theta: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
        """
        Compute reduced gradient: g = ∂Φ/∂θ - (∂R/∂θ)^T λ

        First term is usually zero for trajectory-only loss.
        Second term is the adjoint contribution.
        """
        # ∂R/∂θ: Jacobian of residual w.r.t. parameters
        def residual_wrt_theta(theta_var):
            return self.compute_residual(w, theta_var)

        # Compute VJP: (∂R/∂θ)^T λ
        _, vjp_fn = jax.vjp(residual_wrt_theta, theta)
        dR_dtheta_T_lam = vjp_fn(lam)[0]

        # For MSE loss on trajectory only, ∂Φ/∂θ = 0
        # (loss doesn't directly depend on θ, only through w)
        phi_theta = jnp.zeros_like(theta)

        # Reduced gradient
        g = phi_theta - dR_dtheta_T_lam

        return g

    def step(self, theta: jnp.ndarray, verbose: bool = True) -> Tuple[jnp.ndarray, Dict]:
        """
        Execute one SDAG optimization step.

        Returns:
            theta_new: Updated parameters
            info: Dictionary with step information
        """
        if verbose:
            print("\n" + "="*60)
            print("SDAG Step")
            print("="*60)

        # Step 1: Reset / Forward solve
        if verbose:
            print("1. Forward solve (reset trajectory)...")
        w = self.solve_forward(theta)

        # Check residual
        R = self.compute_residual(w, theta)
        res_norm = float(jnp.linalg.norm(R))
        if verbose:
            print(f"   Residual norm: {res_norm:.6e}")

        # Compute loss
        loss = float(self.loss_fn(w))
        if verbose:
            print(f"   Loss: {loss:.6e}")

        # Step 2: Adjoint solve
        if verbose:
            print("2. Solving adjoint system (direct)...")
        lam = self.solve_adjoint_direct(w, theta)

        # Check adjoint residual
        g_rhs = self.grad_loss(w)
        def matvec_AT(v):
            _, vjp_fn = jax.vjp(lambda w_var: self.compute_residual(w_var, theta), w)
            return vjp_fn(v)[0]
        adj_res = g_rhs - matvec_AT(lam)
        adj_res_norm = float(jnp.linalg.norm(adj_res))
        if verbose:
            print(f"   Adjoint residual: {adj_res_norm:.6e}")

        # Step 3: Compute reduced gradient
        if verbose:
            print("3. Computing reduced gradient...")
        g_theta = self.compute_reduced_gradient(w, theta, lam)
        grad_norm = float(jnp.linalg.norm(g_theta))
        if verbose:
            print(f"   Gradient norm: {grad_norm:.6e}")

        # Step 4: Parameter update
        theta_new = theta - self.alpha * g_theta
        if verbose:
            print(f"4. Parameter update (α={self.alpha})")
            print(f"   θ_new - θ_old norm: {float(jnp.linalg.norm(theta_new - theta)):.6e}")

        info = {
            'loss': loss,
            'residual_norm': res_norm,
            'adjoint_residual': adj_res_norm,
            'gradient_norm': grad_norm,
            'theta': np.array(theta_new),
        }

        return theta_new, info

    def optimize(self, theta_init: np.ndarray, n_iterations: int = 10, verbose: bool = True) -> Dict:
        """
        Run SDAG optimization for multiple iterations.

        Args:
            theta_init: Initial parameter vector
            n_iterations: Number of optimization iterations
            verbose: Print progress

        Returns:
            results: Dictionary with optimization history
        """
        theta = jnp.array(theta_init)

        history = {
            'loss': [],
            'residual_norm': [],
            'adjoint_residual': [],
            'gradient_norm': [],
            'theta': [np.array(theta)],
        }

        print("\n" + "="*80)
        print("SDAG Optimization")
        print("="*80)
        print(f"Initial parameters: {theta}")
        print(f"Target parameters:  {self.dae_solver.p}")
        print(f"Iterations: {n_iterations}")
        print("="*80)

        for k in range(n_iterations):
            print(f"\nIteration {k+1}/{n_iterations}")

            theta, info = self.step(theta, verbose=verbose)

            # Store history
            history['loss'].append(info['loss'])
            history['residual_norm'].append(info['residual_norm'])
            history['adjoint_residual'].append(info['adjoint_residual'])
            history['gradient_norm'].append(info['gradient_norm'])
            history['theta'].append(info['theta'])

            # Check convergence
            if info['gradient_norm'] < 1e-6:
                print("\nConverged!")
                break

        history['theta_final'] = np.array(theta)

        print("\n" + "="*80)
        print("Optimization Complete")
        print("="*80)
        print(f"Final parameters: {history['theta_final']}")
        print(f"Final loss: {history['loss'][-1]:.6e}")
        print(f"Final gradient norm: {history['gradient_norm'][-1]:.6e}")

        return history
