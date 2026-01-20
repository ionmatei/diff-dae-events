"""
Augmented Lagrangian DAE Optimizer V5 - Self-Sufficient Implementation

Implements the Augmented Lagrangian method for DAE parameter estimation
as described in algorithm_3.tex (Option C).

This file is self-contained and does not depend on other project modules
(except for DAESolver which requires scikits.odes).

Key components:
1. Augmented Lagrangian function L_mu(w, theta, eta)
2. Gradient of L_mu with respect to w (state trajectory)
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, jacfwd, jacrev
from jax.scipy.linalg import lu_factor, lu_solve
from jax import lax
import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
from functools import partial
import time
import re
import os

# ============================================================================
# From src/deer/maths.py - Parallel Scan Functions
# ============================================================================

def scan_binop(element_i: Tuple[jnp.ndarray, jnp.ndarray],
               element_j: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Associative operator for the scan."""
    gti, hti = element_i
    gtj, htj = element_j
    a = gtj @ gti
    b = jnp.einsum("...ij,...j->...i", gtj, hti) + htj
    return a, b


def _interleave(a, b, axis):
    """Given two Tensors of static shape, interleave them along the first axis."""
    from jax._src.lax import lax as lax_internal
    assert a.shape[axis] == b.shape[axis] or a.shape[axis] == b.shape[axis] + 1
    a_pad = [(0, 0, 0)] * a.ndim
    b_pad = [(0, 0, 0)] * b.ndim
    a_pad[axis] = (0, 1 if a.shape[axis] == b.shape[axis] else 0, 1)
    b_pad[axis] = (1, 0 if a.shape[axis] == b.shape[axis] else 1, 1)
    op = jax.lax.bitwise_or if a.dtype == jnp.bool_ else jax.lax.add
    return op(jax.lax.pad(a, lax_internal._const(a, 0), a_pad),
              jax.lax.pad(b, lax_internal._const(b, 0), b_pad))


def associative_scan(fn: Callable, elems, reverse: bool = False, axis: int = 0):
    """
    Associative scan from jax's source code, with fix for slice_in_dim bug.
    See https://github.com/google/jax/issues/21637
    """
    from jax._src import util, core

    if not callable(fn):
        raise TypeError("lax.associative_scan: fn argument should be callable.")
    elems_flat, tree = jax.tree_util.tree_flatten(elems)

    if reverse:
        elems_flat = [jax.lax.rev(elem, [axis]) for elem in elems_flat]

    def combine(a_flat, b_flat):
        a = jax.tree_util.tree_unflatten(tree, a_flat)
        b = jax.tree_util.tree_unflatten(tree, b_flat)
        c = fn(a, b)
        c_flat, _ = jax.tree_util.tree_flatten(c)
        return c_flat

    axis = util.canonicalize_axis(axis, elems_flat[0].ndim)

    if not core.is_constant_dim(elems_flat[0].shape[axis]):
        raise NotImplementedError("associative scan over axis "
            f"of non-constant size: {elems_flat[0].shape[axis]}.")
    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError('Array inputs to associative_scan must have the same '
                         'first dimension. (saw: {})'
                         .format([elem.shape for elem in elems_flat]))

    def get_idxs(elem, slc):
        lst = [slice(None, None, None) for _ in range(len(elem.shape))]
        lst[axis] = slc
        return tuple(lst)

    def _scan(elems):
        """Perform scan on `elems`."""
        num_elems = elems[0].shape[axis]

        if num_elems < 2:
            return elems

        reduced_elems = combine(
            [elem[get_idxs(elem, slice(0, -1, 2))] for elem in elems],
            [elem[get_idxs(elem, slice(1, None, 2))] for elem in elems])

        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = combine(
                [e[get_idxs(e, slice(0, -1, None))] for e in odd_elems],
                [e[get_idxs(e, slice(2, None, 2))] for e in elems])
        else:
            even_elems = combine(
                odd_elems,
                [e[get_idxs(e, slice(2, None, 2))] for e in elems])

        even_elems = [
            jax.lax.concatenate([elem[get_idxs(elem, slice(0, 1, None))], result],
                                dimension=axis)
            for (elem, result) in zip(elems, even_elems)]
        return list(util.safe_map(partial(_interleave, axis=axis), even_elems, odd_elems))

    scans = _scan(elems_flat)

    if reverse:
        scans = [jax.lax.rev(scanned, [axis]) for scanned in scans]

    return jax.tree_util.tree_unflatten(tree, scans)


def matmul_recursive(mats: jnp.ndarray, vecs: jnp.ndarray, y0: jnp.ndarray) -> jnp.ndarray:
    """
    Perform the matrix multiplication recursively, y[i + 1] = mats[i] @ y[i] + vec[i].

    Arguments
    ---------
    mats: jnp.ndarray
        The matrices to be multiplied, shape ``(nsamples - 1, ny, ny)``
    vecs: jnp.ndarray
        The vector to be multiplied, shape ``(nsamples - 1, ny)``
    y0: jnp.ndarray
        The initial condition, shape ``(ny,)``

    Returns
    -------
    result: jnp.ndarray
        The result of the matrix multiplication, shape ``(nsamples, ny)``, including ``y0`` at the beginning.
    """
    eye = jnp.eye(mats.shape[-1], dtype=mats.dtype)[None]
    first_elem = jnp.concatenate((eye, mats), axis=0)
    second_elem = jnp.concatenate((y0[None], vecs), axis=0)

    elems = (first_elem, second_elem)
    _, yt = associative_scan(scan_binop, elems)
    return yt


# ============================================================================
# BDF Coefficients
# ============================================================================

BDF_COEFFICIENTS = {
    1: ([1.0, -1.0], 1.0),
    2: ([3.0/2.0, -2.0, 1.0/2.0], 1.0),
    3: ([11.0/6.0, -3.0, 3.0/2.0, -1.0/3.0], 1.0),
    4: ([25.0/12.0, -4.0, 3.0, -4.0/3.0, 1.0/4.0], 1.0),
    5: ([137.0/60.0, -5.0, 5.0, -10.0/3.0, 5.0/4.0, -1.0/5.0], 1.0),
    6: ([147.0/60.0, -6.0, 15.0/2.0, -20.0/3.0, 15.0/4.0, -6.0/5.0, 1.0/6.0], 1.0),
}

VALID_METHODS = ['backward_euler', 'trapezoidal', 'bdf2', 'bdf3', 'bdf4', 'bdf5', 'bdf6']


# ============================================================================
# From src/discrete_adjoint/dae_jacobian.py - DAEJacobian Class
# ============================================================================

class DAEJacobian:
    """
    Computes Jacobians of DAE residual functions using JAX.

    Supports multiple time discretization methods:
    - 'backward_euler': First-order implicit (A-stable, L-stable)
    - 'trapezoidal': Second-order implicit (A-stable), also known as Crank-Nicolson
    - 'bdf2': Second-order BDF (A-stable, L-stable)
    - 'bdf3': Third-order BDF (A-stable)
    - 'bdf4': Fourth-order BDF (A-stable)
    - 'bdf5': Fifth-order BDF (A-stable)
    - 'bdf6': Sixth-order BDF (A-stable, highest stable BDF)
    """

    def __init__(self, dae_data: dict, method: str = 'trapezoidal'):
        """
        Initialize from DAE specification.

        Args:
            dae_data: Dictionary containing DAE specification with keys:
                - states: list of differential state variables
                - alg_vars: list of algebraic variables
                - parameters: list of parameters
                - f: list of differential equations (dx/dt = f)
                - g: list of algebraic equations (0 = g)
            method: Time discretization method.
        """
        if method not in VALID_METHODS:
            raise ValueError(f"method must be one of {VALID_METHODS}, got '{method}'")
        self.method = method

        self.states = dae_data['states']
        self.alg_vars = dae_data['alg_vars']
        self.parameters = dae_data['parameters']

        self.f_eqs = dae_data['f']
        self.g_eqs = dae_data['g']
        self.h_eqs = dae_data.get('h', None)

        self.state_names = [s['name'] for s in self.states]
        self.alg_names = [a['name'] for a in self.alg_vars]
        self.param_names = [p['name'] for p in self.parameters]

        self.p = jnp.array([p['value'] for p in self.parameters])

        self.n_states = len(self.state_names)
        self.n_alg = len(self.alg_names)
        self.n_total = self.n_states + self.n_alg

        self.optimize_indices = None
        self.p_all_default = None

        self._compile_equations()
        self._compile_jacobian_functions()

    def set_selective_optimization(self, optimize_indices, p_all_default):
        """Configure selective parameter optimization."""
        self.optimize_indices = optimize_indices
        self.p_all_default = jnp.array(p_all_default)

    def _compile_equations(self):
        """Compile equation strings into Python functions."""
        self.f_funcs = []
        for eq in self.f_eqs:
            match = re.match(r'der\((\w+)\)\s*=\s*(.+)', eq)
            if match:
                _, rhs = match.groups()
                self.f_funcs.append(rhs)
            else:
                raise ValueError(f"Invalid f equation format: {eq}")

        self.g_funcs = []
        for eq in self.g_eqs:
            if '=' in eq:
                lhs, rhs = eq.split('=', 1)
                lhs, rhs = lhs.strip(), rhs.strip()

                if lhs == '0' or lhs == '0.0':
                    expr = rhs
                elif rhs == '0' or rhs == '0.0':
                    expr = lhs
                else:
                    expr = f"({lhs}) - ({rhs})"

                self.g_funcs.append(expr)
            else:
                self.g_funcs.append(eq)

        self.h_funcs = []
        if self.h_eqs:
            for eq in self.h_eqs:
                if '=' in eq:
                    _, rhs = eq.split('=', 1)
                    self.h_funcs.append(rhs.strip())
                else:
                    self.h_funcs.append(eq)
        else:
            self.h_funcs = self.state_names.copy()

    def _create_jax_eval_namespace_with_params(self, t, x, z, p, optimize_indices=None, p_all_default=None):
        """Create namespace for JAX equation evaluation with explicit parameters."""
        ns = {
            'sin': jnp.sin, 'cos': jnp.cos, 'tan': jnp.tan,
            'sinh': jnp.sinh, 'cosh': jnp.cosh, 'tanh': jnp.tanh,
            'exp': jnp.exp, 'log': jnp.log, 'log10': jnp.log10,
            'sqrt': jnp.sqrt, 'abs': jnp.abs, 'pow': jnp.power,
            'min': jnp.minimum, 'max': jnp.maximum,
            'time': t, 't': t,
        }

        for i, name in enumerate(self.state_names):
            ns[name] = x[i]

        for i, name in enumerate(self.alg_names):
            ns[name] = z[i]

        if optimize_indices is None:
            for i, name in enumerate(self.param_names):
                ns[name] = p[i]
        else:
            if p_all_default is None:
                raise ValueError("p_all_default must be provided when optimize_indices is specified")
            param_values = list(p_all_default)
            for opt_idx, param_idx in enumerate(optimize_indices):
                param_values[param_idx] = p[opt_idx]
            for i, name in enumerate(self.param_names):
                ns[name] = param_values[i]

        return ns

    def eval_f_jax(self, t, x, z, p):
        """Evaluate f(t, x, z, p) using JAX."""
        ns = self._create_jax_eval_namespace_with_params(
            t, x, z, p,
            optimize_indices=self.optimize_indices,
            p_all_default=self.p_all_default
        )
        dxdt_list = [eval(expr, ns) for expr in self.f_funcs]
        return jnp.array(dxdt_list)

    def eval_g_jax(self, t, x, z, p):
        """Evaluate g(t, x, z, p) using JAX."""
        ns = self._create_jax_eval_namespace_with_params(
            t, x, z, p,
            optimize_indices=self.optimize_indices,
            p_all_default=self.p_all_default
        )
        g_list = [eval(expr, ns) for expr in self.g_funcs]
        return jnp.array(g_list)

    def eval_h_jax(self, t, x, z, p):
        """Evaluate h(t, x, z, p) using JAX."""
        if not self.h_funcs:
            return x

        if self.h_funcs == self.state_names:
            return x

        ns = self._create_jax_eval_namespace_with_params(
            t, x, z, p,
            optimize_indices=self.optimize_indices,
            p_all_default=self.p_all_default
        )

        y_list = []
        for i, expr in enumerate(self.h_funcs):
            val = eval(expr, ns)
            y_list.append(val)

        return jnp.array(y_list)

    def residual_trapezoidal_single(self, t_k, t_kp1, y_k, y_kp1, p):
        """Trapezoidal residual for a single time interval [t_k, t_{k+1}]."""
        h = t_kp1 - t_k

        x_k = y_k[:self.n_states]
        z_k = y_k[self.n_states:]
        x_kp1 = y_kp1[:self.n_states]
        z_kp1 = y_kp1[self.n_states:]

        f_k = self.eval_f_jax(t_k, x_k, z_k, p)
        f_kp1 = self.eval_f_jax(t_kp1, x_kp1, z_kp1, p)
        g_kp1 = self.eval_g_jax(t_kp1, x_kp1, z_kp1, p)

        r_diff = (x_kp1 - x_k) / h - 0.5 * (f_k + f_kp1)
        r_alg = g_kp1

        return jnp.concatenate([r_diff, r_alg])

    def residual_backward_euler_single(self, t_k, t_kp1, y_k, y_kp1, p):
        """Backward Euler residual for a single time interval [t_k, t_{k+1}]."""
        h = t_kp1 - t_k

        x_k = y_k[:self.n_states]
        x_kp1 = y_kp1[:self.n_states]
        z_kp1 = y_kp1[self.n_states:]

        f_kp1 = self.eval_f_jax(t_kp1, x_kp1, z_kp1, p)
        g_kp1 = self.eval_g_jax(t_kp1, x_kp1, z_kp1, p)

        r_diff = (x_kp1 - x_k) / h - f_kp1
        r_alg = g_kp1

        return jnp.concatenate([r_diff, r_alg])

    def residual_single(self, t_k, t_kp1, y_k, y_kp1, p):
        """Compute residual using the selected method."""
        if self.method == 'backward_euler':
            return self.residual_backward_euler_single(t_k, t_kp1, y_k, y_kp1, p)
        elif self.method == 'trapezoidal':
            return self.residual_trapezoidal_single(t_k, t_kp1, y_k, y_kp1, p)
        elif self.method == 'bdf2':
            return self.residual_backward_euler_single(t_k, t_kp1, y_k, y_kp1, p)
        else:
            return self.residual_backward_euler_single(t_k, t_kp1, y_k, y_kp1, p)

    def _compile_jacobian_functions(self):
        """Compile vectorized Jacobian functions using JAX vmap."""
        if self.method == 'backward_euler':
            residual_fn = self.residual_backward_euler_single
        elif self.method == 'trapezoidal':
            residual_fn = self.residual_trapezoidal_single
        else:
            residual_fn = self.residual_single

        def jac_y_k_single(t_k, t_kp1, y_k, y_kp1, p):
            return jacfwd(lambda yk: residual_fn(t_k, t_kp1, yk, y_kp1, p))(y_k)

        def jac_y_kp1_single(t_k, t_kp1, y_k, y_kp1, p):
            return jacfwd(lambda ykp1: residual_fn(t_k, t_kp1, y_k, ykp1, p))(y_kp1)

        def jac_p_single(t_k, t_kp1, y_k, y_kp1, p):
            return jacfwd(lambda pp: residual_fn(t_k, t_kp1, y_k, y_kp1, pp))(p)

        self._jac_y_k_vmapped = vmap(jac_y_k_single, in_axes=(0, 0, 0, 0, None))
        self._jac_y_kp1_vmapped = vmap(jac_y_kp1_single, in_axes=(0, 0, 0, 0, None))
        self._jac_p_vmapped = vmap(jac_p_single, in_axes=(0, 0, 0, 0, None))

        self._jac_y_k_vmapped_jit = jit(self._jac_y_k_vmapped)
        self._jac_y_kp1_vmapped_jit = jit(self._jac_y_kp1_vmapped)
        self._jac_p_vmapped_jit = jit(self._jac_p_vmapped)

        self._compile_f_g_jacobian_functions()
        self._compile_loss_functions()

    def _compile_f_g_jacobian_functions(self):
        """Compile vectorized Jacobian functions for f and g separately."""
        def jac_f_single(t, y, p):
            return jacfwd(lambda yy: self.eval_f_jax(t, yy[:self.n_states], yy[self.n_states:], p))(y)

        def jac_g_single(t, y, p):
            return jacfwd(lambda yy: self.eval_g_jax(t, yy[:self.n_states], yy[self.n_states:], p))(y)

        self._jac_f_vmapped = vmap(jac_f_single, in_axes=(0, 0, None))
        self._jac_g_vmapped = vmap(jac_g_single, in_axes=(0, 0, None))

        self._jac_f_vmapped_jit = jit(self._jac_f_vmapped)
        self._jac_g_vmapped_jit = jit(self._jac_g_vmapped)

    def _compile_loss_functions(self):
        """Compile JIT versions of loss and gradient functions."""
        self.trajectory_loss_jit = jit(self.trajectory_loss)
        self.trajectory_loss_gradient_jit = jit(self.trajectory_loss_gradient)
        self.trajectory_loss_gradient_analytical_jit = jit(self.trajectory_loss_gradient_analytical)

    def compute_jacobian_blocks_jit(self, t_array: np.ndarray, y_array: np.ndarray, p: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Jacobian blocks for all time intervals using JIT-compiled functions."""
        if p is None:
            p = self.p
        p = jnp.array(p)

        if y_array.shape[0] == self.n_total and y_array.shape[1] == len(t_array):
            y_array = y_array.T

        N = len(t_array) - 1
        if N <= 0:
            return [], []

        t_k = jnp.array(t_array[:-1])
        t_kp1 = jnp.array(t_array[1:])
        y_k = jnp.array(y_array[:-1])
        y_kp1 = jnp.array(y_array[1:])

        J_prev = self._jac_y_k_vmapped_jit(t_k, t_kp1, y_k, y_kp1, p)
        J_curr = self._jac_y_kp1_vmapped_jit(t_k, t_kp1, y_k, y_kp1, p)

        J_prev_list = [np.array(J_prev[i]) for i in range(N)]
        J_curr_list = [np.array(J_curr[i]) for i in range(N)]

        return J_prev_list, J_curr_list

    def trajectory_loss(
        self,
        t_array: np.ndarray,
        y_array: np.ndarray,
        y_target_array: np.ndarray,
        p: np.ndarray = None
    ) -> float:
        """Compute trajectory tracking loss as sum of squared errors."""
        if p is None:
            p = self.p
        p = jnp.array(p)

        if y_array.shape[0] == self.n_total and y_array.shape[1] == len(t_array):
            y_array = y_array.T

        if self.h_funcs:
            n_outputs = len(self.h_funcs)
        else:
            n_outputs = self.n_states

        if y_target_array.shape[0] == n_outputs and y_target_array.shape[1] == len(t_array):
            y_target_array = y_target_array.T

        t_jax = jnp.array(t_array)
        y_jax = jnp.array(y_array)
        y_target_jax = jnp.array(y_target_array)

        def eval_h_single(t, y):
            x = y[:self.n_states]
            z = y[self.n_states:]
            return self.eval_h_jax(t, x, z, p)

        eval_h_vec = jax.vmap(eval_h_single, in_axes=(0, 0))
        y_pred = eval_h_vec(t_jax, y_jax)

        errors = y_pred - y_target_jax
        squared_errors = errors ** 2
        loss = jnp.sum(squared_errors)

        return loss

    def trajectory_loss_gradient(
        self,
        t_array: np.ndarray,
        y_array: np.ndarray,
        y_target_array: np.ndarray,
        p: np.ndarray = None
    ) -> np.ndarray:
        """Compute gradient of trajectory loss with respect to states y."""
        if p is None:
            p = self.p
        p = jnp.array(p)

        transposed_input = False
        if y_array.shape[0] == self.n_total and y_array.shape[1] == len(t_array):
            y_array = y_array.T
            transposed_input = True

        if self.h_funcs:
            n_outputs = len(self.h_funcs)
        else:
            n_outputs = self.n_states

        if y_target_array.shape[0] == n_outputs and y_target_array.shape[1] == len(t_array):
            y_target_array = y_target_array.T

        t_jax = jnp.array(t_array)
        y_jax = jnp.array(y_array)
        y_target_jax = jnp.array(y_target_array)

        def loss_fn(y):
            def eval_h_single(t, y_single):
                x = y_single[:self.n_states]
                z = y_single[self.n_states:]
                return self.eval_h_jax(t, x, z, p)

            eval_h_vec = jax.vmap(eval_h_single, in_axes=(0, 0))
            y_pred = eval_h_vec(t_jax, y)
            errors = y_pred - y_target_jax
            return jnp.sum(errors ** 2)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(y_jax)
        grad_np = np.array(grad)

        if transposed_input:
            grad_np = grad_np.T

        return grad_np

    def trajectory_loss_gradient_analytical(
        self,
        t_array: np.ndarray,
        y_array: np.ndarray,
        y_target_array: np.ndarray,
        p: np.ndarray = None
    ):
        """Compute gradient of trajectory loss analytically using chain rule."""
        if p is None:
            p = self.p
        p = jnp.array(p)

        t_jax = jnp.array(t_array)
        y_jax = jnp.array(y_array)
        y_target_jax = jnp.array(y_target_array)

        transposed_input = False
        if y_jax.shape[0] == self.n_total and y_jax.shape[1] == t_jax.shape[0]:
            y_jax = y_jax.T
            transposed_input = True

        if self.h_funcs:
            n_outputs = len(self.h_funcs)
        else:
            n_outputs = self.n_states

        if y_target_jax.shape[0] == n_outputs and y_target_jax.shape[1] == t_jax.shape[0]:
            y_target_jax = y_target_jax.T

        def eval_h_single(t, y):
            x = y[:self.n_states]
            z = y[self.n_states:]
            return self.eval_h_jax(t, x, z, p)

        eval_h_vec = jax.vmap(eval_h_single, in_axes=(0, 0))
        y_pred = eval_h_vec(t_jax, y_jax)
        errors = y_pred - y_target_jax

        def jac_h_single(t, y):
            return jax.jacfwd(lambda yy: self.eval_h_jax(t, yy[:self.n_states], yy[self.n_states:], p))(y)

        jac_h_vec = jax.vmap(jac_h_single, in_axes=(0, 0))
        dh_dy = jac_h_vec(t_jax, y_jax)

        def compute_grad_single(dh_dy_k, error_k):
            return 2.0 * dh_dy_k.T @ error_k

        grad_vec = jax.vmap(compute_grad_single, in_axes=(0, 0))
        grad = grad_vec(dh_dy, errors)

        if transposed_input:
            grad = grad.T

        return grad


# ============================================================================
# From src/discrete_adjoint/dae_jacobian.py - DAEOptimizer Class
# ============================================================================

class DAEOptimizer:
    """
    Iterative optimizer for DAE parameters using adjoint-based gradient descent.

    Minimizes a loss function by adjusting DAE parameters to match an output trajectory.
    Uses the adjoint method for efficient gradient computation.
    """

    def __init__(self, dae_data: dict, dae_solver=None, optimize_params: List[str] = None,
                 loss_type: str = 'sum', method: str = 'trapezoidal'):
        """
        Initialize the DAE optimizer.

        Args:
            dae_data: Dictionary containing DAE specification
            dae_solver: Optional DAESolver instance. If None, creates a new one.
            optimize_params: List of parameter names to optimize. If None, optimizes all parameters.
            loss_type: Type of loss function - 'sum' or 'mean'. Default is 'sum'.
            method: Time discretization method.
        """
        from src.discrete_adjoint.dae_solver import DAESolver

        self.dae_data = dae_data
        self.method = method
        self.jac = DAEJacobian(dae_data, method=method)

        if dae_solver is None:
            self.solver = DAESolver(dae_data)
        else:
            self.solver = dae_solver

        if loss_type not in ['sum', 'mean']:
            raise ValueError(f"loss_type must be 'sum' or 'mean', got '{loss_type}'")
        self.loss_type = loss_type

        self.n_params_total = len(self.jac.param_names)
        self.n_states = self.jac.n_states
        self.n_alg = self.jac.n_alg
        self.n_total = self.jac.n_total

        self.p_all = jnp.array([p['value'] for p in dae_data['parameters']])

        if optimize_params is None:
            self.optimize_params = self.jac.param_names.copy()
            self.optimize_indices = list(range(self.n_params_total))
        else:
            self.optimize_params = optimize_params
            self.optimize_indices = []
            for param_name in optimize_params:
                if param_name in self.jac.param_names:
                    idx = self.jac.param_names.index(param_name)
                    self.optimize_indices.append(idx)
                else:
                    print(f"Warning: Parameter '{param_name}' not found in DAE specification")

        self.n_params = len(self.optimize_indices)
        self.optimize_indices_jax = jnp.array(self.optimize_indices)

        self.param_mask = np.zeros(self.n_params_total, dtype=bool)
        self.param_mask[self.optimize_indices] = True

        self.p_current = jnp.array([self.p_all[i] for i in self.optimize_indices])

        self.jac.set_selective_optimization(self.optimize_indices, self.p_all)

        self.history = {
            'loss': [],
            'gradient_norm': [],
            'params': [],
            'params_all': [],
            'step_size': []
        }

        print(f"DAE Optimizer initialized")
        print(f"  Method: {self.method}")
        print(f"  Total parameters: {self.n_params_total}")
        print(f"  Parameters to optimize: {self.n_params}")
        print(f"  Optimized parameter names: {self.optimize_params}")
        print(f"  Fixed parameters: {[name for i, name in enumerate(self.jac.param_names) if i not in self.optimize_indices]}")
        print(f"  Differential states: {self.n_states}")
        print(f"  Algebraic variables: {self.n_alg}")
        print(f"  Loss type: {self.loss_type}")

    def compute_loss(self, y_pred: jnp.ndarray, y_target: jnp.ndarray) -> float:
        """Compute loss function (sum or mean of squared errors)."""
        errors = y_pred - y_target
        squared_errors = errors ** 2

        if self.loss_type == 'sum':
            loss = jnp.sum(squared_errors)
        else:
            loss = jnp.mean(squared_errors)

        return loss


# ============================================================================
# Main Augmented Lagrangian Optimizer Class
# ============================================================================

class DAEOptimizerAugmentedLagrangianV5(DAEOptimizer):
    """
    Augmented Lagrangian optimizer for DAE parameter identification.

    Minimizes:
        L_mu(w, theta, eta) = Phi(w, theta) + eta^T R(w, theta) + (mu/2) ||R(w, theta)||^2

    where:
        w: State trajectory (differential and algebraic variables)
        theta: Parameters
        eta: Lagrange multipliers for the DAE residuals
        Phi: Objective function (trajectory loss)
        R: DAE residuals (defect constraints)
    """

    def __init__(
        self,
        dae_data: dict,
        dae_solver=None,
        optimize_params: List[str] = None,
        loss_type: str = 'sum',
        method: str = 'trapezoidal',
        verbose: bool = True
    ):
        """Initialize optimizer."""
        super().__init__(
            dae_data=dae_data,
            dae_solver=dae_solver,
            optimize_params=optimize_params,
            loss_type=loss_type,
            method=method
        )

        self.verbose = verbose
        if self.verbose:
            print(f"Augmented Lagrangian V5 Initialized (Method: {self.method})")

        # IMPORTANT: DAEJacobian is configured by default (in super().__init__)
        # to expect reduced parameter vectors if optimize_params is set.
        # However, this class manages parameter reconstruction manually and passes
        # the FULL parameter vector to residuals.
        # We must disable selective optimization in self.jac so it accepts p_all correctly.
        self.jac.optimize_indices = None
        self.jac.p_all_default = None

        # Compile the AL and gradient functions
        self._compile_al_functions()

    def _compile_al_functions(self):
        """Compile JAX functions for AL computation."""

        # 1. Residual function (vectorized over time)
        self._residual_vmap = jit(vmap(self.jac.residual_single, in_axes=(0, 0, 0, 0, None)))

        # 2. Augmented Lagrangian function
        def augmented_lagrangian(
            w: jnp.ndarray,
            theta_vals: jnp.ndarray,
            eta: jnp.ndarray,
            mu: float,
            t_array: jnp.ndarray,
            y_target: jnp.ndarray
        ):
            """
            Compute Augmented Lagrangian value.

            Args:
                w: State trajectory, shape (N, n_total)
                theta_vals: Optimized parameter values
                eta: Multipliers, shape (N-1, n_total) corresponding to intervals
                mu: Penalty parameter
                t_array: Time points, shape (N,)
                y_target: Target outputs, shape (N, n_outputs)
            """
            # 1. Map optimized params to full param vector
            p_all = self.p_all.at[self.optimize_indices_jax].set(theta_vals)

            # 2. Compute Phi(w, theta) - Trajectory Loss
            if self.jac.h_funcs:
                outputs = vmap(self.jac.eval_h_jax, in_axes=(0, 0, 0, None))(
                    t_array,
                    w[:, :self.n_states],
                    w[:, self.n_states:],
                    p_all
                )
            else:
                outputs = w[:, :self.n_states]

            diff = outputs - y_target
            if self.loss_type == 'mean':
                phi = jnp.mean(diff**2)
            else:
                phi = jnp.sum(diff**2)

            # 3. Compute R(w, theta) - Residuals
            t_k = t_array[:-1]
            t_kp1 = t_array[1:]
            w_k = w[:-1]
            w_kp1 = w[1:]

            residuals = self._residual_vmap(t_k, t_kp1, w_k, w_kp1, p_all)

            # 4. Compute constraint terms
            eta_dot_R = jnp.sum(eta * residuals)
            R_norm_sq = jnp.sum(residuals**2)
            penalty = (mu / 2.0) * R_norm_sq

            return phi + eta_dot_R + penalty

        self._augmented_lagrangian_jit = jit(augmented_lagrangian)

        # 3. Gradient of AL w.r.t. w
        grad_fn = grad(augmented_lagrangian, argnums=0)

        def grad_w_augmented_lagrangian(
            w: jnp.ndarray,
            theta_vals: jnp.ndarray,
            eta: jnp.ndarray,
            mu: float,
            t_array: jnp.ndarray,
            y_target: jnp.ndarray
        ):
            grad_full = grad_fn(w, theta_vals, eta, mu, t_array, y_target)
            grad_w = grad_full.at[0].set(jnp.zeros_like(grad_full[0]))
            return grad_w

        self._grad_w_augmented_lagrangian_jit = jit(grad_w_augmented_lagrangian)

        # 4. Gradient of AL w.r.t. theta (optimized parameters)
        grad_theta_fn = grad(augmented_lagrangian, argnums=1)
        self._grad_theta_augmented_lagrangian_jit = jit(grad_theta_fn)

        # 5. Jacobian of Residuals w.r.t. theta
        def residual_single_theta(
            t_k: float,
            t_kp1: float,
            w_k: jnp.ndarray,
            w_kp1: jnp.ndarray,
            theta_vals: jnp.ndarray
        ):
            p_all = self.p_all.at[self.optimize_indices_jax].set(theta_vals)
            return self.jac.residual_single(t_k, t_kp1, w_k, w_kp1, p_all)

        jac_theta_fn = jax.jacfwd(residual_single_theta, argnums=4)
        self._jac_theta_residual_vmap = jit(vmap(
            jac_theta_fn,
            in_axes=(0, 0, 0, 0, None)
        ))

        # 6. Gradient of Phi (Objective) w.r.t theta
        def phi_theta(
            w: jnp.ndarray,
            theta_vals: jnp.ndarray,
            t_array: jnp.ndarray,
            y_target: jnp.ndarray
        ):
            p_all = self.p_all.at[self.optimize_indices_jax].set(theta_vals)

            if self.jac.h_funcs:
                outputs = vmap(self.jac.eval_h_jax, in_axes=(0, 0, 0, None))(
                    t_array,
                    w[:, :self.n_states],
                    w[:, self.n_states:],
                    p_all
                )
            else:
                outputs = w[:, :self.n_states]

            diff = outputs - y_target
            if self.loss_type == 'mean':
                return jnp.mean(diff**2)
            else:
                return jnp.sum(diff**2)

        self._grad_phi_theta_jit = jit(grad(phi_theta, argnums=1))

    def _compute_trapezoidal_adjoint_matrixfree(
        self,
        t_k: jnp.ndarray,
        t_kp1: jnp.ndarray,
        y_k: jnp.ndarray,
        y_kp1: jnp.ndarray,
        p_opt_vals_jax: jnp.ndarray,
        dL_dy_adjoint: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Matrix-free adjoint for trapezoidal method."""
        N = t_k.shape[0]

        J_curr = self.jac._jac_y_kp1_vmapped(t_k, t_kp1, y_k, y_kp1, p_opt_vals_jax)
        J_curr_T = jnp.transpose(J_curr, (0, 2, 1))

        lu_factors, lu_pivots = vmap(lu_factor)(J_curr_T)
        v_all = vmap(lu_solve)((lu_factors, lu_pivots), dL_dy_adjoint)

        t_k_shift = t_k[1:]
        t_kp1_shift = t_kp1[1:]
        y_k_shift = y_k[1:]
        y_kp1_shift = y_kp1[1:]

        J_prev_shift = self.jac._jac_y_k_vmapped(t_k_shift, t_kp1_shift,
                                                  y_k_shift, y_kp1_shift, p_opt_vals_jax)
        J_prev_T_shift = jnp.transpose(J_prev_shift, (0, 2, 1))

        lu_factors_m = lu_factors[:-1]
        lu_pivots_m = lu_pivots[:-1]
        M_blocks = -vmap(lu_solve)((lu_factors_m, lu_pivots_m), J_prev_T_shift)

        return M_blocks, v_all

    def compute_grad_phi_theta(
        self,
        t_array: np.ndarray,
        w: np.ndarray,
        theta: np.ndarray,
        y_target: np.ndarray
    ) -> np.ndarray:
        """Compute gradient of trajectory loss w.r.t optimized parameters."""
        if w.shape[0] != len(t_array) and w.shape[1] == len(t_array):
            w = w.T
        return np.array(self._grad_phi_theta_jit(
            jnp.array(w),
            jnp.array(theta),
            jnp.array(t_array),
            jnp.array(y_target)
        ))

    def compute_augmented_lagrangian(
        self,
        t_array: np.ndarray,
        w: np.ndarray,
        theta: np.ndarray,
        eta: np.ndarray,
        y_target: np.ndarray,
        mu: float
    ) -> float:
        """Compute AL value (wrapper for JIT function)."""
        return float(self._augmented_lagrangian_jit(
            jnp.array(w),
            jnp.array(theta),
            jnp.array(eta),
            float(mu),
            jnp.array(t_array),
            jnp.array(y_target)
        ))

    def compute_grad_w_augmented_lagrangian(
        self,
        t_array: np.ndarray,
        w: np.ndarray,
        theta: np.ndarray,
        eta: np.ndarray,
        y_target: np.ndarray,
        mu: float
    ) -> np.ndarray:
        """Compute gradient of AL w.r.t w (wrapper for JIT function)."""
        if w.shape[0] != len(t_array) and w.shape[1] == len(t_array):
            w = w.T

        if eta.shape[0] == len(t_array):
             pass

        return np.array(self._grad_w_augmented_lagrangian_jit(
            jnp.array(w),
            jnp.array(theta),
            jnp.array(eta),
            float(mu),
            jnp.array(t_array),
            jnp.array(y_target)
        ))

    def compute_grad_theta_augmented_lagrangian(
        self,
        t_array: np.ndarray,
        w: np.ndarray,
        theta: np.ndarray,
        eta: np.ndarray,
        y_target: np.ndarray,
        mu: float
    ) -> np.ndarray:
        """Compute gradient of AL w.r.t theta (wrapper for JIT function)."""
        if w.shape[0] != len(t_array) and w.shape[1] == len(t_array):
            w = w.T

        return np.array(self._grad_theta_augmented_lagrangian_jit(
            jnp.array(w),
            jnp.array(theta),
            jnp.array(eta),
            float(mu),
            jnp.array(t_array),
            jnp.array(y_target)
        ))

    def optimize(
        self,
        t_array: np.ndarray,
        y_target: np.ndarray,
        p_init: np.ndarray = None,
        n_iterations: int = 100,
        tol: float = 1e-4,
        verbose: bool = True,
        solver_rtol: float = 1e-6,
        solver_atol: float = 1e-6
    ) -> Dict:
        """
        Run Augmented Lagrangian optimization loop (Algorithm 3 Option C).

        Args:
            t_array: Time points (N,)
            y_target: Target trajectory (N, n_outputs)
            p_init: Initial guess for optimized parameters (optional)
            n_iterations: Maximum number of AL iterations
            tol: Convergence tolerance
            verbose: Print progress
            solver_rtol: Relative tolerance for internal DAE solver steps
            solver_atol: Absolute tolerance for internal DAE solver steps

        Returns:
            Dictionary with optimization results
        """
        # Import DAESolver here to avoid circular imports
        from src.discrete_adjoint.dae_solver import DAESolver

        # 1. Initialization
        if p_init is None:
            theta = np.array(self.p_current)
        else:
            theta = np.array(p_init)

        mu = getattr(self, 'penalty_mu', 1.0)
        alpha_w = getattr(self, 'alpha_w', 0.01)
        alpha_theta = getattr(self, 'alpha_theta', 0.01)
        n_primal_steps = getattr(self, 'n_primal_steps', 1)

        # --- Solve DAE for initial w ---
        p_all_current = np.array(self.jac.p)
        for i, idx in enumerate(self.optimize_indices):
            p_all_current[idx] = theta[i]

        dae_data_curr = self.dae_data.copy()

        for i, param in enumerate(dae_data_curr['parameters']):
             param['value'] = float(p_all_current[i])

        if verbose:
            print("Solving DAE for initial w guess...")

        solver = DAESolver(dae_data_curr, verbose=False)
        try:
            res = solver.solve(t_span=(t_array[0], t_array[-1]), ncp=len(t_array), rtol=solver_rtol, atol=solver_atol)
            x_init = res['x']
            z_init = res['z']

            if z_init is not None and z_init.size > 0:
                w_sol = np.vstack([x_init, z_init]).T
            else:
                w_sol = x_init.T

            if w_sol.shape[0] != len(t_array):
                print(f"Warning: Initial DAE solve length {w_sol.shape[0]} != t_array {len(t_array)}.")
                raise ValueError("Grid mismatch in initialization.")

            w = w_sol
            if verbose: print("Initial w computed successfully.")

        except Exception as e:
            if verbose: print(f"Initial DAE solve failed: {e}. Fallback to zeros.")
            w = np.zeros((len(t_array), self.n_total))

        # Initialize eta = 0
        eta = np.zeros((len(t_array)-1, self.n_total))

        history = {'loss': [], 'mu': [], 'grad_theta_norm': [], 'residual_norm': []}
        start_time = time.time()

        for k in range(n_iterations):
            iter_start = time.time()

            # --- 1. Optimal Reset / Feasibility (Solve DAE) ---
            p_all_iter = np.array(self.jac.p)
            for i, idx in enumerate(self.optimize_indices):
                p_all_iter[idx] = theta[i]

            dae_data_curr = self.dae_data.copy()
            for i, param in enumerate(dae_data_curr['parameters']):
                 param['value'] = float(p_all_iter[i])

            solver_step = DAESolver(dae_data_curr, verbose=False)

            try:
                res_step = solver_step.solve(
                    t_span=(t_array[0], t_array[-1]),
                    ncp=len(t_array),
                    rtol=solver_rtol,
                    atol=solver_atol
                )
                x_step = res_step['x']
                z_step = res_step['z']

                if z_step is not None and z_step.size > 0:
                    w_step = np.vstack([x_step, z_step]).T
                else:
                    w_step = x_step.T

                if w_step.shape[0] == len(t_array):
                    w = jnp.array(w_step)
                else:
                    if verbose: print(f"Warning: Iter {k} DAE solve grid mismatch. Keeping previous w.")

            except Exception as e:
                print(f"Warning: Iter {k} DAE solve failed: {e}. Keeping previous w.")

            # 2. Primal Step (w)
            for _ in range(n_primal_steps):
                grad_w = self.compute_grad_w_augmented_lagrangian(
                    t_array, w, theta, eta, y_target, mu
                )
                w = w - alpha_w * grad_w

            # Compute new residuals R(w^{k+1/2}, theta^k)
            p_all_iter = np.array(self.jac.p)
            for i, idx in enumerate(self.optimize_indices):
                p_all_iter[idx] = theta[i]

            residuals = np.array(self._residual_vmap(
                jnp.array(t_array[:-1]), jnp.array(t_array[1:]),
                jnp.array(w[:-1]), jnp.array(w[1:]),
                jnp.array(p_all_iter)
            ))

            # 3. Multiplier Update (Dual Ascent) - EXACT ADJOINT
            p_jax = jnp.array(p_all_iter)
            w_jax = jnp.array(w)
            t_jax = jnp.array(t_array)
            y_target_jax = jnp.array(y_target)

            # 1. dL/dy
            dL_dy = self.jac.trajectory_loss_gradient_analytical(t_jax, w_jax, y_target_jax, p_jax)

            if self.loss_type == 'mean':
                 n_outputs = y_target.shape[1]
                 n_time = len(t_array)
                 dL_dy = dL_dy / (n_time * n_outputs)

            dL_dy_adjoint = dL_dy[1:, :]

            # 2. Compute Adjoint System components
            t_k = t_jax[:-1]
            t_kp1 = t_jax[1:]
            y_k = w_jax[:-1]
            y_kp1 = w_jax[1:]

            M_blocks, v_all = self._compute_trapezoidal_adjoint_matrixfree(
                t_k, t_kp1, y_k, y_kp1, p_jax, dL_dy_adjoint
            )

            # 3. Solve Adjoint System (Parallel Scan)
            y0 = v_all[-1]
            vecs = v_all[:-1][::-1]
            mats = M_blocks[::-1]

            N_intervals = len(t_k)

            if N_intervals == 1:
                lambda_adjoint = v_all
            else:
                y_rev = matmul_recursive(mats, vecs, y0)
                lambda_adjoint = y_rev[::-1]

            eta_new = -lambda_adjoint
            eta = eta_new

            # 5. Parameter Update
            grad_theta = self.compute_grad_theta_augmented_lagrangian(
                t_array, w, theta, eta, y_target, mu
            )

            theta = theta - alpha_theta * grad_theta

            # Logging
            al_val = self.compute_augmented_lagrangian(t_array, w, theta, eta, y_target, mu)
            residual_norm = np.linalg.norm(residuals)
            grad_theta_norm = np.linalg.norm(grad_theta)

            history['loss'].append(al_val)
            history['residual_norm'].append(residual_norm)
            history['grad_theta_norm'].append(grad_theta_norm)
            history['mu'].append(mu)

            iter_time = time.time() - iter_start

            if verbose:
                print(f"Iter {k+1:3d} | AL: {al_val:.4e} | ||R||: {residual_norm:.4e} | ||g_theta||: {grad_theta_norm:.4e} | mu: {mu:.1e} | t: {iter_time:.2f}s")

            if residual_norm < tol and grad_theta_norm < tol:
                if verbose: print(f"Converged at iteration {k+1}")
                break

        total_time = time.time() - start_time
        if verbose:
            print(f"Optimization finished in {total_time:.2f}s")

        return {
            'theta_opt': theta,
            'w_opt': w,
            'history': history
        }

    def compute_jacobian_residual_theta(
        self,
        t_array: np.ndarray,
        w: np.ndarray,
        theta: np.ndarray
    ) -> np.ndarray:
        """
        Compute Jacobian of the residual vector w.r.t. optimized parameters.

        Args:
            t_array: Time points (N,)
            w: State trajectory (N, n_total)
            theta: Optimized parameters (n_theta,)

        Returns:
            Jacobian tensor of shape (N-1, n_total, n_theta)
        """
        if w.shape[0] != len(t_array) and w.shape[1] == len(t_array):
            w = w.T

        t_k = t_array[:-1]
        t_kp1 = t_array[1:]
        w_k = w[:-1]
        w_kp1 = w[1:]

        return np.array(self._jac_theta_residual_vmap(
            jnp.array(t_k),
            jnp.array(t_kp1),
            jnp.array(w_k),
            jnp.array(w_kp1),
            jnp.array(theta)
        ))
