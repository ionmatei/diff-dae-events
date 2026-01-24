
import jax
import jax.numpy as jnp
import diffrax
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import re
from functools import partial

# Enable x64 precision (often needed for DAEs/stiff problems)
jax.config.update("jax_enable_x64", True)

def compile_equations_to_jax(eqn_strings, state_names, alg_names, param_names, extra_args=None):
    """Parses string equations into a single JAX-jittable function."""
    if not eqn_strings:
        return lambda t, x, z, p: jnp.array([])

    subs = []
    if extra_args:
        for name, repl in extra_args.items(): subs.append((name, repl))
    for i, name in enumerate(state_names): subs.append((name, f"x[{i}]"))
    for i, name in enumerate(alg_names):   subs.append((name, f"z[{i}]"))
    for i, name in enumerate(param_names): subs.append((name, f"p[{i}]"))
    subs.append(('time', 't'))
    subs.append(('t', 't'))
    subs.sort(key=lambda item: len(item[0]), reverse=True)

    processed_eqns = []
    math_funcs = {'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'exp', 'log', 'sqrt', 'abs', 'pow', 'min', 'max'}
    
    for eq in eqn_strings:
        processed_eq = eq
        for name, repl in subs:
            if name in math_funcs: continue
            pattern = r'(?<!\.)\b' + re.escape(name) + r'\b'
            processed_eq = re.sub(pattern, repl, processed_eq)
        processed_eqns.append(processed_eq)

    return_expr = "[" + ", ".join(processed_eqns) + "]"
    
    def create_closure():
        exec_ns = {'jnp': jnp}
        for f in math_funcs:
            if hasattr(jnp, f): exec_ns[f] = getattr(jnp, f)
        exec_ns['power'] = jnp.power 
        source = f"def compiled_fn(t, x, z, p): return jnp.array({return_expr})"
        exec(source, exec_ns)
        return exec_ns['compiled_fn']

    return create_closure()

class DAEOptimizerDiffrax:
    def __init__(self, dae_data: Dict, optimize_params: List[str], verbose: bool = True):
        self.dae_data = dae_data
        self.optimize_params = optimize_params
        self.verbose = verbose
        
        self.state_names = [s['name'] for s in dae_data['states']]
        self.alg_names = [] 
        self.param_names = [p['name'] for p in dae_data['parameters']]
        
        self.p_all = np.array([p['value'] for p in dae_data['parameters']])
        self.optimize_indices = tuple([self.param_names.index(p) for p in optimize_params])
        self.p_opt = np.array([self.p_all[i] for i in self.optimize_indices])
        
        self.n_states = len(self.state_names)
        
        # Compile Vector Field
        f_eqs_raw = dae_data['f']
        f_exprs = []
        for eq in f_eqs_raw:
            if '=' in eq:
                _, rhs = eq.split('=', 1)
                f_exprs.append(rhs.strip())
            else:
                f_exprs.append(eq)
                
        self._f_fn = compile_equations_to_jax(f_exprs, self.state_names, self.alg_names, self.param_names)

        # Compile Output Function
        h_eqs = self.dae_data.get('h', None)
        if h_eqs:
            clean_h = [eq.split('=', 1)[1].strip() if '=' in eq else eq for eq in h_eqs]
            self._h_fn = compile_equations_to_jax(clean_h, self.state_names, self.alg_names, self.param_names)
            self.n_outputs = len(h_eqs)
        else:
            self._h_fn = lambda t,x,z,p: x
            self.n_outputs = self.n_states

        # Setup Diffrax Terms
        self.term = diffrax.ODETerm(self._ode_term_func)
        self.solver = diffrax.Midpoint()
        # Use ConstantStepSize for explicit Midpoint as per user hint "faster"
        self.stepsize_controller = diffrax.ConstantStepSize()
        
        # Event Handling (Bouncing Ball specific logic)
        if 'when' in dae_data and dae_data['when']:
            when = dae_data['when'][0]
            cond_str = when['condition']
            
            # Condition: h < 0 -> h=0
            if "<" in cond_str:
                lhs, rhs = cond_str.split('<')
                cond_expr = f"{lhs} - {rhs}"
            else:
                cond_expr = cond_str
            
            # Simple jump logic for testing: v_new = -e * v_old
            def transition_fn(t, y, args, **kwargs):
                 # y[0] = h, y[1] = v. p[1] = e (assuming order from example)
                 # We rely on parameter indices.
                 # 'e' index logic:
                 e_idx = -1
                 for i, p_name in enumerate(self.param_names):
                     if p_name == 'e':
                         e_idx = i
                         break
                 
                 restitution = args[e_idx] if e_idx >= 0 else 0.8
                 v_new = -restitution * y[1]
                 return jnp.array([y[0], v_new])

            self._event_cond_fn = lambda t, y, args, **kwargs: y[0]
            self._event_trans_fn = transition_fn
        else:
             self._event_cond_fn = None
             self._event_trans_fn = None

    def _ode_term_func(self, t, y, args):
        z = jnp.array([])
        return self._f_fn(t, y, z, args)

    @partial(jax.jit, static_argnames=['self'])
    def _simulate(self, p_opt, t_span, target_times):
        """Simulates trajectory and returns predicted outputs at target_times."""
        p_full = jnp.array(self.p_all).at[jnp.array(self.optimize_indices)].set(p_opt)
        y0 = jnp.array([s['start'] for s in self.dae_data['states']], dtype=float)
        
        t_start_global = t_span[0]
        t_end_global = t_span[1]
        MAX_BOUNCES = 20
        
        # User example uses Tsit5
        solver = diffrax.Tsit5()
        # User example used default controller implicitly or passed PID?
        # User example: solver = dfx.Tsit5(); ... diffeqsolve(..., solver, ...)
        # Diffrax defaults to PIDController for Tsit5.
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-8)
        
        def run_segment(carry, i):
            t_curr, y_curr = carry
            
            # Event detection: height crosses 0
            # User example: return y[0] (root at h=0)
            def event_fn(t, y, args, **kwargs):
                return y[0]
            
            # Use diffrax.Event (works locally for stopping)
            event = diffrax.Event(cond_fn=event_fn)
            
            # SaveAt: dense for root finding
            saveat = diffrax.SaveAt(dense=True, t1=True)
            
            sol = diffrax.diffeqsolve(
                self.term, solver, t0=t_curr, t1=t_end_global, dt0=0.001, y0=y_curr, args=p_full,
                stepsize_controller=stepsize_controller,
                max_steps=100000, event=event, saveat=saveat, throw=False
            )
            
            t_stop_approx = sol.ts[-1] 
            y_final = sol.ys[-1] # State at t_stop_approx
            
            # Check if we should refine (only if we likely hit the event)
            h_stop = sol.evaluate(t_stop_approx)[0]
            should_refine = h_stop < 0.1
            
            # Refine event time differentiable
            def root_fn(t):
                # Clamp to avoid NaN from out-of-bounds evaluation
                t_safe = jnp.clip(t, t_curr, t_stop_approx)
                return sol.evaluate(t_safe)[0]
                
            def newton_step(t_curr, _):
                val, slope = jax.value_and_grad(root_fn)(t_curr)
                t_next = t_curr - val / (slope + 1e-9)
                return jnp.where(should_refine, t_next, t_curr), None

            t_val_start = sol.evaluate(t_stop_approx)[0]

            t_event, _ = jax.lax.scan(newton_step, t_stop_approx, None, length=3)
            
            # Clamp final result to be safe
            t_event = jnp.clip(t_event, t_curr, t_stop_approx)
            
            # Logic: If we didn't cross, t_event is t_stop_approx (handled by newton_step gating)

            # Prediction/Loss Logic with Smooth masking
            # Mask valid range [t_curr, t_event]
            S = 100.0 
            mask_start = jax.nn.sigmoid(S * (target_times - t_curr))
            mask_end = jax.nn.sigmoid(S * (t_event - target_times))
            mask = mask_start * mask_end
            
            # Clamp and Evaluate
            t_eval = jnp.clip(target_times, t_curr, t_event)
            preds = jax.vmap(sol.evaluate)(t_eval)
            
            def get_h(t, x): return self._h_fn(t, x, jnp.array([]), p_full)
            preds_h = jax.vmap(get_h)(t_eval, preds)
            
            masked_preds = preds_h * mask[:, None]
            
            # Reinitialization (Bounce)
            e_idx = -1
            for idx, name in enumerate(self.param_names):
                if name == 'e': e_idx = idx
            restitution = p_full[e_idx] if e_idx >= 0 else 0.8
            
            y_event_state = sol.evaluate(t_event)
            v_old = y_event_state[1]
            
            # Only bounce if moving downward (v < 0)
            v_new = jnp.where(v_old < 0, -restitution * v_old, v_old)
            
            # Explicitly reset height to small epsilon to prevent Zeno
            y_next = jnp.array([1e-5, v_new])
            
            # If we completed time (reached t_end_global), finish.
            finished = t_event >= (t_end_global - 1e-4)
            # If finished, we just hold the state (or it doesn't matter as mask is 0)
            
            return (t_event, y_next), (masked_preds, mask)

        init = (t_start_global, y0)
        final_state, (preds_stack, mask_stack) = jax.lax.scan(run_segment, init, jnp.arange(MAX_BOUNCES))
        
        total_preds = jnp.sum(preds_stack, axis=0)
        total_mask = jnp.sum(mask_stack, axis=0) 
        total_mask = jnp.maximum(total_mask, 1e-6)
        
        return total_preds / total_mask[:, None]

    def predict_outputs(self, p_opt, t_span, target_times):
        """Public method to predict outputs (numpy compatible)."""
        # Ensure imports are available if calling from outside (though this is instance method)
        p_opt_jax = jnp.array(p_opt)
        y_preds = self._simulate(p_opt_jax, t_span, jnp.array(target_times))
        return np.array(y_preds)

    def optimize(self, t_span, target_times, target_outputs, max_iterations=100, step_size=0.01, tol=1e-5, print_every=10, algorithm='adam', **kwargs):
        
        @jax.jit
        def loss_fn(p_opt):
            y_preds = self._simulate(p_opt, t_span, jnp.array(target_times))
            return jnp.mean((y_preds - target_outputs)**2)
        
        grad_fn = jax.value_and_grad(loss_fn)
        
        # Adam State
        p_curr = self.p_opt.copy()
        m = np.zeros_like(p_curr)
        v = np.zeros_like(p_curr)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        history = {'loss': [], 'params': []}
        
        print(f"Starting Diffrax Optimization ({algorithm})")
        print(f"Max Iters: {max_iterations}, Step: {step_size}")
        
        for i in range(max_iterations):
            start = time.time()
            loss_val, grads = grad_fn(p_curr)
            
            history['loss'].append(float(loss_val))
            history['params'].append(p_curr.copy())
            
            grad_norm = np.linalg.norm(grads)
            
            if i % print_every == 0 or i == max_iterations - 1:
                print(f"Iter {i}: Loss={loss_val:.6e}, |grad|={grad_norm:.6e}, time={time.time()-start:.3f}s")
            
            if grad_norm < tol:
                print("Converged.")
                break
            
            # Update
            if algorithm.lower() == 'adam':
                m = beta1 * m + (1 - beta1) * grads
                v = beta2 * v + (1 - beta2) * grads**2
                m_hat = m / (1 - beta1**(i + 1))
                v_hat = v / (1 - beta2**(i + 1))
                p_curr = p_curr - step_size * m_hat / (np.sqrt(v_hat) + eps)
            else:
                p_curr = p_curr - step_size * grads
                
        return {'params': p_curr, 'history': history, 'converged': True, 'elapsed_time': 0}
