import jax
import jax.numpy as jnp
import diffrax as dfx

def bouncing_ball(
    y0=jnp.array([1.0, 0.0]),   # [h0, v0]
    t0=0.0,
    t1=5.0,
    g=9.81,
    e=0.9,
    dt0=1e-3,
    max_bounces=50,
):
    # ODE vector field
    def f(t, y, args):
        g = args["g"]
        return jnp.array([y[1], -g])

    # Event: height crosses 0
    def event_fn(t, y, args, **kwargs):
        return y[0]

    term = dfx.ODETerm(f)
    solver = dfx.Tsit5()

    def run_segment(y_init, t_start, args):
        # 1. Run until event with diffrax.Event (stops correctly but gradient might be broken)
        event = dfx.Event(cond_fn=event_fn)
        
        # We need dense output for the root finding
        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=t_start,
            t1=t1,
            dt0=dt0,
            y0=y_init,
            args=args,
            saveat=dfx.SaveAt(dense=True, t1=True), # Dense for rootfind
            event=event,
            max_steps=1_000_000,
        )
        
        # 2. Extract approximate event time (non-differentiable stop)
        t_stop_approx = sol.ts[-1] 
        # Check if we actually stopped due to event (h ~ 0) or hit t1
        y_stop = sol.ys[-1]
        
        # 3. Differentiable Refinement
        # We assume t_stop_approx is close to the root. 
        # We use a few Newton steps on the dense interpolation to find exact t_event.
        # This restores the gradient path: params -> sol -> h(t) -> t_event
        
        def root_fn(t):
           # Evaluate height at t
           return sol.evaluate(t)[0]
           
        # Initial guess
        t_ev = t_stop_approx
        
        # 3 Newton steps (usually sufficient for cubic spline)
        # We use lax.scan or unroll for differentiable loop
        def newton_step(t_curr, _):
            h_val = root_fn(t_curr)
            # Use finite difference or autodiff for slope? Autodiff is better
            # slope = jax.grad(root_fn)(t_curr)
            # But wait, root_fn depends on sol.evaluate which is differentiable w.r.t t? Yes.
            
            # Note: diffrax.evaluate might return vector. root_fn returns scalar.
            # value_and_grad works for scalar output.
            val, slope = jax.value_and_grad(root_fn)(t_curr)
            
            # Avoid divide by zero
            t_next = t_curr - val / (slope + 1e-9)
            return t_next, None
            
        t_ev_refined, _ = jax.lax.scan(newton_step, t_ev, None, length=3)
        
        # If we didn't actually cross zero (e.g. hit t1), t_stop_approx is t1.
        # Newton might wander off if we are not careful, but for bouncing ball it's simple.
        # We can gate the update?
        # If y_stop[0] is large (didn't hit event), we shouldn't refine locally.
        # But 'Event' logic handled the stop.
        
        # Let's verify if we hit event
        hit_event = y_stop[0] < 1e-2 # Rough check
        
        final_t = jnp.where(hit_event, t_ev_refined, t_stop_approx)
        
        return final_t, sol

    args = {"g": g}
    impact_ts = []
    impact_ys = []

    y = y0
    t = t0

    for _ in range(max_bounces):
        t_event, sol = run_segment(y, t, args)
        
        # Get state at refined time
        y_event = sol.evaluate(t_event)
        
        impact_ts.append(t_event)
        impact_ys.append(y_event)

        if t_event >= t1 - 1e-4:
            break

        # Bounce logic
        v = y_event[1]
        v_bounced = jnp.where(v < 0.0, -e * v, v)
        # Reset h to small epsilon to avoid Zeno
        y = jnp.array([1e-5, v_bounced])
        t = t_event
        
        if jnp.abs(v) < 0.1 and y_event[0] < 1e-2:
             # Stop if low energy
             # break # Can't break in JIT easily but here we are python loop
             pass

    return jnp.array(impact_ts), jnp.stack(impact_ys, axis=0)


if __name__ == "__main__":
    try:
        print("Running simulation...")
        ts_imp, ys_imp = bouncing_ball(t1=3.0, e=0.8)
        print("Impact times:", ts_imp)
        print("Impact states sample:", ys_imp[:3])
        
        print("\nChecking gradient...")
        def loss(g_val):
            # Sum of impact times should depend on g
            ts, ys = bouncing_ball(t1=3.0, e=0.8, g=g_val)
            return jnp.sum(ts)
            
        g_init = 9.81
        l, grad = jax.value_and_grad(loss)(g_init)
        print(f"Loss: {l}, Grad: {grad}")
        
    except Exception as e:
        print("FAILED:", e)
        import traceback
        traceback.print_exc()
