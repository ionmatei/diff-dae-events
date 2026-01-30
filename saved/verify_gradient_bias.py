
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

# Add src and root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from src.discrete_adjoint.dae_solver import DAESolver
import debug.verify_residual as dense
from debug.dae_padded_gradient import DAEPaddedGradient

def verify_gradient_bias():
    print("="*80)
    print("VERIFICATION: Gradient Bias vs Finite Difference")
    print("="*80)

    # 1. Setup system
    config_path = os.path.join(root_dir, 'config/config_bouncing_ball.yaml')
    dae_data, solver_cfg = dense.load_system(config_path)
    solver = DAESolver(dae_data, verbose=False)
    
    t_span = (0.0, 2.0)
    ncp = 100
    
    # Identify 'g' and 'e' parameter indices
    param_names = [p['name'] for p in dae_data['parameters']]
    g_idx = param_names.index('g')
    e_idx = param_names.index('e')
    
    # 2. Generate Target Data (Reference: g=9.81, e=0.8)
    print("Generating Reference Target Data (g=9.81, e=0.8)...")
    reference_p = [p['value'] for p in dae_data['parameters']]
    solver.update_parameters(reference_p)
    sol_ref = solver.solve_augmented(t_span, ncp=ncp)
    target_times, _ = dense.prepare_loss_targets(sol_ref, dae_data['states'], *t_span)

    print(f"Target Data Generation:")
    print(f"  Segments: {len(sol_ref.segments)}")
    for i, seg in enumerate(sol_ref.segments):
        print(f"    Seg {i}: t range [{seg.t[0]:.4f}, {seg.t[-1]:.4f}], pts={len(seg.t)}")
        print(f"      x_start: {seg.x[0]}, x_end: {seg.x[-1]}")
    print(f"  Events: {len(sol_ref.events)}")
    for i, ev in enumerate(sol_ref.events):
        print(f"    Event {i}: t={ev.t_event:.4f}, idx={ev.event_idx}")
        print(f"      x_pre: {ev.x_pre}, x_post: {ev.x_post}")

    # 3. Instantiate Padded Gradient Computer
    padded_gradient_computer = DAEPaddedGradient(dae_data, max_blocks=50, max_pts=500, max_targets=500)
    funcs = dense.create_jax_functions(dae_data)
    n_x, n_z, n_p = funcs[-1]

    # Generate target data through the same sigmoid-blended prediction
    # used in the loss to ensure L=0 at reference parameters.
    ts_ref = [s.t for s in sol_ref.segments]
    ys_ref = [jnp.concatenate([s.x, s.z], axis=1) for s in sol_ref.segments]
    ev_ref = [e.t_event for e in sol_ref.events]
    W_p_ref, TS_p_ref, bt_ref, bi_ref, bp_ref, _ = padded_gradient_computer._pad_problem_data(ts_ref, ys_ref, ev_ref)
    t_final_ref = sol_ref.segments[-1].t[-1]
    n_tgt = len(target_times)
    
    tt_padded = np.zeros(padded_gradient_computer.max_targets, dtype=np.float64)
    tt_padded[:n_tgt] = np.asarray(target_times)
    
    target_data = DAEPaddedGradient._predict_trajectory_padded_kernel(
        jnp.array(W_p_ref), jnp.array(TS_p_ref), jnp.array(bt_ref), 
        jnp.array(bi_ref), jnp.array(bp_ref), jnp.array(tt_padded), 
        t_final_ref, 150.0, n_x
    )[:n_tgt]

    # 4. Define Sweep Logic
    epsilon = 1e-6 # For FD
    
    def run_sensitivity_sweep(param_name, bias_values):
        p_idx = param_names.index(param_name)
        print(f"\nSENSITIVITY SWEEP: parameter '{param_name}'")
        print(f"{param_name + ' bias':<10} | {'Method':<16} | {'(dL/dg, dL/de)':<25} | {'Expectation':<12} | {'Diff to FD':<12}")
        print("-" * 100)

        sweep_plot_data = []

        for p_val in bias_values:
            # Update simulation parameters
            current_p = list(reference_p)
            current_p[p_idx] = p_val
            p_opt = jnp.array(current_p)
            
            # Determine Expectation (for MSE loss centered at reference)
            ref_val = reference_p[p_idx]
            if abs(p_val - ref_val) < 1e-4:
                expectation = "ZERO"
            elif p_val < ref_val:
                expectation = "NEGATIVE"
            else:
                expectation = "POSITIVE"

            # A. Refined Finite Difference (FD) Approach
            def get_loss_standard_at_p(p_query_list):
                solver.update_parameters(p_query_list)
                sol_q = solver.solve_augmented(t_span, ncp=ncp)

                ts_q = [s.t for s in sol_q.segments]
                ys_q = [jnp.concatenate([s.x, s.z], axis=1) for s in sol_q.segments]
                ev_q = [e.t_event for e in sol_q.events]
                W_p_q, TS_p_q, bt_q, bi_q, bp_q, _ = padded_gradient_computer._pad_problem_data(ts_q, ys_q, ev_q)
                t_f_q = sol_q.segments[-1].t[-1]

                # Use the same _loss_fn_padded as the adjoint for consistency
                tt_p, td_p, nt = padded_gradient_computer._pad_targets(target_times, target_data)
                W_d, TS_d, bt_d, bi_d, bp_d, tt_d, td_d = jax.device_put(
                    (W_p_q, TS_p_q, bt_q, bi_q, bp_q, tt_p, td_p))
                return float(DAEPaddedGradient._loss_fn_padded(
                    W_d, jnp.array(p_query_list), TS_d, bt_d, bi_d, bp_d,
                    tt_d, td_d, jnp.int32(nt), t_f_q, 150.0, n_x,
                    adaptive_horizon=True
                ))

            # dL/dg
            p_plus_g = list(current_p); p_plus_g[g_idx] += epsilon
            p_minus_g = list(current_p); p_minus_g[g_idx] -= epsilon
            grad_fd_g = (get_loss_standard_at_p(p_plus_g) - get_loss_standard_at_p(p_minus_g)) / (2 * epsilon)

            # dL/de
            p_plus_e = list(current_p); p_plus_e[e_idx] += epsilon
            p_minus_e = list(current_p); p_minus_e[e_idx] -= epsilon
            grad_fd_e = (get_loss_standard_at_p(p_plus_e) - get_loss_standard_at_p(p_minus_e)) / (2 * epsilon)

            # B. Padded Adjoint (JIT)
            solver.update_parameters(current_p)
            sol = solver.solve_augmented(t_span, ncp=ncp)
            
            # Collect data for plotting
            t_plot = np.concatenate([s.t for s in sol.segments])
            h_plot = np.concatenate([s.x[:, 0] for s in sol.segments])
            sweep_plot_data.append((p_val, t_plot, h_plot))
            
            grad_padded = padded_gradient_computer.compute_total_gradient(sol, p_opt, target_times, target_data, adaptive_horizon=True)
            val_padded_g = float(grad_padded[g_idx])
            val_padded_e = float(grad_padded[e_idx])

            # C. Dense Adjoint (Baseline)
            W_flat, structure_packed, grid_taus_packed = dense.pack_solution(sol, dae_data)
            dL_dW_dense, dL_dp_dense_all, _ = padded_gradient_computer.compute_loss_gradients(sol, p_opt, target_times, target_data, adaptive_horizon=True)
            
            def R_global(W, p_active):
                # Map p_active back to full p vector for unpacking
                p_full = list(current_p)
                p_full[g_idx] = p_active[0]
                p_full[e_idx] = p_active[1]
                return dense.unpack_and_compute_residual(
                    W, jnp.array(p_full), dae_data, structure_packed, funcs, (jnp.array(p_full), [g_idx, e_idx]), grid_taus_packed, 
                    t_final=sol.segments[-1].t[-1] if sol.segments else 2.0
                )
            
            p_active_vals = jnp.array([current_p[g_idx], current_p[e_idx]])
            dR_dW_dense = jax.jacfwd(R_global, 0)(W_flat, p_active_vals)
            dR_dp_dense = jax.jacfwd(R_global, 1)(W_flat, p_active_vals)
            
            lambda_dense = jnp.linalg.solve(dR_dW_dense.T + 1e-12*jnp.eye(dR_dW_dense.shape[0]), -dL_dW_dense)
            grad_dense_val = dL_dp_dense_all[jnp.array([g_idx, e_idx])] + jnp.dot(lambda_dense, dR_dp_dense)
            val_dense_g = float(grad_dense_val[0])
            val_dense_e = float(grad_dense_val[1])

            # D. Event Time Sensitivities
            event_time_sens = []
            idx_scan = 0
            for kind, count, *extra in structure_packed:
                length = extra[0] if kind == 'segment' else count
                if kind == 'event_time':
                    event_time_sens.append(float(dL_dW_dense[idx_scan]))
                idx_scan += length

            # Status check for the PRIMARY parameter of the sweep
            primary_padded = val_padded_g if param_name == 'g' else val_padded_e
            primary_fd = grad_fd_g if param_name == 'g' else grad_fd_e
            
            sign_ok = "PASS" if (expectation == "ZERO" and abs(primary_padded) < 1e-6) or \
                               (expectation == "NEGATIVE" and primary_padded < 0) or \
                               (expectation == "POSITIVE" and primary_padded > 0) else "FAIL"

            print(f"{p_val:<10.2f} | {'FD (g,e)':<16} | ({grad_fd_g:<10.4e}, {grad_fd_e:<10.4e}) | {expectation:<12} | {'0.0':<12}")
            print(f"{'':<10} | {'Padded (g,e)':<16} | ({val_padded_g:<10.4e}, {val_padded_e:<10.4e}) | {sign_ok:<12} | {abs(primary_padded - primary_fd):<12.2e}")
            print(f"{'':<10} | {'Dense (g,e)':<16} | ({val_dense_g:<10.4e}, {val_dense_e:<10.4e}) | {sign_ok:<12} | {abs(val_dense_g - grad_fd_g) if param_name=='g' else abs(val_dense_e - grad_fd_e):<12.2e}")
            if event_time_sens:
                sens_str = ", ".join([f"{s:.2e}" for s in event_time_sens])
                print(f"{'':<10} | {'dL/d(event_t)':<16} | [{sens_str}]")
            print("-" * 100)
        
        return sweep_plot_data

    # 4. Run Sweeps
    g_biases = [9.0, 9.81, 10.0]
    g_plot_data = run_sensitivity_sweep('g', g_biases)

    e_biases = [0.5, 0.8, 0.9]
    e_plot_data = run_sensitivity_sweep('e', e_biases)

    # 5. Generate Comparison Plots
    print("\nGenerating Trajectory Comparison Plot (Gravity)...")
    plt.figure(figsize=(10, 6))
    for p_val, t, h in g_plot_data:
        label = f"g={p_val:.2f}"
        if abs(p_val - 9.81) < 1e-4: label += " (Ref)"
        plt.plot(t, h, label=label, linewidth=2)
    plt.scatter(target_times, target_data[:, 0], color='black', marker='x', s=20, alpha=0.5, label="Targets")
    plt.xlabel("Time (s)"); plt.ylabel("Height (h)"); plt.title("Gravity Sensitivities"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(current_dir, "gradient_bias_trajectories.png"))
    
    print("Generating Trajectory Comparison Plot (Restitution)...")
    plt.figure(figsize=(10, 6))
    for p_val, t, h in e_plot_data:
        label = f"e={p_val:.2f}"
        if abs(p_val - 0.8) < 1e-4: label += " (Ref)"
        plt.plot(t, h, label=label, linewidth=2)
    plt.scatter(target_times, target_data[:, 0], color='black', marker='x', s=20, alpha=0.5, label="Targets")
    plt.xlabel("Time (s)"); plt.ylabel("Height (h)"); plt.title("Restitution Sensitivities"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(current_dir, "gradient_bias_restitution.png"))

if __name__ == "__main__":
    verify_gradient_bias()
