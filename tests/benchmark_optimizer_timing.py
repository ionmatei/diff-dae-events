"""
Timing benchmark for event-aware DAE optimization.

Measures:
1. Forward simulation time
2. Adjoint solver time (gradient computation)
3. Total iteration time
4. Overhead (everything else)
"""

import json
import numpy as np
import sys
import os
import time
from functools import wraps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax.numpy as jnp
from discrete_adjoint.dae_optimizer_parallel_optimized import DAEOptimizerParallelOptimized
from discrete_adjoint.dae_solver import DAESolver


class TimingProfiler:
    """Collects timing statistics for optimization operations."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.timings = {
            'forward_sim': [],
            'adjoint_solve': [],
            'param_update': [],
            'data_prep': [],
            'total_iter': [],
        }

    def record(self, name, elapsed):
        if name in self.timings:
            self.timings[name].append(elapsed)

    def summary(self):
        print("\n" + "="*80)
        print("TIMING BENCHMARK RESULTS")
        print("="*80)

        for name, times in self.timings.items():
            if len(times) > 0:
                times_arr = np.array(times)
                # Skip first iteration (JIT compilation)
                if len(times_arr) > 1:
                    times_no_jit = times_arr[1:]
                    print(f"\n{name}:")
                    print(f"  First iter (with JIT): {times_arr[0]*1000:.2f} ms")
                    print(f"  Mean (excl. first):    {np.mean(times_no_jit)*1000:.2f} ms")
                    print(f"  Std:                   {np.std(times_no_jit)*1000:.2f} ms")
                    print(f"  Min:                   {np.min(times_no_jit)*1000:.2f} ms")
                    print(f"  Max:                   {np.max(times_no_jit)*1000:.2f} ms")
                else:
                    print(f"\n{name}:")
                    print(f"  Single measurement: {times_arr[0]*1000:.2f} ms")

        # Compute breakdown
        if len(self.timings['total_iter']) > 1:
            print("\n" + "-"*40)
            print("TIME BREAKDOWN (excluding first iteration)")
            print("-"*40)

            total = np.mean(self.timings['total_iter'][1:])
            forward = np.mean(self.timings['forward_sim'][1:])
            adjoint = np.mean(self.timings['adjoint_solve'][1:])
            param = np.mean(self.timings['param_update'][1:])
            data = np.mean(self.timings['data_prep'][1:])
            overhead = total - forward - adjoint - param - data

            print(f"  Forward simulation: {forward*1000:7.2f} ms ({forward/total*100:5.1f}%)")
            print(f"  Adjoint solve:      {adjoint*1000:7.2f} ms ({adjoint/total*100:5.1f}%)")
            print(f"  Parameter update:   {param*1000:7.2f} ms ({param/total*100:5.1f}%)")
            print(f"  Data preparation:   {data*1000:7.2f} ms ({data/total*100:5.1f}%)")
            print(f"  Other overhead:     {overhead*1000:7.2f} ms ({overhead/total*100:5.1f}%)")
            print(f"  ---------------------------------")
            print(f"  TOTAL:              {total*1000:7.2f} ms (100.0%)")


def run_benchmark(n_iterations=20, max_steps=500, max_events=20):
    """Run the timing benchmark."""

    profiler = TimingProfiler()

    print("="*80)
    print("DAE Optimizer Timing Benchmark")
    print("="*80)

    # Load DAE
    print("\nLoading bouncing ball DAE...")
    with open('dae_examples/dae_specification_bouncing_ball.json', 'r') as f:
        dae_data = json.load(f)

    # Set true parameters
    g_true, e_true = 9.81, 0.8
    for param in dae_data['parameters']:
        if param['name'] == 'g':
            param['value'] = g_true
        elif param['name'] == 'e':
            param['value'] = e_true

    # Create reference solver and generate target data
    print("Generating reference trajectory...")
    solver_ref = DAESolver(dae_data, verbose=False)
    result_ref = solver_ref.solve_with_events(
        t_span=(0.0, 3.0), ncp=300, rtol=1e-4, atol=1e-4,
        min_event_delta=0.01, verbose=False
    )

    t_target = result_ref['t']
    y_target = result_ref['x'][0:1, :].T

    print(f"  Time points: {len(t_target)}")
    print(f"  Events: {len(result_ref['event_times'])}")

    # Create optimizer
    print("\nCreating optimizer...")
    solver = DAESolver(dae_data, verbose=False)
    optimizer = DAEOptimizerParallelOptimized(
        dae_data=dae_data,
        dae_solver=solver,
        optimize_params=['g', 'e'],
        loss_type='mean',
        method='bdf6',
        rtol=1e-4, atol=1e-4,
        verbose=False
    )

    # Initial parameters
    p_init = np.array([10.5, 0.7])
    p = jnp.array(p_init)
    algo_step_size = 0.002
    min_event_delta = 0.01

    # Prepare static data
    var_indices_list = []
    all_var_names = optimizer.jac.state_names + optimizer.jac.alg_names
    for event_def in dae_data.get('when', []):
        reinit_str = event_def['reinit']
        found_var = None
        for name in sorted(all_var_names, key=len, reverse=True):
            if reinit_str.strip().startswith(name):
                found_var = name
                break
        if found_var:
            if found_var in optimizer.jac.state_names:
                idx = optimizer.jac.state_names.index(found_var)
            else:
                idx = optimizer.jac.n_states + optimizer.jac.alg_names.index(found_var)
            var_indices_list.append(idx)
        else:
            var_indices_list.append(-1)

    event_var_indices_jax = jnp.array(var_indices_list, dtype=jnp.int32)

    print(f"\nRunning {n_iterations} iterations with timing...")
    print(f"  MAX_STEPS: {max_steps}")
    print(f"  MAX_EVENTS: {max_events}")
    print("-"*60)

    for iteration in range(n_iterations):
        t_iter_start = time.perf_counter()

        # === 1. Parameter Update (Python side) ===
        t_param_start = time.perf_counter()
        p_all = np.array(optimizer.p_all)
        for i, opt_idx in enumerate(optimizer.optimize_indices):
            p_all[opt_idx] = float(p[i])
        for i in range(optimizer.n_params_total):
            optimizer.solver.p[i] = float(p_all[i])

        for param in dae_data['parameters']:
            for j, p_meta in enumerate(dae_data['parameters']):
                if p_meta['name'] == param['name']:
                    param['value'] = float(optimizer.solver.p[j])
                    break

        optimizer.solver.x0 = np.array([s['start'] for s in dae_data['states']])
        optimizer.solver.z0 = np.array([a.get('start', 0.0) for a in dae_data['alg_vars']])
        t_param_end = time.perf_counter()
        profiler.record('param_update', t_param_end - t_param_start)

        # === 2. Forward Simulation ===
        t_sim_start = time.perf_counter()
        result = optimizer.solver.solve_with_events(
            t_span=(float(t_target[0]), float(t_target[-1])),
            ncp=len(t_target), rtol=optimizer.rtol, atol=optimizer.atol,
            min_event_delta=min_event_delta, verbose=False
        )
        t_sim_end = time.perf_counter()
        profiler.record('forward_sim', t_sim_end - t_sim_start)

        t_sol = result['t']
        x_sol, z_sol = result['x'], result['z']
        event_times = result.get('event_times', [])
        event_indices = result.get('event_indices', [])

        n_actual = len(t_sol)
        n_ev = len(event_times)

        # === 3. Data Preparation ===
        t_prep_start = time.perf_counter()

        # Padding
        t_pad = np.zeros(max_steps)
        limit = min(n_actual, max_steps)
        t_pad[:limit] = t_sol[:limit]
        if limit < max_steps:
            t_pad[limit:] = t_sol[limit-1] if limit > 0 else 0.0

        y_sol_arr = np.vstack([x_sol, z_sol])
        y_pad = np.zeros((optimizer.jac.n_total, max_steps))
        y_pad[:, :limit] = y_sol_arr[:, :limit]
        if limit > 0:
            y_pad[:, limit:] = y_sol_arr[:, limit-1:limit]

        ev_times_pad = np.zeros(max_events)
        ev_indices_pad = np.zeros(max_events, dtype=int)
        limit_ev = min(n_ev, max_events)
        if limit_ev > 0:
            ev_times_pad[:limit_ev] = np.array(event_times)[:limit_ev]
            ev_indices_pad[:limit_ev] = np.array(event_indices)[:limit_ev]

        # y_target interpolation
        y_target_interp = np.zeros((n_actual, y_target.shape[1]))
        n_out = y_target.shape[1]
        for dim in range(n_out):
            y_target_interp[:, dim] = np.interp(t_sol, t_target, y_target[:, dim])

        y_target_pad = np.zeros((max_steps, n_out))
        y_target_pad[:limit, :] = y_target_interp[:limit, :]

        # Extract coefficients
        current_coeffs = []
        for k in range(len(var_indices_list)):
            reinit_str = dae_data['when'][k]['reinit']
            idx = var_indices_list[k]
            if idx != -1:
                if idx < optimizer.jac.n_states:
                    vname = optimizer.jac.state_names[idx]
                else:
                    vname = optimizer.jac.alg_names[idx - optimizer.jac.n_states]
                ce = optimizer._get_coefficient_value_optimized(reinit_str, vname)
                current_coeffs.append(ce)
            else:
                current_coeffs.append(0.0)

        event_coeffs_jax = jnp.array(current_coeffs, dtype=jnp.float64)
        limit_jax = jnp.array(limit, dtype=jnp.int32)

        t_prep_end = time.perf_counter()
        profiler.record('data_prep', t_prep_end - t_prep_start)

        # === 4. Adjoint Solve (JIT-compiled gradient) ===
        t_adjoint_start = time.perf_counter()

        p_new, grad_p = optimizer._compute_gradient_combined_padded(
            jnp.array(t_pad),
            jnp.array(y_pad),
            jnp.array(y_target_pad),
            p,
            algo_step_size,
            limit_jax,
            jnp.array(ev_times_pad),
            jnp.array(ev_indices_pad),
            event_var_indices_jax,
            event_coeffs_jax
        )
        # Force synchronization to get accurate timing
        grad_p.block_until_ready()

        t_adjoint_end = time.perf_counter()
        profiler.record('adjoint_solve', t_adjoint_end - t_adjoint_start)

        t_iter_end = time.perf_counter()
        profiler.record('total_iter', t_iter_end - t_iter_start)

        # Update for next iteration
        p = p_new

        # Progress
        if iteration % 5 == 0:
            grad_norm = float(jnp.linalg.norm(grad_p))
            print(f"  Iter {iteration:3d}: N={limit}, Ev={n_ev}, "
                  f"GradNorm={grad_norm:.2e}, Time={profiler.timings['total_iter'][-1]*1000:.1f}ms")

    # Print summary
    profiler.summary()

    return profiler


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark DAE optimizer timing')
    parser.add_argument('--iterations', '-n', type=int, default=20,
                        help='Number of iterations to run')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum padded trajectory length')
    parser.add_argument('--max-events', type=int, default=20,
                        help='Maximum number of events')
    args = parser.parse_args()

    run_benchmark(
        n_iterations=args.iterations,
        max_steps=args.max_steps,
        max_events=args.max_events
    )
