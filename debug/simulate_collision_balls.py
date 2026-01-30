"""
Simulate the colliding balls DAE.

Generates ground-truth data from the default parameters and plots the trajectories
of the three balls.
"""

import numpy as np
import yaml
import json
import os
import sys
import matplotlib.pyplot as plt

import jax
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_platform_name", "cpu")

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from src.discrete_adjoint.dae_solver import DAESolver

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    solver_cfg = cfg['dae_solver']
    dae_spec_path = solver_cfg['dae_specification_file']
    # If path is relative, make it absolute relative to root
    if not os.path.isabs(dae_spec_path):
        dae_spec_path = os.path.join(root_dir, dae_spec_path)
        
    with open(dae_spec_path, 'r') as f:
        dae_data = json.load(f)
    return dae_data, solver_cfg


def run_simulation():
    print("=" * 70)
    print("SIMULATION: Colliding Balls DAE")
    print("=" * 70)

    # --- 1. Load config ---
    config_path = os.path.join(root_dir, 'config/config_bouncing_balls.yaml')
    dae_data, solver_cfg = load_config(config_path)

    t_start = solver_cfg['start_time']
    t_stop = solver_cfg['stop_time']
    ncp = solver_cfg['ncp']
    t_span = (t_start, t_stop)

    # Resolve parameter names and values
    param_names = [p['name'] for p in dae_data['parameters']]
    true_p = [p['value'] for p in dae_data['parameters']]
    print(f"Parameters: {dict(zip(param_names, true_p))}")

    # change parameter values if needed
    true_p[param_names.index('e_g')] = 0.79
    true_p[param_names.index('e_b')] = 0.63
    # --- 2. Run Solution ---
    solver = DAESolver(dae_data, verbose=True)
    solver.update_parameters(true_p)
    sol = solver.solve_augmented(t_span, ncp=ncp)

    print(f"Simulation completed. Segments: {len(sol.segments)}, Events: {len(sol.events)}")

    # --- 3. Plotting ---
    print("\nGenerating plots...")

    # Flatten simulated data for plotting (concatenating segments)
    sim_t = []
    sim_x = []
    for seg in sol.segments:
        sim_t.append(seg.t)
        sim_x.append(seg.x)
    
    if not sim_t:
        print("No simulation data generated.")
        return

    sim_t = np.concatenate(sim_t)
    sim_x = np.concatenate(sim_x)

    # Prepare figure for 3 Balls
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)

    # Indices for the 3 balls
    # Ball 1: x=0, y=1
    # Ball 2: x=4, y=5
    # Ball 3: x=8, y=9
    # Checking bounds just in case
    n_dim = sim_x.shape[1]
    
    ball_indices = []
    ball_labels = []
    ball_colors = ['b', 'r', 'g']

    if n_dim >= 2:
        ball_indices.append((0, 1))
        ball_labels.append('Ball 1')
    if n_dim >= 6:
        ball_indices.append((4, 5))
        ball_labels.append('Ball 2')
    if n_dim >= 10:
        ball_indices.append((8, 9))
        ball_labels.append('Ball 3')

    # --- Plot Y (Height) ---
    ax_y = fig.add_subplot(gs[0, 0])
    for k, (idx_x, idx_y) in enumerate(ball_indices):
        color = ball_colors[k % len(ball_colors)]
        label = ball_labels[k]
        ax_y.plot(sim_t, sim_x[:, idx_y], color=color, label=f'{label}')

    ax_y.set_xlabel('Time')
    ax_y.set_ylabel('Height (y)')
    ax_y.set_title('Height Trajectories')
    ax_y.legend()
    ax_y.grid(True, alpha=0.3)

    # --- Plot X (Horizontal) ---
    ax_x = fig.add_subplot(gs[0, 1])
    for k, (idx_x, idx_y) in enumerate(ball_indices):
        color = ball_colors[k % len(ball_colors)]
        label = ball_labels[k]
        ax_x.plot(sim_t, sim_x[:, idx_x], color=color, label=f'{label}')

    ax_x.set_xlabel('Time')
    ax_x.set_ylabel('Position (x)')
    ax_x.set_title('Horizontal Trajectories')
    ax_x.legend()
    ax_x.grid(True, alpha=0.3)

    # --- Plot Trajectory (X vs Y) ---
    ax_xy = fig.add_subplot(gs[1, :])
    for k, (idx_x, idx_y) in enumerate(ball_indices):
        color = ball_colors[k % len(ball_colors)]
        label = ball_labels[k]
        ax_xy.plot(sim_x[:, idx_x], sim_x[:, idx_y], color=color, label=f'{label}')
        # Mark start
        ax_xy.plot(sim_x[0, idx_x], sim_x[0, idx_y], 'o', color=color, markersize=8, markeredgecolor='k')
        # Mark end
        ax_xy.plot(sim_x[-1, idx_x], sim_x[-1, idx_y], 's', color=color, markersize=8, markeredgecolor='k')

    ax_xy.set_xlabel('Position (x)')
    ax_xy.set_ylabel('Height (y)')
    ax_xy.set_title('2D Trajectories (X vs Y)')
    ax_xy.legend()
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect('equal')

    plt.tight_layout()
    plot_path = os.path.join(current_dir, 'simulation_result_balls.png')
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    run_simulation()
