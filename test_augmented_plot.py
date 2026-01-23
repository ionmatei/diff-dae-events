
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from src.discrete_adjoint.dae_solver import DAESolver, AugmentedSolution

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_augmented_solution(aug_sol, title="Augmented Solution Trajectory"):
    # Stitch segments for plotting
    t_all = []
    x_all = []
    
    print(f"Plotting: Found {len(aug_sol.segments)} segments.")
    for i, seg in enumerate(aug_sol.segments):
        # Verify segment data
        if len(seg.t) == 0:
            print(f"Warning: Segment {i} is empty!")
            continue
            
        t_all.append(seg.t)
        x_all.append(seg.x)
        print(f"  Seg {i}: {len(seg.t)} points")
        
    if not t_all:
        print("Error: No data to plot!")
        return
        
    t_concat = np.concatenate(t_all)
    x_concat = np.concatenate(x_all)
    
    print(f"Total points to plot: {len(t_concat)}")
    print(f"State shape: {x_concat.shape}")
    
    # Create subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot Height
    # Plot each segment individually to show where the points are
    # Using marker='.' to confirm density
    start_idx = 0
    colors = ['b', 'c', 'm', 'k'] # Cycle colors for segments to distinguish them? No, keep simple.
    
    for i, seg in enumerate(aug_sol.segments):
        ax1.plot(seg.t, seg.x[:, 0], '.-', markersize=3, linewidth=1, color='b')
        ax2.plot(seg.t, seg.x[:, 1], '.-', markersize=3, linewidth=1, color='g')

    # Add events to both plots
    for ev in aug_sol.events:
        # Height events
        ax1.axvline(x=ev.t_event, color='r', linestyle=':', alpha=0.5)
        ax1.plot(ev.t_event, ev.x_pre[0], 'rx', markersize=6)
        
        # Velocity events (velocity jump is the main feature)
        ax2.axvline(x=ev.t_event, color='r', linestyle=':', alpha=0.5)
        ax2.plot(ev.t_event, ev.x_pre[1], 'rx', markersize=6)
        ax2.plot(ev.t_event, ev.x_post[1], 'ro', markersize=4, markerfacecolor='none')

    ax1.set_ylabel('Height (m)')
    ax1.set_title(title + " (Raw Output)")
    ax1.grid(True)
    
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    filename = "bouncing_ball_augmented_trajectory.png"
    plt.savefig(filename, dpi=100)
    print(f"Plot saved to '{filename}'")

def test_bouncing_ball_simulation():
    config_path = "config/config_bouncing_ball.yaml"
    print(f"Loading config from {config_path}")
    config = load_config(config_path)
    
    solver_cfg = config['dae_solver']
    dae_file = solver_cfg['dae_specification_file']
    t_start = solver_cfg.get('start_time', 0.0)
    t_stop = solver_cfg.get('stop_time', 3.0)
    atol = solver_cfg.get('atol', 1e-6)
    rtol = solver_cfg.get('rtol', 1e-6)
    
    # Use ncp from config to set density
    ncp = solver_cfg.get('ncp', 500)
    
    print(f"DAE File: {dae_file}")
    with open(dae_file, 'r') as f:
        dae_data = json.load(f)
        
    solver = DAESolver(dae_data, verbose=True)
    
    print(f"\nSolving augmented system from t={t_start} to t={t_stop}...")
    print(f"  Target density: ncp={ncp}")
    
    aug_sol = solver.solve_augmented(
        t_span=(t_start, t_stop),
        atol=atol,
        rtol=rtol,
        ncp=ncp # Pass ncp to enforce max_step
    )
    
    print(f"\nSimulation Complete.")
    print(f"Total Segments: {len(aug_sol.segments)}")
    print(f"Total Events: {len(aug_sol.events)}")
    
    # Print event details
    for i, ev in enumerate(aug_sol.events):
        print(f"Event {i}: t={ev.t_event:.6f}")
        print(f"  Pre:  h={ev.x_pre[0]:.6f}, v={ev.x_pre[1]:.6f}")
        print(f"  Post: h={ev.x_post[0]:.6f}, v={ev.x_post[1]:.6f}")

    plot_augmented_solution(aug_sol)

if __name__ == "__main__":
    test_bouncing_ball_simulation()
