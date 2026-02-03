"""
Compare DAE solver and PyTorch model simulation with identical true parameters.
Traces event detection step by step to identify divergence.
"""

import os, sys, json, yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint, odeint_event

torch.set_default_dtype(torch.float64)

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

from src.pytorch.bouncing_balls import BouncingBallsModel
from src.discrete_adjoint.dae_solver import DAESolver

EVENT_NAMES = [
    "Floor B1", "Floor B2", "Floor B3",
    "Ceil B1", "Ceil B2", "Ceil B3",
    "LWall B1", "LWall B2", "LWall B3",
    "RWall B1", "RWall B2", "RWall B3",
    "Coll B1-B2", "Coll B1-B3", "Coll B2-B3",
]

# --- Load config and JSON spec ---
with open(os.path.join(root_dir, 'config/config_bouncing_balls.yaml')) as f:
    config = yaml.safe_load(f)
solver_cfg = config['dae_solver']
with open(solver_cfg['dae_specification_file']) as f:
    dae_data = json.load(f)

p_true = {p['name']: p['value'] for p in dae_data['parameters']}
initial_state = [s['start'] for s in dae_data['states']]
t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
ncp = solver_cfg.get('ncp', 600)
t_end = t_span[1]

print("True params:", p_true)
print("t_span:", t_span, " ncp:", ncp)

# --- 1. DAE Solver ---
true_p_list = [p['value'] for p in dae_data['parameters']]
solver = DAESolver(dae_data, verbose=False)
solver.update_parameters(true_p_list)
sol_true = solver.solve_augmented(t_span, ncp=ncp)

all_t_dae, all_x_dae = [], []
for seg in sol_true.segments:
    if len(seg.t) > 0:
        all_t_dae.append(np.array(seg.t))
        all_x_dae.append(np.array(seg.x))
t_dae = np.concatenate(all_t_dae)
x_dae = np.concatenate(all_x_dae)
print(f"\nDAE Solver: {len(t_dae)} points, {len(sol_true.segments)} segments")
print("DAE segment boundaries:")
for i, seg in enumerate(sol_true.segments):
    if len(seg.t) > 0:
        print(f"  seg {i}: t=[{seg.t[0]:.6f}, {seg.t[-1]:.6f}] ({len(seg.t)} pts)")

# --- 2. PyTorch model - step-by-step event detection with debugging ---
model = BouncingBallsModel(
    g=p_true['g'], 
    e_g=p_true['e_g'], 
    # e_g=0.78,     
    e_b=p_true['e_b'],
    # e_b=0.86,    
    d_sq=p_true['d_sq'],
    x_min=p_true['x_min'], x_max=p_true['x_max'],
    y_min=p_true['y_min'], y_max=p_true['y_max'],
    initial_state=initial_state, ncp=ncp
)

print(f"\n{'='*70}")
print("Step-by-step PyTorch event detection (using model._find_next_event)")
print(f"{'='*70}")

t0, state = model.get_initial_state()
current_t = t0
current_state = state
pt_events = []

for event_num in range(20):
    current_t_val = current_t.detach().item()
    if current_t_val >= t_end:
        break

    print(f"\n--- Finding event {event_num} from t={current_t_val:.6f} ---")

    event_t, event_idx, state_at_event = model._find_next_event(
        current_state, current_t, t_end
    )

    if event_t is None:
        print("  No event found. Stopping.")
        break

    print(f"  >> Selected: ev={event_idx} ({EVENT_NAMES[event_idx]}) "
          f"at t={event_t.item():.6f}")
    print(f"     State at event: {state_at_event.detach().numpy()}")

    new_state = model.state_update(state_at_event, event_idx)
    print(f"     State after update: {new_state.detach().numpy()}")

    pt_events.append((event_t.item(), event_idx))
    current_state = new_state
    current_t = event_t

# --- 3. Summary ---
print(f"\n{'='*70}")
print("Summary")
print(f"{'='*70}")
print(f"\nDAE events ({len(sol_true.segments)-1}):")
for i in range(len(sol_true.segments) - 1):
    seg = sol_true.segments[i]
    print(f"  {i}: t={seg.t[-1]:.6f}")

print(f"\nPyTorch events ({len(pt_events)}):")
for i, (et, ei) in enumerate(pt_events):
    print(f"  {i}: t={et:.6f}, ev={ei} ({EVENT_NAMES[ei]})")

# --- 4. Plot ---
# Re-simulate PyTorch for dense output
with torch.no_grad():
    times_pt, traj_pt = model.simulate_fixed_grid(t_span[1], n_points=ncp)
t_pt = times_pt.numpy()
x_pt = traj_pt.numpy()

state_names = [s['name'] for s in dae_data['states']]
fig, axes = plt.subplots(4, 3, figsize=(18, 16))

for i in range(12):
    row, col = i // 3, i % 3
    ax = axes[row, col]
    ax.plot(t_dae, x_dae[:, i], 'b-', linewidth=1.5, label='DAE Solver', alpha=0.8)
    ax.plot(t_pt, x_pt[:, i], 'r--', linewidth=1.5, label='PyTorch', alpha=0.8)
    ax.set_title(state_names[i])
    ax.set_xlabel('t')
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

    for et, _ in pt_events:
        ax.axvline(et, color='r', alpha=0.15, linewidth=0.5)
    for seg in sol_true.segments:
        if len(seg.t) > 0:
            ax.axvline(seg.t[-1], color='b', alpha=0.15, linewidth=0.5)

fig.suptitle(
    f'Solver comparison (true params: e_g={p_true["e_g"]}, e_b={p_true["e_b"]})\n'
    f'DAE: {len(sol_true.segments)} segments | PyTorch: {len(pt_events)} events',
    fontsize=14
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compare_solvers.png')
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to: {out_path}")
