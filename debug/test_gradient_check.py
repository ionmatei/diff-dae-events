"""
Diagnostic: check loss at true params and verify gradients with finite differences.
Uses simulate_at_targets (direct evaluation, no interpolation).
Position-only loss (continuous across events).
"""
import os, sys, json, yaml
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

from src.pytorch.bouncing_balls import BouncingBallsModel
from src.discrete_adjoint.dae_solver import DAESolver

# --- Load config ---
with open(os.path.join(root_dir, 'config/config_bouncing_balls.yaml')) as f:
    config = yaml.safe_load(f)
solver_cfg = config['dae_solver']
with open(solver_cfg['dae_specification_file']) as f:
    dae_data = json.load(f)

p_true = {p['name']: p['value'] for p in dae_data['parameters']}
initial_state = [s['start'] for s in dae_data['states']]
t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
ncp = solver_cfg['ncp']

# --- Generate target data from DAE solver ---
true_p = [p['value'] for p in dae_data['parameters']]
solver = DAESolver(dae_data, verbose=False)
solver.update_parameters(true_p)
sol_true = solver.solve_augmented(t_span, ncp=ncp)

all_t, all_x = [], []
for seg in sol_true.segments:
    if len(seg.t) > 0:
        all_t.append(seg.t)
        all_x.append(seg.x)
target_times = np.concatenate([np.array(t[:-1]) for t in all_t])
target_data = np.concatenate([np.array(x[:-1]) for x in all_x])

target_times_t = torch.tensor(target_times, dtype=torch.float64)
target_data_t = torch.tensor(target_data, dtype=torch.float64)

# Position indices (continuous across events)
pos_idx = [0, 1, 4, 5, 8, 9]
state_names = [s['name'] for s in dae_data['states']]

print(f"Target data: {len(target_times)} points, last t={target_times[-1]:.6f}")
print(f"Using position-only loss (indices {pos_idx})")


def compute_loss(e_g_val, e_b_val, adjoint=False):
    """Compute position-only loss using simulate_at_targets."""
    model = BouncingBallsModel(
        g=p_true['g'], e_g=e_g_val, e_b=e_b_val, d_sq=p_true['d_sq'],
        x_min=p_true['x_min'], x_max=p_true['x_max'],
        y_min=p_true['y_min'], y_max=p_true['y_max'],
        initial_state=initial_state, ncp=ncp, adjoint=adjoint
    )

    y_pred = model.simulate_at_targets(target_times_t)
    loss = torch.mean((y_pred[:, pos_idx] - target_data_t[:, pos_idx]) ** 2)
    return loss, model, y_pred


def compute_loss_and_grad(e_g_val, e_b_val, adjoint=False):
    """Compute loss and gradients."""
    loss, model, _ = compute_loss(e_g_val, e_b_val, adjoint=adjoint)
    loss.backward()
    g_eg = model.e_g.grad.item() if model.e_g.grad is not None else 0.0
    g_eb = model.e_b.grad.item() if model.e_b.grad is not None else 0.0
    return loss.item(), g_eg, g_eb


# === Test 1: Loss at true parameters ===
print("=" * 60)
print("Test 1: Loss at TRUE parameters (should be ~0)")
print("=" * 60)
loss_true, model_true, y_pred_true = compute_loss(p_true['e_g'], p_true['e_b'])
print(f"  Loss at true params (pos only): {loss_true.item():.6e}")

# Per-state breakdown (all states for info)
for i in range(12):
    name = state_names[i]
    mse = torch.mean((y_pred_true[:, i] - target_data_t[:, i]) ** 2).item()
    max_err = torch.max(torch.abs(y_pred_true[:, i] - target_data_t[:, i])).item()
    marker = " *" if i in pos_idx else ""
    if mse > 1e-8:
        print(f"  {name}: MSE={mse:.6e}, max_err={max_err:.6e}{marker}")


# === Test 2: Loss at biased parameters ===
print("\n" + "=" * 60)
print("Test 2: Loss at BIASED parameters (e_g+0.1, e_b+0.1)")
print("=" * 60)
loss_biased, _, _ = compute_loss(p_true['e_g'] + 0.1, p_true['e_b'] + 0.1)
print(f"  Loss at biased params (pos only): {loss_biased.item():.6e}")


# === Test 3: Gradient check with finite differences ===
print("\n" + "=" * 60)
print("Test 3: Gradient verification (autograd vs finite diff)")
print("=" * 60)

e_g_test = p_true['e_g'] + 0.1
e_b_test = p_true['e_b'] + 0.1

# Autograd gradients
loss_ag, grad_eg_ag, grad_eb_ag = compute_loss_and_grad(e_g_test, e_b_test)
print(f"  Autograd: loss={loss_ag:.6e}, grad_eg={grad_eg_ag:.6e}, grad_eb={grad_eb_ag:.6e}")

# Finite difference gradients
eps = 1e-5
loss_eg_plus, _, _ = compute_loss(e_g_test + eps, e_b_test)
loss_eg_minus, _, _ = compute_loss(e_g_test - eps, e_b_test)
grad_eg_fd = (loss_eg_plus.item() - loss_eg_minus.item()) / (2 * eps)

loss_eb_plus, _, _ = compute_loss(e_g_test, e_b_test + eps)
loss_eb_minus, _, _ = compute_loss(e_g_test, e_b_test - eps)
grad_eb_fd = (loss_eb_plus.item() - loss_eb_minus.item()) / (2 * eps)

print(f"  Fin diff: loss={loss_ag:.6e}, grad_eg={grad_eg_fd:.6e}, grad_eb={grad_eb_fd:.6e}")

print(f"\n  Gradient comparison:")
if abs(grad_eg_fd) > 1e-10:
    print(f"    e_g: autograd/fd = {grad_eg_ag/grad_eg_fd:.4f}")
else:
    print(f"    e_g: fd is ~0, autograd={grad_eg_ag:.6e}")
if abs(grad_eb_fd) > 1e-10:
    print(f"    e_b: autograd/fd = {grad_eb_ag/grad_eb_fd:.4f}")
else:
    print(f"    e_b: fd is ~0, autograd={grad_eb_ag:.6e}")
