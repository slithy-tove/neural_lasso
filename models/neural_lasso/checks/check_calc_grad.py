import torch
import torch.nn as nn
from models.neural_lasso.utils import calc_grad

# Hyperparameters
N_OBS = 10
INPUT_DIM = 5
FEAT_IDX = 0
OBS_IDX = 0

# Generate random input tensor
X = torch.randn(N_OBS, INPUT_DIM, requires_grad=True)

# Simple MLP model
model = nn.Sequential(
    nn.Linear(INPUT_DIM, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# Compute gradients w.r.t. input
grads = calc_grad(model, X)

# Finite difference perturbation
eps = 1e-3
pert = torch.zeros_like(X)
pert[OBS_IDX, FEAT_IDX] = eps
X_p = X + pert

# Model outputs
y = model(X)
y_p = model(X_p)

# Compare analytical gradient to finite difference approximation
approx_grad = (y_p[OBS_IDX] - y[OBS_IDX]) / eps
print(f"Analytical grad: {grads[OBS_IDX, FEAT_IDX].item():.6f}, Finite diff approx: {approx_grad.item():.6f}")
