"""
An investigation into whether a two-matrix linear model exhibits the particular kind of rank collapse predicted from theory.
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

INPUT_DIM = 50
BOTTLENECK = 5
N_OBS = 10000
EPS = 0.3
N_EPOCHS = 20000
LR = 1e-4
LAMBDA_REG = 0.1

class TwoLinear(nn.Module):
    def __init__(self, input_dim, bottleneck):
        super().__init__()
        self.bottleneck_weight = nn.Linear(input_dim, bottleneck, bias=False)
        self.prediction_head = nn.Linear(bottleneck, 1, bias=False)

    def forward(self, X):
        X = self.bottleneck_weight(X)  # (n_obs, bottleneck)
        return self.prediction_head(X).squeeze(-1)  # (n_obs,)

# Generate linear regression data
X = torch.randn(N_OBS, INPUT_DIM)
true_w = torch.randn(INPUT_DIM, 1)
y = X @ true_w + EPS * torch.randn(N_OBS, 1)
y = y.squeeze(-1)

# Center and scale each column of X to variance 1
X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, unbiased=False, keepdim=True) + 1e-12)

# Center and scale y to variance 1
y = (y - y.mean()) / (y.std(unbiased=False) + 1e-12)

# Train the TwoLinear model
model = TwoLinear(INPUT_DIM, BOTTLENECK)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
train_losses = []

for _ in tqdm(range(N_EPOCHS), desc="Training"):
    optimizer.zero_grad()
    preds = model(X)
    obj = criterion(preds, y)
    reg = LAMBDA_REG * (model.bottleneck_weight.weight.norm(dim=0, p=3).sum() + model.prediction_head.weight.norm(p=2))
    loss = obj + reg
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

# Plot training error
os.makedirs("img", exist_ok=True)
plt.figure()
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Over Time")
plt.savefig("img/training.png")
plt.close()

trained_bottleneck = model.bottleneck_weight.weight.detach().numpy()  # (bottleneck, input_dim)

# Cosine similarity heatmap
norms = np.linalg.norm(trained_bottleneck, axis=1, keepdims=True)
cosine_sim = trained_bottleneck @ trained_bottleneck.T / (norms * norms.T + 1e-12)

plt.figure(figsize=(6, 5))
sns.heatmap(cosine_sim, annot=True, fmt=".2f", cmap="viridis")
plt.title("Cosine Similarity Between Bottleneck Rows")
plt.savefig("img/cosine.png")
plt.close()

# Bar chart of the prediction head and heatmap of the bottleneck weight
prediction_head_weights = model.prediction_head.weight.detach().cpu().numpy().flatten()
bottleneck_weights = model.bottleneck_weight.weight.detach().cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(np.arange(len(prediction_head_weights)), prediction_head_weights, color="steelblue")
axes[0].set_title("Prediction Head Weights")
axes[0].set_xlabel("Bottleneck Unit")
axes[0].set_ylabel("Weight")

sns.heatmap(bottleneck_weights, ax=axes[1], cmap="viridis", cbar=True)
axes[1].set_title("Bottleneck Weight Matrix")
axes[1].set_xlabel("Input Dimension")
axes[1].set_ylabel("Bottleneck Unit")

plt.tight_layout()
plt.savefig("img/weights.png")
plt.close()
