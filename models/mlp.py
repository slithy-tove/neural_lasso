import torch
import torch.nn as nn
import torch.utils.data as data
import plotly.express as px
import numpy as np
from models import SparseEstimator
from data import load_data
from tqdm import tqdm

class MLP(SparseEstimator, nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        SparseEstimator.__init__(self)

        self.update_header(**kwargs)
        dataset_name, sparsity, hdim, hnum = kwargs["dataset_name"], kwargs["sparsity"], kwargs["hdim"], kwargs["hnum"]

        self.dataset = load_data(dataset_name)
        self.input_dim = self.dataset.input_dim
        self.sparsity = sparsity

        self.X_tensor = torch.from_numpy(self.dataset.X).float()
        self.y_tensor = torch.from_numpy(self.dataset.y).float()

        self.bottleneck_weight = nn.Linear(self.input_dim, sparsity)

        layers = []
        # first hidden layer
        layers.append(nn.Linear(sparsity, hdim))
        layers.append(nn.ReLU())
        # intermediate hidden layers
        for _ in range(hnum - 2):
            layers.append(nn.Linear(hdim, hdim))
            layers.append(nn.ReLU())
        # output layer
        layers.append(nn.Linear(hdim, 1))

        self.loss_history = []

        # by default, don't regularize
        self.use_reg = False

        self.mlp = nn.Sequential(*layers)

    def forward(self, X):
        """
        X: torch.Tensor of shape (n_obs, input_dim)
        """
        X_b = self.bottleneck_weight(X)  # (n_obs, sparsity)
        return self.mlp(X_b).squeeze(dim=-1)

    def fit(self, **kwargs):
        """
        Fit model to some data.
        """
        self.update_header(**kwargs)
        n_epochs, batch_size, lr = kwargs["n_epochs"], kwargs["batch_size"], kwargs["lr"]

        td = data.TensorDataset(self.X_tensor, self.y_tensor)
        loader = data.DataLoader(td, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in tqdm(range(n_epochs)):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.forward(xb)
                loss = criterion(preds, yb)
                if self.use_reg:
                    reg = self.get_reg_loss(xb)
                    loss = loss + reg
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            avg_loss = epoch_loss / len(td)
            self.loss_history.append(avg_loss)

    def plot_predictions(self):
        self.eval()
        with torch.no_grad():
            self.X_tensor = torch.from_numpy(self.dataset.X).float()
            preds = self.forward(self.X_tensor).cpu().numpy()
        mse = np.mean((preds - self.dataset.y) ** 2)
        fig = px.scatter(
            x=self.dataset.y,
            y=preds,
            labels={"x": "Ground Truth", "y": "Predictions"},
            title=f"Predictions vs Ground Truth (MSE: {mse:.4f})",
        )
        self.save_fig(fig=fig, name="predictions")

    def plot_training(self):
        # plot training loss over time
        fig = px.line(
            y=self.loss_history,
            labels={"x": "Epoch", "y": "MSE Loss"},
            title="Training Loss Over Epochs",
        )
        self.save_fig(fig=fig, name="training")

    def visualize(self):
        self.plot_predictions()
        self.plot_training()

    def get_reg_loss(self, X):
        raise NotImplementedError