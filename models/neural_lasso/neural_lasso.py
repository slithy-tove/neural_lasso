import torch
import torch.nn as nn
import torch.utils.data as data
import plotly.express as px
from models import SparseEstimator
from data import load_data

class NeuralLasso(SparseEstimator, nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        SparseEstimator.__init__(self)

        self.update_header(**kwargs)
        dataset_name, sparsity, hdim, hnum = kwargs["dataset_name"], kwargs["sparsity"], kwargs["hdim"], kwargs["hnum"]

        self.dataset = load_data(dataset_name)
        self.input_dim = self.dataset.input_dim
        self.sparsity = sparsity

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
        self.loss_history = []

        self.update_header(**kwargs)
        n_epochs, batch_size, lr = kwargs["n_epochs"], kwargs["batch_size"], kwargs["lr"]

        X_tensor = torch.from_numpy(self.dataset.X).float()
        y_tensor = torch.from_numpy(self.dataset.y).float()

        td = data.TensorDataset(X_tensor, y_tensor)
        loader = data.DataLoader(td, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.forward(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            avg_loss = epoch_loss / len(td)
            self.loss_history.append(avg_loss)

        self.visualize()
        self.save_header()

    def plot_predictions(self):
        # make a plotly scatterplot of predictions vs ground truth, append to internal list of figures
        self.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(self.dataset.X).float()
            preds = self.forward(X_tensor).cpu().numpy()
        fig = px.scatter(
            x=self.dataset.y,
            y=preds,
            labels={"x": "Ground Truth", "y": "Predictions"},
            title="Predictions vs Ground Truth",
        )
        
        self.save_fig(fig = fig, name = "predictions")

    def plot_training(self):
        # plot training loss over time
        fig = px.line(
            y=self.loss_history,
            labels={"x": "Epoch", "y": "MSE Loss"},
            title="Training Loss Over Epochs",
        )

        self.save_fig(fig = fig, name = "training")

    def visualize(self):
        self.plot_predictions()
        self.plot_training()
