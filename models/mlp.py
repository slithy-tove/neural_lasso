import torch
import torch.nn as nn
import torch.utils.data as data
import plotly.express as px
import numpy as np
import pandas as pd
from models import SparseEstimator
from data import load_data
from tqdm import tqdm

class MLP(SparseEstimator, nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        SparseEstimator.__init__(self)

        self.update_header(**kwargs)
        dataset_name = kwargs["dataset_name"]
        sparsity = kwargs["sparsity"]
        hdim = kwargs["hdim"]
        hnum = kwargs["hnum"]

        self.dataset = load_data(dataset_name)
        self.input_dim = self.dataset.input_dim
        self.sparsity = sparsity
        self.ds_type = getattr(self.dataset, "ds_type", "reg")
        self.bottleneck = kwargs.get("bottleneck", sparsity)

        # tensors for train / test splits
        self.X_train_tensor = torch.from_numpy(self.dataset.X_train).float()
        self.y_train_tensor = torch.from_numpy(self.dataset.y_train).float()
        self.X_test_tensor = torch.from_numpy(self.dataset.X_test).float()
        self.y_test_tensor = torch.from_numpy(self.dataset.y_test).float()

        self.bottleneck_weight = nn.Linear(self.input_dim, self.bottleneck)

        layers = []
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.bottleneck, hdim))
        layers.append(nn.ReLU())
        for _ in range(hnum - 2):
            layers.append(nn.Linear(hdim, hdim))
            layers.append(nn.ReLU())
        out_dim = 1 if self.ds_type == "reg" else self.dataset.n_cls
        layers.append(nn.Linear(hdim, out_dim))

        self.mlp = nn.Sequential(*layers)

        self.use_reg = False

        # histories
        self.train_loss_history = []
        self.test_loss_history = []
        self.train_mse_history = []
        self.test_mse_history = []
        self.train_reg_history = []
        self.test_reg_history = []

    def forward(self, X):
        """
        X: torch.Tensor of shape (n_obs, input_dim)
        """
        X_b = self.bottleneck_weight(X)  # (n_obs, bottleneck)
        return self.mlp(X_b).squeeze(dim=-1)

    def fit(self, **kwargs):
        """
        Fit model to some data.
        """
        self.update_header(**kwargs)
        n_epochs = kwargs["n_epochs"]
        batch_size = kwargs["batch_size"]
        lr = kwargs["lr"]

        train_dataset = data.TensorDataset(self.X_train_tensor, self.y_train_tensor)
        test_dataset = data.TensorDataset(self.X_test_tensor, self.y_test_tensor)

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=len(train_dataset) if batch_size is None else batch_size,
            shuffle=True,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=len(test_dataset) if batch_size is None else batch_size,
            shuffle=False,
        )

        criterion = nn.MSELoss() if self.ds_type == "reg" else nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        for epoch in tqdm(range(n_epochs)):
            # ---------- training ----------
            self.train()
            epoch_loss = 0.0
            epoch_mse = 0.0
            epoch_reg = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = self.forward(xb)
                if self.ds_type == "cls":
                    target = torch.argmax(yb, dim=1)
                    loss = criterion(preds, target)
                else:
                    loss = criterion(preds, yb.squeeze())
                reg = self.get_reg_loss(xb) if self.use_reg else torch.tensor(0.0, device=loss.device)
                total_loss = loss + reg
                total_loss.backward()
                optimizer.step()
                batch_sz = xb.size(0)
                epoch_loss += total_loss.item() * batch_sz
                if self.ds_type == "reg":
                    epoch_mse += loss.item() * batch_sz
                epoch_reg += reg.item() * batch_sz
            avg_train_loss = epoch_loss / len(train_dataset)
            avg_train_mse = epoch_mse / len(train_dataset) if self.ds_type == "reg" else None
            avg_train_reg = epoch_reg / len(train_dataset)
            self.train_loss_history.append(avg_train_loss)
            self.train_reg_history.append(avg_train_reg)
            if self.ds_type == "reg":
                self.train_mse_history.append(avg_train_mse)

            # ---------- testing ----------
            self.eval()
            with torch.no_grad():
                epoch_loss = 0.0
                epoch_mse = 0.0
                epoch_reg = 0.0
                for xb, yb in test_loader:
                    preds = self.forward(xb)
                    if self.ds_type == "cls":
                        loss = criterion(preds, target)
                    else:
                        loss = criterion(preds, yb.squeeze())
                    reg = self.get_reg_loss(xb) if self.use_reg else torch.tensor(0.0, device=loss.device)
                    total_loss = loss + reg
                    batch_sz = xb.size(0)
                    epoch_loss += total_loss.item() * batch_sz
                    if self.ds_type == "reg":
                        epoch_mse += loss.item() * batch_sz
                    epoch_reg += reg.item() * batch_sz
                avg_test_loss = epoch_loss / len(test_dataset)
                avg_test_mse = epoch_mse / len(test_dataset) if self.ds_type == "reg" else None
                avg_test_reg = epoch_reg / len(test_dataset)
                self.test_loss_history.append(avg_test_loss)
                self.test_reg_history.append(avg_test_reg)
                if self.ds_type == "reg":
                    self.test_mse_history.append(avg_test_mse)

    def plot_predictions(self):
        self.eval()
        with torch.no_grad():
            X_all = torch.from_numpy(self.dataset.X).float()
            preds = self.forward(X_all).cpu().numpy()
        mse = np.mean((preds - self.dataset.y) ** 2)
        fig = px.scatter(
            x=self.dataset.y,
            y=preds,
            labels={"x": "Ground Truth", "y": "Predictions"},
            title=f"Predictions vs Ground Truth (MSE: {mse:.4f})",
        )
        self.save_fig(fig=fig, name="predictions")

    def plot_training(self):
        epochs = list(range(1, len(self.train_loss_history) + 1))
        data_dict = {
            "Epoch": epochs,
            "Train Loss": self.train_loss_history,
            "Test Loss": self.test_loss_history,
        }
        if self.ds_type == "reg":
            data_dict["Train MSE"] = self.train_mse_history
            data_dict["Test MSE"] = self.test_mse_history
        data_dict["Train Reg"] = self.train_reg_history
        data_dict["Test Reg"] = self.test_reg_history
        df = pd.DataFrame(data_dict)
        fig = px.line(
            df,
            x="Epoch",
            y=[col for col in df.columns if col != "Epoch"],
            labels={"value": "Metric", "variable": "Component"},
            title="Training and Test Metrics Over Epochs",
        )
        self.save_fig(fig=fig, name="training")

    def visualize(self):
        self.plot_predictions()
        self.plot_training()

    def get_reg_loss(self, X):
        raise NotImplementedError
