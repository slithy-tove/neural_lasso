import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from .utils import HistoryItem

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, ds_type, n_cls=None):
        """
        A generic MLP class with logging and regularization capabilities.
        Inputs:
        input_dim: int, dimension of the input data
        hidden_dims: List[int], the number of neurons in each hidden layer.
        ds_type: str, "reg" or "cls", whether we are solving a regression or classification problem.
        n_cls: NoneType or int, how many classes we are classifying to (only required if ds_type == "cls")
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.ds_type = ds_type
        self.n_cls = n_cls

        if len(hidden_dims) < 2:
            raise ValueError("hidden_dims must contain at least two elements")

        # First layer (input -> first hidden)
        self.first_layer = nn.Linear(input_dim, hidden_dims[0])

        # Downstream layers (remaining hidden layers + output)
        downstream = [nn.ReLU()]
        prev_dim = hidden_dims[0]
        for h in hidden_dims[1:]:
            downstream.append(nn.Linear(prev_dim, h))
            downstream.append(nn.ReLU())
            prev_dim = h
        out_dim = n_cls if ds_type == "cls" else 1
        downstream.append(nn.Linear(prev_dim, out_dim))
        self.downstream_layers = nn.Sequential(*downstream)

    def forward(self, X):
        x = self.first_layer(X)
        y_pred = self.downstream_layers(x)  # (n_obs, out_dim)
        return y_pred.squeeze(-1)

    def _train(self, X_train, y_train, X_test, y_test, optim, batch_size, lr, n_epochs):
        hist = []
        # Prepare data
        X_train_tensor = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y_train_tensor = torch.tensor(
            np.asarray(y_train),
            dtype=torch.long if self.ds_type == "cls" else torch.float32,
        )
        X_test_tensor = torch.tensor(np.asarray(X_test), dtype=torch.float32)
        y_test_tensor = torch.tensor(
            np.asarray(y_test),
            dtype=torch.long if self.ds_type == "cls" else torch.float32,
        )

        if batch_size is None:
            train_loader = [(X_train_tensor, y_train_tensor)]
        else:
            train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Loss function
        criterion = nn.CrossEntropyLoss() if self.ds_type == "cls" else nn.MSELoss()
        optimizer = optim(self.parameters(), lr=lr)

        for _ in tqdm(range(n_epochs), desc="Training"):
            self.train()
            train_crit_sum = 0.0
            train_reg_sum = 0.0
            batch_count = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = self(xb)
                crit = criterion(preds, yb)
                reg = self.get_reg_loss(xb)
                loss = crit + reg
                loss.backward()
                optimizer.step()
                train_crit_sum += crit.item()
                train_reg_sum += reg.item()
                batch_count += 1
            avg_crit = train_crit_sum / batch_count if batch_count else 0.0
            avg_reg = train_reg_sum / batch_count if batch_count else 0.0

            self.eval()
            preds_test = self(X_test_tensor)
            crit_test = criterion(preds_test, y_test_tensor)
            reg_test = self.get_reg_loss(X_test_tensor)
            hist.append(
                self.log(
                    crit=avg_crit,
                    crit_test=crit_test.item(),
                    reg=avg_reg,
                    reg_test=reg_test.item(),
                    X_train=X_train_tensor,
                    X_test=X_test_tensor,
                )
            )
        return hist

    def get_reg_loss(self, X):
        """
        Calculate the regularization part of the loss function (can be modified in child classes).
        """
        return 0.0

    def log(self, crit, crit_test, reg, reg_test, X_train=None, X_test=None):
        """
        Log information relevant to the model at this epoch (can be modified in child classes).
        """
        return None
