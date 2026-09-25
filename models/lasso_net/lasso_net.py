from .src import LassoNetClassifier, LassoNetRegressor
from models import MLP, SparseEstimator
from data import load_data
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import partial


class LassoNet(SparseEstimator):
    def __init__(self, dataset_name, sparsity, hdim, hnum, bottleneck=None, **kwargs):
        super().__init__()
        self.dataset = load_data(dataset_name)
        self.sparsity = sparsity
        self.hidden_dims = tuple(hnum * [hdim])
        self.ds_type = self.dataset.ds_type
        self.input_dim = self.dataset.input_dim

        # define output dimension based on problem type
        if self.ds_type == "reg":
            self.output_dim = 1
        elif self.ds_type == "cls":
            self.output_dim = getattr(self.dataset, "n_cls", None)
        else:
            raise ValueError(f"Unsupported ds_type: {self.ds_type}")

        # initialize an MLP that maps input_dim -> hidden_dims -> output_dim
        self.mlp = MLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            ds_type=self.ds_type,
            n_cls=self.output_dim if self.ds_type == "cls" else None,
        )

        # determine bottleneck size
        self.bottleneck = bottleneck if bottleneck is not None else sparsity

        self.update_header(
            model_name = "lasso_net",
            dataset_name=dataset_name,
            sparsity=sparsity,
            hdim=hdim,
            hnum=hnum,
        )

    def fit(self, init_epochs, path_epochs, init_lr, path_lr, batch_size=None, M=10, **kwargs):
        """
        Fit model to some data.
        """
        # partially initialize optimizer with learning rate
        epochs_tuple = (init_epochs, path_epochs)
        optim_init = partial(torch.optim.Adam, lr=init_lr)
        optim_path = partial(torch.optim.SGD, lr=path_lr)
        optim_tuple = (optim_init, optim_path)

        if self.ds_type == "reg":
            self.model = LassoNetRegressor(
                hidden_dims=self.hidden_dims,
                n_iters=epochs_tuple,
                batch_size=batch_size,
                optim=optim_tuple,
                M=M,
            )
        elif self.ds_type == "cls":
            self.model = LassoNetClassifier(
                hidden_dims=self.hidden_dims,
                n_iters=epochs_tuple,
                batch_size=batch_size,
                optim=optim_tuple,
                M=M,
            )

        # extract X_train, X_test, y_train, y_test from the SparseData object self.dataset
        X_train = self.dataset.X_train
        X_test = self.dataset.X_test
        y_train = self.dataset.y_train
        y_test = self.dataset.y_test

        self.hist = self.model.path(X=X_train, y=y_train, X_val=X_test, y_val=y_test, return_state_dicts = True)

        # find the selected set of variables
        selected_over_time = np.array([hi.selected for hi in self.hist])  # (n_steps, input_dim), boolean
        n_selected = np.sum(selected_over_time, axis=1)  # (n_steps,)

        # header logging (mirroring NeuralLasso)
        self.update_header(
            init_epochs=init_epochs,
            path_epochs=path_epochs,
            init_lr=init_lr,
            path_lr=path_lr,
            M=M,
            dataset=self.dataset.__class__.__name__,
            sparsity=self.sparsity,
            hdim=self.hidden_dims[0] if self.hidden_dims else None,
            hnum=len(self.hidden_dims),
            bottleneck=self.bottleneck,
        )

        # Replicate the workflow from NeuralLasso: find first index achieving desired sparsity,
        # retrain an MLP on those features, and produce predictions & weight plots.
        if np.any(n_selected == self.sparsity):
            first_idx = np.where(n_selected == self.sparsity)[0][0]
            self.selected_idx = np.where(selected_over_time[first_idx])[0]

            # Subset training and test data to selected features
            self.X_train_sub = X_train[:, self.selected_idx]
            self.X_test_sub = X_test[:, self.selected_idx]

            # Retrain a plain MLP on the selected features
            self.refit_mlp = MLP(
                input_dim=self.sparsity,
                hidden_dims=self.hidden_dims,
                ds_type=self.ds_type,
                n_cls=self.dataset.n_cls if self.ds_type == "cls" else None,
            )
            self.refit_mlp._train(
                X_train=self.X_train_sub,
                y_train=y_train,
                X_test=self.X_test_sub,
                y_test=y_test,
                lr=init_lr,
                optim=torch.optim.Adam,
                batch_size=batch_size,
                n_epochs=init_epochs,
            )
        else:
            print("Warning: Desired sparsity not achieved; skipping refit and visualizations.")

    def plot_predictions(self):
        """
        Plot predictions vs ground truth for the refitted MLP on selected features.
        """
        if not hasattr(self, "refit_mlp"):
            print("Warning: Model was not refitted; skipping predictions plot.")
            return

        X_train_tensor = torch.from_numpy(self.X_train_sub).float()
        X_test_tensor = torch.from_numpy(self.X_test_sub).float()

        train_pred = self.refit_mlp(X_train_tensor).detach().cpu().numpy()
        test_pred = self.refit_mlp(X_test_tensor).detach().cpu().numpy()

        # calculate MSE for train and test
        train_mse = np.mean((self.dataset.y_train - train_pred.squeeze()) ** 2)
        test_mse = np.mean((self.dataset.y_test - test_pred.squeeze()) ** 2)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Train", "Test"))
        fig.add_trace(
            go.Scatter(x=self.dataset.y_train, y=train_pred, mode="markers", name="Train"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=self.dataset.y_test, y=test_pred, mode="markers", name="Test"),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="True", row=1, col=1)
        fig.update_xaxes(title_text="True", row=1, col=2)
        fig.update_yaxes(title_text="Predicted", row=1, col=1)
        fig.update_yaxes(title_text="Predicted", row=1, col=2)
        fig.update_layout(
            title_text=f"Predictions vs Ground Truth (Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f})"
        )
        self.save_fig(fig=fig, name="predictions")

    def plot_weights(self):
        """
        Plot average gradient magnitudes for selected features.
        """
        if not hasattr(self, "refit_mlp"):
            print("Warning: Model was not refitted; skipping weights plot.")
            return

        X_train_tensor = torch.from_numpy(self.X_train_sub).float()
        X_train_tensor.requires_grad_(True)
        output = self.refit_mlp(X_train_tensor).sum()
        grads = torch.autograd.grad(output, X_train_tensor, retain_graph=False)[0]
        avg_grad = grads.abs().mean(dim=0).detach().cpu().numpy()
        labels = [self.dataset.feature_names[i] for i in self.selected_idx]
        fig = go.Figure(data=[go.Bar(x=labels, y=avg_grad)])
        fig.update_layout(title_text="Average Gradient Magnitudes for Selected Features")
        self.save_fig(fig=fig, name="weights")

    def plot_traces(self):
        """
        Plot evolution of L2 regularization weights over lambda values.
        """
        lambda_range = np.array([hi.lambda_ for hi in self.hist])
        weight_history = np.array([hi.state_dict["skip.weight"] for hi in self.hist])  # (n_hist, 1, input_dim)
        weight_history = weight_history[:, 0, :] # (n_hist, input_dim)

        fig = go.Figure()
        for i in range(weight_history.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=lambda_range,
                    y=weight_history[:, i],
                    mode="lines",
                    name=self.dataset.feature_names[i],
                )
            )
        fig.update_layout(
            title_text="L2 Regularization Weights Evolution",
            xaxis_title="Lambda",
            yaxis_title="Weight",
            legend_title="Feature",
        )
        self.save_fig(fig=fig, name="traces")

    def visualize(self):
        """
        Call all plot_* methods to generate visualizations.
        """
        self.plot_predictions()
        self.plot_weights()
        self.plot_traces()
