import time
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from models import MLP, SparseEstimator
from models.utils import HistoryItem
from data import load_data


class NeuralLasso(MLP, SparseEstimator):
    def __init__(
        self,
        dataset_name,
        sparsity,
        hdim,
        hnum,
        bottleneck=None,
        **extra_kwargs,
    ):
        SparseEstimator.__init__(self)

        # Load dataset and set basic attributes
        self.dataset = load_data(dataset_name)
        self.ds_type = self.dataset.ds_type
        self.input_dim = self.dataset.input_dim
        self.n_cls = getattr(self.dataset, "n_cls", None)

        params = {"model_name" : "neural_lasso",
                  "dataset_name": dataset_name,
                  "sparsity": sparsity,
                  "hdim": hdim,
                  "hnum": hnum,
                  "bottleneck": bottleneck}
        self.update_header(**params)

        # extract X_train, y_train, X_test, y_test from dataset, save to self
        self.X_train = self.dataset.X_train
        self.y_train = self.dataset.y_train
        self.X_test = self.dataset.X_test
        self.y_test = self.dataset.y_test

        # Determine bottleneck size
        self.sparsity = sparsity
        self.bottleneck = bottleneck if bottleneck is not None else sparsity

        # Hidden layer configuration
        self.hidden_dims = hnum * [hdim]
        self.all_dims = [self.bottleneck] + self.hidden_dims

        # Initialize parent MLP 
        MLP.__init__(
            self,
            input_dim=self.input_dim,
            hidden_dims=self.all_dims,
            ds_type=self.ds_type,
            n_cls=self.n_cls,
        )

        # Time trackers
        self.fit_time = 0.0
        self.grad_calc_time = 0.0

    def calc_grad(self, model, X):
        """
        Calculate the gradients of a model with respect to its input X.
        """
        start = time.time()
        X = X.detach().clone() if isinstance(X, torch.Tensor) else torch.as_tensor(X).float()
        X.requires_grad_()
        y_pred = model(X)
        grads, = torch.autograd.grad(y_pred.sum(), X, create_graph=True)  # same shape as X
        self.grad_calc_time += time.time() - start
        return grads

    def get_reg_loss(self, X):
        grad = self.calc_grad(self, X)  # (n_obs, input_dim)
        X_b = self.first_layer(X)  # (n_obs, bottleneck)
        b_grad = self.calc_grad(self.downstream_layers, X_b)  # (n_obs, bottleneck)
        n_obs = X.shape[0]
        if self.reg_type == "l1":
            reg = grad.abs().sum() / n_obs
        elif self.reg_type == "l2":
            reg = grad.norm(dim = 0, p = 2).sum() / math.sqrt(n_obs)
        if self.reg_type == "group":
            part1 = self.first_layer.weight.norm(dim=0, p=2).sum()
            part2 = b_grad.norm(dim=1, p=2).mean()
            reg = part1 + part2
        elif self.reg_type == "group_squared":
            part1 = (self.first_layer.weight.norm(dim=0, p=2).sum()) ** 2
            part2 = (b_grad ** 2).mean(dim = 0).sum()
            reg = part1 + part2
        else:
            reg = torch.tensor(0.0)
        return self.lambda_reg * reg

    def get_grad_spectrum(self, X):
        r"""
        If the gradients on the output of the bottleneck neurons from data point $x^{(\ell)}$ is $h^{(\ell)}$, this return the eigenvalues of the matrix
        $$
        A = \sum_{\ell}h^{(\ell)}(h^{(\ell)})^{T}
        $$
        """
        # Convert to tensor if needed
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        n_obs = X.shape[0]
        X_b = self.first_layer(X)  # (n_obs, bottleneck)
        grads = self.calc_grad(self.downstream_layers, X_b).detach().numpy()  # (n_obs, bottleneck)
        A = np.einsum("ni,nj->ij", grads, grads) / n_obs  # (bottleneck, bottleneck)
        eigvals = np.linalg.eigvalsh(A)  # sorted ascending
        return eigvals

    def fit(
        self,
        lambda_min,
        lambda_max,
        n_lambda,
        reg_type,
        init_epochs,
        path_epochs,
        init_lr,
        path_lr,
        batch_size=None,
        **extra_kwargs,
    ):
        """
        Fit model to some data.
        """
        self.lambda_min, self.lambda_max, self.n_lambda = lambda_min, lambda_max, n_lambda
        lambda_vals = np.linspace(self.lambda_min, self.lambda_max, self.n_lambda)
        self.reg_type = reg_type

        optim = torch.optim.Adam

        # Update header with all of these new parameters above
        self.update_header(
            lambda_min=self.lambda_min,
            lambda_max=self.lambda_max,
            n_lambda=self.n_lambda,
            reg_type=self.reg_type,
            init_epochs=init_epochs,
            path_epochs=path_epochs,
            init_lr=init_lr,
            path_lr=path_lr,
        )

        # PART 0: Fit with no regularization to initialize parameters
        self.lambda_reg = 0
        self.hist = []  # list of lists
        start_fit = time.time()
        self.hist.append(self._train(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            lr=init_lr,
            optim=optim,
            batch_size=batch_size,
            n_epochs=init_epochs,
        ))

        # PART 1: Fit over each lambda in the series
        for lam in lambda_vals:
            print()
            print(f"Training with lambda={lam}")
            self.lambda_reg = lam
            self.hist.append(self._train(
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                y_test=self.y_test,
                lr=path_lr,
                optim=optim,
                batch_size=batch_size,
                n_epochs=path_epochs,
            ))
        self.fit_time += time.time() - start_fit

        # Flatten history and record lambda change points
        flat_hist = []
        self.lambda_change_points = []  # indices where a new lambda segment starts
        self.lambda_values = [0] + list(lambda_vals)
        for segment in self.hist:
            self.lambda_change_points.append(len(flat_hist))
            flat_hist.extend(segment)
        self.hist = flat_hist
        self.n_hist = len(self.hist)

        # PART 2: Refit to selected `sparsity` features
        nonzero_hist = np.array([hi.nonzero for hi in self.hist])  # (n_epochs, input_dim)
        n_nonzero = nonzero_hist.sum(axis=1)  # (n_epochs,)
        self.sparsified_ok = True
        if (n_nonzero == self.sparsity).any():
            first_idx = np.where(n_nonzero == self.sparsity)[0][0]  # first regularization value for which we attain the desired sparsity
            self.selected_idx = np.where(nonzero_hist[first_idx])[0]  # (sparsity,)

            self.X_train_sub = self.X_train[:, self.selected_idx]
            self.X_test_sub = self.X_test[:, self.selected_idx]

            self.refit_mlp = MLP(
                input_dim=self.sparsity,
                hidden_dims=self.hidden_dims,
                ds_type=self.ds_type,
                n_cls=self.n_cls,
            )
            self.refit_mlp._train(
                X_train=self.X_train_sub,
                y_train=self.y_train,
                X_test=self.X_test_sub,
                y_test=self.y_test,
                lr=init_lr,
                optim=optim,
                batch_size=batch_size,
                n_epochs=init_epochs
            )
        else:
            print("Warning: No correctly sparsified index was found! Skipping refit stage.")
            self.sparsified_ok = False

    def log(self, crit, crit_test, reg, reg_test, X_train=None, X_test=None):
        """
        Log all quantities that we want to save for later.
        """
        # Use stored data if not provided
        if X_train is None:
            X_train = torch.from_numpy(self.X_train).float()
        
        # Convert tensor to numpy for get_grad_spectrum if needed
        if isinstance(X_train, torch.Tensor):
            X_train_np = X_train.detach().numpy()
        else:
            X_train_np = X_train
        
        # record input feature gradients
        grad = self.calc_grad(self, X_train)  # (n_obs, input_dim)
        input_grads = grad.abs().mean(dim=0).detach().numpy()  # (input_dim,)
        # record spectrum of gradient matrix
        spectrum = self.get_grad_spectrum(X_train_np)  # (bottleneck,)
        spectrum /= np.linalg.norm(spectrum, ord = 1) # (normalize to show relative proportions of the eigenvalues)
        # find which variables are nonzero
        thresh = 1e-3
        nonzero = input_grads >= thresh  # (input_dim,)
        
        bottleneck_weights = self.first_layer.weight.norm(dim=0, p=2).detach().numpy()  # (bottleneck,)

        return HistoryItem(
            lambda_reg = self.lambda_reg,
            input_grads=input_grads,
            crit=crit,
            crit_test=crit_test,
            reg=reg,
            reg_test=reg_test,
            spectrum=spectrum,
            nonzero=nonzero,
            bottleneck_weight=bottleneck_weights,
        )

    def plot_predictions(self):
        if not getattr(self, "sparsified_ok", True):
            print("Warning: Model was not sparsified correctly; skipping predictions plot.")
            return
        """
        make a two-paneled plot (train and test), plotting predictions vs ground truth for self.refit_mlp
        """
        # Prepare data
        X_train_tensor = torch.from_numpy(self.X_train_sub).float()
        X_test_tensor = torch.from_numpy(self.X_test_sub).float()

        # Predictions
        train_pred = self.refit_mlp(X_train_tensor).detach().cpu().numpy()
        test_pred = self.refit_mlp(X_test_tensor).detach().cpu().numpy()

        # Compute MSE for train and test
        mse_train = np.mean((train_pred.squeeze() - self.y_train.squeeze()) ** 2)
        mse_test = np.mean((test_pred.squeeze() - self.y_test.squeeze()) ** 2)

        # Plot true vs predicted
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Train", "Test"))
        fig.add_trace(
            go.Scatter(x=self.y_train, y=train_pred, mode="markers", name="Train"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=self.y_test, y=test_pred, mode="markers", name="Test"),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="True", row=1, col=1)
        fig.update_xaxes(title_text="True", row=1, col=2)
        fig.update_yaxes(title_text="Predicted", row=1, col=1)
        fig.update_yaxes(title_text="Predicted", row=1, col=2)
        fig.update_layout(title_text="Predictions vs Ground Truth")

        # Add MSE annotations atop each subplot
        fig.add_annotation(
            x=0.5,
            y=1.05,
            xref="x1",
            yref="y1",
            text=f"MSE: {mse_train:.4f}",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
        )
        fig.add_annotation(
            x=0.5,
            y=1.05,
            xref="x2",
            yref="y2",
            text=f"MSE: {mse_test:.4f}",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
        )
        self.save_fig(fig=fig, name="predictions")

    def plot_training(self):
        """
        Extract arrays of training and test loss (both crit and reg) from self.hist, plot all of them
        """
        # Extract histories
        crit = np.array([h.crit for h in self.hist])
        crit_test = np.array([h.crit_test for h in self.hist])
        reg = np.array([h.reg for h in self.hist])
        reg_test = np.array([h.reg_test for h in self.hist])

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Criterion Loss", "Regularization Loss"))
        fig.add_trace(go.Scatter(x=np.arange(self.n_hist), y=crit, mode="lines", name="Train Crit"), row=1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(self.n_hist), y=crit_test, mode="lines", name="Test Crit"), row=1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(self.n_hist), y=reg, mode="lines", name="Train Reg"), row=2, col=1)
        fig.add_trace(go.Scatter(x=np.arange(self.n_hist), y=reg_test, mode="lines", name="Test Reg"), row=2, col=1)
        fig.update_xaxes(title_text="Iteration", row=1, col=1)
        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=2, col=1)

        # Add vertical dotted lines for lambda changes
        for idx, lam in zip(self.lambda_change_points, self.lambda_values):
            fig.add_shape(
                type="line",
                x0=idx,
                x1=idx,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(dash="dot", color="gray")
            )
            fig.add_annotation(
                x=idx,
                y=1,
                yref="paper",
                text=f"λ={lam:.3g}",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(color="gray")
            )

        # Add annotation with timing information
        fig.add_annotation(
            x=0.5,
            y=1.12,
            xref="paper",
            yref="paper",
            text=f"Fit time: {self.fit_time:.2f}s, Grad calc time: {self.grad_calc_time:.2f}s",
            showarrow=False,
            xanchor="center",
            font=dict(size=12, color="black")
        )

        fig.update_layout(title_text="Training and Test Losses Across Lambda Sweep")
        self.save_fig(fig=fig, name="training")

    def plot_traces(self):
        # extract history of weight training
        weight_history = np.array([h.input_grads for h in self.hist])
        bottleneck_history = np.array([h.bottleneck_weight for h in self.hist])

        # two-paneled line plot for histories with unified legend and matching colors
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Weight History", "Bottleneck History"))
        colors = px.colors.qualitative.Plotly
        for i in range(self.input_dim):
            name = self.dataset.feature_names[i]
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=np.arange(self.n_hist),
                    y=weight_history[:, i],
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    line=dict(color=color),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            if bottleneck_history.shape[1] > i:
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(self.n_hist),
                        y=bottleneck_history[:, i],
                        mode="lines",
                        name=name,
                        legendgroup=name,
                        line=dict(color=color),
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
        fig.update_xaxes(title_text="Iteration")
        fig.update_yaxes(title_text="Weight")
        fig.update_layout(title_text="Weight and Bottleneck Evolution Over Lambda Sweep")

        # Add vertical dotted lines for lambda changes
        for idx, lam in zip(self.lambda_change_points, self.lambda_values):
            fig.add_shape(
                type="line",
                x0=idx,
                x1=idx,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(dash="dot", color="gray")
            )
            fig.add_annotation(
                x=idx,
                y=1,
                yref="paper",
                text=f"λ={lam:.3g}",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(color="gray")
            )

        self.save_fig(fig=fig, name="traces")

    def plot_weights(self):
        if not getattr(self, "sparsified_ok", True):
            print("Warning: Model was not sparsified correctly; skipping weight plot.")
            return
        grad = self.calc_grad(self.refit_mlp, self.X_train_sub)  # (n_obs, sparsity)
        avg_grad = grad.abs().mean(dim=0)  # (sparsity,)
        labels = [self.dataset.feature_names[i] for i in self.selected_idx]
        fig = go.Figure(data=[go.Bar(x=labels, y=avg_grad.detach().cpu().numpy())])
        fig.update_layout(title_text="Average Gradient Magnitudes for Selected Features")
        self.save_fig(fig=fig, name="weights")

    def plot_spectrum(self):
        # make a line plot of the value of each eigenvalue over time (spectrum_history = np.array([h.spectrum for h in self.hist])
        spectrum_history = np.array([h.spectrum for h in self.hist])  # (n_epochs, sparsity)
        fig = go.Figure()
        for idx in range(spectrum_history.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(self.n_hist),
                    y=spectrum_history[:, idx],
                    mode="lines",
                    name=f"Eigenvalue {idx+1}",
                )
            )
        fig.update_layout(
            title_text="Gradient Spectrum Evolution Across Lambda Sweep",
            xaxis_title="Iteration",
            yaxis_title="Eigenvalue",
        )

        # Add vertical dotted lines for lambda changes
        for idx, lam in zip(self.lambda_change_points, self.lambda_values):
            fig.add_shape(
                type="line",
                x0=idx,
                x1=idx,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(dash="dot", color="gray")
            )
            fig.add_annotation(
                x=idx,
                y=1,
                yref="paper",
                text=f"λ={lam:.3g}",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(color="gray")
            )

        self.save_fig(fig=fig, name="spectrum")

    def visualize(self):
        self.plot_predictions()
        self.plot_training()
        self.plot_traces()
        self.plot_weights()
        self.plot_spectrum()
