import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from models import MLP
from data import load_data
from .utils import calc_grad, reset_parameters

class NeuralLasso(MLP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_refit = False
        self.use_reg = True
        self.refit_bottleneck_weight = nn.Linear(self.sparsity, self.sparsity)
        self.spectrum_history = []

    def forward(self, X):
        """
        X: torch.Tensor of shape (n_obs, input_dim)
        """
        if self.is_refit:
            X_selected = X[:, self.selected_idx]
            X_b = self.refit_bottleneck_weight(X_selected)
        else:
            X_b = self.bottleneck_weight(X)

        return self.mlp(X_b).squeeze(-1)

    def get_reg_loss(self, X):
        grad = calc_grad(self, X)  # (n_obs, input_dim)
        n_obs = X.shape[0]
        if self.reg_type == "l1":
            reg = grad.abs().sum() / n_obs
        elif self.reg_type == "l2":
            reg = grad.norm(dim=0, p=2).sum() / math.sqrt(n_obs)
        return self.lambda_reg * reg

    def get_grad_spectrum(self, X):
        """
        If the gradients on the output of the bottleneck neurons from data point $x^{(\ell)}$ is $h^{(\ell)}$, this return the eigenvalues of the matrix
        $$
        A = \sum_{\ell}h^{(\ell)}(h^{(\ell)})^{T}
        $$
        """
        X_b = self.bottleneck_weight(X) # (n_obs, sparsity)
        grads = calc_grad(self.mlp, X_b).detach().numpy() # (n_obs, sparsity)
        A = np.einsum("ni,nj->nj", grads, grads) # (sparsity, sparsity)
        eigvals = np.linalg.eigvalsh(A)  # sorted ascending
        return eigvals

    def fit(self, **kwargs):
        """
        Fit model to some data.
        """
        self.lambda_min, self.lambda_max, self.n_lambda = kwargs["lambda_min"], kwargs["lambda_max"], kwargs["n_lambda"]
        self.lambda_vals = np.linspace(self.lambda_min, self.lambda_max, self.n_lambda)
        self.reg_type = kwargs["reg_type"]
        self.weight_history = []
        self.bottleneck_history = []

        # PART 1: Fit over each lambda in the series
        for lam in self.lambda_vals:
            print()
            print(f"Training with lambda={lam}")
            self.lambda_reg = lam
            MLP.fit(self, **kwargs)
            # record input feature gradients
            grad = calc_grad(self, self.X_tensor)  # (n_obs, input_dim)
            weights = grad.abs().mean(dim = 0).detach().numpy() # (input_dim,)
            self.weight_history.append(weights) 
            # record input feature bottleneck weights
            self.bottleneck_history.append(self.bottleneck_weight.weight.abs().mean(axis=0).detach().numpy())  # (input_dim,)
            # record spectrum of gradient matrix 
            eigs = self.get_grad_spectrum(self.X_tensor) # (sparsity,)
            self.spectrum_history.append(eigs)
            
        self.weight_history = np.array(self.weight_history)
        self.bottleneck_history = np.array(self.bottleneck_history)
        self.spectrum_history = np.array(self.spectrum_history)

        # PART 2: Refit to the smallest lambda which attained the desired sparsity
        thresh = 1e-3
        num_nonzero = (self.weight_history > thresh).sum(axis=1)  # (n_lambda,)
        if num_nonzero[-1] <= self.sparsity:
            min_lambda_idx = np.min(np.where(num_nonzero <= self.sparsity)[0])
        else:
            min_lambda_idx = -1 # keep last one if it didn't sparsify
        cutoff_wt = self.weight_history[min_lambda_idx]  # (input_dim,)
        self.selected_idx = np.argsort(cutoff_wt)[-self.sparsity:]  # (sparsity,)

        # refit the model using only the selected features and no regularization
        self.mlp.apply(reset_parameters)
        self.is_refit = True
        self.lambda_reg = 0
        MLP.fit(self, **kwargs)

    def plot_traces(self):
        # two-paneled line plot for histories with unified legend and matching colors
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Weight History", "Bottleneck History"))
        colors = px.colors.qualitative.Plotly
        for i in range(self.input_dim):
            name = self.dataset.feature_names[i]
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=self.lambda_vals,
                    y=self.weight_history[:, i],
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    line=dict(color=color),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.lambda_vals,
                    y=self.bottleneck_history[:, i],
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    line=dict(color=color),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
        fig.update_xaxes(title_text="Lambda")
        fig.update_yaxes(title_text="Weight")
        fig.update_layout(title_text="Weight and Bottleneck Evolution Over Lambda Sweep")
        self.save_fig(fig=fig, name="traces")

    def plot_weights(self):
        grad = calc_grad(self, self.X_tensor) # (n_obs, sparsity)
        avg_grad = grad.abs().mean(dim = 0)[self.selected_idx] # (sparsity,)
        labels = [self.dataset.feature_names[i] for i in self.selected_idx]
        fig = go.Figure(data=[go.Bar(x=labels, y=avg_grad.detach().cpu().numpy())])
        fig.update_layout(title_text="Average Gradient Magnitudes for Selected Features")
        self.save_fig(fig=fig, name="weights")

    def plot_spectrum(self):
        # plot a line plot of how each ordered eigenvalue of the matrix evolves over training
        fig = go.Figure()
        for i in range(self.sparsity):
            fig.add_trace(go.Scatter(x=self.lambda_vals,
                                     y=self.spectrum_history[:, i],
                                     mode="lines",
                                     name=f"Eigenvalue {i+1}"))
        fig.update_xaxes(title_text="Lambda")
        fig.update_yaxes(title_text="Eigenvalue")
        fig.update_layout(title_text="Gradient Spectrum Evolution Over Lambda Sweep")
        self.save_fig(fig=fig, name="spectrum")

    def visualize(self):
        super().visualize()
        self.plot_traces()
        self.plot_weights()
        self.plot_spectrum()
