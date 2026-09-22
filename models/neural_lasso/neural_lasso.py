import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import math
from copy import deepcopy
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
        self.refit_bottleneck_weight = nn.Linear(self.bottleneck, self.bottleneck)
        self.refit_mlp = deepcopy(self.mlp)
        self.spectrum_history = []

    def forward(self, X):
        """
        X: torch.Tensor of shape (n_obs, input_dim)
        """
        if self.is_refit:
            X_selected = X[:, self.selected_idx]
            X_b = self.refit_bottleneck_weight(X_selected)
            return self.refit_mlp(X_b).squeeze(-1)
        else:
            X_b = self.bottleneck_weight(X)
            return self.mlp(X_b).squeeze(-1)

    def get_reg_loss(self, X):
        grad = calc_grad(self, X)  # (n_obs, input_dim)
        X_b = self.bottleneck_weight(X) # (n_obs, bottleneck)
        b_grad = calc_grad(self.mlp, X_b) # (n_obs, bottleneck)
        n_obs = X.shape[0]
        if self.reg_type == "l1":
            reg = grad.abs().sum() / n_obs
        elif self.reg_type == "l2":
            reg = grad.norm(dim=0, p=2).sum() / math.sqrt(n_obs)
        elif self.reg_type == "new1":
            part1 = self.bottleneck_weight.weight.norm(dim = 1, p = 2).sum()
            part2 = b_grad.norm(dim = 1, p = 2).mean()
            reg = part1 + part2
        elif self.reg_type == "new2":
            part1 = self.bottleneck_weight.weight.norm(dim = 1, p = 2).sum()
            part2 = b_grad.norm(dim = 0, p = 2).sum() / math.sqrt(n_obs)     
            reg = part1 + part2
        elif self.reg_type == "p_reg":
            part1 = self.bottleneck_weight.weight.norm(dim = 1, p = 2).sum()
            part2 = b_grad.norm(dim = 1, p = self.reg_p).mean()
            reg = part1 + part2

        return self.lambda_reg * reg

    def get_grad_spectrum(self, X):
        """
        If the gradients on the output of the bottleneck neurons from data point $x^{(\ell)}$ is $h^{(\ell)}$, this return the eigenvalues of the matrix
        $$
        A = \sum_{\ell}h^{(\ell)}(h^{(\ell)})^{T}
        $$
        """
        n_obs = X.shape[0]
        X_b = self.bottleneck_weight(X) # (n_obs, bottleneck)
        grads = calc_grad(self.mlp, X_b).detach().numpy() # (n_obs, bottleneck)
        A = np.einsum("ni,nj->ij", grads, grads) / n_obs # (bottleneck, bottleneck)
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

        # fit extra to first lambda
        long_kwargs = kwargs.copy()
        long_kwargs["n_epochs"] = 10 * kwargs["n_epochs"]

        # add which type of $\ell^{p}$ regularization we are using, if applicable
        if "reg_p" in kwargs:
            self.reg_p = kwargs["reg_p"]

        # PART 1: Fit over each lambda in the series
        for lam in self.lambda_vals:
            print()
            print(f"Training with lambda={lam}")
            self.lambda_reg = lam
            lam_kwargs = long_kwargs if lam == self.lambda_vals[0] else kwargs
            MLP.fit(self, **lam_kwargs)
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
        self.is_refit = True
        self.lambda_reg = 0
        MLP.fit(self, **long_kwargs)

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
        # compute additional visualizations
        X_b = self.bottleneck_weight(self.X_tensor)
        grads = calc_grad(self.mlp, X_b).detach().cpu().numpy()  # (n_obs, bottleneck)
        weight_matrix = self.bottleneck_weight.weight.detach().cpu().numpy()  # (bottleneck, input_dim)

        # create subplots: line plot, heatmap, and gradient distributions
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Gradient Spectrum", "Bottleneck Weights Heatmap", "Gradient Distributions per Bottleneck"),
            specs=[[{}, {}], [{"type": "xy"}, {"type": "scene"}]]
        )

        # line plot of eigenvalues over lambda
        for i in range(self.bottleneck):
            fig.add_trace(
                go.Scatter(
                    x=self.lambda_vals,
                    y=self.spectrum_history[:, i],
                    mode="lines",
                    name=f"Eigenvalue {i+1}"
                ),
                row=1,
                col=1,
            )

        # heatmap of bottleneck weights
        fig.add_trace(
            go.Heatmap(
                z=weight_matrix,
                x=self.dataset.feature_names,
                y=[f"Bottleneck {i}" for i in range(self.bottleneck)],
                colorscale="Viridis"
            ),
            row=1,
            col=2,
        )

        # histograms of gradients for each bottleneck neuron
        for i in range(self.bottleneck):
            fig.add_trace(
                go.Histogram(
                    x=grads[:, i],
                    name=f"Bottleneck {i}",
                    nbinsx=30,
                    opacity=0.75,
                    showlegend=False
                ),
                row=2,
                col=1,
            )

        # assert sparsity is 3 and add 3D scatter of gradients
        assert self.sparsity == 3, "Sparsity must be 3 for 3D scatter plot"
        fig.add_trace(
            go.Scatter3d(
                x=grads[:, 0],
                y=grads[:, 1],
                z=grads[:, 2],
                mode="markers",
                marker=dict(size=1, opacity=0.1),
                name="Gradient Point Cloud"
            ),
            row=2,
            col=2,
        )

        # axis titles
        fig.update_xaxes(title_text="Lambda", row=1, col=1)
        fig.update_yaxes(title_text="Eigenvalue", row=1, col=1)
        fig.update_xaxes(title_text="Input Feature", row=1, col=2)
        fig.update_yaxes(title_text="Bottleneck", row=1, col=2)
        fig.update_xaxes(title_text="Gradient Value", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)

        fig.update_layout(
            title_text="Gradient Spectrum and Bottleneck Analyses",
            height=800,
            scene = dict(aspectmode = "data")
        )

        self.save_fig(fig=fig, name="spectrum")

    def visualize(self):
        super().visualize()
        self.plot_traces()
        self.plot_weights()
        self.plot_spectrum()
