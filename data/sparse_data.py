"""
SparseData
A standardized class used to store data which can help with developing the SLAB algorithm.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from tqdm import tqdm

class SparseData:
    def __init__(self, data_path, input_names, response_name):
        """
        Initialize a new SparseData object, loading from a single CSV file.
        Arguments:
        data_path: str, location of the csv containing all variables.
        input_names: list of str, names of the input variables (columns).
        response_name: str, name of the response variable (column).
        """
        # Read the full data
        df = pd.read_csv(data_path)
        # Split into X and y
        X_raw = df[input_names].values  # (n_obs, input_dim)
        y_raw = df[response_name].values.ravel()  # (n_obs,)
        # Drop columns with zero variance
        variances = np.var(X_raw, axis=0)
        nonzero_mask = variances != 0
        if not np.all(nonzero_mask):
            X_raw = X_raw[:, nonzero_mask]
            input_names = [name for keep, name in zip(nonzero_mask, input_names) if keep]
        # Scale to mean 0 and variance 1
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        self.X = scaler_X.fit_transform(X_raw)
        self.y = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()
        self.feature_names = np.array(input_names)
        self.n_obs, self.input_dim = self.X.shape

    def fit_lm(self, features = None):
        """
        Fit and save a linear model predicting self.X from self.y, with the option to only use a subset of the features.
        Inputs:
        features: NoneType or boolean array of shape (input_dim,). If specified, limits which features we can use.
        Returns: coefficients and MSE of this model.
        """
        if features is not None:
            X_sub = self.X[:, features]
        else:
            X_sub = self.X
        model = LinearRegression(fit_intercept=False)
        model.fit(X_sub, self.y)
        predictions = model.predict(X_sub)
        mse = mean_squared_error(self.y, predictions)
        return model.coef_, predictions, mse

    def visualize_lm(self, features = None, suffix = ""):
        """
        Make graphics showing the performance of a given linear model on this dataset.
        """
        coef, predictions, mse = self.fit_lm(features)
        os.makedirs("img", exist_ok=True)

        # Predictions scatter plot
        plt.scatter(self.y, predictions)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Predictions vs True (MSE={mse:.4})")
        plt.tight_layout()
        plt.savefig(f"img/predictions{suffix}.png")
        plt.close()

        # Weights bar chart
        if features is not None:
            included_indices = np.where(features)[0]
            filtered_coef = coef  # coef already corresponds to included features
            filtered_names = self.feature_names[included_indices]
            colors = ['blue'] * len(included_indices)
        else:
            filtered_coef = coef
            filtered_names = self.feature_names
            colors = ['blue'] * self.input_dim
        x_pos = np.arange(len(filtered_coef))
        plt.bar(x_pos, filtered_coef, color=colors)
        plt.xticks(x_pos, filtered_names, rotation=45, ha='right')
        plt.ylabel("Coefficient")
        plt.title("Weights of Linear Model")
        plt.tight_layout()
        plt.savefig(f"img/weights{suffix}.png")
        plt.close()

    def best_subset(self, sparsity, suffix = ""):
        """
        Use brute-force to find the subset of input features which best explains the response.
        Inputs:
        sparsity: int, how many features to use.
        Returns: list of feature indices (subset) that minimizes MSE.
        """
        best_mse = float('inf')
        best_features = None
        pbar = tqdm(combinations(range(self.input_dim), sparsity), desc="Iterating Over Combinations")
        for combo in pbar:
            features = np.zeros(self.input_dim, dtype=bool)
            features[list(combo)] = True
            _, _, mse = self.fit_lm(features)
            if mse < best_mse:
                best_mse = mse
                best_features = features
        self.visualize_lm(best_features, suffix=f"_best{suffix}")
        feature_idx = np.where(best_features)[0].astype("int")
        return self.feature_names[feature_idx]