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
    def __init__(self, X_train, y_train, X_test, y_test, input_names, ds_type):
        """
        Initialize a new SparseData object, loading from provided arrays.
        Arguments:
        X_train, X_test: np.ndarrays of shape (n_obs, input_dim)
        y_train, y_test: np.ndarrays of shape (n_obs,)
        input_names: list of str, names of the input variables (columns).
        ds_type: str, "cls" or "reg", whether this is a classification or regression problem.
        """
        # Scale to mean 0 and variance 1
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        self.X_train = scaler_X.fit_transform(X_train)
        self.X_test = scaler_X.transform(X_test)
        self.ds_type = ds_type

        if ds_type == "reg":
            self.y_scaler = scaler_y
            self.y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            self.y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
        else:
            # Ravel y arrays, map classes to consecutive indices, and store number of classes
            y_train_flat = y_train.ravel()
            y_test_flat = y_test.ravel()
            unique_classes = np.unique(np.concatenate([y_train_flat, y_test_flat]))
            class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
            self.y_train = np.vectorize(class_to_idx.get)(y_train_flat)
            self.y_test = np.vectorize(class_to_idx.get)(y_test_flat)
            self.n_cls = len(unique_classes)

        self.feature_names = np.array(input_names)
        self.input_dim = self.X_train.shape[1]
