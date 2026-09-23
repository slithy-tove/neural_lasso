import os
import numpy as np
from data import SparseData

def load_data(name):
    path = os.path.join("data", "cache", f"{name}.npz")
    with np.load(path, allow_pickle=True) as data:
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        names = data["var_names"]
        ds_type = data["ds_type"]
    return SparseData(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, input_names=names, ds_type = ds_type)
