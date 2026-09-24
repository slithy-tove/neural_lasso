from .src import LassoNetClassifier, LassoNetRegressor
from data import load_data
import torch
from functools import partial

class LassoNet:
    def __init__(self, dataset_name, sparsity, hdim, hnum, bottleneck = None, **kwargs):
        self.dataset = load_data(dataset_name)
        self.sparsity = sparsity
        self.hidden_dims = hnum * [hdim]
        self.ds_type = self.dataset.ds_type
        self.input_dim = self.dataset.input_dim

        # TODO: define and initialize an MLP which maps from input_dim through hidden_dims, to self.output_dim, where self.output_dim = 1 if ds_type = "reg", and if ds_type = "cls" output_dim is self.dataset.n_cls

    def fit(self, n_epochs, lr, batch_size = None, optim = torch.optim.SGD, M = 10):
        """
        Fit model to some data.
        """
        # partially intialize optimizer with learning rate
        lr_init, lr_path = lr
        optim_init, optim_path = optim
        optim_init = partial(optim_init, lr = lr_init)
        optim_path = partial(optim_path, lr = lr_path)
        if self.ds_type == "reg":
            self.model = LassoNetRegressor(hidden_dims = self.hidden_dims, n_iters = n_epochs, batch_size = batch_size, optim = optim, M = M)
        elif self.ds_type == "cls":
            self.model = LassoNetClassifier(hidden_dims = self.hidden_dims, n_iters = n_epochs, batch_size = batch_size, optim = optim, M = M)     

        # extract X_train, X_test, y_train, y_test from the SparseData object self.dataset
        X_train = self.dataset.X_train
        X_test = self.dataset.X_test
        y_train = self.dataset.y_train
        y_test = self.dataset.y_test

        self.hist = self.model.path(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)
        # find the selected set of variables

