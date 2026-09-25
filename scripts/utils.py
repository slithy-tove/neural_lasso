import itertools
from models import MLP, NeuralLasso, LassoNet
from data import load_data
from time import time

MODELS = {"mlp" : MLP,
          "neural_lasso" : NeuralLasso,
          "lasso_net" : LassoNet}

def make_iter(**kwargs):
    """
    An iterable over combinations of keywords.

    Yields dictionaries mapping each keyword to one of its possible values,
    covering the Cartesian product of all provided value lists.
    """
    kw = list(kwargs.keys())
    v = list(kwargs.values())
    n_kw = len(kw)
    for lst in v:
        assert isinstance(lst, list), "Each value must be a list"
    combos = itertools.product(*v)
    for combo in combos:
        yield {kw[i]: combo[i] for i in range(n_kw)}

def fit_model(**kwargs):
    """
    Fit a model from the SparseEstimator class to a given dataset.
    """
    model_name = kwargs["model_name"]
    # instantiate model
    model = MODELS[model_name](**kwargs)
    # fit model
    model.fit(**kwargs)
    # save results
    model.visualize()
    model.save_header()

def fit_models(**kwargs):
    """
    Fit a range of models over a range of different datasets, with a range of hyperparameter combinations. Each keyword argument in kwargs should be a list of possible values (even if just a list of length 1).
    """
    for combo in make_iter(**kwargs):
        combo_str = ", ".join([f"{k} = {combo[k]}" for k in combo.keys()])
        print("Running:", combo_str)
        start = time()
        fit_model(**combo)
        end = time()
        print(f"Finished in time {end - start:.4e}s")
        print()
