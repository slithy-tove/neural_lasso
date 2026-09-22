from .utils import fit_models
import numpy as np

kwargs = {"model_name" : ["neural_lasso"],
          "dataset_name" : ["boston"],
          "sparsity" : [3],
          "n_epochs" : [10000],
          "batch_size" : [None],
          "lambda_min" : [0],
          "lambda_max" : [1],
          "n_lambda" : [20],
          "reg_type" : ["p_reg"],
          "reg_p" : [0, 0.25, 0.5, 1, 1.5, 2],
          "lr" : [1e-3],
          "hdim" : [128],
          "hnum" : [3]}

fit_models(**kwargs)
