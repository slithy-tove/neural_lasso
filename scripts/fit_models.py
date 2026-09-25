from .utils import fit_models
import numpy as np
from torch.optim import Adam

kwargs = {"model_name" : ["neural_lasso"],
          "dataset_name" : ["housing"],
          "sparsity" : [3],
          "batch_size" : [None],
          "lambda_min" : [0],
          "lambda_max" : [1],
          "n_lambda" : [10],
          "reg_type" : ["l1", "l2", "group", "group_squared"],
          "init_lr" : [1e-3],
          "path_lr" : [1e-3],
          "init_epochs" : [10000], 
          "path_epochs" : [1000],   
          "hdim" : [128],
          "hnum" : [3]}

fit_models(**kwargs)
