from .utils import fit_models
import numpy as np
from torch.optim import Adam

kwargs = {"model_name" : ["lasso_net", "neural_lasso"],
          "dataset_name" : ["mnist"],
          "sparsity" : [3],
          "batch_size" : [None],
          "lambda_min" : [0],
          "lambda_max" : [1],
          "n_lambda" : [10],
          "reg_type" : ["group"],
          "init_lr" : [1e-3],
          "path_lr" : [1e-3],
          "init_epochs" : [2500], 
          "path_epochs" : [250],   
          "hdim" : [128],
          "hnum" : [3]}

fit_models(**kwargs)
