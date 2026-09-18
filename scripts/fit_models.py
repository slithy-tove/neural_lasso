from .utils import fit_models

kwargs = {"model_name" : ["mlp"],
          "dataset_name" : ["boston"],
          "sparsity" : [3, 4],
          "n_epochs" : [1, 10],
          "batch_size" : [128],
          "lr" : [1e-3],
          "hdim" : [64, 128],
          "hnum" : [1]}

fit_models(**kwargs)
