import torch

def calc_grad(model, X):
    """
    Calculate the gradients of a model with respect to its input X.
    """
    X = X.detach().clone() if isinstance(X, torch.Tensor) else torch.as_tensor(X).float()
    X.requires_grad_()
    y_pred = model(X)
    grads, = torch.autograd.grad(y_pred.sum(), X, create_graph = True) # same shape as X
    return grads

def reset_parameters(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
