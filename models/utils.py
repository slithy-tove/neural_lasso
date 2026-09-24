import numpy as np
from dataclasses import dataclass, field

@dataclass
class HistoryItem:
    lambda_reg: float = None
    input_grads: np.ndarray = None
    crit: float = None
    reg: float = None
    crit_test: float = None
    reg_test: float = None
    spectrum: np.ndarray = None
    nonzero: np.ndarray = None
    bottleneck_weight: np.ndarray = None
