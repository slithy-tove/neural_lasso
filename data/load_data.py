"""
load_data
A helper function for conveniently loading entire datasets.
"""
from pathlib import Path
import pandas as pd
from data import SparseData

root_dir = Path("data") / "cache"
# dictionary containing the input variable names for each dataset
input_vars = {
    "boston": [
        "crim",
        "zn",
        "indus",
        "chas",
        "nox",
        "rm",
        "age",
        "dis",
        "rad",
        "tax",
        "ptratio",
        "b",
        "lstat",
    ],
    "abalone": ["ln", "dm", "ht", "wh", "shd", "vs", "shl"],
    "buildings": ["rc", "sa", "wa", "ra", "oh", "ga"],
    "prostate": ["lcavol", "lweight", "age", "lbph", "lcp"],
    "mouse": pd.read_csv(root_dir / "mouse.csv").columns.drop(
        ["cell_id", "time"]
    ),
    "organs": pd.read_csv(root_dir / "organs.csv").columns.drop(
        ["cell_id", "time"]
    ),
    "stress": pd.read_csv(root_dir / "stress.csv").columns.drop(
        ["cell_id", "p_mito"]
    ),
    "Fluorouracil": pd.read_csv(root_dir / "Fluorouracil.csv").columns.drop("dose"),
    "Dasatinib": pd.read_csv(root_dir / "Dasatinib.csv").columns.drop("dose"),
    "Lapatinib": pd.read_csv(root_dir / "Lapatinib.csv").columns.drop("dose"),
}

# dictionary containing the response variable name for each dataset
resp_var = {
    "boston": "medv",
    "abalone": "r",
    "buildings": "hl",
    "prostate": "lpsa",
    "mouse": "time",
    "organs": "time",
    "stress": "p_mito",
    "Fluorouracil": "dose",
    "Dasatinib": "dose",
    "Lapatinib": "dose",
}

def local_load_data(name):
    """
    Load a SparseData object with path name + ".csv" and the appropriate features
    """
    input_names = input_vars[name]
    response_name = resp_var[name]
    data_path = root_dir / (name + ".csv")
    return SparseData(data_path, input_names, response_name)

LASSONET_NAMES = {"MNIST", "MNIST-Fashion", "MICE", "COIL", "ISOLET", "Activity"}
def load_data(name):
    """
    Load a SparseData object with path name + ".csv" and the appropriate features
    """
    if name in LASSONET_NAMES:
        lassonet_load_data(name)
