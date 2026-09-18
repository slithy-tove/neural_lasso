
from pathlib import Path
import pandas as pd
import scanpy as sc

sc.settings.verbosity = 1
N_TOP_GENES = 500

# Default save target inside data/cache/
DEFAULT_OUTPUT = Path(__file__).parent / "stress.csv"


def download_and_export_sc_data(
    output_csv=DEFAULT_OUTPUT, n_top_genes=N_TOP_GENES
):
    print("1. Downloading PBMC single-cell dataset...")
    adata = sc.datasets.pbmc3k_processed()

    print(
        "2. Extracting natural continuous stress response variable ('percent_mito')..."
    )
    # 'percent_mito' is a built-in continuous metadata variable (0.0 to 1.0)
    response_variable = adata.obs["percent_mito"].values

    print(f"3. Selecting top {n_top_genes} highly variable gene predictors...")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()

    print("4. Formatting dataset...")
    X_df = adata.to_df()

    # Clean up column headers for valid CSV formatting
    X_df.columns = [col.replace("'", "") for col in X_df.columns]

    # Assign continuous cellular stress level as the target column
    X_df["p_mito"] = response_variable

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"5. Saving to {output_path.resolve()}...")
    X_df.to_csv(output_path, index_label="cell_id")

    print("\nDataset successfully saved!")
    print(
        f"Shape: {X_df.shape[0]} rows (cells) x {X_df.shape[1]} columns ({X_df.shape[1]-1} predictors + 1 response)"
    )
    print(f"Null values in 'p_mito': {X_df['p_mito'].isna().sum()}")


if __name__ == "__main__":
    download_and_export_sc_data(
        output_csv=DEFAULT_OUTPUT, n_top_genes=N_TOP_GENES
    )
