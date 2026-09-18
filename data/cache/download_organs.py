from pathlib import Path
import pandas as pd
import scanpy as sc
import scvelo as scv

sc.settings.verbosity = 1
N_TOP_GENES = 500

# Set default save target to organs.csv in the same directory as this script
DEFAULT_OUTPUT = Path(__file__).parent / "organs.csv"


def download_and_export_sc_data(
    output_csv=DEFAULT_OUTPUT, n_top_genes=N_TOP_GENES
):
    print("1. Downloading organogenesis dataset (Dentate Gyrus via scvelo)...")
    adata = scv.datasets.dentategyrus()

    print("2. Preprocessing cell counts...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print("3. Calculating continuous response variable ('time')...")
    cell_cycle_genes = ["Top2a", "Mki67", "Cdk1", "Ube2c"]
    valid_cc_genes = [g for g in cell_cycle_genes if g in adata.var_names]

    if len(valid_cc_genes) > 0:
        print(f"   Using marker genes: {valid_cc_genes}")
        # Extract underlying numpy array (.values) to avoid pandas index alignment mismatches
        response_score = adata[:, valid_cc_genes].to_df().mean(axis=1).values
    else:
        print("   Using PCA component 1 as continuous proxy trajectory...")
        sc.tl.pca(adata, svd_solver="arpack")
        response_score = adata.obsm["X_pca"][:, 0]

    print(f"4. Selecting top {n_top_genes} highly variable continuous genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()

    print("5. Formatting dataset...")
    X_df = adata.to_df()

    # Clean up gene column names for valid CSV header formatting
    X_df.columns = [col.replace("'", "") for col in X_df.columns]

    # Explicitly assign raw numpy array values to ensure 'time' receives full numeric values
    X_df["time"] = (
        response_score.values
        if hasattr(response_score, "values")
        else response_score
    )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"6. Saving to {output_path.resolve()}...")
    X_df.to_csv(output_path, index_label="cell_id")

    print("\nDataset successfully saved!")
    print(
        f"Shape: {X_df.shape[0]} rows (cells) x {X_df.shape[1]} columns ({X_df.shape[1]-1} predictors + 1 response)"
    )
    print(f"Null values in 'time': {X_df['time'].isna().sum()}")


if __name__ == "__main__":
    download_and_export_sc_data(
        output_csv=DEFAULT_OUTPUT, n_top_genes=N_TOP_GENES
    )
