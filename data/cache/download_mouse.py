import pandas as pd
import scanpy as sc

# Set logging level to reduce clutter
sc.settings.verbosity = 1
N_TOP_GENES = 500

def download_and_export_sc_data(
    output_csv="single_cell_dataset.csv", n_top_genes=200
):
    print("1. Downloading dataset (Paul et al., 2015)...")
    adata = sc.datasets.paul15()

    print("2. Preprocessing cell counts...")
    # Normalize reads per cell to 10,000 and log-transform (continuous scale)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print(f"3. Selecting top {n_top_genes} highly variable continuous genes...")
    # Filter to 10s-100s of most informative continuous gene features
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()

    print("4. Calculating continuous response variable (Diffusion Pseudotime)...")
    # Compute neighborhood graph and trajectory
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)

    # Set root cell for trajectory inference (Erythroid progenitor root)
    adata.uns["iroot"] = 0
    sc.tl.dpt(adata)

    print("5. Formatting dataset...")
    # Extract continuous predictor matrix (Genes)
    X_df = adata.to_df()

    # Clean up gene names to ensure valid CSV column headers
    X_df.columns = [col.replace("'", "") for col in X_df.columns]

    # Add the continuous response variable as a single column
    X_df["response_pseudotime"] = adata.obs["dpt_pseudotime"].values

    print(f"6. Saving to {output_csv}...")
    X_df.to_csv(output_csv, index_label="cell_id")

    print("\nDataset successfully saved!")
    print(
        f"Shape: {X_df.shape[0]} rows (cells) x {X_df.shape[1]} columns ({X_df.shape[1]-1} predictors + 1 response)"
    )
    print("\nSample Preview:")
    print(X_df.iloc[:5, -5:])


if __name__ == "__main__":
    download_and_export_sc_data(
        output_csv="single_cell_dataset.csv", n_top_genes=N_TOP_GENES
    )
