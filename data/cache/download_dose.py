import os
import numpy as np
import urllib.request
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd

# User-defined hyperparameters
N_GENES = 500

file_path = "sciplex3.h5ad"

# Load into AnnData
adata = sc.read_h5ad(file_path)

# Verify step
print(adata)
print("\nAvailable metadata columns:")
product_name = adata.obs["product_name"]

drugs = ["Fluorouracil", "Dasatinib", "Lapatinib"]
dose_column = "dose"

# Ensure output directories exist
os.makedirs("img", exist_ok=True)

for drug in drugs:
    # Subset AnnData for the current drug
    adata_drug = adata[adata.obs["product_name"] == drug]

    # Compute variance of each gene across cells
    # .X may be sparse; convert to dense for variance calculation
    gene_vars = np.var(adata_drug.X.toarray() if hasattr(adata_drug.X, "toarray") else adata_drug.X, axis=0)

    # Identify the N_GENES most variable genes
    top_gene_idx = np.argsort(gene_vars)[-N_GENES:]
    top_gene_vars = gene_vars[top_gene_idx]

    # Print minimum, maximum, and quartile variances for the selected genes
    min_var = np.min(top_gene_vars)
    q1_var = np.percentile(top_gene_vars, 25)
    median_var = np.percentile(top_gene_vars, 50)
    q3_var = np.percentile(top_gene_vars, 75)
    max_var = np.max(top_gene_vars)
    print(f"{drug} variance stats:")
    print(f"  Min: {min_var:.6f}")
    print(f"  25th percentile (Q1): {q1_var:.6f}")
    print(f"  Median (Q2): {median_var:.6f}")
    print(f"  75th percentile (Q3): {q3_var:.6f}")
    print(f"  Max: {max_var:.6f}")

    # Create data matrix with selected genes and dose information
    gene_matrix = adata_drug.X[:, top_gene_idx]
    if hasattr(gene_matrix, "toarray"):
        gene_matrix = gene_matrix.toarray()
    df = pd.DataFrame(gene_matrix, columns=adata_drug.var_names[top_gene_idx])
    df[dose_column] = adata_drug.obs[dose_column].values

    # Save to CSV
    csv_path = f"{drug}.csv"
    df.to_csv(csv_path, index=False)
