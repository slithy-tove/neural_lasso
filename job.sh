#!/bin/bash

#SBATCH --partition=devel
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=01:00:00
#SBATCH --output=output.txt

module load uv 
uv sync
uv run python -m scripts.fit_models
date
