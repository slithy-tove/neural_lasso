#!/bin/bash

#SBATCH --partition=devel
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:15:00
#SBATCH --output=output.txt

module load uv 
uv sync
uv add pertpy
uv run python download_dose.py
date
