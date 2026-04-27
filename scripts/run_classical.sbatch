#!/bin/bash
#SBATCH --job-name=fr_classical
#SBATCH --account=class_cse572spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/classical_%j.out
#SBATCH --error=logs/classical_%j.err

set -e
mkdir -p logs

module load mamba/latest
source activate fakereview

cd "${SLURM_SUBMIT_DIR}"
python classical_models.py
