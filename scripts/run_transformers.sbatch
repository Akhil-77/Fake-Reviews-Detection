#!/bin/bash
#SBATCH --job-name=fr_transformers
#SBATCH --account=class_cse572spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/transformers_%j.out
#SBATCH --error=logs/transformers_%j.err

set -e
mkdir -p logs

module load mamba/latest
source activate fakereview

cd "${SLURM_SUBMIT_DIR}"
python transformer_models.py
