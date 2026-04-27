#!/bin/bash
#SBATCH --job-name=fr_dataprep
#SBATCH --account=class_cse572spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/dataprep_%j.out
#SBATCH --error=logs/dataprep_%j.err

set -e
mkdir -p logs

module load mamba/latest
source activate fakereview

cd "${SLURM_SUBMIT_DIR}"
python data_prep.py
