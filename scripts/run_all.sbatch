#!/bin/bash
# Run the entire training pipeline in a single job.
# Use this if you want fire-and-forget. Total wall time ~3-4 hrs.
#
# Submit with:
#   sbatch scripts/run_all.sbatch

#SBATCH --job-name=fr_all
#SBATCH --account=class_cse572spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/all_%j.out
#SBATCH --error=logs/all_%j.err

set -e
mkdir -p logs

module load mamba/latest
source activate fakereview

cd "${SLURM_SUBMIT_DIR}"

# ── A100-friendly hyperparameters (override teammate's CPU-laptop defaults)
export DEEP_EPOCHS=5
export DEEP_BATCH_SIZE=256
export TRANSFORMER_EPOCHS=3
export TRANSFORMER_BATCH_SIZE=32
export TRANSFORMER_TRAIN_LIMIT=0          # 0 = use all 486K rows
export TRANSFORMER_TEST_LIMIT=0           # 0 = use all 122K rows
export TRANSFORMER_MAX_SEQ_LEN=128
export RF_TRAIN_SAMPLE_SIZE=0             # 0 = train RF on full dataset

echo "=========================================="
echo "  Step 1/4 — Data preparation"
echo "=========================================="
python data_prep.py

echo ""
echo "=========================================="
echo "  Step 2/4 — Classical models (5 baselines)"
echo "=========================================="
python classical_models.py

echo ""
echo "=========================================="
echo "  Step 3/4 — LSTM / BiLSTM"
echo "=========================================="
python deep_models.py

echo ""
echo "=========================================="
echo "  Step 4/4 — BERT / RoBERTa"
echo "=========================================="
python transformer_models.py

echo ""
echo "✅  All models trained. Artifacts in ./artifacts/"
