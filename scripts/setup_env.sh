#!/bin/bash
# Run ONCE on Sol to create the conda env. Takes ~5 min.
#
# Usage:
#   bash scripts/setup_env.sh

set -e

module load mamba/latest

ENV_NAME="fakereview"

if mamba env list | grep -q "^${ENV_NAME} "; then
    echo "Env '${ENV_NAME}' already exists. Activating."
else
    echo "Creating env '${ENV_NAME}' (Python 3.11)..."
    mamba create -n "${ENV_NAME}" python=3.11 -y
fi

source activate "${ENV_NAME}"

echo "Installing PyTorch (CUDA 12.1)..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "Installing remaining requirements..."
pip install -r requirements.txt

echo ""
echo "Verifying CUDA visibility..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "Setup complete. Activate with: source activate ${ENV_NAME}"
