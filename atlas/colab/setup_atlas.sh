#!/bin/bash
# Colab setup script for ATLAS with Google Drive integration
set -e

echo "========================================================"
echo "=== [ATLAS] Setting up Project ATLAS in Colab ==="
echo "========================================================"

DRIVE_DIR="/content/drive/MyDrive/learn-a-world"
ATLAS_DRIVE_DIR="$DRIVE_DIR/atlas"

# 1. Check Google Drive Mount & create ATLAS subfolders
if [ -d "$DRIVE_DIR" ]; then
    echo "[Drive] Google Drive is mounted at $DRIVE_DIR"
    echo "[Drive] Setting up dedicated ATLAS subfolder at $ATLAS_DRIVE_DIR..."
    mkdir -p "$ATLAS_DRIVE_DIR/shards"
    mkdir -p "$ATLAS_DRIVE_DIR/runs"
    mkdir -p "$ATLAS_DRIVE_DIR/checkpoints"
    mkdir -p "$ATLAS_DRIVE_DIR/results"
    echo "[Drive] Dedicated ATLAS folders created successfully."
else
    echo "[Drive Warning] $DRIVE_DIR not detected."
    echo "Mount Google Drive in Colab before running campaigns if persistent storage is desired."
fi

# 2. Dependencies
echo "[Setup] Installing requirements..."
pip install --quiet --upgrade pip
pip install --quiet crafter gymnasium pytest stable-baselines3

# 3. Setup local working dirs
mkdir -p checkpoints results data/atlas atlas_runs

# 4. Checkpoint sync (copy champion checkpoint if available on Drive)
if [ -f "$DRIVE_DIR/checkpoints/jepa_crafter_v3_state_readout_v2.pt" ]; then
    echo "[Setup] Champion checkpoint found on Drive: $DRIVE_DIR/checkpoints/jepa_crafter_v3_state_readout_v2.pt"
fi

# 5. Check GPU
echo "[Setup] Checking compute hardware..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo "========================================================"
echo "=== [ATLAS] Colab Environment Setup Complete ==="
echo "========================================================"
