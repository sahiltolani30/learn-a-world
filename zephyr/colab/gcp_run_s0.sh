#!/usr/bin/env bash
# Zephyr Stage-0 launcher for a GCP Deep Learning VM (A100/L4).
# Run INSIDE the VM via SSH. Assumes: PyTorch preinstalled (DLVM image), CUDA GPU.
#
# Usage:
#   bash zephyr/colab/gcp_run_s0.sh /path/to/ML   [optional: extra run_s0 args]
set -euo pipefail

REPO_DIR="${1:?usage: bash zephyr/colab/gcp_run_s0.sh /path/to/ML [extra args]}"
shift || true
cd "$REPO_DIR"

DATA_DIR="${ZEPHYR_DATA_DIR:-$REPO_DIR/zephyr_cloud/data}"
CKPT_DIR="${ZEPHYR_CKPT_DIR:-$REPO_DIR/zephyr_cloud/checkpoints}"
REPORT="${ZEPHYR_REPORT:-$REPO_DIR/zephyr_cloud/reports/zephyr_stage0_report.json}"
STEPS="${ZEPHYR_STEPS:-50000}"
FRAMES="${ZEPHYR_FRAMES:-250000}"
NUM_ENVS="${ZEPHYR_NUM_ENVS:-8}"

mkdir -p "$DATA_DIR" "$CKPT_DIR" "$(dirname "$REPORT")"

echo "[gcp] python: $(python3 -V), torch: $(python3 -c 'import torch; print(torch.__version__)')"
python3 - <<'EOF'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible"
p = torch.cuda.get_device_name(0)
print(f"[gcp] GPU: {p}")
EOF

if ! python3 -c "import crafter, gymnasium" 2>/dev/null; then
  echo "[gcp] installing crafter + gymnasium..."
  pip install -q "crafter>=1.8.0" "gymnasium>=0.29.1"
fi

# Idempotent + resumable: rerun this script any number of times.
exec python3 zephyr/colab/run_s0.py \
  --data-dir "$DATA_DIR" \
  --ckpt-dir "$CKPT_DIR" \
  --report "$REPORT" \
  --frames "$FRAMES" --steps "$STEPS" --num-envs "$NUM_ENVS" --device cuda \
  "$@"
