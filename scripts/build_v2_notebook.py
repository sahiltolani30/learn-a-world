"""
Generates notebooks/02_train_v2_colab.ipynb for Google Colab execution.
"""

import json
from pathlib import Path

def create_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Learn-a-World v2: Full Training & Promotion Campaign\n",
                "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahiltolani30/learn-a-world/blob/v2/notebooks/02_train_v2_colab.ipynb)\n",
                "\n",
                "**A JEPA-style latent world model with learned reward-value planning, ~300k starting dataset, multi-step unroll consistency, and locked promotional evaluation.**\n",
                "\n",
                "### Campaign Workflow:\n",
                "1. **Drive Persistence**: Mounts Google Drive so all ~300k transitions, checkpoints, and scorecards survive session disconnects.\n",
                "2. **Stage 1 (Dataset Collection)**: Collects 300,000 sequence transitions (150,000 `structured_random` + 150,000 `teacher_hybrid`).\n",
                "3. **Stage 2 (Task Heads Training)**: Trains `EventHead` (Focal BCE), `HealthHead` (Huber), `QEnsemble` (TwoHot 51 bins), and `PolicyPrior` (BC) for 5,000 steps on GPU.\n",
                "4. **Stage 3 (Phase 3 Joint Rounds)**: 3-round collect-retrain loop with multi-step imagination unrolls and dynamic consistency.\n",
                "5. **Stage 4 (Phase 4 Locked Comparison)**: 100-episode evaluation on fixed seeds (1000–1099) across 4 agents (`v1_hybrid`, `v1_pure`, `v2_pure`, `random`) with automatic promotion decision rule.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 1: Mount Google Drive & Environment Setup\n",
                "Mounts Google Drive to persist all shards, model checkpoints, and results permanently."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "import os\n",
                "from pathlib import Path\n",
                "\n",
                "# 1. Mount Google Drive\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "# 2. Create persistent workspace directories in Drive\n",
                "DRIVE_DIR = Path('/content/drive/MyDrive/learn-a-world')\n",
                "(DRIVE_DIR / 'checkpoints').mkdir(parents=True, exist_ok=True)\n",
                "(DRIVE_DIR / 'data' / 'sequences').mkdir(parents=True, exist_ok=True)\n",
                "(DRIVE_DIR / 'results').mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "print('Google Drive persistent directory initialized at:', DRIVE_DIR)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 2: Clone or Pull Repository on Branch `v2`"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%bash\n",
                "cd /content\n",
                "if [ ! -d \"learn-a-world\" ]; then\n",
                "  git clone https://github.com/sahiltolani30/learn-a-world.git\n",
                "fi\n",
                "cd learn-a-world\n",
                "git fetch origin\n",
                "git checkout v2\n",
                "git pull origin v2\n",
                "echo \"--> Active branch and latest commit:\"\n",
                "git log -1 --oneline\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 3: Install Dependencies & Verify GPU"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%bash\n",
                "cd /content/learn-a-world\n",
                "pip install -q -r requirements.txt\n",
                "pip install -q crafter gymnasium\n",
                "nvidia-smi\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 4: Verify Base Checkpoint & Run Pre-Flight Tests\n",
                "Ensures the base v1 checkpoint (`checkpoints/jepa_crafter.pt`) is present and passes the complete 43-test test suite."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, shutil\n",
                "from pathlib import Path\n",
                "\n",
                "# Verify / copy base v1 checkpoint\n",
                "src_ckpt = Path('/content/learn-a-world/checkpoints/jepa_crafter.pt')\n",
                "dst_ckpt = Path('/content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter.pt')\n",
                "\n",
                "if not dst_ckpt.exists() and src_ckpt.exists():\n",
                "    shutil.copyfile(src_ckpt, dst_ckpt)\n",
                "    print(f'Copied base v1 checkpoint to Drive: {dst_ckpt}')\n",
                "else:\n",
                "    print(f'Base checkpoint ready: {dst_ckpt}')\n",
                "\n",
                "# Run pre-flight verification test suite (43 tests)\n",
                "!cd /content/learn-a-world && pytest tests/ -v\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 5: Option A — Launch Full One-Click Campaign (~300k Steps)\n",
                "Executes all 4 stages automatically:\n",
                "- **Stage 1**: 300k Dataset (150k structured random exploration + 150k teacher hybrid play)\n",
                "- **Stage 2**: 5,000 Head Training Steps (EventHead + HealthHead + QEnsemble + PolicyPrior)\n",
                "- **Stage 3**: 3-Round Joint Fine-Tuning & Retrain Loop (`run_step3_rounds`)\n",
                "- **Stage 4**: 100-Episode Locked Final Comparison (seeds 1000..1099) & Promotion Decision"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%bash\n",
                "cd /content/learn-a-world\n",
                "python training/run_v2_colab_campaign.py \\\n",
                "    --drive-dir /content/drive/MyDrive/learn-a-world \\\n",
                "    --n-transitions 150000 \\\n",
                "    --head-steps 5000 \\\n",
                "    --rounds 3 \\\n",
                "    --round-transitions 30000 \\\n",
                "    --joint-steps 1000 \\\n",
                "    --eval-episodes 100\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6: Option B — Run Staged (Modular Execution)\n",
                "If you prefer running stage by stage to inspect intermediate progress, use the separate cells below:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6A: Collect 300k starting transitions (150k structured_random + 150k teacher_hybrid)\n",
                "!cd /content/learn-a-world && python training/run_v2_colab_campaign.py \\\n",
                "    --drive-dir /content/drive/MyDrive/learn-a-world \\\n",
                "    --stage collect \\\n",
                "    --n-transitions 150000\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6B: Train Task Heads for 5,000 steps\n",
                "!cd /content/learn-a-world && python training/run_v2_colab_campaign.py \\\n",
                "    --drive-dir /content/drive/MyDrive/learn-a-world \\\n",
                "    --stage heads \\\n",
                "    --head-steps 5000\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6C: Phase 3 Joint Fine-Tuning & 3-Round Collect-Retrain Loop\n",
                "!cd /content/learn-a-world && python training/run_v2_colab_campaign.py \\\n",
                "    --drive-dir /content/drive/MyDrive/learn-a-world \\\n",
                "    --stage rounds \\\n",
                "    --rounds 3 \\\n",
                "    --round-transitions 30000 \\\n",
                "    --joint-steps 1000\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6D: Phase 4 Locked 100-Episode Final Comparison & Promotion Decision\n",
                "!cd /content/learn-a-world && python training/run_v2_colab_campaign.py \\\n",
                "    --drive-dir /content/drive/MyDrive/learn-a-world \\\n",
                "    --stage final \\\n",
                "    --eval-episodes 100\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 7: Inspect Final Promotion Report & Decision Scorecard\n",
                "Displays the comparison table across all 4 agents, promotion criteria audit, and per-achievement unlock rates."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "from IPython.display import Markdown, display\n",
                "\n",
                "report_file = '/content/drive/MyDrive/learn-a-world/results/promotion_decision.md'\n",
                "if not os.path.exists(report_file):\n",
                "    report_file = '/content/learn-a-world/results/promotion_decision.md'\n",
                "\n",
                "if os.path.exists(report_file):\n",
                "    display(Markdown(open(report_file).read()))\n",
                "else:\n",
                "    print('Report not found yet. Run Step 5 or 6D first.')\n"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "02_train_v2_colab.ipynb",
                "provenance": []
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    out_file = Path("notebooks/02_train_v2_colab.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"Successfully generated notebook at: {out_file} ({len(cells)} cells)")

if __name__ == "__main__":
    create_notebook()
