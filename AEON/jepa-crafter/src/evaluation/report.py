"""Report generator synthesizing metrics, validation gates, and benchmark results."""
import os
import sys
import json
from typing import Dict, Any, List, Optional
import torch
import crafter

from src.utils.logging import save_json


class ReportGenerator:
    """Generates comprehensive final_report.md and exports all required benchmark artifacts."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

    def write_environment_manifest(self, filepath: Optional[str] = None) -> str:
        """Write environment_manifest.txt with exact versions of dependencies and hardware."""
        if filepath is None:
            filepath = os.path.join(self.run_dir, "environment_manifest.txt")

        manifest = [
            f"Python: {sys.version}",
            f"PyTorch: {torch.__version__}",
            f"CUDA Available: {torch.cuda.is_available()}",
        ]
        if torch.cuda.is_available():
            manifest.append(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            manifest.append(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        elif torch.backends.mps.is_available():
            manifest.append("Device: Apple MPS")
        else:
            manifest.append("Device: CPU")

        manifest.append(f"Crafter version: {getattr(crafter, '__version__', 'unknown')}")
        manifest.append(f"Crafter location: {crafter.__file__}")

        content = "\n".join(manifest) + "\n"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

    def generate_final_report(
        self,
        config: Dict[str, Any],
        heldout_metrics: Dict[str, Any],
        action_metrics: Dict[str, Any],
        achievement_summary: Dict[str, Any],
        diagnostics: Dict[str, Any],
        output_filename: str = "final_report.md",
    ) -> str:
        """Generate markdown final report adhering to Section 18 specifications."""
        filepath = os.path.join(self.run_dir, output_filename)

        report = f"""# JEPA-Crafter World Model & Agent — Final Research Report

## 1. Executive Summary
- **Primary Objective:** Build a from-scratch, action-conditioned JEPA world model that learns from raw visual interaction (64×64 RGB), retains experience across episodes, and utilizes its learned model for latent planning.
- **Architectural Grounding:** 100% learned parameters initialized randomly, anonymous opaque action tokens (0..16), stop-gradient EMA target visual encoder, multi-horizon cosine prediction loss, and Cross-Entropy Method (CEM) latent planning.
- **Official Crafter Benchmark Score:** **{achievement_summary.get('crafter_score', 0.0):.2f}%** across {achievement_summary.get('num_episodes', 0)} evaluated episodes.
- **Mean Reward:** {achievement_summary.get('mean_reward', 0.0):.3f}
- **Mean Episode Length:** {achievement_summary.get('mean_length', 0.0):.1f} steps

---

## 2. World Model Learning & Validation Outcomes

### 2.1 Held-Out Prediction (Test B)
- **1-Step Cosine Distance (Model):** {heldout_metrics.get('model_cosine_distance', {}).get(1, 0.0):.4f}
- **1-Step Cosine Distance (Persistence Baseline):** {heldout_metrics.get('persistence_cosine_distance', {}).get(1, 0.0):.4f}
- **Improvement over Persistence:** {heldout_metrics.get('pct_improvement_over_persistence', 0.0):.2f}%
- **Multi-Horizon Cosine Distances:**
  - Horizon 1: {heldout_metrics.get('model_cosine_distance', {}).get(1, 0.0):.4f}
  - Horizon 2: {heldout_metrics.get('model_cosine_distance', {}).get(2, 0.0):.4f}
  - Horizon 4: {heldout_metrics.get('model_cosine_distance', {}).get(4, 0.0):.4f}
  - Horizon 8: {heldout_metrics.get('model_cosine_distance', {}).get(8, 0.0):.4f}
- **Baseline Comparison (Horizon 1):**
  - Persistence: {heldout_metrics.get('persistence_cosine_distance', {}).get(1, 0.0):.4f}
  - No-Action Ablation: {heldout_metrics.get('no_action_cosine_distance', {}).get(1, 0.0):.4f}
  - Shuffled-Action Ablation: {heldout_metrics.get('shuffled_action_cosine_distance', {}).get(1, 0.0):.4f}

### 2.2 Action-Conditioning Verification (Test C)
- **Action-Conditioned Error:** {action_metrics.get('mean_action_conditioned_error', 0.0):.4f}
- **No-Action Control Error:** {action_metrics.get('mean_no_action_error', 0.0):.4f}
- **Shuffled-Action Control Error:** {action_metrics.get('mean_shuffled_action_error', 0.0):.4f}
- **Action Prediction Divergence:** {action_metrics.get('action_prediction_divergence', 0.0):.6f}
- **Test C Gate Status:** {'PASS' if action_metrics.get('test_c_passed') else 'FAIL / INVESTIGATE'}

### 2.3 Representation Geometry & Anti-Collapse (Test D)
- **Effective Rank / Participation Ratio:** {diagnostics.get('effective_rank', 0.0):.2f}
- **Feature Standard Deviation:** {diagnostics.get('latent_std', 0.0):.4f}
- **Mean Channel Std:** {diagnostics.get('mean_channel_std', 0.0):.4f}
- **Collapsed Channel Fraction:** {diagnostics.get('collapsed_channel_fraction', 0.0):.2%}
- **Average Pairwise Cosine Similarity:** {diagnostics.get('avg_pairwise_cosine_sim', 0.0):.4f}
- **Collapse Warning:** {'ACTIVE' if diagnostics.get('collapse_warning') else 'NONE (Healthy manifold)'}

---

## 3. Official Crafter Achievement Spectrum
Per-achievement success rates (percentage of episodes where achievement was unlocked):

| Achievement | Success Rate (%) |
|---|---|
"""
        rates = achievement_summary.get("achievement_rates", {})
        for ach, val in rates.items():
            report += f"| `{ach}` | {val:.2f}% |\n"

        report += """
---

## 4. Architectural Specifications & Integrity Invariants
1. **Random Initialization:** All encoder and predictor weights initialized randomly; zero pretrained weights.
2. **Zero Information Leakage:** No `info` dictionary keys, achievement strings, inventory mappings, or semantic entity IDs ever entered the model, replay buffer, or planner.
3. **Anonymous Action Tokens:** The model interfaces exclusively with anonymous tokens `0..16`. Token-to-environment permutation is randomly generated at run initialization.
4. **EMA Target Encoder:** Stop-gradient EMA target branch with momentum increasing from 0.996 to 0.999.
5. **Autoregressive Multi-Step Rollout:** Recursive latent rollout predicting future spatial latent grids conditioned on action embeddings.

---

## 5. Artifact Manifest
- `environment_manifest.txt`
- `final_report.md`
- `achievement_rates.csv`
- `heldout_prediction.json`
- `action_conditioning.json`
- `metrics.jsonl` / `metrics.csv`
"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)

        return filepath
