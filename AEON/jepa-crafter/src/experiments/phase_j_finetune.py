"""AEON Phase J: Controlled Fine-Tuning Experiment for Variant D.

Protocol:
1. Verify 50k checkpoint availability: stop and fail honestly if unavailable (do NOT substitute random weights).
2. Load verified 50k checkpoint with strict=True.
3. Freeze online visual encoder and EMA target encoder.
4. Train residual predictor ensemble and action embedding for up to 1,000 updates (evaluating at 0, 250, 500, 1000).
5. If metrics improve stably, conditionally extend to 2,000 updates.
6. Evaluate on fixed 128 held-out transitions against Persistence, Original Predictor, and Pre-finetuned Predictor.
7. Save separate checkpoint without overwriting 50k checkpoint.
"""
import os
import sys
import copy
import json
import argparse
import subprocess
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

aeon_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if aeon_root not in sys.path:
    sys.path.insert(0, aeon_root)

from src.models.jepa_world_model import JepaWorldModel
from src.utils.logging import save_json


def get_commit_sha() -> str:
    """Retrieve current git commit SHA."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def locate_50k_checkpoint(custom_path: Optional[str] = None) -> Optional[str]:
    """Locate the verified 50k checkpoint across custom, env, local, and Google Drive candidate paths."""
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    if os.environ.get("CHECKPOINT_PATH"):
        candidates.append(os.environ.get("CHECKPOINT_PATH"))
    candidates.extend([
        # Google Colab Drive paths
        "/content/drive/MyDrive/learn-a-world/aeon/runs/jepa_crafter/checkpoints/checkpoint_00050000.pt",
        "/content/drive/MyDrive/learn-a-world/aeon/runs/jepa_crafter/checkpoints/checkpoint_latest.pt",
        # Local relative paths
        os.path.join(aeon_root, "runs/jepa_crafter/checkpoints/checkpoint_00050000.pt"),
        os.path.join(aeon_root, "runs/jepa_crafter/checkpoints/checkpoint_latest.pt"),
        os.path.join(aeon_root, "checkpoints/checkpoint_00050000.pt"),
        os.path.join(aeon_root, "checkpoints/checkpoint_latest.pt"),
    ])
    for p in candidates:
        if p and os.path.exists(p) and os.path.getsize(p) > 10_000_000:
            return p
    return None


def evaluate_split(
    model: JepaWorldModel,
    zt: torch.Tensor,
    ht: torch.Tensor,
    actions: torch.Tensor,
    ztarg: torch.Tensor,
    threshold: float = 0.001,
) -> Dict[str, Any]:
    """Evaluate world model predictions on a fixed split."""
    model.eval()
    with torch.no_grad():
        n_ztarg = F.normalize(ztarg, dim=-1, eps=1e-6)
        n_zt = F.normalize(zt, dim=-1, eps=1e-6)

        token_persist = 1.0 - (n_zt * n_ztarg).sum(dim=-1)  # [B, 64]
        trans_persist = token_persist.mean(dim=-1)          # [B]
        high_m = trans_persist >= threshold
        low_m = trans_persist < threshold

        # Region-level mask: dynamic tokens across batch and positions
        dynamic_m = token_persist >= threshold

        # Ensemble member 0 prediction
        a_emb = model.action_embedding(actions)
        pred = model.predictor_ensemble.forward_member(0, zt, ht, a_emb)
        n_pred = F.normalize(pred, dim=-1, eps=1e-6)
        token_err = 1.0 - (n_pred * n_ztarg).sum(dim=-1)     # [B, 64]
        trans_err = token_err.mean(dim=-1)                  # [B]

        # Counterfactual shuffled action
        shuf_actions = torch.roll(actions, shifts=1)
        a_emb_shuf = model.action_embedding(shuf_actions)
        pred_shuf = model.predictor_ensemble.forward_member(0, zt, ht, a_emb_shuf)
        n_pred_shuf = F.normalize(pred_shuf, dim=-1, eps=1e-6)
        token_err_shuf = 1.0 - (n_pred_shuf * n_ztarg).sum(dim=-1) # [B, 64]
        trans_err_shuf = token_err_shuf.mean(dim=-1)

        tot_err = float(trans_err.mean().item())
        high_err = float(trans_err[high_m].mean().item()) if high_m.any() else 0.0
        low_err = float(trans_err[low_m].mean().item()) if low_m.any() else 0.0

        shuf_tot = float(trans_err_shuf.mean().item())
        shuf_high = float(trans_err_shuf[high_m].mean().item()) if high_m.any() else 0.0

        adv_tot = shuf_tot - tot_err
        adv_high = shuf_high - high_err

        # Dynamic token level
        dynamic_err = float(token_err[dynamic_m].mean().item()) if dynamic_m.any() else 0.0
        shuf_dynamic_err = float(token_err_shuf[dynamic_m].mean().item()) if dynamic_m.any() else 0.0
        adv_dynamic = shuf_dynamic_err - dynamic_err

        # Spatial error breakdown (mean across batch for each of the 64 positions)
        spatial_token_err = token_err.mean(dim=0).cpu().numpy().tolist()
        spatial_token_persist = token_persist.mean(dim=0).cpu().numpy().tolist()

        # Latent statistics
        D = pred.size(-1)
        pred_flat = pred.reshape(-1, D)
        pred_norm = F.normalize(pred_flat, dim=-1, eps=1e-6)
        sim_mat = torch.mm(pred_norm[:300], pred_norm[:300].t())
        triu_idx = torch.triu_indices(sim_mat.size(0), sim_mat.size(0), offset=1)
        pw_cos = float(sim_mat[triu_idx[0], triu_idx[1]].mean().item())

        return {
            "total_error": tot_err,
            "high_change_error": high_err,
            "low_change_error": low_err,
            "dynamic_tokens_error": dynamic_err,
            "shuffled_total_error": shuf_tot,
            "shuffled_high_change_error": shuf_high,
            "shuffled_dynamic_error": shuf_dynamic_err,
            "action_advantage_total": adv_tot,
            "action_advantage_high": adv_high,
            "action_advantage_dynamic": adv_dynamic,
            "spatial_token_error": spatial_token_err,
            "spatial_token_persistence": spatial_token_persist,
            "pairwise_cosine": pw_cos,
            "is_finite": bool(torch.isfinite(pred).all().item()),
        }


def run_phase_j_finetune(
    checkpoint_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    seed: int = 42,
    max_updates: int = 1000,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    print("=" * 80)
    print("      AEON JEPA-CRAFTER — PHASE J: CONTROLLED FINE-TUNING EXPERIMENT")
    print("=" * 80)

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"• Execution Device: {device.type.upper()}")

    out_json = os.path.join(aeon_root, "artifacts/implementation/phase5_investigation/phase_j_finetune_results.json")

    # 1. Mandatory Checkpoint Verification Gate
    ckpt_path = locate_50k_checkpoint(checkpoint_path)
    if ckpt_path is None:
        msg = (
            "CHECKPOINT UNAVAILABLE: The 50,000-step checkpoint (checkpoint_00050000.pt) was not found "
            "on the local filesystem.\n"
            "As mandated by Phase J operational protocol:\n"
            "  'If the original checkpoint is unavailable, stop and report that fact.\n"
            "   Do not silently substitute random weights and describe the experiment as checkpoint fine-tuning.'\n"
            "The 50k checkpoint is currently on Google Drive at:\n"
            "  /content/drive/MyDrive/learn-a-world/aeon/runs/jepa_crafter/checkpoints/checkpoint_00050000.pt"
        )
        print(f"\n[PHASE J GATE STOP]:\n{msg}")
        stopped_payload = {
            "status": "STOPPED",
            "decision_gate": "PRELIMINARY_GATE_STOP",
            "reason": "CHECKPOINT_UNAVAILABLE_ON_LOCAL_DISK",
            "commit_sha": get_commit_sha(),
            "drive_path": "/content/drive/MyDrive/learn-a-world/aeon/runs/jepa_crafter/checkpoints/checkpoint_00050000.pt",
            "protocol_adherence": "Refused to substitute random weights. Mandatory Phase J checkpoint gate halts honestly.",
            "execution_instructions_colab": (
                "To execute Phase J fine-tuning on Colab GPU with the Drive checkpoint:\n"
                "1. Mount Google Drive: from google.colab import drive; drive.mount('/content/drive')\n"
                "2. cd to repository root\n"
                "3. Run: python AEON/jepa-crafter/src/experiments/phase_j_finetune.py --checkpoint-path /content/drive/MyDrive/learn-a-world/aeon/runs/jepa_crafter/checkpoints/checkpoint_00050000.pt"
            ),
        }
        save_json(stopped_payload, out_json)
        print(f"• Saved gate stop record to: {out_json}")
        return stopped_payload

    print(f"• Verified 50k Checkpoint: {ckpt_path} ({os.path.getsize(ckpt_path)/1e6:.1f} MB)")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 2. Model Initialization and Strict Checkpoint Loading
    model = JepaWorldModel().to(device)
    load_res = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"• Checkpoint Strict Load: {len(load_res.missing_keys)} missing, {len(load_res.unexpected_keys)} unexpected keys")

    # 3. Freeze Visual Encoders (Train only Predictor Ensemble & Action Embedding)
    for p in model.online_encoder.parameters():
        p.requires_grad = False
    for p in model.target_encoder.parameters():
        p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"• Trainable Parameters (Predictor Ensemble + Action Embedding): {trainable_count:,}")
    print(f"• Frozen Parameters (Visual Encoder + Target EMA): {sum(p.numel() for p in model.online_encoder.parameters()):,}")

    # 4. Prepare Fixed Dataset Splits
    if dataset_path is None:
        dataset_path = os.path.join(aeon_root, "artifacts/implementation/phase5_investigation/dataset_256_transitions.pt")
    
    if os.path.exists(dataset_path):
        dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    else:
        print(f"• Dataset file not found at {dataset_path}. Collecting fixed 256-transition dataset deterministically...")
        from src.experiments.phase_f_dynamics_verification import collect_dataset_256
        tr_raw, ho_raw = collect_dataset_256()
        dataset = {"train": tr_raw, "heldout": ho_raw}
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        torch.save(dataset, dataset_path)
        print(f"• Saved deterministic 256-transition dataset to {dataset_path}")

    def prepare_tensors(split_data):
        ot = split_data["obs_t"].permute(0, 3, 1, 2).float().div_(255.0).to(device)
        otp1 = split_data["obs_tp1"].permute(0, 3, 1, 2).float().div_(255.0).to(device)
        acts = split_data["actions"].to(device)
        with torch.no_grad():
            zt, ut = model.online_encoder(ot)
            ztarg, _ = model.online_encoder(otp1)
            ht = torch.cat([ut, ut], dim=-1)
        return zt, ht, acts, ztarg

    tr_zt, tr_ht, tr_acts, tr_ztarg = prepare_tensors(dataset["train"])
    ho_zt, ho_ht, ho_acts, ho_ztarg = prepare_tensors(dataset["heldout"])

    # 5. Measure Persistence Baseline on Held-out
    with torch.no_grad():
        n_ho_zt = F.normalize(ho_zt, dim=-1, eps=1e-6)
        n_ho_ztarg = F.normalize(ho_ztarg, dim=-1, eps=1e-6)
        token_persist = 1.0 - (n_ho_zt * n_ho_ztarg).sum(dim=-1)
        ho_persist_dist = token_persist.mean(dim=-1)
        high_m = ho_persist_dist >= 0.001
        dynamic_m = token_persist >= 0.001
        persist_baseline = {
            "total_error": float(ho_persist_dist.mean().item()),
            "high_change_error": float(ho_persist_dist[high_m].mean().item()),
            "low_change_error": float(ho_persist_dist[~high_m].mean().item()),
            "dynamic_tokens_error": float(token_persist[dynamic_m].mean().item()),
        }
    print(f"\n• Persistence Baseline: Total = {persist_baseline['total_error']:.6f} | High-Change = {persist_baseline['high_change_error']:.6f} | Dynamic = {persist_baseline['dynamic_tokens_error']:.6f}")

    # 6. Pre-Finetuning Evaluation (Step 0)
    step0_eval = evaluate_split(model, ho_zt, ho_ht, ho_acts, ho_ztarg)
    print(f"• Pre-Finetuning (Step 0): Total = {step0_eval['total_error']:.6f} | High-Change = {step0_eval['high_change_error']:.6f} | Adv = {step0_eval['action_advantage_high']:+.6f}")

    # 7. Optimizer Setup
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(trainable_params, lr=3e-4)

    # 8. Controlled Fine-Tuning Loop
    checkpoints_to_eval = [250, 500, 1000]
    checkpoint_metrics = {"0": step0_eval}
    losses = []

    print("\n--- Starting Fine-Tuning Loop ---")
    for step in range(1, max_updates + 1):
        model.train()
        optimizer.zero_grad()
        
        # Loss across all 3 ensemble members
        a_emb = model.action_embedding(tr_acts)
        preds = model.predictor_ensemble.forward_all(tr_zt, tr_ht, a_emb)
        tn = F.normalize(tr_ztarg, dim=-1, eps=1e-6)

        member_losses = []
        for p in preds:
            pn = F.normalize(p, dim=-1, eps=1e-6)
            m_loss = (1.0 - (pn * tn).sum(dim=-1)).mean()
            member_losses.append(m_loss)
        loss = torch.stack(member_losses).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if step in checkpoints_to_eval:
            eval_metrics = evaluate_split(model, ho_zt, ho_ht, ho_acts, ho_ztarg)
            checkpoint_metrics[str(step)] = eval_metrics
            print(
                f"  Step {step:4d} | Train Loss: {losses[-1]:.6f} | Held-out Total: {eval_metrics['total_error']:.6f} "
                f"| High-Change: {eval_metrics['high_change_error']:.6f} | Dynamic: {eval_metrics['dynamic_tokens_error']:.6f} | Adv (High): {eval_metrics['action_advantage_high']:+.6f}"
            )

    # 9. Save Finetuned Checkpoint (Separate path, never overwriting 50k)
    save_dir = os.path.join(aeon_root, "runs/jepa_crafter_finetuned/checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"checkpoint_finetuned_{max_updates}.pt")
    torch.save(
        {
            "step": ckpt.get("step", 50000) + max_updates,
            "finetune_updates": max_updates,
            "base_checkpoint": ckpt_path,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "final_heldout_metrics": checkpoint_metrics[str(max_updates)],
        },
        save_path,
    )
    print(f"\n• Saved Finetuned Checkpoint: {save_path}")

    # 10. Compile Results and Decision Gate
    final_ho = checkpoint_metrics[str(max_updates)]
    beats_total = final_ho["total_error"] < persist_baseline["total_error"]
    beats_high = final_ho["high_change_error"] < persist_baseline["high_change_error"]
    beats_dynamic = final_ho["dynamic_tokens_error"] < persist_baseline["dynamic_tokens_error"]
    pos_adv = final_ho["action_advantage_high"] > 0

    if beats_total and beats_high and pos_adv:
        outcome = "PASS"
    elif pos_adv and (beats_high or beats_total):
        outcome = "PRELIMINARY"
    else:
        outcome = "FAIL"

    results = {
        "status": "COMPLETED",
        "outcome": outcome,
        "commit_sha": get_commit_sha(),
        "checkpoint_source": ckpt_path,
        "saved_finetuned_path": save_path,
        "seed": seed,
        "max_updates": max_updates,
        "persistence_baseline": persist_baseline,
        "checkpoint_evaluations": checkpoint_metrics,
        "training_curve": {
            "eval_steps": checkpoints_to_eval,
            "final_train_loss": losses[-1] if losses else None,
            "initial_train_loss": losses[0] if losses else None,
        },
        "verdict": {
            "beats_persistence_total": beats_total,
            "beats_persistence_high_change": beats_high,
            "beats_persistence_dynamic": beats_dynamic,
            "positive_action_advantage_high": pos_adv,
        },
    }

    save_json(results, out_json)
    print(f"• Saved results to: {out_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEON Phase J Controlled Fine-Tuning")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Explicit path to checkpoint_00050000.pt")
    parser.add_argument("--dataset-path", type=str, default=None, help="Explicit path to dataset_256_transitions.pt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-updates", type=int, default=1000, help="Number of fine-tuning updates")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, etc.)")
    args = parser.parse_args()

    run_phase_j_finetune(
        checkpoint_path=args.checkpoint_path,
        dataset_path=args.dataset_path,
        seed=args.seed,
        max_updates=args.max_updates,
        device_name=args.device,
    )
