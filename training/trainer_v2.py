"""
Head-Only Diagnostic Trainer for Learn-a-World v2 (Phase 1).
Trains EventHead and HealthHead on frozen JEPA target-encoder latents:
- Loss = 1.0 * FocalBCE(EventHead, gamma=2.0) + 0.5 * Huber(HealthHead)
- Evaluates per-achievement PR-AUC and recall@0.5 on held-out episodes
- Generates results/v2_head_validation.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from eval.metrics import CRAFTER_ACHIEVEMENTS
from experiments.utils import get_device, load_config
from models.heads import NUM_ACHIEVEMENTS
from models.jepa_v2 import JEPAv2
from training.sequence_buffer import EpisodeBuffer

def compute_binary_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Computes exact Area Under the Precision-Recall Curve (PR-AUC).
    Numerically integrates precision vs recall across all unique score thresholds.
    """
    y_true = (y_true > 0).astype(np.int32)
    positives = int(np.sum(y_true))
    if positives == 0:
        return 0.0
    if positives == len(y_true):
        return 1.0

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)

    recalls = tp / positives
    precisions = tp / (tp + fp)

    recalls = np.concatenate(([0.0], recalls))
    precisions = np.concatenate(([precisions[0]], precisions))

    # Numerical integration compatible with both NumPy 1.x (trapz) and 2.x (trapezoid)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    pr_auc = float(trapz_fn(precisions, recalls))
    return float(np.clip(pr_auc, 0.0, 1.0))

def evaluate_heads(
    model: JEPAv2,
    val_buffer: EpisodeBuffer,
    seq_len: int = 16,
    num_batches: int = 20,
    batch_size: int = 16,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Evaluates diagnostic performance on held-out validation episodes:
    - Per-achievement PR-AUC and recall@0.5
    - Macro Tier-1 PR-AUC
    - Health prediction Mean Absolute Error (MAE)
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    model.eval()

    all_event_targets: List[np.ndarray] = []
    all_event_probs: List[np.ndarray] = []
    all_health_targets: List[np.ndarray] = []
    all_health_preds: List[np.ndarray] = []

    with torch.no_grad():
        for _ in range(num_batches):
            try:
                batch = val_buffer.sample_sequences(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    mode="event_balanced",
                    device=dev,
                )
            except Exception:
                batch = val_buffer.sample_sequences(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    mode="uniform",
                    device=dev,
                )

            obs = batch["obs"]          # [B, T, 3, 64, 64]
            actions = batch["actions"]  # [B, T]
            rewards = batch["rewards"]  # [B, T]
            events = batch["events"]    # [B, T, 22]
            mask = batch["mask_before"] # [B, T, 22]

            B, T = actions.shape
            obs_flat = obs.view(B * T, 3, 64, 64)
            z_flat = model.encode(obs_flat, use_target=True)
            z = z_flat.view(B, T, -1)

            probs = model.event_head.predict_proba(z, actions, mask)
            health_pred = model.health_head(z, actions)

            event_counts = events.sum(dim=-1)
            health_target = (rewards - event_counts).clamp(-1.0, 1.0)

            all_event_targets.append(events.view(-1, NUM_ACHIEVEMENTS).cpu().numpy())
            all_event_probs.append(probs.view(-1, NUM_ACHIEVEMENTS).cpu().numpy())
            all_health_targets.append(health_target.view(-1).cpu().numpy())
            all_health_preds.append(health_pred.view(-1).cpu().numpy())

    targets_arr = np.concatenate(all_event_targets, axis=0)
    probs_arr = np.concatenate(all_event_probs, axis=0)
    health_tgt_arr = np.concatenate(all_health_targets, axis=0)
    health_pred_arr = np.concatenate(all_health_preds, axis=0)

    # Per-achievement metrics
    per_ach_metrics: Dict[str, Dict[str, float]] = {}
    tier1_achs = ["collect_wood", "collect_drink", "collect_sapling", "eat_plant", "wake_up"]
    tier1_pr_aucs: List[float] = []

    for i, ach_name in enumerate(CRAFTER_ACHIEVEMENTS):
        y_true = targets_arr[:, i]
        y_pred = probs_arr[:, i]
        pos_count = int(np.sum(y_true))

        pr_auc = compute_binary_pr_auc(y_true, y_pred)
        if pos_count > 0:
            recall_at_half = float(np.mean(y_pred[y_true > 0] >= 0.5))
        else:
            recall_at_half = 0.0

        per_ach_metrics[ach_name] = {
            "pr_auc": float(round(pr_auc, 4)),
            "recall_at_0.5": float(round(recall_at_half, 4)),
            "positive_samples": pos_count,
        }

        if ach_name in tier1_achs and pos_count > 0:
            tier1_pr_aucs.append(pr_auc)

    macro_tier1_pr_auc = float(np.mean(tier1_pr_aucs)) if tier1_pr_aucs else 0.5
    health_mae = float(np.mean(np.abs(health_pred_arr - health_tgt_arr)))

    return {
        "macro_tier1_pr_auc": float(round(macro_tier1_pr_auc, 4)),
        "health_mae": float(round(health_mae, 4)),
        "per_achievement": per_ach_metrics,
        "representation_problem": bool(macro_tier1_pr_auc < 0.5),
    }

def train_heads(
    config_path: str = "experiments/configs/crafter_v2.yaml",
    checkpoint_path: str = "checkpoints/jepa_crafter.pt",
    buffer: Optional[EpisodeBuffer] = None,
    total_steps: int = 500,
    batch_size: int = 32,
    seq_len: int = 16,
    device_override: Optional[str] = None,
    out_json_path: str = "results/v2_head_validation.json",
    out_checkpoint_path: str = "checkpoints/jepa_crafter_v2.pt",
) -> Tuple[JEPAv2, Dict[str, Any]]:
    """
    Runs diagnostic training for EventHead and HealthHead.
    """
    device = get_device(device_override)
    cfg = load_config(config_path)

    loss_cfg = cfg.get("loss", {})
    t_cfg = cfg.get("training_v2", {})

    event_weight = loss_cfg.get("event_weight", 1.0)
    health_weight = loss_cfg.get("health_weight", 0.5)
    q_weight = loss_cfg.get("q_weight", 1.0)
    prior_weight = loss_cfg.get("prior_weight", 0.5)
    focal_gamma = loss_cfg.get("focal_gamma", 2.0)
    gamma = loss_cfg.get("gamma", 0.99)
    n_step = loss_cfg.get("n_step", 5)
    target_ema = loss_cfg.get("target_ema", 0.995)
    lr_heads = t_cfg.get("lr_heads", 3e-4)

    # Initialize JEPAv2 from v1 checkpoint
    print(f"--> Initializing JEPAv2 with base checkpoint: {checkpoint_path}")
    model = JEPAv2.from_v1_checkpoint(
        checkpoint_path=checkpoint_path,
        config=cfg,
        device=device,
        freeze_base=True,
    )

    optimizer = optim.AdamW(
        list(model.event_head.parameters()) +
        list(model.health_head.parameters()) +
        list(model.q_ensemble.parameters()) +
        list(model.policy_prior.parameters()),
        lr=lr_heads,
        weight_decay=1e-4,
    )

    # Prepare data
    if buffer is None:
        raise ValueError("Buffer must be provided to train_heads.")

    train_buf, val_buf = buffer.split_train_val(val_ratio=0.10)
    if val_buf.num_episodes == 0:
        val_buf = train_buf  # Fallback if dataset is very small

    print(f"--> Beginning Head Training for {total_steps} steps (Batch: {batch_size}, SeqLen: {seq_len}, Device: {device})...")

    history = {"step": [], "loss": [], "loss_event": [], "loss_health": [], "loss_q": [], "loss_prior": []}

    model.train()
    for step in range(1, total_steps + 1):
        batch = train_buf.sample_sequences(
            batch_size=batch_size,
            seq_len=seq_len,
            mode="event_balanced",
            device=device,
        )

        obs = batch["obs"]          # [B, T, 3, 64, 64]
        actions = batch["actions"]  # [B, T]
        rewards = batch["rewards"]  # [B, T]
        dones = batch["dones"]      # [B, T]
        events = batch["events"]    # [B, T, 22]
        mask = batch["mask_before"] # [B, T, 22]

        B, T = actions.shape

        with torch.no_grad():
            obs_flat = obs.view(B * T, 3, 64, 64)
            z_flat = model.encode(obs_flat, use_target=True)
            z = z_flat.view(B, T, -1)

        # 1. EventHead loss
        event_logits = model.event_head(z, actions, mask)
        loss_event = model.event_head.compute_loss(event_logits, events, gamma=focal_gamma)

        # 2. HealthHead loss
        event_counts = events.sum(dim=-1)
        health_target = (rewards - event_counts).clamp(-1.0, 1.0)
        health_pred = model.health_head(z, actions)
        loss_health = model.health_head.compute_loss(health_pred, health_target)

        # 3. PolicyPrior BC loss
        prior_logits = model.policy_prior(z, mask)
        loss_prior = model.policy_prior.compute_loss(prior_logits, actions)

        # 4. Q-Ensemble n-step return regression
        with torch.no_grad():
            n_step_targets = torch.zeros(B, T, device=device)
            # Compute bootstrap actions at all steps
            bootstrap_actions = model.sample_prior_action(z, mask, temperature=1.0)
            target_q_vals = model.q_mean(z, bootstrap_actions, use_target=True)

            for t in range(T):
                ret = torch.zeros(B, device=device)
                discount = 1.0
                bootstrapped = False
                for k in range(n_step):
                    idx = t + k
                    if idx >= T:
                        break
                    ret = ret + discount * rewards[:, idx]
                    if dones[:, idx].any():
                        # Stop bootstrap if done
                        bootstrapped = True
                        break
                    discount *= gamma

                # Add bootstrap terminal value if sequence did not terminate
                if not bootstrapped and (t + n_step) < T:
                    ret = ret + discount * target_q_vals[:, t + n_step]

                n_step_targets[:, t] = ret

        q_logits = model.q_ensemble(z, actions)
        loss_q = model.q_ensemble.compute_loss(q_logits, n_step_targets)

        total_loss = (
            event_weight * loss_event +
            health_weight * loss_health +
            q_weight * loss_q +
            prior_weight * loss_prior
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.event_head.parameters()) +
            list(model.health_head.parameters()) +
            list(model.q_ensemble.parameters()) +
            list(model.policy_prior.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        model.update_target_q(decay=target_ema)

        if step % 50 == 0 or step == total_steps:
            print(f"    Step {step:4d}/{total_steps} | Loss: {total_loss.item():.4f} (Event: {loss_event.item():.4f}, Health: {loss_health.item():.4f}, Q: {loss_q.item():.4f}, Prior: {loss_prior.item():.4f})")
            history["step"].append(step)
            history["loss"].append(float(total_loss.item()))
            history["loss_event"].append(float(loss_event.item()))
            history["loss_health"].append(float(loss_health.item()))
            history["loss_q"].append(float(loss_q.item()))
            history["loss_prior"].append(float(loss_prior.item()))

    print("--> Running validation diagnostic evaluation...")
    val_report = evaluate_heads(
        model=model,
        val_buffer=val_buf,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
    )
    val_report["training_steps"] = total_steps
    val_report["final_loss"] = float(total_loss.item())

    # Save validation report
    out_file = Path(out_json_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(val_report, f, indent=2)
    print(f"--> Saved validation report to: {out_file}")

    # Save checkpoint
    model.save_checkpoint(out_checkpoint_path, step=total_steps, optimizer_state=optimizer.state_dict())
    print(f"--> Saved trained v2 model checkpoint to: {out_checkpoint_path}")

    return model, val_report

def train_joint(
    model: JEPAv2,
    buffer: EpisodeBuffer,
    total_steps: int = 1000,
    batch_size: int = 32,
    seq_len: int = 16,
    unroll_steps: int = 4,
    lr_base: float = 1e-4,
    lr_heads: float = 3e-4,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Joint Fine-Tuning (Phase 3):
    Unfreezes encoder and predictor at lr_base (1e-4) while training task heads at lr_heads (3e-4).
    Computes multi-step consistency losses by unrolling predictor for k=1..unroll_steps.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    model.unfreeze_base_model()
    model.to(dev)

    cfg = config or {}
    loss_cfg = cfg.get("loss", {})
    event_weight = loss_cfg.get("event_weight", 1.0)
    health_weight = loss_cfg.get("health_weight", 0.5)
    q_weight = loss_cfg.get("q_weight", 1.0)
    prior_weight = loss_cfg.get("prior_weight", 0.5)
    target_ema = loss_cfg.get("target_ema", 0.995)

    base_params = list(model.base_model.encoder.parameters()) + list(model.base_model.predictor.parameters())
    head_params = (
        list(model.event_head.parameters()) +
        list(model.health_head.parameters()) +
        list(model.q_ensemble.parameters()) +
        list(model.policy_prior.parameters()) +
        list(model.gru_belief.parameters())
    )

    optimizer = optim.AdamW([
        {"params": base_params, "lr": lr_base, "weight_decay": 1e-5},
        {"params": head_params, "lr": lr_heads, "weight_decay": 1e-4},
    ])

    print(f"--> Beginning Joint Training for {total_steps} steps (LR_base={lr_base}, LR_heads={lr_heads}, Unroll={unroll_steps})...")

    model.train()
    history = {"step": [], "loss": [], "loss_unroll": [], "loss_dyn": []}

    for step in range(1, total_steps + 1):
        batch = buffer.sample_sequences(
            batch_size=batch_size,
            seq_len=seq_len,
            mode="event_balanced",
            device=dev,
        )

        obs = batch["obs"]          # [B, T, 3, 64, 64]
        actions = batch["actions"]  # [B, T]
        rewards = batch["rewards"]  # [B, T]
        events = batch["events"]    # [B, T, 22]
        mask = batch["mask_before"] # [B, T, 22]

        B, T = actions.shape
        obs_flat = obs.view(B * T, 3, 64, 64)

        # Online encoder
        z_online = model.base_model.encoder(obs_flat).view(B, T, -1)

        # Target encoder (stop-gradient target)
        with torch.no_grad():
            z_target = model.base_model.target_encoder(obs_flat).view(B, T, -1)

        K = min(unroll_steps, T - 1)
        loss_dyn = 0.0
        loss_unroll_event = 0.0
        loss_unroll_health = 0.0

        # Multi-step imagination rollout from anchor step t=0
        z_hat = z_online[:, 0]
        for k in range(1, K + 1):
            act_prev = actions[:, k - 1]
            z_hat = model.predict_next(z_hat, act_prev)

            # 1. Latent dynamic alignment with target encoder
            loss_dyn = loss_dyn + F.mse_loss(z_hat, z_target[:, k])

            # 2. Multi-step EventHead loss on imagined latents
            event_pred = model.event_head(z_hat, actions[:, k], mask[:, k])
            loss_unroll_event = loss_unroll_event + model.event_head.compute_loss(event_pred, events[:, k])

            # 3. Multi-step HealthHead loss on imagined latents
            event_cnt = events[:, k].sum(dim=-1)
            h_tgt = (rewards[:, k] - event_cnt).clamp(-1.0, 1.0)
            h_pred = model.health_head(z_hat, actions[:, k])
            loss_unroll_health = loss_unroll_health + model.health_head.compute_loss(h_pred, h_tgt)

        loss_dyn = loss_dyn / K
        loss_unroll_event = loss_unroll_event / K
        loss_unroll_health = loss_unroll_health / K

        # Anti-collapse variance regularization on online latents
        z_flat_online = z_online.view(B * T, -1)
        std_z = torch.sqrt(z_flat_online.var(dim=0) + 1e-4)
        loss_var = torch.mean(F.relu(1.0 - std_z))

        # Standard step 0 heads & Q loss on z_online
        prior_logits = model.policy_prior(z_online, mask)
        loss_prior = model.policy_prior.compute_loss(prior_logits, actions)

        q_logits = model.q_ensemble(z_online, actions)
        with torch.no_grad():
            bootstrap_acts = model.sample_prior_action(z_online, mask, temperature=1.0)
            q_mean_targets = model.q_mean(z_online, bootstrap_acts, use_target=True)
        loss_q = model.q_ensemble.compute_loss(q_logits, q_mean_targets)

        total_loss = (
            1.0 * loss_dyn +
            1.0 * loss_var +
            event_weight * loss_unroll_event +
            health_weight * loss_unroll_health +
            q_weight * loss_q +
            prior_weight * loss_prior
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(base_params + head_params, max_norm=1.0)
        optimizer.step()

        # Update EMA networks
        model.update_base_target(decay=0.99)
        model.update_target_q(decay=target_ema)

        if step % 50 == 0 or step == total_steps:
            print(f"    Joint Step {step:4d}/{total_steps} | Loss: {total_loss.item():.4f} (Dyn: {loss_dyn.item():.4f}, UnrollEvent: {loss_unroll_event.item():.4f}, Var: {loss_var.item():.4f})")
            history["step"].append(step)
            history["loss"].append(float(total_loss.item()))
            history["loss_dyn"].append(float(loss_dyn.item()))
            history["loss_unroll"].append(float(loss_unroll_event.item()))

    return history

def run_step3_rounds(
    base_checkpoint_path: str = "checkpoints/jepa_crafter_v2.pt",
    config_path: str = "experiments/configs/crafter_v2.yaml",
    rounds: int = 3,
    transitions_per_round: int = 30_000,
    joint_steps_per_round: int = 1_000,
    eval_episodes: int = 50,
    eval_max_steps: int = 1000,
    out_dir: str = "results",
    ckpt_dir_override: Optional[str] = None,
    device_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the 3-Round Collect-Retrain Loop (Phase 3):
    For round r in 1..rounds:
      1. Collect transitions with current pure v2 planner (+ 10% random exploration).
      2. Joint fine-tune encoder, predictor, and task heads.
      3. Evaluate with eval_v2.py and track best checkpoint.
      4. Saves results to results/v2_step3_rounds.json.
    """
    from eval.eval_v2 import evaluate_v2
    from training.collect_sequences import collect_episodes

    device = get_device(device_override)
    cfg = load_config(config_path)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(ckpt_dir_override) if ckpt_dir_override else Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(f"         PHASE 3: JOINT FINE-TUNE & COLLECT-RETRAIN ({rounds} ROUNDS)")
    print("=" * 75)

    current_ckpt = base_checkpoint_path
    best_score = -1.0
    best_ckpt = str(ckpt_dir / "jepa_crafter_v2_best.pt")
    rounds_history = []

    # Initial buffer
    dataset_dir = out_path / "sequences"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    buffer = EpisodeBuffer(capacity_transitions=transitions_per_round * (rounds + 1))

    for r in range(1, rounds + 1):
        print(f"\n>>> [Round {r}/{rounds}] Collecting {transitions_per_round} transitions with planner_v2...")
        round_buf = collect_episodes(
            policy_name="planner_v2",
            n_transitions=transitions_per_round,
            out_dir=str(dataset_dir),
            chunk_size=min(5000, max(50, transitions_per_round)),
            checkpoint_path=current_ckpt,
            seed=300 + r * 10,
        )

        for ep in round_buf.episodes:
            buffer.add_episode(**ep)

        # Initialize model from base v1 checkpoint (always, then overlay v2 weights if available)
        model = JEPAv2.from_v1_checkpoint(
            checkpoint_path=base_checkpoint_path,
            config=cfg,
            device=device,
            freeze_base=False,
        )
        if Path(current_ckpt).exists() and current_ckpt != base_checkpoint_path:
            model.load_checkpoint(current_ckpt, device=device)

        # Joint fine-tune
        print(f"\n>>> [Round {r}/{rounds}] Joint fine-tuning for {joint_steps_per_round} steps...")
        train_joint(
            model=model,
            buffer=buffer,
            total_steps=joint_steps_per_round,
            batch_size=min(32, max(2, buffer.total_transitions // 16)),
            seq_len=min(16, max(4, buffer.total_transitions // 2)),
            unroll_steps=cfg.get("loss", {}).get("unroll_steps", 4),
            config=cfg,
            device=device,
        )

        round_ckpt = str(ckpt_dir / f"jepa_crafter_v2_r{r}.pt")
        model.save_checkpoint(round_ckpt, step=r * joint_steps_per_round)

        # Evaluate round
        print(f"\n>>> [Round {r}/{rounds}] Screening evaluation ({eval_episodes} eps, seeds 1000..{1000 + eval_episodes - 1})...")
        eval_res = evaluate_v2(
            checkpoint_path=round_ckpt,
            config_path=config_path,
            num_episodes=eval_episodes,
            seed_start=1000,
            max_steps=eval_max_steps,
            device_override=str(device),
        )

        round_entry = {
            "round": r,
            "transitions_collected": transitions_per_round,
            "total_dataset_transitions": buffer.total_transitions,
            "checkpoint": round_ckpt,
            "crafter_score": eval_res["crafter_score"],
            "unlocked_count": eval_res["unlocked_count"],
            "mean_survival_steps": eval_res["mean_survival_steps"],
            "mean_reward": eval_res["mean_reward"],
            "debug_decomposition": eval_res["debug_decomposition"],
        }
        rounds_history.append(round_entry)

        if eval_res["crafter_score"] > best_score:
            best_score = eval_res["crafter_score"]
            model.save_checkpoint(best_ckpt, step=r * joint_steps_per_round)
            print(f"🎉 New best checkpoint saved to: {best_ckpt} (Score: {best_score:.2f}%)")

        current_ckpt = round_ckpt

        # Save cumulative rounds log
        rounds_file = out_path / "v2_step3_rounds.json"
        with open(rounds_file, "w") as f:
            json.dump({"rounds": rounds_history, "best_score": best_score, "best_checkpoint": best_ckpt}, f, indent=2)
        print(f"--> Saved Step 3 cumulative history to: {rounds_file}")

    return {"rounds": rounds_history, "best_score": best_score, "best_checkpoint": best_ckpt}

def main():
    parser = argparse.ArgumentParser(description="Train Task Heads and Joint JEPA v2")
    parser.add_argument("--mode", type=str, default="heads", choices=["heads", "joint", "rounds"],
                        help="Training mode: heads, joint, or rounds")
    parser.add_argument("--config", type=str, default="experiments/configs/crafter_v2.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter.pt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--transitions-per-round", type=int, default=30_000)
    parser.add_argument("--joint-steps", type=int, default=1_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-json", type=str, default="results/v2_head_validation.json")
    args = parser.parse_args()

    if args.mode == "rounds":
        run_step3_rounds(
            base_checkpoint_path=args.checkpoint,
            config_path=args.config,
            rounds=args.rounds,
            transitions_per_round=args.transitions_per_round,
            joint_steps_per_round=args.joint_steps,
            eval_episodes=args.eval_episodes,
            device_override=args.device,
        )
    else:
        # Collect mini dataset if buffer not present
        from training.collect_sequences import collect_episodes
        print("--> Collecting mini dataset for trainer run...")
        buf = collect_episodes(
            policy_name="structured_random",
            n_transitions=1500,
            out_dir="data/sequences_smoke",
            chunk_size=1000,
            seed=100,
        )

        train_heads(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            buffer=buf,
            total_steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device_override=args.device,
            out_json_path=args.out_json,
        )

if __name__ == "__main__":
    main()

