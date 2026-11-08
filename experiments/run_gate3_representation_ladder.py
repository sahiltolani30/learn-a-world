"""ORBIT Gate 3 Matched Representation Experiment Execution.

Trains and qualifies:
- Candidate A: Tabula-Rasa ORBIT Spatial Encoder (From Scratch, Conv2d, 8x8x192)
- Candidate B: Frozen Meta AI DINOv2 ViT-S/14 Backbone + Trainable Spatial Adapter (8x8x192)

Both arms are trained on identical replay data (10,000 transitions, seeds 3000..3029)
for exactly 1,000 updates under identical batching, sequence lengths, optimizer settings,
and downstream architectures. Evaluated on 20 held-out seeds (3030..3049).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from true_jepa.config import JEPAConfig, TrainConfig
from true_jepa.model import LatentFrame, TrueJEPA
from true_jepa.objectives import train_world_model_step, reconstruct_belief_prefix
from true_jepa.replay import PixelReplay


class CachedDINOv2Replay:
    """Bitwise-equivalent accelerated replay sampler for frozen DINOv2 representations."""

    def __init__(self, raw_replay: PixelReplay, token_cache: List[np.ndarray]) -> None:
        self.raw_replay = raw_replay
        self.token_cache = token_cache
        self._episodes = raw_replay._episodes
        self._frames = raw_replay._frames

    def sample_with_prefix(
        self, batch_size: int, sequence_length: int, rng: np.random.Generator
    ) -> Dict[str, torch.Tensor]:
        if not self._episodes:
            raise RuntimeError("cannot sample an empty replay")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")

        token_shape = self.token_cache[0].shape[1:]
        obs = np.zeros((batch_size, sequence_length + 1) + token_shape, dtype=np.float32)
        actions = np.zeros((batch_size, sequence_length), dtype=np.int64)
        rewards = np.zeros((batch_size, sequence_length), dtype=np.float32)
        terminated = np.zeros((batch_size, sequence_length), dtype=bool)
        truncated = np.zeros((batch_size, sequence_length), dtype=bool)
        valid = np.zeros((batch_size, sequence_length), dtype=bool)

        sampled_indices = []
        starts = []

        for row in range(batch_size):
            ep_idx = int(rng.integers(0, len(self._episodes)))
            episode = self._episodes[ep_idx]
            ep_tokens = self.token_cache[ep_idx]
            length = int(episode.actions.shape[0])
            use = min(length, sequence_length)
            start = 0 if length <= sequence_length else int(rng.integers(0, length - sequence_length + 1))
            sampled_indices.append(ep_idx)
            starts.append(start)

            obs[row, : use + 1] = ep_tokens[start : start + use + 1]
            if use < sequence_length:
                obs[row, use + 1 :] = ep_tokens[-1]
            actions[row, :use] = episode.actions[start : start + use]
            rewards[row, :use] = episode.rewards[start : start + use]
            terminated[row, :use] = episode.terminated[start : start + use]
            truncated[row, :use] = episode.truncated[start : start + use]
            valid[row, :use] = True

        max_prefix = max(starts)
        prefix_obs = np.zeros((batch_size, max_prefix + 1) + token_shape, dtype=np.float32)
        prefix_actions = np.zeros((batch_size, max_prefix), dtype=np.int64)
        prefix_rewards = np.zeros((batch_size, max_prefix), dtype=np.float32)
        prefix_lengths = np.array(starts, dtype=np.int64)

        for row, (ep_idx, start) in enumerate(zip(sampled_indices, starts)):
            if start > 0:
                ep = self._episodes[ep_idx]
                ep_tokens = self.token_cache[ep_idx]
                prefix_obs[row, : start + 1] = ep_tokens[: start + 1]
                prefix_actions[row, :start] = ep.actions[:start]
                prefix_rewards[row, :start] = ep.rewards[:start]

        return {
            "obs": torch.from_numpy(obs),
            "actions": torch.from_numpy(actions),
            "rewards": torch.from_numpy(rewards),
            "terminated": torch.from_numpy(terminated),
            "truncated": torch.from_numpy(truncated),
            "valid": torch.from_numpy(valid),
            "prefix_obs": torch.from_numpy(prefix_obs),
            "prefix_actions": torch.from_numpy(prefix_actions),
            "prefix_rewards": torch.from_numpy(prefix_rewards),
            "prefix_lengths": torch.from_numpy(prefix_lengths),
        }


def extract_dinov2_token_cache(
    raw_replay: PixelReplay, model: TrueJEPA, device: torch.device, cache_path: Path = Path("data/gate3/dinov2_token_cache.pt")
) -> List[np.ndarray]:
    if cache_path.exists():
        print(f"\nLoading cached frozen DINOv2 patch tokens from {cache_path}...")
        token_cache = torch.load(cache_path, weights_only=False)
        print(f"Loaded {len(token_cache)} cached episodes ({sum(c.shape[0] for c in token_cache)} frames).\n")
        return token_cache

    print(f"\nPre-extracting frozen DINOv2 patch tokens for {len(raw_replay._episodes)} episodes ({raw_replay._frames} frames)...")
    start = time.perf_counter()
    token_cache = []
    with torch.no_grad():
        for ep_idx, ep in enumerate(raw_replay._episodes):
            obs_tensor = torch.from_numpy(ep.obs)
            chunks = []
            for i in range(0, obs_tensor.shape[0], 64):
                chunk = obs_tensor[i : i + 64].to(device)
                x_norm = model.encoder.preprocess(chunk)
                feats = model.encoder.backbone.forward_features(x_norm)
                chunks.append(feats["x_norm_patchtokens"].cpu())
            ep_tokens = torch.cat(chunks, dim=0).numpy()
            token_cache.append(ep_tokens)
            if (ep_idx + 1) % 15 == 0 or (ep_idx + 1) == len(raw_replay._episodes):
                current_frames = sum(c.shape[0] for c in token_cache)
                print(f"  Extracted {ep_idx + 1:2d}/{len(raw_replay._episodes)} episodes ({current_frames:5d} frames)...")
    elapsed = time.perf_counter() - start
    print(f"Pre-extraction complete in {elapsed:.2f}s ({raw_replay._frames / elapsed:.1f} fps).\n")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(token_cache, cache_path)
    print(f"Saved token cache to {cache_path}")
    return token_cache


def compute_effective_rank(tokens: torch.Tensor) -> float:
    """Computes Roy-Vetterli singular value effective rank over flattened tokens [M, D]."""
    M = tokens.shape[0] * tokens.shape[1]
    D = tokens.shape[2]
    flat = tokens.detach().reshape(M, D).float()
    centered = flat - flat.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    S_sum = S.sum()
    if S_sum <= 1e-12:
        return 1.0
    p = S / S_sum
    p = p[p > 1e-12]
    entropy = -(p * torch.log(p)).sum().item()
    return float(np.exp(entropy))


def compute_active_dimensions(tokens: torch.Tensor, threshold: float = 1e-4) -> int:
    """Computes count of dimensions with channel variance above threshold."""
    M = tokens.shape[0] * tokens.shape[1]
    D = tokens.shape[2]
    flat = tokens.detach().reshape(M, D).float()
    var = flat.var(dim=0, unbiased=True)
    return int((var > threshold).sum().item())


def compute_spatial_diversity(tokens: torch.Tensor) -> float:
    """Computes average pairwise cosine similarity between spatial tokens within each frame."""
    B, N, D = tokens.shape
    normed = F.normalize(tokens.detach().float(), p=2, dim=-1)
    sim_matrix = torch.bmm(normed, normed.transpose(1, 2))
    mask = ~torch.eye(N, dtype=torch.bool, device=tokens.device).unsqueeze(0).expand(B, N, N)
    off_diag = sim_matrix[mask].reshape(B, N * (N - 1))
    return float(off_diag.mean().item())
    entropy = -(p * torch.log(p)).sum().item()
    return float(np.exp(entropy))


def compute_active_dimensions(tokens: torch.Tensor, threshold: float = 1e-4) -> int:
    """Computes count of dimensions with channel variance above threshold."""
    M = tokens.shape[0] * tokens.shape[1]
    D = tokens.shape[2]
    flat = tokens.detach().reshape(M, D).float()
    var = flat.var(dim=0, unbiased=True)
    return int((var > threshold).sum().item())


def compute_spatial_diversity(tokens: torch.Tensor) -> float:
    """Computes average pairwise cosine similarity between spatial tokens within each frame."""
    B, N, D = tokens.shape
    normed = F.normalize(tokens.detach().float(), p=2, dim=-1)
    sim_matrix = torch.bmm(normed, normed.transpose(1, 2))
    mask = ~torch.eye(N, dtype=torch.bool, device=tokens.device).unsqueeze(0).expand(B, N, N)
    off_diag = sim_matrix[mask].reshape(B, N * (N - 1))
    return float(off_diag.mean().item())


def evaluate_candidate_diagnostics(
    model: TrueJEPA,
    eval_dataset: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluates representation health and predictive utility across held-out seeds."""
    model.eval()

    all_tokens = []
    all_target_tokens = []
    drift_l2_list = []
    drift_cos_list = []

    err_model_list = []
    err_persist_list = []
    err_shuffled_list = []

    horizons = [1, 2, 4, 8]
    tf_errors: Dict[int, List[float]] = {h: [] for h in horizons}
    fr_errors: Dict[int, List[float]] = {h: [] for h in horizons}

    with torch.no_grad():
        for ep in eval_dataset:
            obs = torch.from_numpy(ep["obs"]).to(device) # [T+1, 64, 64, 3]
            actions = torch.from_numpy(ep["action"]).to(device) # [T]
            rewards = torch.from_numpy(ep["reward"]).to(device) # [T]
            T = actions.shape[0]

            if T < 10:
                continue

            # Encode all observations
            online_frames = model.encode(obs)
            target_frames = model.target_encode(obs)

            # Accumulate sample tokens for rank & variance diagnostics (subsample up to 64 frames)
            sample_indices = torch.linspace(0, T, min(64, T + 1), dtype=torch.long)
            all_tokens.append(online_frames.tokens[sample_indices].cpu())
            all_target_tokens.append(target_frames.tokens[sample_indices].cpu())

            # Online-Target Drift
            o_tok = online_frames.tokens[sample_indices]
            t_tok = target_frames.tokens[sample_indices]
            l2_drift = ((o_tok - t_tok).norm(dim=-1) / (t_tok.norm(dim=-1).clamp_min(1e-6))).mean().item()
            cos_align = F.cosine_similarity(o_tok, t_tok, dim=-1).mean().item()
            drift_l2_list.append(l2_drift)
            drift_cos_list.append(cos_align)

            # 1-Step Predictive Utility & Controls
            hidden = model.initial_belief(batch=1, device=device)
            for t in range(min(T - 1, 50)):
                curr_frame = LatentFrame(
                    tokens=online_frames.tokens[t:t+1],
                    pooled=online_frames.pooled[t:t+1],
                )
                tgt_frame = LatentFrame(
                    tokens=target_frames.tokens[t+1:t+2],
                    pooled=target_frames.pooled[t+1:t+2],
                )
                act = actions[t:t+1]
                pred = model.predict(curr_frame, act, hidden)

                # Normalized tokens
                p_norm = F.normalize(pred.tokens, p=2, dim=-1)
                t_norm = F.normalize(tgt_frame.tokens, p=2, dim=-1)
                c_norm = F.normalize(curr_frame.tokens, p=2, dim=-1)

                e_model = (p_norm - t_norm).square().mean().item()
                e_persist = (c_norm - t_norm).square().mean().item()
                err_model_list.append(e_model)
                err_persist_list.append(e_persist)

                # Shuffled Action Control
                shuf_act = (act + 7) % 17
                pred_shuf = model.predict(curr_frame, shuf_act, hidden)
                p_shuf_norm = F.normalize(pred_shuf.tokens, p=2, dim=-1)
                e_shuf = (p_shuf_norm - t_norm).square().mean().item()
                err_shuffled_list.append(e_shuf)

                # Step belief forward with true next observation
                prev_r = rewards[t:t+1]
                hidden = model.belief_step(curr_frame, act, prev_r, hidden)

            # Multi-Step Rollout Horizons
            for start_idx in range(0, min(T - 8, 40), 8):
                h_tf = model.initial_belief(batch=1, device=device)
                h_fr = h_tf.clone()
                cur_fr = LatentFrame(
                    tokens=online_frames.tokens[start_idx:start_idx+1],
                    pooled=online_frames.pooled[start_idx:start_idx+1],
                )

                for step in range(1, 9):
                    step_idx = start_idx + step - 1
                    act = actions[step_idx:step_idx+1]
                    rew = rewards[step_idx:step_idx+1]
                    cur_tf = LatentFrame(
                        tokens=online_frames.tokens[step_idx:step_idx+1],
                        pooled=online_frames.pooled[step_idx:step_idx+1],
                    )
                    tgt = LatentFrame(
                        tokens=target_frames.tokens[step_idx+1:step_idx+2],
                        pooled=target_frames.pooled[step_idx+1:step_idx+2],
                    )

                    # Teacher-forced step
                    pred_tf = model.predict(cur_tf, act, h_tf)
                    h_tf = model.belief_step(cur_tf, act, rew, h_tf)

                    # Free-running closed-loop step
                    pred_fr = model.predict(cur_fr, act, h_fr)
                    h_fr = model.belief_step(pred_fr, act, rew, h_fr)
                    cur_fr = pred_fr

                    if step in horizons:
                        t_norm = F.normalize(tgt.tokens, p=2, dim=-1)
                        ptf_norm = F.normalize(pred_tf.tokens, p=2, dim=-1)
                        pfr_norm = F.normalize(pred_fr.tokens, p=2, dim=-1)

                        tf_errors[step].append((ptf_norm - t_norm).square().mean().item())
                        fr_errors[step].append((pfr_norm - t_norm).square().mean().item())

    # Aggregate global representation metrics across all sampled tokens
    concatenated_tokens = torch.cat(all_tokens, dim=0) # [TotalFrames, 64, 192]
    effective_rank = compute_effective_rank(concatenated_tokens)
    active_dimensions = compute_active_dimensions(concatenated_tokens)
    spatial_diversity = compute_spatial_diversity(concatenated_tokens[:64])

    mean_err_model = float(np.mean(err_model_list))
    mean_err_persist = float(np.mean(err_persist_list))
    mean_err_shuffled = float(np.mean(err_shuffled_list))
    rel_pred_err = mean_err_model / max(mean_err_persist, 1e-8)
    shuffled_action_sensitivity = mean_err_shuffled - mean_err_model

    return {
        "active_dimensions": active_dimensions,
        "effective_rank": effective_rank,
        "spatial_diversity": spatial_diversity,
        "target_drift_l2": float(np.mean(drift_l2_list)),
        "target_cosine_alignment": float(np.mean(drift_cos_list)),
        "one_step_model_error": mean_err_model,
        "one_step_persist_error": mean_err_persist,
        "one_step_shuffled_error": mean_err_shuffled,
        "relative_prediction_error": rel_pred_err,
        "shuffled_action_sensitive": bool(shuffled_action_sensitivity > 0),
        "shuffled_action_gap": float(shuffled_action_sensitivity),
        "teacher_forced_errors_by_horizon": {h: float(np.mean(tf_errors[h])) for h in horizons},
        "free_running_errors_by_horizon": {h: float(np.mean(fr_errors[h])) for h in horizons},
    }


def train_candidate(
    candidate_name: str,
    encoder_type: str,
    replay: Any,
    eval_dataset: List[Dict[str, Any]],
    cfg: JEPAConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    updates: int = 1000,
    output_dir: Path = Path("results/gate3"),
    model: Optional[TrueJEPA] = None,
) -> Tuple[TrueJEPA, Dict[str, Any]]:
    print(f"\n{'='*70}\nStarting Training: {candidate_name} ({encoder_type})\n{'='*70}")
    if model is None:
        model = TrueJEPA(cfg, num_actions=17, encoder_type=encoder_type).to(device)

    # Disjoint parameter groups
    actor_params = list(model.heads.policy.parameters())
    value_params = list(model.heads.value.parameters())
    actor_set = set(actor_params)
    value_set = set(value_params)
    world_params = [
        p for p in model.parameters()
        if p.requires_grad and p not in actor_set and p not in value_set
    ]

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total Parameters: {total_params:,} | Trainable: {trainable_params:,} | Frozen: {frozen_params:,}")

    optimizer = torch.optim.AdamW(
        world_params,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    rng = np.random.default_rng(seed=42)
    training_curve = []
    start_time = time.perf_counter()

    for step in range(1, updates + 1):
        model.train()
        batch = replay.sample_with_prefix(train_cfg.batch_size, train_cfg.sequence_length, rng)
        metrics = train_world_model_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            cfg=train_cfg,
            update=step,
        )

        if step % 50 == 0 or step == updates:
            elapsed = time.perf_counter() - start_time
            fps = step / max(elapsed, 1e-4)
            log_entry = {
                "step": step,
                "jepa_loss": float(metrics["jepa"]),
                "reward_loss": float(metrics["reward"]),
                "continue_loss": float(metrics["continuation"]),
                "total_loss": float(metrics["loss"]),
                "elapsed_sec": float(elapsed),
                "fps": float(fps),
            }
            training_curve.append(log_entry)
            print(f"Step {step:4d}/{updates:4d} | Loss: {metrics['loss']:.4f} (JEPA: {metrics['jepa']:.4f}) | FPS: {fps:.1f}")

    total_time = time.perf_counter() - start_time
    print(f"\nTraining completed in {total_time:.2f}s ({updates/total_time:.1f} steps/s)")

    # Save trained checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{candidate_name.lower().replace(' ', '_')}.pt"
    model.save_checkpoint(str(checkpoint_path), update=updates)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Evaluate diagnostics on held-out seeds
    print(f"Evaluating held-out diagnostics for {candidate_name}...")
    eval_metrics = evaluate_candidate_diagnostics(model, eval_dataset, device)
    eval_metrics["parameter_counts"] = {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
    }
    eval_metrics["training_telemetry"] = {
        "total_runtime_sec": total_time,
        "average_throughput_fps": updates / total_time,
        "final_jepa_loss": training_curve[-1]["jepa_loss"],
        "final_total_loss": training_curve[-1]["total_loss"],
    }
    eval_metrics["training_curve"] = training_curve

    return model, eval_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="results/gate3")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using compute device: {device}")

    # 1. Load exact pinned datasets
    train_path = Path("data/gate3/train_replay_10k.npz")
    eval_path = Path("data/gate3/eval_heldout_seeds.pt")
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError("Gate 3 datasets not found. Run scripts/prepare_gate3_datasets.py first.")

    print(f"Loading training replay from {train_path}...")
    replay = PixelReplay.load(str(train_path))
    print(f"Replay loaded: {replay._frames} frames in {len(replay._episodes)} episodes.")

    print(f"Loading held-out evaluation dataset from {eval_path}...")
    eval_dataset = torch.load(eval_path, weights_only=False)
    print(f"Evaluation dataset loaded: {len(eval_dataset)} episodes.")

    cfg = JEPAConfig(token_dim=192)
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=3e-4,
        weight_decay=1e-6,
        grad_clip=10.0,
        amp=False, # Exact float32 for clean cross-arm parity
    )
    output_dir = Path(args.output_dir)

    # 2. Train Candidate A (From Scratch)
    _, metrics_a = train_candidate(
        candidate_name="Candidate_A_Scratch",
        encoder_type="scratch",
        replay=replay,
        eval_dataset=eval_dataset,
        cfg=cfg,
        train_cfg=train_cfg,
        device=device,
        updates=args.updates,
        output_dir=output_dir,
    )

    # 3. Train Candidate B (Frozen DINOv2 ViT-S/14)
    model_b = TrueJEPA(cfg, num_actions=17, encoder_type="dinov2").to(device)
    token_cache = extract_dinov2_token_cache(replay, model_b, device)
    dino_replay = CachedDINOv2Replay(replay, token_cache)

    _, metrics_b = train_candidate(
        candidate_name="Candidate_B_DINOv2",
        encoder_type="dinov2",
        replay=dino_replay,
        eval_dataset=eval_dataset,
        cfg=cfg,
        train_cfg=train_cfg,
        device=device,
        updates=args.updates,
        output_dir=output_dir,
        model=model_b,
    )

    # 4. Comparative Promotion Analysis
    rel_a = metrics_a["relative_prediction_error"]
    rel_b = metrics_b["relative_prediction_error"]
    rank_a = metrics_a["effective_rank"]
    rank_b = metrics_b["effective_rank"]

    if rel_b < 0.95 and (rel_b < rel_a * 0.95) and rank_b >= rank_a:
        verdict = "PROMOTE_CANDIDATE_B_DINOV2"
        reason = f"Candidate B (DINOv2) demonstrates superior predictive utility (RelPredErr: {rel_b:.3f} vs {rel_a:.3f}) and higher effective rank ({rank_b:.1f} vs {rank_a:.1f})."
    elif rel_a < 0.95 and (rel_a <= rel_b * 0.95):
        verdict = "RETAIN_CANDIDATE_A_SCRATCH"
        reason = f"Candidate A (Scratch) outperforms Candidate B in predictive utility (RelPredErr: {rel_a:.3f} vs {rel_b:.3f})."
    elif abs(rel_a - rel_b) / max(rel_a, rel_b) <= 0.05:
        verdict = "PERFORMANCE_PARITY_EVALUATE_TRADEOFFS"
        reason = f"Both candidates show comparable predictive error ({rel_a:.3f} vs {rel_b:.3f}). Compare compute and downstream utility."
    elif rel_a >= 1.0 and rel_b >= 1.0:
        verdict = "INCONCLUSIVE"
        reason = f"Both candidates fail to outperform static persistence (RelPredErr A: {rel_a:.3f}, B: {rel_b:.3f}). Dynamics modeling needs investigation."
    else:
        verdict = "PROMOTE_CANDIDATE_B_DINOV2" if rel_b < rel_a else "RETAIN_CANDIDATE_A_SCRATCH"
        reason = f"Selected candidate with lower relative prediction error (RelPredErr A: {rel_a:.3f}, B: {rel_b:.3f})."

    results_payload = {
        "protocol": "ORBIT_GATE3_MATCHED_REPRESENTATION_LADDER",
        "spec_version": "v1.3",
        "device": str(device),
        "updates": args.updates,
        "train_dataset": {
            "path": str(train_path),
            "transitions": replay._frames,
            "episodes": len(replay._episodes),
        },
        "eval_dataset": {
            "path": str(eval_path),
            "episodes": len(eval_dataset),
        },
        "candidate_a_scratch": metrics_a,
        "candidate_b_dinov2": metrics_b,
        "verdict": verdict,
        "verdict_reason": reason,
    }

    metrics_file = output_dir / "gate3_representation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n{'='*70}\nGate 3 Execution Summary\n{'='*70}")
    print(f"Candidate A (Scratch) RelPredErr: {rel_a:.4f} | Rank: {rank_a:.1f} | Active Dims: {metrics_a['active_dimensions']}/192")
    print(f"Candidate B (DINOv2)  RelPredErr: {rel_b:.4f} | Rank: {rank_b:.1f} | Active Dims: {metrics_b['active_dimensions']}/192")
    print(f"\nVerdict: {verdict}")
    print(f"Reason:  {reason}")
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
