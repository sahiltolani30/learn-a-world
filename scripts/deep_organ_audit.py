#!/usr/bin/env python3
"""
scripts/deep_organ_audit.py

Fine-Grained 5-Organ Sub-Component Audit for SAPNA.
Breaks down each of the 5 organs into 18 measurable sub-components:
  Organ 1 (Eyes): Conv Stem, Token Projection, Patch Decoder, Latent Pooling, Target Encoder EMA
  Organ 2 (Memory): Action Embedding, GRU Recurrence Delta, Multi-Step History Retention
  Organ 3 (Simulator): FiLM Action Modulation, Spatial Transformer, Ensemble Spread, Reward Head, Done Head
  Organ 4 (Decision): Policy Entropy & Unimix, Action Concentration, Value Critic, Advantage Normalizer
  Organ 5 (Engine): 15-Step Closed-Loop Drift, Lambda-Return Credit Assignment, Buffer Contiguity

Usage:
  python3 scripts/deep_organ_audit.py --checkpoint /path/to/state.pt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sapna.encoder import ObservationEncoder, TargetEncoder
from sapna.dynamics import (
    ActionEmbedding,
    TemporalCore,
    SpatialPredictor,
    RewardHead,
    DoneHead,
)
from sapna.policy import Actor, Critic, SlowCritic, unimix, symlog, symexp
from sapna.imagination import twohot_score, lambda_returns, imagine_batch
from sapna.buffer import SAPNAReplayBuffer
from sapna.train import RunningMeanStd

ACTION_NAMES = [
    "noop", "move_left", "move_right", "move_up", "move_down", "do",
    "sleep", "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword"
]


def audit_organ_1_eyes(enc: ObservationEncoder, target_enc: TargetEncoder, batch_obs: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    """Audit Organ 1 (The Eyes): Sub-components 1.1 to 1.5."""
    with torch.no_grad():
        obs_proc, _ = enc._preprocess_obs(batch_obs.to(device))
        
        # 1.1 Conv Stem Feature Map
        h1 = enc.stage1(obs_proc)
        h2 = enc.stage2(h1)
        h3 = enc.stage3(h2)
        fmap = enc.spatial_pool(h3)  # (B, 64, 8, 8)
        stem_norm = float(fmap.norm(dim=1).mean().item())
        stem_std = float(fmap.std(dim=0).mean().item())

        # 1.2 Token Projection & Spatial Variance
        tokens = enc.token_proj(fmap).flatten(2).transpose(1, 2)  # (B, 64, 64)
        tok_spatial_var = float(tokens.var(dim=1).mean().item())  # Variance across 64 patches
        tok_feature_var = float(tokens.var(dim=-1).mean().item()) # Variance across channels

        # 1.3 Pixel Decoder Reconstruction Quality
        recon = enc.decode_tokens(tokens)
        recon_mse = float(F.mse_loss(recon, obs_proc).item())
        
        # 1.4 Global Mean Pooling & Latent LayerNorm
        pooled_mean = tokens.mean(dim=1)  # (B, 64)
        proj = enc.pooled_proj(pooled_mean)
        z = enc.ln(proj)  # (B, 256)
        
        z_cent = z - z.mean(dim=0, keepdim=True)
        cov = (z_cent.T @ z_cent) / max(1, z.size(0) - 1)
        cov_np = cov.cpu().numpy()
        evals = np.linalg.eigvalsh(cov_np)
        eff_rank = float(((np.sum(evals) ** 2) / (np.sum(evals ** 2) + 1e-6)))
        min_feature_var = float(z.var(dim=0).min().item())
        mean_feature_var = float(z.var(dim=0).mean().item())

        # 1.5 Target Encoder Alignment
        _, target_z = target_enc.forward_tokens(batch_obs.to(device))
        target_cos = float(F.cosine_similarity(z, target_z, dim=-1).mean().item())

    return {
        "1.1_stem_norm": stem_norm,
        "1.1_stem_spatial_std": stem_std,
        "1.2_token_spatial_var": tok_spatial_var,
        "1.2_token_channel_var": tok_feature_var,
        "1.3_recon_pixel_mse": recon_mse,
        "1.4_latent_effective_rank": eff_rank,
        "1.4_latent_min_var": min_feature_var,
        "1.4_latent_mean_var": mean_feature_var,
        "1.5_target_encoder_cosine": target_cos,
    }


def audit_organ_2_memory(core: TemporalCore, act_emb: ActionEmbedding, enc: ObservationEncoder, seq_obs: torch.Tensor, seq_acts: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    """Audit Organ 2 (Working Memory): Sub-components 2.1 to 2.3."""
    with torch.no_grad():
        # 2.1 Action Embedding Disentanglement
        all_actions = torch.arange(17, device=device)
        all_embs = act_emb(all_actions)  # (17, 32)
        norm_embs = F.normalize(all_embs, p=2, dim=-1)
        pairwise_cos = norm_embs @ norm_embs.T
        off_diag = pairwise_cos - torch.eye(17, device=device)
        max_action_sim = float(off_diag.max().item())
        mean_action_sim = float(off_diag.abs().mean().item())

        # 2.2 TemporalCore Step Dynamics
        B, T = seq_obs.shape[0], seq_obs.shape[1]
        _, z_seq = enc.forward_tokens(seq_obs.flatten(0, 1).to(device))
        z_seq = z_seq.view(B, T, 256)
        
        h = torch.zeros(B, 256, device=device)
        h_trace = [h.clone()]
        deltas = []
        for t in range(min(T - 1, 15)):
            a_t = act_emb(seq_acts[:, t].long().to(device))
            h_next = core(z_seq[:, t], h, a_emb=a_t)
            deltas.append((h_next - h).norm(dim=-1).mean().item())
            h = h_next
            h_trace.append(h.clone())

        avg_step_delta = float(np.mean(deltas))

        # 2.3 Multi-step Memory Horizon Retention
        cos_h0_h5 = float(F.cosine_similarity(h_trace[0], h_trace[min(5, len(h_trace)-1)], dim=-1).mean().item())
        cos_h0_h15 = float(F.cosine_similarity(h_trace[0], h_trace[-1], dim=-1).mean().item())
        h_max_norm = float(h.norm(dim=-1).max().item())

    return {
        "2.1_max_action_pairwise_cos": max_action_sim,
        "2.1_mean_action_pairwise_cos": mean_action_sim,
        "2.2_gru_avg_step_delta": avg_step_delta,
        "2.3_memory_drift_cos_5_steps": cos_h0_h5,
        "2.3_memory_drift_cos_15_steps": cos_h0_h15,
        "2.3_memory_max_norm": h_max_norm,
    }


def audit_organ_3_simulator(
    pred: SpatialPredictor, rew_head: RewardHead, done_head: DoneHead, enc: ObservationEncoder,
    core: TemporalCore, act_emb: ActionEmbedding, buf: SAPNAReplayBuffer, device: torch.device
) -> Dict[str, Any]:
    """Audit Organ 3 (The Simulator): Sub-components 3.1 to 3.5."""
    with torch.no_grad():
        seq = buf.sample_sequences(B=8, T=16)
        obs_seq = seq.obs.float() / 255.0
        toks_seq, z_seq = enc.forward_tokens(obs_seq.flatten(0, 1).to(device))
        toks_seq = toks_seq.view(8, 16, 64, 64)
        z_seq = z_seq.view(8, 16, 256)

        # 3.1 FiLM & Action Modulation
        a_0 = seq.action[:, 0].long().to(device)
        h_0 = torch.zeros(8, 256, device=device)
        a_emb_0 = pred.action_embedding(a_0)
        film = pred.film(a_emb_0)
        film_gamma, film_beta = film.chunk(2, dim=-1)
        film_modulation_strength = float((film_gamma.abs().mean() + film_beta.abs().mean()).item())

        # 3.2 1-Step Prediction Accuracy
        mean_pred, all_preds = pred.forward_tokens_clean(toks_seq[:, 0], a_0, h_0)
        z_pred_1 = enc.pool_tokens(mean_pred)
        z_true_1 = z_seq[:, 1]
        pred_cos = float(F.cosine_similarity(z_pred_1, z_true_1, dim=-1).mean().item())
        pred_l2 = float(F.mse_loss(z_pred_1, z_true_1).item())

        # 3.3 Multi-Head Predictor Disagreement
        head_pooled = torch.stack([enc.pool_tokens(all_preds[k]) for k in range(4)], dim=0) # (4, 8, 256)
        ensemble_disagreement = float(head_pooled.std(dim=0).mean().item())

        # 3.4 Reward Head Discrimination (Milestone vs Non-Milestone Arrival States)
        all_starts, ms_starts = buf._get_valid_starts(T=16)
        non_ms_starts = [s for s in all_starts if s not in set(ms_starts)]
        
        def score_starts(start_list):
            scores = []
            for s_idx in start_list[:16]:
                w_obs = torch.from_numpy(buf.obs[s_idx:s_idx+16].astype(np.float32) / 255.0).to(device)
                w_act = torch.from_numpy(buf.action[s_idx:s_idx+16].astype(np.int64)).to(device)
                h = torch.zeros(1, 256, device=device)
                for t in range(15):
                    tok_t, z_t = enc.forward_tokens(w_obs[t:t+1])
                    h = core(z_t, h, a_emb=act_emb(w_act[t:t+1]))
                    if t == 14:
                        p_toks, _ = pred.forward_tokens_clean(tok_t, w_act[t:t+1], h)
                        arr = torch.cat([enc.pool_tokens(p_toks), h], dim=-1)
                        scores.append(twohot_score(rew_head(arr)).item())
            return scores

        ms_scores = score_starts(ms_starts)
        rand_scores = score_starts(non_ms_starts)
        pred_rew_ms = float(np.mean(ms_scores)) if ms_scores else 0.0
        pred_rew_rand = float(np.mean(rand_scores)) if rand_scores else 0.0
        reward_discrimination_gap = pred_rew_ms - pred_rew_rand

        # 3.5 Done Head Calibration
        arr_samples = torch.cat([z_seq[:, 0], h_0], dim=-1)
        done_logits = done_head(arr_samples)
        done_probs = torch.sigmoid(done_logits).squeeze(-1)
        mean_predicted_done_prob = float(done_probs.mean().item())

    return {
        "3.1_film_modulation_strength": film_modulation_strength,
        "3.2_1step_pred_latent_cosine": pred_cos,
        "3.2_1step_pred_latent_l2": pred_l2,
        "3.3_ensemble_head_disagreement": ensemble_disagreement,
        "3.4_pred_reward_milestones": pred_rew_ms,
        "3.4_pred_reward_random": pred_rew_rand,
        "3.4_reward_discrimination_gap": reward_discrimination_gap,
        "3.5_mean_predicted_death_prob": mean_predicted_done_prob,
    }


def audit_organ_4_decision(
    actor: Actor, critic: Critic, slow_critic: SlowCritic,
    enc: ObservationEncoder, buf: SAPNAReplayBuffer, adv_norm: RunningMeanStd, device: torch.device
) -> Dict[str, Any]:
    """Audit Organ 4 (The Decision Maker): Sub-components 4.1 to 4.5."""
    with torch.no_grad():
        # Sample 128 states from buffer
        audit_batch = buf.sample_sequences(B=min(128, max(8, buf.size // 16)), T=8)
        audit_obs = audit_batch.obs[:, 0].to(device)
        z_pooled = enc(audit_obs)
        h_zeros = torch.zeros(len(audit_obs), 256, device=device)
        state_input = torch.cat([z_pooled, h_zeros], dim=-1)

        # 4.1 & 4.2 Actor Action Distribution & Entropy
        logits = actor(state_input)
        raw_probs = torch.softmax(logits, dim=-1)
        probs = unimix(raw_probs, unimix_ratio=0.01)
        entropies = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
        mean_entropy = float(entropies.mean().item())
        
        action_hist = probs.mean(dim=0).cpu().numpy()
        top_action_idx = int(np.argmax(action_hist))
        top_action_prob = float(action_hist[top_action_idx])
        
        # Movement actions: move_left(1), move_right(2), move_up(3), move_down(4)
        movement_fraction = float(np.sum(action_hist[1:5]))

        # 4.3 Critic Value Spread
        critic_logits = critic(state_input)
        v_scores = twohot_score(critic_logits)
        critic_mean_v = float(v_scores.mean().item())
        critic_std_v = float(v_scores.std().item())

        # 4.4 Slow Critic Tracking Alignment
        slow_logits = slow_critic(state_input)
        slow_v_scores = twohot_score(slow_logits)
        critic_slow_drift = float((v_scores - slow_v_scores).abs().mean().item())

        # 4.5 Advantage Normalizer Statistics
        adv_mean = float(adv_norm.mean.item()) if hasattr(adv_norm, "mean") else 0.0
        adv_std = float(adv_norm.std.item()) if hasattr(adv_norm, "std") else 1.0

    return {
        "4.1_unimix_entropy_nats": mean_entropy,
        "4.2_dominant_action_name": ACTION_NAMES[top_action_idx],
        "4.2_dominant_action_prob": top_action_prob,
        "4.2_movement_action_fraction": movement_fraction,
        "4.3_critic_mean_value": critic_mean_v,
        "4.3_critic_q_spread_std": critic_std_v,
        "4.4_critic_slow_divergence": critic_slow_drift,
        "4.5_advantage_norm_mean": adv_mean,
        "4.5_advantage_norm_std": adv_std,
    }


def audit_organ_5_engine(
    enc: ObservationEncoder, core: TemporalCore, act_emb: ActionEmbedding,
    pred: SpatialPredictor, rew_head: RewardHead, done_head: DoneHead,
    actor: Actor, critic: Critic, slow_critic: SlowCritic,
    buf: SAPNAReplayBuffer, device: torch.device
) -> Dict[str, Any]:
    """Audit Organ 5 (The Engine): Sub-components 5.1 to 5.4."""
    with torch.no_grad():
        # 5.1 15-step Closed-Loop Autoregressive Rollout Drift
        sample_seq = buf.sample_sequences(B=8, T=16)
        obs_0 = sample_seq.obs[:, 0].to(device)
        toks_0, z_0 = enc.forward_tokens(obs_0)
        h_0 = torch.zeros(8, 256, device=device)

        imag = imagine_batch(
            tokens0=toks_0,
            z0=z_0,
            h0=h_0,
            actor=actor,
            critic=critic,
            predictor=pred,
            temporal_core=core,
            action_embedding=act_emb,
            pool_fn=enc.pool_tokens,
            reward_head=rew_head,
            done_head=done_head,
            slow_critic=slow_critic,
            H=15,
            unimix_ratio=0.01,
        )

        final_z = imag["final_z"]
        autoregressive_drift_15 = float(F.cosine_similarity(z_0, final_z, dim=-1).mean().item())
        imag_reward_sum = float(imag["rewards"].sum(dim=-1).mean().item())

        # 5.2 Lambda Returns Credit Propagation
        G = lambda_returns(
            imag["rewards"],
            imag["conts"],
            imag["values"],
            imag["bootstrap"],
            gamma=0.997,
            lam=0.95,
        )
        return_std = float(G.std().item())
        raw_adv = G - imag["values"]
        mean_raw_adv = float(raw_adv.mean().item())

        # 5.3 Buffer Contiguity Check
        all_starts, ms_starts = buf._get_valid_starts(T=16)
        buffer_total_windows = len(all_starts)
        buffer_milestone_windows = len(ms_starts)

    return {
        "5.1_imagined_15step_drift_cosine": autoregressive_drift_15,
        "5.1_imagined_15step_mean_cum_reward": imag_reward_sum,
        "5.2_lambda_return_std": return_std,
        "5.2_raw_advantage_mean": mean_raw_adv,
        "5.3_buffer_valid_windows": buffer_total_windows,
        "5.3_buffer_milestone_windows": buffer_milestone_windows,
    }


def run_full_deep_audit(checkpoint_path: str) -> Dict[str, Any]:
    """Execute complete 5-organ, 18-sub-component audit."""
    print("=" * 88)
    print(f"  SAPNA DEEP MULTI-ORGAN SUB-COMPONENT AUDIT")
    print(f"  Target Checkpoint: {checkpoint_path}")
    print("=" * 88)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    step = int(state.get("step", 0))

    # Instantiate modules
    enc = ObservationEncoder().to(device)
    enc.load_state_dict(state["encoder"])
    enc.eval()

    target_enc = TargetEncoder(enc).to(device)
    target_enc.load_state_dict(state["target_encoder"])
    target_enc.eval()

    core = TemporalCore().to(device)
    core.load_state_dict(state["temporal_core"])
    core.eval()

    act_emb = ActionEmbedding().to(device)
    act_emb.load_state_dict(state["action_embedding"])
    act_emb.eval()

    pred = SpatialPredictor().to(device)
    pred.load_state_dict(state["predictor"])
    pred.eval()

    rew_head = RewardHead().to(device)
    rew_head.load_state_dict(state["reward_head"])
    rew_head.eval()

    done_head = DoneHead().to(device)
    done_head.load_state_dict(state["done_head"])
    done_head.eval()

    actor = Actor().to(device)
    actor.load_state_dict(state["actor"])
    actor.eval()

    critic = Critic().to(device)
    critic.load_state_dict(state["critic"])
    critic.eval()

    slow_critic = SlowCritic(critic).to(device)
    slow_critic.load_state_dict(state["slow_critic"])
    slow_critic.eval()

    buf = SAPNAReplayBuffer(capacity=len(state["buffer"]["obs"]))
    buf.load_state(state["buffer"])

    adv_norm = RunningMeanStd().to(device)
    if "adv_normalizer" in state and state["adv_normalizer"] is not None:
        adv_norm.load_state_dict(state["adv_normalizer"])

    sample_seq = buf.sample_sequences(B=8, T=16)

    # 1. Run Audits
    o1 = audit_organ_1_eyes(enc, target_enc, sample_seq.obs[:, 0], device)
    o2 = audit_organ_2_memory(core, act_emb, enc, sample_seq.obs, sample_seq.action, device)
    o3 = audit_organ_3_simulator(pred, rew_head, done_head, enc, core, act_emb, buf, device)
    o4 = audit_organ_4_decision(actor, critic, slow_critic, enc, buf, adv_norm, device)
    o5 = audit_organ_5_engine(enc, core, act_emb, pred, rew_head, done_head, actor, critic, slow_critic, buf, device)

    # 2. Print Granular Breakdown
    def status_pill(val, lower, upper=None, invert=False):
        if upper is not None:
            ok = (lower <= val <= upper)
        elif invert:
            ok = (val <= lower)
        else:
            ok = (val >= lower)
        return "[PASS]" if ok else "[WARN]"

    print(f"\n{'ORGAN 1: THE EYES (Perception / Encoder)':<50} | {'Observed':<18} | {'Benchmark':<10} | Status")
    print("-" * 88)
    print(f"  1.1 Conv Stem Feature Map Norm                  | {o1['1.1_stem_norm']:<18.4f} | > 5.0      | {status_pill(o1['1.1_stem_norm'], 5.0)}")
    print(f"  1.2 Token Spatial Variance across 64 Tiles      | {o1['1.2_token_spatial_var']:<18.4f} | > 0.05     | {status_pill(o1['1.2_token_spatial_var'], 0.05)}")
    print(f"  1.3 Pixel Reconstruction MSE (decode_tokens)    | {o1['1.3_recon_pixel_mse']:<18.4f} | < 0.05     | {status_pill(o1['1.3_recon_pixel_mse'], 0.05, invert=True)}")
    print(f"  1.4 Latent Representation Effective Rank (PR)   | {o1['1.4_latent_effective_rank']:<18.2f} | > 1.0      | {status_pill(o1['1.4_latent_effective_rank'], 1.0)}")
    print(f"  1.4 Latent Mean Variance across 256 Features    | {o1['1.4_latent_mean_var']:<18.4f} | > 0.05     | {status_pill(o1['1.4_latent_mean_var'], 0.05)}")
    print(f"  1.5 Target Encoder Cosine Alignment             | {o1['1.5_target_encoder_cosine']:<18.4f} | > 0.85     | {status_pill(o1['1.5_target_encoder_cosine'], 0.85)}")

    print(f"\n{'ORGAN 2: WORKING MEMORY (TemporalCore Recurrence)':<50} | {'Observed':<18} | {'Benchmark':<10} | Status")
    print("-" * 88)
    print(f"  2.1 Action Embedding Max Pairwise Overlap       | {o2['2.1_max_action_pairwise_cos']:<18.4f} | < 0.85     | {status_pill(o2['2.1_max_action_pairwise_cos'], 0.85, invert=True)}")
    print(f"  2.2 GRU Single-step Dynamic State Transition    | {o2['2.2_gru_avg_step_delta']:<18.4f} | > 0.10     | {status_pill(o2['2.2_gru_avg_step_delta'], 0.10)}")
    print(f"  2.3 Multi-step Memory Horizon Retention (15 st) | {o2['2.3_memory_drift_cos_15_steps']:<18.4f} | < 0.90     | {status_pill(o2['2.3_memory_drift_cos_15_steps'], 0.90, invert=True)}")

    print(f"\n{'ORGAN 3: THE SIMULATOR (Predictor & Foresight)':<50} | {'Observed':<18} | {'Benchmark':<10} | Status")
    print("-" * 88)
    print(f"  3.1 FiLM Action Modulation Strength             | {o3['3.1_film_modulation_strength']:<18.4f} | > 0.10     | {status_pill(o3['3.1_film_modulation_strength'], 0.10)}")
    print(f"  3.2 1-Step Latent Prediction Cosine Alignment   | {o3['3.2_1step_pred_latent_cosine']:<18.4f} | > 0.70     | {status_pill(o3['3.2_1step_pred_latent_cosine'], 0.70)}")
    print(f"  3.3 Ensemble Multi-Head Disagreement Spread     | {o3['3.3_ensemble_head_disagreement']:<18.4f} | > 0.01     | {status_pill(o3['3.3_ensemble_head_disagreement'], 0.01)}")
    print(f"  3.4 Milestone vs Random Reward Discrimination   | {o3['3.4_reward_discrimination_gap']:<18.4f} | > 0.00     | {status_pill(o3['3.4_reward_discrimination_gap'], 0.00)}")
    print(f"  3.5 Terminal Death Probability Calibration      | {o3['3.5_mean_predicted_death_prob']:<18.4f} | < 0.10     | {status_pill(o3['3.5_mean_predicted_death_prob'], 0.10, invert=True)}")

    print(f"\n{'ORGAN 4: THE DECISION MAKER (Actor & Critic)':<50} | {'Observed':<18} | {'Benchmark':<10} | Status")
    print("-" * 88)
    print(f"  4.1 Policy Entropy (unimix)                     | {o4['4.1_unimix_entropy_nats']:<18.4f} | >= 0.50    | {status_pill(o4['4.1_unimix_entropy_nats'], 0.50)}")
    print(f"  4.2 Dominant Action Probability                 | {o4['4.2_dominant_action_prob']*100:<17.1f}% | < 50.0%    | {status_pill(o4['4.2_dominant_action_prob'], 0.50, invert=True)}")
    print(f"      [Top Action: {o4['4.2_dominant_action_name']}]")
    print(f"  4.2 Movement Action Fraction (left/right/up/dn) | {o4['4.2_movement_action_fraction']*100:<17.1f}% | >= 10.0%   | {status_pill(o4['4.2_movement_action_fraction'], 0.10)}")
    print(f"  4.3 Critic Value Spread Across States           | {o4['4.3_critic_q_spread_std']:<18.4f} | > 0.05     | {status_pill(o4['4.3_critic_q_spread_std'], 0.05)}")

    print(f"\n{'ORGAN 5: THE ENGINE (Imagination & Memory Replay)':<50} | {'Observed':<18} | {'Benchmark':<10} | Status")
    print("-" * 88)
    print(f"  5.1 15-Step Autoregressive Dream Drift Cosine   | {o5['5.1_imagined_15step_drift_cosine']:<18.4f} | >= 0.60    | {status_pill(o5['5.1_imagined_15step_drift_cosine'], 0.60)}")
    print(f"  5.2 Lambda-Return Variance (Credit Backprop)    | {o5['5.2_lambda_return_std']:<18.4f} | > 0.01     | {status_pill(o5['5.2_lambda_return_std'], 0.01)}")
    print(f"  5.3 Contiguous Replay Windows in Buffer         | {o5['5.3_buffer_valid_windows']:<18d} | >= 64      | {status_pill(o5['5.3_buffer_valid_windows'], 64)}")

    report = {
        "step": step,
        "organ_1_eyes": o1,
        "organ_2_memory": o2,
        "organ_3_simulator": o3,
        "organ_4_decision": o4,
        "organ_5_engine": o5,
    }

    # Save JSON report
    out_dir = Path(checkpoint_path).resolve().parent.parent / "reports"
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir / f"deep_organ_audit_step_{step}.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 88)
    print(f"[+] Detailed Audit Saved to: {out_file}")
    print("=" * 88)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAPNA Deep Organ Audit")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to state.pt checkpoint")
    args = parser.parse_args()
    run_full_deep_audit(args.checkpoint)
