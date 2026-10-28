"""
End-to-End Self-Contained Training and Evaluation Pipeline for Godfather JEPA.

Integrates all components:
1. VisualConv4Encoder & TargetEMA (stop-gradient momentum encoder).
2. ExpanderProjector with decoupled VICReg loss (Var(z) >= 1.0).
3. SpatialConvGRU & VectorBeliefCore (translation equivariance).
4. EpistemicPredictorEnsemble (K=5 disagreement curiosity).
5. 65-Bin SymlogCategoricalCritic & AmortizedPolicyActor (lambda=0.95).
6. SurvivalContinuationHead (state-dependent mortality discounting).
7. MilestoneReplayBuffer (milestone-prioritized sampling).
8. Procedural training seed stream (guaranteed zero evaluation leakage).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from godfather_jepa.envs.crafter_env import CrafterEnvWrapper
from godfather_jepa.eval.benchmark import BenchmarkRunner
from godfather_jepa.losses.focal_loss import class_weighted_focal_loss
from godfather_jepa.losses.vicreg import decoupled_vicreg_loss
from godfather_jepa.models import (
    AmortizedPolicyActor,
    BoundedResidualPredictor,
    DenseRewardHead,
    EpistemicPredictorEnsemble,
    ExpanderProjector,
    SpatialConvGRU,
    SpatialFeatureProjector,
    SurvivalContinuationHead,
    SymlogCategoricalCritic,
    SymlogTwoHotCoder,
    TargetEMA,
    TopologicalAnchorGraph,
    VectorBeliefCore,
    VisualConv4Encoder,
    VisualOdometryHead,
    update_target_ema,
)
from godfather_jepa.memory import (
    OnlineEpisodicStore,
    SemanticConsolidatedBank,
    OfflineConsolidationEngine,
)
from godfather_jepa.planning.causal_hypergraph import CausalHypergraph
from godfather_jepa.planning.homeostatic_arbiter import HomeostaticArbiter
from godfather_jepa.planning.latent_mpc_planner import LatentMPCPlanner
from godfather_jepa.training.checkpoint_adapter import audit_and_adapt_checkpoint
from godfather_jepa.training.latent_ac import compute_lambda_returns
from godfather_jepa.training.replay_buffer import MilestoneReplayBuffer
from godfather_jepa.utils.seed_guard import generate_procedural_seed, assert_training_seed_pure


class GodfatherJEPAPipeline:
    """
    Unified training and evaluation pipeline for Godfather JEPA.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        checkpoint_init: Optional[str] = None,
        learning_rate: float = 3e-4,
        latent_dim: int = 512,
        horizon: int = 15,
        gamma: float = 0.997,
        lambd: float = 0.95,
    ):
        self.device = device
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.gamma = gamma
        self.lambd = lambd

        # 1. World Model
        self.encoder = VisualConv4Encoder(latent_dim=latent_dim).to(device)
        self.target_encoder = TargetEMA(self.encoder, initial_tau=0.996).to(device)
        self.projector = ExpanderProjector(in_dim=latent_dim, proj_dim=1024).to(device)
        self.spatial_projector = SpatialFeatureProjector(in_channels=256, out_channels=64).to(device)

        self.spatial_gru = SpatialConvGRU(channels=64).to(device)
        self.vector_belief = VectorBeliefCore(latent_dim=latent_dim).to(device)
        self.ensemble = EpistemicPredictorEnsemble(k_heads=5, latent_dim=latent_dim).to(device)
        self.predictor = self.ensemble.heads[0]
        self.continuation_head = SurvivalContinuationHead(state_dim=latent_dim * 2).to(device)
        self.reward_head = DenseRewardHead(state_dim=latent_dim * 2).to(device)
        self.odometry_head = VisualOdometryHead(latent_dim=latent_dim).to(device)
        self.semantic_bank = SemanticConsolidatedBank()
        self.causal_hypergraph = self.semantic_bank.hypergraph
        self.consolidation_engine = OfflineConsolidationEngine(self.semantic_bank)
        self.arbiter = HomeostaticArbiter(
            thirst_threshold=3.0,
            hunger_threshold=3.0,
            health_threshold=4.0,
            dusk_step_threshold=135,
        )

        # 2. Actor-Critic
        self.critic = SymlogCategoricalCritic(state_dim=latent_dim * 2).to(device)
        self.target_critic = TargetEMA(self.critic, initial_tau=0.996).to(device)
        self.actor = AmortizedPolicyActor(state_dim=latent_dim * 2).to(device)
        self.symlog_coder = SymlogTwoHotCoder()

        # 3. Optimizers
        wm_params = (
            list(self.encoder.parameters())
            + list(self.projector.parameters())
            + list(self.spatial_projector.parameters())
            + list(self.spatial_gru.parameters())
            + list(self.vector_belief.parameters())
            + list(self.ensemble.parameters())
            + list(self.continuation_head.parameters())
            + list(self.reward_head.parameters())
            + list(self.odometry_head.parameters())
        )
        self.wm_opt = torch.optim.Adam(wm_params, lr=learning_rate)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        # 4. Checkpoint warm-start adaptation if supplied
        if checkpoint_init and os.path.exists(checkpoint_init):
            success, report = audit_and_adapt_checkpoint(
                checkpoint_init, self.encoder, target_actor=self.actor, device=device
            )
            print(f"[Adapter] Audit verdict: {report.verdict}")

        # 5. Buffer
        self.buffer = MilestoneReplayBuffer(capacity=2000, priority_ratio=0.10)

        # 6. Latent MPC Trajectory Planner
        self.mpc_planner = LatentMPCPlanner(
            predictor=self.predictor,
            reward_head=self.reward_head,
            continuation_head=self.continuation_head,
            critic=self.critic,
            symlog_coder=self.symlog_coder,
            spatial_gru=self.spatial_gru,
            vector_belief=self.vector_belief,
            causal_hypergraph=self.causal_hypergraph,
            horizon=10,
            num_candidates=64,
            cem_iterations=2,
            gamma=self.gamma,
            device=device,
        )

    def select_action(
        self,
        obs: np.ndarray,
        prev_action: int = 0,
        h_spatial: Optional[torch.Tensor] = None,
        h_vector: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Fast O(1) pixel-only action selection.
        """
        self.encoder.eval()
        self.actor.eval()
        self.spatial_projector.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.uint8, device=self.device).unsqueeze(0)
            z, feat = self.encoder(obs_t)

            act_t = torch.tensor([prev_action], dtype=torch.long, device=self.device)
            # SpatialFeatureProjector maps 256 channels to 64 channels
            feat_proj = self.spatial_projector(feat)
            h_spatial = self.spatial_gru(feat_proj, act_t, H=h_spatial)
            h_vector = self.vector_belief(z, act_t, h_spatial, h=h_vector)

            composite_state = torch.cat([z, h_vector], dim=-1)
            action_t = self.actor.sample_action(composite_state, deterministic=deterministic)
            action = int(action_t.item())

        return action, h_spatial, h_vector

    def select_action_stateful(
        self,
        obs: np.ndarray,
        state: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
        use_mpc: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Stateful pixel-only action selection preserving:
        - Recurrent belief (h_spatial, h_vector)
        - Visual odometry and TopologicalAnchorGraph
        - Internal mental inventory & vital estimates
        - Homeostatic-teleological arbitration
        """
        if state is None:
            state = {
                "h_spatial": None,
                "h_vector": None,
                "prev_action": 0,
                "prev_z": None,
                "anchor_graph": TopologicalAnchorGraph(),
                "inventory": {"wood": 0, "stone": 0, "wood_pickaxe": 0},
                "workstations": set(),
                "internal_vitals": {"drink": 9.0, "food": 9.0, "health": 9.0},
                "step": 0,
            }

        h_spatial = state["h_spatial"]
        h_vector = state["h_vector"]
        prev_act = state["prev_action"]
        prev_z = state["prev_z"]
        anchor = state["anchor_graph"]
        inv = state["inventory"]
        ws = state["workstations"]
        vitals = state["internal_vitals"]
        step = state["step"]

        # Update empirical mental inventory when positive reward is received on action 5 (Do)
        last_rew = state.get("last_reward", 0.0)
        if last_rew > 0.0 and prev_act == 5:
            if inv.get("wood_pickaxe", 0) == 0:
                inv["wood"] = inv.get("wood", 0) + 1
            else:
                inv["stone"] = inv.get("stone", 0) + 1

        self.encoder.eval()
        self.actor.eval()
        self.spatial_projector.eval()
        self.odometry_head.eval()

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.uint8, device=self.device).unsqueeze(0)
            z, feat = self.encoder(obs_t)

            act_t = torch.tensor([prev_act], dtype=torch.long, device=self.device)
            feat_proj = self.spatial_projector(feat)
            h_spatial = self.spatial_gru(feat_proj, act_t, H=h_spatial)
            h_vector = self.vector_belief(z, act_t, h_spatial, h=h_vector)
            composite = torch.cat([z, h_vector], dim=-1)

            # 1. Update visual odometry if previous latent exists
            if prev_z is not None:
                delta_p = self.odometry_head(prev_z, act_t, z)
                anchor.update_odometry(delta_p)

            # 2. Survival continuation probability
            contin_prob = float(self.continuation_head(composite).item())

            # 3. Policy actor action
            actor_action_t = self.actor.sample_action(composite, deterministic=deterministic)
            actor_action = int(actor_action_t.item())

            # 4. Teleological target selection & action determination
            target_goal = "make_wood_pickaxe" if inv.get("wood_pickaxe", 0) == 0 else "collect_stone"
            if use_mpc:
                teleo_action, mpc_debug = self.mpc_planner.plan(
                    z_0=z,
                    h_vector_0=h_vector,
                    h_spatial_0=h_spatial,
                    inventory=inv,
                    workstations=ws,
                    anchor_graph=anchor,
                    target_goal=target_goal,
                )
            else:
                subgoal, determined_action = self.causal_hypergraph.determine_subgoal(
                    target_goal=target_goal,
                    current_inventory=inv,
                    workstations_present=ws,
                )
                if subgoal in ("place_table", "make_wood_pickaxe", "make_wood_sword", "place_furnace"):
                    teleo_action = determined_action
                else:
                    teleo_action = actor_action if actor_action in (1, 2, 3, 4, 5) else determined_action

                # Workstation navigation override if return needed
                if "table_placed" in ws and inv.get("wood", 0) >= 1 and inv.get("wood_pickaxe", 0) == 0:
                    return_act = anchor.get_return_action("table")
                    if return_act is not None and return_act != 0:
                        teleo_action = return_act

            # 6. Homeostatic arbitration (preemptive drinking reflex / danger)
            # Detect water adjacency from center-bottom visual patch (RGB blue channel dominance)
            water_adjacent = bool((obs[30:34, 30:34, 2].mean() > 140) and (obs[30:34, 30:34, 0].mean() < 80))
            # Stateful vital decay: decrement per step rather than overwriting from global step count
            vitals["drink"] = max(0.0, vitals.get("drink", 9.0) - (1.0 / 9.0))
            vitals["food"] = max(0.0, vitals.get("food", 9.0) - (1.0 / 25.0))

            final_action, mode = self.arbiter.arbitrate(
                vitals=vitals,
                step=step,
                teleological_action=teleo_action,
                water_adjacent=water_adjacent,
                continuation_prob=contin_prob,
            )

            # 7. Update internal mental inventory
            if final_action == 8:  # place table
                ws.add("table_placed")
                inv["wood"] = max(0, inv.get("wood", 0) - 2)
                anchor.register_landmark("table", latent_key=z.cpu().numpy())
            elif final_action == 11:  # make wood pickaxe
                inv["wood_pickaxe"] = 1
                inv["wood"] = max(0, inv.get("wood", 0) - 1)
            elif mode == "homeostatic_drink":
                vitals["drink"] = 9.0

        next_state = {
            "h_spatial": h_spatial,
            "h_vector": h_vector,
            "prev_action": final_action,
            "prev_z": z,
            "anchor_graph": anchor,
            "inventory": inv,
            "workstations": ws,
            "internal_vitals": vitals,
            "step": step + 1,
        }
        return final_action, next_state

    def collect_episode(self, env_seed: int, max_steps: int = 400) -> Dict[str, Any]:
        """
        Collects a full episode using procedural seed, recording into fast OnlineEpisodicStore
        and consolidating discoveries into permanent SemanticConsolidatedBank.
        """
        assert_training_seed_pure(env_seed)
        env = CrafterEnvWrapper(seed=env_seed, max_steps=max_steps, training=True)
        res = env.reset()
        obs = res[0] if isinstance(res, tuple) else res

        obs_list, act_list, rew_list, done_list = [obs], [], [], []
        achievements_dict: Dict[str, List[int]] = {}

        online_store = OnlineEpisodicStore(target_goal="make_wood_pickaxe")
        state = None
        done = False
        steps = 0
        total_rew = 0.0

        while not done and steps < max_steps:
            act, state = self.select_action_stateful(obs, state=state, deterministic=False)
            step_res = env.step(act)
            if len(step_res) == 5:
                obs_next, reward, terminated, truncated, info = step_res
                done = terminated or truncated
            else:
                obs_next, reward, done, info = step_res

            # Detect first-time unlocks in this episode
            unlocked = None
            step_ach = info.get("achievements", {})
            for a_name, cnt in step_ach.items():
                if cnt > 0 and (a_name not in achievements_dict or not any(achievements_dict[a_name])):
                    unlocked = a_name
                    break

            item_deltas = {}
            if act == 5 and reward > 0 and unlocked == "collect_wood":
                item_deltas["wood"] = 1
            elif act == 8:
                item_deltas["wood"] = -2
            elif act == 11:
                item_deltas["wood"] = -1
            elif act == 5 and reward > 0 and unlocked == "collect_stone":
                item_deltas["stone"] = 1

            # Record transition into online episodic working memory
            prev_z = state.get("prev_z")
            z_np = prev_z.detach().cpu().numpy().squeeze(0) if prev_z is not None else np.zeros(512, dtype=np.float32)
            online_store.record_step(
                step_idx=steps,
                z_t=z_np,
                action=act,
                reward=reward,
                done=done,
                z_next=z_np,
                item_deltas=item_deltas,
                unlocked_achievement=unlocked,
            )

            obs_list.append(obs_next)
            act_list.append(act)
            rew_list.append(reward)
            done_list.append(done)

            for ach, cnt in step_ach.items():
                if ach not in achievements_dict:
                    achievements_dict[ach] = []
                achievements_dict[ach].append(cnt)

            obs = obs_next
            steps += 1
            total_rew += reward

        # Consolidate newly discovered empirical evidence into permanent semantic memory
        consolidation_stats = self.consolidation_engine.consolidate_episode(online_store)

        obs_arr = np.array(obs_list[:-1], dtype=np.uint8)
        act_arr = np.array(act_list, dtype=np.int64)
        rew_arr = np.array(rew_list, dtype=np.float32)
        done_arr = np.array(done_list, dtype=bool)
        ach_arr = {k: np.array(v, dtype=np.int32) for k, v in achievements_dict.items()}

        self.buffer.add_episode(obs_arr, act_arr, rew_arr, done_arr, ach_arr)

        return {
            "steps": steps,
            "reward": total_rew,
            "seed": env_seed,
            "consolidation": consolidation_stats,
        }

    def train_world_model_step(self, batch_size: int = 16, seq_len: int = 16) -> Dict[str, float]:
        """
        Trains encoder, spatial belief, ensemble predictor, and continuation head.
        """
        if len(self.buffer) < batch_size:
            return {}

        batch = self.buffer.sample_batch(batch_size=batch_size, seq_len=seq_len, device=self.device)
        obs = batch["obs"]      # (B, T, 64, 64, 3)
        acts = batch["actions"] # (B, T)
        rews = batch["rewards"] # (B, T)
        dones = batch["dones"]  # (B, T)

        b, t = acts.shape

        # Flatten frames for Conv4 encoder
        flat_obs = obs.reshape(b * t, 64, 64, 3)
        z, feat = self.encoder(flat_obs)
        z = z.reshape(b, t, -1)
        feat = feat.reshape(b, t, 256, 8, 8)

        # Project 256-dim feature maps into 64-dim spatial memory channel space
        feat_proj = self.spatial_projector(feat.reshape(b * t, 256, 8, 8)).reshape(b, t, 64, 8, 8)

        # Target encoder representations
        with torch.no_grad():
            target_z, _ = self.target_encoder.forward(flat_obs)
            target_z = target_z.reshape(b, t, -1)

        # 1. VICReg loss on projector
        p = self.projector(z.reshape(b * t, -1))
        l_var, l_cov = decoupled_vicreg_loss(p, gamma=1.0)
        vicreg_loss_val = l_var + 0.04 * l_cov

        # 2. Sequential belief tracking & predictor ensemble loss
        h_spatial = None
        h_vector = None
        pred_losses = []
        contin_losses = []

        for step in range(t - 1):
            feat_s = feat_proj[:, step]
            act_s = acts[:, step]
            z_s = z[:, step]

            h_spatial = self.spatial_gru(feat_s, act_s, H=h_spatial)
            h_vector = self.vector_belief(z_s, act_s, h_spatial, h=h_vector)

            # Ensemble forward
            preds = self.ensemble(z_s, h_vector, act_s)  # (K, B, 512)
            next_target = target_z[:, step + 1]           # (B, 512)
            p_loss = self.ensemble.prediction_loss(preds, next_target)
            pred_losses.append(p_loss)

            # Continuation head loss
            composite = torch.cat([z_s, h_vector], dim=-1)
            pred_contin = self.continuation_head(composite)
            target_contin = (~dones[:, step]).float()
            c_loss = class_weighted_focal_loss(pred_contin, target_contin)
            contin_losses.append(c_loss)

            # Extrinsic reward prediction loss
            pred_rew = self.reward_head(composite, act_s)
            target_rew = rews[:, step]
            r_loss = F.mse_loss(pred_rew, target_rew)
            contin_losses.append(r_loss)

            # Kinematic displacement mapping for Crafter primitive actions:
            # Action 1: Left (-1, 0), Action 2: Right (+1, 0), Action 3: Up (0, -1), Action 4: Down (0, +1)
            kinematic_deltas = torch.zeros(b, 2, device=self.device)
            kinematic_deltas[act_s == 1] = torch.tensor([-1.0, 0.0], device=self.device)
            kinematic_deltas[act_s == 2] = torch.tensor([1.0, 0.0], device=self.device)
            kinematic_deltas[act_s == 3] = torch.tensor([0.0, -1.0], device=self.device)
            kinematic_deltas[act_s == 4] = torch.tensor([0.0, 1.0], device=self.device)

            next_z_step = z[:, step + 1]
            pred_delta_p = self.odometry_head(z_s, act_s, next_z_step)
            odo_loss = F.mse_loss(pred_delta_p, kinematic_deltas)
            contin_losses.append(odo_loss)

        total_wm_loss = vicreg_loss_val + torch.stack(pred_losses).mean() + torch.stack(contin_losses).mean()

        self.wm_opt.zero_grad()
        total_wm_loss.backward()
        nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        nn.utils.clip_grad_norm_(self.spatial_projector.parameters(), 1.0)
        self.wm_opt.step()

        # Update target encoder via EMA
        self.target_encoder.update(tau=0.996)

        return {
            "wm_loss": total_wm_loss.item(),
            "vicreg_var": l_var.item(),
            "vicreg_cov": l_cov.item(),
        }

    def train_actor_critic_step(self, batch_size: int = 16) -> Dict[str, float]:
        """
        Latent imagination rollouts over horizon H=15 computing multi-step lambda-returns.
        """
        if len(self.buffer) < batch_size:
            return {}

        batch = self.buffer.sample_batch(batch_size=batch_size, seq_len=4, device=self.device)
        obs_start = batch["obs"][:, 0]  # (B, 64, 64, 3)

        self.encoder.eval()
        self.spatial_projector.eval()
        with torch.no_grad():
            z, feat = self.encoder(obs_start)
            h_spatial = self.spatial_projector(feat)
            h_vector = torch.zeros(batch_size, self.latent_dim, device=self.device)

        # Unroll imagination trajectory inside learned world model over horizon H=15
        imagined_states = []
        imagined_actions = []
        imagined_rewards = []
        imagined_discounts = []

        cur_z = z
        cur_h = h_vector
        cur_sp = h_spatial

        for step in range(self.horizon):
            composite = torch.cat([cur_z, cur_h], dim=-1)
            imagined_states.append(composite)

            # Actor selects action in latent space
            log_probs = self.actor(composite)
            dist = torch.distributions.Categorical(logits=log_probs)
            act = dist.sample()
            imagined_actions.append(act)

            with torch.no_grad():
                # Predict next latent via ensemble mean
                preds = self.ensemble(cur_z, cur_h, act)
                next_z = preds.mean(dim=0)

                # Extrinsic reward prediction from learned reward head
                extrinsic_r = self.reward_head(composite, act)

                # Curiosity intrinsic reward from ensemble variance
                curiosity_r = self.ensemble.disagreement(preds)
                total_reward = extrinsic_r + 0.10 * curiosity_r
                imagined_rewards.append(total_reward)

                # Continuation discounting
                contin_prob = self.continuation_head(composite)
                discount = self.gamma * contin_prob
                imagined_discounts.append(discount)

                cur_sp = self.spatial_gru(cur_sp, act, H=cur_sp)
                cur_h = self.vector_belief(next_z, act, cur_sp, h=cur_h)
                cur_z = next_z

        states_t = torch.stack(imagined_states, dim=0)       # (H, B, 1024)
        actions_t = torch.stack(imagined_actions, dim=0)     # (H, B)
        rewards_t = torch.stack(imagined_rewards, dim=0)     # (H, B)
        discounts_t = torch.stack(imagined_discounts, dim=0) # (H, B)

        # 1. Compute multi-step lambda-returns with slow EMA target critic
        with torch.no_grad():
            target_logits = self.target_critic.forward(states_t, use_target=True)
            target_vals = self.symlog_coder.decode(target_logits)
            lambda_targets = compute_lambda_returns(rewards_t, target_vals, discounts_t, lambd=self.lambd)

        # 2. Update 65-bin symlog categorical critic
        critic_logits = self.critic(states_t)
        critic_loss = self.symlog_coder.loss(
            critic_logits.reshape(-1, self.symlog_coder.num_bins),
            lambda_targets.reshape(-1),
        )

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # Update target critic EMA
        self.target_critic.update(tau=0.996)

        # 3. Update Actor policy
        with torch.no_grad():
            baseline_vals = self.symlog_coder.decode(self.critic(states_t))
            advantages = lambda_targets - baseline_vals

        log_probs = self.actor(states_t)
        action_log_probs = log_probs.gather(-1, actions_t.unsqueeze(-1)).squeeze(-1)
        actor_loss = -(action_log_probs * advantages).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    def save_checkpoint(self, out_path: str) -> None:
        """
        Saves full model state dict for Colab / Drive persistence.
        """
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        ckpt = {
            "encoder": self.encoder.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "projector": self.projector.state_dict(),
            "spatial_projector": self.spatial_projector.state_dict(),
            "spatial_gru": self.spatial_gru.state_dict(),
            "vector_belief": self.vector_belief.state_dict(),
            "ensemble": self.ensemble.state_dict(),
            "continuation_head": self.continuation_head.state_dict(),
            "reward_head": self.reward_head.state_dict(),
            "odometry_head": self.odometry_head.state_dict(),
            "critic": self.critic.state_dict(),
            "actor": self.actor.state_dict(),
            "latent_dim": self.latent_dim,
            "horizon": self.horizon,
        }
        torch.save(ckpt, out_path)
        bank_path = str(Path(out_path).parent / "godfather_memory_bank.pt")
        self.semantic_bank.save(bank_path)
        print(f"[Checkpoint] Successfully saved Godfather JEPA model to {out_path} and memory bank to {bank_path}")

    def load_checkpoint(self, ckpt_path: str) -> bool:
        """
        Loads full model state dict and associated memory bank from disk or Drive.
        """
        if not os.path.exists(ckpt_path):
            print(f"[Checkpoint] Warning: File not found: {ckpt_path}")
            return False

        ckpt = torch.load(ckpt_path, map_location=self.device)
        if "encoder" in ckpt:
            self.encoder.load_state_dict(ckpt["encoder"])
        if "target_encoder" in ckpt:
            self.target_encoder.load_state_dict(ckpt["target_encoder"])
        if "projector" in ckpt:
            self.projector.load_state_dict(ckpt["projector"])
        if "spatial_projector" in ckpt:
            self.spatial_projector.load_state_dict(ckpt["spatial_projector"])
        if "spatial_gru" in ckpt:
            self.spatial_gru.load_state_dict(ckpt["spatial_gru"])
        if "vector_belief" in ckpt:
            self.vector_belief.load_state_dict(ckpt["vector_belief"])
        if "ensemble" in ckpt:
            self.ensemble.load_state_dict(ckpt["ensemble"])
        if "continuation_head" in ckpt:
            self.continuation_head.load_state_dict(ckpt["continuation_head"])
        if "reward_head" in ckpt:
            self.reward_head.load_state_dict(ckpt["reward_head"])
        if "odometry_head" in ckpt:
            self.odometry_head.load_state_dict(ckpt["odometry_head"])
        if "critic" in ckpt:
            self.critic.load_state_dict(ckpt["critic"])
        if "actor" in ckpt:
            self.actor.load_state_dict(ckpt["actor"])

        bank_path = str(Path(ckpt_path).parent / "godfather_memory_bank.pt")
        if os.path.exists(bank_path):
            self.semantic_bank.load(bank_path)
            print(f"[Checkpoint] Successfully loaded Godfather JEPA model from {ckpt_path} and memory bank from {bank_path}")
        else:
            print(f"[Checkpoint] Successfully loaded Godfather JEPA model from {ckpt_path}")

        return True

    def evaluate(self, seeds: Optional[Sequence[int]] = None, use_mpc: bool = True) -> Dict[str, Any]:
        """
        Runs standardized benchmark evaluation with preserved recurrent state across steps.
        """
        runner = BenchmarkRunner(seeds=seeds)

        def stateful_policy(obs: np.ndarray, state: Optional[Dict[str, Any]] = None):
            return self.select_action_stateful(obs, state=state, deterministic=True, use_mpc=use_mpc)

        return runner.run(stateful_policy)


def main():
    parser = argparse.ArgumentParser(description="Godfather JEPA Training Runner")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--updates-per-episode", type=int, default=5, help="Number of gradient update steps per episode")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v3_phase2.pt")
    parser.add_argument("--shard-dir", type=str, default=None, help="Directory containing sequence .npz shards to pre-populate buffer")
    parser.add_argument("--out-dir", type=str, default="godfather_runs/", help="Output directory for checkpoints and eval results")
    args = parser.parse_args()

    device = torch.device(args.device)
    pipeline = GodfatherJEPAPipeline(device=device, checkpoint_init=args.checkpoint)

    # Optional: Load Drive / local sequence shards into MilestoneReplayBuffer
    if args.shard_dir and os.path.exists(args.shard_dir):
        shard_files = sorted(Path(args.shard_dir).glob("*.npz"))
        if shard_files:
            print(f"[Bootstrap] Ingesting {len(shard_files)} sequence shards from {args.shard_dir}...")
            loaded = pipeline.buffer.load_npz_shards([str(f) for f in shard_files], max_episodes=500)
            print(f"[Bootstrap] Successfully loaded {loaded} episodes into MilestoneReplayBuffer ({len(pipeline.buffer.milestone_indices)} milestones).")

    print(f"=== Starting Godfather JEPA Training on {device} (updates_per_episode={args.updates_per_episode}) ===")
    for ep in range(args.episodes):
        seed = generate_procedural_seed(rank=0, episode=ep, generation=0)
        res = pipeline.collect_episode(env_seed=seed, max_steps=250)
        for _ in range(args.updates_per_episode):
            wm_stats = pipeline.train_world_model_step()
            ac_stats = pipeline.train_actor_critic_step()

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1:03d} | Seed: {seed} | Steps: {res['steps']} | Reward: {res['reward']:.1f} | WM Loss: {wm_stats.get('wm_loss', 0.0):.4f}")

    # Save final checkpoint
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(out_dir / "godfather_jepa_final.pt")
    pipeline.save_checkpoint(ckpt_path)

    print("\n=== Running Locked-Seed Benchmark (2050..2059) ===")
    bench_results = pipeline.evaluate(seeds=list(range(2050, 2060)))
    print(f"Benchmark Crafter Score: {bench_results['crafter_score']:.2f}%")
    print(f"Mean Survival Length: {bench_results['mean_survival_length']:.1f} steps")


if __name__ == "__main__":
    main()
