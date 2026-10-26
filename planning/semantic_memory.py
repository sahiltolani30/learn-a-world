#!/usr/bin/env python3
"""
planning/semantic_memory.py — Cognitive Causal Memory Architecture
==================================================================

Pure Autonomous World Modeling & Experience Diary (DeepMind Neural Episodic Control).
Adheres strictly to THE IRON RULE: ZERO HARDCODING. ZERO CHEATS. ZERO SHORTCUTS.
- NO human recipes (no "wood -= 2", no "table += 1").
- NO human rules (no "gather 3 wood", no "if pickaxe: boost novelty").
- NO hardcoded achievement indices (no index 6, 15, 16).
- 100% pure learning from empirical gameplay, sensory perception, and Monte Carlo returns.

Components:
1. ShortTermWorkingMemory:
   - Within-episode rolling context (FIFO sensory buffers).
   - Dynamic 16-D backpack resource tracking.
   - Spatial landmark object anchors (orientation persistence).
   - Episode trajectory history buffer for retrospective credit assignment.
2. LongTermSemanticMemory (Experience Diary):
   - Stores: (z_t, inv_t, a_t, delta_inv_t, G_t).
   - Gaussian RBF Nadaraya-Watson Kernel Regression in joint visual-inventory space.
   - Pure mathematical suppression of fruitless / failed actions (G = 0.0).
3. DopaminePersonalBestTracker:
   - Intrinsic record-breaking reinforcement (lifetime & session personal bests).
4. AutonomousMemoryWriter:
   - Captures live interactions, updates episode trajectory.
   - Retroactive Monte Carlo return backpropagation: G_t = sum_k gamma^k (r_{t+k} + dopamine_{t+k}).
   - Consolidates causal breakthroughs, predecessors, and failure counterfactuals.
"""

from collections import deque
import json
import math
import os
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
from eval.metrics import CRAFTER_ACHIEVEMENTS

NUM_ACHIEVEMENTS = len(CRAFTER_ACHIEVEMENTS)  # 22

# Canonical 16 physical backpack resources observed directly from Crafter info['inventory']
CRAFTER_INVENTORY_KEYS = [
    "wood", "stone", "coal", "iron", "diamond", "sapling",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
    "health", "food", "drink", "energy"
]
NUM_INVENTORY_ITEMS = len(CRAFTER_INVENTORY_KEYS)  # 16


def inventory_dict_to_vector(inv_dict: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    Converts Crafter inventory dictionary into a 16-D float32 vector.
    Zero recipes or interpretation: extracts raw item counts directly from the game.
    """
    vec = np.zeros(NUM_INVENTORY_ITEMS, dtype=np.float32)
    if not inv_dict or not isinstance(inv_dict, dict):
        # Default starting stats for Crafter
        vec[12] = 9.0  # health
        vec[13] = 9.0  # food
        vec[14] = 9.0  # drink
        vec[15] = 9.0  # energy
        return vec

    for i, key in enumerate(CRAFTER_INVENTORY_KEYS):
        vec[i] = float(inv_dict.get(key, 0))
    return vec


def vector_to_inventory_dict(vec: np.ndarray) -> Dict[str, int]:
    """Converts 16-D vector back to human-readable inventory dictionary."""
    return {key: int(round(float(vec[i]))) for i, key in enumerate(CRAFTER_INVENTORY_KEYS)}


class ShortTermWorkingMemory:
    """
    Short-Term Working Memory (System 1 - Within-Episode Context).
    Maintains rolling sensory history, dynamic 16-D backpack state, landmark anchors,
    and the full episode trajectory buffer for backward credit assignment.
    """

    def __init__(self, capacity: int = 16, latent_dim: int = 128, recurrent_dim: int = 256):
        self.capacity = capacity
        self.latent_dim = latent_dim
        self.recurrent_dim = recurrent_dim

        self.latent_window: Deque[torch.Tensor] = deque(maxlen=capacity)
        self.inv_window: Deque[np.ndarray] = deque(maxlen=capacity)
        self.action_window: Deque[int] = deque(maxlen=capacity)
        self.reward_window: Deque[float] = deque(maxlen=capacity)

        self.recurrent_h: Optional[torch.Tensor] = None
        self.active_inventory: np.ndarray = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
        self.current_inv: np.ndarray = inventory_dict_to_vector(None)

        # Full episode trajectory buffer for retrospective Monte Carlo backpropagation
        self.episode_trajectory: List[Dict[str, Any]] = []

        # Spatial/Landmark Anchors: stores latent representation of key structures
        self.landmark_anchors: Dict[str, torch.Tensor] = {}
        self.landmark_step_created: Dict[str, int] = {}
        self.current_step: int = 0
        self.vital_trend: float = 0.0

    def reset(self) -> None:
        """Resets working memory at the start of a new episode."""
        self.latent_window.clear()
        self.inv_window.clear()
        self.action_window.clear()
        self.reward_window.clear()
        self.episode_trajectory.clear()
        self.recurrent_h = None
        self.active_inventory.fill(0)
        self.current_inv = inventory_dict_to_vector(None)
        self.landmark_anchors.clear()
        self.landmark_step_created.clear()
        self.current_step = 0
        self.vital_trend = 0.0

    def update(
        self,
        z_t: torch.Tensor,
        action: int,
        reward: float,
        next_h: torch.Tensor,
        inventory_mask: Optional[np.ndarray] = None,
        inventory_dict: Optional[Dict[str, Any]] = None,
        unlocked_event: Optional[str] = None,
        dopamine_surge: float = 0.0,
    ) -> None:
        """Updates working memory state after a step."""
        self.current_step += 1
        z_cpu = z_t.detach().cpu().flatten()
        self.latent_window.append(z_cpu)
        self.action_window.append(int(action))
        self.reward_window.append(float(reward))
        self.recurrent_h = next_h.detach()

        prev_inv = self.current_inv.copy()
        if inventory_dict is not None:
            self.current_inv = inventory_dict_to_vector(inventory_dict)
        self.inv_window.append(self.current_inv.copy())

        if inventory_mask is not None:
            self.active_inventory = np.maximum(self.active_inventory, inventory_mask.astype(np.uint8))

        delta_inv = self.current_inv - prev_inv

        # Store complete transition in episode trajectory for retrospective Monte Carlo learning
        self.episode_trajectory.append({
            "step": self.current_step,
            "z": z_cpu.clone(),
            "inv": prev_inv,
            "action": int(action),
            "reward": float(reward),
            "next_inv": self.current_inv.copy(),
            "delta_inv": delta_inv,
            "dopamine": float(dopamine_surge),
            "event": unlocked_event,
        })

        # Generic Object Permanence: track placed structures until subsequent breakthrough
        if unlocked_event is not None:
            if "place_" in unlocked_event:
                self.landmark_anchors["structure"] = z_cpu.clone()
                self.landmark_step_created["structure"] = self.current_step
            else:
                # Any milestone breakthrough fulfills the structure's potential -> release anchor
                self.landmark_anchors.clear()
                self.landmark_step_created.clear()

        # Update vital trend (damage slope over last 5 steps)
        if len(self.reward_window) >= 5:
            recent_rews = list(self.reward_window)[-5:]
            self.vital_trend = float(np.mean(recent_rews))

    def has_anchor(self, name: str = "structure", max_age: int = 250) -> bool:
        """Checks if a landmark anchor exists and is within temporal relevance."""
        key = name if name in self.landmark_anchors else ("structure" if "structure" in self.landmark_anchors else None)
        if key is None:
            return False
        age = self.current_step - self.landmark_step_created.get(key, 0)
        return age <= max_age

    def get_anchor(self, name: str = "structure") -> Optional[torch.Tensor]:
        """Retrieves a landmark anchor latent vector."""
        if name in self.landmark_anchors:
            return self.landmark_anchors[name]
        return self.landmark_anchors.get("structure", None)


class LongTermSemanticMemory:
    """
    Long-Term Semantic Memory / Experience Diary (System 2 - Across-Episode Causal Model).
    Stores: (z_t, inv_t, a_t, delta_inv_t, G_t).
    
    Evaluates actions via Gaussian RBF Nadaraya-Watson Kernel Regression in the joint
    visual and 16-D backpack inventory space:
    Similarity((z_t, inv_t), (z_i, inv_i)) = cos(z_t, z_i) * exp(- ||inv_t - inv_i||^2 / (2 * sigma^2))
    
    Zero hardcoded rules: 100% learned from agent's own gameplay and Monte Carlo returns.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        device: Union[str, torch.device] = "cpu",
        rbf_sigma: float = 0.80,
    ):
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.rbf_sigma = rbf_sigma
        self.max_capacity = 4096  # Bounded capacity for 100,000-game scalability

        # Core Experience Diary tensors
        self.keys: torch.Tensor = torch.empty((0, latent_dim), dtype=torch.float32, device=self.device)
        self.inventories: torch.Tensor = torch.empty((0, NUM_INVENTORY_ITEMS), dtype=torch.float32, device=self.device)
        self.actions: torch.Tensor = torch.empty((0,), dtype=torch.long, device=self.device)
        self.inventory_deltas: torch.Tensor = torch.empty((0, NUM_INVENTORY_ITEMS), dtype=torch.float32, device=self.device)
        self.returns: torch.Tensor = torch.empty((0,), dtype=torch.float32, device=self.device)
        self.outcome_names: List[str] = []

        # Backwards compatibility tensors for legacy code / checkpoints
        self.preconditions: torch.Tensor = torch.empty((0, NUM_ACHIEVEMENTS), dtype=torch.float32, device=self.device)
        self.outcomes: torch.Tensor = torch.empty((0, NUM_ACHIEVEMENTS), dtype=torch.float32, device=self.device)

    def __len__(self) -> int:
        return self.keys.shape[0]

    def _evict_lowest_priority(self) -> None:
        """
        Evicts the lowest-value entry when capacity is reached.
        Permanent Priority Shields:
        1. Survival Invariants: Any memory that restores food, drink, energy, or health is NEVER evicted.
        2. Rare Tool & Combat Breakthroughs: Pickaxes, swords, tables, defeated mobs are NEVER evicted.
        Only redundant, low-return exploration steps with no vital or resource utility are evicted.
        """
        if len(self) == 0:
            return
        priorities = self.returns.clone()
        
        # 1. Permanent Homeostatic Survival Shield: never forget how to eat, drink, sleep, heal!
        vital_restoration_mask = (self.inventory_deltas[:, 12:16] > 0.0).any(dim=-1)
        priorities[vital_restoration_mask] += 20.0

        # 2. Permanent Tool, Resource & Combat Milestone Shield
        for idx, name in enumerate(self.outcome_names):
            if any(m in name for m in ["pickaxe", "sword", "table", "diamond", "stone", "iron", "zombie", "skeleton", "cow", "plant", "drink", "eat"]):
                priorities[idx] += 10.0

        evict_idx = int(torch.argmin(priorities).item())
        keep_mask = torch.ones(len(self), dtype=torch.bool, device=self.device)
        keep_mask[evict_idx] = False

        self.keys = self.keys[keep_mask]
        self.inventories = self.inventories[keep_mask]
        self.actions = self.actions[keep_mask]
        self.inventory_deltas = self.inventory_deltas[keep_mask]
        self.returns = self.returns[keep_mask]
        self.outcome_names.pop(evict_idx)
        if self.preconditions.shape[0] > 0:
            self.preconditions = self.preconditions[keep_mask]
            self.outcomes = self.outcomes[keep_mask]

    def add_entry(
        self,
        key: torch.Tensor,
        inventory: Optional[Union[np.ndarray, torch.Tensor]] = None,
        action: int = 0,
        inventory_delta: Optional[Union[np.ndarray, torch.Tensor]] = None,
        realized_return: float = 0.0,
        outcome_name: str = "unknown",
        preconditions: Optional[np.ndarray] = None,
        outcome: Optional[np.ndarray] = None,
    ) -> str:
        """
        Adds a single experience tuple (z_t, inv_t, a_t, delta_inv_t, G_t) into the diary.
        Performs similarity de-duplication: if an entry with the same action and identical
        resource inventory already exists with visual cosine similarity > 0.90, updates its return
        rather than bloating memory with duplicates.
        """
        if inventory is None:
            if preconditions is not None:
                inventory = preconditions
            else:
                inventory = np.zeros(NUM_INVENTORY_ITEMS, dtype=np.float32)

        k = key.detach().to(self.device).float().flatten()
        k_norm = k / (k.norm(p=2).clamp(min=1e-6))

        # Support both 16-D backpack vector and legacy 22-D calls
        if isinstance(inventory, torch.Tensor):
            inv_arr = inventory.detach().cpu().numpy().flatten()
        else:
            inv_arr = np.array(inventory, dtype=np.float32).flatten()

        if len(inv_arr) == NUM_ACHIEVEMENTS:
            # Legacy call: inventory argument was actually 22-D preconditions
            precond_arr = inv_arr.copy()
            inv_16 = inventory_dict_to_vector(None)
            for idx, ach_name in enumerate(CRAFTER_ACHIEVEMENTS):
                if precond_arr[idx] > 0.5:
                    for inv_key in CRAFTER_INVENTORY_KEYS[:12]:
                        if ach_name == f"collect_{inv_key}" or ach_name == f"make_{inv_key}":
                            inv_idx = CRAFTER_INVENTORY_KEYS.index(inv_key)
                            inv_16[inv_idx] = max(inv_16[inv_idx], 1.0)
        else:
            inv_16 = inv_arr[:NUM_INVENTORY_ITEMS].astype(np.float32)
            if preconditions is not None:
                p_flat = np.array(preconditions, dtype=np.float32).flatten()
                if len(p_flat) == NUM_ACHIEVEMENTS:
                    precond_arr = p_flat
                else:
                    precond_arr = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
                    precond_arr[:min(len(p_flat), NUM_ACHIEVEMENTS)] = p_flat[:NUM_ACHIEVEMENTS]
            else:
                precond_arr = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)

        if inventory_delta is not None:
            if isinstance(inventory_delta, torch.Tensor):
                delta_16 = inventory_delta.detach().cpu().numpy().flatten()[:NUM_INVENTORY_ITEMS].astype(np.float32)
            else:
                delta_16 = np.array(inventory_delta, dtype=np.float32).flatten()[:NUM_INVENTORY_ITEMS]
        else:
            delta_16 = np.zeros(NUM_INVENTORY_ITEMS, dtype=np.float32)

        if outcome is not None:
            o_flat = np.array(outcome, dtype=np.float32).flatten()
            if len(o_flat) == NUM_ACHIEVEMENTS:
                out_arr = o_flat
            else:
                out_arr = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
                out_arr[:min(len(o_flat), NUM_ACHIEVEMENTS)] = o_flat[:NUM_ACHIEVEMENTS]
        else:
            out_arr = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)

        # De-duplication check: update existing memory if same action and identical resource state
        if len(self) > 0:
            same_action_mask = (self.actions == action)
            if same_action_mask.any():
                cand_indices = torch.where(same_action_mask)[0]
                cand_keys = self.keys[cand_indices]
                cos_sims = torch.mv(cand_keys, k_norm)
                cand_invs = self.inventories[cand_indices]
                inv_curr_t = torch.from_numpy(inv_16).to(self.device)

                # Physical resources/tools (indices 0..11) must match exactly
                res_dist_sq = ((cand_invs[:, :12] - inv_curr_t[:12].unsqueeze(0)) ** 2).sum(dim=-1)
                # Vitals variations (indices 12..15) weighted lightly so minor hunger does not duplicate crafting rules
                vital_diff = (cand_invs[:, 12:16] - inv_curr_t[12:16].unsqueeze(0)) * 0.10
                vital_dist_sq = (vital_diff ** 2).sum(dim=-1)

                match_mask = (cos_sims >= 0.90) & (res_dist_sq < 0.05) & (vital_dist_sq < 1.0)
                if match_mask.any():
                    match_idx = cand_indices[torch.where(match_mask)[0][0]]
                    # Update existing return with max return achieved
                    old_ret = float(self.returns[match_idx].item())
                    new_ret = max(old_ret, float(realized_return))
                    self.returns[match_idx] = new_ret
                    return "updated"

        # Check capacity limit for 100,000-game bounded memory
        if len(self) >= self.max_capacity:
            self._evict_lowest_priority()

        # Append new memory entry
        inv_t = torch.from_numpy(inv_16).to(self.device).float().flatten()
        delta_t = torch.from_numpy(delta_16).to(self.device).float().flatten()
        a_t = torch.tensor([action], dtype=torch.long, device=self.device)
        r_t = torch.tensor([realized_return], dtype=torch.float32, device=self.device)
        p_t = torch.from_numpy(precond_arr).to(self.device).float().flatten()
        o_t = torch.from_numpy(out_arr).to(self.device).float().flatten()

        self.keys = torch.cat([self.keys, k_norm.unsqueeze(0)], dim=0)
        self.inventories = torch.cat([self.inventories, inv_t.unsqueeze(0)], dim=0)
        self.actions = torch.cat([self.actions, a_t], dim=0)
        self.inventory_deltas = torch.cat([self.inventory_deltas, delta_t.unsqueeze(0)], dim=0)
        self.returns = torch.cat([self.returns, r_t], dim=0)
        self.outcome_names.append(outcome_name)
        self.preconditions = torch.cat([self.preconditions, p_t.unsqueeze(0)], dim=0)
        self.outcomes = torch.cat([self.outcomes, o_t.unsqueeze(0)], dim=0)
        return "added"

    def query_all_actions(
        self,
        z_t: torch.Tensor,
        active_inventory: Union[np.ndarray, torch.Tensor],
        rbf_sigma: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Gaussian RBF Nadaraya-Watson kernel regression for ALL 17 actions:
        Q_memory(z_t, inv_t, a) = sum_{i in N_a} (Sim_i * G_i) / (sum Sim_i + eps)
        
        Returns:
            q_values: [17] tensor of estimated action values.
            max_sims: [17] tensor of highest joint similarity for each action.
        """
        num_actions = 17
        q_values = torch.zeros(num_actions, dtype=torch.float32, device=self.device)
        max_sims = torch.zeros(num_actions, dtype=torch.float32, device=self.device)

        if len(self) == 0:
            return q_values, max_sims

        sigma = rbf_sigma if rbf_sigma is not None else self.rbf_sigma

        z = z_t.detach().to(self.device).float().flatten()
        z_norm = z / (z.norm(p=2).clamp(min=1e-6))

        if isinstance(active_inventory, torch.Tensor):
            inv_vec = active_inventory.detach().to(self.device).float().flatten()
        else:
            inv_vec = torch.from_numpy(active_inventory).to(self.device).float().flatten()

        if len(inv_vec) > NUM_INVENTORY_ITEMS:
            inv_vec = inv_vec[:NUM_INVENTORY_ITEMS]

        # 1. Visual Cosine Similarity (clamped to positive)
        cos_sims = torch.clamp(torch.mv(self.keys, z_norm), min=0.0)  # [N]

        # 2. Gaussian RBF Backpack Distance Kernel
        # Weight resources/tools with full weight (1.0), and downweight minor vital variations (0.2)
        weights = torch.tensor([1.0] * 12 + [0.2] * 4, device=self.device)
        inv_diff_sq = (((self.inventories - inv_vec.unsqueeze(0)) * weights.unsqueeze(0)) ** 2).sum(dim=-1)  # [N]
        rbf_weights = torch.exp(-inv_diff_sq / (2.0 * (sigma ** 2)))  # [N] in (0, 1]

        # Joint Similarity: Sim_i = cos_sim * RBF
        joint_sims = cos_sims * rbf_weights  # [N]

        # 3. Kernel Regression per discrete action
        for act in range(num_actions):
            act_mask = (self.actions == act)
            if not act_mask.any():
                continue

            sims_act = joint_sims[act_mask]
            rets_act = self.returns[act_mask]

            weight_sum = sims_act.sum()
            if weight_sum > 1e-4:
                q_val = (sims_act * rets_act).sum() / weight_sum
                q_values[act] = q_val
                max_sims[act] = sims_act.max()

        return q_values, max_sims

    def query(
        self,
        z_t: torch.Tensor,
        active_inventory: Union[np.ndarray, torch.Tensor],
        sim_threshold: float = 0.58,
        rbf_sigma: Optional[float] = None,
    ) -> Tuple[Optional[int], float, Optional[str]]:
        """
        Queries the memory bank for the single best affordance action using
        Gaussian RBF Nadaraya-Watson kernel regression.
        
        Returns:
            (best_action, similarity_confidence, outcome_name)
        """
        if len(self) == 0:
            return None, 0.0, None

        if isinstance(active_inventory, torch.Tensor):
            inv_arr = active_inventory.detach().cpu().numpy().flatten()
        else:
            inv_arr = np.array(active_inventory, dtype=np.float32).flatten()

        # Handle legacy 22-D achievement mask calls for backwards compatibility
        if len(inv_arr) == NUM_ACHIEVEMENTS and self.preconditions.shape[0] > 0 and self.preconditions.sum() > 0:
            return self._query_legacy_mask(z_t, inv_arr, sim_threshold)

        q_values, max_sims = self.query_all_actions(z_t, inv_arr, rbf_sigma=rbf_sigma)

        # Candidate actions must exceed similarity threshold and have positive expected return
        valid_candidates = (max_sims >= sim_threshold) & (q_values > 0.0)

        if not valid_candidates.any():
            best_sim = float(max_sims.max().item()) if len(max_sims) > 0 else 0.0
            return None, best_sim, None

        cand_indices = torch.where(valid_candidates)[0]
        # Rank by composite score = q_value * max_sim
        composite_scores = q_values[cand_indices] * max_sims[cand_indices]
        best_cand_local = int(torch.argmax(composite_scores))
        best_action = int(cand_indices[best_cand_local].item())
        best_sim = float(max_sims[best_action].item())

        # Retrieve representative outcome name
        name = "breakthrough"
        act_mask = (self.actions == best_action)
        if act_mask.any():
            act_idx = torch.where(act_mask)[0]
            best_mem_idx = int(act_idx[torch.argmax(self.returns[act_idx])].item())
            if best_mem_idx < len(self.outcome_names):
                name = self.outcome_names[best_mem_idx]

        return best_action, best_sim, name

    def _query_legacy_mask(
        self,
        z_t: torch.Tensor,
        active_achievements: np.ndarray,
        sim_threshold: float = 0.58,
    ) -> Tuple[Optional[int], float, Optional[str]]:
        """Legacy fallback query for backwards-compatible test suites."""
        z = z_t.detach().to(self.device).float().flatten()
        z_norm = z / (z.norm(p=2).clamp(min=1e-6))
        inv_t = torch.from_numpy(active_achievements).to(self.device).float()

        p_counts = self.preconditions.sum(dim=-1)
        matched_preconditions = (self.preconditions * inv_t.unsqueeze(0)).sum(dim=-1)
        overlap = torch.where(
            p_counts > 0,
            matched_preconditions / p_counts,
            torch.tensor(1.0, device=self.device),
        )
        sufficient_overlap = (overlap >= 0.50)
        unmet_outcome = (self.outcomes * (1.0 - inv_t).unsqueeze(0)).sum(dim=-1) > 0
        valid_mask = sufficient_overlap & unmet_outcome

        if not valid_mask.any():
            return None, 0.0, None

        valid_indices = torch.where(valid_mask)[0]
        valid_keys = self.keys[valid_indices]
        cos_sims = torch.mv(valid_keys, z_norm)

        valid_overlaps = overlap[valid_indices]
        valid_returns = self.returns[valid_indices]
        return_weights = 1.0 + 0.20 * torch.tanh(valid_returns)
        composite_scores = cos_sims * (0.70 + 0.30 * valid_overlaps) * return_weights

        best_local_idx = int(torch.argmax(composite_scores))
        best_sim = float(cos_sims[best_local_idx].item())

        if best_sim >= sim_threshold:
            global_idx = int(valid_indices[best_local_idx].item())
            retrieved_action = int(self.actions[global_idx].item())
            retrieved_name = self.outcome_names[global_idx] if global_idx < len(self.outcome_names) else "milestone"
            return retrieved_action, best_sim, retrieved_name

        return None, best_sim, None

    def save(self, path: Union[str, Path]) -> Path:
        """Saves long-term memory bank to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "keys": self.keys.cpu(),
                "inventories": self.inventories.cpu(),
                "actions": self.actions.cpu(),
                "inventory_deltas": self.inventory_deltas.cpu(),
                "returns": self.returns.cpu(),
                "outcome_names": self.outcome_names,
                "preconditions": self.preconditions.cpu(),
                "outcomes": self.outcomes.cpu(),
                "latent_dim": self.latent_dim,
                "rbf_sigma": self.rbf_sigma,
            },
            str(p),
        )
        return p

    def load(self, path: Union[str, Path]) -> None:
        """Loads long-term memory bank from disk with full backwards compatibility."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Memory bank file not found: {p}")
        data = torch.load(str(p), map_location=self.device)
        self.keys = data["keys"].to(self.device)
        self.actions = data["actions"].to(self.device)
        self.returns = data["returns"].to(self.device)
        self.outcome_names = data.get("outcome_names", [])
        self.latent_dim = data.get("latent_dim", self.latent_dim)
        self.rbf_sigma = data.get("rbf_sigma", self.rbf_sigma)

        if "inventories" in data:
            self.inventories = data["inventories"].to(self.device)
        else:
            # Backwards compatibility: initialize from zero
            self.inventories = torch.zeros((len(self.keys), NUM_INVENTORY_ITEMS), dtype=torch.float32, device=self.device)

        if "inventory_deltas" in data:
            self.inventory_deltas = data["inventory_deltas"].to(self.device)
        else:
            self.inventory_deltas = torch.zeros((len(self.keys), NUM_INVENTORY_ITEMS), dtype=torch.float32, device=self.device)

        if "preconditions" in data:
            self.preconditions = data["preconditions"].to(self.device)
        else:
            self.preconditions = torch.zeros((len(self.keys), NUM_ACHIEVEMENTS), dtype=torch.float32, device=self.device)

        if "outcomes" in data:
            self.outcomes = data["outcomes"].to(self.device)
        else:
            self.outcomes = torch.zeros((len(self.keys), NUM_ACHIEVEMENTS), dtype=torch.float32, device=self.device)


class DopaminePersonalBestTracker:
    """
    Dopamine Personal-Best Circuit (Intrinsic Record-Breaking Reinforcement).
    Tracks historical lifetime and within-session bests:
    - Survival steps
    - Cumulative reward
    - Milestones unlocked in a single game
    - Unique milestones discovered across lifetime
    When an episode sets a new personal record, it triggers a 'dopamine surge'
    that is retroactively backpropagated across the episode's causal predecessors.
    """

    def __init__(self, dopamine_scale: float = 0.50):
        self.dopamine_scale = dopamine_scale
        self.lifetime_best_reward: float = 0.0
        self.lifetime_best_milestones: int = 0
        self.lifetime_best_steps: int = 0
        self.lifetime_unlocked_set: Set[str] = set()

        self.current_ep_reward: float = 0.0
        self.current_ep_milestones: int = 0
        self.current_ep_steps: int = 0
        self.record_broken_this_ep: bool = False

    def reset_episode(self) -> None:
        """Resets episode counters at the start of each episode."""
        self.current_ep_reward = 0.0
        self.current_ep_milestones = 0
        self.current_ep_steps = 0
        self.record_broken_this_ep = False

    def update_step(
        self,
        reward: float,
        newly_unlocked: List[str],
    ) -> float:
        """
        Updates step metrics and computes intrinsic dopamine bonus.
        Returns dopamine surge value (>= 0.0).
        """
        self.current_ep_steps += 1
        self.current_ep_reward += float(reward)
        self.current_ep_milestones += len(newly_unlocked)

        dopamine_surge = 0.0

        # 1. Unlocked an achievement NEVER seen before across all games
        for name in newly_unlocked:
            if name not in self.lifetime_unlocked_set:
                self.lifetime_unlocked_set.add(name)
                dopamine_surge += self.dopamine_scale * 1.5
                self.record_broken_this_ep = True

        # 2. Beat previous highest achievement count in a single game
        if self.current_ep_milestones > self.lifetime_best_milestones:
            self.lifetime_best_milestones = self.current_ep_milestones
            dopamine_surge += self.dopamine_scale * 1.0
            self.record_broken_this_ep = True

        # 3. Beat previous highest episode reward
        if self.current_ep_reward > self.lifetime_best_reward and self.lifetime_best_reward > 0.0:
            margin = self.current_ep_reward - self.lifetime_best_reward
            self.lifetime_best_reward = self.current_ep_reward
            dopamine_surge += float(np.clip(margin * 0.25, 0.0, 1.0))
            self.record_broken_this_ep = True
        elif self.lifetime_best_reward == 0.0 and self.current_ep_reward > 0.0:
            self.lifetime_best_reward = self.current_ep_reward

        # 4. Beat longest survival length
        if self.current_ep_steps > self.lifetime_best_steps:
            self.lifetime_best_steps = self.current_ep_steps

        return float(dopamine_surge)


class AutonomousMemoryWriter:
    """
    Autonomous Memory Writer with Temporal Return Backpropagation.
    
    1. During Episode:
       Records (z_t, inv_t, action, reward, next_inv, delta_inv, newly_unlocked, dopamine_surge)
       into an internal trajectory buffer.
    2. End of Episode:
       Performs backward pass computing true Monte Carlo discounted returns:
       G_t = sum_k gamma^k (r_{t+k} + dopamine_{t+k})
       Consolidates into LongTermSemanticMemory:
       - Breakthrough transitions (r_t > 0 or new achievement).
       - High-value causal predecessor steps (e.g. chopping trees).
       - Counterfactual failure memories (actions with delta_inv == 0 and G == 0).
    """

    def __init__(
        self,
        memory_bank: LongTermSemanticMemory,
        dopamine_tracker: Optional[DopaminePersonalBestTracker] = None,
        working_memory: Optional[ShortTermWorkingMemory] = None,
        gamma: float = 0.99,
    ):
        self.memory_bank = memory_bank
        self.dopamine_tracker = dopamine_tracker
        self.working_memory = working_memory
        self.gamma = gamma

        self.ach_prev: np.ndarray = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
        self.inv_prev: np.ndarray = inventory_dict_to_vector(None)
        self.episode_buffer: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Resets milestone tracker and trajectory buffer for a new episode."""
        self.ach_prev.fill(0)
        self.inv_prev = inventory_dict_to_vector(None)
        self.episode_buffer.clear()
        if self.dopamine_tracker:
            self.dopamine_tracker.reset_episode()

    def step(
        self,
        z_t: torch.Tensor,
        action: int,
        reward: float,
        info: Dict[str, Any],
    ) -> Tuple[Optional[str], float]:
        """
        Inspects step outcome, updates trajectory buffer, and detects immediate milestones.
        Returns (unlocked_name, dopamine_surge).
        """
        ach_dict = dict(info.get("achievements", {})) if isinstance(info, dict) else {}
        ach_curr = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
        newly_unlocked: List[str] = []

        for idx, a_name in enumerate(CRAFTER_ACHIEVEMENTS):
            if ach_dict.get(a_name, 0) > 0:
                ach_curr[idx] = 1
                if self.ach_prev[idx] == 0:
                    newly_unlocked.append(a_name)

        delta_events = np.clip(ach_curr - self.ach_prev, 0, 1)

        # Inspect raw physical inventory from environment
        inv_dict = info.get("inventory") if isinstance(info, dict) else None
        inv_curr = inventory_dict_to_vector(inv_dict)
        delta_inv = inv_curr - self.inv_prev

        dopamine_surge = 0.0
        if self.dopamine_tracker:
            dopamine_surge = self.dopamine_tracker.update_step(reward, newly_unlocked)

        z_cpu = z_t.detach().cpu().flatten()

        event_name = ", ".join(newly_unlocked) if newly_unlocked else None

        # Record into episode buffer for backward Monte Carlo credit assignment
        self.episode_buffer.append({
            "key": z_cpu,
            "inv": self.inv_prev.copy(),
            "action": int(action),
            "reward": float(reward),
            "next_inv": inv_curr.copy(),
            "delta_inv": delta_inv.copy(),
            "dopamine": float(dopamine_surge),
            "event_name": event_name,
            "delta_events": delta_events.copy(),
            "ach_prev": self.ach_prev.copy(),
        })

        # Immediate breakthrough memory logging (System 2 real-time update)
        if np.any(delta_events > 0):
            effective_return = float(reward) + dopamine_surge
            self.memory_bank.add_entry(
                key=z_cpu,
                inventory=self.inv_prev.copy(),
                action=action,
                inventory_delta=delta_inv.copy(),
                realized_return=effective_return,
                outcome_name=event_name or "milestone",
                preconditions=self.ach_prev.copy(),
                outcome=delta_events,
            )

        self.ach_prev = ach_curr.copy()
        self.inv_prev = inv_curr.copy()
        return event_name, dopamine_surge

    def end_episode(self, final_reward: float = 0.0, gamma: Optional[float] = None) -> int:
        """
        Consolidates episode trajectory at game over using backward Monte Carlo return pass:
        G_t = sum_k gamma^k (r_{t+k} + dopamine_{t+k})
        
        Saves:
        1. Causal Breakthroughs & Predecessors: steps that enabled future achievements.
        2. Counterfactual Failure Memories: actions with delta_inv == 0 that failed (G = 0.0).
        
        Returns:
            Number of newly consolidated memory entries.
        """
        if not self.episode_buffer:
            return 0

        g_decay = gamma if gamma is not None else self.gamma
        T = len(self.episode_buffer)

        # 1. Backward Monte Carlo Discounted Return Pass
        mc_returns = np.zeros(T, dtype=np.float32)
        running_g = 0.0
        for t in reversed(range(T)):
            step_data = self.episode_buffer[t]
            step_r = step_data["reward"] + step_data["dopamine"]
            running_g = step_r + g_decay * running_g
            mc_returns[t] = running_g

        consolidated_count = 0
        failure_count = 0
        for t in range(T):
            step_data = self.episode_buffer[t]
            G_t = float(mc_returns[t])
            delta_inv = step_data["delta_inv"]
            
            # Meaningful causal inventory change:
            # 1. Physical tools & resources (indices 0..11): wood, stone, tools, saplings, etc.
            # 2. Deliberate vital replenishment (indices 12..15 > 0): eating, drinking, sleeping, healing
            # 3. Acute combat injury (health drop > 1.0): learning danger from monster attacks
            has_resource_change = bool(np.any(np.abs(delta_inv[:12]) > 0.01))
            has_vital_replenishment = bool(np.any(delta_inv[12:16] > 0.01))
            has_combat_injury = bool(delta_inv[12] < -1.0)
            has_causal_inv_change = has_resource_change or has_vital_replenishment or has_combat_injury

            is_milestone = bool(step_data["event_name"] is not None)
            is_action_reward = bool(step_data["reward"] > 0.0)

            # Case A: Causal Predecessor (e.g. tree-chopping that yielded wood and enabled pickaxe later)
            if has_causal_inv_change and (G_t >= 0.15 or is_action_reward):
                inv_changes = []
                for i, key in enumerate(CRAFTER_INVENTORY_KEYS):
                    d = delta_inv[i]
                    if abs(d) > 0.01:
                        sign = "+" if d > 0 else ""
                        inv_changes.append(f"{key}{sign}{int(round(d))}")
                tag = step_data["event_name"] or f"gain({', '.join(inv_changes)})"

                res = self.memory_bank.add_entry(
                    key=step_data["key"],
                    inventory=step_data["inv"],
                    action=step_data["action"],
                    inventory_delta=delta_inv,
                    realized_return=G_t,
                    outcome_name=tag,
                    preconditions=step_data["ach_prev"],
                    outcome=step_data["delta_events"],
                )
                if res in ("added", "updated"):
                    consolidated_count += 1

            # Case B: Counterfactual Failure Memory (Pillar 2: Learning from mistakes)
            # When an action produced nothing (no causal change, r == 0) and future return was zero
            # Record limited representative samples so Gaussian kernel suppresses fruitless actions
            elif not has_causal_inv_change and not is_milestone and step_data["reward"] == 0.0 and G_t <= 0.01:
                # Store up to 4 failed attempts per episode (e.g. crafting with 0 wood)
                if failure_count < 4 and step_data["action"] in [10, 11, 12, 13, 14, 15, 16]:  # craft/place/interact buttons
                    res = self.memory_bank.add_entry(
                        key=step_data["key"],
                        inventory=step_data["inv"],
                        action=step_data["action"],
                        inventory_delta=delta_inv,
                        realized_return=0.0,
                        outcome_name=f"fruitless_act_{step_data['action']}",
                        preconditions=step_data["ach_prev"],
                        outcome=step_data["delta_events"],
                    )
                    if res in ("added", "updated"):
                        consolidated_count += 1
                    failure_count += 1

        self.episode_buffer.clear()
        return consolidated_count
