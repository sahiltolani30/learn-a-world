"""
src/diagnostics.py

A-to-Z Forensic Diagnostic Suite for Tabula Rasa JEPA World Model on Crafter.
Provides deep, transparent introspection into:
1. Perception (Visual Attention / Saliency Heatmaps)
2. Latent Dynamics (4-Head Predictor Ensemble Uncertainty & Imagination)
3. Action Valuation (TwinCritic Q-Values across all 17 Crafter Actions)
4. Policy Probabilities (ActorPrior Action Distribution)
5. Memory Forensic X-Ray (Two-Tier Buffer Inventory & Milestone Retention)
6. Counterfactual Probing ("Why did the model pick Action X instead of Action Y?")
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

CRAFTER_ACTION_NAMES: List[str] = [
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
]


class AgentForensics:
    """
    Forensic Analyzer for inspecting every internal operation of the JEPA Agent.
    """

    def __init__(self, agent: Any, action_names: Optional[List[str]] = None):
        self.agent = agent
        self.action_names = action_names or CRAFTER_ACTION_NAMES
        self.num_actions = len(self.action_names)
        self.device = getattr(agent, "device", torch.device("cpu"))

    def explain_step(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        compute_saliency: bool = True,
    ) -> Dict[str, Any]:
        """
        Deep introspection of a single decision step.

        Returns:
            Dictionary containing:
            - 'latent_z': (256,) latent representation
            - 'action_probs': (17,) probability distribution
            - 'q_values': (17,) Q-value estimates
            - 'uncertainties': (17,) predictor ensemble disagreement
            - 'saliency_map': (64, 64) visual attention heatmap (if requested)
            - 'ranked_actions': list of tuples (rank, name, q_value, prob, uncertainty)
            - 'best_action': top action index
            - 'best_action_name': top action name
        """
        self.agent.eval()

        # 1. Prepare raw observation
        x = self.agent.validate_and_prepare_obs(obs)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (1, C, H, W)

        # 2. Compute visual saliency via input gradient (if enabled)
        saliency_map = None
        if compute_saliency:
            x_grad = x.clone().detach().requires_grad_(True)
            z_grad = self.agent.encoder(x_grad)
            q1_grad, q2_grad = self.agent.critic(z_grad)
            q_min_grad = torch.min(q1_grad, q2_grad)
            top_action = torch.argmax(q_min_grad, dim=-1)
            q_target = q_min_grad[0, top_action]
            q_target.backward()

            if x_grad.grad is not None:
                # Max gradient magnitude across color channels
                saliency = torch.max(torch.abs(x_grad.grad[0]), dim=0)[0].detach().cpu().numpy()
                s_min, s_max = saliency.min(), saliency.max()
                if s_max > s_min:
                    saliency_map = (saliency - s_min) / (s_max - s_min)
                else:
                    saliency_map = np.zeros_like(saliency)

        # 3. Forward passes (no grad)
        with torch.no_grad():
            z = self.agent.encoder(x)
            q1, q2 = self.agent.critic(z)
            q_vals = torch.min(q1, q2)[0].cpu().numpy()

            actor_logits = self.agent.actor(z)
            probs = torch.softmax(actor_logits, dim=-1)[0].cpu().numpy()

            # Predictor Ensemble Disagreement across all discrete actions
            uncertainties = np.zeros(self.num_actions, dtype=np.float32)
            for a_idx in range(self.num_actions):
                a_tensor = torch.tensor([a_idx], dtype=torch.long, device=self.device)
                _, all_preds = self.agent.predictor(z, a_tensor)  # (num_heads, 1, 256)
                disagreement = self.agent.predictor.compute_epistemic_disagreement(all_preds)
                uncertainties[a_idx] = float(disagreement.item())

        best_idx = int(np.argmax(q_vals))

        # Build ranked table
        ranked_actions = []
        sorted_indices = np.argsort(-q_vals)
        for rank, idx in enumerate(sorted_indices, start=1):
            ranked_actions.append({
                "rank": rank,
                "index": int(idx),
                "name": self.action_names[idx] if idx < len(self.action_names) else f"action_{idx}",
                "q_value": float(q_vals[idx]),
                "prob": float(probs[idx]),
                "uncertainty": float(uncertainties[idx]),
            })

        return {
            "latent_z": z[0].cpu().numpy(),
            "action_probs": probs,
            "q_values": q_vals,
            "uncertainties": uncertainties,
            "saliency_map": saliency_map,
            "ranked_actions": ranked_actions,
            "best_action": best_idx,
            "best_action_name": self.action_names[best_idx],
        }

    @staticmethod
    def inspect_memory(buffer: Any) -> Dict[str, Any]:
        """
        Forensic X-Ray of the Two-Tier Replay Memory Buffer.
        Audits stored transitions, milestone archive health, and retention ratios.
        """
        t1_len = len(getattr(buffer, "tier1", []))
        t2_len = len(getattr(buffer, "tier2", []))
        total_len = len(buffer)
        t2_capacity = getattr(buffer, "tier2_capacity", 0)
        t1_capacity = getattr(buffer, "tier1_capacity", 0)

        # Audit milestones present in Tier 2
        milestone_counts: Dict[str, int] = {}
        if hasattr(buffer, "tier2"):
            for trans in buffer.tier2:
                info = getattr(trans, "info", {})
                if isinstance(info, dict):
                    for ach in info.get("unlocked_achievements", []):
                        milestone_counts[ach] = milestone_counts.get(ach, 0) + 1

        return {
            "total_transitions": total_len,
            "tier1_count": t1_len,
            "tier1_capacity": t1_capacity,
            "tier2_count": t2_len,
            "tier2_capacity": t2_capacity,
            "tier2_utilization_pct": (t2_len / max(1, t2_capacity)) * 100.0,
            "milestone_catalog": milestone_counts,
        }

    def plot_forensic_panel(
        self,
        obs: np.ndarray,
        explanation: Dict[str, Any],
        title_suffix: str = "",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Renders a comprehensive 4-panel visual diagnostic card:
        1. Raw Crafter RGB Canvas
        2. Visual Attention Saliency Heatmap
        3. Q-Value Landscape across all 17 actions
        4. World Model Epistemic Uncertainty across all 17 actions
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Project Prometheus JEPA Decision Forensics {title_suffix}", fontsize=14, fontweight="bold")

        # 1. Raw Crafter Visual Frame
        ax_img = axes[0, 0]
        # Normalize/format observation for display
        img = obs.copy()
        if img.shape[0] in (1, 3, 4) and img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        ax_img.imshow(img)
        ax_img.set_title("1. Visual Input Screen (64x64 RGB)", fontsize=11, fontweight="bold")
        ax_img.axis("off")

        # 2. Visual Attention Saliency Heatmap
        ax_sal = axes[0, 1]
        sal = explanation.get("saliency_map")
        if sal is not None:
            ax_sal.imshow(img)
            im = ax_sal.imshow(sal, cmap="hot", alpha=0.6)
            plt.colorbar(im, ax=ax_sal, fraction=0.046, pad=0.04)
            ax_sal.set_title("2. Visual Attention Heatmap (|dQ/dx|)", fontsize=11, fontweight="bold")
        else:
            ax_sal.text(0.5, 0.5, "Saliency Not Computed", ha="center", va="center")
            ax_sal.set_title("2. Visual Attention Heatmap", fontsize=11, fontweight="bold")
        ax_sal.axis("off")

        # 3. Action Q-Values Bar Chart
        ax_q = axes[1, 0]
        q_vals = explanation["q_values"]
        names = [name.replace("_", "\n") for name in self.action_names]
        colors = ["#2ecc71" if i == explanation["best_action"] else "#3498db" for i in range(len(q_vals))]
        bars = ax_q.bar(range(len(q_vals)), q_vals, color=colors, edgecolor="black", alpha=0.85)
        ax_q.set_xticks(range(len(q_vals)))
        ax_q.set_xticklabels(names, rotation=90, fontsize=8)
        ax_q.set_ylabel("TwinCritic Q(z, a)", fontsize=10)
        ax_q.set_title(f"3. Action Value Spectrum (Chosen: '{explanation['best_action_name']}')", fontsize=11, fontweight="bold")
        ax_q.grid(axis="y", linestyle="--", alpha=0.5)

        # 4. Predictor Ensemble Uncertainty
        ax_unc = axes[1, 1]
        unc = explanation["uncertainties"]
        bars2 = ax_unc.bar(range(len(unc)), unc, color="#e74c3c", edgecolor="black", alpha=0.8)
        ax_unc.set_xticks(range(len(unc)))
        ax_unc.set_xticklabels(names, rotation=90, fontsize=8)
        ax_unc.set_ylabel("Predictor Head Disagreement", fontsize=10)
        ax_unc.set_title("4. World Model Uncertainty / Curiosity", fontsize=11, fontweight="bold")
        ax_unc.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
