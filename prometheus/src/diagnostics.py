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


ALL_CRAFTER_ACHIEVEMENTS: List[str] = [
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
]


def compute_crafter_score(success_rates: Dict[str, float]) -> float:
    """
    Official Crafter Score formula (Hafner, 2021):
    Score = exp( (1/22) * sum_{i=1}^{22} ln(1 + S_i) ) - 1
    where S_i is in percent [0.0, 100.0].
    """
    log_sum = 0.0
    for ach in ALL_CRAFTER_ACHIEVEMENTS:
        s_i = success_rates.get(ach, 0.0)
        log_sum += np.log(1.0 + s_i)
    return float(np.exp(log_sum / len(ALL_CRAFTER_ACHIEVEMENTS)) - 1.0)


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
        t1_size = getattr(buffer, "t1_size", 0)
        t2_size = getattr(buffer, "t2_size", 0)
        total_len = len(buffer) if hasattr(buffer, "__len__") else (t1_size + t2_size)
        t1_cap = getattr(buffer, "tier1_capacity", 100000)
        t2_cap = getattr(buffer, "tier2_capacity", 20000)

        # Audit milestones in Tier 2
        milestone_count = 0
        if hasattr(buffer, "t2_is_milestone") and buffer.t2_is_milestone is not None:
            milestone_count = int(np.sum(buffer.t2_is_milestone[:t2_size]))

        novelty_thresh = buffer.get_novelty_threshold() if hasattr(buffer, "get_novelty_threshold") else 0.0

        return {
            "total_transitions": total_len,
            "tier1_count": t1_size,
            "tier1_capacity": t1_cap,
            "tier2_count": t2_size,
            "tier2_capacity": t2_cap,
            "tier2_utilization_pct": (t2_size / max(1, t2_cap)) * 100.0,
            "tier2_milestones": milestone_count,
            "milestone_catalog": f"{milestone_count} milestone transitions archived",
            "novelty_threshold": float(novelty_thresh),
        }

    def plot_forensic_panel(
        self,
        obs: Any,
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
        # Normalize/format observation for display (supports Tensor, ResetObs, ndarray)
        if isinstance(obs, torch.Tensor):
            img = obs.detach().cpu().numpy()
        else:
            img = np.asarray(obs)
        img = img.copy()

        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
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

    def run_benchmark(
        self,
        num_episodes: int = 50,
        planning: bool = False,
        base_seed: int = 1000,
        stochastic: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs N=50 independent benchmark evaluation games to compute:
        1. Official Crafter Score (geometric mean of success rates across 22 achievements)
        2. Per-game average achievements unlocked
        3. Mean survival steps and episodic returns
        4. Success rate for every single achievement
        """
        from src.env.wrapper import make_crafter_env

        self.agent.eval()
        ep_returns = []
        ep_lengths = []
        ep_achievements = []
        ach_counts = {ach: 0 for ach in ALL_CRAFTER_ACHIEVEMENTS}

        for ep_idx in range(num_episodes):
            seed = base_seed + ep_idx * 137
            env = make_crafter_env(seed=seed)
            obs = env.reset()
            ep_ret = 0.0
            ep_steps = 0
            done = False
            unlocked_this_ep = set()

            while not done:
                action = self.agent.act(obs, eval_mode=not stochastic, planning=planning)
                obs, reward, term, trunc, info = env.step(action)
                ep_ret += float(reward)
                ep_steps += 1
                done = term or trunc

                eval_info = info.get("eval_info", {})
                for ach in eval_info.get("unlocked_achievements", []):
                    unlocked_this_ep.add(ach)

            ep_returns.append(ep_ret)
            ep_lengths.append(ep_steps)
            ep_achievements.append(unlocked_this_ep)

            for ach in unlocked_this_ep:
                if ach in ach_counts:
                    ach_counts[ach] += 1

        success_rates = {
            ach: (ach_counts[ach] / num_episodes) * 100.0
            for ach in ALL_CRAFTER_ACHIEVEMENTS
        }
        crafter_score = compute_crafter_score(success_rates)
        mean_achs_per_game = float(np.mean([len(s) for s in ep_achievements]))
        all_unlocked = sorted(list(set.union(*ep_achievements))) if ep_achievements else []

        print(f"Policy: {'stochastic' if stochastic else 'greedy'} (N={num_episodes} games)")
        return {
            "num_episodes": num_episodes,
            "policy": "stochastic" if stochastic else "greedy",
            "crafter_score": crafter_score,
            "mean_return": float(np.mean(ep_returns)),
            "std_return": float(np.std(ep_returns)),
            "mean_length": float(np.mean(ep_lengths)),
            "std_length": float(np.std(ep_lengths)),
            "mean_achievements_per_game": mean_achs_per_game,
            "total_unique_achievements": len(all_unlocked),
            "unlocked_list": all_unlocked,
            "success_rates": success_rates,
            "episode_details": [
                {
                    "episode": i + 1,
                    "return": ep_returns[i],
                    "steps": ep_lengths[i],
                    "achievements": sorted(list(ep_achievements[i])),
                }
                for i in range(num_episodes)
            ],
        }

    def plot_benchmark_results(
        self,
        benchmark_results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plots the official Crafter benchmark achievement success rates.
        """
        success_rates = benchmark_results["success_rates"]
        sorted_achs = sorted(ALL_CRAFTER_ACHIEVEMENTS, key=lambda a: success_rates[a], reverse=True)
        rates = [success_rates[a] for a in sorted_achs]
        labels = [a.replace("_", " ") for a in sorted_achs]

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ["#2ecc71" if r > 0 else "#bdc3c7" for r in rates]
        bars = ax.barh(range(len(sorted_achs)), rates, color=colors, edgecolor="black", alpha=0.85)
        ax.set_yticks(range(len(sorted_achs)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Success Rate (%) across Evaluation Games", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 105)
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        title = (
            f"Official Crafter Benchmark Evaluation ({benchmark_results['num_episodes']} Games)\n"
            f"Official Crafter Score: {benchmark_results['crafter_score']:.2f}% | "
            f"Mean Achievements/Game: {benchmark_results['mean_achievements_per_game']:.2f} | "
            f"Mean Return: {benchmark_results['mean_return']:.2f}"
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(rate + 1.0, bar.get_y() + bar.get_height() / 2, f"{rate:.1f}%", va="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


def evaluate_counterfactual_verdict(
    rank_b: int,
    prob_b: float,
    rank_c: int,
) -> Dict[str, Any]:
    """
    Evaluate counterfactual probe verdict strictly using within-scene ranks (UPGRADE_1_2_3_SPEC.md P0.2).
    PASS if: rank(b) <= 3 AND prob(b) >= 0.5 AND rank(c) > rank(b) + 3.
    """
    passed = bool((rank_b <= 3) and (prob_b >= 0.5) and (rank_c > rank_b + 3))
    return {
        "passed": passed,
        "rank_b": rank_b,
        "prob_b": prob_b,
        "rank_c": rank_c,
        "verdict": "PASS" if passed else "FAIL",
    }


def run_counterfactual_probe(agent: Any) -> Dict[str, Any]:
    """
    Executes counterfactual scene probing comparing:
    - Scene (b): Facing grass with Wood=2 (affordance present -> place_table expected)
    - Scene (c): Facing grass with Wood=0 (precondition absent -> place_table suppressed)
    Verdict computed purely via within-scene ranking (P0.2).
    """
    import crafter
    from src.agent.actor_critic import symexp

    agent.eval()
    dev = getattr(agent, "device", torch.device("cpu"))
    act_pt = CRAFTER_ACTION_NAMES.index("place_table")

    # Construct Scene (b): Facing grass with Wood=2
    env_b = crafter.Env(seed=42)
    obs_b = env_b.reset()
    p_b = env_b._player
    p_b.inventory["wood"] = 2
    px, py = p_b.pos
    dx, dy = p_b.facing
    env_b._world[px + dx, py + dy] = "grass"
    obs_b = env_b.render()
    with torch.no_grad():
        t_b = torch.from_numpy(obs_b.transpose((2, 0, 1))).unsqueeze(0).to(dev).float() / 255.0
        z_b = agent.encoder(t_b)
        readout_b = agent.get_readout(z_b) if hasattr(agent, "get_readout") else z_b
        probs_b = agent.actor.get_probs(readout_b).squeeze(0).cpu().numpy()
        q1_b, q2_b = agent.critic(readout_b)
        q_b = torch.min(q1_b, q2_b).squeeze(0).cpu().numpy()

    # Construct Scene (c): Facing grass with Wood=0
    env_c = crafter.Env(seed=42)
    obs_c = env_c.reset()
    p_c = env_c._player
    p_c.inventory["wood"] = 0
    px, py = p_c.pos
    dx, dy = p_c.facing
    env_c._world[px + dx, py + dy] = "grass"
    obs_c = env_c.render()
    with torch.no_grad():
        t_c = torch.from_numpy(obs_c.transpose((2, 0, 1))).unsqueeze(0).to(dev).float() / 255.0
        z_c = agent.encoder(t_c)
        readout_c = agent.get_readout(z_c) if hasattr(agent, "get_readout") else z_c
        probs_c = agent.actor.get_probs(readout_c).squeeze(0).cpu().numpy()
        q1_c, q2_c = agent.critic(readout_c)
        q_c = torch.min(q1_c, q2_c).squeeze(0).cpu().numpy()

    # Calculate within-scene rank (1-indexed) based on policy probabilities / Q
    rank_b = int(np.where(np.argsort(probs_b)[::-1] == act_pt)[0][0]) + 1
    rank_c = int(np.where(np.argsort(probs_c)[::-1] == act_pt)[0][0]) + 1
    prob_b = float(probs_b[act_pt])
    prob_c = float(probs_c[act_pt])

    verdict_info = evaluate_counterfactual_verdict(rank_b, prob_b, rank_c)
    verdict_info.update({
        "prob_b": prob_b,
        "prob_c": prob_c,
        "q_b_pt": float(q_b[act_pt]),
        "q_c_pt": float(q_c[act_pt]),
    })
    return verdict_info


def run_stagewise_policy_probe(agent: Any) -> Dict[str, Any]:
    """
    Probes neural activations across the four canonical tech-tree stages (P0.6):
    1. Fresh spawn (0 Wood, bare hands)
    2. Wood in inventory (2 Wood, no table)
    3. Crafting station (1 Wood, facing table)
    4. Mining ready (has wood pickaxe, facing stone)
    """
    import crafter

    agent.eval()
    dev = getattr(agent, "device", torch.device("cpu"))
    stage_results = {}

    stages = [
        ("Stage 1: Fresh Spawn", 0, None, "grass"),
        ("Stage 2: Wood in Inventory (Wood=2)", 2, None, "grass"),
        ("Stage 3: Facing Table (Wood=1)", 1, "table", "table"),
        ("Stage 4: Mining Ready (Wood Pickaxe, Facing Stone)", 0, None, "stone"),
    ]

    for name, wood_cnt, facing_item, target_tile in stages:
        env = crafter.Env(seed=42)
        env.reset()
        p = env._player
        p.inventory["wood"] = wood_cnt
        if name.startswith("Stage 4"):
            p.inventory["wood_pickaxe"] = 1
        px, py = p.pos
        dx, dy = p.facing
        env._world[px + dx, py + dy] = target_tile
        frame = env.render()

        with torch.no_grad():
            t = torch.from_numpy(frame.transpose((2, 0, 1))).unsqueeze(0).to(dev).float() / 255.0
            z = agent.encoder(t)
            readout = agent.get_readout(z) if hasattr(agent, "get_readout") else z
            probs = agent.actor.get_probs(readout).squeeze(0).cpu().numpy()
            q1, q2 = agent.critic(readout)
            q_vals = torch.min(q1, q2).squeeze(0).cpu().numpy()

        sorted_indices = np.argsort(probs)[::-1]
        top_actions = [
            {
                "rank": r + 1,
                "action": CRAFTER_ACTION_NAMES[idx],
                "prob": float(probs[idx]),
                "q": float(q_vals[idx]),
            }
            for r, idx in enumerate(sorted_indices[:5])
        ]
        stage_results[name] = top_actions

    return stage_results
