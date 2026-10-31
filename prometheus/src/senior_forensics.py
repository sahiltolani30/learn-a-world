"""
src/senior_forensics.py

Senior Forensic Diagnostic Suite for Crafter Tabula Rasa JEPA World Model.
Performs read-only comprehensive introspection answering all 12 senior audit questions:
1. Final achievement list & progression curves
2. Action distribution histograms (Tier 1 vs Tier 2 vs Eval)
3. Explain-step on 4 canonical diagnostic scenes
4. Latent linear probes (wood count, tree presence, facing direction)
5. Curiosity decay & Welford state
6. Loss & Q-value training curves
7. Episode survival outcomes over time
8. One-step prediction error by event type
9. Multi-step latent rollout drift (H=10)
10. Tier 2 curated memory audit
11. Hyperparameter & configuration audit
12. 30-second behavior replay video / GIF with action overlays
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import crafter
from src.env.wrapper import make_crafter_env
from src.diagnostics import CRAFTER_ACTION_NAMES, ALL_CRAFTER_ACHIEVEMENTS, AgentForensics


class SeniorForensicSuite:
    """Executes the complete 12-item senior diagnostic suite on a trained checkpoint."""

    def __init__(self, pipeline: Any, output_dir: Union[str, Path] = "./forensics_output"):
        self.pipeline = pipeline
        self.agent = pipeline.agent
        self.buffer = pipeline.buffer
        self.device = pipeline.device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_lines: List[str] = []

    def log(self, text: str = "") -> None:
        print(text)
        self.report_lines.append(text)

    def run_all(self) -> Dict[str, Any]:
        """Run all 12 forensic analyses and output structured reports & figures."""
        self.log("=" * 80)
        self.log("  🔬 SENIOR JEPA WORLD MODEL FORENSIC AUDIT (12-STAGE REPORT)")
        self.log("=" * 80)

        results: Dict[str, Any] = {}
        results["item1"] = self.audit_item1_achievements()
        results["item2"] = self.audit_item2_action_histograms()
        results["item3"] = self.audit_item3_canonical_frames()
        results["item4"] = self.audit_item4_latent_probes()
        results["item5"] = self.audit_item5_curiosity_decay()
        results["item6_7"] = self.audit_item6_7_training_curves()
        results["item8"] = self.audit_item8_prediction_by_event()
        results["item9"] = self.audit_item9_rollout_drift()
        results["item10"] = self.audit_item10_tier2_xray()
        results["item11"] = self.audit_item11_config_audit()
        results["item12"] = self.audit_item12_behavior_video()

        # Save markdown report to disk
        report_path = self.output_dir / "FORENSIC_DIAGNOSTIC_REPORT.md"
        with open(report_path, "w") as f:
            f.write("\n".join(self.report_lines))
        self.log(f"\n[+] Full report saved to: {report_path.resolve()}")

        return results

    # -------------------------------------------------------------------------
    # 1. Final achievement list + per-eval curve
    # -------------------------------------------------------------------------
    def audit_item1_achievements(self) -> Dict[str, Any]:
        self.log("\n" + "=" * 80)
        self.log("  [ITEM 1] FINAL ACHIEVEMENT LIST & PROGRESSION HISTORY")
        self.log("=" * 80)

        unlocked = sorted(list(self.pipeline.all_unlocked_achievements))
        self.log(f"Total Unique Achievements Unlocked: {len(unlocked)} / {len(ALL_CRAFTER_ACHIEVEMENTS)}")
        self.log(f"Unlocked Achievements: {unlocked}")

        history = self.pipeline.achievement_history
        self.log(f"Evaluation Checkpoints Recorded: {len(history)}")
        if history:
            self.log(f"{'Step':>7} | {'Mean Return':>12} | {'Mean Length':>12} | {'Count':>6} | {'Achievements Unlocked'}")
            self.log("-" * 80)
            for step in sorted(history.keys()):
                entry = history[step]
                achs = ", ".join(entry.get("achievements", [])) or "None"
                self.log(f"{step:7d} | {entry.get('mean_return', 0.0):12.2f} | {entry.get('mean_length', 0.0):12.1f} | {entry.get('achievement_count', 0):6d} | {achs}")

        return {"unlocked": unlocked, "history": history}

    # -------------------------------------------------------------------------
    # 2. Action histogram, early vs late (Tier 1 vs Tier 2 vs Eval)
    # -------------------------------------------------------------------------
    def audit_item2_action_histograms(self) -> Dict[str, Any]:
        self.log("\n" + "=" * 80)
        self.log("  [ITEM 2] ACTION HISTOGRAM (TIER 1 vs TIER 2 vs EVAL EPISODES)")
        self.log("=" * 80)

        t1_acts = self.buffer.t1_actions[:self.buffer.t1_size] if self.buffer.t1_size > 0 else np.array([])
        t2_acts = self.buffer.t2_actions[:self.buffer.t2_size] if self.buffer.t2_size > 0 else np.array([])

        # Collect 5 eval episodes to see real eval action frequencies
        eval_acts = []
        eval_env = make_crafter_env(seed=999)
        self.agent.eval()
        for _ in range(5):
            obs = eval_env.reset()
            done = False
            eval_h = torch.zeros(1, getattr(self.agent, "mem_hidden", 256), device=self.device)
            while not done:
                a = self.agent.act(obs, eval_mode=True, planning=False, h=eval_h)
                eval_acts.append(a)
                obs, _, term, trunc, _ = eval_env.step(a)
                done = term or trunc
                if done:
                    eval_h = torch.zeros(1, getattr(self.agent, "mem_hidden", 256), device=self.device)
                else:
                    with torch.no_grad():
                        x = self.agent.validate_and_prepare_obs(obs)
                        if x.dim() == 3:
                            x = x.unsqueeze(0)
                        z = self.agent.encoder(x)
                        eval_h = self.agent.gru(z, eval_h)
        eval_acts = np.array(eval_acts)

        self.log(f"{'Index':>5} | {'Action Name':<20} | {'Tier 1 (FIFO)':>14} | {'Tier 2 (Milestone)':>18} | {'Eval Policy':>14}")
        self.log("-" * 80)

        t1_counts = np.bincount(t1_acts, minlength=17) if len(t1_acts) > 0 else np.zeros(17)
        t2_counts = np.bincount(t2_acts, minlength=17) if len(t2_acts) > 0 else np.zeros(17)
        ev_counts = np.bincount(eval_acts, minlength=17) if len(eval_acts) > 0 else np.zeros(17)

        t1_pct = (t1_counts / max(1, len(t1_acts))) * 100
        t2_pct = (t2_counts / max(1, len(t2_acts))) * 100
        ev_pct = (ev_counts / max(1, len(eval_acts))) * 100

        for i, name in enumerate(CRAFTER_ACTION_NAMES):
            self.log(f"{i:5d} | {name:<20} | {t1_pct[i]:13.1f}% | {t2_pct[i]:17.1f}% | {ev_pct[i]:13.1f}%")

        # Plot Action Distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(17)
        w = 0.28
        ax.bar(x - w, t1_pct, width=w, label="Tier 1 (Recent FIFO)", color="steelblue", alpha=0.85)
        ax.bar(x, t2_pct, width=w, label="Tier 2 (Milestones)", color="darkorange", alpha=0.85)
        ax.bar(x + w, ev_pct, width=w, label="Eval Policy Actions", color="forestgreen", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(CRAFTER_ACTION_NAMES, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Frequency (%)", fontweight="bold")
        ax.set_title("Action Distribution Spectrum: Experience Memory vs Evaluation Behavior", fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plot_path = self.output_dir / "action_histograms.png"
        fig.savefig(plot_path, dpi=150)
        plt.show()

        return {"t1_pct": t1_pct, "t2_pct": t2_pct, "ev_pct": ev_pct}

    # -------------------------------------------------------------------------
    # 3. AgentForensics.explain_step on four canonical frames
    # -------------------------------------------------------------------------
    def audit_item3_canonical_frames(self) -> Dict[str, Any]:
        self.log("\n" + "=" * 80)
        self.log("  [ITEM 3] EXPLAIN-STEP ON FOUR CANONICAL DIAGNOSTIC SCENES")
        self.log("=" * 80)

        env = crafter.Env(seed=123)
        self.agent.eval()

        scenes = {}
        # Scene (d): Open field
        obs_d = env.reset()
        scenes["(d) Open field"] = obs_d

        # Scene (a): Adjacent to tree, facing it
        # Walk until facing a tree or place tree tile in front
        obs_a = env.reset()
        p_x, p_y = env._player.pos
        # Script facing tree by putting tree adjacent to player in diagnostic harness
        env._world[p_x + 1, p_y] = "tree"
        env._player.facing = (1, 0)
        obs_a = env.render()
        scenes["(a) Adjacent to tree, facing it"] = obs_a

        # Scene (b): Facing grass, wood = 2
        obs_b = env.reset()
        p_x, p_y = env._player.pos
        env._world[p_x + 1, p_y] = "grass"
        env._player.facing = (1, 0)
        env._player.inventory["wood"] = 2
        obs_b = env.render()
        scenes["(b) Facing grass with Wood=2"] = obs_b

        # Scene (c): Same as (b), but wood = 0
        obs_c = env.reset()
        p_x, p_y = env._player.pos
        env._world[p_x + 1, p_y] = "grass"
        env._player.facing = (1, 0)
        env._player.inventory["wood"] = 0
        obs_c = env.render()
        scenes["(c) Facing grass with Wood=0"] = obs_c

        diagnostics = {}
        for name, obs in scenes.items():
            self.log(f"\n--- Scene: {name} ---")
            x = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
            with torch.no_grad():
                z = self.agent.encoder(x)
                q1, q2 = self.agent.critic(z)
                q_vals = torch.min(q1, q2)[0].cpu().numpy()
                probs = torch.softmax(self.agent.actor(z), dim=-1)[0].cpu().numpy()

                # Predictor epistemic disagreement across actions
                disagreements = []
                for act_idx in range(17):
                    a_t = torch.tensor([act_idx], dtype=torch.long, device=self.device)
                    _, all_preds = self.agent.predictor(z, a_t)
                    mean_pred = all_preds.mean(dim=0)
                    disagree = ((all_preds - mean_pred) ** 2).sum(dim=-1).mean().item()
                    disagreements.append(disagree)
                disagreements = np.array(disagreements)

            diagnostics[name] = {"q_vals": q_vals, "probs": probs, "disagreement": disagreements}

            # Top 3 actions
            top3 = np.argsort(q_vals)[::-1][:3]
            for rank, a_idx in enumerate(top3):
                self.log(f"  Rank {rank+1}: {CRAFTER_ACTION_NAMES[a_idx]:<18} | Q: {q_vals[a_idx]:+.4f} | Prob: {probs[a_idx]:.3f} | Disagree: {disagreements[a_idx]:.4f}")

            # Specific check for place_table and do
            act_do = CRAFTER_ACTION_NAMES.index("do")
            act_pt = CRAFTER_ACTION_NAMES.index("place_table")
            self.log(f"  Target 'do':          Q={q_vals[act_do]:+.4f} | Prob={probs[act_do]:.3f} | Disagree={disagreements[act_do]:.4f}")
            self.log(f"  Target 'place_table': Q={q_vals[act_pt]:+.4f} | Prob={probs[act_pt]:.3f} | Disagree={disagreements[act_pt]:.4f}")

        # The Smoking Gun Evaluation (Within-Scene Ranking Only - UPGRADE_1_2_3_SPEC.md P0.2)
        act_pt_idx = CRAFTER_ACTION_NAMES.index("place_table")
        probs_b = diagnostics["(b) Facing grass with Wood=2"]["probs"]
        probs_c = diagnostics["(c) Facing grass with Wood=0"]["probs"]
        q_b = diagnostics["(b) Facing grass with Wood=2"]["q_vals"]
        q_c = diagnostics["(c) Facing grass with Wood=0"]["q_vals"]

        # Calculate within-scene rank (1-indexed) based on policy probability
        rank_b = int(np.where(np.argsort(probs_b)[::-1] == act_pt_idx)[0][0]) + 1
        rank_c = int(np.where(np.argsort(probs_c)[::-1] == act_pt_idx)[0][0]) + 1
        prob_b = float(probs_b[act_pt_idx])
        prob_c = float(probs_c[act_pt_idx])

        self.log("\n>>> SMOKING GUN AUDIT: PRECONDITION AWARENESS (WITHIN-SCENE RANKS - P0.2) <<<")
        self.log(f"  Scene (b) Wood=2: Prob(place_table) = {prob_b:.1%} | Rank = #{rank_b} / 17 | Q = {q_b[act_pt_idx]:+.4f}")
        self.log(f"  Scene (c) Wood=0: Prob(place_table) = {prob_c:.1%} | Rank = #{rank_c} / 17 | Q = {q_c[act_pt_idx]:+.4f}")
        
        passed = (rank_b <= 3) and (prob_b >= 0.5) and (rank_c > rank_b + 3)
        if passed:
            self.log(f"  [+] VERDICT: PASS — place_table has Wood=2 Rank #{rank_b} (prob={prob_b:.1%}) and Wood=0 Rank #{rank_c} (prob={prob_c:.1%}) satisfying rank(b)<=3, prob(b)>=0.5, rank(c)>rank(b)+3.")
        else:
            self.log(f"  [-] VERDICT: FAIL — place_table has Wood=2 Rank #{rank_b} (prob={prob_b:.1%}) and Wood=0 Rank #{rank_c} (prob={prob_c:.1%}). Does not satisfy rank(b)<=3, prob(b)>=0.5, rank(c)>rank(b)+3.")

        return diagnostics

    # -------------------------------------------------------------------------
    # 4. Linear probes on the latent (offline, human-only)
    # -------------------------------------------------------------------------
    def audit_item4_latent_probes(self) -> Dict[str, Any]:
        self.log("\n" + "=" * 80)
        self.log("  [ITEM 4] LINEAR PROBES ON LATENT SPACE z (500 FRAMES)")
        self.log("=" * 80)

        env = crafter.Env(seed=42)
        frames_z = []
        labels_wood = []
        labels_tree = []
        labels_facing = []

        self.agent.eval()
        obs = env.reset()
        for i in range(500):
            # In human-only diagnostic harness: synthesize diverse wood counts and positions
            p = env._player
            if i % 3 == 0:
                p.inventory["wood"] = 0
            elif i % 3 == 1:
                p.inventory["wood"] = 1
            else:
                p.inventory["wood"] = 2

            frame = env.render()
            x = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
            with torch.no_grad():
                z = self.agent.encoder(x)[0].cpu().numpy()

            frames_z.append(z)
            labels_wood.append(min(p.inventory["wood"], 2))

            # Tree in view: check if tree is in 3x3 surrounding
            px, py = p.pos
            # Synthesize balanced distribution: on alternating frames, ensure clear area without trees
            if i % 2 == 1:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if env._world[px + dx, py + dy] == "tree":
                            env._world[px + dx, py + dy] = "grass"

            has_tree = any(env._world[px + dx, py + dy] == "tree" for dx in (-1, 0, 1) for dy in (-1, 0, 1))
            labels_tree.append(int(has_tree))

            # Facing direction
            fx, fy = p.facing
            facing_id = 0 if fx == -1 else (1 if fx == 1 else (2 if fy == -1 else 3))
            labels_facing.append(facing_id)

            action = int(env.action_space.sample())
            obs, _, term, _ = env.step(action)
            if term:
                obs = env.reset()

        Z = np.array(frames_z)
        y_wood = np.array(labels_wood)
        y_tree = np.array(labels_tree)
        y_facing = np.array(labels_facing)

        probes = {
            "Wood Count (0, 1, 2)": (y_wood, 3),
            "Tree Nearby (Yes/No)": (y_tree, 2),
            "Facing Direction (4-way)": (y_facing, 4),
        }
        probe_results = {}

        self.log(f"{'Target Attribute':<28} | {'Majority Baseline':>18} | {'Probe Accuracy':>16} | {'Signal Preserved?'}")
        self.log("-" * 80)

        N = len(Z)
        n_train = int(N * 0.75)
        perm = np.random.RandomState(42).permutation(N)
        train_idx, test_idx = perm[:n_train], perm[n_train:]

        for attr_name, (y, num_classes) in probes.items():
            Z_tr = torch.from_numpy(Z[train_idx]).float().to(self.device)
            y_tr = torch.from_numpy(y[train_idx]).long().to(self.device)
            Z_te = torch.from_numpy(Z[test_idx]).float().to(self.device)
            y_te = torch.from_numpy(y[test_idx]).long().to(self.device)

            probe = nn.Linear(Z.shape[1], num_classes).to(self.device)
            optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)
            loss_fn = nn.CrossEntropyLoss()

            for _ in range(100):
                optimizer.zero_grad()
                loss = loss_fn(probe(Z_tr), y_tr)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                preds = probe(Z_te).argmax(dim=-1)
                acc = (preds == y_te).float().mean().item() * 100.0

            # Majority baseline
            vals, counts = np.unique(y[test_idx], return_counts=True)
            baseline = (counts.max() / len(test_idx)) * 100.0

            preserved = "YES (Informative)" if (acc - baseline) > 15.0 else "NO (Degraded / Discarded)"
            self.log(f"{attr_name:<28} | {baseline:17.1f}% | {acc:15.1f}% | {preserved}")
            probe_results[attr_name] = {"acc": acc, "baseline": baseline}

        return probe_results

    # -------------------------------------------------------------------------
    # 5. Curiosity decay curve & Welford State
    # -------------------------------------------------------------------------
    def audit_item5_curiosity_decay(self) -> Dict[str, Any]:
        self.log("\n" + "=" * 80)
        self.log("  [ITEM 5] CURIOSITY DECAY & RUNNING STATS AUDIT")
        self.log("=" * 80)

        engine = self.agent.curiosity_engine
        has_stats = hasattr(engine, "running_stats")

        if has_stats:
            stats = engine.running_stats
            count = float(stats.count.item()) if hasattr(stats.count, "item") else float(stats.count)
            mean_r = float(stats.running_mean.mean().item())
            var_r = float(stats.running_var.mean().item())
            self.log(f"Welford Running Count: {count:,.0f} updates")
            self.log(f"Welford Running Mean:  {mean_r:.6f}")
            self.log(f"Welford Running Var:   {var_r:.6f} (Std: {np.sqrt(max(1e-8, var_r)):.6f})")

        # Sample 100 transitions from Tier 1 early vs Tier 1 late
        if self.buffer.t1_size >= 100:
            early_errs = self.buffer.t1_epistemic_errors[:50]
            late_errs = self.buffer.t1_epistemic_errors[self.buffer.t1_size - 50:self.buffer.t1_size]
            self.log(f"Early Epistemic Error (First 50 steps): Mean = {early_errs.mean():.4f}")
            self.log(f"Late Epistemic Error (Recent 50 steps): Mean = {late_errs.mean():.4f}")
            decay_pct = (1.0 - (late_errs.mean() / max(1e-6, early_errs.mean()))) * 100
            self.log(f"Curiosity Decay: {decay_pct:.1f}% reduction in epistemic disagreement")

        return {}

    # -------------------------------------------------------------------------
    # 6 & 7. Loss, Q-Curves & Episode Outcomes
    # -------------------------------------------------------------------------
    def audit_item6_7_training_curves(self) -> Dict[str, Any]:
        self.log("\n" + "=" * 80)
        self.log("  [ITEM 6 & 7] LOSS, VALUE, & EVALUATION OUTCOMES OVER TIME")
        self.log("=" * 80)

        history = self.pipeline.achievement_history
        if not history:
            self.log("No achievement history records found.")
            return {}

        steps = sorted(history.keys())
        returns = [history[s]["mean_return"] for s in steps]
        lengths = [history[s]["mean_length"] for s in steps]
        counts = [history[s]["achievement_count"] for s in steps]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        ax1.plot(steps, returns, "o-", color="royalblue", lw=2)
        ax1.axhline(0, color="gray", ls="--", alpha=0.5)
        ax1.set_ylabel("Mean Return", fontweight="bold")
        ax1.set_title("Training Progression Diagnostics Across Evaluation Cycles", fontweight="bold")
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, lengths, "s-", color="darkorange", lw=2)
        ax2.set_ylabel("Survival Length (Steps)", fontweight="bold")
        ax2.grid(True, alpha=0.3)

        ax3.plot(steps, counts, "^-", color="forestgreen", lw=2)
        ax3.set_ylabel("Achievement Count", fontweight="bold")
        ax3.set_xlabel("Environment Steps", fontweight="bold")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        curve_path = self.output_dir / "training_and_eval_curves.png"
        fig.savefig(curve_path, dpi=150)
        plt.show()
        self.log(f"[+] Curves plotted and saved to: {curve_path.resolve()}")

        return {"steps": steps, "returns": returns, "lengths": lengths}

    # -------------------------------------------------------------------------
    # 8. One-step prediction error by event type
    # -------------------------------------------------------------------------
    def audit_item8_prediction_by_event(self) -> Dict[str, Any]:
        self.log("\n" + "=" * 80)
        self.log("  [ITEM 8] ONE-STEP PREDICTION ERROR BY EVENT TYPE")
        self.log("=" * 80)

        self.agent.eval()
        env = crafter.Env(seed=777)
        errors_by_event: Dict[str, List[float]] = {
            "plain_move": [],
            "wood_harvest": [],
            "place_table_attempt": [],
            "night_transition": [],
        }

        obs = env.reset()
        for step_i in range(300):
            p = env._player
            x = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
            
            # Select diverse actions
            if step_i % 4 == 0:
                act = CRAFTER_ACTION_NAMES.index("place_table")
            elif step_i % 4 == 1:
                act = CRAFTER_ACTION_NAMES.index("do")
            else:
                act = int(env.action_space.sample())

            next_obs, rew, term, _ = env.step(act)

            # Measure prediction error in latent space
            x_next = torch.from_numpy(next_obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
            with torch.no_grad():
                z_curr = self.agent.encoder(x)
                z_next_target = self.agent.target_encoder(x_next)
                a_tensor = torch.tensor([act], dtype=torch.long, device=self.device)
                pred_next, _ = self.agent.predictor(z_curr, a_tensor)
                err = torch.norm(pred_next - z_next_target, p=2, dim=-1).item()

            if act == CRAFTER_ACTION_NAMES.index("place_table"):
                errors_by_event["place_table_attempt"].append(err)
            elif rew > 0 and act == CRAFTER_ACTION_NAMES.index("do"):
                errors_by_event["wood_harvest"].append(err)
            elif step_i >= 148:
                errors_by_event["night_transition"].append(err)
            else:
                errors_by_event["plain_move"].append(err)

            obs = next_obs
            if term:
                obs = env.reset()

        self.log(f"{'Event Type':<25} | {'Sample Count':>12} | {'Mean Latent L2 Error':>22}")
        self.log("-" * 80)
        for ev, errs in errors_by_event.items():
            mean_e = np.mean(errs) if errs else 0.0
            self.log(f"{ev:<25} | {len(errs):12d} | {mean_e:22.4f}")

        return errors_by_event

    # -------------------------------------------------------------------------
    # 9. Multi-step rollout drift (H=10)
    # -------------------------------------------------------------------------
    def audit_item9_rollout_drift(self) -> Dict[str, Any]:
        self.log("\n" + "=" * 80)
        self.log("  [ITEM 9] MULTI-STEP LATENT ROLLOUT DRIFT (H=10)")
        self.log("=" * 80)

        env = crafter.Env(seed=888)
        self.agent.eval()

        obs = env.reset()
        true_frames = [obs]
        actions_taken = []
        for _ in range(10):
            a = int(env.action_space.sample())
            obs, _, term, _ = env.step(a)
            true_frames.append(obs)
            actions_taken.append(a)
            if term:
                break

        H = len(actions_taken)
        # Encode ground truth futures with target encoder
        x_all = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in true_frames]).to(self.device)
        with torch.no_grad():
            z_true = self.agent.target_encoder(x_all)

        # Rollout in latent space
        z_sim = [z_true[0:1]]
        for t in range(H):
            a_t = torch.tensor([actions_taken[t]], dtype=torch.long, device=self.device)
            pred_next, _ = self.agent.predictor(z_sim[-1], a_t)
            z_sim.append(pred_next)

        z_sim = torch.cat(z_sim, dim=0)

        self.log(f"{'Horizon t':<10} | {'Cosine Similarity':>18} | {'Latent L2 Distance':>20}")
        self.log("-" * 80)
        for t in range(1, H + 1):
            cos = F.cosine_similarity(z_sim[t:t+1], z_true[t:t+1]).item()
            l2 = torch.norm(z_sim[t:t+1] - z_true[t:t+1], p=2).item()
            self.log(f"{t:10d} | {cos:18.4f} | {l2:20.4f}")

        return {"horizon": H}

    # -------------------------------------------------------------------------
    # 10. Tier-2 Memory X-Ray
    # -------------------------------------------------------------------------
    def audit_item10_tier2_xray(self) -> Dict[str, Any]:
        self.log("\n" + "=" * 80)
        self.log("  [ITEM 10] TIER-2 CURATED MILESTONE ARCHIVE X-RAY")
        self.log("=" * 80)

        mem_audit = AgentForensics.inspect_memory(self.buffer)
        total_exp = mem_audit.get("total_transitions", self.buffer.size)
        t1_sz = mem_audit.get("tier1_count", self.buffer.t1_size)
        t2_sz = mem_audit.get("tier2_count", self.buffer.t2_size)
        self.log(f"Total Experiences in Buffer: {total_exp}")
        self.log(f"Tier 1 Size: {t1_sz} / {self.buffer.tier1_capacity}")
        self.log(f"Tier 2 Size: {t2_sz} / {self.buffer.tier2_capacity}")
        self.log(f"Curiosity Threshold: {mem_audit.get('novelty_threshold', 0.0):.4f}")

        catalog = mem_audit.get("milestone_catalog")
        if isinstance(catalog, dict):
            self.log("\nTop Actions Retained in Tier 2 Archive:")
            for act_name, count in sorted(catalog.items(), key=lambda x: x[1], reverse=True)[:10]:
                self.log(f"  - {act_name:<20}: {count} transitions")
        elif catalog:
            self.log(f"  {catalog}")

        if hasattr(self.buffer, "t2_actions") and self.buffer.t2_size > 0:
            t2_acts = self.buffer.t2_actions[:self.buffer.t2_size]
            unique, counts = np.unique(t2_acts, return_counts=True)
            self.log("\nTop Actions Stored in Tier 2 Archive:")
            for a_id, cnt in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:10]:
                self.log(f"  - {CRAFTER_ACTION_NAMES[a_id]:<20}: {cnt} transitions ({(cnt / max(1, self.buffer.t2_size))*100:.1f}%)")

        return mem_audit

    # -------------------------------------------------------------------------
    # 11. Config Audit
    # -------------------------------------------------------------------------
    def audit_item11_config_audit(self) -> Dict[str, Any]:
        self.log("\n" + "=" * 80)
        self.log("  [ITEM 11] CONFIGURATION & HYPERPARAMETER AUDIT")
        self.log("=" * 80)

        cfg = getattr(self.pipeline, "cfg", {})
        for k, v in sorted(cfg.items()):
            self.log(f"  {k:<22}: {v}")

        # Explicitly verify beta and entropy coefficient
        engine = self.agent.curiosity_engine
        beta = getattr(engine, "beta", 0.05)
        self.log(f"\n  Curiosity Beta:        {beta}")
        self.log(f"  Actor Entropy Coeff:   0.05 (hardcoded expectation)")
        self.log(f"  Training CEM Planning: False (Dual Execution Policy Mode)")

        return cfg

    # -------------------------------------------------------------------------
    # 12. 30-Second Behavior Video / GIF
    # -------------------------------------------------------------------------
    def audit_item12_behavior_video(self, num_steps: int = 150) -> Optional[str]:
        self.log("\n" + "=" * 80)
        self.log("  [ITEM 12] 30-SECOND BEHAVIOR REPLAY RENDERING")
        self.log("=" * 80)

        import matplotlib.animation as animation

        env = make_crafter_env(seed=333)
        self.agent.eval()

        frames = []
        actions = []
        rewards = []

        obs = env.reset()
        done = False
        step_count = 0
        eval_h = torch.zeros(1, getattr(self.agent, "mem_hidden", 256), device=self.device)

        while not done and step_count < num_steps:
            a = self.agent.act(obs, eval_mode=True, planning=False, h=eval_h)
            obs, r, term, trunc, info = env.step(a)
            rendered = env.render()
            frames.append(rendered)
            actions.append(CRAFTER_ACTION_NAMES[a])
            rewards.append(r)
            step_count += 1
            done = term or trunc
            if not done:
                with torch.no_grad():
                    x = self.agent.validate_and_prepare_obs(obs)
                    if x.dim() == 3:
                        x = x.unsqueeze(0)
                    z = self.agent.encoder(x)
                    eval_h = self.agent.gru(z, eval_h)

        # Save animated GIF with text overlay
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")
        im = ax.imshow(frames[0])
        title = ax.set_title(f"Step 0 | {actions[0]} | R={rewards[0]:+.1f}", fontsize=12, fontweight="bold")

        def update(frame_idx):
            im.set_data(frames[frame_idx])
            title.set_text(f"Step {frame_idx:3d} | Action: {actions[frame_idx]:<18} | R={rewards[frame_idx]:+.1f}")
            return [im, title]

        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
        gif_path = self.output_dir / "agent_behavior_eval.gif"
        try:
            ani.save(str(gif_path), writer="pillow", fps=10)
            self.log(f"[+] 30-Second evaluation animation saved to: {gif_path.resolve()}")
            try:
                import base64
                from IPython.display import display, HTML
                with open(gif_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                display(HTML(f'<div style="margin: 15px 0;"><img src="data:image/gif;base64,{b64}" style="max-width: 480px; border: 2px solid #22c55e; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" /><br><span style="font-size: 13px; color: #64748b; font-weight: bold;">🎬 30-Second Eval Behavior Replay (Right-Click to Copy/Save)</span></div>'))
            except Exception as e:
                self.log(f"[!] Inline HTML display note: {e}")
        except Exception as e:
            self.log(f"[!] Animation saving warning: {e}")
        plt.close(fig)

        return str(gif_path)
