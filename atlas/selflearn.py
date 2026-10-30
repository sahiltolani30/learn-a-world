"""SelfLearnLoop: iterative self-supervised play, training, and goal mining loop.

Resumable cycle:
[Collect episodes -> Shard -> Cache latents -> Train heads -> Mine GoalBank & Lift -> Screen -> Checkpoint]
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from atlas.arbiter import AtlasAgent
from atlas.audit_pack import AuditPackExporter
from atlas.backbone import FrozenBackbone
from atlas.collect import EpisodeCollector, ShardLoader
from atlas.config import NUM_ACHIEVEMENTS, NUM_ACTIONS, REPO_ROOT, AtlasConfig
from atlas.goals import GoalEntry
from atlas.health import HealthMonitor
from atlas.screen import screen_agent
from atlas.skills import build_her_batch
from atlas.train import HeadTrainer
from eval.metrics import CRAFTER_ACHIEVEMENTS


class IterationInstrumenter:
    """Instruments agent.act during an iteration's episode collection to record telemetry."""

    def __init__(
        self,
        agent: AtlasAgent,
        iter_idx: int = 0,
        audit_exporter: Optional[AuditPackExporter] = None,
    ):
        self.agent = agent
        self.iter_idx = iter_idx
        self.audit_exporter = audit_exporter
        self.orig_act = agent.act
        self.stats: Dict[str, Any] = {
            "goals_selected_by_kind": {"event": 0, "vital": 0, "frontier": 0},
            "arrivals": 0,
            "probes_started": 0,
            "probe_unlocks": 0,
            "probe_unlocks_bank_matched": 0,
            "wander_starts": 0,
        }

    def __enter__(self):
        def instrumented_act(obs_uint8: np.ndarray, reward_prev: float, step: int):
            prev_goal = self.agent.active_goal
            prev_mode = self.agent.mode
            prev_m_bin_sum = int(np.sum(self.agent.belief.m_bin))

            action, telemetry = self.orig_act(obs_uint8, reward_prev, step)

            # 1. Goal selection: active_goal changed from None to not None
            if prev_goal is None and self.agent.active_goal is not None:
                kind = getattr(self.agent.active_goal, "kind", "unknown")
                if kind in self.stats["goals_selected_by_kind"]:
                    self.stats["goals_selected_by_kind"][kind] += 1
                else:
                    self.stats["goals_selected_by_kind"][kind] = 1

            # 2. Arrivals: cos(z, goal) >= arrival_cosine
            arrival_cos = float(telemetry.get("arrival_cos", 0.0))
            if arrival_cos >= self.agent.cfg.skills.arrival_cosine:
                self.stats["arrivals"] += 1
                if self.audit_exporter is not None:
                    self.audit_exporter.record_frame(f"arrival_it{self.iter_idx}_step{step}", obs_uint8)

            # 3. Probes started
            if prev_mode != "PROBE" and self.agent.mode == "PROBE":
                self.stats["probes_started"] += 1

            # 4. Probe unlocks (new belief unlocks during probe)
            if prev_mode == "PROBE":
                curr_m_bin_sum = int(np.sum(self.agent.belief.m_bin))
                if curr_m_bin_sum > prev_m_bin_sum:
                    diff = curr_m_bin_sum - prev_m_bin_sum
                    self.stats["probe_unlocks"] += diff
                    if getattr(self.agent, "last_probe_decision", None) and self.agent.last_probe_decision.get("matched_bank", False):
                        self.stats["probe_unlocks_bank_matched"] += diff

                if self.audit_exporter is not None and getattr(self.agent, "last_probe_decision", None):
                    dec = self.agent.last_probe_decision
                    # V3 C2: unlocked_dims during this probe only, plus probe_id
                    unlocked_dims = dec.get("unlocked_dims", [])
                    self.audit_exporter.record_decision(
                        iter_idx=self.iter_idx,
                        step=step,
                        goal_kind=dec.get("goal_kind", "unknown"),
                        goal_dim=dec.get("goal_dim", []),
                        mode="PROBE",
                        arrival_cos=dec.get("arrival_cos", 0.0),
                        chosen_action=action,
                        blended_scores=dec.get("blended_scores", [0.0] * 17),
                        probe_event_probs=dec.get("probe_event_probs", [[0.0] * 22] * 17),
                        unlocked_dims=unlocked_dims,
                        matched_bank=dec.get("matched_bank", False),
                        probe_id=dec.get("probe_id", 0),
                    )
                    self.audit_exporter.record_frame(f"probe_it{self.iter_idx}_step{step}", obs_uint8)

            # 5. Wander starts
            if prev_mode != "WANDER" and self.agent.mode == "WANDER":
                self.stats["wander_starts"] += 1

            return action, telemetry

        self.agent.act = instrumented_act
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.agent.act = self.orig_act


def mine_goals_and_lift(
    agent: AtlasAgent,
    cached_episodes: List[Dict[str, Any]],
    iter_idx: int,
) -> None:
    """Mines event, vital, and frontier goals + predecessor lift from cached play."""
    if not cached_episodes:
        return

    # 1) Mine PredecessorLift
    all_events = [ep["events"] for ep in cached_episodes]
    agent.lift.mine(all_events, window=agent.cfg.goals.predecessor_window)
    agent.lift.assert_consistent()

    # 2) Mine GoalEntries
    # Always include all episodes that unlocked any achievement
    event_eps = [ep for ep in cached_episodes if np.any(ep["events"])]
    other_eps = [ep for ep in cached_episodes if not np.any(ep["events"])]
    if len(other_eps) > 100:
        sample_indices = np.random.choice(len(other_eps), size=100, replace=False)
        selected_other = [other_eps[i] for i in sample_indices]
    else:
        selected_other = other_eps

    mining_eps = event_eps + selected_other
    print(f"  [Mining] Extracting goals from {len(mining_eps)} trajectories ({len(event_eps)} with achievements)...")

    for ep in mining_eps:
        z = ep["z"]
        h = ep["h"]
        a = ep["actions"]
        events = ep["events"]
        T = len(z)
        if T < 2:
            continue

        s = np.concatenate([z, h], axis=-1)

        # Event goals: view right before unlock
        for t in range(1, T):
            fired = np.where(events[t])[0]
            if len(fired) > 0:
                key_step = t - 1
                s_key = torch.from_numpy(s[key_step: key_step + 1]).float().to(agent.device)
                m_key = torch.from_numpy(ep["mask_before"][key_step: key_step + 1]).float().to(agent.device)
                with torch.no_grad():
                    profile = agent.backbone.event_logits_all_actions(s_key, m_key).squeeze(0).cpu().numpy()

                entry = GoalEntry(
                    key=z[key_step],
                    context=h[key_step],
                    action_profile=profile,
                    event_dims=fired.tolist(),
                    kind="event",
                    born_iter=iter_idx,
                )
                agent.goals.add(entry)

        # Vital goals: windowed realized-vital mining from training inventory targets (V3 F7)
        if "inventory" in ep and ep["inventory"].shape[-1] >= 16:
            inv = ep["inventory"]
            # Target vitals in dims 12..15, normalized by /16.0; multiply by 16.0 to get raw Crafter units
            vitals = inv[:, 12:16] * 16.0
            vital_window = getattr(agent.cfg.goals, "vital_window", 30)
            vital_min_delta = getattr(agent.cfg.goals, "vital_min_delta", 1.0)
            for t in range(T - 1):
                max_step = min(T, t + vital_window + 1)
                future_max = np.max(vitals[t + 1: max_step], axis=0)
                diff = future_max - vitals[t]
                if np.any(diff >= vital_min_delta) and not events[t].any():
                    entry = GoalEntry(
                        key=z[t],
                        context=h[t],
                        action_profile=np.zeros((NUM_ACTIONS, NUM_ACHIEVEMENTS), dtype=np.float32),
                        event_dims=[],
                        kind="vital",
                        born_iter=iter_idx,
                    )
                    agent.goals.add(entry)

        # Frontier goals: top 15% epistemic disagreement
        with torch.no_grad():
            s_t = torch.from_numpy(s).float().to(agent.device)
            a_t = torch.from_numpy(a).long().to(agent.device)
            d_vals = agent.epistemic_net.disagreement(s_t, a_t).cpu().numpy()

        q_val = np.quantile(d_vals, agent.cfg.goals.frontier_quantile)
        frontier_steps = np.where(d_vals >= q_val)[0]
        # Pick up to 2 top frontier states per episode to avoid flooding bank
        if len(frontier_steps) > 2:
            frontier_steps = frontier_steps[np.argsort(d_vals[frontier_steps])[-2:]]

        for step_idx in frontier_steps:
            entry = GoalEntry(
                key=z[step_idx],
                context=h[step_idx],
                action_profile=np.zeros((NUM_ACTIONS, NUM_ACHIEVEMENTS), dtype=np.float32),
                event_dims=[],
                kind="frontier",
                born_iter=iter_idx,
            )
            agent.goals.add(entry)

    # 3) Update world_frequency for all goals across processed episodes (V2.2 F10)
    all_z_episodes = [ep["z"] for ep in cached_episodes if len(ep["z"]) > 0]
    num_total_eps = len(all_z_episodes)
    if num_total_eps > 0 and len(agent.goals.entries) > 0:
        normed_eps = []
        for z_ep in all_z_episodes:
            z_norm = z_ep / (np.linalg.norm(z_ep, axis=-1, keepdims=True) + 1e-8)
            normed_eps.append(z_norm)

        for entry in agent.goals.entries:
            key_norm = entry.key / (np.linalg.norm(entry.key) + 1e-8)
            match_count = 0
            for z_norm in normed_eps:
                if float(np.max(np.dot(z_norm, key_norm))) >= 0.85:
                    match_count += 1
            entry.world_frequency = float(match_count / num_total_eps)

    # 3) Mine ItemEventMap (V3 F3 learned inventory-to-event bridge)
    support = np.zeros((12, NUM_ACHIEVEMENTS), dtype=np.int32)
    for ep in cached_episodes:
        if "inventory" in ep and len(ep["inventory"]) >= 2 and "events" in ep:
            inv = ep["inventory"]
            evs = ep["events"]
            T_ep = min(len(inv), len(evs))
            if T_ep >= 2:
                # inc[i][t] = inventory[t+1][i] - inventory[t][i] > +0.3 (a real gain of item i)
                inv_diff = inv[1:T_ep, :12] - inv[:T_ep - 1, :12]
                inc = inv_diff > 0.3
                ev_slice = evs[1:T_ep, :NUM_ACHIEVEMENTS] > 0.5
                # support(i, e) = count of steps where ev[e][t] AND inc[i][t]
                support += np.dot(inc.T.astype(np.int32), ev_slice.astype(np.int32))

    item_event_map: Dict[int, Optional[int]] = {}
    for i in range(12):
        best_e = int(np.argmax(support[i]))
        if support[i, best_e] >= 5:
            item_event_map[i] = best_e
        else:
            item_event_map[i] = None
    agent.item_event_map = item_event_map


class SelfLearnLoop:
    """Orchestrates bootstrap, iterative collection, head training, and screening."""

    def __init__(
        self,
        agent: AtlasAgent,
        run_id: str = "atlas_v1",
        run_dir: Optional[Union[str, Path]] = None,
        shards_dir: Optional[Union[str, Path]] = None,
        results_dir: Optional[Union[str, Path]] = None,
        cfg: Optional[AtlasConfig] = None,
    ):
        self.agent = agent
        self.cfg = cfg or AtlasConfig()
        self.run_id = run_id
        self.run_dir = Path(run_dir or f"atlas_runs/{run_id}")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.shards_dir = Path(shards_dir) if shards_dir else (self.run_dir / "shards")
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = Path(results_dir) if results_dir else (self.run_dir / "results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.journal_path = self.run_dir / "game_journal.jsonl"
        self.health = HealthMonitor(self.cfg.health)
        self.collector = EpisodeCollector(
            agent=self.agent,
            out_dir=self.shards_dir,
            cfg=self.cfg.collect,
            journal_path=self.journal_path,
            health=self.health,
        )
        self.trainer = HeadTrainer(
            backbone=self.agent.backbone,
            option_net=self.agent.option_net,
            reach_net=self.agent.reach_net,
            epistemic_net=self.agent.epistemic_net,
            safety_net=self.agent.safety_net,
            readout_net=self.agent.readout_net,
            critic_net=self.agent.critic_net,
            cfg=self.cfg.train,
            log_dir=self.run_dir,
        )
        self.agent.critic_net = self.trainer.critic_net

        self.state_file = self.run_dir / "state.pt"
        self.all_cached_episodes: List[Dict[str, Any]] = []
        self.start_iter: int = 0
        self._state_already_loaded: bool = False

        try:
            self.git_commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=str(REPO_ROOT),
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
            )
        except Exception:
            self.git_commit = "unknown"

        self.audit_exporter = AuditPackExporter(self.run_dir)
        self.enable_audit: bool = True

        # Wrap collector.run_episode to ensure journal entries carry run_segment (V2.2 F1)
        # and positions are appended to player_positions.jsonl (V2.2 F7c)
        orig_run_ep = self.collector.run_episode

        def run_ep_with_segment(seed: int, iter_idx: int = 0) -> Dict[str, Any]:
            ep_data = orig_run_ep(seed, iter_idx=iter_idx)

            if self.journal_path is not None:
                all_unlocked = [
                    CRAFTER_ACHIEVEMENTS[idx]
                    for idx in range(NUM_ACHIEVEMENTS)
                    if ep_data["events"][:, idx].any()
                ]
                mode_counts: Dict[str, int] = {}
                for t_item in ep_data.get("telemetry", []):
                    m = t_item.get("mode", "UNKNOWN")
                    mode_counts[m] = mode_counts.get(m, 0) + 1

                journal_entry = {
                    "run_segment": self.git_commit,
                    "iter": iter_idx,
                    "seed": seed,
                    "game_id": f"it{iter_idx}_seed{seed}",
                    "length": ep_data["episode_length"],
                    "total_reward": round(float(np.sum(ep_data["rewards"])), 2),
                    "died": bool(ep_data["dones"][-1]) if len(ep_data["dones"]) > 0 else False,
                    "achievements": all_unlocked,
                    "num_achievements": len(all_unlocked),
                    "mode_counts": mode_counts,
                    "overrides_count": sum(1 for t_item in ep_data.get("telemetry", []) if t_item.get("override", False)),
                    "positions": ep_data["positions"],
                    "moved_steps": ep_data["moved_steps"],
                    "moved_fraction": ep_data["moved_fraction"],
                    "action_histogram": ep_data["action_histogram"],
                    "steps": ep_data.get("telemetry", []),
                }
                try:
                    with open(self.journal_path, "a") as jf:
                        jf.write(json.dumps(journal_entry) + "\n")
                except Exception:
                    pass

            # Diagnostic player positions logging
            pos_record = {
                "game_id": f"it{iter_idx}_seed{seed}",
                "seed": seed,
                "iter": iter_idx,
                "run_segment": self.git_commit,
                "length": ep_data["episode_length"],
                "positions": ep_data["positions"],
                "moved_steps": ep_data["moved_steps"],
                "moved_fraction": ep_data["moved_fraction"],
            }
            try:
                with open(self.run_dir / "player_positions.jsonl", "a") as pf:
                    pf.write(json.dumps(pos_record) + "\n")
            except Exception:
                pass

            return ep_data

        self.collector.journal_path = None
        self.collector.run_episode = run_ep_with_segment

    def save_state(self, current_iter: int) -> None:
        """Atomic state checkpoint."""
        tmp_file = self.state_file.with_suffix(".tmp")
        payload = {
            "iter": current_iter,
            "health_snapshot": self.health.get_snapshot(),
            "option_net": self.agent.option_net.state_dict(),
            "reach_net": self.agent.reach_net.state_dict(),
            "epistemic_net": self.agent.epistemic_net.state_dict(),
            "safety_net": self.agent.safety_net.state_dict(),
            "readout_net": self.agent.readout_net.state_dict() if self.agent.readout_net else None,
            "critic_net": self.trainer.critic_net.state_dict() if hasattr(self.trainer, "critic_net") else None,
            "item_event_map": getattr(self.agent, "item_event_map", None),
            "torch_rng": torch.get_rng_state(),
            "np_rng": np.random.get_state(),
        }
        torch.save(payload, tmp_file)
        os.replace(tmp_file, self.state_file)

        # Save goal bank, lift, and persistent map
        self.agent.goals.save(self.run_dir / "goal_bank")
        self.agent.lift.save(self.run_dir / "lift.npz")
        self.agent.topomap.save(self.run_dir / "topomap.npz")

        # Audit Pack export (V2.2 F8)
        if getattr(self, "enable_audit", True):
            self.audit_exporter.export(
                current_iter=current_iter,
                git_commit=self.git_commit,
                cfg=self.cfg,
                cached_episodes=self.all_cached_episodes,
                results_dir=self.results_dir,
            )

    def load_state(self) -> int:
        """Loads state checkpoint if present; returns resume iteration."""
        self._state_already_loaded = True
        if not self.state_file.exists():
            return 0

        ckpt = torch.load(self.state_file, map_location=self.agent.device, weights_only=False)
        self.agent.option_net.load_state_dict(ckpt["option_net"])
        self.agent.reach_net.load_state_dict(ckpt["reach_net"])
        self.agent.epistemic_net.load_state_dict(ckpt["epistemic_net"])
        self.agent.safety_net.load_state_dict(ckpt["safety_net"])
        if ckpt.get("readout_net") is not None and self.agent.readout_net:
            self.agent.readout_net.load_state_dict(ckpt["readout_net"])
        if ckpt.get("critic_net") is not None and hasattr(self.trainer, "critic_net"):
            self.trainer.critic_net.load_state_dict(ckpt["critic_net"])
            self.agent.critic_net = self.trainer.critic_net
        if "item_event_map" in ckpt:
            self.agent.item_event_map = ckpt["item_event_map"]

        resume_iter = int(ckpt.get("iter", 0)) + 1
        if "torch_rng" in ckpt:
            torch.set_rng_state(ckpt["torch_rng"].cpu())
        if "np_rng" in ckpt:
            np.random.set_state(ckpt["np_rng"])

        if "health_snapshot" in ckpt:
            self.health.load_snapshot(ckpt["health_snapshot"], resume_iter=resume_iter)

        self.agent.goals.load(self.run_dir / "goal_bank")
        self.agent.lift.load(self.run_dir / "lift.npz")
        self.agent.topomap.load(self.run_dir / "topomap.npz")

        self.start_iter = resume_iter
        print(f"[SelfLearnLoop] Resumed from iteration {resume_iter}")
        return resume_iter

    def remine_shards(
        self,
        shards_dir: Optional[Union[str, Path]] = None,
        marker_path: Optional[Union[str, Path]] = None,
        chunk_size: int = 40,
    ) -> Dict[str, Any]:
        """Performs one-time re-mine of existing shards in chunks <= 40 episodes."""
        s_dir = Path(shards_dir or self.shards_dir)
        m_path = Path(marker_path or (self.run_dir / "REMINED"))

        def _shard_iter(p: Path) -> int:
            try:
                parts = p.stem.split("_")
                return int(parts[1])
            except Exception:
                return 0

        shard_files = sorted(
            [p for p in s_dir.glob("shard_*.npz") if not p.name.endswith(".latents.npz")],
            key=_shard_iter,
        )

        total_episodes = 0
        all_events_for_lift: List[np.ndarray] = []
        chunk: List[Dict[str, Any]] = []

        for shard_path in shard_files:
            latents_path = s_dir / f"{shard_path.stem}.latents.npz"
            if not latents_path.exists():
                latents_path = self.trainer.cache_latents(shard_path, cache_dir=s_dir)

            shard_data = ShardLoader.load_shard(shard_path, load_obs=False)
            with np.load(latents_path) as latents_data:
                z_full = np.array(latents_data["z"])
                h_full = np.array(latents_data["h"])
                res_full = np.array(latents_data["target_residuals"])

            ep_lengths = shard_data["episode_lengths"]
            start = 0
            for ep_len in ep_lengths:
                end = start + ep_len
                ep_dict = {
                    "z": z_full[start:end].copy(),
                    "h": h_full[start:end].copy(),
                    "target_residuals": res_full[start:end].copy(),
                    "actions": shard_data["actions"][start:end].copy(),
                    "rewards": shard_data["rewards"][start:end].copy(),
                    "dones": shard_data["dones"][start:end].copy(),
                    "events": shard_data["events"][start:end].copy(),
                    "mask_before": shard_data["mask_before"][start:end].copy(),
                    "inventory": shard_data["inventory"][start:end].copy(),
                    "positions": shard_data["positions"][start:end].copy() if "positions" in shard_data else np.zeros((ep_len, 2), dtype=np.float32),
                }
                chunk.append(ep_dict)
                all_events_for_lift.append(ep_dict["events"])
                start = end

                if len(chunk) >= chunk_size:
                    mine_goals_and_lift(self.agent, chunk, iter_idx=39)
                    total_episodes += len(chunk)
                    chunk.clear()
                    gc.collect()

            del z_full, h_full, res_full, shard_data
            gc.collect()

        if chunk:
            mine_goals_and_lift(self.agent, chunk, iter_idx=39)
            total_episodes += len(chunk)
            chunk.clear()
            gc.collect()

        if all_events_for_lift:
            self.agent.lift.mine(all_events_for_lift, window=self.agent.cfg.goals.predecessor_window)
            self.agent.lift.assert_consistent()

        event_goals = sum(1 for g in self.agent.goals.entries if g.kind == "event")
        vital_goals = sum(1 for g in self.agent.goals.entries if g.kind == "vital")
        frontier_goals = sum(1 for g in self.agent.goals.entries if g.kind == "frontier")
        lift_edges = int(np.sum((self.agent.lift.lift > 1.0) & (self.agent.lift.ci_low > 1.0)))

        remine_stats = {
            "episodes": total_episodes,
            "event_goals": event_goals,
            "vital_goals": vital_goals,
            "frontier_goals": frontier_goals,
            "lift_edges": lift_edges,
        }

        # Save repaired state components
        self.agent.goals.save(self.run_dir / "goal_bank")
        self.agent.lift.save(self.run_dir / "lift.npz")

        # Write atomic REMINED marker
        tmp_m = m_path.with_suffix(".tmp")
        with open(tmp_m, "w") as f:
            json.dump(remine_stats, f, indent=2)
        os.replace(tmp_m, m_path)

        # Log to train_log.jsonl
        log_entry = {
            "iter": 39,
            "remine_stats": remine_stats,
        }
        with open(self.trainer.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return remine_stats

    def run_iterations(
        self,
        num_iters: int,
        minutes_budget: Optional[float] = None,
        fresh: bool = False,
    ) -> None:
        """Runs the self-learn loop for num_iters or until minutes_budget expires."""
        if fresh:
            self.start_iter = 0
            self._state_already_loaded = True
        elif not self._state_already_loaded:
            self.start_iter = self.load_state()
            self._state_already_loaded = True
        t0 = time.time()
        end_iter = self.start_iter + num_iters

        print(f"[Resume] starting at iteration {self.start_iter}, seeds {10000 + self.start_iter * 100}..")
        last_dev_ach = self.health.progress_history[-1][1] if self.health.progress_history else 0.0

        for it in range(self.start_iter, end_iter):
            assert it >= self.start_iter, f"Loop iteration {it} is less than start_iter {self.start_iter}"
            if minutes_budget is not None and (time.time() - t0) / 60.0 >= minutes_budget:
                print(f"[SelfLearnLoop] Time budget of {minutes_budget}m reached. Graceful exit.")
                break

            print(f"\n--- [SelfLearnLoop] Iteration {it}/{end_iter - 1} ---")
            seed_base = 10000 + it * 100

            # 1. Collect with loop instrumentation
            with IterationInstrumenter(self.agent, iter_idx=it, audit_exporter=self.audit_exporter) as inst:
                shard_path = self.collector.run_episodes(
                    n=self.cfg.collect.episodes_per_iter,
                    seed_base=seed_base,
                    iter_idx=it,
                )

            iter_extra_log = {
                "goals_selected_by_kind": inst.stats["goals_selected_by_kind"],
                "arrivals": inst.stats["arrivals"],
                "probes_started": inst.stats["probes_started"],
                "probe_unlocks": inst.stats["probe_unlocks"],
                "probe_unlocks_bank_matched": inst.stats.get("probe_unlocks_bank_matched", 0),
                "wander_starts": inst.stats["wander_starts"],
                "git_commit": self.git_commit,
            }

            # Remove any stale sidecar from a previously interrupted attempt of this iteration
            stale_latents = self.shards_dir / f"{shard_path.stem}.latents.npz"
            if stale_latents.exists() and stale_latents.stat().st_mtime < shard_path.stat().st_mtime:
                stale_latents.unlink(missing_ok=True)

            # 2. Latent caching
            latents_path = self.trainer.cache_latents(shard_path, cache_dir=self.shards_dir)

            # Load cached shard into episode memory without raw pixel images
            shard_data = ShardLoader.load_shard(shard_path, load_obs=False)
            with np.load(latents_path) as latents_data:
                z_full = np.array(latents_data["z"])
                h_full = np.array(latents_data["h"])
                res_full = np.array(latents_data["target_residuals"])

            assert len(z_full) == len(shard_data["actions"]), (
                f"Length mismatch: {shard_path.name} has {len(shard_data['actions'])} actions, "
                f"but latents has {len(z_full)} steps."
            )

            ep_lengths = shard_data["episode_lengths"]

            start = 0
            for ep_len in ep_lengths:
                end = start + ep_len
                self.all_cached_episodes.append({
                    "z": z_full[start:end].copy(),
                    "h": h_full[start:end].copy(),
                    "target_residuals": res_full[start:end].copy(),
                    "actions": shard_data["actions"][start:end].copy(),
                    "rewards": shard_data["rewards"][start:end].copy(),
                    "dones": shard_data["dones"][start:end].copy(),
                    "events": shard_data["events"][start:end].copy(),
                    "mask_before": shard_data["mask_before"][start:end].copy(),
                    "inventory": shard_data["inventory"][start:end].copy(),
                    "positions": shard_data["positions"][start:end].copy() if "positions" in shard_data else np.zeros((ep_len, 2), dtype=np.float32),
                })
                start = end

            del z_full, h_full, res_full, shard_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 3. Train heads (90% train, 10% val split by episode)
            num_eps = len(self.all_cached_episodes)
            if num_eps < 2:
                train_eps = self.all_cached_episodes
                val_eps = self.all_cached_episodes
            else:
                val_count = max(1, int(0.1 * num_eps))
                train_eps = self.all_cached_episodes[:-val_count]
                val_eps = self.all_cached_episodes[-val_count:]

            train_metrics = self.trainer.train_epoch(train_eps, val_eps, iter_idx=it, extra_log=iter_extra_log)

            # Observe OptionNet confidence on validation HER batch
            val_her = build_her_batch(val_eps, device=self.agent.device) if val_eps else None
            if val_her is not None and val_her["s_pos"].shape[0] > 0:
                with torch.no_grad():
                    logits = self.agent.option_net.logits(val_her["s_pos"], val_her["g_pos"])
                    probs = torch.softmax(logits, dim=-1)
                    mean_max_prob = float(torch.max(probs, dim=-1).values.mean().item())
                self.health.observe_confidence(mean_max_prob)

            # 4. Mine GoalBank and PredecessorLift
            mine_goals_and_lift(self.agent, self.all_cached_episodes, iter_idx=it)

            # 5. Dev screen every 5 iterations
            if (it + 1) % 5 == 0:
                print(f"[SelfLearnLoop] Running dev screen at iteration {it}...")
                screen_res = screen_agent(
                    agent=self.agent,
                    seeds=self.cfg.screen.dev_seeds,
                    num_episodes=10,
                    tag=f"{self.run_id}_dev_it{it}",
                    out_dir=self.results_dir,
                )
                last_dev_ach = float(
                    screen_res.get("mean_achievements_per_ep")
                    if "mean_achievements_per_ep" in screen_res
                    else screen_res.get("summary", {}).get("achievements_per_episode", 0.0)
                )
                probe_frac = float(screen_res.get("per_mode_telemetry", {}).get("probe_fraction", 0.0))
                self.health.observe_probe_fraction(probe_frac)

            # Honest monitor feed: carry forward last known dev screen ach (V2.2 F2)
            self.health.observe_progress(it, last_dev_ach)

            # 6. Runtime health check and alarms
            health_status = self.health.check(it, log_file=self.trainer.log_file)
            if health_status.get("beta_reset", False):
                self.agent.beta_t = 0.8
                print(f"[SelfLearnLoop] Beta reset to 0.8 triggered by health flatline alarm at iter {it}")

            # 7. Write health report cards at iterations 30 and 50
            if it in (30, 50):
                report_path = self.results_dir / f"atlas_health_it{it}.json"
                self.health.generate_report_card(
                    iter_idx=it,
                    agent=self.agent,
                    val_her_batch=val_her,
                    out_path=report_path,
                )

            # 8. Save atomic state
            self.save_state(it)

        print("[SelfLearnLoop] Execution finished.")
