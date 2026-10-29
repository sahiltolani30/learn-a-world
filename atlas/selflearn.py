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
from atlas.backbone import FrozenBackbone
from atlas.collect import EpisodeCollector, ShardLoader
from atlas.config import NUM_ACHIEVEMENTS, NUM_ACTIONS, REPO_ROOT, AtlasConfig
from atlas.goals import GoalEntry
from atlas.health import HealthMonitor
from atlas.screen import screen_agent
from atlas.skills import build_her_batch
from atlas.train import HeadTrainer


class IterationInstrumenter:
    """Instruments agent.act during an iteration's episode collection to record telemetry."""

    def __init__(self, agent: AtlasAgent):
        self.agent = agent
        self.orig_act = agent.act
        self.stats: Dict[str, Any] = {
            "goals_selected_by_kind": {"event": 0, "vital": 0, "frontier": 0},
            "arrivals": 0,
            "probes_started": 0,
            "probe_unlocks": 0,
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

            # 3. Probes started
            if prev_mode != "PROBE" and self.agent.mode == "PROBE":
                self.stats["probes_started"] += 1

            # 4. Probe unlocks (new belief unlocks during probe)
            if prev_mode == "PROBE":
                curr_m_bin_sum = int(np.sum(self.agent.belief.m_bin))
                if curr_m_bin_sum > prev_m_bin_sum:
                    self.stats["probe_unlocks"] += (curr_m_bin_sum - prev_m_bin_sum)

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

        # Vital goals: health-head positive spikes without unlock
        with torch.no_grad():
            s_t = torch.from_numpy(s).float().to(agent.device)
            a_t = torch.from_numpy(a).long().to(agent.device)
            vital_pred = agent.backbone.health(s_t, a_t).cpu().numpy()

        for t in range(T):
            if vital_pred[t] >= agent.cfg.goals.vital_spike and not events[t].any():
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
            cfg=self.cfg.train,
            log_dir=self.run_dir,
        )

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

    def save_state(self, current_iter: int) -> None:
        """Atomic state checkpoint."""
        tmp_file = self.state_file.with_suffix(".tmp")
        payload = {
            "iter": current_iter,
            "option_net": self.agent.option_net.state_dict(),
            "reach_net": self.agent.reach_net.state_dict(),
            "epistemic_net": self.agent.epistemic_net.state_dict(),
            "safety_net": self.agent.safety_net.state_dict(),
            "readout_net": self.agent.readout_net.state_dict() if self.agent.readout_net else None,
            "torch_rng": torch.get_rng_state(),
            "np_rng": np.random.get_state(),
        }
        torch.save(payload, tmp_file)
        os.replace(tmp_file, self.state_file)

        # Save goal bank, lift, and persistent map
        self.agent.goals.save(self.run_dir / "goal_bank")
        self.agent.lift.save(self.run_dir / "lift.npz")
        self.agent.topomap.save(self.run_dir / "topomap.npz")

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

        if "torch_rng" in ckpt:
            torch.set_rng_state(ckpt["torch_rng"].cpu())
        if "np_rng" in ckpt:
            np.random.set_state(ckpt["np_rng"])

        self.agent.goals.load(self.run_dir / "goal_bank")
        self.agent.lift.load(self.run_dir / "lift.npz")
        self.agent.topomap.load(self.run_dir / "topomap.npz")

        resume_iter = int(ckpt.get("iter", 0)) + 1
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

        for it in range(self.start_iter, end_iter):
            if minutes_budget is not None and (time.time() - t0) / 60.0 >= minutes_budget:
                print(f"[SelfLearnLoop] Time budget of {minutes_budget}m reached. Graceful exit.")
                break

            print(f"\n--- [SelfLearnLoop] Iteration {it}/{end_iter - 1} ---")
            seed_base = 10000 + it * 100

            # 1. Collect with loop instrumentation
            with IterationInstrumenter(self.agent) as inst:
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
                ach_per_ep = float(screen_res.get("summary", {}).get("achievements_per_episode", 0.0))
                self.health.observe_progress(it, ach_per_ep)
                probe_frac = float(screen_res.get("per_mode_telemetry", {}).get("probe_fraction", 0.0))
                self.health.observe_probe_fraction(probe_frac)

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
