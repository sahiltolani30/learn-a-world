"""Phase 0 causal-chain traces and controlled ablations.

The diagnostic path may record engine inventory/achievement values, but labels
the run ``privileged_diagnostic`` and never feeds those values to the planner in
the strict track.  This makes controller failures observable without turning
diagnostic state into a claimed visual capability.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import gymnasium as gym
import numpy as np
import torch

import envs  # noqa: F401 - register Crafter-v0
from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner
from scripts.screen_v3_50eps import _resolve_checkpoint


@dataclass(frozen=True)
class Phase0Ablation:
    name: str
    track: str = "strict_visual_jepa"
    memory_mode: str = "off"  # off, empty, preloaded
    landmarks: bool = True
    policy_prior: bool = True


def default_ablations(include_privileged: bool = False) -> List[Phase0Ablation]:
    configs = [
        Phase0Ablation("strict_no_memory", memory_mode="off"),
        Phase0Ablation("strict_empty_memory", memory_mode="empty"),
        Phase0Ablation("strict_landmarks_off", memory_mode="off", landmarks=False),
        Phase0Ablation("strict_policy_prior_off", memory_mode="off", policy_prior=False),
    ]
    if include_privileged:
        configs.append(
            Phase0Ablation(
                "privileged_inventory_diagnostic",
                track="privileged_diagnostic",
                memory_mode="off",
            )
        )
    return configs


class ChainTraceRecorder:
    """Stores step-level planner evidence and a conservative failure label."""

    def __init__(self, *, track: str, ablation: str):
        self.track = track
        self.ablation = ablation
        self.episodes: List[Dict[str, Any]] = []
        self._current: Optional[Dict[str, Any]] = None

    def start_episode(self, seed: int) -> None:
        if self._current is not None:
            raise RuntimeError("Previous diagnostic episode was not finalized")
        self._current = {"seed": int(seed), "steps": [], "actual_state_recorded": self.track == "privileged_diagnostic"}

    def record_step(
        self,
        *,
        step: int,
        action: int,
        reward: float,
        done: bool,
        planner_debug: Optional[Dict[str, Any]] = None,
        predicted_transition: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._current is None:
            raise RuntimeError("start_episode must be called before record_step")
        row: Dict[str, Any] = {
            "step": int(step),
            "action": int(action),
            "reward": float(reward),
            "done": bool(done),
            "planner": _json_safe(planner_debug or {}),
        }
        if predicted_transition is not None:
            row["predicted_transition"] = _json_safe(predicted_transition)
        if self.track == "privileged_diagnostic" and info is not None:
            row["actual_inventory"] = _json_safe(info.get("inventory", {}))
            row["actual_achievements"] = _json_safe(info.get("achievements", {}))
        self._current["steps"].append(row)

    def finish_episode(self, final_info: Optional[Dict[str, Any]], total_reward: float) -> Dict[str, Any]:
        if self._current is None:
            raise RuntimeError("No active diagnostic episode")
        achievements = dict(final_info.get("achievements", {})) if final_info else {}
        reached = {k: bool(v > 0) for k, v in achievements.items()}
        # These labels are diagnostic vocabulary only. They are not supplied to
        # the model and are not part of the transferable planning core.
        if not reached.get("collect_wood", False):
            failure = "wood_acquisition"
        elif not reached.get("place_table", False):
            failure = "table_execution"
        elif not reached.get("make_wood_pickaxe", False):
            failure = "pickaxe_execution"
        elif not reached.get("collect_stone", False):
            failure = "stone_detection_or_mining"
        else:
            failure = "later_chain_or_survival"
        self._current.update({
            "total_reward": float(total_reward),
            "steps_taken": len(self._current["steps"]),
            "final_achievements": _json_safe(achievements) if self.track == "privileged_diagnostic" else {},
            "first_failed_link": failure,
        })
        self.episodes.append(self._current)
        finished = self._current
        self._current = None
        return finished

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for ep in self.episodes:
            key = ep["first_failed_link"]
            counts[key] = counts.get(key, 0) + 1
        return {
            "track": self.track,
            "ablation": self.ablation,
            "episodes": len(self.episodes),
            "first_failed_link_counts": counts,
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _build_memory(mode: str, memory_bank_path: Optional[str], device: torch.device):
    if mode == "off":
        return None
    from planning.semantic_memory import LongTermSemanticMemory

    bank = LongTermSemanticMemory(latent_dim=128, device=device)
    if mode == "preloaded" and memory_bank_path and Path(memory_bank_path).exists():
        bank.load(memory_bank_path)
    return bank


def run_phase0_diagnostics(
    *,
    checkpoint_path: str,
    output_json: Union[str, Path] = "results/phase0_chain_diagnostics.json",
    num_episodes: int = 3,
    seed_start: int = 2050,
    max_steps: int = 300,
    device: Union[str, torch.device] = "cpu",
    memory_bank_path: Optional[str] = None,
    ablations: Optional[Sequence[Phase0Ablation]] = None,
    include_privileged: bool = False,
) -> Dict[str, Any]:
    """Run same seeds across Phase 0 controller ablations."""
    dev = torch.device(device)
    resolved = _resolve_checkpoint(checkpoint_path)
    model = JEPAv3()
    if Path(resolved).exists():
        model.load_checkpoint(resolved, device=dev)
    else:
        model.to(dev)
    model.eval()
    configs = list(ablations) if ablations is not None else default_ablations(include_privileged)
    if ablations is None and memory_bank_path and Path(memory_bank_path).exists():
        configs.insert(2, Phase0Ablation("strict_preloaded_memory", memory_mode="preloaded"))
    all_results: List[Dict[str, Any]] = []
    started = time.time()

    for config in configs:
        memory = _build_memory(config.memory_mode, memory_bank_path, dev)
        planner = RecurrentMPCPlanner(
            model=model,
            horizon=10,
            num_candidates=32,
            cem_iterations=1,
            prior_mix=0.5,
            semantic_memory=memory,
            use_landmark_guidance=config.landmarks,
            use_policy_prior=config.policy_prior,
            device=dev,
        )
        belief = AgentBeliefV3(recurrent_dim=256)
        recorder = ChainTraceRecorder(track=config.track, ablation=config.name)
        env = gym.make("Crafter-v0", render_size=64, max_steps=max_steps)
        episode_records: List[Dict[str, Any]] = []
        try:
            for ep_idx in range(num_episodes):
                seed = seed_start + ep_idx
                obs, info = env.reset(seed=seed)
                belief.reset()
                # Only the explicit privileged ablation receives engine state.
                if config.track == "privileged_diagnostic":
                    belief.update_inventory(info.get("inventory", {}))
                recorder.start_episode(seed)
                total_reward = 0.0
                for step in range(max_steps):
                    action, debug = planner.plan(obs, belief)
                    with torch.no_grad():
                        obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(dev) / 255.0
                        z = model.encode(obs_t, use_target=True)
                        h = belief.get_h_tensor(device=dev).unsqueeze(0)
                        a = torch.tensor([action], dtype=torch.long, device=dev)
                        s, next_h = model.recurrent_step(z, a, h)
                        transition = model.predict_transition(s, a)
                        pred = {k: v.squeeze(0) for k, v in transition.items()}
                        mask = belief.get_mask_tensor(device=dev).unsqueeze(0)
                        event_probs = model.event_head.predict_proba(s, a, mask).squeeze(0)
                    next_obs, reward, terminated, truncated, next_info = env.step(action)
                    done = bool(terminated or truncated)
                    total_reward += float(reward)
                    recorder.record_step(
                        step=step,
                        action=action,
                        reward=reward,
                        done=done,
                        planner_debug=debug,
                        predicted_transition=pred,
                        info=next_info if config.track == "privileged_diagnostic" else None,
                    )
                    belief.update(event_probs, reward, next_h=next_h.squeeze(0))
                    if config.track == "privileged_diagnostic":
                        belief.update_inventory(next_info.get("inventory", {}))
                    obs = next_obs
                    info = next_info
                    if done:
                        break
                episode_records.append(recorder.finish_episode(info, total_reward))
        finally:
            env.close()
        result = recorder.summary()
        result["episodes_detail"] = episode_records
        result["config"] = asdict(config)
        all_results.append(result)
        print(f"[Phase 0] {config.name}: {result['first_failed_link_counts']}")

    result = {
        "phase": "phase0",
        "status": "completed",
        "checkpoint": resolved,
        "num_episodes": num_episodes,
        "seed_range": [seed_start, seed_start + num_episodes - 1],
        "inventory_used_by_strict_track": False,
        "results": all_results,
        "elapsed_seconds": time.time() - started,
    }
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result
