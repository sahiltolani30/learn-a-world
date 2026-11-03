"""Data collection workers gathering raw experience into ReplayBuffer."""
from typing import Optional, Dict, Any, List
import numpy as np
import torch

from src.env.crafter_adapter import CrafterAdapter
from src.env.vector_adapter import VectorCrafterAdapter
from src.data.replay_buffer import ReplayBuffer
from src.exploration.action_selector import ActionSelector
from src.models.jepa_world_model import JepaWorldModel
from src.planning.cem_planner import CemPlanner


class ExperienceCollector:
    """Collects environment transitions from a single environment into ReplayBuffer."""

    def __init__(
        self,
        adapter: CrafterAdapter,
        replay_buffer: ReplayBuffer,
        action_selector: ActionSelector,
        model: Optional[JepaWorldModel] = None,
        planner: Optional[CemPlanner] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.adapter = adapter
        self.replay = replay_buffer
        self.action_selector = action_selector
        self.model = model
        self.planner = planner
        self.device = device

        self.current_obs: Optional[np.ndarray] = None
        self.current_episode_id = 0
        self.total_env_steps = 0
        self.episode_history_obs: List[np.ndarray] = []
        self.episode_history_actions: List[int] = []

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment and initialize episode in replay buffer."""
        self.current_episode_id += 1
        obs = self.adapter.reset(seed=seed)
        self.current_obs = obs
        self.replay.start_episode(obs, episode_id=self.current_episode_id)
        
        self.episode_history_obs = [obs]
        self.episode_history_actions = [0]
        return obs

    def step(self, mode: str = "warmup", role: str = "explorer") -> Dict[str, Any]:
        """Execute one environment step and add transition to replay buffer."""
        if self.current_obs is None:
            self.reset()

        latent = None
        context = None
        if self.model is not None and mode != "warmup":
            self.model.eval()
            with torch.no_grad():
                ctx_len = min(4, len(self.episode_history_obs))
                obs_slice = self.episode_history_obs[-ctx_len:]
                act_slice = self.episode_history_actions[-ctx_len:]

                obs_arr = np.zeros((4, 64, 64, 3), dtype=np.uint8)
                act_arr = np.zeros(4, dtype=np.int64)
                mask_arr = np.zeros(4, dtype=bool)

                start_idx = 4 - ctx_len
                for idx, (o, a) in enumerate(zip(obs_slice, act_slice)):
                    obs_arr[start_idx + idx] = o
                    act_arr[start_idx + idx] = a
                    mask_arr[start_idx + idx] = True

                ch_obs = np.transpose(obs_arr, (0, 3, 1, 2))
                t_obs = torch.from_numpy(ch_obs).float().div_(255.0).unsqueeze(0).to(self.device)
                t_act = torch.from_numpy(act_arr).unsqueeze(0).to(self.device)
                t_mask = torch.from_numpy(mask_arr).unsqueeze(0).to(self.device)

                latent, context = self.model.encode_actual_history(t_obs, t_act, t_mask)

        action_token = self.action_selector.select_action(
            latent=latent,
            context=context,
            model=self.model,
            mode=mode,
            role=role,
            planner=self.planner,
        )

        next_obs, reward, terminated, truncated = self.adapter.step(action_token)
        self.total_env_steps += 1

        self.replay.add(
            action=action_token,
            reward=reward,
            next_obs=next_obs,
            terminated=terminated,
            truncated=truncated,
        )

        self.episode_history_obs.append(next_obs)
        self.episode_history_actions.append(action_token)
        if len(self.episode_history_obs) > 8:
            self.episode_history_obs.pop(0)
            self.episode_history_actions.pop(0)

        step_info = {
            "action": action_token,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "done": (terminated or truncated),
            "step": self.total_env_steps,
        }

        if terminated or truncated:
            self.current_obs = None
        else:
            self.current_obs = next_obs

        return step_info


class BatchedExperienceCollector:
    """Collects transitions in lockstep across N parallel environments (e.g. 8 games).
    
    Guarantees:
    - N games step in parallel on CPU.
    - Completed episodes are committed contiguously into the circular ReplayBuffer.
    - Zero boundary crossing during multi-horizon sequence sampling.
    """

    def __init__(
        self,
        vector_adapter: VectorCrafterAdapter,
        replay_buffer: ReplayBuffer,
        action_selector: ActionSelector,
        model: Optional[JepaWorldModel] = None,
        planner: Optional[CemPlanner] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.vec_adapter = vector_adapter
        self.replay = replay_buffer
        self.action_selector = action_selector
        self.model = model
        self.planner = planner
        self.device = device
        self.num_envs = vector_adapter.num_envs

        self.episode_id_counter = 1
        self.total_env_steps = 0
        self.completed_episodes = 0

        self.obs_buffers: List[List[np.ndarray]] = [[] for _ in range(self.num_envs)]
        self.act_buffers: List[List[int]] = [[] for _ in range(self.num_envs)]
        self.rew_buffers: List[List[float]] = [[] for _ in range(self.num_envs)]
        self.term_buffers: List[List[bool]] = [[] for _ in range(self.num_envs)]
        self.trunc_buffers: List[List[bool]] = [[] for _ in range(self.num_envs)]

        self.reset_all()

    def reset_all(self) -> None:
        """Reset all environments and initialize observation buffers."""
        init_obs = self.vec_adapter.reset()
        for i in range(self.num_envs):
            self.obs_buffers[i] = [init_obs[i].copy()]
            self.act_buffers[i] = []
            self.rew_buffers[i] = []
            self.term_buffers[i] = []
            self.trunc_buffers[i] = []

    def step(self, mode: str = "warmup", role: str = "explorer") -> Dict[str, Any]:
        """Step all N environments in parallel and commit completed episodes contiguously."""
        actions: List[int] = []

        if mode == "warmup" or self.model is None:
            actions = [int(np.random.randint(0, 17)) for _ in range(self.num_envs)]
        else:
            self.model.eval()
            for i in range(self.num_envs):
                hist_obs = self.obs_buffers[i]
                hist_act = self.act_buffers[i]
                ctx_len = min(4, len(hist_obs))

                obs_arr = np.zeros((4, 64, 64, 3), dtype=np.uint8)
                act_arr = np.zeros(4, dtype=np.int64)
                mask_arr = np.zeros(4, dtype=bool)

                start_idx = 4 - ctx_len
                for idx, (o, a) in enumerate(zip(hist_obs[-ctx_len:], [0] + hist_act[-ctx_len + 1:] if ctx_len > 1 else [0])):
                    obs_arr[start_idx + idx] = o
                    act_arr[start_idx + idx] = a
                    mask_arr[start_idx + idx] = True

                ch_obs = np.transpose(obs_arr, (0, 3, 1, 2))
                with torch.no_grad():
                    t_obs = torch.from_numpy(ch_obs).float().div_(255.0).unsqueeze(0).to(self.device)
                    t_act = torch.from_numpy(act_arr).unsqueeze(0).to(self.device)
                    t_mask = torch.from_numpy(mask_arr).unsqueeze(0).to(self.device)
                    latent, context = self.model.encode_actual_history(t_obs, t_act, t_mask)

                act = self.action_selector.select_action(
                    latent=latent,
                    context=context,
                    model=self.model,
                    mode=mode,
                    role=role,
                    planner=self.planner,
                )
                actions.append(act)

        transitions = self.vec_adapter.step(actions)
        self.total_env_steps += self.num_envs

        dones_in_batch = 0
        rewards_in_batch = 0.0

        for i, (next_obs, rew, term, trunc) in enumerate(transitions):
            self.act_buffers[i].append(actions[i])
            self.rew_buffers[i].append(float(rew))
            self.term_buffers[i].append(bool(term))
            self.trunc_buffers[i].append(bool(trunc))
            self.obs_buffers[i].append(next_obs)
            rewards_in_batch += float(rew)

            if term or trunc:
                dones_in_batch += 1
                self.completed_episodes += 1
                ep_obs = np.stack(self.obs_buffers[i], axis=0)
                ep_act = np.array(self.act_buffers[i], dtype=np.uint8)
                ep_rew = np.array(self.rew_buffers[i], dtype=np.float32)
                ep_term = np.array(self.term_buffers[i], dtype=bool)
                ep_trunc = np.array(self.trunc_buffers[i], dtype=bool)

                self.replay.add_episode_trajectory(
                    obs_sequence=ep_obs,
                    actions=ep_act,
                    rewards=ep_rew,
                    terminated=ep_term,
                    truncated=ep_trunc,
                    episode_id=self.episode_id_counter,
                )
                self.episode_id_counter += 1

                self.obs_buffers[i] = [self.vec_adapter.current_obs[i].copy()]
                self.act_buffers[i] = []
                self.rew_buffers[i] = []
                self.term_buffers[i] = []
                self.trunc_buffers[i] = []

        return {
            "num_steps": self.num_envs,
            "total_env_steps": self.total_env_steps,
            "dones": dones_in_batch,
            "rewards": rewards_in_batch,
            "completed_episodes": self.completed_episodes,
        }
