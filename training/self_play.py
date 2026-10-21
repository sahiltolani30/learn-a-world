"""
Self-Play Agent and Evaluation Loop for Learn-a-World.
Implements the continuous closed-loop:
Play -> Remember (Buffer) -> Learn (JEPA) -> Imagine (MPC) -> Play Better
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import gymnasium as gym
import numpy as np
import torch

from models.curiosity import CuriosityModule
from models.jepa import JEPAWorldModel
from planning.mpc import LatentMPCPlanner
from training.buffer import ReplayBuffer

class SelfPlayAgent:
    """
    Closed-loop self-play agent combining latent-space MPC planning and curiosity exploration.
    """

    def __init__(
        self,
        env: gym.Env,
        model: JEPAWorldModel,
        buffer: ReplayBuffer,
        planner: LatentMPCPlanner,
        curiosity: Optional[CuriosityModule] = None,
        epsilon: float = 0.2,
        device: Optional[torch.device] = None,
    ):
        self.env = env
        self.model = model
        self.buffer = buffer
        self.planner = planner
        self.curiosity = curiosity
        self.epsilon = epsilon
        self.device = device if device is not None else next(model.parameters()).device

        # Pre-render goal observation for planning
        self.goal_obs = self._extract_goal_frame()

    def _extract_goal_frame(self) -> np.ndarray:
        """Extracts a frame where the agent is at the goal position."""
        unwrapped = self.env.unwrapped
        saved_pos = unwrapped.agent_pos.copy()
        unwrapped.agent_pos = unwrapped.goal_pos.copy()
        goal_frame = unwrapped.render()
        unwrapped.agent_pos = saved_pos
        return goal_frame

    def run_episode(
        self,
        max_steps: int = 100,
        random_start: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes one self-play episode.
        Appends transitions into the replay buffer with optional curiosity reward.
        Returns episode diagnostics and unique state coverage.
        """
        obs, info = self.env.reset(options={"random_start": random_start})
        self.planner.reset_history()
        if self.curiosity is not None:
            self.curiosity.reset_memory()

        visited_positions: Set[Tuple[int, int]] = set()
        total_reward = 0.0
        success = False
        steps = 0

        pos = tuple(info.get("agent_pos", (0, 0)))
        visited_positions.add((int(pos[0]), int(pos[1])))

        for step in range(max_steps):
            steps += 1

            # Epsilon-directed exploration vs MPC goal planning
            if np.random.random() < self.epsilon:
                if self.curiosity is not None:
                    action = self.curiosity.select_curious_action(obs)
                else:
                    action = self.env.action_space.sample()
            else:
                action = self.planner.plan(obs, self.goal_obs)

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # Add curiosity bonus if enabled
            effective_reward = float(reward)
            if self.curiosity is not None:
                bonus = self.curiosity.compute_intrinsic_reward(obs, action, next_obs)
                effective_reward += bonus

            # Store in replay buffer
            self.buffer.add(obs, action, next_obs, effective_reward, terminated)

            pos = tuple(info.get("agent_pos", (0, 0)))
            visited_positions.add((int(pos[0]), int(pos[1])))
            total_reward += float(reward)

            if terminated:
                success = True
                break
            if truncated:
                break

            obs = next_obs

        return {
            "success": success,
            "steps": steps,
            "total_reward": total_reward,
            "unique_states_visited": len(visited_positions),
        }

def evaluate_agent(
    env: gym.Env,
    planner: Optional[LatentMPCPlanner] = None,
    num_episodes: int = 20,
    max_steps: int = 25,
    random_policy: bool = False,
    start_positions: Optional[List[List[int]]] = None,
) -> Dict[str, float]:
    """
    Evaluates agent success rate and average steps to goal over fixed test episodes.
    """
    unwrapped = env.unwrapped
    saved_pos = unwrapped.agent_pos.copy()
    unwrapped.agent_pos = unwrapped.goal_pos.copy()
    goal_obs = unwrapped.render()
    unwrapped.agent_pos = saved_pos

    successes = 0
    total_steps_list: List[int] = []

    # Default held-out test starts testing goal-seeking within planning horizon
    test_starts = (
        start_positions
        if start_positions is not None
        else [
            [1, 6], [1, 5], [2, 7], [3, 7], [4, 7],
            [3, 5], [5, 7], [3, 6], [2, 5], [1, 3],
            [2, 3], [3, 3], [5, 6], [6, 7], [7, 7],
        ]
    )

    actual_episodes = min(num_episodes, len(test_starts)) if start_positions or test_starts else num_episodes

    for ep in range(actual_episodes):
        start_opt = {"start_pos": test_starts[ep % len(test_starts)]}
        obs, info = env.reset(seed=1000 + ep, options=start_opt)
        if planner is not None:
            planner.reset_history()

        ep_success = False
        steps = 0

        for s in range(max_steps):
            steps += 1
            if random_policy or planner is None:
                action = env.action_space.sample()
            else:
                action = planner.plan(obs, goal_obs)

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                ep_success = True
                break
            if truncated:
                break

        if ep_success:
            successes += 1
            total_steps_list.append(steps)
        else:
            total_steps_list.append(max_steps)

    success_rate = (successes / actual_episodes) * 100.0
    avg_steps = float(np.mean(total_steps_list))

    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "num_episodes": actual_episodes,
    }
