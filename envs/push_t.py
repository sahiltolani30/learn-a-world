"""
Custom 2D Push-T Gymnasium Environment.
Features:
- 2D visual physics: end-effector circle pushes a T-shaped block toward a target zone.
- Continuous sliding friction and rotation dynamics.
- Rendered visual observation: Box(0, 255, (64, 64, 3), uint8).
- Discrete action space: 5 actions (0: NOOP, 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT).
- Strict compliance with Gymnasium API (passes check_env).
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw

class PushTEnv(gym.Env):
    """
    Gymnasium environment where an agent pushes a T-shaped object to a target pose.
    Observation: Box(0, 255, (64, 64, 3), uint8)
    Action: Discrete(5)
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    # Action deltas: (dx, dy)
    ACTIONS = {
        0: (0.0, 0.0),    # NOOP
        1: (0.0, -2.5),   # UP
        2: (0.0, 2.5),    # DOWN
        3: (-2.5, 0.0),   # LEFT
        4: (2.5, 0.0),    # RIGHT
    }

    def __init__(
        self,
        max_steps: int = 100,
        render_size: int = 64,
        render_mode: str = "rgb_array",
    ):
        super().__init__()
        self.max_steps = max_steps
        self.render_size = render_size
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(render_size, render_size, 3),
            dtype=np.uint8,
        )

        # Physics parameters
        self.agent_radius = 3.0
        self.friction = 0.85
        self.rot_friction = 0.80

        # State variables
        self.agent_pos = np.array([16.0, 16.0], dtype=np.float32)
        self.block_pos = np.array([32.0, 32.0], dtype=np.float32)
        self.block_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.block_angle = 0.0
        self.block_ang_vel = 0.0
        self.target_pos = np.array([44.0, 44.0], dtype=np.float32)
        self.target_angle = 0.0
        self.step_count = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random

        # Randomize agent start near corner
        self.agent_pos = np.array([rng.uniform(12.0, 20.0), rng.uniform(12.0, 20.0)], dtype=np.float32)
        # Block near center
        self.block_pos = np.array([rng.uniform(28.0, 36.0), rng.uniform(28.0, 36.0)], dtype=np.float32)
        self.block_vel = np.zeros(2, dtype=np.float32)
        self.block_angle = float(rng.uniform(-0.5, 0.5))
        self.block_ang_vel = 0.0
        # Target pose
        self.target_pos = np.array([44.0, 44.0], dtype=np.float32)
        self.target_angle = 0.0
        self.step_count = 0

        obs = self.render()
        info = {
            "dist_to_goal": float(np.linalg.norm(self.block_pos - self.target_pos)),
            "coverage": 0.0,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        dx, dy = self.ACTIONS[int(action)]

        # Move agent
        prev_agent_pos = self.agent_pos.copy()
        self.agent_pos[0] = np.clip(self.agent_pos[0] + dx, 4.0, self.render_size - 4.0)
        self.agent_pos[1] = np.clip(self.agent_pos[1] + dy, 4.0, self.render_size - 4.0)
        agent_v = self.agent_pos - prev_agent_pos

        # Check collision with block center of mass
        delta = self.block_pos - self.agent_pos
        dist = float(np.linalg.norm(delta))
        collision_dist = self.agent_radius + 6.0  # approximate T-block collision sphere

        if dist < collision_dist and dist > 1e-4:
            normal = delta / dist
            push_force = np.dot(agent_v, normal)
            if push_force > 0:
                self.block_vel += normal * (push_force * 1.2)
                # Torque = lever arm cross product
                perp = np.array([-normal[1], normal[0]])
                torque = float(np.dot(agent_v, perp)) * 0.05
                self.block_ang_vel += torque

        # Integrate block dynamics
        self.block_pos += self.block_vel
        self.block_angle += self.block_ang_vel
        self.block_vel *= self.friction
        self.block_ang_vel *= self.rot_friction

        # Boundary bounce
        self.block_pos[0] = np.clip(self.block_pos[0], 8.0, self.render_size - 8.0)
        self.block_pos[1] = np.clip(self.block_pos[1], 8.0, self.render_size - 8.0)

        # Distance to target
        dist_to_goal = float(np.linalg.norm(self.block_pos - self.target_pos))
        angle_diff = abs(float(self.block_angle - self.target_angle)) % (2 * math.pi)
        coverage = float(np.clip(1.0 - (dist_to_goal / 40.0), 0.0, 1.0))
        reward = float(coverage * 0.1)

        terminated = bool(dist_to_goal < 6.0 and angle_diff < 0.3)
        if terminated:
            reward += 1.0
        truncated = bool(self.step_count >= self.max_steps)

        obs = self.render()
        info = {
            "dist_to_goal": dist_to_goal,
            "coverage": coverage,
            "success": terminated,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Renders 64x64 RGB observation of target, T-block, and agent."""
        img = Image.new("RGB", (self.render_size, self.render_size), (245, 247, 250))
        draw = ImageDraw.Draw(img)

        # 1. Draw Target zone (pale green outline)
        self._draw_t_shape(draw, self.target_pos, self.target_angle, fill=(200, 245, 210), outline=(34, 197, 94))

        # 2. Draw T-block (navy blue)
        self._draw_t_shape(draw, self.block_pos, self.block_angle, fill=(30, 58, 138), outline=(15, 23, 42))

        # 3. Draw Agent circle (coral red)
        ax, ay = self.agent_pos
        r = self.agent_radius
        draw.ellipse([ax - r, ay - r, ax + r, ay + r], fill=(239, 68, 68), outline=(185, 28, 28))

        return np.array(img, dtype=np.uint8)

    def _draw_t_shape(
        self,
        draw: ImageDraw.ImageDraw,
        pos: np.ndarray,
        angle: float,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
    ) -> None:
        """Helper to render rotated T-shape polygon."""
        cx, cy = float(pos[0]), float(pos[1])
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        def rotate(x: float, y: float) -> Tuple[float, float]:
            return (cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a)

        # T-bar: horizontal rect [-8, -4] to [8, 0]
        # T-stem: vertical rect [-3, 0] to [3, 8]
        points = [
            rotate(-8, -4), rotate(8, -4), rotate(8, 0),
            rotate(3, 0), rotate(3, 8), rotate(-3, 8),
            rotate(-3, 0), rotate(-8, 0),
        ]
        draw.polygon(points, fill=fill, outline=outline)

    def close(self):
        pass
