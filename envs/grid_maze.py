"""
Custom Grid Maze Gymnasium Environment.
Features:
- Configurable grid layout with walls, start, goal, and special slippery zones.
- Continuous rendered visual observations: (64, 64, 3) uint8 image.
- 4 discrete actions: UP, DOWN, LEFT, RIGHT.
- Controllable physics: normal friction vs slippery ice zones (used for Phase 4 adaptation).
- Strict compliance with Gymnasium API (passes env_checker).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridMazeEnv(gym.Env):
    """
    Custom 2D Grid Maze environment with rendered image observations.
    Observation: Box(0, 255, (64, 64, 3), uint8)
    Actions: Discrete(4) -> 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    # Movement vectors: (row_delta, col_delta)
    ACTIONS = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),   # RIGHT
    }

    # Palette (RGB)
    COLOR_FLOOR = np.array([241, 245, 249], dtype=np.uint8)     # Off-white / light slate
    COLOR_WALL = np.array([30, 41, 59], dtype=np.uint8)          # Dark slate
    COLOR_GOAL = np.array([34, 197, 94], dtype=np.uint8)         # Emerald green
    COLOR_AGENT = np.array([239, 68, 68], dtype=np.uint8)        # Vibrant red / coral
    COLOR_SLIPPERY = np.array([56, 189, 248], dtype=np.uint8)    # Ice cyan

    # Default 9x9 layout:
    # 1: Wall, 0: Floor, 2: Goal, 3: Slippery zone
    DEFAULT_LAYOUT = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 2, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 3, 1, 1, 0, 1],
        [1, 0, 0, 3, 3, 3, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    def __init__(
        self,
        layout: Optional[List[List[int]]] = None,
        max_steps: int = 100,
        render_size: int = 64,
        slippery_active: bool = False,
        slip_probability: float = 0.5,
        render_mode: str = "rgb_array",
    ):
        super().__init__()
        self.raw_layout = np.array(layout if layout is not None else self.DEFAULT_LAYOUT, dtype=np.int32)
        self.height, self.width = self.raw_layout.shape
        self.max_steps = max_steps
        self.render_size = render_size
        self.slippery_active = slippery_active
        self.slip_probability = slip_probability
        self.render_mode = render_mode

        # Find initial agent and goal positions
        self._find_special_tiles()

        # Gymnasium spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.render_size, self.render_size, 3),
            dtype=np.uint8,
        )

        # Dynamic state
        self.agent_pos = np.array([1, 1], dtype=np.int32)
        self.step_count = 0

    def _find_special_tiles(self):
        """Locates startable floor tiles and goal position."""
        self.goal_pos = np.argwhere(self.raw_layout == 2)
        if len(self.goal_pos) > 0:
            self.goal_pos = self.goal_pos[0]
        else:
            self.goal_pos = np.array([self.height - 2, self.width - 2], dtype=np.int32)

        self.spawn_positions = np.argwhere((self.raw_layout == 0) | (self.raw_layout == 3))
        # Exclude goal from spawn
        self.spawn_positions = [
            pos for pos in self.spawn_positions if not np.array_equal(pos, self.goal_pos)
        ]

    def set_slippery(self, active: bool, probability: float = 0.5):
        """Toggle slippery physics (key mechanism for Phase 4 adaptation tests)."""
        self.slippery_active = active
        self.slip_probability = probability

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0

        # Choose start position: options can specify start_pos, else default spawn (1, 1) or random
        if options and "start_pos" in options:
            self.agent_pos = np.array(options["start_pos"], dtype=np.int32)
        elif options and options.get("random_start", False):
            idx = self.np_random.integers(0, len(self.spawn_positions))
            self.agent_pos = self.spawn_positions[idx].copy()
        else:
            self.agent_pos = np.array([1, 1], dtype=np.int32)

        obs = self._render_frame()
        info = {
            "agent_pos": self.agent_pos.copy(),
            "goal_pos": self.goal_pos.copy(),
            "is_on_ice": bool(self.raw_layout[tuple(self.agent_pos)] == 3),
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        delta = self.ACTIONS[int(action)]

        # Check if currently on ice tile
        curr_tile = self.raw_layout[tuple(self.agent_pos)]
        if self.slippery_active and curr_tile == 3:
            # When on ice with slippery active, random deviation or slip forward extra tile
            if self.np_random.random() < self.slip_probability:
                # Random drift action
                drift_action = self.np_random.integers(0, 4)
                delta = self.ACTIONS[int(drift_action)]

        intended_pos = self.agent_pos + np.array(delta, dtype=np.int32)

        # Collision check with walls (boundary and interior)
        if (
            0 <= intended_pos[0] < self.height
            and 0 <= intended_pos[1] < self.width
            and self.raw_layout[tuple(intended_pos)] != 1
        ):
            self.agent_pos = intended_pos

        reached_goal = np.array_equal(self.agent_pos, self.goal_pos)
        terminated = bool(reached_goal)
        truncated = bool(self.step_count >= self.max_steps)

        reward = 1.0 if reached_goal else -0.01  # Small step penalty to encourage shortest path

        obs = self._render_frame()
        info = {
            "agent_pos": self.agent_pos.copy(),
            "goal_pos": self.goal_pos.copy(),
            "reached_goal": reached_goal,
            "step_count": self.step_count,
            "is_on_ice": bool(self.raw_layout[tuple(self.agent_pos)] == 3),
        }
        return obs, reward, terminated, truncated, info

    def _render_frame(self) -> np.ndarray:
        """Renders the maze grid into a 64x64x3 RGB image."""
        cell_h = self.render_size // self.height
        cell_w = self.render_size // self.width

        # Base canvas
        canvas = np.zeros((self.height * cell_h, self.width * cell_w, 3), dtype=np.uint8)

        # Draw grid tiles
        for r in range(self.height):
            for c in range(self.width):
                tile_type = self.raw_layout[r, c]
                if tile_type == 1:
                    color = self.COLOR_WALL
                elif tile_type == 2:
                    color = self.COLOR_GOAL
                elif tile_type == 3:
                    color = self.COLOR_SLIPPERY
                else:
                    color = self.COLOR_FLOOR
                canvas[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = color

        # Draw Agent on top
        ar, ac = self.agent_pos
        pad_h = max(1, cell_h // 5)
        pad_w = max(1, cell_w // 5)
        canvas[
            ar * cell_h + pad_h : (ar + 1) * cell_h - pad_h,
            ac * cell_w + pad_w : (ac + 1) * cell_w - pad_w,
        ] = self.COLOR_AGENT

        # Pad or resize to exact (render_size, render_size) if not exact multiple
        if canvas.shape[0] != self.render_size or canvas.shape[1] != self.render_size:
            from PIL import Image
            img = Image.fromarray(canvas)
            img = img.resize((self.render_size, self.render_size), Image.NEAREST)
            canvas = np.array(img, dtype=np.uint8)

        return canvas

    def render(self) -> np.ndarray:
        return self._render_frame()

    def close(self):
        return None
