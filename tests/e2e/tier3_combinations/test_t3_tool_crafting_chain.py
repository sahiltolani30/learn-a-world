"""Tier 3: Tool Crafting Chain & Causal Progression Combinations.

Covers combinatorial interactions between:
- F1 (Pixels), F2 (17 Actions), F3 (Causal Tool Progression), F7 (Dynamics Predictor),
  F12 (Station Affordance Probing), F16 (Multi-Game Loop).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from envs.crafter_wrapper import CrafterEnv
from zephyr.config import INVENTORY_KEYS
from zephyr.wm import ZephyrWM


def test_t3_chain_01_visual_perception_to_wood_gathering(crafter_env_factory, tiny_test_cfg):
    """T3-CHAIN-01: Perception -> action 5 -> wood gathering state & visual change."""
    env = crafter_env_factory(seed=7000, max_steps=50)
    obs_0, _ = env.reset(seed=7000)
    target_pos = tuple(env._env._player.pos + env._env._player.facing)
    env._env._world[target_pos] = "tree"
    assert env._env._player.inventory["wood"] == 0

    wm = ZephyrWM(tiny_test_cfg).to("cpu").eval()
    with torch.no_grad():
        z_0 = wm.encode(torch.from_numpy(np.ascontiguousarray(obs_0)).unsqueeze(0))

    obs_1, reward, term, trunc, info = env.step(5)

    assert reward == 1.0, f"Expected 1.0 for collect_wood, got {reward}"
    assert env._env._player.inventory["wood"] == 1
    assert info["achievements"]["collect_wood"] == 1

    with torch.no_grad():
        z_1 = wm.encode(torch.from_numpy(np.ascontiguousarray(obs_1)).unsqueeze(0))

    latent_diff = torch.norm(z_1 - z_0, p=2).item()
    assert latent_diff > 0.05, f"Latent difference {latent_diff} too small"
    assert (obs_1[-10:, :, :] != obs_0[-10:, :, :]).any(), "HUD strip did not update"


def test_t3_chain_02_wood_gathering_accumulation_to_table_placement(crafter_env_factory):
    """T3-CHAIN-02: Wood gathering accumulation (wood>=2) -> table placement (action 8)."""
    env = crafter_env_factory(seed=7000, max_steps=50)
    env.reset(seed=7000)
    target_pos = tuple(env._env._player.pos + env._env._player.facing)
    env._env._world[target_pos] = "grass"
    env._env._player.inventory["wood"] = 2

    obs_table, reward, term, trunc, info = env.step(8)

    assert env._env._player.inventory["wood"] == 0
    assert info["achievements"]["place_table"] == 1
    assert reward == 1.0
    tile = env._env._world[target_pos]
    material = tile[0] if isinstance(tile, tuple) else tile
    assert material == "table"


def test_t3_chain_03_crafting_station_chebyshev_proximity_boundary(crafter_env_factory):
    """T3-CHAIN-03: Table proximity boundary (Chebyshev dist <= 1 vs > 1) for wood pickaxe."""
    env = crafter_env_factory(seed=7000, max_steps=50)
    env.reset(seed=7000)

    # Case A: Table distant (> 1), wood=1 -> craft fails
    env._env._player.inventory["wood"] = 1
    env._env._player.inventory["wood_pickaxe"] = 0
    obs, reward, term, trunc, info = env.step(11)
    assert env._env._player.inventory["wood"] == 1
    assert env._env._player.inventory["wood_pickaxe"] == 0
    assert reward == 0.0
    assert info["achievements"]["make_wood_pickaxe"] == 0

    # Case B: Table adjacent (dist <= 1), wood=1 -> craft succeeds
    target_pos = tuple(env._env._player.pos + env._env._player.facing)
    env._env._world[target_pos] = "table"
    obs, reward, term, trunc, info = env.step(11)
    assert env._env._player.inventory["wood"] == 0
    assert env._env._player.inventory["wood_pickaxe"] == 1
    assert reward == 1.0
    assert info["achievements"]["make_wood_pickaxe"] == 1


def test_t3_chain_04_station_proximity_to_tool_fabrication(crafter_env_factory, mock_wm, mock_heads):
    """T3-CHAIN-04: Tool fabrication state transition & readout head representation."""
    env = crafter_env_factory(seed=7000, max_steps=50)
    env.reset(seed=7000)
    target_pos = tuple(env._env._player.pos + env._env._player.facing)
    env._env._world[target_pos] = "table"
    env._env._player.inventory["wood"] = 1
    env._env._player.inventory["wood_pickaxe"] = 0

    obs, reward, term, trunc, info = env.step(11)

    assert env._env._player.inventory["wood"] == 0
    assert env._env._player.inventory["wood_pickaxe"] == 1
    assert reward == 1.0
    assert info["achievements"]["make_wood_pickaxe"] == 1

    with torch.no_grad():
        z = mock_wm.encode(torch.from_numpy(np.ascontiguousarray(obs)).unsqueeze(0))
        h = mock_wm.init_state(1, "cpu")
        s = mock_heads.state_input(z, h)
        readout = mock_heads.readout(s)
    assert readout.shape == (1, len(INVENTORY_KEYS))
    pickaxe_idx = INVENTORY_KEYS.index("wood_pickaxe")
    assert not torch.isnan(readout[0, pickaxe_idx])


def test_t3_chain_05_tool_possession_to_stone_mining_causal_affordance(crafter_env_factory):
    """T3-CHAIN-05: Possessing pickaxe enables stone mining affordance (action 5)."""
    env = crafter_env_factory(seed=7000, max_steps=50)
    env.reset(seed=7000)
    target_pos = tuple(env._env._player.pos + env._env._player.facing)
    env._env._world[target_pos] = "stone"
    env._env._player.inventory["wood_pickaxe"] = 1
    env._env._player.inventory["stone"] = 0

    obs, reward, term, trunc, info = env.step(5)

    assert env._env._player.inventory["stone"] == 1
    assert info["achievements"]["collect_stone"] == 1
    assert reward == 1.0
    tile = env._env._world[target_pos]
    material = tile[0] if isinstance(tile, tuple) else tile
    assert material == "path"


def test_t3_chain_06_negative_causal_invariance_barehanded_stone_striking(crafter_env_factory):
    """T3-CHAIN-06: Barehanded stone striking (wood_pickaxe=0) has zero causal effect."""
    env = crafter_env_factory(seed=7000, max_steps=50)
    env.reset(seed=7000)
    target_pos = tuple(env._env._player.pos + env._env._player.facing)
    env._env._world[target_pos] = "stone"
    env._env._player.inventory["wood_pickaxe"] = 0
    env._env._player.inventory["stone"] = 0

    obs, reward, term, trunc, info = env.step(5)

    assert env._env._player.inventory["stone"] == 0
    tile = env._env._world[target_pos]
    material = tile[0] if isinstance(tile, tuple) else tile
    assert material == "stone"
    assert reward == 0.0
    assert info["achievements"]["collect_stone"] == 0


def test_t3_chain_07_full_multi_hop_closed_loop_progression_sequence(crafter_env_factory):
    """T3-CHAIN-07: Full causal progression Wood -> Table -> Pickaxe -> Stone."""
    env = crafter_env_factory(seed=7000, max_steps=50)
    env.reset(seed=7000)
    target_pos = tuple(env._env._player.pos + env._env._player.facing)

    total_rewards = 0.0

    # Hop 1: Collect wood (first tree)
    env._env._world[target_pos] = "tree"
    obs, r1, _, _, info1 = env.step(5)
    total_rewards += r1
    assert info1["achievements"]["collect_wood"] == 1

    # Hop 2: Collect wood (second tree)
    env._env._world[target_pos] = "tree"
    obs, r2, _, _, _ = env.step(5)
    total_rewards += r2
    assert env._env._player.inventory["wood"] == 2

    # Hop 3: Place table
    env._env._world[target_pos] = "grass"
    obs, r3, _, _, info3 = env.step(8)
    total_rewards += r3
    assert info3["achievements"]["place_table"] == 1
    tile = env._env._world[target_pos]
    material = tile[0] if isinstance(tile, tuple) else tile
    assert material == "table"

    # Hop 4: Make wood pickaxe (needs wood=1 and table)
    env._env._player.inventory["wood"] = 1
    obs, r4, _, _, info4 = env.step(11)
    total_rewards += r4
    assert info4["achievements"]["make_wood_pickaxe"] == 1
    assert env._env._player.inventory["wood_pickaxe"] == 1

    # Hop 5: Mine stone
    env._env._world[target_pos] = "stone"
    obs, r5, _, _, info5 = env.step(5)
    total_rewards += r5
    assert info5["achievements"]["collect_stone"] == 1
    assert env._env._player.inventory["stone"] == 1

    assert total_rewards == 4.0
    assert env._env._player.inventory["wood_pickaxe"] == 1
    assert env._env._player.inventory["stone"] == 1
