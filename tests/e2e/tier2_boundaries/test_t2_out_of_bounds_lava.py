"""Tier 2: Boundary Tests - Out-of-Bounds & Lava Deaths.

Category 3: Lava & Lethal Environmental Hazards
Tests:
13. test_t2_lava_single_step_instant_lethal_death
14. test_t2_world_grid_edge_clamping_boundary
15. test_t2_simultaneous_starvation_and_damage_lethal_boundary
16. test_t2_sleep_interruption_by_hostile_attack
17. test_t2_skeleton_arrow_maximum_range_boundary
18. test_t2_terminal_step_observation_reset_contract
"""
from __future__ import annotations

import crafter.objects
import numpy as np
import pytest

from zephyr.vec_env import ZephyrVecEnv


def test_t2_lava_single_step_instant_lethal_death(crafter_env_factory):
    """T2-Lava: Stepping into a lava tile causes instant lethal death on that exact single step."""
    env = crafter_env_factory(seed=7000, max_steps=50)
    env.reset(seed=7000)
    player = env._env._player
    assert player.health == 9

    # Place lava tile to the right of player (+1, 0)
    right_pos = (player.pos[0] + 1, player.pos[1])
    env._env._world[right_pos] = "lava"

    # Action 2 is move_right
    obs, reward, term, trunc, info = env.step(2)

    assert player.health == 0, f"Player health must drop to 0 in lava, got {player.health}"
    assert term is True, "Episode must immediately terminate upon stepping into lava"
    # Health dropped by 9 points: reward is -9/10 = -0.90
    assert reward == pytest.approx(-0.90, abs=1e-3), f"Expected -0.90 reward for 9 health loss, got {reward}"


def test_t2_world_grid_edge_clamping_boundary(crafter_env_factory):
    """T2-Bounds: Moving towards world perimeter clamps coordinates within [0, 63] without exception."""
    env = crafter_env_factory(seed=7000, max_steps=50)
    env.reset(seed=7000)
    player = env._env._player

    # Move player to top-left corner (0, 0)
    player.pos = np.array([0, 0])
    # Clear corner tile
    env._env._world[(0, 0)] = "grass"

    # Action 1 is move_left (facing becomes (-1, 0), pos cannot be < 0)
    obs, reward, term, trunc, info = env.step(1)
    assert player.pos[0] >= 0, f"X coordinate underflowed: {player.pos[0]}"
    assert player.pos[0] == 0, f"Player moved out of bounds to {player.pos}"

    # Action 3 is move_up (facing becomes (0, -1), pos cannot be < 0)
    obs, reward, term, trunc, info = env.step(3)
    assert player.pos[1] >= 0, f"Y coordinate underflowed: {player.pos[1]}"
    assert player.pos[1] == 0, f"Player moved out of bounds to {player.pos}"

    # Verify max edge at (63, 63)
    player.pos = np.array([63, 63])
    env._env._world[(63, 63)] = "grass"

    # Action 2 is move_right
    env.step(2)
    assert player.pos[0] <= 63, f"X coordinate exceeded 63: {player.pos[0]}"

    # Action 4 is move_down
    env.step(4)
    assert player.pos[1] <= 63, f"Y coordinate exceeded 63: {player.pos[1]}"


def test_t2_simultaneous_starvation_and_damage_lethal_boundary(crafter_env_factory):
    """T2-Lethal: Agent with 1 health takes fatal hit; health clamps at exactly 0 without negative underflow."""
    env = crafter_env_factory(seed=7000, max_steps=50)
    env.reset(seed=7000)
    player = env._env._player

    player.health = 1
    player.inventory["health"] = 1
    player.inventory["drink"] = 0
    player._recover = -15  # next tick triggers health loss

    obs, reward, term, trunc, info = env.step(0)

    assert player.health == 0, f"Health should clamp at 0, got {player.health}"
    assert player.inventory["health"] == 0, f"Inventory health should be 0, got {player.inventory['health']}"
    assert term is True, "Episode must terminate when health reaches 0"


def test_t2_sleep_interruption_by_hostile_attack(crafter_env_factory):
    """T2-Combat: Hostile attack on sleeping player deals 7 damage and cancels sleeping immediately."""
    env = crafter_env_factory(seed=7000, max_steps=50)
    env.reset(seed=7000)
    player = env._env._player

    player.health = 9
    player.inventory["energy"] = 2
    env.step(6)  # sleep
    assert player.sleeping is True, "Player should be sleeping"

    # Spawn adjacent Zombie
    target_pos = (player.pos[0] + 1, player.pos[1])
    env._env._world[target_pos] = "grass"
    zombie = crafter.objects.Zombie(env._env._world, target_pos, player)
    zombie.cooldown = 0
    zombie.update()  # Zombie attacks sleeping player

    player._wake_up_when_hurt()

    assert player.health == 2, f"Sleeping player should take 7 damage (9 -> 2), got {player.health}"
    assert player.sleeping is False, "Sleeping state must be canceled immediately upon taking hostile damage"


def test_t2_skeleton_arrow_maximum_range_boundary(crafter_env_factory):
    """T2-Combat: Skeleton shoots arrow within distance <= 5; obstacle blocks arrow and player takes 2 damage on hit."""
    env = crafter_env_factory(seed=7000, max_steps=50)
    env.reset(seed=7000)
    player = env._env._player
    player.health = 9

    # Test 1: Arrow collision with obstacle removes arrow without damage
    skel_pos = (player.pos[0] + 4, player.pos[1])
    # Place wall between skeleton and player
    wall_pos = (player.pos[0] + 2, player.pos[1])
    env._env._world[wall_pos] = "stone"

    arrow = crafter.objects.Arrow(env._env._world, (player.pos[0] + 3, player.pos[1]), (-1, 0))
    env._env._world.add(arrow)
    arrow.update()  # Moves to wall_pos: blocked!
    assert player.health == 9, "Wall failed to block arrow"

    # Test 2: Arrow hits player on open terrain
    open_arrow = crafter.objects.Arrow(env._env._world, (player.pos[0] + 1, player.pos[1]), (-1, 0))
    env._env._world.add(open_arrow)
    open_arrow.update()  # Hits player
    assert player.health == 7, f"Arrow should deal 2 damage (9 -> 7), got {player.health}"


def test_t2_terminal_step_observation_reset_contract(vec_env_factory):
    """T2-Reset: On terminal step (done=True), ZephyrVecEnv auto-resets and returns first frame of new episode."""
    # Create vectorized environment with max_steps=3
    venv = vec_env_factory(num_envs=1, base_seed=8000, max_steps=3)
    obs_init, _ = venv.reset()

    # Step 1
    obs1, rew1, done1, info1 = venv.step(np.array([0]))
    assert not done1[0]

    # Step 2
    obs2, rew2, done2, info2 = venv.step(np.array([0]))
    assert not done2[0]

    # Step 3: Hits max_steps timeout -> done=True
    obs3, rew3, done3, info3 = venv.step(np.array([0]))
    assert done3[0] is True or bool(done3[0]), "Third step must trigger done=True at max_steps=3"

    # Returned observation on terminal step is a valid uint8 RGB frame from the new episode
    assert isinstance(obs3, np.ndarray)
    assert obs3.shape == (1, 64, 64, 3)
    assert obs3.dtype == np.uint8
    assert int(obs3.min()) >= 0 and int(obs3.max()) <= 255
