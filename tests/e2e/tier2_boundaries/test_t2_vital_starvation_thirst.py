"""Tier 2: Boundary Tests - Vital Starvation and Thirst.

Category 2: Vital Depletion Starving/Thirsting at Tick 189/234
Tests:
7. test_t2_thirst_exact_step_188_vs_189_boundary
8. test_t2_hunger_exact_step_233_vs_234_boundary
9. test_t2_health_degeneration_first_damage_tick_205
10. test_t2_dehydration_death_spiral_tick_333
11. test_t2_sleeping_vital_rate_reduction_boundary
12. test_t2_vitals_zero_clamping_and_saturation
"""
from __future__ import annotations

import pytest
import numpy as np


def test_t2_thirst_exact_step_188_vs_189_boundary(crafter_env_factory):
    """T2-Vitals: Drink starts at 9, decrements every 21 ticks, and hits exactly 0 at tick 189."""
    env = crafter_env_factory(seed=7000, max_steps=200)
    env.reset(seed=7000)

    for step in range(1, 190):
        env.step(0)
        drink = env._env._player.inventory["drink"]
        if step == 188:
            assert drink > 0, f"Drink depleted prematurely at step {step} (drink={drink})"
        elif step == 189:
            assert drink == 0, f"Drink should reach exactly 0 at tick 189, got {drink}"


def test_t2_hunger_exact_step_233_vs_234_boundary(crafter_env_factory):
    """T2-Vitals: Food starts at 9, decrements every 26 ticks, and hits exactly 0 at tick 234."""
    env = crafter_env_factory(seed=7000, max_steps=250)
    env.reset(seed=7000)

    for step in range(1, 235):
        # Keep hydrated so thirst doesn't cause health degeneration
        env._env._player.inventory["drink"] = 9
        env.step(0)
        food = env._env._player.inventory["food"]
        if step == 233:
            assert food > 0, f"Food depleted prematurely at step {step} (food={food})"
        elif step == 234:
            assert food == 0, f"Food should reach exactly 0 at tick 234, got {food}"


def test_t2_health_degeneration_first_damage_tick_205(crafter_env_factory):
    """T2-Vitals: When drink is 0, recovery decrements by 1/tick; after 16 ticks (< -15), first damage occurs."""
    env = crafter_env_factory(seed=7000, max_steps=100)
    env.reset(seed=7000)
    player = env._env._player

    # Set zero drink with zero recovery counter
    player.inventory["drink"] = 0
    player.inventory["food"] = 9
    player.inventory["energy"] = 9
    player.inventory["health"] = 9
    player._recover = 0
    player._thirst = 0

    # Step 15 ticks: _recover goes from -1 down to -15 (not < -15), health stays 9
    for i in range(15):
        obs, reward, term, trunc, info = env.step(0)
        assert player.health == 9, f"Health dropped prematurely at step {i+1}, health={player.health}"

    # 16th tick: _recover hits -16 (< -15), triggering health -= 1 and resetting _recover to 0
    obs, reward, term, trunc, info = env.step(0)
    assert player.health == 8, f"Health should decrement to 8 on 16th tick of dehydration, got {player.health}"
    assert reward == pytest.approx(-0.10, abs=1e-3), f"Expected -0.1 reward for health drop, got {reward}"


def test_t2_dehydration_death_spiral_tick_333(crafter_env_factory):
    """T2-Vitals: Sustained dehydration causes 9 health losses over exactly 9 * 16 = 144 ticks."""
    env = crafter_env_factory(seed=7000, max_steps=200)
    env.reset(seed=7000)
    player = env._env._player

    # Start dehydration state
    player.inventory["drink"] = 0
    player.inventory["food"] = 9
    player.inventory["energy"] = 9
    player.inventory["health"] = 9
    player._recover = 0

    # 9 health * 16 ticks = 144 ticks until death
    for tick in range(1, 145):
        # Keep food and energy full so hunger/fatigue don't accelerate death
        player.inventory["food"] = 9
        player.inventory["energy"] = 9
        obs, reward, term, trunc, info = env.step(0)
        if tick < 144:
            assert not term, f"Agent died prematurely at tick {tick}"
        elif tick == 144:
            assert term, f"Agent should be dead at tick 144 (189+144=333 timeline), health={player.health}"
            assert player.health == 0, f"Health should reach 0, got {player.health}"


def test_t2_sleeping_vital_rate_reduction_boundary(crafter_env_factory):
    """T2-Vitals: Sleeping reduces metabolic depletion rate by 50% (+0.5/tick thirst/hunger instead of +1.0)."""
    env = crafter_env_factory(seed=7000, max_steps=100)
    env.reset(seed=7000)
    player = env._env._player

    # Test awake increment
    player.sleeping = False
    player._thirst = 0
    env.step(0)
    awake_thirst_delta = player._thirst
    assert awake_thirst_delta == pytest.approx(1.0, abs=1e-4)

    # Set energy < 9 so sleep action is accepted
    player.inventory["energy"] = 2
    env.step(6)  # action 6 = sleep
    assert player.sleeping is True, "Player should be sleeping when energy < 9"

    player._thirst = 0
    env.step(0)  # while sleeping, any action or noop steps sleep
    sleeping_thirst_delta = player._thirst
    assert sleeping_thirst_delta == pytest.approx(0.5, abs=1e-4), (
        f"Sleeping thirst increment should be 0.5/tick, got {sleeping_thirst_delta}"
    )


def test_t2_vitals_zero_clamping_and_saturation(crafter_env_factory):
    """T2-Vitals: Inventory vitals (food, drink, health) never exceed 9 or underflow below 0."""
    env = crafter_env_factory(seed=7000, max_steps=50)
    env.reset(seed=7000)
    player = env._env._player

    # Test saturation: cannot exceed 9 in normal Crafter mechanics
    player.inventory["health"] = 9
    player.inventory["food"] = 9
    player.inventory["drink"] = 9

    # Drink water facing water tile
    target_pos = tuple(player.pos + player.facing)
    env._env._world[target_pos] = "water"
    env.step(5)  # do
    assert player.inventory["drink"] <= 9, f"Drink should not exceed max 9, got {player.inventory['drink']}"

    # Test lower bound clamping: cannot be negative
    player.inventory["drink"] = 0
    player.inventory["food"] = 0
    env.step(0)
    assert player.inventory["drink"] >= 0, "Drink underflowed below 0"
    assert player.inventory["food"] >= 0, "Food underflowed below 0"
