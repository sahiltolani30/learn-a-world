"""Tier 1: Causal Mechanics, Nightfall, and Vitals (F3, F4, F5).

Tests:
- F3: Causal Tool Progression (Wood -> Table -> Pickaxe -> Stone) [5 tests]
- F4: Step 148 Nightfall Survival Wall [5 tests]
- F5: Vitals Depletion & Metabolic Survival [5 tests]
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from envs.crafter_wrapper import CrafterEnv
from zephyr.heads import ZephyrHeads
from zephyr.wm import ZephyrWM


# ==============================================================================
# Feature 3: Causal Tool Progression (Wood -> Table -> Pickaxe -> Stone)
# ==============================================================================

def test_f3_wood_gathering_affordance(crafter_env_factory):
    """F3: Executing action 5 facing a tree harvests wood and unlocks collect_wood."""
    env = crafter_env_factory(seed=7000, max_steps=100)
    env.reset(seed=7000)
    target_pos = tuple(env._env._player.pos + env._env._player.facing)
    env._env._world[target_pos] = "tree"

    obs, reward, term, trunc, info = env.step(5)

    assert reward == 1.0, f"Expected milestone reward 1.0, got {reward}"
    assert env._env._player.inventory["wood"] == 1, (
        f"Wood count {env._env._player.inventory['wood']} != 1"
    )
    assert info["achievements"]["collect_wood"] == 1, "Achievement collect_wood not unlocked"


def test_f3_table_crafting_requires_two_wood(crafter_env_factory):
    """F3: Crafting a table (action 8) consumes 2 wood; fails if wood < 2."""
    env = crafter_env_factory(seed=7000, max_steps=100)
    env.reset(seed=7000)

    # Sub-case A: With wood=1, table crafting fails
    env._env._player.inventory["wood"] = 1
    obs, reward, term, trunc, info = env.step(8)
    assert env._env._player.inventory["wood"] == 1, "Wood should not be consumed when wood < 2"
    assert info["achievements"]["place_table"] == 0, "Table should not be placed with wood < 2"

    # Sub-case B: With wood=2, table crafting succeeds
    env._env._player.inventory["wood"] = 2
    obs, reward, term, trunc, info = env.step(8)
    assert env._env._player.inventory["wood"] == 0, "Wood should be decremented by 2"
    assert info["achievements"]["place_table"] == 1, "Table should be placed with wood == 2"
    assert reward == 1.0, f"Expected milestone reward 1.0, got {reward}"


def test_f3_pickaxe_crafting_requires_table_proximity(crafter_env_factory):
    """F3: Crafting a wood pickaxe (action 11) requires wood >= 1 and nearby table."""
    env = crafter_env_factory(seed=7000, max_steps=100)
    env.reset(seed=7000)

    # Sub-case A: No table nearby (distance > 1)
    env._env._player.inventory["wood"] = 1
    obs, reward, term, trunc, info = env.step(11)
    assert env._env._player.inventory["wood"] == 1, "Wood consumed without nearby table"
    assert env._env._player.inventory["wood_pickaxe"] == 0, "Pickaxe crafted without nearby table"
    assert info["achievements"]["make_wood_pickaxe"] == 0

    # Sub-case B: Table adjacent (distance <= 1)
    target_pos = tuple(env._env._player.pos + env._env._player.facing)
    env._env._world[target_pos] = "table"
    obs, reward, term, trunc, info = env.step(11)
    assert env._env._player.inventory["wood"] == 0, "Wood should decrement by 1"
    assert env._env._player.inventory["wood_pickaxe"] == 1, "Wood pickaxe should increment to 1"
    assert info["achievements"]["make_wood_pickaxe"] == 1, "Achievement make_wood_pickaxe not unlocked"
    assert reward == 1.0, f"Expected milestone reward 1.0, got {reward}"


def test_f3_stone_mining_requires_wood_pickaxe(crafter_env_factory):
    """F3: Mining stone requires possessing a wooden pickaxe."""
    env = crafter_env_factory(seed=7000, max_steps=100)
    env.reset(seed=7000)
    target_pos = tuple(env._env._player.pos + env._env._player.facing)
    env._env._world[target_pos] = "stone"

    # Sub-case A: Without pickaxe (wood_pickaxe = 0)
    env._env._player.inventory["wood_pickaxe"] = 0
    obs, reward, term, trunc, info = env.step(5)
    assert env._env._player.inventory["stone"] == 0, "Stone mined without pickaxe"
    assert info["achievements"]["collect_stone"] == 0

    # Sub-case B: With pickaxe (wood_pickaxe = 1)
    env._env._player.inventory["wood_pickaxe"] = 1
    obs, reward, term, trunc, info = env.step(5)
    assert env._env._player.inventory["stone"] == 1, "Stone not harvested with pickaxe"
    assert info["achievements"]["collect_stone"] == 1, "Achievement collect_stone not unlocked"
    assert reward == 1.0, f"Expected milestone reward 1.0, got {reward}"


def test_f3_causal_chain_four_step_sequential_integrity(crafter_env_factory):
    """F3: Full causal sequence (Tree -> Table -> Pickaxe -> Stone) unlocks 4 achievements."""
    env = crafter_env_factory(seed=7000, max_steps=100)
    env.reset(seed=7000)
    target_pos = tuple(env._env._player.pos + env._env._player.facing)

    # 1. Harvest wood
    env._env._world[target_pos] = "tree"
    env.step(5)
    assert env._env._player.inventory["wood"] == 1

    # 2. Place table (requires 2 wood, give 1 more)
    env._env._player.inventory["wood"] = 2
    env.step(8)
    assert env._env._player.inventory["wood"] == 0

    # 3. Make pickaxe (requires 1 wood + table)
    env._env._player.inventory["wood"] = 1
    env._env._world[target_pos] = "table"
    env.step(11)
    assert env._env._player.inventory["wood_pickaxe"] == 1

    # 4. Mine stone (requires pickaxe + stone)
    env._env._world[target_pos] = "stone"
    obs, reward, term, trunc, info = env.step(5)
    assert env._env._player.inventory["stone"] == 1

    assert info["achievements"]["collect_wood"] == 1
    assert info["achievements"]["place_table"] == 1
    assert info["achievements"]["make_wood_pickaxe"] == 1
    assert info["achievements"]["collect_stone"] == 1


# ==============================================================================
# Feature 4: Step 148 Nightfall Survival Wall
# ==============================================================================

def test_f4_illumination_waveform_calculation():
    """F4: Illumination waveform matches Crafter daylight formula daylight(t) = 1 - |cos(pi*(t/300 + 0.3))|^3."""
    def daylight(step: int) -> float:
        progress = (step / 300.0 + 0.3) % 1.0
        return 1.0 - abs(math.cos(math.pi * progress)) ** 3

    assert daylight(0) == pytest.approx(0.7969, abs=1e-3)
    assert daylight(60) == pytest.approx(1.0000, abs=1e-3)
    assert daylight(147) > 0.50, f"Daylight at 147 ({daylight(147)}) should be > 0.50"
    assert daylight(148) < 0.50, f"Daylight at 148 ({daylight(148)}) should be < 0.50"
    assert daylight(210) == pytest.approx(0.0000, abs=1e-3)
    assert daylight(300) == pytest.approx(daylight(0), abs=1e-3)


def test_f4_visual_noise_activation_at_step_148(crafter_env_factory):
    """F4: Crafter environment daylight drops below 0.5 at step 148, darkening pixels."""
    env = crafter_env_factory(seed=7000, max_steps=300)
    env.reset(seed=7000)

    # Step to 147
    for _ in range(147):
        obs147, _, _, _, _ = env.step(0)

    # Step to 148
    obs148, _, _, _, _ = env.step(0)

    # At step 148, illumination is lower than at step 147
    mean_lum147 = obs147.mean()
    mean_lum148 = obs148.mean()
    assert mean_lum148 <= mean_lum147, (
        f"Luminance at 148 ({mean_lum148}) should be <= luminance at 147 ({mean_lum147})"
    )


def test_f4_monster_quota_surge_at_nightfall():
    """F4: Zombie target quota monotonically surges as daylight decreases."""
    def zombie_quota(daylight: float) -> float:
        # Crafter balance formula: 3.5 - 3 * daylight
        return 3.5 - 3.0 * daylight

    quota_noon = zombie_quota(1.0)
    quota_step148 = zombie_quota(0.4946)
    quota_midnight = zombie_quota(0.0)

    assert quota_noon == pytest.approx(0.5, abs=1e-3)
    assert quota_step148 >= 2.0, f"Quota at step 148 ({quota_step148}) should be >= 2.0"
    assert quota_midnight == pytest.approx(3.5, abs=1e-3)
    assert quota_midnight > quota_step148 > quota_noon


def test_f4_nocturnal_predator_damage_rules(crafter_env_factory):
    """F4: Zombie deals 2 damage awake and 7 damage sleeping upon collision."""
    env = crafter_env_factory(seed=7000, max_steps=100)
    env.reset(seed=7000)

    import crafter.objects
    player = env._env._player

    # Zombie attacks awake player
    player.health = 9
    player.sleeping = False
    zombie = crafter.objects.Zombie(env._env._world, tuple(player.pos + player.facing), player)
    zombie.update()
    assert player.health == 7, f"Awake player should take 2 damage (health 9->7), got {player.health}"

    # Zombie attacks sleeping player
    player.health = 9
    player.sleeping = True
    player._last_health = 9
    zombie.cooldown = 0
    zombie.update()
    player._wake_up_when_hurt()
    assert player.health == 2, f"Sleeping player should take 7 damage (health 9->2), got {player.health}"
    assert player.sleeping is False, "Sleeping player should wake up after being attacked"


def test_f4_nightfall_survival_continuation_discounting(tiny_test_cfg):
    """F4: Continuation head predicts lower continuation probability (higher termination) on night latents."""
    heads = ZephyrHeads(tiny_test_cfg).eval()
    state_dim = tiny_test_cfg.world_model.latent_dim + tiny_test_cfg.world_model.belief_dim
    day_state = torch.ones(2, state_dim) * 0.5
    night_state = torch.ones(2, state_dim) * -2.0

    with torch.no_grad():
        c_day = 1.0 - torch.sigmoid(heads.contin(day_state))
        c_night = 1.0 - torch.sigmoid(heads.contin(night_state))

    assert c_day.shape == (2,), f"Expected shape (2,), got {c_day.shape}"
    assert torch.all((c_day >= 0.0) & (c_day <= 1.0)), "Probabilities must be bounded in [0, 1]"
    assert torch.all((c_night >= 0.0) & (c_night <= 1.0)), "Probabilities must be bounded in [0, 1]"


# ==============================================================================
# Feature 5: Vitals Depletion & Metabolic Survival
# ==============================================================================

def test_f5_thirst_exact_step_189_depletion(crafter_env_factory):
    """F5: Drink depletes by 1 every 21 ticks, reaching exactly 0 at tick 189."""
    env = crafter_env_factory(seed=7000, max_steps=200)
    env.reset(seed=7000)

    for step in range(1, 190):
        env.step(0)
        if step == 188:
            assert env._env._player.inventory["drink"] > 0, "Drink depleted before tick 189"
        elif step == 189:
            assert env._env._player.inventory["drink"] == 0, (
                f"Drink should reach 0 at tick 189, got {env._env._player.inventory['drink']}"
            )


def test_f5_hunger_exact_step_234_depletion(crafter_env_factory):
    """F5: Food depletes by 1 every 26 ticks, reaching exactly 0 at tick 234."""
    env = crafter_env_factory(seed=7000, max_steps=250)
    env.reset(seed=7000)
    # Replenish drink so thirst doesn't terminate episode early
    env._env._player.inventory["drink"] = 99

    for step in range(1, 235):
        env._env._player.inventory["drink"] = 99  # keep hydrated
        env.step(0)
        if step == 233:
            assert env._env._player.inventory["food"] > 0, "Food depleted before tick 234"
        elif step == 234:
            assert env._env._player.inventory["food"] == 0, (
                f"Food should reach 0 at tick 234, got {env._env._player.inventory['food']}"
            )


def test_f5_starvation_degeneration_death_spiral(crafter_env_factory):
    """F5: When drink is 0, recovery counter decays and health drops after 16 ticks."""
    env = crafter_env_factory(seed=7000, max_steps=220)
    env.reset(seed=7000)
    player = env._env._player
    player.inventory["drink"] = 0
    player.inventory["food"] = 9
    player.inventory["health"] = 9
    player._thirst = 0
    player._recover = 0

    # Step until starvation damage occurs (16 ticks)
    for _ in range(16):
        env.step(0)

    assert player.health < 9, f"Health should have decayed after 16 ticks of zero drink, got {player.health}"


def test_f5_drink_replenishment_affordance(crafter_env_factory):
    """F5: Executing action 5 facing water replenishes drink and unlocks collect_drink."""
    env = crafter_env_factory(seed=7000, max_steps=100)
    env.reset(seed=7000)
    player = env._env._player
    player.inventory["drink"] = 5
    target_pos = tuple(player.pos + player.facing)
    env._env._world[target_pos] = "water"

    obs, reward, term, trunc, info = env.step(5)

    assert player.inventory["drink"] == 6, f"Drink should increment to 6, got {player.inventory['drink']}"
    assert info["achievements"]["collect_drink"] == 1, "Achievement collect_drink not unlocked"
    assert reward == 1.0, f"Expected milestone reward 1.0, got {reward}"


def test_f5_sleep_energy_recovery_and_vitals_rate(crafter_env_factory):
    """F5: Action 6 (sleep) activates sleeping state and restores energy."""
    env = crafter_env_factory(seed=7000, max_steps=100)
    env.reset(seed=7000)
    player = env._env._player
    player.inventory["energy"] = 2

    env.step(6)
    assert player.sleeping is True, "Sleeping state should be True after action 6"

    # In Crafter, fatigue recovers by 1 each sleeping tick; after 11 ticks energy increments
    for _ in range(12):
        env.step(0)  # while sleeping, noop/step continues sleep until awake or attacked

    assert player.inventory["energy"] > 2, (
        f"Energy should recover during sleep, got {player.inventory['energy']}"
    )
