"""Tests verifying that CEM planner executes only first action and replans every step."""
import pytest
import torch
from src.models.jepa_world_model import JepaWorldModel
from src.planning.cem_planner import CemPlanner
from src.planning.candidate_scorer import CandidateScorer
from src.exploration.role_definitions import ROLE_WEIGHTS


def test_planner_replans_first_action_only():
    model = JepaWorldModel()
    model.eval()

    planner = CemPlanner(
        model=model,
        horizon=4,
        population_size=16,
        num_iterations=2,
        num_elites=4,
    )

    latent = torch.randn(1, 64, 128)
    context = torch.randn(1, 256)

    # Step 1: plan first action
    action_1 = planner.plan_first_action(latent, context, ROLE_WEIGHTS["balanced_planner"])
    assert isinstance(action_1, int)
    assert 0 <= action_1 < 17
    assert len(planner.last_plan_info["best_sequence"]) == 4
    assert planner.last_plan_info["best_sequence"][0] == action_1

    # Step 2: simulate receiving new real environment state and replanning from scratch
    new_latent = torch.randn(1, 64, 128)
    new_context = torch.randn(1, 256)
    action_2 = planner.plan_first_action(new_latent, new_context, ROLE_WEIGHTS["balanced_planner"])
    assert isinstance(action_2, int)
    assert 0 <= action_2 < 17


def test_random_shooting_fallback():
    model = JepaWorldModel()
    model.eval()

    planner = CemPlanner(
        model=model,
        horizon=4,
        population_size=16,
        algorithm="random_shooting",
    )

    latent = torch.randn(1, 64, 128)
    context = torch.randn(1, 256)
    action = planner.plan_first_action(latent, context)
    assert isinstance(action, int)
    assert 0 <= action < 17
