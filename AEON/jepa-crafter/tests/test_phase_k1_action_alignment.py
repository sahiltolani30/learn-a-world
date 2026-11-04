"""Unit test proving and verifying the action-indexing alignment in Phase K."""
import pytest
import numpy as np


def test_action_alignment_logic():
    """Verify that action taken from obs[i] to obs[i+1] is act[i], not act[i+1]."""
    # Simulate an environment trajectory of 10 steps
    # obs_list[0] = initial reset observation O_0
    # step 0: env.step(a0) -> obs_list[1] = O_1
    # step 1: env.step(a1) -> obs_list[2] = O_2
    # step 2: env.step(a2) -> obs_list[3] = O_3
    # step 3: env.step(a3) -> obs_list[4] = O_4
    # step 4: env.step(a4) -> obs_list[5] = O_5
    
    obs_list = [f"O_{t}" for t in range(15)]
    act_list = [f"a_{t}" for t in range(14)]
    
    context_length = 4
    max_horizon = 8
    
    # Suppose current context window ends at index i = 3:
    # Context observations are [O_0, O_1, O_2, O_3]
    # Current state z_t is O_3
    i = 3
    ctx_obs = obs_list[i - context_length + 1 : i + 1]
    assert ctx_obs[-1] == "O_3", "z_t must be O_3"
    
    # The next observation at horizon 1 is O_4:
    target_h1 = obs_list[i + 1]
    assert target_h1 == "O_4", "Target at horizon 1 must be O_4"
    
    # The action that transitioned O_3 -> O_4 was step 3 action: a_3 (which is act_list[3])
    correct_transition_action = act_list[i]
    assert correct_transition_action == "a_3"
    
    # Buggy indexing in Phase K:
    buggy_fut_act = act_list[i + 1 : i + 1 + max_horizon]
    assert buggy_fut_act[0] == "a_4", "Bug: fut_act[0] was a_4 instead of a_3!"
    
    # Correct indexing:
    correct_fut_act = act_list[i : i + max_horizon]
    assert correct_fut_act[0] == "a_3", "Correct: fut_act[0] is a_3!"
    assert correct_fut_act[1] == "a_4", "Correct: fut_act[1] is a_4!"
    assert len(correct_fut_act) == max_horizon
