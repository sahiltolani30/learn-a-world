"""Unit tests for V3 F4: Confidence-gated exploration focus in blind contexts."""

import numpy as np
import pytest
import torch

from atlas.arbiter import AtlasAgent
from atlas.backbone import FrozenBackbone
from atlas.config import AtlasConfig, NUM_ACTIONS


class StubEpistemic(torch.nn.Module):
    def __init__(self, spike_action: int, spike_value: float = 5.0, base_value: float = 0.1):
        super().__init__()
        self.spike_action = spike_action
        self.spike_value = spike_value
        self.base_value = base_value

    def disagreement_all_actions(self, s):
        # returns [B, NUM_ACTIONS]
        B = s.shape[0]
        d = torch.full((B, NUM_ACTIONS), self.base_value, device=s.device)
        d[:, self.spike_action] = self.spike_value
        return d

    def disagreement(self, s, a):
        d_all = self.disagreement_all_actions(s)
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        return d_all.gather(1, a)


def test_blind_probe_epistemic_disagreement_ranking():
    """V3 F4 verification:

    Zero event probs (blind context) + high disagreement flips probe ranking
    toward the max-disagreement action.
    """
    cfg = AtlasConfig()
    cfg.goals.blind_probe_boost = 1.0
    cfg.goals.probe_entropy = 0.5
    cfg.safety.risk_threshold = 10.0  # disable safety override for pure test

    fb = FrozenBackbone(config=cfg, device="cpu")

    agent = AtlasAgent(backbone=fb, cfg=cfg)
    agent.epsilon = 0.0  # no random exploration for deterministic test

    # Hook stub epistemic net with spike on action 7
    SPIKE_ACTION = 7
    agent.epistemic_net = StubEpistemic(spike_action=SPIKE_ACTION, spike_value=10.0, base_value=0.1)

    # Put agent directly into PROBE mode
    agent.mode = "PROBE"
    agent.probe_steps_remaining = 3

    obs = np.zeros((64, 64, 3), dtype=np.uint8)

    # Case 1: Blind context (all event probs near 0)
    # Action chosen should be SPIKE_ACTION because curiosity dominates
    act, telem = agent.act(obs, reward_prev=0.0, step=0)
    assert act == SPIKE_ACTION
    assert agent.last_probe_decision is not None
    scores = agent.last_probe_decision["blended_scores"]
    assert np.argmax(scores) == SPIKE_ACTION

    # Case 2: Affordance present (affordance >= 0.05 on action 3)
    # When affordance is dominant (e.g. 10.0), it should outrank curiosity
    def mock_event_logits_high_affordance(s, mask):
        probs = torch.zeros((1, NUM_ACTIONS, 22), device=s.device)
        probs[:, 3, :] = 5.0  # action 3 has massive affordance
        return probs

    agent.backbone.event_logits_all_actions = mock_event_logits_high_affordance
    agent.mode = "PROBE"
    agent.probe_steps_remaining = 3
    act2, _ = agent.act(obs, reward_prev=0.0, step=1)
    assert act2 == 3
