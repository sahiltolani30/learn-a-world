"""Tests for opaque action permutation and mapping."""
import pytest
from src.env.opaque_actions import OpaqueActionMapping


def test_permutation_properties():
    mapping1 = OpaqueActionMapping(num_actions=17, seed=42)
    mapping2 = OpaqueActionMapping(num_actions=17, seed=42)
    mapping3 = OpaqueActionMapping(num_actions=17, seed=99)
    
    assert sorted(mapping1.permutation) == list(range(17))
    assert mapping1.permutation == mapping2.permutation
    assert mapping1.permutation != mapping3.permutation
    
    # Bijective check
    for token in range(17):
        env_act = mapping1.token_to_env_action(token)
        assert mapping1.env_action_to_token(env_act) == token


def test_serialization_roundtrip():
    mapping = OpaqueActionMapping(num_actions=17, seed=123)
    data = mapping.to_dict()
    restored = OpaqueActionMapping.from_dict(data)
    
    assert mapping.permutation == restored.permutation
    for token in range(17):
        assert mapping.token_to_env_action(token) == restored.token_to_env_action(token)
