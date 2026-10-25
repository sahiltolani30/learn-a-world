"""
tests/test_checkpoint_loader.py
Unit tests for modernized JEPAv3 checkpoint loader and parameter verification.
"""

import pytest
import torch
from pathlib import Path
from models.jepa_v3 import JEPAv3

CHECKPOINT_PATH = Path("checkpoints/jepa_crafter_v3_phase2.pt")

def test_v3_checkpoint_loading_and_integrity():
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")

    model = JEPAv3()
    res = model.load_checkpoint(str(CHECKPOINT_PATH), device="cpu")
    
    assert "encoder" in res["loaded_components"]
    assert "event_head" in res["loaded_components"]
    assert "q_ensemble" in res["loaded_components"]
    assert "policy_prior" in res["loaded_components"]
    
    audit = res["audit"]
    assert audit["all_healthy"] is True
    assert audit["encoder"]["l2_norm"] > 1.0
    assert audit["event_head"]["l2_norm"] > 1.0
    assert audit["q_ensemble"]["l2_norm"] > 1.0

def test_checkpoint_integrity_audit():
    model = JEPAv3()
    audit = model.verify_integrity()
    assert "encoder" in audit
    assert "event_head" in audit
    assert "q_ensemble" in audit
    assert audit["all_healthy"] is True
