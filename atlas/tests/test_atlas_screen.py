"""Unit tests for screening protocol and JSON schema."""

import json
import tempfile
from pathlib import Path

from atlas.arbiter import AtlasAgent
from atlas.backbone import FrozenBackbone
from atlas.config import AtlasConfig
from atlas.screen import screen_agent


def test_screen_provenance_json_schema():
    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        report = screen_agent(
            agent=agent,
            seeds=[2050, 2051],
            num_episodes=2,
            max_steps=5,
            tag="test_schema",
            out_dir=tmp_dir,
        )

        json_path = Path(tmp_dir) / "atlas_screen_test_schema.json"
        assert json_path.exists(), "Screen JSON report was not written!"

        with open(json_path, "r") as f:
            data = json.load(f)

        # Enforce all required protocol fields
        assert data["protocol_id"] == "strict_visual_atlas_clean_v1"
        assert data["strict_inference"] is True
        assert data["privileged_inventory_used"] is False
        assert data["privileged_achievements_used_for_action"] is False
        assert data["achievements_used_for_scoring_only"] is True

        assert "git_commit" in data
        assert "timestamp" in data
        assert "backbone" in data
        assert "sha256" in data["backbone"] and len(data["backbone"]["sha256"]) == 64
        assert "heads_state_sha256" in data and len(data["heads_state_sha256"]) == 64

        assert data["seeds"] == [2050, 2051]
        assert data["num_episodes"] == 2
        assert "crafter_score" in data
        assert len(data["unlock_percentages"]) == 22

        telemetry = data["per_mode_telemetry"]
        for key in ["probe_fraction", "nav_fraction", "fallback_fraction", "override_rate"]:
            assert key in telemetry, f"Missing telemetry metric: {key}"

        # ATLAS V2 provenance fields
        assert data["bootstrap"] == "none"
        assert data["prior_weight"] == 0.0
        assert data["train_from_scratch"] is True
        assert data["run_line"] == "atlas_v2"
        assert data["resumed_from"] == {"run_id": "atlas_v2_scratch", "iteration": 39}
        assert data["memory_source"] == "own_play_v2"
        assert data["remine"] is True


def test_screen_refuses_nonzero_prior_weight():
    """Screening with prior_weight != 0.0 raises ValueError refusing to write."""
    import pytest

    cfg = AtlasConfig()
    cfg.backbone.prior_weight = 1.0
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(ValueError, match="prior_weight=1.0"):
            screen_agent(
                agent=agent,
                seeds=[2050],
                num_episodes=1,
                max_steps=5,
                tag="test_reject",
                out_dir=tmp_dir,
            )


def test_screen_refuses_missing_provenance():
    """Screening raises ValueError if resumed_from, memory_source, or remine is missing/None."""
    import pytest

    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(ValueError, match="Screening report refuses to write without provenance field 'remine'"):
            screen_agent(
                agent=agent,
                seeds=[2050],
                num_episodes=1,
                max_steps=5,
                tag="test_reject_remine",
                out_dir=tmp_dir,
                remine=None,
            )

