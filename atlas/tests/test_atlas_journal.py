"""Unit tests for ATLAS game journal logging and inspection tool."""

import json
from pathlib import Path
import pytest
import numpy as np

from atlas.inspect_games import (
    export_best_and_worst,
    export_html_report,
    generate_game_markdown,
    load_journal,
    print_summary,
)


def test_journal_round_trip(tmp_path: Path):
    journal_file = tmp_path / "test_journal.jsonl"
    
    mock_game = {
        "iter": 35,
        "seed": 10001,
        "game_id": "it35_seed10001",
        "length": 150,
        "total_reward": 2.5,
        "died": False,
        "achievements": ["collect_sapling", "place_plant"],
        "num_achievements": 2,
        "overrides_count": 1,
        "steps": [
            {"t": 0, "a": "move_up", "mode": "NAVIGATE", "risk": 0.02, "override": False, "r": 0.0},
            {"t": 1, "a": "do", "mode": "NAVIGATE", "risk": 0.01, "override": False, "r": 1.0, "unlocked": ["collect_sapling"]},
            {"t": 2, "a": "sleep", "mode": "PROBE", "risk": 0.85, "override": True, "r": 0.0},
        ],
    }
    
    with open(journal_file, "w") as f:
        f.write(json.dumps(mock_game) + "\n")
        
    loaded = load_journal(journal_file)
    assert len(loaded) == 1
    assert loaded[0]["game_id"] == "it35_seed10001"
    assert loaded[0]["total_reward"] == 2.5
    assert len(loaded[0]["steps"]) == 3
    
    # Test HTML export
    html_out = tmp_path / "dashboard.html"
    export_html_report(loaded, html_out)
    assert html_out.exists()
    assert "ATLAS Game Journal Dashboard" in html_out.read_text()

    # Test Best & Worst Export
    bw = export_best_and_worst(loaded, tmp_path / "analysis")
    assert bw["best_md"].exists()
    assert bw["worst_md"].exists()
    assert bw["best_json"].exists()
    assert bw["worst_json"].exists()
    assert "Complete Step-by-Step Decision Chronicle" in bw["best_md"].read_text()
