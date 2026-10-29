"""Project ATLAS: Associative Topological LATent Skills."""

import importlib.util
import sys
from pathlib import Path

# Safe pre-registration of eval.metrics to avoid pulling unused stable_baselines3 from eval/__init__.py
if "eval.metrics" not in sys.modules:
    _metrics_p = Path(__file__).resolve().parent.parent / "eval" / "metrics.py"
    if _metrics_p.exists():
        _spec = importlib.util.spec_from_file_location("eval.metrics", str(_metrics_p))
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules["eval.metrics"] = _mod
            _spec.loader.exec_module(_mod)

from atlas.arbiter import AtlasAgent
from atlas.backbone import FrozenBackbone
from atlas.config import AtlasConfig

__all__ = ["AtlasConfig", "FrozenBackbone", "AtlasAgent"]
