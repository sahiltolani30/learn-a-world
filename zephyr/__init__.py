"""Project Zephyr — true-JEPA actor-critic world model agent.

A self-contained agent package. It must never import `models.*`, `planning.*`, or
`training.*` (the previous agent line); enforced by tests/test_zephyr_isolation.py.
Read-only whitelist: `envs` (deterministic Crafter wrapper) and `eval.metrics`
(official score function). See docs/ZEPHYR_AGENT_PLAN.md.
"""
from __future__ import annotations

__version__ = "0.1.0"

from zephyr.config import ZephyrConfig  # noqa: F401
