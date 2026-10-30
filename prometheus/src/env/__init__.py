"""
src/env

Zero-Cheat Environment Abstraction & Wrappers for Crafter and Atari ALE.
Authoritative Specification: PROJECT.md, ORIGINAL_REQUEST.md (R3, R4)
"""

from src.env.wrapper import (
    ZeroCheatWrapper,
    FrameStackWrapper,
    make_crafter_env,
    ResetObs,
    FORBIDDEN_CHEAT_KEYS,
)
from src.env.atari_wrapper import (
    AtariEnvWrapper,
    make_atari_env,
    resolve_atari_game_id,
    MockAtariEnv,
)

__all__ = [
    "ZeroCheatWrapper",
    "FrameStackWrapper",
    "make_crafter_env",
    "AtariEnvWrapper",
    "make_atari_env",
    "ResetObs",
    "FORBIDDEN_CHEAT_KEYS",
    "resolve_atari_game_id",
    "MockAtariEnv",
]
