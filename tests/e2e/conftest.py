"""Zephyr E2E Shared Fixtures.

Provides deterministic test environments, mock Google Drive checkpoint workspaces,
synthetic 64x64x3 RGB observation synthesizers, configuration fixtures, and AST
legacy purity validation.
"""
from __future__ import annotations

import gc
import os
import pathlib
import random
import shutil
import sys
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import pytest
import torch

from envs.crafter_wrapper import CrafterEnv, patch_crafter_determinism
from zephyr.buffer import EpisodeRecord, ZephyrReplay
from zephyr.config import ZephyrConfig
from zephyr.heads import ZephyrHeads
from zephyr.vec_env import ZephyrVecEnv
from zephyr.wm import ZephyrWM
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tests"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zephyr_test_utils import tiny_cfg


# Ensure Crafter creature iteration determinism is patched across all tests
patch_crafter_determinism()


# ==============================================================================
# 1. Determinism & Seed Management
# ==============================================================================

@pytest.fixture
def deterministic_seed() -> int:
    """Sets random seeds across Python, NumPy, and PyTorch for strict determinism."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


# ==============================================================================
# 2. Environments
# ==============================================================================

@pytest.fixture
def crafter_env_factory() -> Generator[Callable[..., CrafterEnv], None, None]:
    """Factory creating isolated, deterministic single CrafterEnv instances."""
    created_envs: List[CrafterEnv] = []

    def _make(seed: int = 7000, max_steps: int = 1000) -> CrafterEnv:
        env = CrafterEnv(seed=seed, max_steps=max_steps)
        created_envs.append(env)
        return env

    yield _make

    for env in created_envs:
        try:
            env.close()
        except Exception:
            pass


@pytest.fixture
def fixed_crafter_env(crafter_env_factory) -> Generator[CrafterEnv, None, None]:
    """Pre-seeded deterministic CrafterEnv instance with seed 7000."""
    env = crafter_env_factory(seed=7000, max_steps=500)
    obs, info = env.reset(seed=7000)
    yield env


@pytest.fixture
def vec_env_factory() -> Generator[Callable[..., ZephyrVecEnv], None, None]:
    """Factory creating deterministic ZephyrVecEnv instances with in_process=True."""
    created_vecs: List[ZephyrVecEnv] = []

    def _make(num_envs: int = 2, base_seed: int = 8000, max_steps: int = 120) -> ZephyrVecEnv:
        venv = ZephyrVecEnv(
            num_envs=num_envs,
            base_seed=base_seed,
            max_steps=max_steps,
            in_process=True,
        )
        created_vecs.append(venv)
        return venv

    yield _make

    for venv in created_vecs:
        try:
            venv.close()
        except Exception:
            pass


# ==============================================================================
# 3. Mock Checkpoints & Serialization Workspaces
# ==============================================================================

@pytest.fixture
def mock_drive_workspace(tmp_path: pathlib.Path) -> Dict[str, pathlib.Path]:
    """Creates a sandboxed mock Google Drive directory structure."""
    drive_root = tmp_path / "mock_drive" / "MyDrive" / "crafter_jepa_checkpoints"
    shards_dir = drive_root / "shards"
    reports_dir = drive_root / "reports"

    shards_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root": drive_root,
        "shards": shards_dir,
        "reports": reports_dir,
        "state_pt": drive_root / "state.pt",
        "state_tmp": drive_root / "state.pt.tmp",
        "wm_pt": drive_root / "zephyr_wm_latest.pt",
        "heads_pt": drive_root / "zephyr_heads_latest.pt",
    }


@pytest.fixture
def mock_checkpoint_factory(mock_drive_workspace, tiny_test_cfg):
    """Factory producing valid small-dimension checkpoints on demand."""
    def _create(
        wm: Optional[ZephyrWM] = None,
        heads: Optional[ZephyrHeads] = None,
        step: int = 10,
        generation: int = 1,
    ) -> Dict[str, pathlib.Path]:
        cfg = tiny_test_cfg
        _wm = wm or ZephyrWM(cfg)
        _heads = heads or ZephyrHeads(cfg)

        wm_path = mock_drive_workspace["wm_pt"]
        heads_path = mock_drive_workspace["heads_pt"]
        state_path = mock_drive_workspace["state_pt"]

        _wm.save_checkpoint(str(wm_path), step=step)
        _heads.save_checkpoint(str(heads_path), step=step)

        # Full trainer state archive
        state_payload = {
            "format": "zephyr_run_v1",
            "world_model": _wm.state_dict(),
            "heads": _heads.state_dict(),
            "slow_critic": _heads.critic.state_dict(),
            "opt_states": {},
            "rng": {
                "torch": torch.get_rng_state(),
                "torch_cuda": None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "manifest": {
                "step": step,
                "generation": generation,
                "collector_generation": generation,
                "episodes": 5,
                "achievements": {"collect_wood": 3, "place_table": 1},
            },
        }
        torch.save(state_payload, str(state_path))

        return {
            "wm": wm_path,
            "heads": heads_path,
            "state": state_path,
        }

    return _create


@pytest.fixture
def corrupted_checkpoint_factory(mock_drive_workspace):
    """Factory producing corrupted checkpoints for boundary testing."""
    def _create(corruption_type: str = "truncated") -> pathlib.Path:
        target = mock_drive_workspace["root"] / f"corrupt_{corruption_type}.pt"
        if corruption_type == "truncated":
            with open(target, "wb") as f:
                f.write(b"PK\x03\x04" + b"\x00" * 40)  # Truncated zip
        elif corruption_type == "invalid_pickle":
            with open(target, "w") as f:
                f.write("NOT_A_VALID_PYTORCH_ARCHIVE")
        elif corruption_type == "missing_keys":
            torch.save({"unrelated_key": 123}, str(target))
        elif corruption_type == "shape_mismatch":
            # Incompatible weights tensor
            torch.save({"world_model": {"encoder.proj.weight": torch.randn(128, 4096)}}, str(target))
        return target

    return _create


# ==============================================================================
# 4. Synthetic Observation Generators (Strictly 64x64x3 RGB)
# ==============================================================================

@pytest.fixture(scope="session")
def synthetic_obs_generator() -> Callable[..., np.ndarray]:
    """Generates valid 64x64x3 RGB uint8 observations with customizable patterns."""
    def _make(
        pattern: str = "random",
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        seed: int = 42,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)

        def _generate_frame() -> np.ndarray:
            if pattern == "random":
                frame = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            elif pattern == "black":
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
            elif pattern == "white":
                frame = np.full((64, 64, 3), 255, dtype=np.uint8)
            elif pattern == "daylight":
                frame = rng.integers(120, 220, size=(64, 64, 3), dtype=np.uint8)
            elif pattern == "nightfall":
                frame = rng.integers(0, 10, size=(64, 64, 3), dtype=np.uint8)
            elif pattern == "station_table":
                frame = np.full((64, 64, 3), 60, dtype=np.uint8)
                frame[24:40, 24:40] = np.array([139, 69, 19], dtype=np.uint8)  # Wood table
            elif pattern == "tree":
                frame = np.full((64, 64, 3), 80, dtype=np.uint8)
                frame[20:44, 28:36] = np.array([100, 70, 30], dtype=np.uint8)  # Trunk
                frame[12:28, 20:44] = np.array([30, 140, 30], dtype=np.uint8)  # Leaves
            else:
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
            return np.ascontiguousarray(frame, dtype=np.uint8)

        if batch_size is None and seq_len is None:
            return _generate_frame()
        elif batch_size is not None and seq_len is None:
            return np.stack([_generate_frame() for _ in range(batch_size)])
        elif batch_size is not None and seq_len is not None:
            return np.stack([
                np.stack([_generate_frame() for _ in range(seq_len)])
                for _ in range(batch_size)
            ])
        else:
            raise ValueError("Invalid batch/seq dimension combination")

    return _make


@pytest.fixture(scope="session")
def synthetic_batch_generator(synthetic_obs_generator) -> Callable[..., Dict[str, torch.Tensor]]:
    """Generates synthetic replay-like batches conforming to buffer contracts."""
    def _make(batch_size: int = 4, seq_len: int = 8, seed: int = 42) -> Dict[str, torch.Tensor]:
        obs = synthetic_obs_generator("random", batch_size=batch_size, seq_len=seq_len, seed=seed)
        obs_next = synthetic_obs_generator("random", batch_size=batch_size, seq_len=seq_len, seed=seed + 1)
        rng = np.random.default_rng(seed)

        return {
            "obs": torch.from_numpy(obs),
            "obs_next": torch.from_numpy(obs_next),
            "actions": torch.from_numpy(rng.integers(0, 17, size=(batch_size, seq_len))),
            "rewards": torch.from_numpy(rng.uniform(0.0, 1.0, size=(batch_size, seq_len)).astype(np.float32)),
            "rewards_int": torch.zeros((batch_size, seq_len), dtype=torch.float32),
            "done": torch.zeros((batch_size, seq_len), dtype=torch.bool),
            "valid": torch.ones((batch_size, seq_len), dtype=torch.bool),
            "inventory": torch.zeros((batch_size, seq_len, 16), dtype=torch.float32),
            "inventory_next": torch.zeros((batch_size, seq_len, 16), dtype=torch.float32),
            "achievements_new": torch.zeros((batch_size, seq_len, 22), dtype=torch.bool),
            "achievements_cum": torch.zeros((batch_size, seq_len, 22), dtype=torch.bool),
        }

    return _make


# ==============================================================================
# 5. Configurations & Models
# ==============================================================================

@pytest.fixture
def device() -> str:
    return "cpu"


@pytest.fixture
def tiny_test_cfg() -> ZephyrConfig:
    """Fast CPU configuration for instant unit and tier execution."""
    return tiny_cfg()


@pytest.fixture(scope="session")
def smoke_cfg() -> ZephyrConfig:
    """Lightweight test configuration with small dimensions and batch sizes."""
    return tiny_cfg()


@pytest.fixture
def standard_test_cfg() -> ZephyrConfig:
    """Full-scale benchmark ZephyrConfig for architecture contract testing."""
    return ZephyrConfig()


@pytest.fixture
def mock_wm(tiny_test_cfg) -> ZephyrWM:
    """Pre-instantiated ZephyrWM instance on CPU."""
    return ZephyrWM(tiny_test_cfg).to("cpu")


@pytest.fixture
def dummy_wm(smoke_cfg, device) -> ZephyrWM:
    return ZephyrWM(smoke_cfg).to(device).eval()


@pytest.fixture
def mock_heads(tiny_test_cfg) -> ZephyrHeads:
    """Pre-instantiated ZephyrHeads instance on CPU."""
    return ZephyrHeads(tiny_test_cfg).to("cpu")


@pytest.fixture
def dummy_heads(smoke_cfg, device) -> ZephyrHeads:
    return ZephyrHeads(smoke_cfg).to(device).eval()


@pytest.fixture
def sample_obs_batch() -> np.ndarray:
    """Generates uint8 [4, 64, 64, 3] random frames."""
    return np.random.randint(0, 256, (4, 64, 64, 3), dtype=np.uint8)


@pytest.fixture
def populated_replay() -> ZephyrReplay:
    """Replay buffer pre-populated with 3 dummy episodes for sampling tests."""
    replay = ZephyrReplay(capacity_frames=1000)
    for ep in range(3):
        rec = EpisodeRecord(
            obs=np.zeros((20, 64, 64, 3), dtype=np.uint8),
            actions=np.zeros(20, dtype=np.int16),
            rewards=np.zeros(20, dtype=np.float32),
            rewards_int=np.zeros(20, dtype=np.float32),
            done=np.zeros(20, dtype=bool),
            inventory=np.zeros((20, 16), dtype=np.float32),
            achievements_cum=np.zeros((20, 22), dtype=bool),
            achievements_new=np.zeros((20, 22), dtype=bool),
            seed=1000 + ep,
            source="test",
        )
        rec.done[-1] = True
        replay.add_episode(rec)
    return replay


# ==============================================================================
# 6. Sandboxing & Legacy Code Isolation Validator
# ==============================================================================

@pytest.fixture(autouse=True)
def legacy_purity_validator():
    """Validates that no legacy repository code is loaded or imported."""
    forbidden = ("models.", "planning.", "training.")
    yield
    violations = [
        mod for mod in sys.modules.keys()
        if any(mod.startswith(p) for p in forbidden)
    ]
    assert not violations, f"Legacy code isolation violation: imported {violations}"


@pytest.fixture
def memory_guard():
    """Asserts that no memory leaks occur during test execution."""
    gc.collect()
    mem_before = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()

    yield

    gc.collect()
    if torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated()
        assert mem_after <= mem_before + 1024 * 1024, "GPU memory leak detected"
