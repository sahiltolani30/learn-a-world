"""
Automated unit tests for Phase 0 gate requirements.
"""

import numpy as np
import pytest

def test_imports():
    import torch
    import gymnasium
    import crafter
    import stable_baselines3
    import huggingface_hub
    try:
        import gradio
    except ImportError:
        pass
    
    assert torch.__version__ is not None
    assert gymnasium.__version__ is not None
    assert hasattr(crafter, 'Env')
    assert stable_baselines3.__version__ is not None

def test_torch_compute():
    import torch
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.ones((4, 4), device=device)
    res = t + t
    assert res.sum().item() == 32.0

def test_crafter_environment_rollout():
    import crafter
    env = crafter.Env(seed=123)
    obs = env.reset()
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.uint8
    assert env.action_space.n == 17
    
    for _ in range(10):
        action = np.random.randint(0, env.action_space.n)
        next_obs, reward, done, info = env.step(action)
        assert next_obs.shape == (64, 64, 3)
        assert next_obs.dtype == np.uint8
        assert isinstance(info, dict)
        if done:
            break

def test_crafter_gymnasium_wrapper():
    from envs.crafter_wrapper import CrafterEnv
    env = CrafterEnv(seed=42)
    obs, info = env.reset()
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.uint8
    assert env.action_space.n == 17
    
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == (64, 64, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

def test_utils_device_and_config():
    from experiments.utils import get_device, get_output_dir, load_config, set_seed
    device = get_device()
    assert device.type in ["mps", "cuda", "cpu"]
    
    cfg = load_config("experiments/configs/default.yaml")
    assert cfg["project"] == "learn-a-world"
    assert "model" in cfg
    
    out_dir = get_output_dir("./test_out")
    assert out_dir.exists()
    import shutil
    shutil.rmtree(out_dir)
    
    set_seed(123)
