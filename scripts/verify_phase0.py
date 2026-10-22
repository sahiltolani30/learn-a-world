#!/usr/bin/env python3
"""
Phase 0 Verification Script for Learn-a-World.
Checks toolchain, library imports, Crafter environment execution, and device capabilities.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np

def test_imports():
    print("--> 1. Testing Core Imports...")
    import torch
    import gymnasium as gym
    import crafter
    import stable_baselines3
    import huggingface_hub
    try:
        import gradio
    except ImportError:
        pass
    
    import importlib.metadata
    crafter_version = importlib.metadata.version('crafter')
    print(f"    PyTorch version: {torch.__version__}")
    print(f"    Gymnasium version: {gym.__version__}")
    print(f"    Crafter version: {crafter_version}")
    print(f"    Stable-Baselines3 version: {stable_baselines3.__version__}")
    print("    [PASS] All core libraries imported successfully!")
    return torch

def test_device(torch):
    print("\n--> 2. Testing Compute Acceleration (MPS / CUDA / CPU)...")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("    [PASS] Apple Silicon MPS (Metal Performance Shaders) is AVAILABLE!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"    [PASS] NVIDIA CUDA is AVAILABLE! ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("    [INFO] Running on CPU.")
    
    # Run a simple tensor op on selected device
    x = torch.randn(10, 10, device=device)
    y = x @ x
    assert y.shape == (10, 10)
    print(f"    [PASS] Tensor matrix multiplication verified on device '{device}'.")
    return device

def test_crafter_env():
    print("\n--> 3. Testing Crafter Game Environment...")
    import crafter
    
    env = crafter.Env(seed=42)
    obs = env.reset()
    
    assert isinstance(obs, np.ndarray), f"Expected numpy array obs, got {type(obs)}"
    assert obs.shape == (64, 64, 3), f"Expected shape (64, 64, 3), got {obs.shape}"
    assert obs.dtype == np.uint8, f"Expected dtype uint8, got {obs.dtype}"
    assert env.action_space.n == 17, f"Expected 17 discrete actions in Crafter, got {env.action_space.n}"
    
    print(f"    Initial observation shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"    Action space: {env.action_space}")
    
    for step in range(10):
        action = np.random.randint(0, env.action_space.n)
        next_obs, reward, done, info = env.step(action)
        assert next_obs.shape == (64, 64, 3), f"Step {step} bad obs shape: {next_obs.shape}"
        assert isinstance(reward, (int, float)), f"Bad reward type: {type(reward)}"
    
    print(f"    [PASS] Successfully executed 10 Crafter steps with valid 64x64x3 frames!")
    return True

def main():
    print("==================================================")
    print("       Learn-a-World: Phase 0 Verification       ")
    print("==================================================")
    try:
        torch_mod = test_imports()
        test_device(torch_mod)
        test_crafter_env()
        print("\n==================================================")
        print("   ALL PHASE 0 GATE TESTS PASSED SUCCESSFULLY!    ")
        print("==================================================")
    except Exception as e:
        print(f"\n[FAIL] Phase 0 Verification failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
