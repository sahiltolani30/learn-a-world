# Project AEON: Architecture & Interface Decisions

## Decision 001: Strict Gated Phase Execution (Plan v2.0)
- **Context:** Previous iterations exhibited fragmentation across parallel phases.
- **Decision:** Follow the canonical 15-phase dependency order (Phases 0–14) defined in Plan v2.0. No downstream phase is unlocked until its prerequisite gate passes.
- **Date:** 2026-11-03

## Decision 002: Lockstep Vector Collection with Episode Boundary Preservation
- **Context:** Crafter step latency on CPU is ~15ms, making single-worker collection slow for 100k+ step regimes.
- **Decision:** Implemented `VectorCrafterAdapter` (N=8) and `BatchedExperienceCollector`. All workers share the same opaque permutation. Completed trajectories are committed as contiguous sequences via `add_episode_trajectory`.
- **Date:** 2026-11-03

## Decision 003: FiLM + Transformer Spatial Latent Predictor
- **Context:** NVIDIA T4 GPU memory limit (15 GB) on Google Colab requires lean parameter footprint and minimal intermediate activation caching.
- **Decision:** Adopted 2 Transformer blocks per ensemble member with FiLM conditioning on 64 spatial tokens ($8\times 8$ grid of 128-d). Total trainable parameters: 3,346,149. Fits cleanly in ~2.4 GB VRAM under mixed precision.
- **Date:** 2026-11-03

## Decision 004: Atomic Drive Checkpoint Persistence
- **Context:** Colab instances are subject to timeouts and preemption.
- **Decision:** `CheckpointStore.save_atomic` writes to a temporary file (`.tmp`) and renames atomically to prevent corrupted snapshots. Checkpoints save every 10,000 steps directly to `/content/drive/MyDrive/learn-a-world/aeon/`.
- **Date:** 2026-11-03
