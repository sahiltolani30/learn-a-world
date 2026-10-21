# Learn-a-World

**A JEPA-style latent world model that learns by playing.**

An agent plays simulated worlds (a custom maze → Push-T → the game Crafter). Every step it takes is stored in a capped replay buffer. A world model (encoder + action-conditioned latent predictor, custom-trained in PyTorch — no LLM APIs anywhere) learns the world's rules from that self-generated experience. A planner (MPC) dreams candidate action sequences through the model and acts on the best one — so play improves, generating richer experience: Play → Remember → Learn → Imagine → Play better.

Headline experiment: when the world's dynamics change at test time, the model keeps learning from its own prediction errors (test-time adaptation) and recovers — while a frozen model and a PPO baseline degrade.

## Status
Blueprint stage. Full design document: docs/BLUEPRINT.md
Build stages: custom maze → planner + PPO baseline → Push-T → Crafter → adaptation → public release (model on Hugging Face + live demo).

## Stack
Python · PyTorch · Gymnasium · Crafter · stable-baselines3 (PPO baseline) · Hugging Face

## Why
World models are where the frontier funding went in 2026 (AMI Labs, World Labs, DeepMind Genie 3). This project builds the same science at open, reproducible scale — custom-trained, honestly measured, fully public.
