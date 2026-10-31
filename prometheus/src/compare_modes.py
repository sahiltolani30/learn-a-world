"""
Comparative Evaluation and Decision Forensics for JEPA Decision Modes.
Evaluates:
  1. Stochastic Exploration Mode (Actor sampling + curiosity)
  2. Greedy Reflex Mode (Deterministic Twin-Critic argmax)
  3. Deliberative CEM Planning Mode (50-candidate latent MPC lookahead)
Includes recurrent memory analysis and decision explainability.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import crafter

from src.agent.agent import JEPAAgent
from src.memory.storage import CheckpointManager
from src.diagnostics import CRAFTER_ACTION_NAMES


def run_episode(
    agent: JEPAAgent,
    seed: int = 42,
    mode: str = "greedy",
    max_steps: int = 500,
) -> Dict[str, Any]:
    """
    Runs a single game episode under specified decision mode with memory tracking.
    """
    env = crafter.Env(seed=seed)
    obs = env.reset()
    done = False
    
    total_reward = 0.0
    steps = 0
    action_history = []
    memory_history = []
    latencies = []
    unlocked_achievements = set()
    decision_snapshots = []
    
    device = agent.device
    mem_h = torch.zeros(1, 256, device=device)
    
    eval_mode = (mode != "stochastic")
    planning = (mode == "planning")
    
    while not done and steps < max_steps:
        t0 = time.perf_counter()
        
        # Prepare observation and encode
        with torch.no_grad():
            x = agent.validate_and_prepare_obs(obs)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            z = agent.encoder(x)
            readout = agent.get_readout(z, mem_h)
            
            # Select action based on mode
            plan_info = None
            if planning:
                action, plan_info = agent.plan(obs, h=mem_h)
            elif eval_mode:
                q1, q2 = agent.critic(readout)
                q_min = torch.min(q1, q2)
                action = int(torch.argmax(q_min, dim=-1).item())
            else:
                action_t = agent.actor.sample(readout, deterministic=False)
                action = int(action_t.item() if action_t.numel() == 1 else action_t[0].item())
                
            # Snapshot critical decisions for analysis (every 25 steps or when reward/milestone occurs)
            if steps % 30 == 0 or steps in [10, 50, 100]:
                q1, q2 = agent.critic(readout)
                q_vals = torch.min(q1, q2).squeeze(0).cpu().numpy()
                probs = F.softmax(agent.actor(readout), dim=-1).squeeze(0).cpu().numpy()
                
                decision_snapshots.append({
                    "step": steps,
                    "chosen_action": action,
                    "chosen_action_name": CRAFTER_ACTION_NAMES[action],
                    "q_values": q_vals,
                    "actor_probs": probs,
                    "plan_info": plan_info,
                    "mem_norm": float(mem_h.norm().item()),
                })
                
            # Update recurrent memory GRU
            mem_h = agent.gru_step(z, mem_h)
            memory_history.append(mem_h.detach().cpu())
            
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms
        
        # Environment step
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        action_history.append(action)
        
        # Check newly unlocked achievements
        for ach, count in info.get("achievements", {}).items():
            if count > 0:
                unlocked_achievements.add(ach)
                
    return {
        "mode": mode,
        "seed": seed,
        "total_reward": total_reward,
        "steps": steps,
        "achievements": sorted(list(unlocked_achievements)),
        "mean_latency_ms": float(np.mean(latencies)),
        "action_history": action_history,
        "decision_snapshots": decision_snapshots,
        "memory_history": memory_history,
    }


def analyze_memory_dynamics(memory_history: List[torch.Tensor]) -> Dict[str, Any]:
    """
    Analyzes what is happening inside the recurrent memory GRU over the episode.
    """
    if not memory_history:
        return {}
        
    stacked_h = torch.cat(memory_history, dim=0)  # (T, 256)
    norms = stacked_h.norm(dim=1).numpy()
    
    # Cosine similarities between consecutive memory states
    cos_sims = []
    for t in range(len(memory_history) - 1):
        h1 = memory_history[t]
        h2 = memory_history[t + 1]
        sim = F.cosine_similarity(h1, h2, dim=-1).item()
        cos_sims.append(sim)
        
    # Active dimension count (dimensions with variance > 0.01 across episode)
    dim_vars = stacked_h.var(dim=0).numpy()
    active_dims = int(np.sum(dim_vars > 0.01))
    
    return {
        "initial_norm": float(norms[0]),
        "final_norm": float(norms[-1]),
        "mean_norm": float(np.mean(norms)),
        "mean_consecutive_cosine": float(np.mean(cos_sims)) if cos_sims else 1.0,
        "active_memory_dims": active_dims,
    }


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Print clean summary table comparing all 3 modes."""
    print("\n" + "=" * 90)
    print(" 🏆 TRIPLE-MODE PERFORMANCE BENCHMARK (Exact Same Environment Seed = 42)")
    print("=" * 90)
    print(f"{'Decision Mode':<28} | {'Return':<8} | {'Length':<8} | {'Latency':<12} | {'Achievements Unlocked'}")
    print("-" * 90)
    for res in results:
        achs = ", ".join(res["achievements"]) if res["achievements"] else "None"
        print(f"{res['mode']:<28} | {res['total_reward']:<+8.2f} | {res['steps']:<8} | {res['mean_latency_ms']:<6.2f} ms/step | {achs}")
    print("=" * 90)


def print_decision_forensics(res: Dict[str, Any]) -> None:
    """Prints deep breakdown of why the model took specific decisions."""
    mode = res["mode"]
    snapshots = res.get("decision_snapshots", [])
    if not snapshots:
        return
        
    print(f"\n🔎 DECISION FORENSICS & WHY THE MODEL CHOSE WHAT IT CHOSE ({mode.upper()} MODE):")
    print("-" * 90)
    
    # Pick a representative snapshot (e.g. step 30)
    snap = snapshots[min(1, len(snapshots) - 1)]
    step = snap["step"]
    chosen = snap["chosen_action_name"]
    q_vals = snap["q_values"]
    probs = snap["actor_probs"]
    plan_info = snap["plan_info"]
    
    print(f"Frame at Step {step}: Chosen Action -> '{chosen}' (Action {snap['chosen_action']})")
    print(f"Memory Magnitude ||h||: {snap['mem_norm']:.3f} (Preserves temporal context)")
    
    # Top 3 actions according to Actor
    top3_actor = np.argsort(probs)[::-1][:3]
    print("\n  [1] Reflex Actor Prior Policy \u03c0(a | z, h) - Reflex Preference:")
    for rank, idx in enumerate(top3_actor):
        print(f"      Rank {rank+1}: {CRAFTER_ACTION_NAMES[idx]:<18} | Probability: {probs[idx]*100:5.1f}%")
        
    # Top 3 actions according to Twin Critic
    top3_critic = np.argsort(q_vals)[::-1][:3]
    print("\n  [2] Twin Critic Value Q(z, h, a) - Expected Long-Term Payoff:")
    for rank, idx in enumerate(top3_critic):
        print(f"      Rank {rank+1}: {CRAFTER_ACTION_NAMES[idx]:<18} | Q-Value: {q_vals[idx]:+7.3f}")
        
    # If Grandmaster Planning Mode, show the 5-step lookahead trajectory
    if plan_info is not None:
        traj = plan_info.get("best_trajectory", [])
        traj_names = [CRAFTER_ACTION_NAMES[a] for a in traj] if traj else []
        print("\n  [3] Deliberative CEM Planner (50 Candidate Futures Lookahead):")
        print(f"      Imagined 5-Step Trajectory: {' -> '.join(traj_names)}")
        print(f"      Projected Latent Return:   {plan_info.get('best_return', 0.0):+.4f}")
        print(f"      Planning Compute Time:     {plan_info.get('planning_latency_ms', 0.0):.2f} ms")
    print("-" * 90)


def print_memory_deep_dive(results: List[Dict[str, Any]]) -> None:
    """Print analysis of recurrent memory dynamics across the runs."""
    print("\n🧠 RECURRENT MEMORY ANALYSIS (What is Stored in h_t \u2208 \u211d\u00b2\u2075\u2076):")
    print("=" * 90)
    print("  \u2022 Role of h_t: Integrates past latent observations z_{0..t} via GRUCell(z_t, h_{t-1}).")
    print("  \u2022 Anti-Amnesia: Retains off-screen landmark traces (e.g. table location, water ponds).")
    print("  \u2022 Readout Fusion: Readout = LayerNorm(Linear([z_t, h_t])) combines vision with memory.")
    print("-" * 90)
    for res in results:
        mem_stats = analyze_memory_dynamics(res["memory_history"])
        if mem_stats:
            print(f"  [{res['mode'].upper()}] Active Memory Dims: {mem_stats['active_memory_dims']}/256 | Mean Norm: {mem_stats['mean_norm']:.2f} | Smoothness (CosSim): {mem_stats['mean_consecutive_cosine']:.4f}")
    print("=" * 90)
