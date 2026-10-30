"""Replay / Inspect CLI for Project ATLAS decision traces (T4).

Pure log renderer. Reads ONLY trace/*.jsonl + game_journal.jsonl + optional audit/probes/*.
No env, no nets, no torch.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Contractual Crafter action names for display-only rendering
CRAFTER_ACTION_NAMES = [
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "sleep", "place_stone", "place_table", "place_furnace",
    "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
    "make_iron_pickaxe", "make_wood_sword", "make_stone_sword", "make_iron_sword",
]


def _get_action_name(idx: int) -> str:
    if 0 <= idx < len(CRAFTER_ACTION_NAMES):
        return CRAFTER_ACTION_NAMES[idx]
    return str(idx)


def _resolve_trace_dir(run_dir: Path) -> Path:
    """Resolves trace directory supporting Colab Drive, local, or direct trace dir paths."""
    # 1. Exact match run_dir / "trace"
    if (run_dir / "trace").is_dir():
        return run_dir / "trace"
    # 2. If run_dir itself is the trace directory
    if run_dir.is_dir() and run_dir.name == "trace":
        return run_dir
    # 3. If run_dir itself contains trace_it*.jsonl
    if run_dir.is_dir() and list(run_dir.glob("trace_it*.jsonl")):
        return run_dir
    # 4. Common path mappings between local & Colab Drive
    run_name = run_dir.name
    candidates = [
        Path("/content/drive/MyDrive/learn-a-world/atlas/runs") / run_name,
        Path("/content/drive/MyDrive/learn-a-world") / run_dir,
        Path("atlas/runs") / run_name,
        Path("atlas_runs") / run_name,
    ]
    for cand in candidates:
        if (cand / "trace").is_dir():
            return cand / "trace"
        if cand.is_dir() and cand.name == "trace":
            return cand
    return run_dir / "trace"


def _load_trace_lines(run_dir: Path, iter_idx: Optional[int] = None) -> List[Dict[str, Any]]:
    trace_dir = _resolve_trace_dir(run_dir)
    if not trace_dir.exists():
        sys.stderr.write(f"Trace directory not found: {trace_dir}\n")
        sys.exit(2)

    if iter_idx is not None:
        trace_file = trace_dir / f"trace_it{iter_idx}.jsonl"
        if not trace_file.exists():
            sys.stderr.write(f"Missing trace file: {trace_file}\n")
            sys.exit(2)
        files = [trace_file]
    else:
        files = sorted(trace_dir.glob("trace_it*.jsonl"))
        if not files:
            sys.stderr.write(f"No trace files found in {trace_dir}\n")
            sys.exit(2)

    lines: List[Dict[str, Any]] = []
    for f in files:
        with open(f, "r") as fp:
            for line_str in fp:
                line_str = line_str.strip()
                if line_str:
                    lines.append(json.loads(line_str))
    return lines


def run_story(args: argparse.Namespace) -> None:
    if getattr(args, "trace_file", None):
        trace_path = Path(args.trace_file)
        if "*" in args.trace_file:
            import glob
            matched = sorted(glob.glob(args.trace_file))
            if not matched:
                sys.stderr.write(f"No trace files matched pattern: {args.trace_file}\n")
                sys.exit(2)
            trace_path = Path(matched[0])
        if not trace_path.exists():
            sys.stderr.write(f"Missing trace file: {trace_path}\n")
            sys.exit(2)
        ep_lines = []
        with open(trace_path, "r") as fp:
            for line_str in fp:
                line_str = line_str.strip()
                if line_str:
                    ep_lines.append(json.loads(line_str))
        if getattr(args, "ep", None) is not None:
            ep_lines = [l for l in ep_lines if l.get("ep_idx") == args.ep]
    else:
        if args.run_dir is None or args.iter is None or args.ep is None:
            sys.stderr.write("story requires either --trace-file or (--run-dir, --iter, --ep)\n")
            sys.exit(2)
        run_dir = Path(args.run_dir)
        lines = _load_trace_lines(run_dir, iter_idx=args.iter)
        ep_lines = [line for line in lines if line.get("ep_idx") == args.ep]
    if not ep_lines:
        sys.stderr.write(f"No trace entries found for episode {args.ep} in iteration {args.iter}\n")
        sys.exit(2)

    for line in ep_lines:
        t = line.get("t", 0)
        if args.from_step is not None and t < args.from_step:
            continue
        if args.to_step is not None and t > args.to_step:
            continue

        mode = line.get("mode", "UNKNOWN")
        goal = line.get("goal")
        goal_str = "none"
        cos_val = 0.0
        psi_val = 0.0
        if goal:
            kind = goal.get("kind", "goal")
            dims = goal.get("dims", [])
            goal_str = f"{kind}{dims}"
            cos_val = goal.get("cos", 0.0)
            psi_val = goal.get("psi", 0.0)

        scores = line.get("scores", [])
        src = line.get("scores_source", "scores")
        # Top 3 scores
        if scores and len(scores) == len(CRAFTER_ACTION_NAMES):
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]
            top_scores_str = ", ".join(f"{_get_action_name(a)} {sc:.2f}" for a, sc in ranked)
        else:
            top_scores_str = "n/a"

        risk_val = line.get("override", {}).get("risk_intended", 0.0)
        eps_fired = line.get("epsilon_fired", False)
        eps_str = "fired" if eps_fired else "off"
        final_action = line.get("a_final", 0)
        final_name = _get_action_name(final_action)

        print(
            f"t={t:03d} {mode:<8} goal={goal_str} cos={cos_val:.2f} ψ={psi_val:.2f} | "
            f"{src}: {top_scores_str} | risk {risk_val:.2f} | ε {eps_str} | → {final_name}"
        )

        # Print !!! lines for notable events
        override = line.get("override", {})
        if override.get("fired", False):
            a_int = _get_action_name(override.get("a_intended", -1))
            a_fin = _get_action_name(override.get("a_final", -1))
            r_int = override.get("risk_intended", 0.0)
            r_min = override.get("min_risk", 0.0)
            print(f"  !!! [OVERRIDE] Intended '{a_int}' (risk={r_int:.2f}) -> Overridden to '{a_fin}' (min_risk={r_min:.2f})")

        unlocked = line.get("newly_unlocked", [])
        if unlocked:
            print(f"  !!! [UNLOCK] Unlocked achievement dimensions: {unlocked}")

        abort_reason = line.get("abort_reason", "none")
        if abort_reason != "none":
            print(f"  !!! [ABORT] Goal aborted: reason={abort_reason}")

        if line.get("arrived", False):
            print(f"  !!! [ARRIVE] Arrived at target key (arrival_cos={cos_val:.2f})")


def run_filter(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    lines = _load_trace_lines(run_dir, iter_idx=args.iter)

    for line in lines:
        if args.ep is not None and line.get("ep_idx") != args.ep:
            continue
        if args.mode is not None and line.get("mode") != args.mode:
            continue

        reasons = []
        override = line.get("override", {})
        if override.get("fired", False):
            reasons.append("override")

        abort_reason = line.get("abort_reason", "none")
        if abort_reason != "none":
            reasons.append(f"abort:{abort_reason}")

        unlocked = line.get("newly_unlocked", [])
        if unlocked:
            reasons.append(f"unlock:{unlocked}")

        if line.get("arrived", False):
            reasons.append("arrive")

        # Check filter flags
        matched = False
        if args.override and "override" in reasons:
            matched = True
        if args.abort and any(r.startswith("abort:") for r in reasons):
            matched = True
        if args.unlocked and any(r.startswith("unlock:") for r in reasons):
            matched = True
        if not (args.override or args.abort or args.unlocked):
            # If no specific reason flag is specified, print if any notable reason or matching mode
            matched = True

        if matched:
            t = line.get("t", 0)
            ep = line.get("ep_idx", 0)
            mode = line.get("mode", "UNKNOWN")
            r_str = " | ".join(reasons) if reasons else "normal"
            print(f"t={t:03d} ep={ep} mode={mode:<8} reason={r_str}")


def run_summary(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    lines = _load_trace_lines(run_dir, iter_idx=args.iter)

    # Group by (iter, ep_idx)
    episodes: Dict[Any, List[Dict[str, Any]]] = {}
    for line in lines:
        key = (line.get("iter"), line.get("ep_idx"))
        if key not in episodes:
            episodes[key] = []
        episodes[key].append(line)

    for (it, ep_idx), ep_steps in episodes.items():
        mode_counts = Counter(s.get("mode") for s in ep_steps)
        action_counts = Counter(_get_action_name(s.get("a_final", 0)) for s in ep_steps)
        top_3_actions = action_counts.most_common(3)
        overrides_count = sum(1 for s in ep_steps if s.get("override", {}).get("fired", False))
        aborts = Counter(s.get("abort_reason") for s in ep_steps if s.get("abort_reason", "none") != "none")
        all_unlocked = []
        for s in ep_steps:
            all_unlocked.extend(s.get("newly_unlocked", []))

        print(f"--- Iteration {it} | Episode {ep_idx} (Steps: {len(ep_steps)}) ---")
        mode_str = ", ".join(f"{k}: {v}" for k, v in mode_counts.items())
        print(f"  Modes:      {mode_str}")
        act_str = ", ".join(f"{k}: {v}" for k, v in top_3_actions)
        print(f"  Top Actions: {act_str}")
        print(f"  Overrides:  {overrides_count}")
        if aborts:
            abort_str = ", ".join(f"{k}: {v}" for k, v in aborts.items())
            print(f"  Aborts:     {abort_str}")
        print(f"  Unlocked:   {list(set(all_unlocked))}")

    # Check for probes json
    if args.iter is not None:
        probe_file = run_dir / "audit" / "probes" / f"probes_it{args.iter}.json"
        if probe_file.exists():
            with open(probe_file, "r") as fp:
                probe_data = json.load(fp)
            s_probes = probe_data.get("probes", {}).get("s", {})
            metrics = []
            for k, v in s_probes.items():
                m_val = v.get("r2") if "r2" in v else v.get("auroc")
                if m_val is not None:
                    metrics.append(f"{k}: {m_val:.2f}")
            if metrics:
                print(f"  Probe best (s): {', '.join(metrics[:5])}")


def run_goal(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    lines = _load_trace_lines(run_dir, iter_idx=args.iter)

    ep_lines = [line for line in lines if line.get("ep_idx") == args.ep]
    if not ep_lines:
        sys.stderr.write(f"No trace entries found for episode {args.ep} in iteration {args.iter}\n")
        sys.exit(2)

    current_goal_key = None
    for line in ep_lines:
        t = line.get("t", 0)
        goal = line.get("goal")
        mode = line.get("mode")
        arrived = line.get("arrived", False)
        abort_reason = line.get("abort_reason", "none")
        unlocked = line.get("newly_unlocked", [])

        if goal:
            goal_key = (goal.get("kind"), tuple(goal.get("dims", [])), goal.get("goal_mode"))
            if goal_key != current_goal_key:
                current_goal_key = goal_key
                kind = goal.get("kind")
                dims = goal.get("dims")
                gmode = goal.get("goal_mode")
                budget = goal.get("budget")
                n_try = goal.get("n_try")
                n_succ = goal.get("n_succ")
                print(f"t={t:03d} [GOAL SELECTED] {kind}{dims} mode='{gmode}' budget={budget} (tries: {n_try}, succ: {n_succ})")

        if arrived:
            cos = goal.get("cos", 0.0) if goal else 0.0
            print(f"t={t:03d} [GOAL ARRIVAL] Arrived at goal (cos={cos:.2f}) -> Entering PROBE")

        if unlocked:
            print(f"t={t:03d} [GOAL PROBE SUCCESS] Unlocked achievements: {unlocked}")

        if abort_reason != "none":
            print(f"t={t:03d} [GOAL ABORT] Aborted: {abort_reason}")
            current_goal_key = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Project ATLAS Decision Trace Replay CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # story
    p_story = subparsers.add_parser("story", help="Render step-by-step decision story")
    p_story.add_argument("--trace-file", type=str, default=None, help="Direct path or glob to trace JSONL file")
    p_story.add_argument("--run-dir", type=str, default=None, help="Path to run directory")
    p_story.add_argument("--iter", type=int, default=None, help="Iteration index")
    p_story.add_argument("--ep", type=int, default=None, help="Episode index")
    p_story.add_argument("--from", dest="from_step", type=int, default=None, help="Start step")
    p_story.add_argument("--to", dest="to_step", type=int, default=None, help="End step")
    p_story.set_defaults(func=run_story)

    # filter
    p_filter = subparsers.add_parser("filter", help="Filter trace lines by event")
    p_filter.add_argument("--run-dir", type=str, required=True, help="Path to run directory")
    p_filter.add_argument("--iter", type=int, default=None, help="Filter by iteration")
    p_filter.add_argument("--ep", type=int, default=None, help="Filter by episode")
    p_filter.add_argument("--mode", type=str, default=None, help="Filter by mode")
    p_filter.add_argument("--override", action="store_true", help="Match overrides")
    p_filter.add_argument("--abort", action="store_true", help="Match aborts")
    p_filter.add_argument("--unlocked", action="store_true", help="Match unlocks")
    p_filter.set_defaults(func=run_filter)

    # summary
    p_summary = subparsers.add_parser("summary", help="Summarize episodes from trace")
    p_summary.add_argument("--run-dir", type=str, required=True, help="Path to run directory")
    p_summary.add_argument("--iter", type=int, default=None, help="Iteration index")
    p_summary.set_defaults(func=run_summary)

    # goal
    p_goal = subparsers.add_parser("goal", help="Show full goal lifecycle view")
    p_goal.add_argument("--run-dir", type=str, required=True, help="Path to run directory")
    p_goal.add_argument("--iter", type=int, required=True, help="Iteration index")
    p_goal.add_argument("--ep", type=int, required=True, help="Episode index")
    p_goal.set_defaults(func=run_goal)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
