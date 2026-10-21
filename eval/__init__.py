"""Evaluation and metrics module for Learn-a-World."""

from eval.benchmark_table import format_markdown_table, run_phase2_benchmarks
from eval.metrics import compute_crafter_score, evaluate_crafter_agent, compute_multi_step_drift

__all__ = [
    "run_phase2_benchmarks",
    "format_markdown_table",
    "compute_crafter_score",
    "evaluate_crafter_agent",
    "compute_multi_step_drift",
]

