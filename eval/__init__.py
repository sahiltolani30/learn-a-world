"""Evaluation and metrics module for Learn-a-World."""

try:
    from eval.adaptation_benchmark import run_adaptation_benchmarks
except ImportError:
    run_adaptation_benchmarks = None

try:
    from eval.benchmark_table import format_markdown_table, run_phase2_benchmarks
except ImportError:
    format_markdown_table = None
    run_phase2_benchmarks = None

try:
    from eval.metrics import compute_crafter_score, evaluate_crafter_agent, compute_multi_step_drift
except ImportError:
    compute_crafter_score = None
    evaluate_crafter_agent = None
    compute_multi_step_drift = None

__all__ = [
    "run_phase2_benchmarks",
    "run_adaptation_benchmarks",
    "format_markdown_table",
    "compute_crafter_score",
    "evaluate_crafter_agent",
    "compute_multi_step_drift",
]


