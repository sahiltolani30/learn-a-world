"""Zephyr E2E Test Runner & Diagnostic Suite.

Provides standalone CLI execution, tier-specific filtering, structured JSON
reporting, microsecond timing, and deep failure diagnostics.

Usage:
    python tests/e2e/runner.py --tier all
    python tests/e2e/runner.py --tier tier1,tier2 --fail-fast
    python tests/e2e/runner.py --dry-run
    pytest tests/e2e/runner.py -v
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
E2E_ROOT = pathlib.Path(__file__).resolve().parent

TIER_MAPPING: Dict[str, str] = {
    "tier1": "tests/e2e/tier1_features",
    "tier2": "tests/e2e/tier2_boundaries",
    "tier3": "tests/e2e/tier3_combinations",
    "tier4": "tests/e2e/tier4_real_world",
    "tier5": "tests/e2e/tier5_adversarial",
}

TIER_ALIASES: Dict[str, str] = {
    "t1": "tier1", "features": "tier1",
    "t2": "tier2", "boundaries": "tier2", "bva": "tier2",
    "t3": "tier3", "combinations": "tier3", "pairwise": "tier3",
    "t4": "tier4", "scenarios": "tier4", "benchmark": "tier4",
    "t5": "tier5", "adversarial": "tier5",
}

TIER_TITLES: Dict[str, str] = {
    "tier1": "Tier 1: Feature Coverage",
    "tier2": "Tier 2: Boundary Value Analysis",
    "tier3": "Tier 3: Pairwise Combinations",
    "tier4": "Tier 4: Real-World Scenarios",
    "tier5": "Tier 5: Adversarial Hardening",
}


@dataclasses.dataclass
class TestCaseResult:
    __test__ = False
    nodeid: str
    tier: str
    name: str
    outcome: str  # "passed", "failed", "skipped", "error"
    duration_sec: float
    message: Optional[str] = None
    traceback: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    file: Optional[str] = None
    line: Optional[int] = None


@dataclasses.dataclass
class TierSummary:
    tier: str
    title: str
    discovered: int = 0
    executed: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_sec: float = 0.0
    results: List[TestCaseResult] = dataclasses.field(default_factory=list)

    @property
    def pass_rate_pct(self) -> float:
        if self.executed == 0:
            return 100.0 if self.discovered > 0 else 0.0
        return (self.passed / self.executed) * 100.0

    @property
    def status(self) -> str:
        if self.failed > 0 or self.errors > 0:
            return "FAILED"
        if self.executed == 0 and self.discovered == 0:
            return "SKIPPED"
        return "PASSED"


class E2ECollectorPlugin:
    """Pytest plugin to record execution metrics and failure traces."""

    def __init__(self, target_tiers: List[str]):
        self.target_tiers = target_tiers
        self.tier_summaries: Dict[str, TierSummary] = {
            t: TierSummary(tier=t, title=TIER_TITLES.get(t, t)) for t in target_tiers
        }
        self.failures: List[TestCaseResult] = []
        self._start_time: float = 0.0

    def _get_tier_for_path(self, path_str: str) -> Optional[str]:
        norm_path = path_str.replace("\\", "/")
        for tier, rel_path in TIER_MAPPING.items():
            if rel_path in norm_path:
                return tier
        return None

    def pytest_collection_modifyitems(self, session, config, items):
        for item in items:
            tier = self._get_tier_for_path(str(item.fspath))
            if tier and tier in self.tier_summaries:
                self.tier_summaries[tier].discovered += 1

    def pytest_runtest_logreport(self, report):
        if report.when != "call" and not (report.failed and report.when in ("setup", "teardown")):
            return

        tier = self._get_tier_for_path(str(report.fspath))
        if not tier or tier not in self.tier_summaries:
            return

        summary = self.tier_summaries[tier]
        summary.executed += 1
        summary.duration_sec += report.duration

        outcome = report.outcome
        if report.failed:
            outcome = "failed"
            summary.failed += 1
        elif report.skipped:
            outcome = "skipped"
            summary.skipped += 1
        elif report.passed:
            outcome = "passed"
            summary.passed += 1

        message, tb, file, line = None, None, None, None
        if report.failed:
            message = str(report.longrepr)
            if hasattr(report.longrepr, "reprcrash"):
                crash = report.longrepr.reprcrash
                file = crash.path
                line = crash.lineno
                message = crash.message
            tb = str(report.longrepr)

        stdout = getattr(report, "capstdout", "")
        stderr = getattr(report, "capstderr", "")

        res = TestCaseResult(
            nodeid=report.nodeid,
            tier=tier,
            name=report.location[2] if len(report.location) > 2 else report.nodeid,
            outcome=outcome,
            duration_sec=report.duration,
            message=message,
            traceback=tb,
            stdout=stdout,
            stderr=stderr,
            file=str(report.fspath),
            line=line,
        )
        summary.results.append(res)
        if outcome == "failed":
            self.failures.append(res)


class E2ETestRunner:
    """Manages test execution across tiers and report generation."""

    def __init__(
        self,
        tiers: List[str],
        fail_fast: bool = False,
        filter_expr: Optional[str] = None,
        dry_run: bool = False,
        verbose: bool = False,
        output_json: str = "reports/e2e_results.json",
        timeout_sec: Optional[int] = None,
    ):
        self.tiers = tiers
        self.fail_fast = fail_fast
        self.filter_expr = filter_expr
        self.dry_run = dry_run
        self.verbose = verbose
        self.output_json = output_json
        self.timeout_sec = timeout_sec

    def run(self) -> int:
        collector = E2ECollectorPlugin(self.tiers)
        target_dirs = [
            str(REPO_ROOT / TIER_MAPPING[t]) for t in self.tiers if (REPO_ROOT / TIER_MAPPING[t]).exists()
        ]

        if not target_dirs:
            print("[ERROR] No valid tier directories exist on disk.")
            return 4

        pytest_args = target_dirs + ["-q"]
        if self.fail_fast:
            pytest_args.append("-x")
        if self.filter_expr:
            pytest_args.extend(["-k", self.filter_expr])
        if self.dry_run:
            pytest_args.append("--collect-only")
        if self.verbose:
            pytest_args.append("-v")

        start_time = time.perf_counter()
        exit_code = pytest.main(pytest_args, plugins=[collector])
        total_duration = time.perf_counter() - start_time

        # Render Terminal UI Table
        self.render_terminal_table(collector.tier_summaries, total_duration, int(exit_code))

        # Output Failures
        if collector.failures:
            self.render_failures(collector.failures)

        # Emit JSON Report
        self.save_json_report(collector.tier_summaries, collector.failures, total_duration, int(exit_code))

        return 0 if exit_code == 0 else (1 if collector.failures else int(exit_code))

    def render_terminal_table(
        self, summaries: Dict[str, TierSummary], total_duration: float, exit_code: int
    ) -> None:
        print("\n" + "=" * 100)
        print("                                 ZEPHYR E2E TEST EXECUTION REPORT                                 ")
        print("=" * 100)
        header = f"{'Tier':<32} {'Discovered':>10} {'Executed':>10} {'Passed':>8} {'Failed':>8} {'Skipped':>9} {'Pass Rate':>11} {'Time':>8}"
        print(header)
        print("-" * 100)

        tot_disc, tot_exec, tot_pass, tot_fail, tot_skip = 0, 0, 0, 0, 0
        for tier in self.tiers:
            if tier not in summaries:
                continue
            s = summaries[tier]
            tot_disc += s.discovered
            tot_exec += s.executed
            tot_pass += s.passed
            tot_fail += s.failed
            tot_skip += s.skipped
            line = (
                f"{s.title:<32} {s.discovered:>10} {s.executed:>10} {s.passed:>8} {s.failed:>8} "
                f"{s.skipped:>9} {s.pass_rate_pct:>10.1f}% {s.duration_sec:>7.1f}s"
            )
            print(line)

        print("-" * 100)
        tot_rate = (tot_pass / tot_exec * 100.0) if tot_exec > 0 else (100.0 if tot_disc > 0 else 0.0)
        tot_line = (
            f"{'TOTAL':<32} {tot_disc:>10} {tot_exec:>10} {tot_pass:>8} {tot_fail:>8} "
            f"{tot_skip:>9} {tot_rate:>10.1f}% {total_duration:>7.1f}s"
        )
        print(tot_line)
        print("=" * 100)
        status_str = "[PASSED]" if (exit_code == 0 and tot_fail == 0) else "[FAILED]"
        print(f"OVERALL STATUS: {status_str} {tot_rate:.2f}% Pass Rate (Exit Code {exit_code})")
        print(f"Report JSON written to: {self.output_json}")
        print("=" * 100 + "\n")

    def render_failures(self, failures: List[TestCaseResult]) -> None:
        print("=" * 100)
        print(f"                                       FAILURE DIAGNOSTICS ({len(failures)})                                      ")
        print("=" * 100)
        for idx, f in enumerate(failures, start=1):
            title = TIER_TITLES.get(f.tier, f.tier)
            print(f"[{idx}/{len(failures)}] {title}")
            print(f"Test ID:   {f.nodeid}")
            print(f"Location:  {f.file}:{f.line}")
            print(f"Message:   {f.message}")
            if f.traceback:
                print("Traceback:")
                for l in f.traceback.strip().splitlines():
                    print(f"  {l}")
            if f.stdout:
                print(f"Captured Stdout:\n  {f.stdout.strip()}")
            if f.stderr:
                print(f"Captured Stderr:\n  {f.stderr.strip()}")
            print("-" * 100)

    def save_json_report(
        self,
        summaries: Dict[str, TierSummary],
        failures: List[TestCaseResult],
        total_duration: float,
        exit_code: int,
    ) -> None:
        out_path = pathlib.Path(self.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        tot_disc = sum(s.discovered for s in summaries.values())
        tot_exec = sum(s.executed for s in summaries.values())
        tot_pass = sum(s.passed for s in summaries.values())
        tot_fail = sum(s.failed for s in summaries.values())
        tot_skip = sum(s.skipped for s in summaries.values())
        tot_err = sum(s.errors for s in summaries.values())
        tot_rate = (tot_pass / tot_exec * 100.0) if tot_exec > 0 else 0.0

        payload = {
            "metadata": {
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "platform": sys.platform,
                "python_version": sys.version,
                "pytest_version": pytest.__version__,
                "cmd_args": sys.argv,
            },
            "summary": {
                "overall_status": "PASSED" if (exit_code == 0 and tot_fail == 0) else "FAILED",
                "exit_code": exit_code,
                "total_discovered": tot_disc,
                "total_executed": tot_exec,
                "total_passed": tot_pass,
                "total_failed": tot_fail,
                "total_skipped": tot_skip,
                "total_errors": tot_err,
                "pass_rate_pct": round(tot_rate, 2),
                "duration_sec": round(total_duration, 3),
            },
            "tiers": {
                t: {
                    "title": s.title,
                    "status": s.status,
                    "discovered": s.discovered,
                    "executed": s.executed,
                    "passed": s.passed,
                    "failed": s.failed,
                    "skipped": s.skipped,
                    "pass_rate_pct": round(s.pass_rate_pct, 2),
                    "duration_sec": round(s.duration_sec, 3),
                    "tests": [
                        {
                            "nodeid": r.nodeid,
                            "name": r.name,
                            "outcome": r.outcome,
                            "duration_sec": round(r.duration_sec, 4),
                        }
                        for r in s.results
                    ],
                }
                for t, s in summaries.items()
            },
            "failures": [
                {
                    "test_id": f.nodeid,
                    "tier": f.tier,
                    "file": f.file,
                    "line": f.line,
                    "message": f.message,
                    "traceback": f.traceback,
                    "stdout": f.stdout,
                    "stderr": f.stderr,
                }
                for f in failures
            ],
        }

        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)


def test_e2e_runner_meta_check():
    """Pytest hook ensuring runner modules and tier directories are discoverable."""
    assert (REPO_ROOT / "tests/e2e").exists(), "tests/e2e directory must exist"
    assert TIER_MAPPING["tier1"] == "tests/e2e/tier1_features"
    assert TIER_MAPPING["tier2"] == "tests/e2e/tier2_boundaries"
    assert TIER_MAPPING["tier3"] == "tests/e2e/tier3_combinations"
    assert TIER_MAPPING["tier4"] == "tests/e2e/tier4_real_world"


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zephyr E2E Test Runner")
    parser.add_argument(
        "--tier", "-t",
        default="all",
        help="Test tier(s) to run: all, tier1, tier2, tier3, tier4 (comma-separated)",
    )
    parser.add_argument("--fail-fast", "-x", action="store_true", help="Stop on first failure")
    parser.add_argument("--filter", "-k", default=None, help="Pytest keyword expression filter")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Discover tests without running")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose test execution")
    parser.add_argument(
        "--output-json", "-o",
        default="reports/e2e_results.json",
        help="Output path for JSON test report",
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()

    # Resolve requested tiers
    if args.tier.lower() == "all":
        selected_tiers = ["tier1", "tier2", "tier3", "tier4"]
    else:
        raw_tiers = [t.strip().lower() for t in args.tier.split(",")]
        selected_tiers = [TIER_ALIASES.get(t, t) for t in raw_tiers if t]

    runner = E2ETestRunner(
        tiers=selected_tiers,
        fail_fast=args.fail_fast,
        filter_expr=args.filter,
        dry_run=args.dry_run,
        verbose=args.verbose,
        output_json=args.output_json,
    )
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
