"""
tests/test_smoke_and_pipeline.py

Comprehensive Unit and Integration Tests for Milestone 5 Deliverables:
- src/test_smoke.py: Subprocess execution, latency budget (<60s), direct API, 13-stage verification.
- src/train.py: CLI argument parsing, TrainingPipeline stepping, world model & actor-critic updates,
  eval logging, achievement tracking, and CheckpointManager integration.
- notebooks/train_colab.ipynb: JSON structure, nbformat, AST syntax verification of all 7 code cells,
  Drive mounting and checkpoint directory setup.
"""

from typing import Dict, Any, List
from pathlib import Path
import os
import sys
import json
import ast
import time
import shutil
import tempfile
import subprocess
import unittest
import torch

from src.test_smoke import run_smoke_test
from src.train import parse_args, TrainingPipeline, train
from src.memory.storage import CheckpointManager


class TestSmokeTestDeliverable(unittest.TestCase):
    """Verifies src/test_smoke.py execution via CLI subprocess and direct Python API."""

    def test_01_smoke_test_subprocess_execution(self):
        """Executes 'python -m src.test_smoke' via subprocess and asserts returncode 0 under 60s."""
        repo_root = Path(__file__).resolve().parent.parent
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", "src.test_smoke"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=60.0,
        )
        elapsed = time.perf_counter() - t0

        self.assertEqual(
            proc.returncode,
            0,
            f"Smoke test subprocess failed with code {proc.returncode}.\nStderr: {proc.stderr}\nStdout: {proc.stdout}",
        )
        self.assertLess(elapsed, 60.0, f"Execution time {elapsed:.2f}s exceeded 60.0s budget!")

        # Verify key milestone output tokens in stdout
        stdout = proc.stdout
        self.assertIn("C R A F T E R   J E P A   C P U   S M O K E   T E S T", stdout)
        self.assertIn("Stage 1: Environment initialized", stdout)
        self.assertIn("Stage 2: Zero privileged cheat state confirmed", stdout)
        self.assertIn("Stage 3: Dual-Mode Stepping", stdout)
        self.assertIn("Stage 4: Visual Latent Encoding", stdout)
        self.assertIn("Stage 5: 4-Head Predictor Ensemble", stdout)
        self.assertIn("Stage 6: Epistemic Curiosity Engine", stdout)
        self.assertIn("Stage 7: Polyak EMA Target Encoder", stdout)
        self.assertIn("Stage 8: Two-Tier Buffer Mixed Sampling", stdout)
        self.assertIn("Stage 9: VICReg Loss", stdout)
        self.assertIn("Stage 10: Backpropagation & Non-Zero Gradients", stdout)
        self.assertIn("Stage 11: Non-Collapse Verified", stdout)
        self.assertIn("Stage 12: Execution Time Budget", stdout)
        self.assertIn("ALL 13 STAGES PASSED", stdout)

    def test_02_smoke_test_direct_api_telemetry(self):
        """Invokes run_smoke_test() directly and asserts full telemetry payload integrity."""
        report = run_smoke_test()
        self.assertIsInstance(report, dict)

        expected_keys = [
            "stage_1_env_creation_ms",
            "stage_2_zero_cheat_ms",
            "stage_3_fast_latency_ms",
            "stage_3_plan_latency_ms",
            "stage_4_encode_ms",
            "stage_5_predictor_ms",
            "stage_6_curiosity_ms",
            "r_int",
            "stage_7_ema_update_ms",
            "stage_8_buffer_ms",
            "stage_9_vicreg_ms",
            "loss_total",
            "loss_inv",
            "loss_var",
            "loss_cov",
            "stage_10_backward_ms",
            "stage_11_non_collapse_ms",
            "healthy_min_var",
            "effective_rank",
            "total_elapsed_seconds",
        ]
        for k in expected_keys:
            self.assertIn(k, report, f"Missing telemetry metric '{k}' in smoke test report")

        self.assertLess(report["total_elapsed_seconds"], 60.0)
        self.assertGreater(report["healthy_min_var"], 0.1)
        self.assertGreater(report["effective_rank"], 10.0)


class TestTrainingPipelineDeliverable(unittest.TestCase):
    """Verifies src/train.py CLI parsing, training execution, checkpointing, and evaluation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="jepa_train_test_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_03_train_arg_parser_defaults_and_custom_flags(self):
        """Verifies parse_args default values and custom CLI flag overrides."""
        # Test defaults
        defaults = parse_args([])
        self.assertIn(defaults.device, ("cpu", "cuda", "mps"))
        self.assertEqual(defaults.total_steps, 10_000)
        self.assertEqual(defaults.batch_size, 128)
        self.assertEqual(defaults.checkpoint_dir, "./checkpoints")
        self.assertEqual(defaults.eval_interval, 1_000)
        self.assertEqual(defaults.seed, 42)

        # Test custom arguments
        custom_cli = [
            "--device", "cpu",
            "--total_steps", "250",
            "--batch_size", "16",
            "--checkpoint_dir", self.temp_dir,
            "--eval_interval", "50",
            "--seed", "99",
            "--eval_episodes", "2",
            "--lr", "5e-4",
            "--critic_lr", "1e-3",
            "--warmup_steps", "20",
            "--no_resume",
        ]
        parsed = parse_args(custom_cli)
        self.assertEqual(parsed.device, "cpu")
        self.assertEqual(parsed.total_steps, 250)
        self.assertEqual(parsed.batch_size, 16)
        self.assertEqual(parsed.checkpoint_dir, self.temp_dir)
        self.assertEqual(parsed.eval_interval, 50)
        self.assertEqual(parsed.seed, 99)
        self.assertEqual(parsed.eval_episodes, 2)
        self.assertEqual(parsed.lr, 5e-4)
        self.assertEqual(parsed.critic_lr, 1e-3)
        self.assertEqual(parsed.warmup_steps, 20)
        self.assertTrue(parsed.no_resume)

    def test_04_training_pipeline_mini_run_and_checkpoint_resumption(self):
        """Executes a 25-step training pipeline run, asserting updates, metrics, and checkpoint save."""
        config = {
            "device": "cpu",
            "total_steps": 20,
            "batch_size": 4,
            "checkpoint_dir": self.temp_dir,
            "eval_interval": 10,
            "warmup_steps": 5,
            "eval_episodes": 1,
            "seed": 42,
            "no_resume": True,
            "save_interval": 10,
        }

        pipeline = TrainingPipeline(config)
        results = pipeline.train()

        self.assertIsInstance(results, dict)
        self.assertEqual(results["total_steps"], 20)
        self.assertIn("achievement_history", results)
        self.assertIn("final_checkpoint", results)
        self.assertTrue(os.path.exists(results["final_checkpoint"]))

        # Verify checkpoint contents via CheckpointManager
        manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        loaded = manager.load(results["final_checkpoint"])
        self.assertIn("components", loaded)
        self.assertIn("metadata", loaded)
        self.assertIn("agent", loaded["components"])
        self.assertIn("buffer", loaded["components"])
        self.assertEqual(loaded["metadata"]["step"], 20)

        # Test resumption into a new pipeline instance
        resume_config = {
            "device": "cpu",
            "total_steps": 25,
            "batch_size": 4,
            "checkpoint_dir": self.temp_dir,
            "eval_interval": 10,
            "warmup_steps": 5,
            "eval_episodes": 1,
            "seed": 42,
            "no_resume": False,
        }
        resumed_pipeline = TrainingPipeline(resume_config)
        self.assertEqual(resumed_pipeline.current_step, 20)
        self.assertGreaterEqual(resumed_pipeline.buffer.size, 20)


class TestColabNotebookDeliverable(unittest.TestCase):
    """Verifies notebooks/train_colab.ipynb JSON validity, cell structure, and code blocks."""

    def setUp(self):
        repo_root = Path(__file__).resolve().parent.parent
        self.nb_path = repo_root / "notebooks" / "train_colab.ipynb"

    def test_05_notebook_json_structure_and_nbformat(self):
        """Verifies notebook exists, parses cleanly as valid JSON, and matches nbformat 4."""
        self.assertTrue(self.nb_path.exists(), f"Notebook not found at {self.nb_path}")

        with open(self.nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        self.assertEqual(nb.get("nbformat"), 4, f"Expected nbformat 4, got {nb.get('nbformat')}")
        self.assertIn("cells", nb)
        self.assertIsInstance(nb["cells"], list)
        self.assertGreaterEqual(len(nb["cells"]), 7)

    def test_06_notebook_required_7_cells_and_ast_compilation(self):
        """Verifies all 7 required functional code blocks exist, compile under AST, and match requirements."""
        with open(self.nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 7, f"Expected at least 7 code cells, found {len(code_cells)}")

        # Verify syntax of every code cell using Python AST
        for idx, cell in enumerate(code_cells, start=1):
            source = "".join(cell.get("source", []))
            self.assertTrue(len(source.strip()) > 0, f"Code cell {idx} is empty")
            try:
                ast.parse(source)
            except SyntaxError as e:
                self.fail(f"Syntax error in notebook code cell {idx}: {e}")

        # Cell 1: Environment verification
        cell1_src = "".join(code_cells[0]["source"])
        self.assertIn("torch.cuda.is_available()", cell1_src)
        self.assertIn("PyTorch Version", cell1_src)

        # Cell 2: Google Drive mounting & checkpoints
        cell2_src = "".join(code_cells[1]["source"])
        self.assertIn("google.colab", cell2_src)
        self.assertIn("drive.mount('/content/drive')", cell2_src)
        self.assertTrue(
            "/content/drive/MyDrive/learn-a-world/prometheus/checkpoints" in cell2_src
            or "/content/drive/MyDrive/crafter_jepa/checkpoints" in cell2_src
        )

        # Cell 3: Dependency installation
        cell3_src = "".join(code_cells[2]["source"])
        self.assertIn("crafter", cell3_src)
        self.assertIn("gymnasium", cell3_src)
        self.assertIn("matplotlib", cell3_src)

        # Cell 4: Repository setup & path configuration
        cell4_src = "".join(code_cells[3]["source"])
        self.assertIn("sys.path", cell4_src)
        self.assertIn("from src.agent.agent import JEPAAgent", cell4_src)

        # Cell 5: Fast CPU smoke test verification
        cell5_src = "".join(code_cells[4]["source"])
        self.assertIn("run_smoke_test", cell5_src)

        # Cell 6: Full GPU training run with Drive checkpointing
        cell6_src = "".join(code_cells[5]["source"])
        self.assertIn("TrainingPipeline", cell6_src)
        self.assertIn("CHECKPOINT_DIR", cell6_src)

        # Cell 7: Visualization and evaluation
        cell7_src = "".join(code_cells[6]["source"])
        self.assertIn("matplotlib", cell7_src)
        self.assertIn("invariance", cell7_src)
        self.assertIn("variance", cell7_src)
        self.assertIn("achievement", cell7_src)


if __name__ == "__main__":
    unittest.main()
