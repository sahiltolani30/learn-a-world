# MOSAIC Colab Operations

This directory documents the future Colab runner. The bootstrap commit is **not runnable for MOSAIC training yet** because P0/P1 implementation is still pending.

## Drive location

Use the dedicated model directory:

```text
/content/drive/MyDrive/learn-a-world/mosaic/
```

Do not write MOSAIC checkpoints or replay shards into another model's directory.

## Required order

1. Mount Drive.
2. Clone the committed GitHub SHA into `/content/learn-a-world`.
3. Install dependencies from the phase-specific requirements/configuration.
4. Create or verify the `mosaic/` Drive subdirectories.
5. Run the phase's integrity and smoke checks.
6. Run the short profile job.
7. Inspect the manifest and metrics before starting a longer segment.

## First approved job

The first approved Colab job is P1 baseline profiling after P0 is complete. It must use a short fixed budget, record T4 throughput/latency/memory, and compare the recurrent baseline controls. It must not enable atlas, skills, or planner authority.

The command will be:

```bash
python -m mosaic.colab.run_phase \
  --phase p1_baseline \
  --drive-root /content/drive/MyDrive/learn-a-world/mosaic \
  --config mosaic/configs/p1_baseline.yaml \
  --run-id p1_profile_001
```

The runner is implemented in `mosaic/colab/run_phase.py`; execute this command only after the P0 gate is recorded as passing.
