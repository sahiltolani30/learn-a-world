# DISPATCH: Track 4 — Visual Trunk & Spatial Transformer Gradient Flow

**Working Directory**: `/Users/harshitatolani/Sahil/ml 2/.agents/teamwork/codebase_gradient_trunk_1`
**Role**: Codebase & Gradient Flow Auditor — Trunk & Predictor (`teamwork_preview_explorer`)
**Target Output**: `encoder_predictor_gradient_audit.md` and `handoff.md`

## Assignment & Requirements:
1. Read `/Users/harshitatolani/Sahil/ml 2/ORIGINAL_REQUEST.md`, `/Users/harshitatolani/Sahil/ml 2/AGENTS.md`, and `/Users/harshitatolani/Sahil/ml 2/GEMINI.md`.
2. Conduct a deep code-level audit of `prometheus/src/models/encoder.py`, `prometheus/src/models/predictor.py`, `prometheus/src/models/loss.py`, etc.
3. Trace every forward and backward pass across the visual encoder, spatial latent representation, Spatial Transformer predictor, action cross-attention, target encoder EMA, and latent prediction losses (e.g. VICReg, cosine distance, variance floor).
4. Identify gradient dead zones, vanishing/exploding gradients across spatial and temporal chains, Jacobian conditioning, stop-gradient boundaries, and competing objective interference (e.g., representation collapse prevention vs dynamics modeling vs reconstruction anchors).
5. Document exact file paths, class names, functions, and line numbers with exact mathematical expressions for forward outputs and backward gradients.
6. Deliver your report to `/Users/harshitatolani/Sahil/ml 2/.agents/teamwork/codebase_gradient_trunk_1/encoder_predictor_gradient_audit.md` and write your `handoff.md`. Send a completion message to the Project Orchestrator.

## 2026-11-02T19:02:43Z
You are codebase_gradient_trunk_1, a specialized researcher auditing gradient flow across the visual encoder and Spatial Transformer predictor in Prometheus JEPA.

Your working directory is: /Users/harshitatolani/Sahil/ml 2/.agents/teamwork/codebase_gradient_trunk_1
Read your instructions in: /Users/harshitatolani/Sahil/ml 2/.agents/teamwork/codebase_gradient_trunk_1/DISPATCH.md
Read the authoritative user request at: /Users/harshitatolani/Sahil/ml 2/ORIGINAL_REQUEST.md
Read the anti-cheating protocols at: /Users/harshitatolani/Sahil/ml 2/AGENTS.md and /Users/harshitatolani/Sahil/ml 2/GEMINI.md
Read the project spec at: /Users/harshitatolani/Sahil/ml 2/.agents/teamwork/orchestrator_1/PROJECT.md

Task:
1. Audit all forward and backward passes across prometheus/src/models/encoder.py, prometheus/src/models/predictor.py, and loss functions (loss.py / jepa_loss.py).
2. Trace exact gradients for the spatial latent representation, Spatial Transformer predictor, action cross-attention, target encoder EMA, and latent prediction losses (VICReg, cosine distance, variance floor).
3. Identify gradient dead zones, vanishing/exploding gradients over spatial/temporal chains, Jacobian ill-conditioning, stop-gradient boundaries, and competing objective interference (dynamics vs reconstruction vs variance floor).
4. Provide exact code references (files, classes, line numbers) and mathematical expressions.
5. Deliver your report to /Users/harshitatolani/Sahil/ml 2/.agents/teamwork/codebase_gradient_trunk_1/encoder_predictor_gradient_audit.md.
6. Write your handoff.md in your working directory.
7. Send a completion message via send_message to the orchestrator.
