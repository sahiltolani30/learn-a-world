# Dead Ends Log (Append-Only)

| Iteration | Approach Tried | Why It Failed | Files Touched |
|-----------|---------------|---------------|---------------|
| Phase 0 | Pure Single-Predictor MSE Latent Error | Suffers from "Noisy TV" aleatoric trap on stochastic environment noise | None (rejected in Survey R6) |
| Phase 0 | Pure Model-Free Actor-Critic | High sample complexity, local minima in sparse Crafter affordances | None (rejected in Survey R6) |
| Phase 0 | Pure Unbounded CEM Rollouts (H=10) | Combinatorial action explosion ($17^{10}$), compounding open-loop drift | None (rejected in Survey R6) |
