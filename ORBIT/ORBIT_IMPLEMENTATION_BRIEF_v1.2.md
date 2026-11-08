# ORBIT — Research, Architecture, Mathematics, Implementation and Verification Brief

**Document status:** Pre-implementation specification with a separate frozen DINOv2 encoder diagnostic; not an implementation claim
**Version:** 1.2
**Prepared:** 5 November 2026
**Repository target:** `sahiltolani30/learn-a-world`
**Implementation target:** audited extension of existing `true_jepa/` unless a separate package is approved by a recorded architecture decision
**Research objective:** Build a visual, self-learning agent that develops useful world understanding, discovers rules and reusable behaviours through interaction, and improves at the unmodified Crafter benchmark without privileged game-state access or hand-written gameplay strategies.


# ORBIT v1.2 — Binding Forensic, Mathematical and Encoder-Comparison Amendment

> **READ THIS SECTION FIRST.** This amendment is part of the controlling specification. If any statement in the original v1.0 body conflicts with this amendment, **v1.2 wins**. It retains the v1.1 forensic, mathematical, interface, replay, checkpoint, evaluation, and debugging corrections, and adds a controlled frozen DINOv2 visual-encoder diagnostic. It is not a claim that any code has already been fixed.

**Revision date:** 5 November 2026
**v1.2 change:** Add Meta AI DINOv2 ViT-S/14 as a frozen, diagnostic-only encoder arm; preserve the from-scratch ORBIT track as the primary result.
**Review target:** `sahiltolani30/learn-a-world`, branch `main`, reviewed tree `b5c6964e9f5b4892bb6681091a00efd27a4de01c`
**Current implementation candidate:** existing `true_jepa/` line, subject to the gates below. Do not create a second implementation package until the code audit and migration decision are complete.

## A. Controlling decisions and non-negotiable rules

1. **No implementation from diagrams alone.** The implementing AI must translate every module into an interface contract, tensor schema, equation, gradient path, unit test, integration test, failure trace, and gate before coding it.
2. **No guessed values.** Every numerical setting is labelled `DERIVED`, `ENVIRONMENT_DEFINED`, `PAPER_REPRODUCED`, `INHERITED_CANDIDATE`, `EXPERIMENT_SELECTED`, or `OPERATIONAL_LIMIT`. An inherited value is not approved merely because it exists in `config.py`.
3. **No hidden data.** Learner, replay, world model, actor, value function, curiosity, memory, skill discovery, and curriculum student may receive only pixel observations, previous action, raw scalar environment reward, and reset/termination control necessary for correct sequence handling. Evaluator-only fields are stored on a physically/logically separate path.
4. **No hard-coded semantics.** No recipes, material counts, action-name dictionaries, terrain rules, survival thresholds, object-name rules, milestone order, or handcrafted strategy branches. Never reintroduce the `3 wood` budgeter described in the retrospective's F35; it violates this contract.
5. **No automatic phase promotion.** A gate with missing telemetry, unrun tests, failed tests, or an unresolved `DECISION_REQUIRED` is `BLOCKED`, not passed.
6. **One controlled change per experiment.** No simultaneous changes to encoder, replay, policy, loss, optimizer, and exploration. An experiment must have a hypothesis, control, fixed budget, seed manifest, primary metric, stopping rule, and analysis written before the run.
7. **A model is not useful because a loss decreases.** Each learned module must demonstrate nonzero gradients, parameter change, predictive/behavioral utility, and causal influence on downstream decisions.
8. **Do not use achievement labels for training.** Achievement names and booleans are evaluator-only. They may be read after an episode to report the benchmark, never to form labels, replay priorities, curricula, reward shaping, model inputs, or planner decisions.

## B. Repository forensic audit — findings that must be fixed or explicitly gated

The following findings were checked against the listed `main`-branch files at the reviewed tree. They are findings about that snapshot, not assumptions about a future commit. Re-run these checks against the exact implementation commit before each phase.

| ID | Severity | Verified finding | Consequence | Mandatory correction / test |
|---|---|---|---|---|
| R01 | **BLOCKER** | `results/screen_v3_champion_clean_50ep.json` reports `3.17087726205484`; `results/screen_v3_t2_final_results.json` reports `4.55`. Both declare 50 episodes, seeds 2050–2099, and the same checkpoint path string. Per-episode and per-achievement outcomes differ. | The historical 4.55% result is not a reproducible baseline merely because the path strings match. | Re-run from an immutable checkpoint whose SHA-256 is captured, pin code/config/environment/evaluator/calibrator/action-map/memory hashes, save per-episode evaluator outputs, and independently recompute score. Mark the historical 4.55% `PROVENANCE_UNRESOLVED` until reconciled. |
| R02 | **BLOCKER** | `envs/crafter_wrapper.py` returns `inventory`, `achievements`, `semantic`, and `player_pos` in `info`. `eval/metrics.py` calls `action_fn(obs, info)`. | A policy callback can accidentally read privileged fields. | Replace the policy callback signature with the strict pixel/reward/reset API. The evaluator must collect `info` through a separate evaluator-only object that is never passed to the policy. Poison-test this boundary. |
| R03 | **BLOCKER** | `envs/crafter_wrapper.py` sets `self.max_steps` but always returns `truncated=False`. | Its configured time limit is not enforced; evaluation episode semantics can silently differ from the declared budget. | Test the actual wrapper. Enforce truncation exactly at the declared limit and preserve the final observation. Never use this wrapper as the official baseline until tested. |
| R04 | **HIGH** | `envs/crafter_wrapper.py` monkey-patches Crafter's global `_balance_object` and `World.chunks` at import time to impose ordering. | This may change environment dynamics and makes the benchmark environment different from unmodified upstream Crafter. | Do not monkey-patch globally in the canonical benchmark. If deterministic patching is retained for a separate diagnostic protocol, fingerprint it, version it, test parity, and report it as a distinct environment. No import-time global patching. |
| R05 | **BLOCKER** | `true_jepa/replay.py::PixelReplay.sample` chooses a random mid-episode window, but `true_jepa/objectives.py::train_world_model_step` initializes the recurrent belief to zero at the sampled window start. | The recurrent state is wrong for almost every sampled mid-episode window. History-dependent prediction is trained under a false reset. | Sample a history prefix and reconstruct hidden state using current weights. The reference implementation must use a full episode prefix first; only introduce bounded burn-in after measuring approximation error against full-prefix hidden states. |
| R06 | **BLOCKER** | In `train_world_model_step`, `current` is replaced by the predicted frame, while the hidden update consumes that predicted frame but the next reward input is the **real** reward. | The sequence is a mixed teacher-forced/free-running path: predicted visual state + observed reward. Online inference updates belief from real observed frames; pure imagination updates from predicted frames and predicted rewards. These distributions are not the same. | Implement and report two explicit modes: (a) teacher-forced training, using actual encoded frames and actual rewards at every step; (b) free-running closed-loop rollout, using predicted frames, predicted rewards, and predicted continuation. Never label the mixed path simply “closed-loop.” Compare both and test distribution mismatch. |
| R07 | **BLOCKER** | `true_jepa/objectives.py::imagination_policy_step` computes `_continuation` and discards it. Returns terminate at the artificial imagination horizon with zero tail bootstrap. | Death/continuation predictions have no effect on imagined returns; the critic has horizon truncation bias and can learn a self-referential target. | Use a continuation-aware lambda return with a value bootstrap at horizon. Add terminal/truncation unit tests, hand-computed target tests, and an exploding-continuation-module test proving that changing continuation changes return targets. |
| R08 | **HIGH** | The same `true_jepa/objectives.py` trains the value head in the imagination step against its own finite model-generated returns; it does not establish value calibration against held-out real experience. | A critic can agree with a biased world model and still be wrong in the real environment. | Before imagination policy training, train/validate value targets using held-out real n-step/lambda returns (or another explicitly selected real-data target). Report calibration, error by horizon, and return-distribution coverage. Imagination-only value fitting is not a promotion criterion. |
| R09 | **BLOCKER** | `SelfPlayCoordinator.__init__` constructs `world_optimizer` from **all** trainable model parameters, including policy and value heads, then also constructs separate actor and value optimizers over the same policy/value parameters. | Parameters are registered with multiple optimizers, creating ambiguous optimizer state and possible double updates when losses evolve. | Define disjoint explicit parameter groups. World optimizer includes only encoder, belief, predictor, ensemble, reward and continuation components active in that phase. Actor/critic parameters belong only to their own optimizers. Add a test proving each trainable parameter belongs to exactly one optimizer and EMA target parameters belong to none. |
| R10 | **HIGH** | `imagination_policy_step` starts from `batch["obs"][:,0]` and a zero belief, although that observation may be from the middle of an episode. | Imagined policy training starts from a false reset state. | Carry the reconstructed belief from the same burn-in/full-prefix computation used for the selected replay window. Test hidden-state equivalence to a full-episode rollout. |
| R11 | **HIGH** | `PixelReplay.load` recreates `PixelReplay()` without restoring `capacity_frames`; `save` metadata does not store capacity. `add` also leaves one oversized episode in the buffer even if it exceeds capacity. | Resume loses the memory limit; a single episode can exceed configured capacity. | Persist capacity and schema version. Define an explicit oversized-episode policy: reject with error or split at valid episode boundaries without losing boundary metadata. Test exact capacity after save/load and after a single over-capacity episode. |
| R12 | **HIGH** | `SelfPlayCoordinator.load` restores model, optimizers, counters and relation memory but not NumPy RNG state, Torch CPU/CUDA RNG state, GradScaler state, snapshot pool, or action sampling RNG state. | A resumed run is not equivalent to an uninterrupted run; mixed-policy sampling distribution changes because snapshots are lost. | Save/restore all stochastic state and data-generation counters. Initially promise **episode-boundary resume only**; do not claim exact mid-episode resume. Compare uninterrupted and resumed traces from the same checkpoint. |
| R13 | **HIGH** | In parallel collection, mixed random overrides are applied per transition, but the episode source is still recorded as `batched_self_play`; snapshots are not selected in `_collect_parallel` despite the serial collector's snapshot path. | Data provenance does not describe the behavior policy that generated each transition; serial and parallel modes implement different collection mixtures. | Record per-transition behavior source and actual executed action. Either implement snapshot sampling in parallel with identical declared probabilities or remove it from the documented policy. Compare serial and parallel collection distribution. |
| R14 | **HIGH** | Episode `seed` fields in collection paths are episode IDs or slot-derived IDs rather than always the actual integer seed passed to the environment factory. | Seed-level reproducibility and duplicate detection are unreliable. | Store both `episode_id` and `environment_seed` as separate provenance fields. Verify exact seed assignment at environment construction. These fields remain evaluator/collector metadata and are not passed to the policy. |
| R15 | **HIGH** | `true_jepa/model.py::TransitionEnsemble` members are trained against the same targets on the same examples without an explicit bootstrap or independent data assignment. | Disagreement can be small, correlated, or miscalibrated; ensemble size 3 does not establish epistemic uncertainty. | Compare single model, independent bootstrap members, and ensemble variants. Calibrate disagreement against held-out prediction errors and OOD transitions. Do not use disagreement for control until calibration passes. |
| R16 | **HIGH** | `true_jepa/agent.py::act` uses an exact `step` argument to infer reset (`step == 0`); policy API and collector semantics depend on an external counter. | Reset logic is implicit and easy to break after vector reset/resume. | Use explicit `agent.reset()` at environment reset and `agent.act(obs_uint8, reward_prev)`; the step counter is logger/runner metadata only, never a learned input. |
| R17 | **HIGH** | `true_jepa/objectives.py` ignores `truncated` in the continuation loss and treats every non-terminated transition as continuing. This can be valid for external time-limit truncation only if the final observation is bootstrapped correctly. | Current target semantics are underspecified; the actor imagination code does not perform the needed truncation bootstrap. | Define and unit-test separate terminal, time-limit truncation, and continuing transitions. Store final observation on truncation. `terminated` implies zero continuation; truncation bootstrap behavior is explicit in config and benchmark contract. |
| R18 | **MEDIUM** | Current `true_jepa/config.py` defaults (D=192, belief=256, 4 heads, 2 layers, batch=8, sequence=16, 120k replay frames, gamma=.997, EMA .996→.999) are inherited settings, not independently selected ORBIT optima. | Treating defaults as established values repeats unsupported assumptions. | Keep each as `INHERITED_CANDIDATE`; require matched sweeps/preflight before promotion. |
| R19 | **MEDIUM** | `true_jepa/self_play.py::_refresh_relations` builds anonymous relation memory, but the inspected policy/planning path does not use the relation bank to choose actions. | It is not evidence of working rule discovery or a useful planner. | Keep it diagnostic-only. Do not claim it improves behavior until an ablation demonstrates policy influence and actual held-out gain. |
| R20 | **MEDIUM** | The historic Prometheus audit recorded a GRU with no gradients despite optimizer inclusion. This is not proof that every current recurrent core has the same defect. | Passing shape tests or listing a GRU in an optimizer does not prove it learns. | Require per-module gradients, parameter deltas, controlled memory probes, and full-prefix vs reset-hidden comparisons for every recurrent implementation. |

### B.1 Required environment protocol

There must be one named canonical benchmark environment protocol, identified by the exact Crafter package/version, wrapper source hash, max episode length, reward semantics, action count, observation transform, deterministic settings, and seed procedure. A patched deterministic wrapper is a separate protocol. The baseline, candidate, and evaluator must use the same protocol. Never compare a patched and unpatched environment as if only the model changed.

The learner-facing adapter must expose exactly the minimal pixel/reward/termination stream. The evaluator may independently retain Crafter `info` to score achievements after an episode. The evaluator must call the policy through the same strict API as deployment and must not pass `info` as an argument.

## C. Mathematical corrections and fully calculated reference values

### C.1 Discount factor is not prediction-error growth

The older wording that latent error compounds as `O(gamma^H)` is mathematically wrong. **Gamma discounts future returns; it does not govern how a learned transition model's state-prediction error grows.**

A simple deterministic error-bound illustration is:


define one-step model error at step t as `epsilon_t`, and assume a transition map with Lipschitz constant `L` in the relevant region. Then an upper-bound recurrence is:

`e_(t+1) <= L * e_t + epsilon_t`.

If `epsilon_t <= epsilon` and `e_0 = 0`:

- for `L != 1`: `e_H <= epsilon * (L^H - 1) / (L - 1)`;
- for `L == 1`: `e_H <= H * epsilon`.

This is only an illustrative bound under its assumptions. Neural latent dynamics may violate a global Lipschitz bound; the actual quantity must be **measured** with teacher-forced and free-running rollouts. The `gamma` discount belongs in return equations, not in the latent-error law.

### C.2 Discount calculations — exact interpretation

Definitions:

- half-life: `ln(0.5) / ln(gamma)` transitions;
- e-folding length: `-1 / ln(gamma)` transitions;
- discounted weight at offset k: `gamma^k`;
- infinite geometric sum: `1 / (1-gamma)` for `0 <= gamma < 1`.

| gamma | half-life | e-folding length | gamma^25 | gamma^60 | gamma^90 | gamma^175 | geometric sum |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.990 | 68.9676 | 99.4992 | 0.777821 | 0.547157 | 0.404732 | 0.172250 | 100.000 |
| 0.995 | 138.2826 | 199.4996 | 0.882220 | 0.740261 | 0.636909 | 0.415948 | 200.000 |
| 0.997 | 230.7023 | 332.8331 | 0.927639 | 0.835044 | 0.763070 | 0.591089 | 333.333 |

The geometric sum is not an empirically learned planning horizon, and the half-life is not a guarantee that the agent can execute a sequence of that length. Gamma is a return-weighting parameter. The actual gamma must be selected using real return distributions, survival/achievement outcomes, and value calibration. `.997` remains an inherited candidate, not a default truth.

### C.3 Lambda-return calculations and definition

For an imagined/real transition, let `r_t` be scalar environment reward, `c_t` the continuation probability/target, and `d_t = gamma * c_t`. For an imagination horizon H, define the terminal bootstrap as `G_H^lambda = V(s_H)`. Then compute backwards:

`G_t^lambda = r_t + d_t * ((1-lambda) * V(s_(t+1)) + lambda * G_(t+1)^lambda)`.

Boundary behavior:

- if the environment truly terminates after transition t, `c_t=0`, so no reward/critic value from the nonexistent continuation leaks backward;
- if the episode is externally truncated by the evaluator/time limit and the task definition allows bootstrap, use the final observation's value for that transition;
- if the protocol defines the time limit itself as terminal, use `c_t=0` and record that choice; never infer it from reward or a guessed step number;
- `lambda=0` gives a one-step bootstrap target; `lambda=1` gives an H-step rollout bootstrapped at the horizon.

For the inherited candidate `lambda=.95`, products `gamma*lambda` and their trace decay are:

| gamma | gamma*lambda | trace half-life | trace e-fold length | (gamma*lambda)^25 | ^60 | ^90 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.990 | 0.94050 | 11.2994 | 16.3016 | 0.215760 | 0.025207 | 0.004002 |
| 0.995 | 0.94525 | 12.3104 | 17.7601 | 0.244719 | 0.034104 | 0.006298 |
| 0.997 | 0.94715 | 12.7657 | 18.4170 | 0.257317 | 0.038470 | 0.007546 |

This table illustrates trace weighting, not the total dependency length solved by a critic. It does show why a large gamma does not automatically solve a 25–90-step dependency. Real exploration, bootstrapping accuracy, policy learning, and model accuracy still matter.

`lambda=.95` is also inherited from earlier code/research; do not lock it without a return-target and calibration experiment. Add a hand-calculated unit test for at least one two-step terminal example, one continuing example, and one time-limit truncation example.

### C.4 Attention arithmetic

For N spatial tokens, one attention-score matrix has N×N entries per head and sample:

- 4×4 tokens = 16 tokens = 256 score entries;
- 8×8 tokens = 64 tokens = 4,096 score entries;
- ratio = 4,096 / 256 = **16×** pairwise score entries.

This does **not** mean the full transformer is 16× slower: projections, MLPs, memory traffic, kernels and batch size also contribute. Measure actual peak memory, throughput and latency on target hardware before choosing token grid.

### C.5 Action-space combinatorics and exploration mixture

For 17 discrete actions, the number of raw sequences is:

- `17^8 = 6,975,757,441`;
- `17^10 = 2,015,993,900,449`;
- `17^15 = 2,862,423,051,509,815,793`.

This is a scale illustration, not a claim that CEM enumerates all sequences. More candidates can still select model hallucinations; prior failures F03/F17/F18 are the empirical warning.

For uniform mixture exploration:

`p_mix(a|s) = (1-epsilon) * p_policy(a|s) + epsilon/17`.

At `epsilon=.01`, the guaranteed per-action floor is `.01/17 = 0.000588235`, i.e. **0.0588% per action**, not 1% per action. If the policy never selects a particular action, the chance that the uniform branch selects that one action at least once over 15 independent decisions is `1-(1-.01/17)^15 = 0.0087873`, i.e. approximately **0.8787%**; the chance that any uniform branch is activated at least once is `1-(1-.01)^15 = 13.9942%`. Log policy entropy, actual action histogram, uniform-branch selections, and state-conditioned action distribution separately. Do not say “1% action floor.”

### C.6 Memory, tensor, mask and representation arithmetic

| Quantity | Exact calculation | Correct interpretation |
|---|---|---|
| One RGB `uint8` frame | `64*64*3=12,288` bytes = 12 KiB | Raw storage only. |
| 120,000 unique frames | `1,474,560,000` bytes = 1.3733 GiB | Excludes indices, actions, rewards, duplicated sequence buffers, compression staging and model memory. |
| Batch 8 × 17 raw observations | `8*17*12,288=1,671,168` bytes = 1.594 MiB | Input bytes only; float32 materialization is 6.375 MiB. |
| B=8,T=16,N=64,D=192 latent activation | `8*16*64*192=1,572,864` elements | 6 MiB FP32 or 3 MiB FP16 for one such tensor, not a total GPU estimate. |
| D=192, 4 attention heads | `192/4=48` dimensions/head | Shape arithmetic only. |
| TrueJEPA belief input | `192 + 96 + 48 = 336` | Current model: pooled visual D, action embedding D/2, reward embedding D/4. |
| TrueJEPA state vector | `192 + 256 = 448` | Current model pooled D plus belief 256. |
| `floor(.35*64)` mask | 22 tokens | Actual mask fraction `22/64=34.375%`, not 35%. |
| Uniform 17-action entropy | `ln(17)=2.833213` nats = `log2(17)=4.087463` bits | Maximum entropy, not a target policy entropy. |
| EMA `.996` | e-fold 249.4997 updates; half-life 172.9400 updates | Optimizer updates, not environment frames. |
| EMA `.999` | e-fold 999.4999 updates; half-life 692.8005 updates | Optimizer updates, not environment frames. |

All numbers above are arithmetic, not selected model hyperparameters. Run a real forward/backward peak-memory test on the target device; never estimate T4 fit from one activation tensor alone.

### C.7 Crafter score units — independently checked

The repository's `eval/metrics.py` implements:

`score = exp(mean_i(log(1 + success_percent_i))) - 1`

where each of the 22 success rates is in **percentage points `[0,100]`**. For each achievement, count each episode as boolean unlocked/not-unlocked; repeated unlock events do not count multiple times. Do not multiply the result by 100 again.

Recalculation from checked-in artifacts:

- clean screen: `3.170877262054839`, matching `3.17087726205484`;
- T2 screen: `4.546214306875846`, rounding to `4.55`.

The score formula is consistent within each artifact. It does not reconcile why the per-episode records differ. Missing any of the 22 expected achievement keys is a hard error, not an implicit zero.

## D. Corrected data and model contracts

### D.1 Policy boundary

Use an explicit reset operation so reset cannot be confused with a learned feature:

```python
agent.reset()  # called by collector/evaluator exactly at each environment reset
action_id = agent.act(
    obs_uint8: np.ndarray,  # [64,64,3], uint8, RGB
    reward_prev: float,     # previous raw scalar environment reward
) -> int
```

The agent internally retains the previous action it actually returned. The collector must record the action actually executed after any exploration override; it must feed that executed action, not the proposed action, to the agent's next belief update. The policy must not receive `info`, inventory, achievements, semantic map, coordinates, exact environment step, seed, episode ID, or environment object. Reset is an explicit method call, not a policy tensor feature.

The existing 17 native action IDs can be treated as opaque categorical tokens. An independently seeded random bijection may be used as a **separate robustness diagnostic**, but is not mandatory for the first implementation because it changes action-token identities and increases experiment state. Do not hardcode their names or meanings. If a permutation is used, store and hash it and keep it fixed for the whole trained checkpoint lineage.

### D.2 Transition and sequence schema

For one episode with T environment transitions:

- observations: `obs[0..T]`, shape `[T+1,64,64,3]`, `uint8`;
- actions: `actions[0..T-1]`, shape `[T]`, integer opaque/native action IDs;
- rewards: `rewards[0..T-1]`, shape `[T]`, raw `float32` environment reward;
- terminated: `[T]`, true only for genuine environment terminal state;
- truncated: `[T]`, true only for an externally imposed episode cutoff;
- episode ID, actual environment seed, step index, behavior source, executed action, protocol ID and provenance hashes: collector/evaluator metadata only;
- no achievement names/booleans, inventory, coordinates, semantics, terrain internals or hidden state in learner replay.

For a batch of B sequences and T sampled transitions:

| Field | Required shape |
|---|---|
| observations | `[B,T+1,64,64,3]` |
| actions, rewards, terminated, truncated, valid | `[B,T]` |
| target spatial latents | `[B,T,N,D]` after action/observation alignment is verified |
| belief state | `[B,T+1,H_b]` with the exact before/after convention documented |
| burn-in prefix | variable per row or a separately padded/masked prefix, with valid mask |

Never use an unqualified variable `sequence_length`. Call it `sequence_transitions`; derive observation count as `T+1`.

### D.3 Recurrent state and the two distinct training modes

For real interaction, define:

`z_t = E_online(o_t)`

`h_t = B(h_(t-1), z_t, a_(t-1), r_(t-1))`

`z_hat_(t+1) = F(z_t, h_t, a_t)`

At episode start, `h_-1=0`, previous reward is 0, and previous action is a dedicated START token or a separately tested initial-state convention. The chosen convention must be in the model's action-embedding schema and checkpoint manifest.

For teacher-forced replay training, reconstruct belief from **real** preceding observations/actions/rewards and use real `z_t` as the current prediction input at each supervised step. This tests one-step predictive correctness under the real observation distribution.

For closed-loop model rollout, after the first real context, recursively feed **predicted** `z_hat`, predicted reward `r_hat`, predicted continuation, and the chosen next action into the next model step. This tests free-running imagination. Report this separately from teacher-forced prediction.

Do not silently mix predicted visual state with real reward and label the result “fully closed-loop.” Such a mixed rollout can be a separately named ablation, but it is not the default.

**Burn-in algorithm for the first correct implementation:** sample a sequence window plus its entire prefix from the beginning of that same episode; unroll the belief core over the prefix under current parameters in `no_grad`, detach the final belief, then train on the requested window. This is exact with respect to the recurrent computation (assuming identical model/eval settings). It can be slower for long episodes, but correctness comes first. Only after logging the difference between full-prefix belief and fixed-length burn-in belief may the team reduce prefix length. The burn-in length is therefore not guessed in advance.

Add tests:

1. A window sampled at step k must reconstruct the same hidden state as a reference forward pass over observations `0..k` within deterministic numerical tolerance.
2. Resetting hidden at a mid-episode sample must demonstrably change the state and should fail the test.
3. No sequence may cross an episode boundary.
4. Teacher-forced and closed-loop training traces must show their respective inputs at every step.
5. The online actor's belief update and the imagination belief update must use the correct real-vs-predicted inputs.

### D.4 Optimizer and gradient ownership contract

Each trainable parameter must belong to exactly one active optimizer. Target encoder parameters belong to no optimizer and have `requires_grad=False`.

During the world-model phase, the explicit world optimizer may include only the modules whose losses are active: online encoder, belief core, action-conditioned predictor, any validated ensemble, reward head, and continuation head. Policy and value parameters are excluded until their phase begins. During actor-critic training, actor and critic/value optimizers must be disjoint from each other and from the world-model optimizer unless a deliberate shared-backbone gradient experiment is declared with its own test. No optimizer overlap by accident.

For every optimizer step, report per-module:

- trainable parameter count;
- number of parameters with `grad is not None`;
- gradient L2 norm and max absolute gradient;
- number of finite/non-finite gradients;
- parameters whose values changed;
- optimizer membership count;
- optimizer step count.

Fail the update if a required module has no gradients, non-finite gradients, or no parameter change over the controlled learning test. A zero gradient can be mathematically expected for a target/unused module; that status must be explicit, not hidden.

### D.5 Replay and checkpoint correctness

Replay must persist: schema version, capacity frames/transitions, observation layout, episode offsets/lengths, action/reward/termination arrays, actual environment seeds, episode IDs, transition source metadata, and exact hashes. A replay loaded from checkpoint must restore the same capacity and sample semantics. An episode larger than capacity must trigger a documented result (reject or explicitly segment) and never silently exceed memory budget.

A training checkpoint at an episode boundary must persist and restore:

- online model and target encoder;
- every optimizer and scheduler state;
- AMP GradScaler state if enabled;
- replay shards/index/manifest and replay sampler RNG;
- Python RNG, NumPy RNG, Torch CPU RNG, and every CUDA RNG state;
- action sampling RNG, source-mixture state and snapshot pool/versions;
- optimizer update count, environment transition count, environment seed counter, episode IDs and config;
- normalizers/calibrators if later introduced;
- hashes of code, config, environment, replay shards and action map.

For the first implementation, support checkpoint/resume **at episode boundaries only**. Mid-episode environment-state serialization is a separate feature requiring a deterministic snapshot test. A resumed run is called “exact” only if uninterrupted and resumed runs produce identical action/reward/observation traces for a deterministic test fixture under the same environment protocol. Otherwise report the exact source of divergence and call it resumable, not exact.

## E. Actor, critic, imagination and termination — required equations and tests

### E.1 Return target

For every real or imagined transition:

`d_t = gamma * c_t`

`G_H = V(s_H)`

`G_t = r_t + d_t * ((1-lambda) * V(s_(t+1)) + lambda * G_(t+1))`.

For a true terminal, continuation is zero. For time-limit truncation, use the protocol-defined bootstrap rule and final observation. `r_t` remains raw environment reward for the extrinsic critic. Any intrinsic signal, if later studied, is a separately logged training objective and must not silently replace the official extrinsic return.

### E.2 Required critic evaluation before imagination policy updates

- Real-data n-step/lambda targets must be formed with correct terminal/truncation boundaries.
- Compare scalar critic and distributional critic only if justified by empirical target distribution. The old scalar 1-step TD failure (prediction std 0.0157 vs return std about 1.018) is a regression test, not proof that a distributional critic is automatically calibrated.
- Report predicted-value mean/std, target mean/std, calibration by target quantile, explained variance, absolute error, calibration curve and error by episode age/survival phase.
- Constant-mean and persistence-value baselines must be reported.
- The value head must not train solely against its own imagined rewards.

### E.3 Imagination gate

Do not train the actor in imagination until:

1. the chosen prediction horizon has a held-out closed-loop error advantage over persistence;
2. the next-state prediction is action-sensitive (shuffled-action control is worse);
3. reward and continuation heads beat their baselines and are calibrated well enough for the intended decision;
4. hidden-state burn-in/reset correctness passes;
5. the value critic is calibrated on held-out real data;
6. replacing the predictor, reward head, or continuation head with an exploding module makes the imagined decision path fail at the expected call site;
7. replacing the model with a persistence/constant baseline measurably changes the decision outcome.

After the gate, compare no-imagination policy, short-horizon imagination, and the selected actor baseline at equal real environment steps. An increase in imagined return without real held-out improvement is a failed experiment.

## F. Binding phase plan and exit criteria

The v1.0 phases are retained as a high-level research map, but the following ordered gates are mandatory and take priority. Later research tracks are diagnostic-only until promoted.

### Gate 0 — Freeze and reproduce the environment/evaluator baseline

**Deliverables:** exact environment protocol, score evaluator, source hashes, result provenance report, immutable baseline checkpoint or an explicit `NO_VERIFIED_BASELINE` status.

**Tests:**
- run `eval/metrics.py` score formula against the two committed JSON artifacts;
- independently rebuild the 22 achievement rates from per-episode booleans and compare them to stored rates;
- same-checkpoint/same-seed rerun with checkpoint SHA-256 and complete manifest;
- test deterministic repeatability across repeated episodes and identify any nondeterminism;
- test max-step termination and truncation at 1, configured maximum;
- compare canonical unmodified environment with any proposed deterministic patch on paired seeds and report behavioral parity, not just same-seed determinism;
- make evaluator policy call through the strict API without `info`.

**Pass:** one reproducible baseline with exact provenance. If not, ORBIT model work may proceed only on synthetic/unit tests; no gameplay claim or promotion baseline is allowed.

### Gate 1 — Anti-cheat and environment interface

**Deliverables:** pixel-only environment adapter, explicit reset API, action count discovery, evaluator isolation, policy input allowlist.

**Tests:** poison all fields in `info`; change inventory/coordinates/achievement metadata while holding pixels and reward fixed; model inputs/logits/actions must remain bitwise identical within deterministic execution. Static audit checks learner imports and API signatures. The policy receives no step counter, seed, episode ID or environment object.

**Pass:** zero privileged-data path; all learner input provenance is allowlisted.

### Gate 2 — Data/replay/sequence/checkpoint integrity

**Deliverables:** versioned replay, prefix-aware sampler, sample trace, checkpoint manifest, exact episode-boundary resume.

**Tests:** frame/action/reward alignment; episode boundaries; terminal frame retained; truncation final frame retained; capacity restoration; oversized episode behavior; ring wrap; save/load parity; all RNG restoration; snapshot/source-mixture restoration; uninterrupted-vs-resumed trace parity.

**Pass:** every sampled transition has an auditable predecessor/successor, episode and seed; resume is deterministic on fixture or explicitly limited.

### Gate 3 — Representation health and gradient ownership

**Deliverables:** selected spatial encoder, online/EMA target, representation diagnostics, exact optimizer membership, and the matched frozen DINOv2 ViT-S/14 diagnostic report.

**Tests:** shape and dtype; target freeze/eval; EMA scalar arithmetic; target update timing; per-module gradient/parameter-delta report; variance/effective-rank/collapse; online-target drift; held-out visual/predictive probes; AMP vs FP32 parity before enabling mixed precision; DINOv2 frozen-weight hash/gradient/optimizer-membership checks; preprocessing full-frame/no-crop test; 16×16 patch-grid and adapter alignment test; matched scratch-versus-DINOv2 downstream configuration audit.

**Pass:** no collapse, target update exactly matches formula, every intended trainable component demonstrably learns, optimizer membership is disjoint, and the DINOv2 diagnostic has a valid matched report (positive or negative). If the scratch core passes but the DINOv2 comparison cannot be run, report the core as `PASS` and the overall Gate 3 as `PARTIAL/BLOCKED`; do not finalize encoder selection or claim the comparison complete. A negative DINOv2 result is valid; it is not a reason to replace the primary scratch track.

### Gate 4 — Action-conditioned one-step prediction

**Deliverables:** minimal JEPA predictor and correctly normalized loss; persistence control; shuffled-action control; per-token residual visualizations.

**Tests:** manually calculated synthetic masked-loss fixture; target stop-gradient; action permutation consistency if used; action shuffle must worsen held-out error; model must beat persistence using paired episode bootstrap; all prediction outputs finite; exploding predictor must fail at the call site; loss per token and per scene type reported.

**Pass:** action-conditioned model beats persistence on held-out sequences with a confidence interval supporting the result. If not, stop and debug representation/data/predictor before policy work.

### Gate 5 — Recurrent belief, teacher-forced and closed-loop multi-step fidelity

**Deliverables:** full-prefix belief reconstruction, distinct teacher-forced and free-running rollout code paths, horizon diagnostics.

**Tests:** horizons 1,2,4,8 and only extend to 16 if prior horizons justify; compare full-prefix vs bounded burn-in; compare model vs persistence at each horizon; action-shuffle control; prediction norm/calibration; error by episode age and day/night phase using evaluator-only analysis; no horizon is selected because it is convenient.

**Pass:** only the empirically validated horizons are allowed for planning. A horizon failing the gate is disabled rather than silently used.

### Gate 6 — Reward, continuation and real-data value calibration

**Deliverables:** correct reward/continuation targets, terminal/truncation semantics, real-data value target and critic calibration.

**Tests:** terminal means zero continuation; external truncation follows explicit bootstrap config; reward prediction beats constant baseline; continuation precision/recall, PR-AUC, Brier score and calibration beat prevalence-only baseline; hand-calculated lambda-return fixtures; value calibration against held-out real returns; continuation exploding-module sensitivity.

**Pass:** reward/continuation/value inputs are fit for the selected control objective. If continuation is not reliable, no long imagination is permitted.

### Gate 7 — Real-experience policy baseline (no imagination)

**Deliverables:** one chosen policy-learning baseline with exact algorithm, loss, optimizer, data source and hyperparameters documented. This is a research decision; do not invent PPO/AWR/actor-critic settings in code without a measured comparison.

**Tests:** policy input contract; action histogram/entropy; executed-vs-proposed action trace; reset behavior; train/eval mode; policy and critic parameter changes; no `do`/single-action collapse; comparison to random and frozen policy at equal environment steps.

**Pass:** a stable policy control that makes the later imagination ablation meaningful. If it does not learn, stop and debug data/algorithm rather than adding curiosity or skills.

### Gate 8 — Short-horizon imagination policy

**Deliverables:** continuation-aware lambda-return, value bootstrap, validated rollout length, separate actor/critic/world optimizers.

**Tests:** hand-calculated return target; terminal and truncation behavior; return sensitivity to continuation; value bootstrap sensitivity; imagined-vs-real action distribution and return gap; exploding-module tests; no-imagination vs imagination equal-budget ablation; survival and all achievement metrics.

**Pass:** statistically supported real held-out improvement without critical regressions. Otherwise revert the imagination path and keep world model as prediction-only.

### Gate 9 — Research extensions, one at a time

Order is not predetermined; choose one extension only after a written research decision:
- object-centric/slot memory;
- topological visual place memory and relocalization;
- active rule discovery/information gain;
- reusable skills/options;
- autonomous curriculum selection.

For each extension: one primary hypothesis, one matched control, one preregistered metric, one failure bundle, one ablation removing the extension, and no changes to other modules during the comparison. It remains disabled until it passes. None may use hidden inventory, map, coordinates, achievement data or hardcoded rules.

### Gate 10 — Locked benchmark and final promotion

Use an immutable evaluation protocol and locked held-out seeds. Run multiple independent training seeds; use a pilot to estimate variance and determine final sample count/power. The old “five seeds when compute permits” is a practical reporting target, not an automatic statistical guarantee. Report every training run, all evaluation episodes, 22 achievement rates, composite score, confidence intervals, IQM/performance profile where applicable, interaction budget, wall-clock, hardware and artifact hashes. No best-seed selection.

## G. Full debugging and forensic system — mandatory, not optional telemetry

The debug system must let an engineer answer, with artifacts rather than guesses: **what did the agent see; what state did each module compute; what action did it propose; what action was actually executed; what did it predict; what happened; which loss/gradient/update changed the responsible module; and what is the earliest point where prediction and reality diverged?**

### G.1 IDs and lineage

Every event carries stable IDs:

`run_id, code_commit, config_hash, environment_protocol_id, training_seed, episode_id, environment_seed, step_id, decision_id, transition_id, replay_shard_id, replay_sample_id, batch_id, optimizer_update_id, checkpoint_id`.

No ID may be reused. The system must trace a final evaluation episode back to the exact checkpoint/config/environment and trace a training gradient back to the original source transition(s).

### G.2 Per-decision trace schema

At minimum capture:

- input observation hash, shape, dtype and a viewable image artifact;
- previous executed action and previous raw reward;
- reset flag and recurrent-state hash/norm;
- per-module latent tensor shape, mean/std, min/max, norm, finite count, effective rank estimate, and optional compressed tensor dump;
- per-action logits/probabilities, entropy, top-k actions, stochastic sample and RNG-state reference;
- proposed action ID, exploration override flag/source, final executed action ID;
- predicted next-frame/latent hash and summary, predicted reward, predicted continuation, predicted value, uncertainty if active;
- actual next observation hash, actual reward, terminated/truncated, terminal cause from evaluator-only record;
- per-step prediction residual at spatial-token level and pooled level;
- all active goal/skill/object-memory IDs, confidence and evidence provenance, if those modules are enabled;
- exceptions, NaN/Inf flags, clipping, fallback decisions and timing per module.

The learner-visible trace must not contain hidden fields. Evaluator-only metadata is written to a separate file with a separate reader and permissions. Reports may join the streams after the episode for analysis; never join them before action selection or training.

### G.3 Training/update trace schema

Every optimizer update records:

- exact replay IDs, episode IDs, start indices, burn-in prefix lengths and valid masks;
- model mode (teacher-forced, mixed ablation, or closed-loop);
- each loss numerator, denominator and final value (not only a summed scalar);
- per-loss/per-module gradient norms and gradient coverage;
- parameter delta norms and optimizer state finite checks;
- EMA decay value and online-target distance before/after;
- batch action distribution, reward distribution, termination/truncation counts;
- AMP scale, skipped-step flag, gradient clipping norm before/after;
- update wall time, throughput, peak host/GPU memory.

### G.4 Exploding-module and intervention tests

For every module that is claimed to influence a decision, run a test where it is replaced by a module that raises a unique exception. The expected call site must raise. If execution continues, that module is not connected to that decision path.

Then run a controlled perturbation test: replace output with zeros, permute output, or perturb one feature within a bounded range; confirm downstream score/logits/decision changes in the expected location. Capture before/after traces. A model that trains but has no decision influence must be labelled auxiliary/unused, not a planner component.

Required tests include encoder, belief core, transition predictor, reward, continuation, value, actor, uncertainty ensemble, object memory and each future research module when activated.

### G.5 Gradient and parameter ownership audit

At initialization and each new phase:

- enumerate all named parameters and optimizer membership;
- fail if a trainable parameter is in zero or multiple active optimizers, unless explicitly declared frozen/auxiliary;
- verify expected gradients for every active module on a deterministic synthetic batch;
- verify target encoder has no gradients;
- when Candidate E is active, verify the DINOv2 backbone has no gradients, no optimizer membership and an unchanged checkpoint hash;
- verify per-module parameter deltas after a controlled update;
- check for stale optimizer state after parameter-freeze or unfreeze;
- report total trainable/frozen parameter counts.

A module can intentionally have no gradients only when the phase spec says so (for example, target encoder or a disabled head). This is not an error if reported correctly.

### G.6 Prediction inspection and episode microscope

Build a local viewer that can display, step-by-step:

1. actual input frame;
2. predicted next spatial tokens decoded/visualized by a diagnostic-only renderer (renderer is not learner input);
3. actual next frame;
4. per-token error heatmap;
5. selected action probability distribution;
6. predicted vs actual reward/continuation/value;
7. recurrent memory summary across the episode;
8. event timeline from evaluator-only annotations, visibly marked as evaluator-only.

The viewer must support jumping from a bad episode metric to its exact decision trace and replay sample. Preserve the original evidence; never overwrite a failing trace with a later rerun.

### G.7 Earliest-divergence debugging algorithm

When a failure occurs, do not start by changing hyperparameters.

1. Reproduce from exact trace/checkpoint/config/environment hash.
2. Verify raw observation, executed action, reward and boundary alignment.
3. Verify replay sample indices and episode continuity.
4. Recompute current latent, target latent and hidden state from stored raw frames.
5. Compare model next-state prediction against actual next state, token by token.
6. Check reward/continuation targets and model outputs.
7. Check gradient path, optimizer membership and parameter delta for the implicated module.
8. Compare the same decision with persistence, shuffled action and ablated-module controls.
9. Find the earliest transition/module where actual and predicted states diverge beyond the preregistered diagnostic threshold.
10. Write a root-cause report with evidence, competing hypotheses falsified, minimal reproduction test and regression test.

Do not say “the encoder is wrong” or “the planner is confused” without showing which tensor/transition first becomes inconsistent.

### G.8 Failure bundle format

Every gate failure produces an immutable bundle:

```text
failure_<run>_<episode>_<step>/
  manifest.json
  environment_protocol.json
  config.json
  code_commit.txt
  checkpoint_hashes.json
  rng_states/
  observation_frames/
  learner_trace.jsonl
  evaluator_trace.jsonl       # separate provenance and access path
  replay_sample.npz
  predictions.npz
  gradients.json
  parameter_deltas.json
  module_ablation.json
  reproduction.sh
  root_cause.md
  regression_test.py
```

The failure bundle must be sufficient for another engineer to reproduce the same decision without access to hidden user state or undocumented files. If exact reproduction cannot be obtained, explicitly identify the nondeterministic component.

## H. Mandatory regression matrix for all 40 historical failures

All 40 retrospective failure modes must be represented as named tests or documented “not applicable” decisions. No failure may disappear from the test suite merely because the new architecture is different.

| Historical ID | Regression test/invariant required |
|---|---|
| F01 | Uncertainty/novelty alone must not drive uncontrolled wandering; compare reward-grounded control and random baseline. |
| F02 | AST/static audit plus policy-trace proof that no scripted survival/reflex branch selects actions. |
| F03 | Candidate-count/horizon sweep must be controlled; no claim that more candidates improves behavior. |
| F04 | Critic variance and calibration compared with real return distribution and constant-mean baseline. |
| F05 | Replay coverage report for all data strata; zero-example targets must be reported as unsupported, not inferred. No achievement labels used to balance training. |
| F06 | Recurrent history probe for motion/occlusion/night; frame-only ablation compared to recurrence. |
| F07 | State-change head must beat persistence on held-out unique transitions, not just training loss. |
| F08 | Rare-change/label-count sensitivity; compare against persistence and report independent episode support. |
| F09 | Encoder-freeze/unfreeze drift test; downstream-head agreement and full recalibration required before unfreezing. |
| F10 | Exploding transition module must be called by every path that claims to use it. |
| F11 | Preserve raw reward/milestone scale; test that normalization does not erase sparse positive events. |
| F12 | Event/decision threshold selected from validation calibration; never inherit 0.5 by habit. |
| F13 | Curiosity must not be raw displacement-from-start; compare information gain to prediction error reduction and task progress. |
| F14 | Replay provenance and historical data retention audit; prevent forgetting by checking old behavior regression after new data. |
| F15 | Closed-loop prediction fidelity must gate every horizon extension; H=15 is not permitted by default. |
| F16 | Same as F15 for long horizons; planner cannot use an unvalidated H=20. |
| F17 | CEM/refinement iterations tested for model exploitation; more iterations not an assumed improvement. |
| F18 | Candidate count tested for model exploitation; more candidates not an assumed improvement. |
| F19 | Rare singleton evidence must not be silently dropped by clustering/min-support filters; all filtering counts reported. |
| F20 | Empty-precondition hypotheses cannot become action permissions; evidence must include context and contradictions. |
| F21 | Visual localization must re-anchor after collisions/occlusions; dead reckoning alone is insufficient. |
| F22 | Multi-stage plan readiness checked against the current active subgoal, not only final destination. |
| F23 | Confidence/quarantine logic must credit valid observed outcomes and avoid confidence deadlock; simulate boundary cases. |
| F24 | No unconditional interaction bonus; test on empty space and valid object-contact contexts. |
| F25 | Separate inventory-like visual factors from vitals if causal analysis is attempted; prohibit global correlational DAG without controls. |
| F26 | Failed transition cannot falsify a rule unless the context is comparable and prerequisites are empirically supported. |
| F27 | Discovery/unlock event is not proof that a persistent workstation/object is currently nearby. |
| F28 | AST scan and provenance audit reject synthetic recipe/material patches and human-authored semantic dependencies. |
| F29 | Skill/actor action-distribution test catches `do` or any single action dominating without movement/task progress. |
| F30 | Object/goal memory deduplication; monitor duplicate occupancy and effective distinct-memory count. |
| F31 | Reachability/object classifier evaluated with hard negatives from same episode and matched contexts; prevent split leakage. |
| F32 | Memory eviction audit reports per-category retention; no category silently monopolizes capacity. |
| F33 | Every data filter logs counts before/after and reason; compare retained learning signal to unfiltered controls. |
| F34 | Goal/skill selection entropy and max selection share; detect monopoly and enforce exploration only through validated learned uncertainty. |
| F35 | Resource-budget inference must be learned from data; no hard-coded table, required quantities, or recursive recipe sum. |
| F36 | Arrival/localization threshold derived from validation distribution and robust to scene variation; no fixed cosine assumption. |
| F37 | Resource/skill priorities do not disappear after a single event unless learned evidence supports completion. |
| F38 | Survival-risk calibration measured with proper scoring and recall at lethal events; threshold cannot be inherited blindly. |
| F39 | Sleep/survival behavior must be learned from observed outcomes; no hardcoded sleep-at-night or never-sleep rule. |
| F40 | Entropy/action diversity monitored alongside reward and progress; detect reward-driven one-action collapse and prove policy can recover. |

## I. Evaluation and statistical decision rules

### I.1 Required separation of randomness

Keep training seeds, environment seeds, replay-sampling RNG, action-sampling RNG and evaluator seeds separate and recorded. Paired environment seeds are useful for comparing two agents, but do not make different training runs statistically independent. Use the same locked evaluation seeds for both sides of a paired comparison.

### I.2 Required intervals and metrics

For the final score and per-achievement outcomes, report 95% confidence intervals. Use paired episode bootstrap for comparisons on shared seeds. When multiple independent training seeds are available, use hierarchical resampling (sample training runs, then evaluation episodes within each run). Report median/mean and IQM/performance profiles as appropriate. Include all individual run results.

For a binary achievement rate p, the rough independent-episode sample-size planning formula for a 95% normal-approximation margin of error m is:

`n ≈ 1.96^2 * p * (1-p) / m^2`.

At the worst-case `p=.5`, this is approximately 384.16 (round up to **385**) episodes for ±5 percentage points, 2,401 episodes for ±2 points, or 9,604 episodes for ±1 point. These are per-rate planning illustrations, not guarantees for the nonlinear 22-achievement geometric mean; use Wilson intervals for per-achievement rates and paired bootstrap/power analysis for the composite score. With small n, use an exact/binomial method rather than trusting a normal approximation.

Do not claim a 50-episode run precisely estimates rare achievements. For example, one additional success in 50 episodes changes the empirical rate by 2 percentage points. The primary model-selection delta and variance must be estimated from development pilots; then pre-register the required sample size before opening the locked test set.

### I.3 Baseline lock and artifact manifest

Every reported result must include:

- repository commit SHA and working-tree dirty flag;
- checkpoint SHA-256 (not only file path/name);
- config file and canonical config hash;
- environment package/version and wrapper source hash;
- action-map/permutation hash if used;
- evaluator source hash and exact score formula version;
- calibration/normalizer/memory hashes if any;
- train seed(s), evaluation seed list, max steps and actual environment transitions;
- deterministic/stochastic policy mode and action sampling seed;
- per-episode return, length, terminal/truncated status and achievement booleans;
- 22 per-achievement rates and independently recomputed composite score;
- 95% intervals, paired comparison method and all exclusions (ideally none).

No evaluation result can be promoted when any required hash is missing.

## J. Instructions to the implementation AI — stricter execution contract

For every phase, the IDE AI must return these artifacts before moving forward:

1. **Pre-implementation design:** files to change, public APIs, tensor shapes, state alignment, equations, gradient paths, optimizer membership, data provenance and failure modes.
2. **Decision register:** every ambiguity recorded as `DECISION_REQUIRED`, with available evidence, alternatives, cost/risk, falsification test and one recommended option. Stop dependent implementation until user/research owner approval.
3. **Tests written first:** unit, numerical, boundary, gradient, integration, leakage, deterministic replay and failure regression tests relevant to that phase.
4. **Minimal implementation:** no adjacent refactor, no extra architecture module, no unapproved dependency or config default.
5. **Observed results:** exact command, stdout/stderr, exit status, test count, failed tests and logs. Never summarize an unrun test as passing.
6. **Trace evidence:** one deterministic smoke transition and one end-to-end episode with inspectable decision/transition traces.
7. **Phase report:** before/after file list, diff summary, parameter ledger, data manifest, hashes, telemetry, failure bundle and explicit `PASS`, `FAIL` or `BLOCKED` gate result.

The implementation AI may research and recommend alternatives, but it may not resolve unknowns by guessing. Do not ask the model to write all phases in one large change. Implement one gate, verify it, then continue.

## K. Updated first implementation task

The first PR is **not** a new encoder or actor. It must only:

1. record the exact repository commit and allowed file scope;
2. audit the existing `true_jepa/`, `envs/`, `eval/`, `results/` files listed in this amendment;
3. fix/replace the evaluation callback so policy cannot receive `info`;
4. implement canonical environment protocol and test max-step/truncation semantics without global import-time monkey-patching;
5. fix result manifest and independently recompute the 3.1709% and 4.5462% artifact scores;
6. add strict agent reset/action API tests;
7. implement replay capacity persistence and full-prefix recurrent window sampling;
8. implement exact episode-boundary RNG/checkpoint restoration tests;
9. add module/optimizer ownership, gradient coverage and exploding-module test harness;
10. output the Gate 0–2 report.

Do not train a new model until Gate 0, Gate 1 and Gate 2 pass. Do not claim that a baseline is fixed until the checkpoint hash and episode-level results reproduce.

## L. Frozen pretrained DINOv2 diagnostic encoder (v1.2 addition)

### L.1 Purpose and status

Add one pretrained visual backbone to test whether the visual representation is a bottleneck in the current/scratch ORBIT path. The selected diagnostic is **DINOv2 ViT-S/14 (without register tokens)**, developed by Meta AI. “Meta” and “DINOv2” are not two separate encoder choices: DINOv2 is Meta AI’s model family.

This is a **diagnostic comparison, not the primary ORBIT model**. Only the visual backbone is pretrained; the spatial adapter and all ORBIT temporal belief, action-conditioned dynamics, reward/continuation prediction and later policy learning remain trained using the permitted Crafter interaction data. The primary tabula-rasa result continues to use a visual encoder trained from scratch. DINOv2 results must be reported in a separate `PRETRAINED_DIAGNOSTIC` track and must never be blended into the primary score, training curve, or claims of from-scratch learning.

Official model facts to pin in the experiment manifest: ViT-S has 21M parameters, 384-dimensional embeddings, and a patch size of 14; the backbone returns a class token and spatial patch tokens. The DINOv2 model card reports pretraining on LVD-142M (142 million images), identifies Meta AI as developer, and lists Apache License 2.0. The official model card also states that input dimensions must be multiples of 14; otherwise the model crops to the closest smaller multiple. Sources: https://github.com/facebookresearch/dinov2, https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md and https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py. Recheck the exact model card, checkpoint and license at implementation time and record their commit/hash.

### L.2 Input preprocessing and spatial-token contract

Crafter observations are `64×64×3` RGB. A 64-pixel side is not divisible by 14, so passing the raw frame directly would allow DINOv2 to crop the frame silently. That is prohibited. The initial diagnostic preprocessing is:

1. Preserve the complete `64×64` field of view and deterministically resize each frame to `224×224` (16×16 patch positions). Do not center-crop, random-crop, or pad away any part of the observation.
2. Use nearest-neighbor interpolation for the first pixel-art-preserving diagnostic. Record the exact implementation/version and output hash. A different interpolation method is a separately named preprocessing ablation, not an unlogged change.
3. Convert RGB `uint8` to float in `[0,1]`, then use the official DINOv2 ImageNet normalization constants: mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. Record channel order, scaling, interpolation and normalization in the run manifest.
4. Use the 256 spatial patch tokens and preserve their 16×16 row/column arrangement. Do not use the class token as a spatial position. The selected non-register model has no register tokens; if a register variant is ever tested, its register tokens must not be reshaped into the spatial grid.
5. Add an explicit, tested adapter that maps the 16×16 DINO patch grid and 384 feature dimensions to the common spatial grid and token dimension selected for the matched ORBIT comparison. The adapter may pool/resample spatially and project channels, but its exact operations and parameter count must be recorded. No hidden crop or semantic/object mapping is allowed.

### L.3 Training and gradient ownership

- Freeze every DINOv2 backbone parameter and keep the backbone in evaluation mode for the initial experiment. No DINOv2 fine-tuning is allowed in this first diagnostic.
- The DINOv2 checkpoint hash must be identical before and after training; every backbone parameter must have `requires_grad=False`, no gradient, and no optimizer membership. Add a test that fails if any DINOv2 parameter changes.
- The spatial adapter, recurrent belief, action-conditioned predictor, reward head and continuation head follow the selected ORBIT training contract. If the adapter participates in the online/target pathway, give it explicit online/EMA-target ownership and test the update rule.
- Keep the same downstream belief/predictor/outcome-head architecture, losses, replay samples, sequence boundaries, training transitions and optimizer-update budget as the matched from-scratch arm. Record adapter and downstream trainable parameter counts, total parameter count, throughput, wall-clock and peak memory. Equal interaction exposure does not imply equal compute; disclose both.
- Fine-tuning DINOv2, trying a larger DINOv2 variant, changing image interpolation, or testing another pretrained family requires a separate preregistered experiment and must not be folded into this first diagnostic.

### L.4 Matched comparison and decision rules

The key comparison is not “DINOv2 score versus an unrelated historical checkpoint.” It is a controlled encoder substitution:

- **Scratch arm:** the selected clean-room ORBIT encoder trained from random initialization.
- **Pretrained diagnostic arm:** frozen DINOv2 ViT-S/14 plus its documented spatial adapter.
- Both arms receive the same raw episodes, episode/seed split, action/reward stream, replay samples, sequence lengths, downstream belief and predictor, losses, number of environment transitions, optimizer updates and evaluation protocol. Use paired seeds and retain per-seed results. The only intentional representation-side difference is encoder source/pretraining plus the necessary explicitly documented input transform/adapter.

Report, separately for each arm: representation collapse/variance/effective rank; held-out one-step prediction relative to that arm’s own persistence baseline; shuffled-action sensitivity; teacher-forced and free-running error by horizon; reward/continuation prediction quality; downstream real-environment performance when the later policy gate is reached; and compute/throughput. Do not compare raw latent MSE across different representation spaces as if the scales were automatically commensurate.

Interpretation:

- If frozen DINOv2 improves held-out spatial diagnostics **and** action-conditioned prediction under the matched downstream model, this is evidence that the scratch/legacy visual representation may be a bottleneck. It does not prove that the encoder is the only bottleneck.
- If DINOv2 improves static/spatial probes but not action-conditioned prediction, investigate temporal belief, action conditioning, predictor design, training targets and pixel preprocessing.
- If DINOv2 and scratch both fail, investigate data alignment, preprocessing, predictor capacity/objective and measurement validity; do not conclude that the scratch encoder is therefore good.
- If DINOv2 fails, domain mismatch (natural-image pretraining versus Crafter pixel art), patch granularity, resize artifacts or adapter design may be responsible. A negative result does not certify the scratch encoder.
- A claim that one encoder improves gameplay requires later held-out real-environment results under the same policy-learning and evaluation budgets, not representation loss alone.

### L.5 Gate placement and artifact naming

Run this diagnostic only after Gates 0–2 pass and alongside Gate 3 / Phase 3 representation qualification. It does not change the first PR and does not delay the environment/replay/evaluator repair work. The DINOv2 arm can inform encoder selection, but the primary tabula-rasa ORBIT model must remain independently specified and reported.

Required artifacts: model/checkpoint and license manifest; preprocessing manifest and pixel test; frozen-backbone hash/gradient audit; adapter tensor-shape and parameter ledger; matched-seed manifest; prediction/persistence/shuffled-action metrics; compute profile; and a concise interpretation report with `SUPPORTS_ENCODER_BOTTLENECK`, `INCONCLUSIVE`, or `DOES_NOT_SUPPORT_ENCODER_BOTTLENECK` (never a categorical “encoder proven good/bad”).

---

**End of v1.2 binding amendment.** The original v1.0 specification remains below as background and research detail, subject to every superseding instruction above.


---

## 0. Executive decision

Do not start by implementing every component described in the earlier ORBIT concept. That proposal was too broad and promoted memory, causal graphs, skills and planning before their value had been established.

The corrected approach is:

1. Establish a trustworthy, reproducible Crafter baseline and resolve existing evaluation discrepancies.
2. Compare spatial world-model designs using identical data, interaction budgets and evaluation protocols. Include an EMERALD reproduction/reference, the strongest existing repository model, a clean ORBIT candidate, and a separate frozen DINOv2 diagnostic arm. The DINOv2 arm is not part of the primary tabula-rasa result.
3. Build and validate one action-conditioned spatial world-model core.
4. Research the four directions as separate, controlled tracks: **object-based world modelling, active rule discovery, reusable skill discovery, and autonomous curriculum generation**.
5. Integrate a direction only if its ablation demonstrates a measurable improvement.
6. Introduce imagination-trained policy learning only after closed-loop model prediction is demonstrably useful.
7. Promote models only with immutable checkpoints, reproducible evaluations and statistical uncertainty reports.

**No target score is promised.** EMERALD's published Crafter result is evidence that spatial latent world models are a promising direction, not evidence that ORBIT will reproduce that result. EMERALD reports 58.1% in its paper at a 10M-environment-step budget; that budget and experimental setup must not be presented as equivalent to our earlier 390k-step v2 result or a 1M-step run. See [EMERALD paper](https://proceedings.mlr.press/v267/burchi25a.html) and its [official repository](https://github.com/burchim/EMERALD).

## 1. Purpose and measurable objective

### 1.1 Main objective

Develop an agent that learns from the environment's legitimate interaction interface:

- pixels;
- the action it previously executed;
- the scalar reward returned by the environment;
- an episode-reset signal used only to reset internal state;
- the next pixels and scalar reward after acting.

It must learn predictive representations, action consequences, persistent information, useful exploration, repeatable behaviours and task progress from experience.

### 1.2 What counts as success

Success is not a falling prediction loss alone. Evidence must accumulate across these levels:

1. **Perception:** stable spatial representations preserve task-relevant visual information.
2. **Prediction:** the model predicts future latent observations better than persistence and shuffled-action controls.
3. **Action grounding:** predictions change meaningfully with actions and correctly anticipate observed consequences.
4. **Temporal competence:** useful predictions survive closed-loop rollout, not merely teacher-forced one-step prediction.
5. **Memory:** recurrent or episodic memory measurably improves performance on situations where history matters.
6. **Learning:** the agent improves its real behaviour from its own experience.
7. **Progress:** it reaches more varied and deeper achievements across unseen procedural worlds.
8. **Generalisation:** the agent adapts to held-out seeds and, in a separate research track, previously unseen mechanics.

### 1.3 Explicit non-goals for the first core version

- Do not build a hand-coded Crafter solver.
- Do not hardcode crafting recipes, resource dependencies, survival thresholds, achievement order, object names or action-to-situation rules.
- Do not add a causal graph merely because a diagram looks useful.
- Do not add a planner before validating model rollouts.
- Do not create dozens of independent model families.
- Do not use language models, text instructions or external knowledge to choose actions in the clean visual-learning setting.
- Do not use external pretrained visual weights in the primary tabula-rasa result. Any pretrained variant must be reported separately. The v1.2 frozen DINOv2 ViT-S/14 arm is diagnostic-only and is governed by the binding amendment in Section L.

---

## 2. Research evidence and design implications

This section records what the papers support and what they do **not** prove for our project.

| Research | Evidence relevant to ORBIT | Design consequence | Limitation / caution |
|---|---|---|---|
| Crafter, Hafner (ICLR 2022) | A partially observed visual survival world with semantically meaningful achievements, procedural worlds, sparse achievement rewards, long dependencies and reusable behaviours. | Keep pixels as the policy observation; evaluate per-achievement breadth and depth, not only return. | The game's own action interface includes high-level discrete commands. Using the native action interface is allowed; programming when to use those commands is not. |
| EMERALD, ICML 2025 | Spatial latent state plus masked latent Transformer / MaskGIT prediction achieved a strong reported Crafter result; its ablations indicate spatial state and iterative masked prediction matter. | Make a spatial-token masked-prediction model the primary world-model research baseline. Reproduce the paper's configuration as a comparator before claiming a new design is better. | Its 10M-step result is not an expected ORBIT result. Paper, code and result artifacts must be distinguished. The official code declares CC BY-NC-SA 4.0; inspect license obligations before copying or adapting source. |
| DreamerV3 | Strong stability practices: recurrent state-space models, symlog/two-hot targets, KL balancing/free bits for categorical latents, unimix, return normalization and explicit actor/critic/world-model gradient boundaries. | Reuse tested numerical ideas after deriving and unit-testing them. Compare against a Dreamer-style baseline. | Do not copy hyperparameters blindly into a different latent model, reward scale or compute regime. |
| V-JEPA 2 | Separates large-scale action-free visual pretraining from action-conditioned post-training; demonstrates JEPA-style prediction can support planning. | Preserve a clean separation between visual representation learning and action-conditioned dynamics. | Its visual pretraining uses over one million hours of internet video. It is not a fair tabula-rasa comparison to a model trained only on Crafter. |
| DINOv2 (Meta AI) | A pretrained ViT visual backbone; ViT-S/14 has 21M parameters, 384-dimensional features and 14×14-pixel patches. The model card reports LVD-142M pretraining and spatial patch-token outputs. | Add a frozen ViT-S/14 diagnostic to test whether pretrained visual features improve ORBIT’s held-out spatial representation and action-conditioned prediction when the downstream model is held fixed. | Natural-image pretraining is not Crafter-specific; pixel-art domain shift and the required resize/patch adapter can confound the result. Keep it separate from the scratch-trained primary track. |
| TD-MPC2 | Latent model predictive control can be effective when dynamics and value learning are jointly reliable. | Keep latent MPC as a later candidate; compare it against an actor-critic trained in imagination. | Its results cover continuous-control domains; Crafter is discrete-action and partially observed. Adaptation must be tested. |
| Slot Attention | Competitive slot assignment can produce object-centric representations without supplying object labels. | Treat object slots as a research ablation over spatial tokens, not as a required component of the base model. | Slots do not automatically become stable object identities or causal entities. Measure binding and persistence. |
| DIAYN / CIC | Unsupervised skill discovery can learn diverse behaviours from state transitions and latent skill variables. | Test latent skill discovery only after a reliable representation and replay pipeline exist. | Diversity is not the same as task usefulness. Promote only on downstream Crafter performance and transfer. |
| PAIRED / ACCEL | Environment design and level mutation can create adaptive training curricula. | Test an external training-only curriculum teacher after the student baseline is stable. | The teacher must not expose hidden environment state or achievement labels to the policy. Generated training variants are not the official Crafter benchmark. |
| Mars | Counter-commonsense changes to terrain, survival and task dependencies test situated rule induction. | Use an audited, pixel-only changed-mechanics benchmark for the active-rule track. | Mars's published task/instruction interface is not automatically compatible with our no-text policy. Strip instructions and metadata or do not claim a visual-only result. |
| RLiable | Point estimates hide seed variation; recommends bootstrap intervals, IQM, performance profiles and probability of improvement. | Every model comparison reports uncertainty and the full distribution, not a single best run. | Episode seeds and training seeds are different sources of randomness and should not be conflated. |

### 2.1 Required research references

- Crafter: https://danijar.com/project/crafter/ and https://github.com/danijar/crafter
- EMERALD paper: https://proceedings.mlr.press/v267/burchi25a.html
- EMERALD official code and declared license: https://github.com/burchim/EMERALD
- DreamerV3 paper: https://arxiv.org/abs/2301.04104 and code: https://github.com/danijar/dreamerv3
- V-JEPA 2: https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/
- DINOv2 paper: https://arxiv.org/abs/2304.07193
- DINOv2 official repository and model card: https://github.com/facebookresearch/dinov2 and https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
- TD-MPC2: https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html
- Slot Attention: https://research.google/pubs/object-centric-learning-with-slot-attention/
- DIAYN: https://research.google/pubs/diversity-is-all-you-need-learning-skills-without-a-reward-function/
- CIC: https://arxiv.org/abs/2202.00161
- PAIRED: https://research.google/pubs/emergent-complexity-and-zero-shot-transfer-via-unsupervised-environment-design/
- ACCEL: https://proceedings.mlr.press/v162/parker-holder22a.html
- Mars: https://proceedings.neurips.cc/paper_files/paper/2024/hash/1fb6d0b52f5e41b11392841a66dbfe7d-Abstract-Datasets_and_Benchmarks_Track.html
- RLiable: https://github.com/google-research/rliable

---

## 3. Repository audit findings that affect the plan

The following findings were checked against the public `main` branch files available during this review. They are reasons for gates, not reasons to assume the same bug exists in every branch or later commit.

### 3.1 Existing benchmark ambiguity must be resolved first

Two committed result files report the same checkpoint path and the same 50-episode seed range `2050..2099`, but different episode results and metrics:

- `results/screen_v3_champion_clean_50ep.json`: Crafter score `3.17087726205484`, mean survival `172.96`, mean reward `3.34`.
- `results/screen_v3_t2_final_results.json`: Crafter score `4.55`, mean survival `181.36`, mean reward `4.12`.

Both JSON files name `/content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v3_t2_final.pt`, but neither path string proves the two evaluations used byte-identical checkpoint contents. The reports also differ in their per-achievement rates and per-episode unlock lists.

**Required action:** do not call either file the definitive current champion until the evaluation is rerun from a checkpoint whose SHA-256 is recorded, with the exact code commit, config hash, environment version, action mapping hash and seed manifest captured.

Relevant files:

- https://github.com/sahiltolani30/learn-a-world/blob/main/results/screen_v3_champion_clean_50ep.json
- https://github.com/sahiltolani30/learn-a-world/blob/main/results/screen_v3_t2_final_results.json
- https://github.com/sahiltolani30/learn-a-world/blob/main/RESULTS.md

### 3.2 Historical memory-gradient failure

The Prometheus math audit documented a GRU memory core that appeared in the optimizer but did not receive gradients through the training losses because collection/evaluation calls were under `no_grad` and training consumed stored hidden states. A module being present in an optimizer is not evidence that it learns.

**Required action:** every trainable component needs gradient-coverage tests and a controlled learning test, not only shape tests or decreasing aggregate loss.

Reference: https://github.com/sahiltolani30/learn-a-world/blob/main/prometheus/MATH_AUDIT.md

### 3.3 Historical reporting and gate failures

The repository documents telemetry key mismatches, incorrect audit-buffer calls, weakened checks and fail-open gates. A missing metric must be `MISSING/FAIL`, never treated as zero and never automatically passed.

Reference: https://github.com/sahiltolani30/learn-a-world/blob/main/prometheus/ROADMAP.md and the detailed Prometheus fix plans.

### 3.4 Existing successful component tests are not gameplay evidence

MOSAIC's `STATUS.md` reports P0/P1 checks and P2 action-conditioning, persistence-baseline and collapse diagnostics. These are useful world-model validation results, but they do not establish a competent full Crafter policy. The same distinction applies to every ORBIT phase.

Reference: https://github.com/sahiltolani30/learn-a-world/blob/main/mosaic/STATUS.md

### 3.5 No accidental reuse of unrelated projects

Keep changes scoped to the allowed Crafter research files and do not inspect, import or modify unrelated project directories. v1.1 controls package placement: audit `true_jepa/` first and do not create a duplicate package without an approved decision.

---

## 4. Benchmark boundary and anti-cheating contract

This is a hard boundary. It must be implemented and tested before any ORBIT training.

### 4.1 Policy input contract

The policy receives:

```python
agent.reset()
action_id = agent.act(
    obs_uint8: np.ndarray,        # shape [64, 64, 3], dtype uint8
    reward_prev: float,           # raw reward from the previous transition; 0.0 at reset
) -> int                           # native action index treated as an opaque category
```

The canonical Crafter environment is expected to expose an RGB observation with shape `(64, 64, 3)` and a discrete action space with 17 actions. Discover `num_actions` from `env.action_space.n` at runtime and assert the canonical environment returns 17. Do not hardcode action descriptions into model code.

The `reset` flag is a control signal for clearing recurrent state. It must not be concatenated into a learned feature vector. The global step counter may be used by the training runner for scheduling and logging, but must not be supplied as a policy feature in the tabula-rasa core. Day/night information should be inferred from observations and experience.

### 4.2 Data the learner must never receive

Never pass any environment `info` fields, structured game state, evaluator labels or simulator internals to the policy, model loss, exploration module, memory, skill discovery or curriculum student.

Forbidden examples include (field names may vary by wrapper/version):

- inventory dictionaries or arrays;
- achievement booleans, names or unlock counters;
- semantic/object maps;
- player coordinates or exact map state;
- terrain arrays, hidden object lists or NPC internals;
- hidden recipe/dependency tables;
- action names/descriptions as model inputs;
- environment source constants used as decision rules.

The environment adapter may receive `info` from the Gym API. It must discard it before constructing learner-facing objects. The evaluator may inspect `info` in a separate process after an episode to calculate metrics. There must be no import path or shared mutable object from evaluator annotations back into training.

### 4.3 Allowed information

- RGB pixels, including visual inventory and HUD elements visible in the observation.
- The previous action token.
- The raw scalar reward returned by the environment.
- Termination/reset signals required by the RL API.
- The native discrete action interface supplied by Crafter.

Crafter's 17 native actions include high-level environment commands such as placing and crafting. Calling these native actions is **not cheating**. The agent must learn from trials and outcomes when to select them. Hand-writing conditions such as `if wood_count >= 2: place_table` is forbidden.

### 4.4 Opaque action interface

The canonical environment has 17 native actions. Use native discrete action indices as opaque categorical IDs by default. A fixed random bijection is an optional, separately reported robustness diagnostic; v1.1 Section D.1 controls.

Requirements:

1. The action vocabulary size is read from `env.action_space.n`.
2. If the optional permutation diagnostic is enabled, generate it from a recorded seed and store it in a wrapper-only manifest.
3. The agent sees and emits only opaque IDs; no action names or semantic descriptions.
4. If enabled, the same permutation is used for training and evaluation of that checkpoint.
5. Optional multi-run permutation testing is a robustness diagnostic, not a prerequisite for the first clean baseline.
6. A permutation change at inference without remapping/relearning is a different control problem; do not claim zero-shot invariance unless measured.
7. The action mapping hash (identity mapping by default) is included in every checkpoint and result manifest.

### 4.5 Reward boundary

Use the environment-returned scalar reward unchanged as the task reward. Do not add manual bonuses for named achievements, resource collection, crafting steps, object proximity, survival thresholds or preferred action sequences.

Exploration research may use a clearly separated intrinsic information signal, but:

- it must be generated from prediction uncertainty/information gain, not achievement identities or hard-coded rules;
- it must be logged separately from environment reward;
- the extrinsic value target must use only environment reward;
- all intrinsic-reward experiments must have a `no_intrinsic` ablation;
- official benchmark metrics must use evaluator-reported achievement outcomes, not the intrinsic score.

### 4.6 Curriculum boundary

The curriculum teacher may select training environment seeds or valid training variants using student outcomes. It must not pass hidden state, rules, recipe data, achievement labels or task names into the student. Standard Crafter evaluation remains unmodified. Any modified-mechanics experiments must be reported as a separate evaluation track.

### 4.7 Leakage tests

- Static scan for forbidden keys and imports in learner package.
- Runtime poison test: inject a fake `info` object containing unique sentinel values and verify no learner tensor, replay record, logit or loss changes because of those values.
- Policy signature test: reject structured inputs.
- Wrapper permutation test: verify every opaque ID maps bijectively to exactly one native action.
- Data-flow test: evaluator annotations are written to evaluator-only artifacts and never appear in learner replay batches.
- Human-readable action names must not appear in config objects passed into the model.
- No reward shaping source file may import Crafter achievement names.

---

## 5. ORBIT core architecture

### 5.1 Architecture decision: choose by experiment, not taste

Do not lock the spatial grid or latent dimension based on intuition. Run controlled candidates before choosing the production core.

| Candidate | Source / definition | Purpose |
|---|---|---|
| A — EMERALD reference | Reproduce the published spatial categorical latent configuration and masked-latent iterative prediction from the official paper/code. Paper ablation includes a `4×4×32` spatial latent and 3 MaskGIT refinement steps. | Strong published Crafter-relevant reference. |
| B — repository spatial baseline | Prometheus predictor uses 64 spatial tokens with token dimension 64, two Transformer layers, four attention heads and 35% masking (22 of 64 tokens after flooring). | Compare against an existing repository implementation. |
| C — repository JEPA baseline | `true_jepa/config.py` uses observation `(64,64,3)`, patch size 8, 64 tokens, token dimension 192, belief dimension 256, four predictor heads, two predictor layers, MLP dimension 384 and a three-member transition ensemble. | Compare a higher-dimensional JEPA candidate already represented in the repository. |
| D — ORBIT clean-room candidate | A spatial-token JEPA encoder + recurrent temporal belief + action-conditioned masked spatial predictor. Dimensions and mask schedule are selected from A–C results, not guessed. | Primary from-scratch ORBIT implementation candidate. |
| E — DINOv2 pretrained diagnostic | Frozen Meta AI `dinov2_vits14` (ViT-S/14, without registers), explicit full-frame 64×64→224×224 preprocessing, spatial patch-token adapter to the common ORBIT grid, and the same downstream belief/predictor/outcome heads as the matched scratch arm. | Diagnostic-only encoder substitution; never counted as the primary tabula-rasa result. |

Do not compare these candidates with different training budgets, observation preprocessing, action mappings, optimizer-update counts, reward wrappers or evaluation seed sets and then attribute the score difference to architecture. Candidate E is a separate pretrained diagnostic: compare it directly with the selected scratch encoder using the matched protocol in the v1.2 binding amendment. Do not mix its results into the A–D clean-room leaderboard.

The EMERALD paper reports that its spatial latent + MaskGIT variant with three refinement steps achieved 58.1 in its 10M-step setting; its ablation table also reports different scores for other latent arrangements and refinement counts. The official repository result JSON separately records an `achievements_score` of about 57.04. The exact reason for the difference is not inferred here; do not silently substitute one figure for the other. Pin the artifact and inspect its own evaluation protocol before reproducing or comparing it. Three MaskGIT refinement steps are **not** three environment steps and are **not** a three-step imagined rollout. Keep these quantities separate.

### 5.2 Data flow of the selected core

```text
RGB image o_t
    │
    ▼
Online spatial encoder E_θ(o_t) ────────────────┐
    │                                           │
    ├── spatial tokens z_t [N, D]               │
    └── pooled representation p_t               │
                                                ▼
Previous action token + previous reward → recurrent belief update → h_t
                                                │
Current spatial tokens + h_t + candidate action a_t
                                                │
                                                ▼
Action-conditioned masked spatial predictor F_φ
                                                │
                                                ├── predicted next spatial tokens ẑ_{t+1}
                                                ├── reward prediction r̂_t
                                                └── continuation prediction ĉ_t

EMA target encoder E_ξ(o_{t+1}) → stop-gradient target z^target_{t+1}

Real next frame/reward/termination → replay → world-model training
Validated world model → later actor-critic imagination
```

### 5.3 Components

**A. Spatial encoder**

- Consumes only RGB pixels.
- Produces a grid of spatial tokens and a pooled feature.
- Use 2D positional information; preserve spatial arrangement.
- Start with a reproducible CNN/patch-stem baseline. Compare against the paper-faithful EMERALD encoder rather than assuming one patch stem is best.
- Online encoder receives gradients. Target encoder is an EMA copy and receives no gradient.
- For the separate Candidate E diagnostic only, freeze the DINOv2 ViT-S/14 backbone and train the explicit spatial adapter plus the same downstream ORBIT modules. DINOv2 patch tokens must retain their 2D layout; class/register tokens are not spatial cells.
- No text labels, segmentation masks or object names are supplied during training.

**B. Temporal belief**

- Recurrent state summarizes the current and previous visual/action/reward history.
- It must be trained through an explicit sequence loss; storing hidden states in replay and later treating them as current hidden states is not sufficient.
- Episode reset sets recurrent state to its documented initial value.
- During replay training, reconstruct the hidden state using a burn-in prefix under the current model parameters. Do not train on stale hidden-state snapshots unless an explicit stale-state ablation is run.

**C. Action-conditioned spatial predictor**

- Conditions prediction on current spatial tokens, current belief and selected action token.
- Uses masked target-position queries and predicts EMA target spatial tokens at corresponding positions.
- The target encoder is evaluated on the real next observation and stop-gradient is applied to target tokens.
- Report prediction error over all target positions at evaluation, even if training loss only covers masked positions.
- Use an ensemble only as an uncertainty estimator if ensemble calibration proves useful. Ensemble disagreement is not automatically calibrated uncertainty.

**D. Outcome heads**

- Reward head predicts the raw environment reward for the current action transition.
- Continuation head predicts whether the underlying environment continues after the transition.
- Distinguish environment termination from wrapper time-limit truncation. Verify the exact Crafter/Gym wrapper semantics before defining targets.
- Value/actor heads are not introduced until world-model prediction passes the relevant gate.

**E. Replay and training controller**

- Store raw `uint8` frames and transition metadata.
- Store episode IDs, step-in-episode, actions, rewards, terminated/truncated flags and frame references.
- Enforce sequence contiguity and episode boundaries.
- Start with uniform sequence replay. Add priority or rare-event archives only as a separately measured experiment; do not use achievement labels as replay priorities.

### 5.4 Temporal equations and indexing contract

For one episode, observations are indexed `o_0, o_1, …, o_T`; actions and rewards are transitions indexed `a_0, r_0, …, a_{T-1}, r_{T-1}`.

For each real time step:

\[
 z_t = E_\theta(o_t), \qquad \bar z_t = E_\xi(o_t)
\]

\[
 h_t = G_\psi\left(h_{t-1},\; \operatorname{pool}(z_t),\; e_a(a_{t-1}),\; e_r(r_{t-1})\right)
\]

\[
 \hat z_{t+1} = F_\phi(z_t,h_t,a_t,m_t)
\]

where `m_t` is the sampled mask/query set and the supervised target is `stop_gradient(E_ξ(o_{t+1}))`.

At the first observation in an episode, define `h_{-1}=0`, previous action token = `START`, and previous reward = `0.0`. `START` is a control token, not one of the 17 environment actions; its embedding is learned and explicitly tested.

For closed-loop model rollout, use the model's predicted next tokens and predicted reward to update the simulated recurrent state:

\[
 \hat h_{t+1}=G_\psi(\hat h_t,\operatorname{pool}(\hat z_{t+1}),e_a(a_t),e_r(\hat r_t))
\]

Then predict the following latent using `(zhat_{t+1}, hhat_{t+1}, a_{t+1})`. Do not silently replace predicted states with encoded real future frames in a test labelled “closed-loop”. Teacher-forced/open-loop-action-sequence tests and self-fed closed-loop tests must be reported separately.

### 5.5 Masked latent prediction loss

For batch size `B`, token count `N`, feature dimension `D` and binary mask `m_{b,n}`:

\[
 \mathcal L_{mask} =
 \frac{\sum_{b=1}^{B}\sum_{n=1}^{N}m_{b,n}\,
 \lVert \hat z_{b,n}-\operatorname{sg}(z^{target}_{b,n})\rVert_2^2}
 {D\sum_{b,n}m_{b,n}}
\]

Requirements:

- Assert `sum(mask)` is nonzero for every sample used in masked prediction.
- Assert targets have no gradient.
- Assert loss denominator is the actual number of masked feature elements, not the number of samples or token positions alone.
- Log masked token count, actual mask fraction and per-position loss.
- Evaluate the full target map as well as masked-only loss to detect a model that only learns the mask-training subset.

Do not add VICReg covariance regularization by habit. With batch size `B=8`, per-time covariance over 8 samples has centered rank at most 7; it cannot provide a full-rank estimate for a 192-dimensional feature. The initial core should use explicit collapse diagnostics and a justified variance regularizer only if needed. Any covariance term must state sample count `n`, feature dimension `d`, rank bound `min(d,n-1)`, estimator and test. If the design uses `B×T` frames, report temporal correlation and effective rank; do not pretend 128 adjacent frames are 128 independent episodes.

### 5.6 EMA target encoder

The only allowed update is:

\[
 \xi \leftarrow \tau\xi+(1-\tau)\theta
\]

Update after the online optimizer step, not on every forward pass. Target parameters must not be present in an optimizer and target outputs must be stop-gradient.

The existing `true_jepa/config.py` carries a candidate schedule `τ=0.996→0.999` over 20,000 optimizer updates. It is a **repository-inherited candidate**, not a proven ORBIT optimum. The exact e-folding time is `−1/ln(τ)` updates and the half-life is `ln(0.5)/ln(τ)` updates. `1/(1−τ)` is only a close approximation for τ near 1:

| τ | e-folding time | half-life |
|---:|---:|---:|
| 0.996 | 249.4997 optimizer updates (approximation: 250) | 172.9400 optimizer updates |
| 0.999 | 999.4999 optimizer updates (approximation: 1,000) | 692.8005 optimizer updates |

These are optimizer updates, not environment frames. Unit-test the exact EMA update using a scalar parameter before training.

### 5.7 Multi-horizon prediction

Evaluate horizons `H ∈ {1, 2, 4, 8}` first. The set is a diagnostic ladder, not an instruction to train all horizons from the beginning.

For each horizon, report:

- latent MSE and cosine similarity;
- model error relative to persistence error;
- action-shuffled error;
- reward prediction calibration;
- continuation prediction calibration;
- ensemble error/disagreement calibration if an ensemble exists;
- error growth over time;
- teacher-forced versus closed-loop performance.

Define the normalized error ratio:

\[
 R_H = \frac{\operatorname{MSE}(\hat z_{t+H},z^{target}_{t+H})}
 {\operatorname{MSE}(z^{target}_{t},z^{target}_{t+H})+\epsilon}
\]

The denominator is the target-encoder persistence baseline. Use the same target encoder, same observations, same valid token set and same normalization for numerator and denominator. `R_H<1` means the model beats persistence on this metric. Report confidence intervals of the paired difference; an average below one without uncertainty is not a promotion gate.

Cosine similarity `≥0.8` from the existing Prometheus roadmap may be logged for continuity, but it is not sufficient by itself: embedding cosine can stay high while rare, task-relevant details disappear.

---

## 6. The four required research directions

These are four distinct research tracks around one shared core. They are not four separate agents and must not be hardwired into a single action sequence.

### Direction 1 — Object-based world modelling

**Question:** Does an object-centric latent improve persistent entity tracking and action-effect prediction over spatial-token JEPA alone?

**Candidates:**

- D1-A: spatial-token baseline without slots;
- D1-B: Slot Attention over encoder features;
- D1-C: spatial tokens plus learned slots;
- D1-D: slots plus an interaction/dynamics predictor.

Slot counts are experimental capacity values, not semantic object categories. Candidate slot counts may be `{4, 8, 16}`; no slot is assigned a name such as `tree`, `table`, `zombie` or `water`.

**Training:** unsupervised from visual observations and transitions. No object bounding boxes, semantic maps, entity IDs or game labels in the loss.

**Evaluation:**

1. Re-identification after partial occlusion or temporary disappearance.
2. Stability across nearby camera/player movement.
3. Slot churn and duplicate-slot rate.
4. Prediction improvement on interaction sequences.
5. Downstream policy ablation with slots on/off.

If simulator semantic maps are used for an object-tracking diagnostic, they must live in an evaluator-only process and must never be read by the learner or influence weights. Report this as an oracle diagnostic, not a learned capability.

**Promotion:** keep object slots only if they improve a preregistered held-out prediction or downstream policy metric with a confidence interval supporting improvement, without materially harming throughput or memory.

### Direction 2 — Active rule discovery

**Question:** Can the agent choose experiments that distinguish competing hypotheses about unknown world mechanics?

A prediction-error seeker is not automatically a scientific experimenter. The track must test whether the chosen action is expected to reduce uncertainty over a persistent model of transition rules.

**Comparison arms:**

- D2-A: random exploration;
- D2-B: calibrated ensemble disagreement;
- D2-C: prediction-error / learning-progress exploration;
- D2-D: approximate expected information gain over model hypotheses;
- D2-E: task-aware information gain, reported separately.

The main Crafter game has mostly fixed mechanics. It can test learning of its regular dynamics, but by itself it cannot prove discovery of previously unknown rules that change between worlds. Therefore use two datasets:

1. Unmodified Crafter for the official benchmark.
2. A separately versioned, hidden-mechanics environment suite inspired by Mars, with terrain/survival/task-dependency variants. The model sees only pixels, action tokens and scalar rewards. Variant IDs and rule parameters remain evaluator-only.

Mars's original text/instruction pathway must not be passed into ORBIT. If the benchmark cannot be adapted to the strict visual-only contract, do not claim a visual-only Mars result.

**Evaluation:** rule-change detection delay, number of interventions required, calibrated hypothesis uncertainty, adaptation performance, retention of old rules when context returns, and transfer to held-out rule combinations.

**Promotion:** the information-gain arm must outperform random/disagreement controls on adaptation efficiency and downstream performance across held-out rule variants. An uncertainty score that merely rises on noisy scenes does not pass.

### Direction 3 — Reusable skill discovery

**Question:** Can the agent discover temporally extended behaviours from its own transitions and reuse them in new contexts?

**Candidates:**

- D3-A: no skills; primitive-action policy baseline;
- D3-B: DIAYN-style latent skill IDs and discriminator;
- D3-C: CIC-style contrastive transition-to-skill objective;
- D3-D: trajectory segmentation / option discovery from recurring state-transition patterns.

Skills are anonymous latent behaviours. Do not call them `collect wood`, `craft pickaxe`, `sleep` or similar, and do not label training trajectories with achievement names. The environment action interface remains the only source of action semantics.

**Evaluation:**

1. Behavioural diversity and controllability of skill latents.
2. Repeatability across different initial states and procedural seeds.
3. Skill termination and interruption reliability.
4. Composability of two or more discovered skills.
5. Downstream Crafter score/reward/survival against primitive-action control under the same interaction budget.
6. Transfer to held-out layouts and hidden-mechanics variants.

Diversity alone is not a pass. A library of many useless behaviours is a failure.

### Direction 4 — Autonomous curriculum generation

**Question:** Can an external training teacher select or create the next useful challenge based on the student's measured learning frontier?

**Comparison arms:**

- D4-A: ordinary procedurally generated Crafter seeds;
- D4-B: random selection of training variants;
- D4-C: progress-based curriculum;
- D4-D: PAIRED/ACCEL-inspired environment design, only after a reliable student policy exists.

The teacher is not part of the deployed policy. It may use training outcomes to choose the next training seed or valid training variant. It must not provide labels, hidden state, recipes or action hints to the student.

**Training-only variants:** map layout, resource distribution, initial placement and other officially permitted environment parameters. Any change to game mechanics is a separate hidden-rule experiment. Do not alter the standard Crafter reward definition in the official benchmark.

**Evaluation:** total environment steps to reach a fixed development performance level, success on held-out standard Crafter seeds, success on held-out variants, diversity of generated levels, and solvability/validity of generated levels. The evaluator may validate solvability outside the learner.

**Promotion:** curriculum improves held-out performance at a fixed interaction budget and does not reduce performance on the unmodified benchmark. If it only improves on generated training levels, it fails.

---

## 7. Learning objectives and reward/value mathematics

### 7.1 Environment reward

Let `r_t^env` be the exact scalar reward returned by the configured Crafter environment wrapper. Store it unchanged. Do not recompute it from achievement names or internal info.

The learner's task return uses the observed environment reward:

\[
 G_t = r_t^{env}+\gamma c_tG_{t+1}
\]

where `c_t` is the continuation/discount target defined by the verified environment termination semantics.

### 7.2 Separate losses

The initial world-model objective is:

\[
 \mathcal L_{WM}=\lambda_{mask}\mathcal L_{mask}
 +\lambda_{var}\mathcal L_{var}
 +\lambda_{r}\mathcal L_r
 +\lambda_c\mathcal L_c
 +\lambda_{multi}\mathcal L_{multi}
\]

However, **do not invent or silently tune these coefficients**. Each coefficient must be recorded in the parameter ledger with:

- units/scale of its loss term;
- the numerical range observed during a baseline run;
- gradient norm contributed to each shared parameter group;
- calibration/ablation used to select it;
- exact config hash.

Start with only the terms required by the phase. Do not activate future losses early just to keep all modules “busy”.

Reward prediction should use the exact raw reward target. Candidate loss functions (MSE, Huber, symlog/two-hot distributional regression) must be compared on held-out reward prediction calibration. If symlog is used:

\[
 \operatorname{symlog}(x)=\operatorname{sign}(x)\ln(1+|x|)
\]

\[
 \operatorname{symexp}(y)=\operatorname{sign}(y)(e^{|y|}-1)
\]

Unit test both inverse directions over a grid including zero, negative values, small values, and the observed reward/return range. Do not choose two-hot bin support until a data audit of actual symlogged targets is complete. If two-hot bins are used, record the exact bin construction, endpoints, count, clipping rate and inverse expectation calculation.

### 7.3 Continuation and time-limit semantics

Do not equate all `done=True` events. Record separately:

- `terminated`: true terminal condition such as agent death;
- `truncated`: an external episode time limit or wrapper cutoff;
- `final_observation`: the actual final frame for a truncation if available.

If the installed Crafter API does not expose these cleanly, Phase 0 must inspect the actual wrapper/source and implement a tested adapter. Do not guess from reward sign or episode length.

For an actual terminal transition, continuation target is zero. For a time-limit truncation, the underlying environment may still continue; whether the critic bootstraps from the final observation must follow the chosen benchmark semantics and be unit-tested. A transition must not silently disappear at a time limit.

### 7.4 Actor-critic is a later phase

When enabled, keep these distinct:

- world-model optimizer;
- actor optimizer;
- critic optimizer;
- target/slow critic state if used.

Define exact detach boundaries. An imagined policy loss must not accidentally backpropagate into the world model unless a separately specified experiment explicitly tests that design. Log optimizer membership, per-module gradient norms and parameter deltas.

Advantage normalization must use a documented running statistic or another mathematically specified method. Per-batch standardisation can amplify noise when the true advantage variance is near zero. The earlier repository audit documented this failure mode. Include a unit test with a constant advantage stream and a low-signal noisy stream.

---

## 8. Numerical audit: values, dimensions and memory calculations

This section is deliberately explicit. It separates exact arithmetic from inherited candidate hyperparameters.

### 8.1 Environment observation size

Canonical Crafter RGB observation:

\[
64\times64\times3=12,288\text{ uint8 values}
\]

At one byte per uint8 value:

- one frame = 12,288 bytes = 12 KiB;
- float32 conversion = 12,288 × 4 = 49,152 bytes = 48 KiB.

For a batch of `B=8` sequences with `T=16` transitions, the sequence contains `T+1=17` observation frames:

\[
8\times17\times12,288=1,671,168\text{ bytes}
\]

That is 1.594 MiB of raw uint8 observations or 6.375 MiB if all frames are materialised as float32. These are input tensors only, not total GPU memory use.

### 8.2 Token activation size candidate

For the repository `true_jepa` candidate (`N=64`, `D=192`, `B=8`, `T=16`):

\[
8\times16\times64\times192=1,572,864\text{ elements}
\]

One such tensor consumes:

- FP32: 1,572,864 × 4 = 6,291,456 bytes = 6 MiB;
- BF16/FP16: 1,572,864 × 2 = 3 MiB.

This excludes online and target copies, transformer intermediates, attention matrices, gradients, optimizer states, recurrent activations, predictor outputs and temporary buffers. **Never use this one-tensor calculation as a GPU-memory estimate.** Run a real forward/backward peak-memory preflight before selecting batch size or enabling AMP.

### 8.3 Mask count arithmetic

For `N=64` tokens and an existing `mask_ratio=0.35` implementation using `floor`:

\[
\lfloor0.35\times64\rfloor=\lfloor22.4\rfloor=22
\]

Actual mask fraction:

\[
22/64=0.34375=34.375\%
\]

Therefore the code must log `mask_count=22` and `effective_mask_fraction=0.34375`; do not log it as exactly 35%. A future experiment may compare exact mask counts `{16, 22, 32}` (25%, 34.375%, 50%) as a preregistered ablation. Do not silently change mask count mid-run.

### 8.4 Action entropy arithmetic

For a uniform distribution over 17 actions, the maximum categorical entropy is:

\[
H_{max}=\ln(17)=2.833213344\text{ nats}
\]

Equivalent bits:

\[
\log_2(17)=4.087462841\text{ bits}
\]

This is a mathematical reference for the entropy logger, not a required exploration entropy target. The learned policy may become more concentrated as it learns. A collapse alarm must use both entropy and behaviour metrics (action histogram, unique actions, state-conditioned action variation, reward/progress), not entropy alone.

### 8.5 Discount-factor interpretation

The `true_jepa` repository candidate uses `gamma=0.997`. Its mathematical consequences are:

\[
1/(1-0.997)=333.333\text{ steps}
\]

This is the geometric-series effective horizon, not a hard cutoff.

\[
0.997^{100}=0.74048426
\]
\[
0.997^{1000}=0.04956308
\]

Half-life:

\[
\ln(0.5)/\ln(0.997)=230.7023\text{ steps}
\]

`0.997` is an inherited candidate, not automatically correct for every policy/value head. Compare a preregistered small gamma sweep on development runs and select based on held-out return/achievement metrics and calibration. Store gamma in every checkpoint and evaluation manifest. Never change gamma while resuming the same optimizer/replay lineage without an explicit new experiment ID.

### 8.6 Replay RAM calculation

If replay stores one unique uint8 frame per transition, a capacity of 120,000 frames costs:

\[
120,000\times12,288=1,474,560,000\text{ bytes}=1.3733\text{ GiB}
\]

If a naive replay implementation separately stores both `obs` and `next_obs` for 120,000 transitions:

\[
2\times120,000\times12,288=2,949,120,000\text{ bytes}=2.7466\text{ GiB}
\]

These figures exclude arrays for actions, rewards, episode IDs, frame references, valid flags, priority data, indexes, allocator overhead, compression staging, logs and model memory.

**Requirement:** design replay around unique observation frames referenced by transitions/sequences where possible. Before allocating, compute exact byte requirements from config and fail early if projected host memory exceeds the measured budget. Keep the process under the repository's existing conservative 80% host-RAM guard and test the guard itself. Never silently drop replay data because of memory pressure.

### 8.7 Recurrent batch indexing

For batch size `B=8` and `T=16` transitions, the world-model training tensors must have these shapes:

| Tensor | Shape |
|---|---|
| observations | `[B, T+1, 64, 64, 3]` = `[8,17,64,64,3]` |
| actions | `[B, T]` = `[8,16]` |
| rewards | `[B, T]` = `[8,16]` |
| terminated | `[B, T]` |
| truncated | `[B, T]` |
| valid-transition mask | `[B, T]` |
| target token maps | `[B, T, N, D]` after alignment |
| recurrent states | `[B, T+1, H]` or the precisely documented state alignment |

Do not use a single variable called `sequence_length` without declaring whether it counts observations or transitions. In ORBIT config use `sequence_transitions`; derive `observation_count=sequence_transitions+1`.

### 8.8 Covariance and rank check

For `n` samples and feature dimension `d`, the centered sample covariance has rank at most:

\[
\operatorname{rank}(C)\leq\min(d,n-1)
\]

If a covariance loss is computed from batch size 8 at one time step, rank ≤7. It is mathematically invalid to expect a full-rank 192-dimensional covariance estimate from those eight samples. If using 128 `B×T` samples and `d=192`, rank ≤127, still not full rank. The plan therefore forbids blindly copying a covariance regularizer into this setup. Use variance/rank diagnostics and validate the sample geometry before adding decorrelation losses.

### 8.9 Crafter score: exact units and test vectors

The official Crafter aggregate uses per-achievement success rates in **percentage points** `s_i ∈ [0,100]`:

\[
S=\exp\left(\frac{1}{22}\sum_{i=1}^{22}\ln(1+s_i)\right)-1
\]

Do **not** pass fractions in `[0,1]` to this formula and then multiply the result by 100. Do **not** multiply the final result by 100 again. The `s_i` values are already percentages.

For `N_episodes`:

\[
s_i=100\times\frac{\text{episodes that unlocked achievement }i}{N_{episodes}}
\]

Required unit tests:

- all `s_i=0` → score `0`;
- all `s_i=100` → score `100`;
- all 22 rates are computed from booleans, not number of repeated unlock events;
- a 50-episode report has rates in increments of 2 percentage points;
- a 100-episode report has rates in increments of 1 percentage point;
- 22 rates are present, and missing achievement keys are a hard error;
- formula is tested against the two checked-in v3 result files.

Independent calculation from the committed per-achievement percentages gives:

| Artifact | Reported score | Recomputed from 22 percentage rates |
|---|---:|---:|
| `screen_v3_champion_clean_50ep.json` | 3.17087726205484 | 3.170877262054839 |
| `screen_v3_t2_final_results.json` | 4.55 | 4.546214306875846 (rounds to 4.55) |

The formula is consistent with each file's own achievement rates. The two files still disagree because their per-episode and per-achievement data differ. Fixing the formula does not resolve the checkpoint/evaluation provenance discrepancy.

**Do not confuse these metrics:**

- Crafter score: geometric mean over 22 achievement success percentages;
- mean achievements per episode: average count of achievement booleans per episode;
- unique achievements unlocked: number of achievement types with nonzero success across the evaluation set;
- mean episode reward: environment rewards, not the benchmark score;
- survival length: steps alive, not task competence.

“All 22 achievements at least once during evaluation” does not mean an average episode unlocked 22 achievements, nor does it mean every episode unlocked all 22.

### 8.10 Statistical reporting

For each final result report:

- each independent training seed;
- fixed evaluation seed manifest;
- per-episode metrics and per-achievement booleans;
- mean, standard deviation and median;
- IQM where meaningful;
- 95% bootstrap confidence intervals;
- probability of improvement over the frozen baseline;
- full per-achievement success rates;
- training interactions and optimizer updates;
- wall-clock and hardware details.

Use paired evaluation seeds for policy comparisons. Use hierarchical resampling when comparing multiple training runs and evaluation episodes. Do not treat adjacent steps from one episode as independent samples. A development result is not a locked result.

---

## 9. Hyperparameter and configuration governance

Every parameter must be labelled in the parameter ledger as one of:

1. **Mathematically derived:** e.g. token count from image dimensions and patch/grid definition.
2. **Environment-defined:** e.g. observation shape and action count read from the environment.
3. **Paper-reproduced:** copied exactly from a named paper implementation with version/commit and license recorded.
4. **Repository-inherited candidate:** copied from an existing project config but not yet proven for ORBIT.
5. **Experiment-selected:** chosen from a preregistered development sweep with report artifact.
6. **Operational safety limit:** e.g. RAM guard, checkpoint retention; not a model hyperparameter.

No parameter may be silently changed. Every value must include units and semantics.

### 9.1 Initial inherited candidates (not final truth)

| Parameter | Candidate | Provenance | Required action |
|---|---:|---|---|
| Observation | `(64,64,3)` uint8 | Canonical Crafter interface | Runtime assertion |
| Action count | `17` | Canonical Crafter action space | Read `env.action_space.n`; assert 17 |
| Patch size | `8` | `true_jepa/config.py` | Compare to EMERALD reproduction and Prometheus spatial baseline |
| Token count | `64` | 8×8 grid if patch size 8 | Derive from actual encoder output shape |
| Token dimension | `192` | `true_jepa/config.py` | Candidate only; benchmark capacity/throughput |
| Belief dimension | `256` | `true_jepa/config.py` | Candidate only; test memory usefulness and throughput |
| Predictor heads | `4` | `true_jepa/config.py` | Confirm divisible head dimension and ablate only if justified |
| Predictor layers | `2` | `true_jepa/config.py` | Candidate only |
| Predictor MLP dimension | `384` | `true_jepa/config.py` | Candidate only |
| Ensemble size | `3` | `true_jepa/config.py` | Candidate only; test calibration not just disagreement |
| Batch size | `8` | `true_jepa/config.py` | Must pass real peak-memory profiling |
| Sequence transitions | `16` | `true_jepa/config.py` | Explicitly distinguish from 17 observations |
| Replay capacity | `120,000` frames | `true_jepa/config.py` | Capacity selected after RAM budget calculation |
| World-model LR | `3e-4` | `true_jepa/config.py` | Development sweep; not locked |
| Policy LR | `1e-4` | `true_jepa/config.py` | Not active until actor phase; sweep then |
| Weight decay | `1e-6` | `true_jepa/config.py` | Development sweep |
| Grad clip | `10.0` | `true_jepa/config.py` | Confirm norm type and log pre/post clipping |
| Gamma | `0.997` | `true_jepa/config.py` | Development sweep; implications calculated in §8.5 |
| EMA schedule | `.996 → .999`, 20k updates | `true_jepa/config.py` | Compare target lag and validation; updates, not frames |
| Mask count | `22/64` | Prometheus `floor(.35*64)` | Log actual fraction 34.375%; test mask ablation |

This table is a reproducible starting point, not authority to skip measurement. The architecture selection may choose a different candidate after the controlled comparison.

### 9.1.1 DINOv2 diagnostic configuration (separate experiment namespace)

| Setting | Initial diagnostic choice | Provenance / status | Required verification |
|---|---|---|---|
| Backbone | `dinov2_vits14` (non-register) | Official Meta AI DINOv2 model; pretrained diagnostic only | Pin repository commit, weights file and SHA-256; verify license at implementation time |
| Backbone parameters | 21M | Official DINOv2 model card | Confirm loaded checkpoint architecture |
| Patch size / feature width | 14 / 384 | Official DINOv2 model card | Assert actual output shape |
| Input tensor | `224×224` RGB | Explicit design choice; yields 16×16 patch positions | Confirm whole 64×64 frame is represented with no crop |
| Resize interpolation | Nearest-neighbor | First pixel-art-preserving experiment choice | Pin implementation and test deterministic output; alternate interpolation is separate ablation |
| Normalization mean | `(0.485, 0.456, 0.406)` | Official DINOv2 transform defaults | Assert channel order and exact constants |
| Normalization std | `(0.229, 0.224, 0.225)` | Official DINOv2 transform defaults | Assert exact constants |
| Backbone trainability | Frozen | Diagnostic protocol choice | No gradients, no optimizer membership, unchanged checkpoint hash |
| Spatial adapter output | Common `[N,D]` selected for the matched ORBIT model | Experiment-selected after A–D interface is fixed | Log pooling/projection details and parameter count |

These settings define the first diagnostic only; they do not silently set the production encoder, token grid or latent dimension.

### 9.2 Parameter ledger schema

Every experiment must write `orbit/artifacts/parameter_ledger.jsonl` with:

```json
{
  "parameter": "gamma",
  "value": 0.997,
  "unit": "per environment transition",
  "category": "repository_inherited_candidate",
  "source": "true_jepa/config.py",
  "mathematical_effect": "discount half-life 230.7023 transitions",
  "selection_experiment": null,
  "config_hash": "...",
  "approved_by_gate": false
}
```

Do not use the literal sample hash. The implementation must calculate it.

---

## 10. Exact implementation phases, tests and exit gates

No phase may start until all exit tests in the preceding phase pass. If a test is missing, the phase is not complete.

### Phase 0 — Repository, paper and baseline forensic audit

**Implement:**

- isolated `orbit/` package and dedicated branch;
- package/dependency lock;
- code and artifact inventory of allowed existing Crafter model lines;
- checkpoint SHA-256 utility;
- canonical JSON config hashing;
- environment/version report;
- evaluator version and seed-manifest format;
- benchmark-score calculation module with tests;
- result-provenance comparison tool.

**Tests:**

1. Recalculate the two checked-in v3 scores from 22 rates.
2. Compare the two existing reports and flag same-path/same-seed but different episode metrics.
3. Re-evaluate a selected checkpoint with hash, config and seed manifest captured.
4. Verify a second independent evaluator run on the same checkpoint and seeds reproduces per-episode outputs or produces a documented deterministic/non-deterministic delta.
5. Confirm `final_comparison.json` v2 protocol (100 episodes, seeds 1000–1099, max 1000 steps) is not mixed with the 50-episode v3 screens.

**Exit:** one verified baseline artifact with a content hash and exact protocol. If the v3 discrepancy cannot be resolved, mark the old champion as `provenance_unresolved`; do not use it as a definitive comparison anchor.

### Phase 1 — Clean environment and data contract

**Implement:**

- strict learner API;
- sanitized environment wrapper;
- action permutation wrapper;
- seed manager;
- `terminated`/`truncated` adapter;
- episode boundary handling;
- replay schema and frame store;
- run manifest.

**Tests:**

1. Observation shape, dtype, numeric range and channel order.
2. Action count discovered at runtime and canonical value validated.
3. Bijection test for opaque/native action mapping.
4. Policy cannot access `info` sentinel values (static and runtime tests).
5. Reset clears recurrent state; non-reset preserves it.
6. No cross-episode sequence samples.
7. Every sampled training sequence has exactly 16 transitions and 17 aligned observations.
8. Truncation/termination tests against the installed environment version.
9. Replay save/load byte parity and ring-wrap sequence contiguity.
10. Checkpoint includes all RNG states and action mapping hash.
11. Crash during checkpoint save leaves previous checkpoint readable.
12. Resuming does not claim exact environment-state resume unless environment state serialization was tested. Otherwise resume only at a documented episode boundary.

**Exit:** no learner leakage, correct sequence semantics, replay integrity and reproducible environment reset.

### Phase 2 — Baseline ladder and architecture reproduction

**Implement/measure:**

- uniform random policy;
- existing repository best baseline, after provenance resolution;
- feed-forward PPO baseline;
- recurrent PPO baseline;
- DreamerV3-style pixel baseline or official reproduction;
- official EMERALD reproduction/reference under its published config;
- ORBIT spatial JEPA candidate.

Run short smoke and throughput profiles first. Then equal interaction-budget screening. Maintain a separate exact-paper replication report and a fair equal-budget comparison report.

**Important:** EMERALD's official repository currently states CC BY-NC-SA 4.0. Do not copy its source into ORBIT without reviewing and complying with that license. For an independently distributed implementation, either comply with license obligations or implement a clean-room paper-based version without copying implementation text/code. Record provenance.

**Tests:**

- exact environment interactions counted;
- action repeat fixed to 1 for the canonical comparison;
- same preprocessing and reward wrapper;
- training updates and batch counts recorded;
- same development/locked seed partitions;
- model parameter count, wall-clock, FPS, host and GPU memory captured;
- baselines run under matching interaction budgets;
- no pretrained weights in the clean-room comparison; the separate DINOv2 arm is not part of this clean-room result and is tested in Phase 3.

**Exit:** select the initial from-scratch spatial world-model candidate based on prediction, task performance, stability and compute efficiency—not architecture preference. Final representation selection remains subject to the matched DINOv2 diagnostic in Phase 3.

### Phase 3 — Spatial representation qualification and DINOv2 diagnostic

**Implement:** the selected from-scratch encoder, spatial tokens, target EMA encoder, positional embeddings and latent normalisation. In a separate `PRETRAINED_DIAGNOSTIC` run, load frozen Meta AI DINOv2 ViT-S/14, apply the full-frame preprocessing contract in Section L, and connect its patch tokens through the explicit adapter to the same downstream ORBIT model. Do not fine-tune DINOv2 in this phase.

**Tests:**

1. Shape and range assertions for both scratch and DINOv2 paths.
2. Scratch target encoder has `requires_grad=False` and no parameter appears in an optimizer.
3. EMA scalar-update equation test.
4. Target weights change only after a valid EMA update.
5. Scratch online encoder receives nonzero gradients.
6. Effective rank, variance, covariance spectrum and token-wise variance diagnostics for each representation.
7. Representation stability on held-out sequences.
8. Checkpoint roundtrip preserves online and target weights exactly.
9. FP32 baseline before AMP; if AMP is enabled later, compare output/loss/gradient parity and NaN rates.
10. DINOv2 backbone stays frozen/eval; every backbone parameter has no gradient and no optimizer membership; its checkpoint hash is unchanged after training.
11. 64×64→224×224 preprocessing preserves the full frame, performs no crop, uses pinned nearest-neighbor interpolation and exact normalization; test corner markers so any lost border fails.
12. DINOv2 returns the expected 16×16 patch grid; adapter mapping to the shared ORBIT `[N,D]` shape is explicit, deterministic and tested.
13. Same downstream predictor/belief/loss/data/update budget is used for the paired scratch and DINOv2 runs; report differences in trainable parameters and measured compute.
14. Compare held-out one-step prediction to each arm’s own persistence baseline and include shuffled-action controls; do not directly compare unnormalized latent MSE across representation spaces.

**Exit:** the scratch representation core may be reported as passing only when there is no collapse, no silent target-update bug, reproducible output and useful spatial information. A full Phase 3/Gate 3 `PASS` also requires the matched DINOv2 report; it does not have to win. If it cannot be run, record the scratch-core result separately and leave the overall phase `PARTIAL/BLOCKED` with encoder selection provisional. The production/tabula-rasa encoder remains a from-scratch candidate unless a separate architecture decision explicitly changes that objective.

### Phase 4 — One-step action-conditioned world model

**Implement:** masked spatial predictor, action embedding, temporal belief (if enabled in selected candidate), reward predictor and continuation predictor.

**Tests:**

1. Synthetic tensor unit test for masked loss with manually calculated expected value.
2. Mask count is exactly the configured count; denominator is correct.
3. True-action prediction is better than shuffled-action control on held-out transitions.
4. Model beats persistence baseline on a preregistered metric with paired uncertainty.
5. Predictor receives gradients; target encoder does not.
6. Every trainable parameter intended to learn appears exactly once in an optimizer.
7. Each module's parameters change after its designated training test.
8. Reward/continuation labels align to the correct transition.
9. Per-head loss, gradient norms and prediction errors are logged separately.
10. No NaN/Inf in outputs, losses, gradients or optimizer states.

**Exit:** reproducible one-step action-conditioned prediction that beats persistence on held-out data and passes gradient coverage.

### Phase 5 — Multi-step prediction and learned belief

**Implement:** sequence training, recurrent unroll, burn-in and closed-loop latent rollout.

**Tests:**

1. Horizons 1, 2, 4, 8 are evaluated on held-out sequences.
2. Teacher-forced action-sequence evaluation is separate from closed-loop self-fed prediction.
3. Model-vs-persistence ratio for each horizon.
4. Closed-loop rollout test feeds predicted tokens and predicted rewards into the next recurrent step.
5. Burn-in reconstructs hidden state using current weights.
6. Episode reset test catches hidden leakage between episodes.
7. Memory-on vs memory-off ablation on history-dependent probes.
8. GRU/temporal core has nonzero gradients and demonstrably improves a held-out memory probe.
9. Gradient flow through unrolled prediction is verified with `gradcheck`/finite-difference or a smaller analytic synthetic test where practical.
10. Error growth, representation norm and uncertainty over horizon logged.

**Exit:** the model remains useful over the rollout horizons actually intended for planning. If only H=1 or H=2 passes, planning is limited to that horizon or postponed.

### Phase 6 — Object-based world-model research

Implement D1-A through D1-D as isolated variants using the same data protocol.

**Tests:** slot stability, duplicate slots, object re-identification, transition prediction, same-budget throughput and downstream ablation. No semantic training labels.

**Exit:** object-based variant improves a preregistered held-out metric over the selected spatial baseline. Otherwise leave slots out of core.

### Phase 7 — Active rule discovery

Implement D2-A through D2-E. First validate the uncertainty estimate on held-out prediction errors; then run action-selection experiments. Add hidden-mechanics variants only in a separate environment version with evaluator-only rule manifests.

**Tests:** uncertainty calibration, information-gain estimate against observed error reduction, random/disagreement controls, rule-change detection, sample efficiency, held-out variant adaptation and no rule metadata leakage.

**Exit:** active experiments improve actual adaptation efficiency and task outcomes over controls. Otherwise do not claim rule discovery.

### Phase 8 — Reusable skills

Implement DIAYN/CIC/trajectory segmentation experiments after the perception and replay gates.

**Tests:** skill controllability, repeatability, diversity, composability, held-out start-state generalisation and downstream task-performance ablation.

**Exit:** discovered skills improve real Crafter performance or transfer at equal environment steps. If only latent diversity improves, no promotion.

### Phase 9 — Curriculum generation

Implement training-only seed/level selection and PAIRED/ACCEL-inspired experiments after a stable student exists.

**Tests:** environment validity, solvability, generated-level distribution, no student leakage, equal-budget performance on locked standard Crafter seeds, and transfer to a separate held-out variant suite.

**Exit:** improvement on unmodified Crafter and/or declared OOD benchmark at equal interaction budget. Improvement solely on teacher-generated levels is insufficient.

### Phase 10 — Imagination-trained policy

Only after multi-step gates pass, implement actor-critic training from imagined states. Begin with the shortest validated horizon; do not set a long horizon just because the code permits it.

**Tests:**

1. Actor entropy from full action probabilities, never from sampled one-hot actions.
2. Action distribution and movement statistics.
3. Actor/critic/world-model gradient isolation.
4. Value target calibration and reward-value correlation.
5. Real-vs-imagined prediction gap.
6. Actor update improves actual held-out return versus a policy trained without imagination.
7. Survival and per-achievement metrics do not regress unacceptably.
8. Advantage normalizer state persists in checkpoints.
9. Slow/target critic update exactness.
10. No mode collapse, NaNs or optimizer-state corruption.

**Exit:** imagination yields statistically supported improvement on real held-out episodes. A better imagined return without a real-world improvement is a failure.

### Phase 11 — Final scaling and promotion

Scale only after every smaller gate passes. Report 1M-step and 10M-step experiments separately. Do not imply 1M and 10M results are the same compute setting.

Final report must include five independent training seeds when compute permits, a predeclared locked evaluation manifest, per-achievement rates, bootstrap confidence intervals, IQM/probability of improvement, training cost and all artifact hashes. If fewer than five training seeds are feasible, explicitly report the limitation; do not claim statistical certainty.

---

## 11. Debugging and observability: end-to-end requirements

Debugging is a first-class system implemented from Phase 0. It must make it possible to identify the earliest faulty component in a failed decision or training update.

### 11.1 Run manifest

Every run writes an immutable manifest containing:

- run ID and experiment ID;
- git commit SHA and dirty-tree status;
- config file and canonical config SHA-256;
- checkpoint SHA-256;
- environment package/version and wrapper version;
- Python/PyTorch/CUDA/cuDNN versions;
- device model, total RAM and VRAM;
- all random seeds and generator states;
- opaque action-permutation hash;
- replay schema version and data-shard hashes;
- exact train/eval seed manifest IDs;
- environment-step count and optimizer-update count;
- precision mode and deterministic-kernel settings.

A checkpoint path without a content hash is not a checkpoint identity.

### 11.2 Always-on scalar telemetry

Log per optimizer update:

- total and per-component losses, both unweighted and weighted;
- learning rates and optimizer step counts;
- global gradient norm and per-module gradient norm;
- count of parameters with `grad is None`, non-finite gradients and zero gradient;
- parameter delta norm per module;
- online/target encoder distance and EMA delta; for Candidate E, separately log frozen DINOv2 backbone hash plus adapter online/target distance and delta;
- latent mean, standard deviation, minimum/maximum and effective rank;
- per-token variance and masked/unmasked prediction errors;
- prediction error at horizons 1/2/4/8;
- persistence and shuffled-action comparison;
- reward and continuation calibration;
- ensemble disagreement and calibration if enabled;
- actor entropy, action distribution and unique actions if policy exists;
- replay occupancy, sequence validity, episode mix and age distribution;
- host RAM, GPU memory, utilization, FPS, update throughput and data-loader wait;
- checkpoint duration and bytes written.

Missing telemetry keys must cause the gate to fail. `NaN`, `Inf`, a stale value or an empty series must not be reported as a valid zero.

### 11.3 Per-decision trace

For a selected subset of episodes and all failure snapshots, persist:

- episode ID, step index and environment reset ID;
- observation image or exact observation bytes plus SHA-256;
- previous action, previous reward and hidden-state reset marker;
- opaque action token, full action probability vector/logits and sampling RNG state;
- recurrent-state norm and checksum;
- spatial-token norms and a selected latent visualization;
- predicted next latent summary and uncertainty;
- reward/continuation predictions;
- actual next observation hash, actual reward and terminal flags;
- prediction residual by token and feature group;
- model/checkpoint version used for that decision.

Do not dump every full activation for every training step. Use scalar telemetry continuously, sampled deep traces at fixed intervals, and automatic deep traces on gate failure. Define retention and storage budgets.

### 11.4 Model inspection tools

Build inspectable outputs for:

- input frame and normalized tensor;
- spatial token grid, positional embeddings and pooled vector;
- EMA target vs online encoder token maps;
- frozen DINOv2 spatial patch-token grid and the explicit adapter output grid, when Candidate E is enabled;
- masked positions and predicted target positions;
- real vs predicted latent maps by horizon;
- error heatmap by token location;
- belief-state norm and change over time;
- ensemble member outputs and disagreement;
- object slots and slot-attention assignment maps, only when D1 is enabled;
- actor action probabilities and sampled action;
- critic value and reward prediction;
- imagination trajectories against actual held-out trajectories.

Every visualization must include run ID, checkpoint hash, seed, episode and step. Never produce a plot that silently mixes different checkpoint versions.

### 11.5 Deterministic failure replay

For a failure, create a minimal replay bundle containing:

1. checkpoint hash and complete model/optimizer/RNG state;
2. exact initial observation and preceding observation/action/reward history needed to rebuild belief;
3. action permutation manifest;
4. failing action logits/probabilities and selected token;
5. actual next observation, reward and terminal flags;
6. predicted next latent/reward/continuation;
7. replay data indices and episode sequence metadata;
8. config/code/environment hashes.

A replay tool must rerun the model decision and verify hashes. If exact reproduction is impossible due environment nondeterminism, state which component differs and do not label it deterministic.

### 11.6 Gradient and optimizer debugger

At initialization and at least once per training phase:

- enumerate all trainable named parameters;
- assert each intended parameter is in exactly one intended optimizer;
- assert frozen target parameters are in no optimizer;
- run a synthetic batch through the intended loss;
- verify expected parameter groups receive finite, nonzero gradients;
- execute one optimizer step and verify parameters change;
- verify no unexpected module receives gradients;
- check optimizer state for NaN/Inf.

A model component is not considered trained merely because it is registered in an optimizer.

### 11.7 Replay debugger

- Validate every sequence with `(episode_id, step_in_episode)`.
- Verify action/reward/next-frame alignment.
- Check no window crosses episode boundaries.
- Check ring wrap and save/load ordering.
- Validate terminated/truncated masks.
- Log sample provenance: shard ID, episode ID, start step, tier/priority if any.
- Provide a CLI to print and render a sampled sequence.
- Provide an audit that detects stale hidden states, duplicate frame references and missing final observations.

### 11.8 Checkpoint and resume debugger

Checkpoint must contain:

- online encoder;
- EMA target encoder;
- belief/dynamics/outcome heads;
- actor and critic if implemented;
- every optimizer and scheduler state;
- EMA update count;
- all running normalisers;
- replay metadata and shard hashes (large replay payload may be separate);
- all Python/NumPy/PyTorch/CUDA/environment RNG states;
- environment episode-boundary state or an explicit “resume at next episode” policy;
- global environment steps and optimizer updates;
- experiment/config/code hashes.

Atomic write order: write temp file → fsync if supported → rename/replace → verify new artifact → only then prune old checkpoint. Never delete the only valid checkpoint before verifying the new one.

Use step-crossing schedules (`if step >= next_save_step`) rather than modulo checks that can miss events when vectorized steps skip exact multiples.

### 11.9 Failure taxonomy and earliest-divergence report

Every failure report must assign a category and provide evidence:

1. input/observation contract;
2. data leakage;
3. action mapping or action sampling;
4. replay alignment/corruption;
5. spatial encoder/representation collapse;
6. target EMA update;
7. predictor/action-conditioning;
8. recurrent belief/gradient flow;
9. reward/continuation target;
10. uncertainty calibration;
11. actor/critic objective;
12. planner/imagination drift;
13. curriculum/data distribution;
14. checkpoint/resume;
15. evaluation/reporting.

The report must identify the earliest timestamp/module where expected and actual values first diverge. Do not jump directly to “planner issue” if the predicted latent was already wrong.

---

## 12. Evaluation protocol and promotion rules

### 12.1 Seed partitions

Use three disjoint seed sets, generated once and stored in immutable manifests:

- **Training seeds:** used only for experience collection and training.
- **Development seeds:** may be used for architecture/hyperparameter selection.
- **Locked final seeds:** never used for training, architecture selection, early stopping or threshold tuning.

Do not assume old development seeds are still held out. Seeds `2050..2099` have already appeared in v3 development/screening artifacts, so they must not be re-labelled as a pristine final set.

The actual locked seed list should be generated and committed as a manifest before final training. Its contents may be kept away from the training process. Use paired seed lists for candidate comparisons.

### 12.2 Evaluation levels

| Level | Purpose | Episodes / runs | Promotion status |
|---|---|---|---|
| Unit/synthetic | Verify math and code paths | Deterministic test cases | Required; no gameplay claim |
| Smoke | Catch crashes and shape bugs | 1–3 episodes | Never promotion evidence |
| Development screen | Detect gross regressions | 10–20 episodes per checkpoint | Tuning only |
| Phase gate | Validate a new module | At least 30 fixed dev episodes plus module-specific held-out sequences | Unlocks next phase only if all gates pass |
| Final evaluation | Compare promoted models | 100 fixed episodes per trained model, with independent training seeds where feasible | Official result |

These episode counts are protocol choices, not statistical guarantees. If variance remains large, increase episodes and/or training seeds. Report confidence intervals regardless.

### 12.3 Baseline comparisons

All candidate comparisons must match:

- environment implementation/version;
- reward wrapper;
- observation resolution/preprocessing; for the DINOv2 arm, report the exact full-frame resize/interpolation/normalization separately and do not conceal this transform;
- action-repeat setting (use 1 for canonical comparison);
- environment-step budget;
- training-seed count and evaluation seed list;
- evaluation maximum episode length;
- deterministic/stochastic policy mode;
- checkpoint selection rule;
- model-selection budget and number of tuning trials.

Report the exact number of environment transitions, not only optimizer updates. For parallel environments, distinguish vector steps from total environment transitions. For scratch-versus-DINOv2, also match the downstream model, replay sample sequence, training updates and evaluation seeds; disclose that the frozen pretrained backbone has a large external pretraining advantage and report total/frozen/trainable parameter counts plus throughput and memory. Keep the results in separate `TABULA_RASA` and `PRETRAINED_DIAGNOSTIC` namespaces.

### 12.4 Primary and secondary outcomes

Primary outcome for standard Crafter is the official 22-achievement geometric-mean score, calculated exactly as in §8.9.

Secondary metrics:

- 22 per-achievement success rates;
- unique achievement types unlocked across evaluation;
- mean achievements per episode;
- mean and median environment return;
- survival length distribution;
- resource/tool progression measured only by evaluator;
- episode completion/death causes measured only by evaluator;
- action diversity and movement statistics;
- training interactions, throughput and compute budget.

### 12.5 Promotion policy

A module is promoted only if:

1. All unit, integration, leakage and debug tests pass.
2. Its preregistered primary metric is computed from a valid immutable artifact.
3. It beats the declared control on the same held-out data or achieves its explicitly defined non-inferiority target.
4. The paired confidence interval supports the claim; uncertainty is reported.
5. No critical regression occurs in survival, other achievement categories, model calibration or runtime budget.
6. The gain is reproduced in an independent training run or explicitly marked provisional.

No “best dev checkpoint” is promoted based only on a single high-scoring seed. No failed gate can be bypassed by changing its threshold after seeing the run. Any threshold change creates a new experiment version and requires a fresh validation run.

---

## 13. Phase-by-phase debugging checklist

| Phase | Minimum debug evidence before exit |
|---|---|
| 0 | Score-formula tests, result provenance diff, checkpoint hash, config hash, code/environment manifest |
| 1 | Observation/action boundary trace, no-leak tests, permutation hash, replay alignment, termination/truncation audit |
| 2 | Throughput profile, parameter count, full config, same-budget baseline report, memory profile |
| 3 | Token/target visualizations, EMA delta, latent rank/variance, gradient coverage, checkpoint parity |
| 4 | Mask map, prediction residual heatmap, true-vs-shuffled action test, persistence baseline, per-head gradients |
| 5 | Belief trace, burn-in verification, hidden reset, multi-horizon model-vs-persistence, closed-loop GIF/trace |
| 6 | Slot assignments, slot churn, object persistence probes, object-on/off ablation |
| 7 | Hypothesis/uncertainty trace, selected experiment, observed update, hidden-rule manifest separation |
| 8 | Skill code distribution, action/transition traces, termination rates, downstream ablation |
| 9 | Curriculum selection log, generated-level validity, solvability test, student performance and OOD result |
| 10 | Actor entropy/logits, critic calibration, imagination-vs-real traces, gradient isolation, survival/score ablation |
| 11 | Final score recomputation, bootstrap report, all seed manifests, hashes, per-achievement table and artifact audit |

---

## 14. Required package layout

The following v1.0 package diagram is historical reference only. v1.2 Sections K–L and the binding amendment control: first audit and extend the existing `true_jepa/` line; do not create a duplicate package until an explicit architecture decision approves it.

```text
orbit/
├── README.md
├── SPEC.md                         # link/summary of this brief
├── configs/
│   ├── base.yaml
│   ├── candidates/
│   └── experiments/
├── src/
│   ├── api.py                       # strict policy contract
│   ├── env/
│   │   ├── adapter.py               # sanitized Gym boundary
│   │   ├── action_permutation.py    # opaque action mapping
│   │   └── episode.py               # reset/termination semantics
│   ├── models/
│   │   ├── encoder.py
│   │   ├── dinov2_adapter.py          # Phase 3 diagnostic only; frozen external backbone is not the primary encoder
│   │   ├── target_encoder.py
│   │   ├── belief.py
│   │   ├── spatial_predictor.py
│   │   ├── outcome_heads.py
│   │   └── agent.py
│   ├── data/
│   │   ├── frame_store.py
│   │   ├── sequence_replay.py
│   │   └── provenance.py
│   ├── training/
│   │   ├── world_model.py
│   │   ├── actor_critic.py          # added only in Phase 10
│   │   └── schedules.py
│   ├── research/
│   │   ├── object_modeling/         # added in Phase 6
│   │   ├── rule_discovery/          # added in Phase 7
│   │   ├── skill_discovery/         # added in Phase 8
│   │   └── curriculum/              # added in Phase 9
│   ├── evaluation/
│   │   ├── crafter_score.py
│   │   ├── evaluator.py
│   │   ├── statistics.py
│   │   └── report.py
│   └── debug/
│       ├── telemetry.py
│       ├── trace.py
│       ├── replay_failure.py
│       ├── gradient_audit.py
│       └── checkpoint_audit.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── leakage/
│   ├── replay/
│   ├── numerical/
│   ├── determinism/
│   └── phase_gates/
└── artifacts/
    ├── manifests/
    ├── parameter_ledger.jsonl
    ├── experiment_register.jsonl
    └── reports/
```

Large checkpoints and replay shards belong in a model-specific storage directory, not Git. Every artifact still needs hashes and an immutable manifest.

---

## 15. Instructions to the implementing AI/developer

These rules are mandatory. They exist specifically to prevent assumptions, silent substitutions and invented values.

### 15.1 No-assumption rule

- Treat this brief as the controlling specification.
- Do not infer a missing architectural decision from personal preference or general ML convention.
- Do not invent hyperparameters, thresholds, shapes, schedule values, reward weights, rollout horizons or benchmark seeds.
- If a required detail is missing, create a short `DECISION_REQUIRED` entry with: ambiguity, options, relevant evidence, required experiment and proposed default. Stop work on the dependent module until the decision is approved.
- A default from an existing repository config must be labelled “inherited candidate”; it is not automatically an ORBIT-approved value.
- A paper value must include paper/version/source-code commit and whether it is from the paper or a public reimplementation.
- Preserve exact units: environment transitions, vector steps, optimizer updates, frames, episodes, percent and fraction are not interchangeable.

### 15.2 No-silent-change rule

- Do not weaken or delete a test to make implementation pass.
- Do not change a gate threshold after observing results without creating a new experiment version and fresh validation.
- Do not change model shape while resuming an old checkpoint.
- Do not resume optimizer state after changing loss definition, normalizer semantics, action mapping, target EMA schedule or replay schema unless a specifically tested migration is approved.
- Do not silently treat missing telemetry as zero.
- Do not catch exceptions and continue training with defaults.
- Do not automatically fall back to a different observation shape, action count or model config.

### 15.3 Implementation sequence rule

For every component:

1. Write the interface and tensor-shape contract.
2. Write unit tests and numerical expected values first.
3. Implement the smallest module.
4. Run unit tests.
5. Run gradient coverage and parameter-update tests.
6. Run one deterministic integration batch.
7. Run a short smoke episode.
8. Generate debug artifacts and inspect them.
9. Run the phase gate.
10. Only then start the next dependent component.

### 15.4 Required implementation report after each phase

The AI must produce:

- files added/changed;
- interface and tensor shape table;
- exact equations implemented;
- config and parameter ledger diff;
- test command and full pass/fail summary;
- gradient audit summary;
- memory/throughput measurements;
- debug artifacts produced;
- failures encountered and root cause evidence;
- gate decision (`PASS`, `FAIL`, or `BLOCKED`);
- checkpoint/config/code hashes;
- any deviations, each approved or unresolved.

A phase cannot be marked complete with unresolved `DECISION_REQUIRED`, missing tests or unexamined failures.

---

## 16. Known risks and explicit mitigations

| Risk | Why it matters | Required mitigation |
|---|---|---|
| Spatial model is accurate at one step but drifts in imagination | A planner can exploit hallucinations | Multi-horizon, closed-loop and real-vs-imagined gap gates before policy training |
| Pretrained DINOv2 underperforms on Crafter pixel art | Natural-image features may not preserve game-specific visual transitions; resizing can distort the pixel grid | Keep it frozen and diagnostic-only first; audit preprocessing; compare to persistence and scratch under matched downstream training; do not treat a negative result as proof the scratch encoder is sound |
| Reward head learns health rewards but not achievement progress | Return can improve without broad capability | Report official per-achievement metrics; do not add named achievement shaping |
| Object slots are unstable | Skills/memory may bind to the wrong entity | Slot persistence tests and slot-free ablation |
| Uncertainty is not calibrated | Active exploration can chase noise | Held-out calibration and random/disagreement controls |
| Skill discovery maximizes diversity only | Skills may not help solve the game | Downstream policy and transfer ablation |
| Curriculum teacher exploits environment quirks | Student learns generated-level artefacts | Final unmodified Crafter evaluation and held-out variant tests |
| Recurrent state is stale in replay | Model sees inconsistent history | Sequence burn-in with current weights; no stale hidden reuse by default |
| Episode termination is confused with time-limit truncation | Incorrect value bootstrap and continuation learning | Verify API and unit-test both cases |
| High-dimensional covariance regularization is rank deficient | Wrong/unstable collapse loss | Rank/sample audit; do not blindly use covariance loss |
| Checkpoint is only a path, not a content identity | Reproducibility failure | SHA-256, code/config/env/action-map manifests |
| Replay duplicates next frames | RAM overrun | Unique frame store and exact byte budget |
| Evaluation is tuned on final seeds | Inflated performance | Disjoint immutable seed manifests |
| External code has incompatible license | Reuse/distribution risk | License audit before copying/adapting code or weights |

---

## 17. Definition of done for ORBIT v1

ORBIT v1 is not “done” when it has all seven blocks in a diagram. It is done only when:

- the clean environment boundary passes all leakage tests;
- baseline and evaluator are reproducible;
- selected from-scratch spatial JEPA design beats simple predictive controls on held-out sequences;
- the frozen DINOv2 diagnostic has a complete matched report and is kept separate from the primary tabula-rasa result;
- action-conditioned prediction is proven by controlled tests;
- the chosen rollout horizon passes closed-loop fidelity gates;
- recurrent memory is trained and shown to improve a history-dependent capability;
- every activated research direction has a valid control and ablation;
- imagination, if used, improves actual held-out gameplay;
- achievement score and per-achievement rates are independently recomputed;
- all reported results include hashes, seed manifests, confidence intervals and compute budgets;
- a failure can be replayed and traced to the earliest faulty module using the debug tools.

**No score target is part of the definition of done.** A scientifically valid negative result is acceptable; a high score with an unverified data path, hidden rule or irreproducible evaluation is not.

---

## 18. First work item — what to implement before the model

Do not begin with the encoder. The first pull request should implement only:

1. repository isolation and package skeleton;
2. canonical environment wrapper and strict policy API;
3. sanitized data boundary and opaque action mapping;
4. run/checkpoint/config hash manifests;
5. replay schema and sequence integrity tests;
6. exact Crafter score evaluator and unit tests;
7. baseline evaluation runner;
8. debug telemetry schema and deterministic failure-bundle format.

Then reproduce the baseline and resolve the existing result discrepancy. Only after Phase 0 and Phase 1 pass should the team implement candidate world models.

---

## 19. Source and artifact index

### Project evidence

- Root README: https://github.com/sahiltolani30/learn-a-world/blob/main/README.md
- Root results: https://github.com/sahiltolani30/learn-a-world/blob/main/RESULTS.md
- Prometheus roadmap: https://github.com/sahiltolani30/learn-a-world/blob/main/prometheus/ROADMAP.md
- Prometheus math audit: https://github.com/sahiltolani30/learn-a-world/blob/main/prometheus/MATH_AUDIT.md
- Prometheus spatial predictor: https://github.com/sahiltolani30/learn-a-world/blob/main/prometheus/src/models/predictor.py
- True JEPA config: https://github.com/sahiltolani30/learn-a-world/blob/main/true_jepa/config.py
- True JEPA model: https://github.com/sahiltolani30/learn-a-world/blob/main/true_jepa/model.py
- MOSAIC plan: https://github.com/sahiltolani30/learn-a-world/blob/main/mosaic/PLAN.md
- MOSAIC status: https://github.com/sahiltolani30/learn-a-world/blob/main/mosaic/STATUS.md
- ATLAS v3 empirical bottleneck report: https://github.com/sahiltolani30/learn-a-world/blob/main/docs/ATLAS_V3_EMPIRICAL_RESULTS_AND_BOTTLENECK_REPORT.md

### Research

- Crafter benchmark paper/project: https://danijar.com/project/crafter/
- EMERALD paper: https://proceedings.mlr.press/v267/burchi25a.html
- EMERALD official repository/license: https://github.com/burchim/EMERALD
- DreamerV3 paper: https://arxiv.org/abs/2301.04104
- V-JEPA 2 paper: https://arxiv.org/abs/2506.09985
- TD-MPC2 paper: https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html
- Slot Attention: https://research.google/pubs/object-centric-learning-with-slot-attention/
- DIAYN: https://research.google/pubs/diversity-is-all-you-need-learning-skills-without-a-reward-function/
- CIC: https://arxiv.org/abs/2202.00161
- PAIRED: https://research.google/pubs/emergent-complexity-and-zero-shot-transfer-via-unsupervised-environment-design/
- ACCEL: https://proceedings.mlr.press/v162/parker-holder22a.html
- Mars: https://proceedings.neurips.cc/paper_files/paper/2024/hash/1fb6d0b52f5e41b11392841a66dbfe7d-Abstract-Datasets_and_Benchmarks_Track.html
- RLiable: https://github.com/google-research/rliable

---

**End of specification.** Any implementation that violates the anti-cheating contract, bypasses a gate, changes an undocumented value or reports an untraceable result is not an ORBIT-compliant implementation.
