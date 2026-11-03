# JEPA Crafter Agent — Complete Implementation Plan

**Document type:** Engineering implementation specification  
**Status:** Revision 1.3 — complete engineering specification. Core AEON source implementation was present in the audited repository snapshot; current source, runtime tests, and learning performance must be re-verified against the latest commit before implementation or benchmark claims. See Sections 22–29 for the source audit and relationship-learning additions, and Sections 30–58 for the comprehensive observability and forensic-debugging specification.  
**Primary environment:** Crafter (`CrafterReward-v1` and `CrafterNoReward-v1`)  
**Target hardware:** Free Google Colab, one NVIDIA T4 where available  
**Primary objective:** Build and validate a from-scratch visual world model that learns action-conditioned dynamics, autonomously discovers conditional action–effect relationships from experience, stores and reuses those relationships across episodes and worlds, composes them into plans, and continues exploring for new knowledge.

---

## 0. Instructions to the implementation agent

Treat this document as the source of truth. Implement the specified design in stages and do not silently replace it with a conventional PPO/DQN agent, a pretrained vision model, a scripted Crafter solver, or a collection of hard-coded game rules.

The implementation must satisfy all of the following:

1. **No pretrained weights.** Initialize every trainable model parameter randomly. Do not download a pretrained encoder, world model, policy, embedding model, or game-playing checkpoint.
2. **No game-mechanics database.** Do not encode crafting recipes, inventory requirements, object names, resource values, action meanings, technology trees, routes, or action sequences in the learner.
3. **No privileged training inputs.** The learning system may receive only the RGB observation, the opaque action token it issued, the next RGB observation, the single scalar environment reward, and episode termination/truncation flags.
4. **No semantic action names.** The learner receives anonymous integer action tokens. The environment adapter may translate tokens into legal Crafter action indices, but it must not expose action names or descriptions to the model.
5. **JEPA is the central learned dynamics model.** The main predictive objective must be future-latent prediction against a stop-gradient EMA target encoder. Pixel reconstruction is not the primary objective.
6. **The model must be action-conditioned.** The predicted future latent must depend on the current latent state, recent context, and the issued action token(s).
7. **Experience must survive episode resets and notebook restarts.** Save and restore model weights, EMA target weights, optimizer state, replay buffer, counters, random-number states, and the action-token mapping.
8. **Separate learning from scoring.** Crafter `info` fields and achievement names may be read by a separate evaluator for metrics only. They must never reach the learner, planner, replay buffer, reward model, curiosity model, or training logs consumed by the learner.
9. **Prove that the model learns.** A falling training loss alone is not evidence. Implement the held-out prediction tests, action-conditioning tests, collapse tests, planning tests, and behavioural evaluations defined below.
10. **Fail visibly.** If an acceptance test fails, stop progression to the next stage, save diagnostics, identify the failure, and do not label the model as successful.

If a dependency or API version differs from this plan, write a compatibility adapter and a test. Do not change the learning contract to accommodate a library difference.

---

## 1. Project objective and success definition

### 1.1 What is being built

Build one shared agent system containing:

- A visual encoder trained from random initialization.
- An EMA target visual encoder.
- An action-conditioned JEPA predictor that forecasts future visual latents.
- A temporal context state that summarizes recent observations and actions.
- A persistent replay memory containing raw visual experience.
- A compact ensemble of predictive heads to estimate model disagreement.
- An intrinsic exploration mechanism based on novelty and model learning signals.
- A latent-space planner that evaluates candidate action-token sequences using the learned model.
- A set of logical exploration roles that share the same world model and replay memory.
- A separate evaluator that measures Crafter achievements without leaking them into training.

The model is not expected to produce human-readable sentences such as “a table is required before crafting a tool.” Its first requirement is to learn predictive representations that capture the relevant visual and action-dependent regularities. Human-readable concepts are optional diagnostic interpretations, not training labels.

### 1.2 What counts as success

There are three independent success dimensions. Report them separately.

| Dimension | Required evidence | What does not count |
|---|---|---|
| **World-model learning** | Held-out future-latent prediction improves over persistence and action-ablated baselines; predictions respond to action changes; representations do not collapse. | A low training loss by itself. |
| **Useful memory and planning** | The saved model resumes correctly; replay improves held-out prediction; imagined action outcomes correlate with actual outcomes; MPC decisions outperform the same agent without model-based planning. | A planner that only exploits imaginary rewards. |
| **Game-playing ability** | Crafter achievement success rates and score improve over random and over the project’s own ablations on matched interaction budgets. | High episode reward alone, or a few attractive gameplay videos. |

The intended result is a system that improves through experience. Do not claim that it has human-level cognition or that JEPA guarantees discovery of every hidden relationship.

### 1.3 Project milestones

1. Environment interface and information boundary are correct.
2. A randomly initialized JEPA predicts held-out future latents better than simple baselines.
3. Action conditioning is demonstrably useful.
4. Training and experience persist across episodes and notebook restarts.
5. Exploration produces data that improves both world-model quality and task performance.
6. MPC decisions are useful in the real environment, not only in imagination.
7. The multi-role population improves data coverage or performance over one role.
8. The final 1M-step evaluation is repeatable and fully logged.

Do not skip directly to MPC or multi-agent training before milestones 1–4 pass.

---

## 2. Benchmark facts and environment contract

Crafter is a procedurally generated visual survival environment. The official project describes 64×64×3 RGB observations and a categorical action space with 17 actions. The benchmark defines 22 achievements and a 1-million-environment-step interaction budget. The published score is based on achievement success rates, not raw total reward. See the reference links in Section 18.

### 2.1 Permitted learner inputs

For every environment time step, the learner may receive exactly:

```text
observation_rgb: uint8[64, 64, 3]
action_token:    int in [0, 16]       # action chosen by this agent
next_rgb:        uint8[64, 64, 3]
reward:          float32              # one scalar only
terminated:      bool
truncated:       bool
```

The training transition is:

```python
Transition(
    obs=observation_rgb,
    action=action_token,
    reward=reward,
    next_obs=next_rgb,
    terminated=terminated,
    truncated=truncated,
    episode_id=episode_id,  # sampler boundary metadata only
    timestep=timestep,      # sampler boundary metadata only
)
```

`episode_id` and `timestep` are allowed only to keep sequences contiguous and prevent sampling across episode resets. They must not be model features.

### 2.2 Forbidden learner inputs

The following must not be read by any training component:

- `info` dictionaries or nested values.
- Achievement names, achievement flags, achievement unlock events, or achievement counters.
- Inventory dictionaries, internal inventory counts, player coordinates, map arrays, object IDs, terrain IDs, entity state, health fields extracted from internal state, crafting state, or game source internals.
- Action names, action descriptions, action requirement tables, keyboard labels, or semantic action mappings.
- Any text generated by OCR or image-template matching from the observation.
- Any manually written Crafter recipes, dependency graphs, expert trajectories, or scripted action sequences.

The visible HUD and inventory artwork are part of the RGB observation. The model may learn from the pixels as any visual agent would. It must not receive a decoded inventory representation.

### 2.3 Reward handling

The main run uses the scalar reward emitted by `CrafterReward-v1`. The Crafter documentation describes sparse achievement reward and small health-related reward changes. The learner receives only the final scalar, never a breakdown or event label.

A second run must use `CrafterNoReward-v1`, with the external-reward input set to zero. That run evaluates whether the intrinsic exploration system can learn without achievement reward. Keep reward-enabled and reward-free results in separate experiment records.

### 2.4 Opaque action tokens

The environment has 17 legal discrete actions. The learner must not receive their names or meanings.

At the beginning of each independent training run:

1. Create a random permutation of the integers `0..16` using the run’s seeded adapter RNG.
2. The agent chooses an opaque token `0..16`.
3. The adapter maps that token through the saved permutation to the environment’s discrete action index.
4. The model stores and learns from only the opaque token it selected.
5. Save the permutation in the run configuration and checkpoint so resumed training and evaluation use the same mapping.
6. For a new independent seed/run, create a new permutation.

The permutation must never be converted into names or descriptions in model input. It is an interface mapping, not a game-mechanics lookup table. The action token mapping must remain stable within one run so the agent can learn from consequences.

### 2.5 Environment adapter API compatibility

Crafter’s documented API follows the Gym style. The adapter must normalize both older Gym and newer Gymnasium-style step/reset signatures without changing the learner’s observation contract.

Implement:

```python
reset_env(seed) -> obs_rgb
step_env(action_token) -> (next_rgb, scalar_reward, terminated, truncated)
```

For an older four-value `step()` result, derive the episode-end flags from the returned `done` value and the adapter’s time-limit handling. For a newer five-value result, pass through `terminated` and `truncated`. `info` may be routed only to the evaluator process; discard it before returning to the learner.

On startup, assert:

```text
observation shape == (64, 64, 3)
observation dtype is uint8 or safely converted to uint8
action_space.n == 17
all emitted opaque tokens are in [0, 16]
```

Do not assume a library version. Install Crafter in Colab, record actual package versions, run the smoke tests, then pin the tested versions in `requirements.txt`.

---

## 3. System architecture

### 3.1 High-level data flow

```text
                 ┌────────────────────────────┐
                 │     Crafter Environment    │
                 └──────────────┬─────────────┘
                                │ RGB, scalar reward, end flags
                                ▼
                 ┌────────────────────────────┐
                 │    Blind Environment      │
                 │       Adapter             │
                 │ - anonymous action tokens │
                 │ - strips info from learner│
                 └──────────────┬─────────────┘
                                │
                                ▼
                 ┌────────────────────────────┐
                 │     Logical Explorer(s)    │
                 │ random / novelty / MPC    │
                 └──────────────┬─────────────┘
                                │ transitions
                                ▼
                 ┌────────────────────────────┐
                 │  Persistent Replay Buffer  │
                 │ raw RGB + token + scalar   │
                 └──────────────┬─────────────┘
                                │ contiguous sequences
                                ▼
         ┌────────────────────────────────────────────┐
         │               JEPA Learner                 │
         │                                            │
         │ Online Encoder ──► Context State           │
         │       │                   │                │
         │       └──────► Action-conditioned Predictor│
         │                             │              │
         │ EMA Target Encoder ◄────────┘              │
         │       │                                    │
         │       └── stop-gradient target latents     │
         └───────────────────┬────────────────────────┘
                             │
                             ▼
         ┌────────────────────────────────────────────┐
         │   Reward / Termination / Novelty Signals   │
         │   Predictor disagreement and latent memory │
         └───────────────────┬────────────────────────┘
                             │
                             ▼
         ┌────────────────────────────────────────────┐
         │   Latent Model Predictive Controller       │
         │   candidate token sequences → imagined     │
         │   latents → score → execute first token    │
         └────────────────────────────────────────────┘

         Separate side channel:
         Crafter info ──► Evaluator only ──► metrics/videos
                            (never returned to learner)
```

### 3.2 One shared world model, not four separate brains

The initial population uses one shared visual encoder, target encoder, predictor ensemble, replay buffer, and planner. The “agents” are exploration roles that generate different kinds of experience. They are not four independently trained world models.

This is important for the T4 budget: four complete models would multiply memory and training cost without first proving that multiple models help. The population is introduced only after a single-role agent is stable.

### 3.3 Core learning loop

For an observation sequence, the system:

1. Encodes the current and historical RGB frames.
2. Embeds the opaque action tokens.
3. Uses recent observations and actions to form a temporal context state.
4. Predicts future target-encoder latents conditioned on candidate or observed actions.
5. Compares predicted future latents with stop-gradient EMA target latents.
6. Updates the online encoder and predictor; updates the target encoder with EMA.
7. Stores the raw interaction in persistent replay.
8. Uses novelty and predictor disagreement to seek informative experience.
9. Once the model passes validation gates, uses MPC to select actions in latent space.
10. Repeats after every real environment step; MPC replans from the latest actual observation.

---

## 4. Repository structure

Create the following structure. Do not put all logic in one notebook or one Python file.

```text
jepa-crafter/
├── README.md
├── requirements.txt
├── configs/
│   ├── crafter_t4.yaml
│   ├── crafter_t4_reward_free.yaml
│   └── evaluation.yaml
├── notebooks/
│   ├── 00_environment_smoke_test.ipynb
│   ├── 01_collect_random_data.ipynb
│   ├── 02_train_jepa.ipynb
│   ├── 03_train_agent.ipynb
│   └── 04_evaluate_and_ablate.ipynb
├── src/
│   ├── env/
│   │   ├── crafter_adapter.py
│   │   ├── opaque_actions.py
│   │   ├── learner_observation_filter.py
│   │   └── evaluator_tap.py
│   ├── data/
│   │   ├── replay_buffer.py
│   │   ├── sequence_sampler.py
│   │   ├── episode_store.py
│   │   └── checkpoint_store.py
│   ├── models/
│   │   ├── visual_encoder.py
│   │   ├── ema_encoder.py
│   │   ├── temporal_context.py
│   │   ├── action_embedding.py
│   │   ├── latent_predictor.py
│   │   ├── predictor_ensemble.py
│   │   ├── auxiliary_heads.py
│   │   └── jepa_world_model.py
│   ├── learning/
│   │   ├── jepa_losses.py
│   │   ├── learner.py
│   │   ├── optimizer.py
│   │   ├── feature_diagnostics.py
│   │   └── update_scheduler.py
│   ├── exploration/
│   │   ├── latent_memory.py
│   │   ├── novelty.py
│   │   ├── disagreement.py
│   │   ├── role_definitions.py
│   │   └── action_selector.py
│   ├── planning/
│   │   ├── latent_rollout.py
│   │   ├── cem_planner.py
│   │   ├── candidate_scorer.py
│   │   └── planner_diagnostics.py
│   ├── evaluation/
│   │   ├── achievement_recorder.py
│   │   ├── crafter_score.py
│   │   ├── heldout_prediction.py
│   │   ├── action_conditioning_tests.py
│   │   ├── relation_diagnostics.py
│   │   └── report.py
│   ├── training/
│   │   ├── collect.py
│   │   ├── train_world_model.py
│   │   ├── train_agent.py
│   │   └── resume.py
│   └── utils/
│       ├── config.py
│       ├── seeding.py
│       ├── logging.py
│       ├── device.py
│       └── safety_checks.py
├── tests/
│   ├── test_env_contract.py
│   ├── test_no_info_leak.py
│   ├── test_opaque_actions.py
│   ├── test_replay_sequences.py
│   ├── test_ema_update.py
│   ├── test_jepa_shapes.py
│   ├── test_no_episode_crossing.py
│   ├── test_checkpoint_resume.py
│   ├── test_planner_replans.py
│   └── test_evaluator_isolation.py
├── runs/
│   └── .gitkeep
└── artifacts/
    └── .gitkeep
```

The notebooks are thin orchestration and visualization layers. Put reusable behaviour in `src/` and tests in `tests/`.

---

## 5. Exact model specification — T4-sized first version

The first version should be compact and testable rather than trying to reproduce a multi-billion-parameter research model. It must nevertheless implement a genuine JEPA-style objective.

### 5.1 Input preprocessing

- Input: RGB observation, shape `[B, 64, 64, 3]`, `uint8`.
- Convert to channel-first `[B, 3, 64, 64]`.
- Convert to `float32` in `[0, 1]` inside the model pipeline.
- Do not resize or crop the default Crafter observation.
- Do not apply OCR, object detectors, texture classifiers, image captions, or semantic segmentation.
- Use the exact same deterministic preprocessing for online and target encoders.

### 5.2 Visual encoder

Implement a small convolutional encoder from random initialization.

**Required output:** a spatial feature grid of shape `[B, 64, 128]` (an 8×8 grid with 128 channels).

Recommended fixed architecture:

| Layer | Operation | Output shape excluding batch |
|---|---|---|
| Input | RGB image | `3 × 64 × 64` |
| Stem | 3×3 convolution, stride 2, 64 channels; GroupNorm; GELU | `64 × 32 × 32` |
| Block 1 | two 3×3 residual convolutions, 64 channels | `64 × 32 × 32` |
| Downsample 1 | 3×3 convolution, stride 2, 96 channels; GroupNorm; GELU | `96 × 16 × 16` |
| Block 2 | two 3×3 residual convolutions, 96 channels | `96 × 16 × 16` |
| Downsample 2 | 3×3 convolution, stride 2, 128 channels; GroupNorm; GELU | `128 × 8 × 8` |
| Block 3 | two 3×3 residual convolutions, 128 channels | `128 × 8 × 8` |
| Tokenize | flatten spatial grid and transpose | `64 × 128` |
| Position | add learned 8×8 two-dimensional positional embedding | `64 × 128` |

Use GroupNorm rather than BatchNorm to avoid dependence on small or variable batch statistics. Use GELU. Keep the output grid; do not reduce the whole frame to a single vector before prediction.

Implement an attention-pooling function over the 64 tokens to produce a global vector `u_t` of shape `[B, 128]` for temporal context and auxiliary predictions. The spatial grid remains the main JEPA target.

### 5.3 Online and target encoders

- `online_encoder = E_theta` receives gradients.
- `target_encoder = E_ema` has `requires_grad=False` and runs in evaluation mode.
- Initialize target weights as an exact copy of online weights.
- After every successful optimizer update, update target weights with EMA:

\[
\bar\theta \leftarrow m\bar\theta + (1-m)\theta
\]

Start with `m = 0.996`; linearly increase toward `0.999` over the first 50,000 learner updates. Make the momentum configurable.
- Compute target features under `torch.no_grad()`.
- Do not allow target branch gradients to flow into the online encoder or predictor.

### 5.4 Anonymous action embedding

- `num_action_tokens = 17`.
- Learn an embedding table of shape `[17, 64]` from random initialization.
- The embedding index is the opaque action token, not a semantic name.
- Do not add textual action descriptions or action-specific hand-coded features.
- The adapter’s random token-to-environment permutation is saved outside the model.

### 5.5 Temporal context state

Use a GRU to retain recent temporal information.

- Input per time step: concatenation of pooled visual latent `[128]` and action embedding `[64]`, total `[192]`.
- GRU hidden size: `256`.
- Context window used for training samples: the most recent 4 observation/action pairs.
- At episode start, initialize hidden state to zeros.
- Never carry a GRU state from one episode into another.
- The replay sampler must provide the burn-in context needed to reconstruct the hidden state. Do not rely on an uncheckpointed hidden state from a previous notebook cell.

For training, compute the context hidden state from observed past latents/actions. For a simulated rollout, update it from the predicted next latent and the simulated action. This allows the model to imagine more than one step into the future.

### 5.6 Action-conditioned latent predictor

Create three independently initialized predictor members. They share the visual encoder and temporal context module, but each has independent predictor parameters. The members receive independent bootstrap masks during learning. This compact ensemble is used for disagreement estimates; it is not three full agents.

Each predictor member:

1. Receives current spatial latent tokens `[64,128]`.
2. Receives temporal context `[256]`.
3. Receives the current action embedding `[64]`.
4. Projects context and action to 128-dimensional conditioning vectors.
5. Uses feature-wise conditioning (for example, FiLM) on the current visual tokens.
6. Processes the conditioned tokens through 2 Transformer encoder blocks, each with 128 hidden dimensions, 4 attention heads, and a 512-dimensional MLP.
7. Outputs predicted next-frame latent tokens `[64,128]`.

Do not feed any future observation or future target latent into the predictor. The future target is used only by the loss function.

### 5.7 Autoregressive latent rollout

For a sequence of future action tokens, the model must be able to recursively forecast:

\[
\hat Z_{t+1}=P(Z_t,h_t,e(a_t))
\]

\[
\hat u_{t+1}=\operatorname{Pool}(\hat Z_{t+1})
\]

\[
h_{t+1}=\operatorname{GRU}([\hat u_{t+1};e(a_t)],h_t)
\]

Then feed `Z_hat_(t+1)`, `h_(t+1)`, and the next candidate action token into the predictor to obtain `Z_hat_(t+2)`. Continue for the requested horizon.

The rollout function must be differentiable during training and may run under `no_grad()` during planning. Include a unit test proving that changing the candidate action token can change the predicted latent output for at least some held-out contexts after training.

### 5.8 Auxiliary prediction heads

Add small heads from the predicted pooled latent/context to predict:

- Next-step scalar reward (one scalar regression output).
- Next-step episode termination probability (one sigmoid probability).

These heads are auxiliary. They do not replace JEPA and do not receive achievement labels. Train the reward head only on the scalar reward that the adapter emits. Train termination on the `terminated OR truncated` bit.

Do not create heads for inventory items, crafting recipes, terrain categories, object identities, or achievement names.

### 5.9 Expected size and memory discipline

- Aim for a compact model in the low-single-digit to modest-single-digit millions of parameters. Record the actual parameter count at startup.
- Print GPU memory use after model allocation and after the first training batch.
- If memory use is too high, reduce batch size from 32 to 16 before changing the architecture.
- Do not increase image size, hidden dimensions, or ensemble count during the first milestone.

---

## 6. JEPA training objective

### 6.1 Main objective

Let:

- `O_t` be the current RGB observation.
- `Z_t = E_theta(O_t)` be online spatial latent tokens.
- `Zbar_(t+k) = E_ema(O_(t+k))` be target spatial latent tokens.
- `h_t` be temporal context derived from earlier observations and actions.
- `a_t ... a_(t+k-1)` be the opaque future action tokens.
- `Zhat_(t+k)` be the predictor’s imagined future latent after rolling out those actions.

The target must be stop-gradient. Normalize the feature vectors before computing the main loss.

Use cosine feature-prediction loss over a random 50% subset of future target spatial tokens:

\[
L_{JEPA}^{(k)} = \frac{1}{|M|}\sum_{i\in M}\left(1-\cos(\hat z_{t+k,i},\operatorname{sg}(\bar z_{t+k,i}))\right)
\]

`M` is a random subset of spatial target-token positions. The target encoder computes the full future representation, but no future target feature is passed into the predictor. The mask determines which predicted target positions contribute to the loss.

Train with rollout horizons `k ∈ {1, 2, 4, 8}`. Use normalized horizon weights:

```text
raw weights:       [1.00, 0.50, 0.25, 0.125]
normalized weights: divide each by their sum
```

The full JEPA loss is the weighted sum across horizons and the average across the three predictor members, with each member receiving its own replay bootstrap mask.

### 6.2 Auxiliary losses

Use:

```text
L_total = L_JEPA
        + 0.10 * L_reward
        + 0.05 * L_termination
```

- `L_reward`: smooth L1 loss on the scalar next-step reward.
- `L_termination`: binary cross-entropy on episode-end prediction.
- These coefficients are starting values and must remain fixed for the first controlled experiment. Tune only after the baseline is reproducible.

Do not add an achievement classifier, object detector, inventory predictor, recipe predictor, language loss, pixel decoder, or imitation loss.

### 6.3 Anti-collapse monitoring

The target encoder and EMA reduce instability but do not remove the need for collapse checks. Log every 500 learner updates:

- Mean and standard deviation of latent feature values.
- Per-channel feature standard deviation.
- Average pairwise cosine similarity between different observations.
- Effective rank / participation ratio of the latent covariance matrix on a fixed held-out batch.
- Predictor-to-target cosine similarity.

Raise a hard warning if feature standard deviation approaches zero, pairwise cosine similarity approaches 1 for unrelated observations, or effective rank collapses sharply. Save diagnostic plots. Do not hide collapse by reporting only the loss.

If collapse occurs:

1. Verify target gradients are disabled.
2. Verify the target EMA update is happening once per optimizer update.
3. Verify observations are not duplicated or normalized incorrectly.
4. Verify the action and target time alignment.
5. Reduce the learning rate by 2×.
6. Only then test a small variance-floor regularizer on online features. Record it as a separate experiment, not an invisible change.

### 6.4 Optimizer defaults

Initial configuration:

| Parameter | Default |
|---|---:|
| Optimizer | AdamW |
| Encoder learning rate | `1e-4` |
| Predictor/context learning rate | `3e-4` |
| Auxiliary-head learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Batch size | `32` |
| Gradient clipping | global norm `1.0` |
| Warm-up | first `1,000` learner updates |
| Schedule | cosine decay after warm-up |
| Precision | FP16 autocast + GradScaler on T4 |
| Training sequence | 4 context steps + up to 8 prediction steps |
| Random target-token loss mask | 50% |

If a T4 or Colab runtime cannot support batch 32, lower it to 16, then 8. Do not silently remove temporal horizons to make the run fit. Log the effective batch size and actual number of optimizer updates.

### 6.5 Exact training-sequence alignment and update pseudocode

This subsection removes ambiguity about which action predicts which frame.

For a sampled trajectory, observations and actions are indexed as:

```text
observations: O[0], O[1], ..., O[T]
actions:     A[0], A[1], ..., A[T-1]
rewards:     R[0], R[1], ..., R[T-1]
end_flags:   D[0], D[1], ..., D[T-1]
```

`A[j]` is the action taken while the environment was displaying `O[j]`; it leads to `O[j+1]` and emits `R[j]` and `D[j]`.

For a prediction origin `t`:

- The observed context contains `O[t-3] ... O[t]` (four frames), when available.
- The context GRU receives each observed pooled latent and the action that led into that observation. For `O[t-3]`, use a learned `START` token or zero previous-action vector; for later context frames use `A[t-4] ... A[t-1]` as applicable.
- The model is asked to predict the consequences of `A[t]`, `A[t+1]`, ..., `A[t+7]`.
- The target for one-step prediction is the EMA latent of `O[t+1]`, not `O[t]`.
- The targets for horizons 2, 4, and 8 are the EMA latents of `O[t+2]`, `O[t+4]`, and `O[t+8]`.
- The reward and end-flag target for the first step are `R[t]` and `D[t]`.

If an episode begins before four context frames are available, left-pad the context with a learned start embedding and a context-validity mask. Do not draw missing context frames from the previous episode.

**Training-batch pseudocode:**

```python
# Batch contains valid contiguous episode slices only.
# obs_context: [B, 4, 3, 64, 64]
# actions_future: [B, 8]        opaque action tokens
# obs_future_targets: frames at offsets 1, 2, 4, 8
# rewards_first: [B, 1]
# done_first: [B, 1]

online_context = online_encoder(obs_context)      # [B, 4, 64, 128]
pooled_context = attention_pool(online_context)    # [B, 4, 128]
h_t = context_gru_over_observed_history(
    pooled_context,
    previous_action_tokens,
    context_validity_mask,
)

with torch.no_grad():
    target_1 = ema_encoder(obs_future_at_1)
    target_2 = ema_encoder(obs_future_at_2)
    target_4 = ema_encoder(obs_future_at_4)
    target_8 = ema_encoder(obs_future_at_8)

for member in predictor_ensemble:
    z_hat = online_context[:, -1]  # last observed spatial latent grid
    h = h_t
    predictions = {}
    auxiliary_first_step = None

    for k in range(1, 9):
        a = action_embedding(actions_future[:, k - 1])
        z_hat = member.predict_next(z_hat, h, a)
        h = context_gru_update(h, attention_pool(z_hat), a)
        if k == 1:
            auxiliary_first_step = auxiliary_heads(z_hat, h)
        if k in {1, 2, 4, 8}:
            predictions[k] = z_hat

    # Compare predictions with stop-gradient EMA targets at matching offsets.
    # Apply the member's bootstrap mask and the random 50% target-token mask.
    loss_member = weighted_jepa_loss(
        predictions,
        {1: target_1, 2: target_2, 4: target_4, 8: target_8},
        bootstrap_mask=member_bootstrap_mask,
        target_token_mask=random_target_mask,
    )

    if auxiliary_first_step is not None:
        loss_member += 0.10 * smooth_l1(
            auxiliary_first_step.reward, rewards_first
        )
        loss_member += 0.05 * binary_cross_entropy(
            auxiliary_first_step.done_probability, done_first
        )

loss = average_over_ensemble_members(loss_member)
optimizer.zero_grad(set_to_none=True)
scaled_loss = grad_scaler.scale(loss)
scaled_loss.backward()
grad_scaler.unscale_(optimizer)
clip_global_gradient_norm(all_trainable_parameters, 1.0)
grad_scaler.step(optimizer)
grad_scaler.update()
update_target_encoder_ema()
```

The pseudocode is normative for temporal alignment. Helper-function implementation may differ, but the action-to-next-observation relationship and future-target boundaries must remain identical.

**Required alignment tests:** create a tiny synthetic sequence with unique frame markers and action markers. Verify that changing `A[t]` changes the predictor input for `O[t+1]`, and that the target at horizon 4 is exactly `O[t+4]`. Add a test that rejects a sample if any `D[j]` is true for `j` between the sample start and the final target transition.

### 6.6 Minimum quantitative learning gates

Use these as initial engineering acceptance thresholds, not as claims about published JEPA performance. Record all raw values as well as pass/fail.

At the 50k-environment-step JEPA pilot checkpoint:

1. Held-out one-step cosine distance must be at least **5% lower relative to the persistence baseline** on average across 3 development seeds.
2. On the paired action-sensitive test set, action-conditioned prediction error must be at least **5% lower relative to the no-action control** and lower than the shuffled-action control.
3. At least 3 of the 4 evaluated horizons (1, 2, 4, 8) must outperform persistence on held-out data. The 8-step horizon may have higher absolute error than the 1-step horizon.
4. No representation-collapse warning may be active. The report must include feature standard deviations and effective rank, not just a boolean.
5. The result must hold on held-out episodes, not only on training replay.

If a threshold is missed, mark the gate as **FAIL / INVESTIGATE** and do not unlock MPC. The implementation agent may recommend a separate follow-up experiment, but it must not silently lower the threshold after seeing the results.

---

## 7. Replay memory and persistent experience

### 7.1 Replay buffer

Implement a circular sequence-aware replay buffer. Default capacity: 50,000 observations/transitions. Store RGB frames as `uint8`, not `float32`.

Store:

```text
observations:  uint8 [capacity + 1, 64, 64, 3]
actions:       uint8 [capacity]
rewards:       float32 [capacity]
terminated:    bool [capacity]
truncated:     bool [capacity]
episode_ids:   int32 [capacity]
timesteps:     int32 [capacity]
write_pointer: int
size:          int
```

Store each frame once. A transition references its current observation slot and next observation slot. Do not duplicate both images in every transition record. Make wrap-around safe: a sequence sample must not cross a ring-buffer overwrite or episode boundary.

At 50,000 RGB frames, raw image storage is approximately 0.6 GB. Log actual memory consumption. Do not start with a million-frame in-RAM buffer on free Colab.

### 7.2 Sequence sampling rules

- Sample a contiguous sequence containing at least 4 context transitions plus 8 future transitions.
- Never allow a sampled sequence to cross `terminated` or `truncated`.
- Never join the last frame of one episode to the reset frame of the next episode.
- If a sampled transition is too close to a boundary, reject and resample.
- Keep episode IDs only for sampling and evaluation bookkeeping; do not pass them into the model.
- Use a mixture of recent transitions and older replay so the system does not forget earlier episodes.

Recommended sampler: 70% uniform-over-valid-sequences and 30% recent-sequence sampling. Keep this fixed for the first baseline.

### 7.3 Persistent memory layers

The project has three distinct forms of memory:

1. **Parametric memory:** learned weights in the online encoder, predictor, context model, and auxiliary heads.
2. **Target memory:** the EMA encoder, which stabilizes the target feature space.
3. **Experience memory:** replayed RGB/action/reward transitions across episodes.

These are the required persistent memory systems. Do not claim that an LSTM state surviving a single episode is long-term memory. Episode-specific recurrent hidden state must reset; learned parameters and replay must not.

### 7.4 Checkpoint contents

Every checkpoint must include:

```text
online encoder state_dict
EMA target encoder state_dict
context GRU state_dict
action embedding state_dict
all three predictor state_dicts
auxiliary head state_dicts
optimizer state_dict
GradScaler state_dict
learner update counter
environment step counter
replay buffer arrays and pointers
replay episode/timestep metadata
random action-token permutation
Python / NumPy / PyTorch RNG states
configuration snapshot
current exploration-role schedule
normalization statistics for reward / novelty / disagreement
```

Save atomically: write to a temporary file, verify it can be loaded, then rename it to the final checkpoint name. Save a rolling checkpoint every 10,000 environment steps and before a planned Colab disconnect.

### 7.5 Resume validation

After loading a checkpoint, assert:

- The action-token permutation is identical.
- Model and target parameters match the saved state.
- Replay size and write pointer are restored.
- Optimizer state and update count are restored.
- The next sampled sequence is valid and does not cross an episode boundary.
- A short deterministic inference pass produces the same outputs as immediately before saving, within a small numerical tolerance.

The agent must not treat a notebook restart as a new training run.

---

## 8. Exploration and logical agent population

### 8.1 Do not start with four independent agents

Start with one explorer and a random warm-up. Add multiple logical roles only after the shared JEPA is demonstrably learning. The population shares the same learned dynamics model and replay buffer.

### 8.2 Exploration signals

Implement three generic, game-agnostic signals:

**A. Latent novelty**  
Compare the current pooled latent with a small memory of previously visited pooled latents. Use k-nearest-neighbour distance or a distance-to-nearest-centroid calculation. Keep this memory in the learned latent space; do not attach semantic labels. Because the encoder changes during training, refresh stored embeddings periodically using the current EMA target encoder from stored raw observations, or limit the novelty memory to the current episode plus a refreshed reservoir.

**B. Predictor disagreement**  
For the same latent context and action, compare the three independently trained predictor outputs. Use average pairwise cosine distance or variance after feature normalization. Treat this as an epistemic-uncertainty proxy, not a calibrated probability. Cap extreme values and normalize with running statistics.

**C. Learning progress**  
Track prediction error over repeated visits to similar latent regions. Prefer regions where prediction error is improving as the model receives more data, rather than permanently rewarding unpredictable or noisy observations. Keep a short and long moving average; use their positive improvement as an optional bonus.

Do not use raw prediction error as an unlimited curiosity reward. That can make an agent seek random or chaotic situations that never become understood.

### 8.3 Intrinsic reward

Initial intrinsic reward:

\[
r_t^{int}=\beta_n\,\tilde n_t+\beta_d\,\tilde d_t+\beta_p\,\tilde p_t
\]

where:

- `n_t` is latent novelty,
- `d_t` is predictor disagreement,
- `p_t` is positive learning progress,
- tildes indicate clipped, running-normalized values.

For the first agent experiment, set `beta_n = 1.0`, `beta_d = 0.25`, and `beta_p = 0.25`. These are initial coefficients, not established optimal values. Log each component separately. Do not mix intrinsic and external rewards without preserving both columns in diagnostics.

For the reward-enabled run, use the scalar external reward plus a controlled intrinsic term. For the reward-free run, external reward is exactly zero.

### 8.4 Four logical exploration roles

After single-agent baseline passes, implement these roles. They do not get special knowledge of Crafter actions or recipes.

| Role | Generic objective | How it selects experience |
|---|---|---|
| **Explorer** | Visit underrepresented latent states. | Higher weight on latent novelty. |
| **Scientist** | Reduce uncertainty in the learned dynamics. | Higher weight on predictor disagreement, with a cap and a penalty for repeated non-learning noise. |
| **Experimenter** | Collect diverse action-conditioned transitions. | Prefer contexts/actions with high expected information gain and balanced token usage; no fixed action script. |
| **Balanced planner** | Combine predicted external reward, novelty, and survival/termination prediction. | Uses MPC with a balanced score. |

Implementation rules:

- Use one world model and one replay buffer.
- Give each role a separate exploration coefficient profile, not separate game rules.
- Do not hard-code “role A should collect wood” or any semantic objective.
- Each role may run in its own CPU environment worker, but all must use the same saved opaque action-token mapping for that training run.
- On a single T4, one learner process owns the GPU. Environment workers run on CPU and write transitions to a queue/replay buffer.
- Begin with one environment worker. Scale to four only after checking CPU throughput, queue stability, and learner utilization.
- Log the number of steps, unique episode seeds, latent coverage, and model improvement attributable to each role.

Ablate the population against one Explorer role. Keep it only if it improves held-out model quality, achievement breadth, or data efficiency under the same total environment-step budget.

---

## 9. Action selection and latent model-predictive control

### 9.1 Staged action selection

Do not use MPC from the first environment step. A random or novelty-driven warm-up is needed to obtain initial transition data.

1. **Warm-up:** uniform random opaque tokens.
2. **Early learning:** sample actions using a mixture of random choice and a generic novelty/disagreement score.
3. **Model-based control:** enable MPC only after world-model validation gates pass.
4. **Replanning:** after every real environment step, encode the new actual observation and plan again. Never execute an entire imagined sequence without observing the real environment.

### 9.2 Candidate scoring

For each candidate action-token sequence, roll the model forward for horizon `H` and estimate:

- Scalar reward prediction.
- Intrinsic novelty.
- Predictor disagreement / information value.
- Probability of episode termination.
- Model uncertainty penalty for long, unreliable imagined rollouts.

Use the initial score:

\[
J=\sum_{k=0}^{H-1}\gamma^k\left(\hat r_{t+k}+\lambda_n\hat n_{t+k}+\lambda_i\hat i_{t+k}\right)
-\lambda_T\,\hat p_{term}
-\lambda_U\,U_{rollout}
\]

Use `gamma = 0.99` initially. Normalize each score component by running statistics before combining. Keep all coefficients in the YAML config and log them with every evaluation.

Do not reward uncertainty without limit. A high disagreement score is useful only when it encourages informative exploration and does not cause endless pursuit of unstable predictions. Cap the per-step intrinsic bonus and use the termination/rollout-risk penalties.

### 9.3 Planner algorithm

Implement discrete-action Cross-Entropy Method (CEM) for the first MPC version.

Initial settings:

| Parameter | Default |
|---|---:|
| Action vocabulary | 17 opaque tokens |
| Planning horizon | 8 steps |
| Candidate population | 64 sequences |
| CEM iterations | 2 |
| Elite sequences | 8 |
| Replanning | every actual environment step |
| Executed portion | first action token only |
| Model evaluation | ensemble disagreement measured for candidates |

Each CEM iteration:

1. Maintain a categorical probability distribution over 17 tokens at each future time index.
2. Sample 64 candidate sequences from that distribution.
3. Roll each candidate sequence through the latent world model.
4. Score the candidates using the objective above.
5. Select the top 8 elites.
6. Update each time-index categorical distribution from elite token frequencies, with smoothing so no token probability becomes exactly zero.
7. Repeat for 2 iterations.
8. Execute only the first token of the highest-scoring sequence.
9. Receive the actual next RGB observation, discard the imagined future, and plan again.

The implementation must support a `random_shooting` fallback for debugging, but CEM is the primary planner.

### 9.4 Planner validation gate

Enable MPC only when all these checks pass on held-out real transitions:

- One-step latent prediction is better than persistence and no-action baselines.
- Action-conditioned prediction is better than the action-shuffled or action-ablated model on action-sensitive test cases.
- The reward head is not grossly miscalibrated; report calibration error and correlation rather than assuming it is accurate.
- The termination head is not confidently wrong on most held-out episodes.
- Free-running rollout error is measured at horizons 1, 2, 4, and 8.

If planner performance is worse than the same policy without MPC, disable MPC and return to data collection. Do not tune the planner against a single lucky seed.

### 9.5 Detecting model exploitation

For every real action selected by MPC, log:

- Predicted reward and actual reward.
- Predicted termination probability and actual termination.
- Predicted disagreement/uncertainty.
- Predicted future latent and the next observed target latent distance.
- Candidate score versus realized short-term outcome.

Compute the **imagination gap**: average predicted outcome minus actual observed outcome for selected actions. If the planner repeatedly selects actions with high predicted benefit but poor realized outcome, reduce the rollout horizon, increase uncertainty penalty, or disable MPC pending model improvement.

### 9.6 Exact action-selection pseudocode

Use the following policy flow. `current_history` contains only actual observations and the opaque tokens previously chosen by this agent.

```python
def choose_action(obs, current_history, mode, role, model, planner):
    if mode == "warmup":
        return uniform_random_opaque_token(0, 16)

    if random_uniform_0_1() < RANDOM_ACTION_FLOOR:
        return uniform_random_opaque_token(0, 16)

    latent, context = model.encode_actual_history(obs, current_history)

    if mode == "mpc":
        return planner.plan_first_action(
            latent=latent,
            context=context,
            role_weights=ROLE_WEIGHTS[role],
        )

    # Pre-MPC exploration: score each opaque token using model predictions.
    scores = []
    for token in range(17):
        predicted_members = model.predict_one_step_for_all_members(
            latent, context, token
        )
        novelty = latent_memory.novelty(predicted_members.mean_latent())
        disagreement = ensemble_disagreement(predicted_members)
        progress = learning_progress_estimator.score(predicted_members)
        score = (
            ROLE_WEIGHTS[role].novelty * normalize_and_clip(novelty)
            + ROLE_WEIGHTS[role].disagreement * normalize_and_clip(disagreement)
            + ROLE_WEIGHTS[role].progress * normalize_and_clip(progress)
        )
        scores.append(score)

    # Sample rather than always taking argmax so exploration remains diverse.
    return categorical_sample(softmax(scores / EXPLORATION_TEMPERATURE))
```

Use these starting role weights:

```text
Explorer:         novelty=1.00, disagreement=0.25, progress=0.25
Scientist:        novelty=0.25, disagreement=1.00, progress=0.50
Experimenter:     novelty=0.50, disagreement=0.50, progress=1.00
Balanced planner: novelty=0.50, disagreement=0.25, progress=0.25
```

The Balanced planner’s MPC score additionally uses predicted scalar reward, termination risk, and rollout uncertainty. These are generic objectives; none encode a particular Crafter task or action meaning.

If no trained model is available, `mode` must remain `warmup`. If validation gates fail, the orchestrator must refuse to enable `mpc` and log the reason.

---

## 10. Data collection and training schedule

### 10.1 Development budgets

Use progressively larger budgets. Do not consume the full benchmark budget before the diagnostics work.

| Run | Environment steps | Purpose | Required result |
|---|---:|---|---|
| Smoke test | 500 | Verify environment, adapter, replay and logging. | No contract violations. |
| Random data pilot | 10,000 | Collect initial diverse data. | Valid episodes and replay sequences. |
| JEPA pilot | 50,000 total | Train the world model offline/online. | Held-out prediction improves over baselines. |
| Agent pilot | 100,000–200,000 | Test curiosity and gated MPC. | Improvement over matched random baseline. |
| Final run | 1,000,000 | Official-budget-scale Crafter experiment. | Full per-achievement report and score. |

The final 1M step run may need multiple Colab sessions. Checkpoint and resume; do not assume the runtime remains active.

### 10.2 Collection split

- Create fixed random seeds for training and a disjoint fixed set for held-out validation.
- Split by full episodes/seeds, never by randomly splitting individual frames from the same episode.
- Held-out episodes must never be inserted into replay or used for gradient updates.
- Keep a separate evaluator seed list for frozen-policy evaluation.
- Record exact seed lists in each run directory.

### 10.3 Initial warm-up

- Run uniform random opaque actions for the first 10,000 environment steps.
- Use the same adapter and replay path that the final system will use.
- Save all raw transitions.
- Train the JEPA learner after the first 2,000 transitions are available, but do not enable MPC in warm-up.
- Keep 10–20% of collected episodes as a fixed held-out prediction set; do not train on them.

The warm-up is a generic data bootstrap, not a hand-coded policy.

### 10.4 World-model updates

- Use one JEPA optimizer update every 4 environment steps during the initial T4 pilot.
- If the learner is comfortably faster than the environment and GPU memory is stable, test one update per environment step as a separate configuration.
- Batch size 32, reduce only as needed for memory.
- Use the sequence sampler in Section 7.
- Update target EMA only after a successful optimizer update.
- Log environment steps separately from optimizer updates.

### 10.5 Agent training

After the JEPA validation gate passes:

1. Start with Explorer role only.
2. Use novelty/disagreement-guided action selection while retaining a random-action floor of at least 10% for the pilot.
3. Enable MPC on a small fraction of decisions, then increase its use only if realized action outcomes validate its predictions.
4. Record all action-selection modes so results can be separated.
5. Add Scientist, Experimenter, and Balanced Planner only after the one-role baseline is stable.

### 10.6 Evaluation cadence

At 10k, 25k, 50k, 100k, and every additional 100k environment steps:

- Save a checkpoint.
- Evaluate held-out future-latent prediction.
- Evaluate action sensitivity and rollout error.
- Run the frozen policy for a fixed number of held-out episodes.
- Record per-achievement success, episode length, reward, termination reason, and survival indicators available to the evaluator.
- Save representative videos, including poor episodes, not only successful episodes.

---

## 11. Validation: how to prove the model is learning correctly

This section is mandatory. The implementation is incomplete until these tests run automatically and produce a machine-readable report.

### 11.1 Test A — environment and data-boundary correctness

**Tests:**

- Observation shape is exactly 64×64×3.
- Action vocabulary is exactly 17 opaque tokens.
- The learner never receives `info` or any nested `info` values.
- The learner never receives an action name or description.
- Reward is a single scalar.
- Sequence sampler never crosses episode boundaries.
- `episode_id`, seed, timestep, evaluator values, and action mapping are not model input tensors.

**Pass condition:** all unit tests pass. Any information leak is a hard failure and invalidates the experiment.

### 11.2 Test B — future-latent prediction on held-out episodes

Use entire held-out episodes that were never part of the training replay.

For each held-out sequence, compare the predicted latent with the EMA target latent at horizons 1, 2, 4, and 8.

Report:

- Mean cosine distance.
- Mean cosine similarity.
- Normalized feature MSE (secondary metric).
- Median and 90th-percentile prediction error.
- Error separately by horizon.
- Error on early versus late training checkpoints.

Compare against these baselines:

1. **Persistence baseline:** predict the current latent as the future latent.
2. **No-action baseline:** predictor receives current latent/context but no action embedding.
3. **Shuffled-action baseline:** evaluate or train an ablation with actions shuffled across samples while preserving sequence lengths.
4. **Random predictor baseline:** randomly initialized predictor with the same target encoder protocol, used only as a sanity baseline.

**Pass condition:** the trained action-conditioned model beats persistence and random baselines at one step, improves across training checkpoints on held-out episodes, and shows a measurable benefit over action-ablated/shuffled controls on action-sensitive cases. Do not demand that error always decreases at every horizon; report the expected increase in difficulty with horizon.

A model that only memorizes training episodes will fail the held-out split.

### 11.3 Test C — action conditioning is actually used

A common failure is a model that ignores action tokens and simply predicts that the scene remains similar. Test this explicitly.

**Test procedure:**

- Gather matched initial observations from reproducible environment resets or evaluator-only cloned states.
- From each matched state, execute different opaque action tokens in separate branches.
- Record the actual next RGB frame and target latent for each branch.
- Give the same initial observation/context to the predictor, changing only the candidate action token.
- Compare the predicted latent for each action with its corresponding actual target latent.

The evaluator may use deterministic resets or hidden state snapshots to make paired test cases. This mechanism is evaluator-only and must never be used during agent training or planning.

**Metrics:**

- Action-conditioned latent error.
- Action-shuffled latent error.
- No-action latent error.
- Fraction of paired cases where the model predicts the correct direction of latent change better than the ablations.
- Difference between predictions for different actions from the same context.

**Pass condition:** action-conditioned prediction must outperform the no-action and shuffled-action controls on the paired test set. If it does not, the model has not demonstrated that it learned action-conditioned dynamics.

### 11.4 Test D — representation collapse

Evaluate on a fixed batch of visually different held-out frames.

Report:

- Feature standard deviation per channel.
- Percentage of channels with near-zero standard deviation.
- Average pairwise cosine similarity between distinct frames.
- Effective rank / participation ratio.
- Feature norm distribution.

**Pass condition:** representations remain non-degenerate throughout training. A collapsed encoder is a hard failure even if the predictor loss is low.

### 11.5 Test E — multi-step rollout quality

For every held-out sequence:

1. Encode the initial observation.
2. Roll the world model forward using the actual action-token sequence recorded in the episode.
3. Compare each imagined future latent with the EMA target latent of the actual future observation.
4. Repeat at horizons 1, 2, 4, and 8.

Report model error growth with horizon and compare with the persistence baseline. Also report the gap between teacher-forced one-step predictions and fully autoregressive free-running predictions.

**Pass condition:** free-running predictions are meaningfully better than persistence for the tested horizons. A large teacher-forced/free-running gap must be reported and blocks long-horizon MPC, even if one-step prediction is good.

### 11.6 Test F — reward and termination-head calibration

On held-out transitions:

- Report reward prediction MAE and correlation.
- Plot predicted versus actual reward bins.
- Report termination AUROC and Brier score where class balance permits.
- Report calibration by predicted-probability bucket.

Do not use this as proof of world understanding. It is a check that auxiliary heads are usable and not dangerously miscalibrated.

### 11.7 Test G — memory retention across episodes and notebook restarts

Run two experiments with identical environment-step budgets:

- **Persistent run:** retain model weights, EMA, and replay across episode resets and checkpoint resume.
- **Reset-memory control:** intentionally reinitialize model/replay at episode boundaries for an ablation only.

Measure held-out prediction quality and behaviour after multiple episodes. Also save, terminate the Colab runtime, reload, and compare outputs and counters as described in Section 7.5.

**Pass condition:** the persistent run resumes exactly and demonstrates retained learning. The reset-memory control should expose the value, if any, of cross-episode memory.

### 11.8 Test H — latent features retain visual information

This is a diagnostic only, not a training objective.

Use evaluator-only annotations on a small held-out diagnostic set to train a simple linear probe on frozen latent embeddings. Example diagnostics include visible inventory quantities or broad visual state attributes. The annotations must never enter the JEPA training batch or policy input.

Report probe performance separately and clearly label it as post-hoc. A successful probe suggests information is present in the representation; it does not prove the agent reasons with that information.

### 11.9 Test I — conditional relationship learning

The goal is to test whether the model learns that the same action can have different effects depending on visual context, without giving the model a recipe table.

**Procedure:**

1. Collect visually distinct contexts from held-out Crafter worlds.
2. Identify matched test cases in which the same opaque action token produces different outcomes under different visible conditions.
3. Use the evaluator’s hidden state only to organize and score these diagnostic cases; never provide hidden fields or labels to the model.
4. Compare the trained action-conditioned model with no-action and action-shuffled baselines.
5. Measure whether the prediction changes appropriately with context and whether the corresponding real action outcome is better predicted.

The evaluator may later map action tokens to human-readable effects for a report, but that mapping must not be returned to the learner.

**Pass condition:** the context-plus-action model predicts held-out conditional outcomes better than context-only and shuffled-action controls. Avoid claiming a symbolic concept such as “recipe understanding” unless an independent diagnostic supports it.

### 11.10 Test J — model-based planning is useful in the real environment

Run matched evaluation seeds and equal environment-step budgets for:

- The same exploration system without MPC.
- The same system with MPC.
- A random-action baseline.

Report:

- Crafter score and 22 per-achievement success rates.
- External reward and intrinsic components separately.
- Episode length and termination distribution.
- Predicted-versus-actual reward error.
- Imagination gap.
- Number of MPC decisions and average planner latency.

**Pass condition:** MPC improves performance or data efficiency across multiple seeds without increasing model-exploitation failures. If it does not, disable MPC and investigate; do not report a planner win from one seed.

### 11.11 Test K — multi-role population adds value

Compare one Explorer role against the four-role population under the same total environment-step budget and same world-model capacity.

Report:

- Unique latent coverage.
- Predictor error reduction per 10k environment steps.
- Per-achievement success rates.
- Replay distribution by role.
- Role contribution to discoveries and successful episodes.

**Pass condition:** keep the population only if the multi-role setup shows repeatable improvement on a relevant metric. If not, use one role; multiple workers are not automatically better.

---

## 12. Crafter evaluation protocol

### 12.1 Official-style achievement metrics

Crafter defines 22 achievements. For each achievement `i`, calculate its success rate as the percentage of evaluated episodes in which that achievement occurred at least once.

For the published score, let `s_i` be the achievement success rate in percentage points from 0 to 100. Compute per run/seed:

\[
S = \exp\left(\frac{1}{22}\sum_{i=1}^{22}\ln(1+s_i)\right)-1
\]

The score is a geometric-mean aggregation. It rewards broad achievement coverage; increasing a rarely achieved difficult milestone matters more than slightly increasing an already frequent one.

Important implementation details:

- Count an achievement at most once per episode.
- Compute the score separately for each independent seed/run.
- Average the per-run scores only after calculating each run’s score.
- Also report all 22 per-achievement success rates. Never report only the aggregate score.
- In a 1M-step benchmark-style run, track achievements across the episodes completed within the 1M interaction budget.
- In addition, run a separate frozen-policy held-out evaluation and label it as a supplementary evaluation; do not mix its episodes into the benchmark-style training-episode statistics.

### 12.2 Evaluator isolation

The evaluator can consume `info` to record achievement events and other official metrics. It must be a distinct object/process with a one-way output path to metrics files. The learner-facing adapter must return only the permitted inputs from Section 2.1.

Add an automated test that deliberately inserts a fake secret field into `info` and verifies that no learner, replay, model, or planner object receives it.

### 12.3 Required baselines

At minimum compare:

1. Random opaque-action agent.
2. JEPA + single Explorer without MPC.
3. JEPA + single Explorer with MPC.
4. JEPA + four logical roles with MPC, only if the single-role system passes.
5. Reward-free JEPA exploration.

When compute permits, run the following ablations with equal architecture and interaction budget:

- No action embedding.
- Shuffled action tokens.
- No replay across episodes.
- No EMA target encoder.
- No intrinsic reward.
- No ensemble disagreement.
- One-step-only JEPA objective versus multi-horizon JEPA.
- No MPC.

Do not compare runs trained for different numbers of environment steps as if the comparison were controlled.

### 12.4 Random seeds and reporting

- Use at least 3 independent seeds for development decisions.
- Use 5 independent seeds for a final comparison when compute permits.
- Report mean and standard deviation across seeds.
- Show the exact environment steps, optimizer updates, wall-clock time, hardware, package versions, and checkpoint identifiers.
- Do not cherry-pick the best seed. Include every completed seed and document failed/interrupted runs.

---

## 13. T4 / free Colab execution plan

### 13.1 Runtime allocation

- GPU: one T4 where available.
- GPU use: model forward/backward, EMA encoder, latent prediction, MPC model rollouts.
- CPU use: Crafter environments, replay storage, metrics, video writing, evaluator.
- Start with one environment worker and one learner.
- Do not run multiple independent GPU learners.

### 13.2 Memory controls

- Store replay frames as `uint8`.
- Transfer only sampled minibatches to GPU.
- Use FP16 autocast and GradScaler.
- Do not keep entire replay observations as GPU tensors.
- Do not store redundant copies of current/next frames.
- Use batch size 32 initially; reduce to 16 or 8 only if required.
- Limit video recording to evaluation episodes or a small diagnostic subset.
- Save checkpoints to Google Drive or download them before runtime loss.

### 13.3 Colab notebook behaviour

Every notebook must be restart-safe:

- Read the config from `configs/`.
- Detect the latest valid checkpoint.
- Resume from checkpoint when requested.
- Verify replay and action permutation before collecting new data.
- Print current environment steps, learner updates, replay size, current role, and latest validation metrics at startup.
- Save a checkpoint periodically and on clean shutdown.

Do not depend on notebook variables surviving a runtime disconnect.

### 13.4 Throughput measurements

Log these separately:

- Environment steps per second.
- Learner updates per second.
- GPU utilization and peak allocated memory.
- CPU environment-worker utilization.
- Replay sampling time.
- Planner decision latency.

Only add environment workers if the learner can train without starving or overwhelming the queue. More workers are not automatically better on free Colab.

---

## 14. Default configuration file

Create `configs/crafter_t4.yaml` with at least the following keys and defaults. Every key must be parsed and validated; do not leave unused configuration values.

```yaml
project:
  name: jepa_crafter
  seed: 12345
  device: auto
  output_dir: runs/
  checkpoint_every_env_steps: 10000

environment:
  package: crafter
  env_id: CrafterReward-v1
  observation_shape: [64, 64, 3]
  action_count: 17
  opaque_action_permutation: random_per_run
  reward_mode: scalar
  max_environment_steps: 1000000

collection:
  initial_random_steps: 10000
  num_env_workers: 1
  train_seed_start: 1000
  heldout_seed_start: 900000
  heldout_episode_count: 50

replay:
  capacity: 50000
  observation_dtype: uint8
  recent_sequence_fraction: 0.30
  uniform_sequence_fraction: 0.70
  context_length: 4
  prediction_horizon: 8
  min_sequences_before_training: 2000

model:
  encoder_channels: [64, 96, 128]
  latent_tokens: 64
  latent_dim: 128
  action_embedding_dim: 64
  context_hidden_dim: 256
  context_length: 4
  predictor_members: 3
  predictor_transformer_blocks: 2
  transformer_heads: 4
  transformer_mlp_dim: 512
  prediction_horizons: [1, 2, 4, 8]
  target_mask_fraction: 0.50
  ema_momentum_start: 0.996
  ema_momentum_end: 0.999
  ema_momentum_ramp_updates: 50000

optimization:
  optimizer: AdamW
  encoder_lr: 0.0001
  predictor_lr: 0.0003
  auxiliary_lr: 0.0003
  weight_decay: 0.0001
  batch_size: 32
  gradient_clip_norm: 1.0
  warmup_updates: 1000
  precision: fp16
  updates_per_env_step: 0.25
  reward_loss_weight: 0.10
  termination_loss_weight: 0.05

exploration:
  novelty_weight: 1.0
  disagreement_weight: 0.25
  learning_progress_weight: 0.25
  random_action_floor: 0.10
  intrinsic_clip: 5.0

planner:
  enabled_after_validation: true
  algorithm: cem
  horizon: 8
  population: 64
  iterations: 2
  elite_count: 8
  discount: 0.99
  replan_every_real_step: true
  execute_first_action_only: true
  max_intrinsic_bonus_per_step: 2.0

population:
  enabled_after_single_role_validation: true
  roles: [explorer, scientist, experimenter, balanced_planner]
  shared_world_model: true
  shared_replay: true

evaluation:
  num_development_seeds: 3
  num_final_seeds: 5
  report_all_22_achievements: true
  heldout_prediction_horizons: [1, 2, 4, 8]
  evaluate_mpc_ablation: true
  save_videos: true
```

The config may be extended, but every extra key must have a documented effect and a test. Do not insert semantic Crafter knowledge into this file.

---

## 15. Implementation phases and acceptance gates

### Phase 0 — Environment and integrity

**Build:** `crafter_adapter.py`, `opaque_actions.py`, `learner_observation_filter.py`, evaluator tap, seed utilities, environment smoke notebook, unit tests.

**Tasks:**

- Install Crafter and record dependency versions.
- Verify 64×64 RGB and 17 discrete actions.
- Implement random opaque-action permutation.
- Normalize old/new Gym APIs.
- Strip `info` from learner-facing results.
- Create separate evaluator tap.
- Collect and save 500 random transitions.

**Acceptance gate:** all environment and leak tests pass; saved replay has correct shape/dtype; learner cannot access `info`.

### Phase 1 — Random experience and replay

**Build:** replay buffer, sequence sampler, episode store, checkpoint serialization.

**Tasks:**

- Collect 10k random transitions.
- Split training and held-out episodes.
- Validate contiguous sequence extraction.
- Test ring-buffer wrap-around.
- Save and reload a replay checkpoint.

**Acceptance gate:** 1,000 sampled sequences pass all boundary checks; save/reload preserves sampled contents and metadata.

### Phase 2 — JEPA model and offline prediction

**Build:** encoder, target encoder, context GRU, action embedding, three predictor members, auxiliary heads, losses, optimizer, feature diagnostics.

**Tasks:**

- Implement exact tensor shapes.
- Add shape and gradient-flow tests.
- Train using the random dataset.
- Run held-out horizons 1/2/4/8.
- Run persistence/no-action/shuffled-action baselines.
- Monitor collapse diagnostics.

**Acceptance gate:** no collapse; future latent prediction improves over persistence/random; action conditioning shows measurable benefit on paired action-sensitive tests. If not, debug this phase—do not build MPC.

### Phase 3 — Continual training and persistent memory

**Build:** collection loop, online learner scheduler, checkpoint manager, resume notebook.

**Tasks:**

- Train online as new transitions enter replay.
- Update EMA target correctly.
- Save checkpoints every 10k environment steps.
- Interrupt and resume a Colab runtime.
- Verify same action-token permutation and replay.

**Acceptance gate:** deterministic resume test passes; held-out prediction and learned weights persist across episode resets and runtime restart.

### Phase 4 — Intrinsic exploration

**Build:** latent memory, novelty, disagreement, learning progress, action selector.

**Tasks:**

- Add one Explorer role.
- Preserve at least 10% random actions during pilot.
- Log each intrinsic component separately.
- Compare latent coverage and prediction improvement against random collection.

**Acceptance gate:** exploration signals are finite, bounded, and non-degenerate; the Explorer data improves held-out model metrics or useful coverage over random data under matched budgets.

### Phase 5 — Latent MPC

**Build:** latent rollout, CEM planner, candidate scorer, planner diagnostics.

**Tasks:**

- Implement one-step and multi-step autoregressive rollout.
- Validate predicted versus actual latent outcomes.
- Run planning initially on a small proportion of decisions.
- Compare planner with no-MPC agent.
- Track imagination gap.

**Acceptance gate:** model passes Section 9.4; MPC is not enabled as the default policy unless matched multi-seed tests show real benefit.

### Phase 6 — Multi-role population

**Build:** role profiles and multi-worker collection queue.

**Tasks:**

- Add Scientist, Experimenter, and Balanced Planner roles.
- Keep one shared world model and replay.
- Assign CPU environment workers if throughput allows.
- Attribute all data to role IDs for analysis only; role ID is not a visual or dynamics feature.

**Acceptance gate:** population beats or complements the one-role baseline on a predeclared metric under the same total interaction budget. Otherwise revert to one role.

### Phase 7 — Full evaluation and ablations

**Build:** official achievement recorder, score calculation, held-out tests, report generator.

**Tasks:**

- Run 1M steps per final run/seed where compute permits.
- Run at least 3 independent seeds, target 5.
- Run reward-free experiment.
- Run essential ablations.
- Save metrics, configs, videos, checkpoints, package versions.

**Acceptance gate:** final report contains all 22 achievement rates, per-run score, mean/std across seeds, all model diagnostics, ablations, and complete compute accounting.

---

## 16. Failure modes and mandatory responses

| Symptom | Likely cause | Required response |
|---|---|---|
| JEPA train loss falls but held-out loss does not | Memorization, leakage, train/test overlap, or target mismatch | Recheck episode split and sequence sampler; use new held-out seeds; do not advance. |
| Action-conditioned model equals no-action baseline | Model ignores actions or action alignment is wrong | Verify transition indexing, opaque token path, action embedding gradients, and paired action test. |
| Predictions barely change for different actions | Action token not used, predictor underpowered, or action data too narrow | Check token distribution and action embedding; collect broader experience; keep action-conditioning test mandatory. |
| Features collapse | Target/online instability, incorrect gradients, learning rate too high | Run collapse checklist in Section 6.3. |
| One-step works, multi-step fails | Autoregressive error accumulation or context-state mismatch | Reduce MPC horizon; train free-running rollouts; fix predicted-context updates. |
| Planner predicts reward that does not happen | Model exploitation or reward calibration failure | Track imagination gap; shorten horizon; raise uncertainty penalty; disable MPC if persistent. |
| Agent repeats one action or becomes passive | Poor intrinsic normalization, action-selection collapse, bad planner score scale | Log action-token entropy and score components; rebalance generic exploration; keep random floor. |
| Agent only achieves easy milestones | Insufficient exploration breadth or long-horizon planning | Check per-achievement spectrum and novelty coverage; do not optimize only raw reward. |
| Model forgets earlier episodes | Replay sampling bias, weights overwritten, checkpoint missing replay | Increase older replay fraction; verify replay restoration; compare persistence ablation. |
| Colab runtime runs out of RAM | Too many float32 frames or too many workers | Store uint8; reduce replay capacity/workers; avoid GPU-resident replay. |
| Resume changes behaviour unexpectedly | Action permutation, RNG, optimizer, replay, or EMA not restored | Fail resume test; inspect checkpoint contents; do not continue a supposedly identical run. |
| Four roles do not help | Shared data is redundant or policy objectives overlap | Inspect role diversity; change only generic weights; remove roles that add no measurable value. |

Every failure must be logged with the checkpoint, environment steps, learner update count, config, and a short diagnostic note.

---

## 17. Engineering quality requirements

- Use type hints on public functions and dataclasses for transitions/config objects.
- Add docstrings to all public modules and non-trivial algorithms.
- Use explicit device placement; do not assume CUDA exists.
- Set and record Python, NumPy, and PyTorch seeds.
- Check tensor shapes and finite values at module boundaries during development.
- Use structured logging (JSONL or CSV) for metrics.
- Separate training, planning, collection, and evaluation code.
- Do not silently catch exceptions in the training loop. Save diagnostics and fail visibly.
- Add a short unit test for every data boundary, loss, EMA update, and checkpoint operation.
- Provide a `README.md` with exact Colab setup, run order, config selection, checkpoint-resume steps, and expected output files.
- Every experiment directory must contain its exact config, package versions, seed, action-token permutation, logs, metrics, and checkpoint manifest.
- Keep raw videos and large replay checkpoints out of Git.

### 17.1 Minimum automated test suite

The implementation must provide commands equivalent to:

```bash
pytest -q tests/test_env_contract.py
pytest -q tests/test_no_info_leak.py
pytest -q tests/test_opaque_actions.py
pytest -q tests/test_replay_sequences.py
pytest -q tests/test_ema_update.py
pytest -q tests/test_jepa_shapes.py
pytest -q tests/test_no_episode_crossing.py
pytest -q tests/test_checkpoint_resume.py
pytest -q tests/test_planner_replans.py
pytest -q tests/test_evaluator_isolation.py
```

The README must state the expected result for each command. Do not mark tests as skipped to make the suite appear green.

---

## 18. Required final outputs

At the end of a completed run, produce:

1. `final_report.md` — architecture actually implemented, deviations from this plan, validation outcomes, per-achievement results, Crafter score, ablations, known limitations.
2. `metrics.json` and `metrics.csv` — machine-readable logs.
3. `achievement_rates.csv` — one row per run/seed and one column per achievement, plus the aggregate score.
4. `heldout_prediction.json` — latent prediction metrics for horizons 1/2/4/8 and baseline comparisons.
5. `action_conditioning.json` — paired action tests and ablation comparisons.
6. `collapse_diagnostics/` — feature variance, cosine similarity, effective-rank charts.
7. `planner_diagnostics/` — predicted/actual outcomes, imagination gap, latency.
8. `videos/` — selected successful, average, and failed evaluation episodes.
9. `checkpoints/` — resumable checkpoints and manifest.
10. `environment_manifest.txt` — Python, PyTorch, CUDA, GPU, Crafter, Gym/Gymnasium, and dependency versions.
11. `experiment_config.yaml` — exact final config, including seeds and opaque action mapping.

The final report must answer these questions explicitly:

- Did the latent predictor generalize to held-out episodes?
- Did action conditioning improve prediction over action-ablated controls?
- Did the representations avoid collapse?
- How quickly did free-running prediction error grow with horizon?
- Did persistent replay/model weights help across episodes and restart?
- Did MPC improve real game performance over the same non-MPC agent?
- Did four roles help compared with one role?
- Which achievements improved, which remained unsolved, and what is the per-seed variance?
- What failed, and what evidence supports each conclusion?

---

## 19. Scope boundary: Atari comes later

Do not implement Atari in the first Crafter milestone. Keep the environment interface generic enough that another visual environment can later provide the same six learner fields: RGB observation, opaque action token, next RGB observation, scalar reward, and episode-end flags.

After the Crafter system passes the validation gates:

1. Build a separate Atari adapter.
2. Define a new anonymous action-token vocabulary for each game.
3. Preserve the same JEPA training principle and evaluation discipline.
4. Reassess frame size, action repeat, temporal context, and planning horizon for Atari.
5. Do not assume the Crafter action mapping, reward scales, or planner settings transfer unchanged.

Transfer is an experiment, not a guaranteed result.

---

## 20. Research references and implementation grounding

These references ground the benchmark and the JEPA design direction. They are not permission to copy a pretrained model or bypass the from-scratch requirement.

1. **Crafter official repository and environment interface** — documents the 64×64 RGB observation, 17 discrete actions, environment setup, and reward/no-reward variants:  
   https://github.com/danijar/crafter
2. **Crafter PyPI documentation and scoreboard** — documents the 1M-step evaluation budget, 22 achievement success rates, geometric-mean score, and example benchmark results:  
   https://pypi.org/project/crafter/
3. **Hafner, “Benchmarking the Spectrum of Agent Capabilities,” ICLR 2022** — describes Crafter’s design, exploration/generalization challenges, action space, and exact score definition:  
   https://openreview.net/forum?id=1W0z96MFEoH
4. **Meta, “V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning,” 2025** — research context for self-supervised visual world modelling and action-conditioned latent prediction/planning:  
   https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/
5. **V-JEPA 2 research code repository** — reference material for the research direction; do not load its pretrained weights in this project:  
   https://github.com/facebookresearch/vjepa2

---

## 21. Final implementation-agent checklist

Before declaring the build complete, verify every statement below:

- [ ] All model weights begin from random initialization.
- [ ] No game mechanics, recipes, object labels, action names, or scripted action sequences are included in the learner.
- [ ] The learner input is limited to RGB, opaque action token, scalar reward, next RGB, and end flags.
- [ ] `info` is isolated to the evaluator and cannot reach training or planning code.
- [ ] The online encoder, EMA target encoder, action-conditioned latent predictor, and stop-gradient JEPA loss are implemented.
- [ ] Multi-step autoregressive latent rollouts are trained and evaluated.
- [ ] The replay buffer persists across episodes and Colab restarts.
- [ ] The model passes held-out prediction, action-conditioning, and collapse tests.
- [ ] The planner is gated by model validation and replans after every real action.
- [ ] The imagination gap is tracked.
- [ ] The multi-role population shares one model and is retained only if it provides measured value.
- [ ] Crafter metrics include all 22 achievement success rates and the published geometric-mean score.
- [ ] Reward-free and reward-enabled experiments are kept separate.
- [ ] Baselines and ablations use matched interaction budgets.
- [ ] Final reports include all seeds, compute usage, failures, and deviations.

**Definition of done:** the system is not “done” because it runs or because its JEPA loss decreases. It is done only when the code passes the integrity tests, the world model demonstrates action-conditioned generalization on held-out data, memory survives restarts, and the agent’s measured behaviour is compared honestly against controlled baselines.

---

# Revision 1.2 — AEON implementation audit, relationship learning, and initial observability

**Revision date:** 3 November 2026  
**Audited repository:** `sahiltolani30/learn-a-world`  
**Audited AEON commit:** `268d8f9b5df5f123ceda27626b178b4cb9758c20`  
**Purpose:** Extend the original JEPA implementation plan without discarding the existing AEON implementation. This revision is normative wherever it adds detail or conflicts with an earlier generic statement.

## 22. Current implementation audit and status rules

### 22.1 How to interpret this audit

This audit is a source-code inspection of the repository snapshot above. It is **not** a claim that the complete test suite passed, that a Colab run completed, or that AEON learned the intended relationships. A class or test file existing in GitHub means code is present; it does not prove correct runtime behaviour.

Use these status labels:

- **Implemented in source:** a corresponding implementation exists in the audited code.
- **Partial / needs correction:** a component exists, but an observed integration or validation gap prevents us from treating it as complete.
- **Missing:** the required capability is not implemented in the inspected AEON source.
- **Not verified:** no committed runtime evidence or completed experiment establishes that it works.

The plan's former status line (“implementation has not been performed”) is no longer accurate. The core source implementation is present. The implementation is not yet validated end-to-end.

### 22.2 AEON source audit table

| Capability | Current source evidence | Revision 1.2 status | What is still required |
|---|---|---|---|
| RGB environment adapter and `info` quarantine | `src/env/crafter_adapter.py`, `learner_observation_filter.py`, `evaluator_tap.py` | Implemented in source; runtime not verified | Run contract and leak tests in the actual pinned environment. |
| Opaque 17-token action mapping | `src/env/opaque_actions.py` | Implemented in source | Ensure every training, validation, and replay environment uses the exact run mapping where token identity is meant to match. |
| Visual encoder and EMA target encoder | `src/models/visual_encoder.py`, `ema_encoder.py` | Implemented in source | Validate feature quality on real held-out episodes. |
| Action embedding, GRU context, predictor ensemble | `src/models/action_embedding.py`, `temporal_context.py`, `latent_predictor.py`, `predictor_ensemble.py`, `jepa_world_model.py` | Implemented in source | Verify training and evaluation use correctly aligned action/observation pairs. |
| JEPA multi-horizon learner | `src/learning/learner.py`, `jepa_losses.py` | Implemented in source | Run the alignment tests and compare all horizons on a genuine held-out split. |
| Raw transition replay and sequence sampling | `src/data/replay_buffer.py`, `sequence_sampler.py` | Implemented in source | Add stronger wrap-around and stale-frame tests; verify with actual collected data. |
| Checkpoint serialization primitive | `src/data/checkpoint_store.py` | Partial | An atomic save/load helper exists, but orchestration does not yet provide complete, deterministic resume. |
| Novelty and latent k-NN memory | `src/exploration/latent_memory.py`, `novelty.py` | Partial | The memory structure exists, but the inspected collection/action path does not populate it or checkpoint it. |
| Predictor disagreement | `src/exploration/disagreement.py` | Implemented in source; integration needs correction | Separate raw measurement from normalization; do not update normalizers once per candidate while ranking candidates. |
| Learning progress | `src/exploration/disagreement.py` | Partial | `update_and_score()` exists, but the inspected training/collection path does not feed it observed prediction errors. |
| Curiosity action selection | `src/exploration/action_selector.py` | Partial | Candidate scores exist, but novelty memory/progress signals are not correctly maintained; sequential normalizer mutation can make scores token-order dependent. |
| CEM latent planner | `src/planning/cem_planner.py`, `latent_rollout.py`, `candidate_scorer.py` | Implemented in source; utility unverified | Validate imagined outcomes against real outcomes and compare with matched no-MPC policies. |
| Basic planner imagination-gap diagnostics | `src/planning/planner_diagnostics.py` | Partial | Log per-decision predictions, actual outcomes, latent errors and score components, not just aggregate reward gap. |
| Context-sensitive relation diagnostic | `src/evaluation/relation_diagnostics.py` | Diagnostic only | It measures prediction divergence between contexts; it does not discover, store, verify or reuse relationships. |
| Visual inventory/object state tracker | No dedicated module found in inspected AEON tree | Missing | Build a self-supervised, anonymous visual state tracker. |
| Action–precondition–effect discovery | No dedicated learning module found in inspected AEON tree | Missing | Build hypothesis generation, repeated evidence collection, contradiction handling and conditional testing. |
| Persistent relationship memory | No dedicated module found in inspected AEON tree | Missing | Build serializable relation records, evidence index, confidence and retrieval. |
| Dependency graph and multi-step subgoal composition | No dedicated module found in inspected AEON tree | Missing | Compose discovered anonymous relations into task-level plans. |
| True held-out episode evaluator | Current validation path samples `sampler.sample_batch()` from the training replay in `src/training/train_agent.py` | Partial / invalid as held-out evidence | Build a separate held-out collector/store that is never inserted into training replay. |
| Action-conditioning test mapping | `src/evaluation/action_conditioning_tests.py` creates adapters without the training mapping | Partial / invalid until corrected | Pass the exact run mapping to all paired test branches or use a rigorously verified shared token mapping. |
| End-to-end checkpoint resume | Save helper exists; `run.py` has no resume option and inspected `train_agent.py` does not restore a saved run | Missing as an end-to-end feature | Implement full restoration and a deterministic save/reload/continue test. |
| Full decision and memory audit trail | No complete trace system found | Missing | Implement Section 23 before the longer baseline run. |
| Runtime benchmark evidence | AEON notebooks in the audited commit have no executed cells/outputs; no AEON run report was found in the committed AEON artifacts | Not verified | Run and commit experiment manifests and reports. |

### 22.3 Specific defects and integration risks to resolve before trusting metrics

1. **Action mapping mismatch in Test C.** The training run creates an `OpaqueActionMapping` using the run seed. `ActionConditioningTester` constructs `CrafterAdapter(seed=seed)` for its branches without receiving that training mapping. Consequently, token `k` in the model can be evaluated as a different environment action in a branch. The paired error comparison is invalid until the test uses the correct mapping. Keep the model blind to the mapping; the evaluator/adapter owns it.

2. **Test B is not currently a held-out test.** The validation block in `train_agent.py` obtains its validation batch from the same `SequenceSampler` and `ReplayBuffer` used for training. Config fields such as held-out seed start/count do not by themselves create a held-out split. The reported `one_step_beats_persistence` result must not be called generalization until an independent episode-seed collector and store are used, and those episodes are excluded from replay and optimizer updates.

3. **Novelty memory is not demonstrably populated.** `ActionSelector` calls novelty evaluation with `update_memory=False`. The inspected collector/training path contains no corresponding observed-latent insertion call. An empty latent memory cannot provide meaningful long-term novelty coverage. Add a single well-defined update after each real observation (not once for every imagined candidate), and checkpoint the memory or rebuild it deterministically from stored observations.

4. **Learning progress is not demonstrably updated.** The estimator has an `update_and_score()` method, but no observed prediction-error update is wired into the training/collection path. Calling `score()` repeatedly is not a substitute for updating it with measured errors.

5. **Candidate ranking can mutate its own normalization statistics.** Novelty and disagreement normalizers are updated while the selector iterates over candidate tokens; the planner scorer also evaluates candidate novelty sequentially. That can make candidate scores depend on evaluation order. Compute raw candidate values first, normalize all candidates using one frozen statistics snapshot, then update running statistics once from real observed data. Never update normalizers using imagined candidate rollouts.

6. **Checkpoint helper is not a resume system.** The current training checkpoint includes model, optimizer, scaler, mapping, replay, RNG and validation flags. The inspected orchestration does not restore them before continuing. The checkpoint also needs scheduler/update-scheduler state, exploration-memory and normalizer state, relationship memory, role schedule, collector/episode boundary state, config and code-version metadata. Do not label a run restart-safe until the end-to-end resume test passes.

7. **Warm-up config needs to be honoured.** The training mode currently switches using replay size versus `min_sequences_before_training`; verify that `collection.initial_random_steps` actually controls the intended random warm-up. A config field that is not read by orchestration must not be presented as an active setting.

8. **Worker count differs from the original safe rollout order.** The committed `crafter_t4.yaml` config uses eight workers, while the original plan says to begin with one and scale after throughput testing. For the first diagnostic baseline, use one worker. Add workers only after data ordering, episode tracking and throughput are verified.

9. **The current relation diagnostic is not a relation learner.** `conditional_outcome_divergence` only establishes that predictions differ between two contexts. A larger divergence can still be wrong. It must be supplemented by actual held-out outcome accuracy, counterfactual/paired tests and relation-memory retrieval tests.

10. **Role profiles do not prove four independent exploration streams.** The current training orchestration passes a role value into a batched collector call and advances a role index as episodes complete. Treat the multi-role implementation as provisional until traces prove which role generated each transition and demonstrate distinct data coverage. Keep one Explorer role for the first controlled baseline.

These are source-level findings from the inspected commit, not reports of observed runtime failures. Each must be converted into a failing/passing automated test or runtime metric.

## 23. Required observability: every decision and every memory mutation must be explainable

The system must be debuggable after a run without guessing why it behaved as it did. Add an append-only, machine-readable event log and reproducible episode traces. Logging is a read-only side channel: it must not change model inputs, action selection, random-number consumption, gradients or memory updates.

### 23.1 Global trace identity and run manifest

Every run must create a unique `run_id`. Every episode, real environment step, learner update, action decision, prediction, memory event and planner invocation receives a stable ID. Link related events with IDs rather than relying on timestamps alone.

At run start, save:

- Git commit SHA and dirty/uncommitted status, config snapshot and config hash.
- Python, PyTorch, Crafter, Gym/Gymnasium and dependency versions.
- Hardware/device, CUDA availability, GPU memory and precision mode.
- Training seed, environment seed list, held-out seed list and frozen evaluation seed list.
- Opaque action permutation in a restricted evaluator/debug manifest. Do not expose its semantic interpretation to the learner.
- Model architecture and parameter count; checkpoint identifier; replay schema version.
- Debug/logging configuration and whether any trace sampling was enabled.

Every metric must include `run_id`, `env_step`, `learner_update`, `episode_id`, and checkpoint/version context wherever applicable.

### 23.2 Per-action decision trace

For **every real environment action**, write one decision record before the adapter executes it and one outcome record after the next observation arrives.

The decision record must include:

- `decision_id`, current checkpoint/model version, environment step, episode and timestep.
- Current policy mode: `warmup`, `explore`, or `mpc`.
- Role label, random-action-floor value and whether the random branch fired.
- Decision source: `warmup_random`, `random_floor`, `curiosity_sample`, `mpc_cem`, or an explicitly named fallback/override.
- All 17 opaque tokens' raw and normalized score components where applicable: novelty, disagreement, learning progress, predicted reward, termination risk, rollout uncertainty and final score.
- The softmax temperature and all probabilities when sampling from a categorical distribution.
- Selected opaque token, its rank/probability, the reason/source that selected it, and random generator state/seed reference needed for deterministic replay.
- For MPC: candidate sequences, initial categorical distribution, elite sequences, score breakdowns, chosen sequence, first action, horizon, iterations, planner latency and predictor ensemble version. Support `summary` and `full` trace levels so full candidate logs can be enabled for selected episodes without overwhelming normal runs.

The outcome record must include:

- Reference to the observation before and after the action (stored image or replay index plus integrity hash).
- Actual scalar reward, termination/truncation, and evaluator-only episode metadata kept in a separate namespace/file.
- Actual next-frame latent from the current target encoder; per-member prediction error and ensemble disagreement for the chosen action.
- Predicted versus observed reward/termination and the prediction residual.
- Whether the action achieved its predicted visual effect, according to the anonymous state tracker, with confidence.

Never record only “AEON selected token 7”. The trace must show the candidate scores/probabilities and the branch that caused the selection.

### 23.3 Visual-state tracker trace

Implement a visual state tracker that produces an anonymous, confidence-bearing state abstraction from RGB only. It must not use OCR, hand-authored Crafter labels, image-template matching, internal inventory dictionaries or game source state.

Initial outputs should be anonymous structures such as:

- visual slot/prototype ID;
- spatial region or relative location in the image;
- appearance embedding and track ID;
- estimated visual quantity/state representation and uncertainty;
- presence/absence/change probability between frames;
- relative proximity between tracked visual entities when supported by pixels.

For the inventory HUD, track stable visual slots and learn the visual representation of quantity changes. Do not assume the current JEPA global latent already represents exact counts. The tracker may begin with a learned patch/slot encoder and temporal association; avoid manually assigning meanings such as “wood” or “pickaxe” to learner-side features.

For each tracker update, log the before/after visual slots, match confidence, quantity/state belief, uncertainty, changed slots and the evidence pixels/patch references. Diagnostic labels may be added manually only in an evaluator-only annotation file; they must never be fed to training, policy, relation memory or planning.

### 23.4 Relationship discovery trace and evidence ledger

A relationship is not created as confirmed knowledge from one surprising transition. Use three lifecycle states:

1. **Candidate:** an observed action/context/outcome pattern worth testing.
2. **Tentative:** repeated evidence supports the pattern, but conditional requirements or confounds are not resolved.
3. **Confirmed for transfer:** repeated, independent evidence supports the same conditional prediction, and held-out tests succeed. “Confirmed” is an empirical confidence label, not a claim of universal truth.

Each relation record must be anonymous and serializable. Suggested fields:

```python
RelationRecord(
    relation_id: str,
    action_token: int,                    # opaque token only
    precondition_features: ...,          # anonymous visual/state features
    effect_features: ...,                # anonymous before/after deltas
    context_embedding: ...,               # optional retrieval key
    expected_change_distribution: ...,
    support_count: int,
    contradiction_count: int,
    independent_episode_count: int,
    confidence: float,
    status: str,                          # candidate/tentative/confirmed/rejected
    evidence_event_ids: list[str],
    created_at_env_step: int,
    last_updated_env_step: int,
    schema_version: int,
)
```

For every relation event, log:

- Why a candidate was proposed (which observed state changes and action event triggered it).
- Before/after anonymous feature signatures and quantity/state deltas.
- The hypothesized preconditions and effects, each with confidence.
- Which observations support it and which contradict it.
- Whether evidence came from the same episode or independent episodes/world seeds.
- Why confidence/status changed; include thresholds and update formula version.
- Whether a controlled test was run, what variable was changed, and whether the predicted outcome occurred.
- Why a relation was rejected, merged, weakened, or marked context-specific.

Use a calibrated probability or explicitly labelled heuristic confidence. Do not call a raw count or neural similarity “confidence” unless it is calibrated against held-out outcomes.

### 23.5 Memory retrieval and use trace

When the agent retrieves memory, log:

- Query event ID and current anonymous visual/context representation.
- Retrieved relation IDs, retrieval scores, status and confidence.
- Relations rejected by the retrieval threshold and why.
- How each retrieved relation changes the selected subgoal or candidate action distribution.
- Whether the planned relation was applicable under current preconditions.
- Whether execution matched the predicted effect; add support or contradiction evidence only from the observed outcome.

This makes it possible to diagnose four separate failures: the relation was never learned; it was learned incorrectly; it was not retrieved; or it was retrieved but the planner failed to use it correctly.

### 23.6 Planner and subgoal trace

The planner must distinguish:

- low-level CEM action planning over the JEPA model;
- relationship-memory-based subgoal/dependency planning;
- the execution controller that chooses one real action and replans.

For every high-level subgoal, log the relation chain, dependencies, expected intermediate visual conditions, estimated cost/risk, confidence and reason for selection. For every low-level CEM call, log Section 23.2 candidate information. Record when the planner declines to use a low-confidence relation and falls back to exploration.

Do not represent an eight-step CEM horizon as proof of arbitrary long-horizon planning. The relation graph is intended to add reusable multi-step structure beyond the short imagined horizon.

### 23.7 Learning and data-pipeline trace

At each learner update or configured interval, log:

- Replay sample IDs and train/validation/test split IDs.
- Sequence origin and target frame indices for horizons 1, 2, 4 and 8.
- JEPA loss by horizon and predictor member; auxiliary reward/termination losses.
- Prediction error against persistence, no-action and shuffled-action baselines.
- Action-token distribution/entropy and the fraction selected by each decision source.
- Latent mean/std, per-channel variance, collapsed-channel fraction, effective rank and predictor-target similarity.
- Gradient norm, skipped/overflowed AMP update, learning rates, EMA momentum and optimizer update count.
- Data collection throughput, replay size/age, sampling time, update time, GPU memory and planner latency.
- Novelty/disagreement/progress raw values, normalized values, normalizer snapshot/version and the single real-data update that changed those statistics.

The same transition must never silently appear in both training and held-out sets. Persist split membership and assert disjoint episode IDs/seeds.

### 23.8 Human-facing debug tools

Provide a post-run inspector (CLI first; dashboard optional) with these views:

1. **Episode timeline:** frame, action token, selected decision source, score summary, reward, state change and outcome.
2. **Why this action?** All 17 token scores/probabilities and the selected branch/reason.
3. **World-model accuracy:** predicted-versus-observed latent changes, by horizon/member/action token/context bucket.
4. **Visual state diff:** tracked anonymous inventory slots/world patches before and after the action.
5. **Memory explorer:** relation lifecycle, support/contradiction examples, confidence history, retrieval events and checkpoint snapshots.
6. **Subgoal trace:** dependency chain, retrieved relation evidence and where execution diverged.
7. **Root-cause report:** first point in an episode where the agent's prediction, state estimate, memory or action choice diverged from observed evidence.

Optional human-only annotations can map anonymous prototype IDs or opaque tokens to familiar names in a separate evaluator/debug layer. Such annotations are strictly read-only and must never be loaded by the learner, memory, planner or training process.

### 23.9 Debug modes and storage policy

Add configuration such as:

```yaml
debug:
  enabled: true
  level: summary       # off | summary | full
  log_every_n_steps: 1
  trace_selected_episodes: []
  save_observation_keyframes: true
  save_candidate_sequences: true
  save_memory_snapshots: true
  deterministic_replay: true
  max_full_trace_episodes: 20
```

- `off`: only essential training metrics and checkpoint manifests.
- `summary`: every decision and memory lifecycle event, with compact candidate summaries.
- `full`: all token scores, all CEM candidates/elite updates, visual-state diffs and complete relation evidence for selected episodes.

Write append-only JSONL or Parquet records; store large RGB frames/keyframes separately by content hash. Keep a schema version for every log. Ensure logging can be disabled or sampled without changing the agent's actions. Keep raw debug artefacts outside Git; preserve compact manifests and summary reports.

### 23.10 Deterministic episode replay

Implement a replay tool that loads:

- the exact checkpoint before the episode;
- environment seed and episode initialization record;
- the exact opaque action mapping;
- recorded opaque action tokens;
- relevant RNG states and configuration;
- model and memory versions.

It must replay the recorded action sequence and compare observation hashes. If exact deterministic reproduction is not available on the installed Crafter/library version, state the deviation and use recorded observations for policy-decision reanalysis. Never silently claim exact replay when hashes differ.

## 24. New learning system: visual state, relations, long-term memory, and composition

### 24.1 Visual state tracking — no semantic shortcut

The existing encoder/JEPA remains the shared visual representation foundation. Add a separate learned visual state-tracking module that identifies persistent anonymous slots/prototypes and measures their changes across frames. It may use patch tokens, attention slots, temporal association and learned quantity/state embeddings.

The learner receives only RGB and its own action token, reward and end flags as defined in Section 2. It must not receive OCR text, decoded inventory counts, engine objects, map coordinates, Crafter labels, or recipe data. The evaluator may use privileged state only for post-hoc scoring and must write it to a quarantined namespace.

Acceptance checks:

- Same visual slot/prototype remains trackable across small camera/player movements where the pixels support matching.
- Before/after change estimates are reproducible and have uncertainty values.
- Quantity-change estimates are tested against evaluator-only ground truth on a held-out diagnostic set, with no labels used in training.
- Report exact count accuracy, mean absolute count error, slot association accuracy, false-change rate and confidence calibration.
- If quantity tracking is unreliable, relation learning must mark quantity preconditions as uncertain rather than inventing exact counts.

### 24.2 Relationship hypothesis representation

The learning target is an empirical conditional relationship:

`observed context + opaque action token -> distribution over observed visual state changes`.

The context may include anonymous visual slot states, approximate quantities, nearby visual tracks, latent context and uncertainty. The effect may include anonymous slots appearing/disappearing, quantity-state changes, persistent world changes or other reproducible visual deltas.

A candidate may be proposed when the observed state change is larger than the tracker's expected no-action/background variation and is temporally associated with the chosen action. This is only a candidate, not causal proof.

### 24.3 Evidence and causal confidence

A relationship is conditional, not merely “action X once preceded change Y”. For confirmation:

- Gather repeated observations across distinct episodes/world seeds.
- Compare outcomes under similar contexts with and without the suspected precondition where possible.
- Change one observable condition at a time when testing a prerequisite.
- Retain failed attempts as negative evidence; do not erase them after success.
- Separate “effect observed” from “precondition established”.
- Track alternate explanations and uncertainty, including unrelated environmental changes.

Only the evaluator may use hidden state to assemble paired diagnostic cases or score the learner's relation predictions. The learner itself must discover the relationship from permitted observations.

### 24.4 Persistent relationship memory

Add a dedicated store, separate from replay and latent novelty memory. Minimum capabilities:

- Save/load relation records, evidence references, support/contradiction counts, confidence calibration, retrieval index and schema version.
- Preserve memory across episode resets and process/Colab restarts.
- Consolidate duplicate relations only when their observable effects and conditions match.
- Keep context-specific variants distinct when evidence indicates different outcomes.
- Track superseded or rejected relations rather than silently deleting history.
- Support retrieval by current visual/context features and current precondition confidence.
- Never retrieve a relation with a precondition that the current visual tracker cannot verify; return “unknown” rather than assuming it is satisfied.

The relationship store must be included in checkpoints and have a save/reload equivalence test.

### 24.5 Dependency graph and compositional subgoals

Create dependencies from observed anonymous effects and preconditions. If relation A reliably produces a condition required by relation B, the system may hypothesize an edge A → B. An edge needs its own evidence and confidence; do not create a dependency merely because two discoveries happened sequentially.

A planner can then:

1. Select an unachieved observable target state or evaluator-defined diagnostic task for evaluation only.
2. Search the learned relation graph for a path that could produce it.
3. Estimate required resources/preconditions using the visual state tracker.
4. Choose the next attainable subgoal.
5. Use JEPA/CEM for short-horizon action control toward that subgoal.
6. Compare predicted and actual effects after execution.
7. Update relationship evidence and replan when reality differs.

During training, task targets must be generic visual-state objectives or permitted environment rewards. Do not inject named Crafter achievements or recipe graphs into the learner. Human-readable tasks may be used in evaluator-only analysis to score capability.

### 24.6 Knowledge transfer across episodes and worlds

Separate three kinds of retained state:

- **Within-episode state:** current visual tracker tracks and recurrent context; reset at episode end.
- **Across-episode experience:** JEPA weights, replay, calibrated exploration statistics and relation memory; retained according to experiment design.
- **Across-world knowledge:** stable, evidence-backed relations retrievable in a new seed/world, with context/precondition checks.

A new world must not be assumed identical. Track whether the visual context matches a known relation. If it does not, explore and create a new candidate instead of forcing an old relation.

## 25. New evaluation suite for relational learning

Do not use only the aggregate Crafter score. Add these independent gates:

| Gate | Measurement | Required evidence |
|---|---|---|
| `VISUAL_STATE_TRACKING_PASS` | Anonymous slot tracking, state/quantity change accuracy, false-change rate and calibration | Held-out diagnostic frames; evaluator labels remain quarantined. |
| `RELATION_DISCOVERY_PASS` | Correct action-effect predictions, support/contradiction handling and conditionality | Repeated evidence plus matched context tests. |
| `RELATION_MEMORY_PASS` | Save/load equality, retrieval precision and no stale relation use | Restart test and retrieval audit. |
| `CROSS_EPISODE_TRANSFER_PASS` | Success using a relation in an unseen episode without rediscovery | Frozen-memory held-out seed evaluation. |
| `COMPOSITION_PASS` | Successful multi-relation dependency chains | New objective requiring at least two previously discovered relations. |
| `DEBUG_TRACE_PASS` | Every real action and memory mutation has a complete trace and replay reference | Automated trace-integrity audit. |

### 25.1 Relation metrics

Report at least:

- Candidate generation rate and candidate-to-tentative/confirmed/rejected conversion.
- Relation precision, recall and F1 against evaluator-only ground truth on a labelled diagnostic set; labels never enter learning.
- False discovery rate and false rejection rate.
- Support and contradiction counts, by independent episode count.
- Predicted effect distribution calibration (Brier/log score where applicable).
- Conditionality accuracy: same action under matched contexts with different observed preconditions.
- Memory retrieval precision/recall, stale-retrieval rate and percentage of retrieved relations actually applicable.
- Transfer success, rediscovery frequency, extra experiments avoided and number of steps saved versus a no-relation-memory control.
- Multi-relation chain success rate, chain length and error location in failed chains.

### 25.2 Controlled transfer protocol

Use a staged discovery/transfer experiment, but do not assume a fixed number of episodes guarantees success.

1. Discovery partition: allow AEON to explore and update its relationship memory.
2. Freeze partition: save checkpoint, model, visual tracker and relation memory; record hashes.
3. Transfer partition: use disjoint world seeds. Freeze learning for the primary transfer measurement; do not add new relation evidence to the frozen memory. New observations may be used for normal inference and action selection.
4. Optional adaptation partition: separately measure online adaptation, clearly distinct from frozen transfer.
5. Controls: same JEPA without relation memory; reset relation memory; no compositional planner; random policy; and previous AEON baseline where feasible.
6. Use matched environment-step or episode budgets and at least three independent training seeds for development decisions. Report confidence intervals and every run, not only the best.

Report both the frozen-transfer result and the optional online-adaptation result. A system that relearns a relationship in every new world has not demonstrated transfer.

## 26. Revised implementation and experiment order

### 26.1 Decision: do not run the long current configuration yet

Do **not** start the 1M-step benchmark or an uninstrumented 50k-step run using the current orchestration. The current validation and exploration integration gaps can make the metrics misleading. Also do not build every new module before obtaining a baseline; that would make it difficult to identify which change helped or broke the system.

The correct order is:

**instrument and fix the baseline → run a short baseline → freeze baseline evidence → add relationship learning → run an integrated pilot → scale only after gates pass.**

### 26.2 Stage A — make the existing baseline trustworthy

Before a meaningful training run:

1. Fix Test C to use the exact run-specific opaque mapping in every paired branch.
2. Build a true held-out episode collector/store with disjoint seeds. Never sample validation batches from training replay.
3. Verify and wire the actual `initial_random_steps` setting, or remove it from config if not used.
4. Fix novelty memory insertion so it is updated from real observations exactly once per real step; checkpoint its state or deterministically rebuild it.
5. Wire learning-progress updates to observed prediction errors; do not call a score method as if it updates evidence.
6. Make normalization candidate-order invariant: freeze running statistics while scoring all actions/candidates, then update statistics only from real observations/outcomes.
7. Add full action-decision and training traces from Section 23.
8. Implement end-to-end resume or explicitly disable resume claims until it exists. Save all training/collector/exploration state needed for a deterministic continuation.
9. Set `num_env_workers: 1` for the first diagnostic run; increase only after performance and data-integrity tests.
10. Start with one Explorer role. Defer the multi-role population until later.

Keep these as a separately versioned baseline-correction commit. Do not add relation learning in the same commit; preserve an identifiable “corrected JEPA baseline” checkpoint.

### 26.3 Stage B — baseline tests and short runs

Run in this order:

1. Full pytest suite, with each test result saved to `test_results.txt` and a JSON summary.
2. 500-step environment/data smoke test: verify observations, actions, replay sequence alignment, trace creation, checkpoint save/load and clean shutdown.
3. Short instrumented training pilot (suggested 5k–10k environment steps, one worker): verify JEPA updates actually occur; verify the true held-out split remains untouched; inspect action distribution, latent diagnostics and trace completeness.
4. If Stage B is clean, run the original JEPA development pilot to its planned checkpoint (50k total environment steps) with all baseline gates and independent held-out episodes.

This gives a fair baseline for world-model prediction, exploration and behaviour before relational learning is introduced. Do not unlock MPC solely from a train-replay validation score. Do not claim the 50k target passed unless the documented thresholds are met on true held-out episodes.

### 26.4 Stage C — add new capabilities one at a time

Implement in this order, each behind a config flag and with unit tests:

1. Anonymous visual slot/state tracker and before/after state-difference records.
2. Relationship candidate generation from observed action-conditioned state changes.
3. Evidence ledger with support, contradiction, independent-episode count and hypothesis lifecycle.
4. Controlled conditional/precondition testing policy.
5. Persistent relationship store with save/load and retrieval diagnostics.
6. Relation-aware subgoal and dependency planner.
7. Relationship-aware exploration that balances known-skill exploitation and targeted unknown-condition experiments.
8. End-to-end trace views and frozen-memory transfer evaluation.

At each step compare against the corrected JEPA baseline under matched budgets. Keep the feature disabled if it fails its own gate.

### 26.5 Stage D — integrated and scaled experiments

After the relation system passes unit/integration tests:

- Run an instrumented end-to-end micro-world experiment to inspect one discovered relation from first evidence through retrieval in a later episode.
- Run 10k–25k integrated development experiments to expose integration failures before scaling.
- Run the 50k JEPA + relation-memory development study with at least three seeds.
- Run the frozen transfer experiment and controls in Section 25.2.
- Only then test CEM/subgoal integration, multi-role collection, additional workers and longer budgets.
- Run the 1M-step benchmark only after the core model, held-out tests, relationship learning, resume and trace gates are stable.

## 27. Revised initial configuration and feature flags

For the first corrected baseline, use one worker and disable unvalidated modules until their own tests pass. Suggested initial config additions:

```yaml
collection:
  num_env_workers: 1
  initial_random_steps: 10000
  heldout_seed_start: 900000
  heldout_episode_count: 50

validation:
  use_independent_heldout_episodes: true
  require_disjoint_episode_ids: true
  require_disjoint_seeds: true

exploration:
  novelty_memory_enabled: true
  learning_progress_enabled: true
  update_normalizers_from_real_data_only: true

relationship_learning:
  enabled: false               # enable only after baseline is reproduced
  visual_state_tracker: false
  candidate_generation: false
  evidence_ledger: false
  persistent_memory: false
  conditional_testing: false
  compositional_planner: false
  min_independent_episodes_for_confirmation: 3

planner:
  enabled_after_validation: false
  require_true_heldout_pass: true
  require_action_mapping_check: true

debug:
  enabled: true
  level: summary
  log_every_n_steps: 1
  save_candidate_sequences: true
  save_memory_snapshots: true
  deterministic_replay: true
```

The number of independent episodes above is a starting engineering threshold to test, not a universal guarantee of causal proof. Tune it only through a documented calibration experiment. All switches must be parsed, surfaced in the run manifest and covered by tests. Do not leave configuration fields unused.

## 28. Required new modules and tests

Suggested module boundaries (names may differ if documented):

```text
src/
  perception/
    visual_state_tracker.py
    slot_association.py
    state_change.py
  relationships/
    relation_types.py
    candidate_generator.py
    evidence_ledger.py
    conditional_tester.py
    relation_memory.py
    retrieval.py
    dependency_graph.py
  planning/
    subgoal_planner.py
  debugging/
    event_schema.py
    trace_logger.py
    decision_recorder.py
    memory_recorder.py
    episode_replay.py
    debug_report.py

tests/
  test_action_mapping_in_validation.py
  test_true_heldout_split.py
  test_novelty_memory_updates.py
  test_learning_progress_updates.py
  test_candidate_order_invariant_scores.py
  test_checkpoint_full_resume.py
  test_visual_state_tracker.py
  test_relation_candidate_evidence.py
  test_relation_contradictions.py
  test_relation_memory_roundtrip.py
  test_relation_retrieval.py
  test_dependency_composition.py
  test_frozen_cross_episode_transfer.py
  test_decision_trace_completeness.py
  test_memory_event_trace_completeness.py
  test_debug_logging_does_not_change_policy.py
  test_deterministic_episode_replay.py
```

### 28.1 Mandatory integration tests

- Given identical observation/context and action token, every test branch maps that token to the same environment action as the training run.
- No held-out episode ID or seed is present in training replay or optimizer batches.
- Every real observation updates novelty memory once; imagined candidates do not update it.
- Candidate score results remain unchanged when candidate iteration order is permuted, within numerical tolerance.
- Every real action has exactly one decision event and one outcome event, including terminal transitions.
- Every relation status or confidence mutation references one or more evidence event IDs.
- A relation retrieved for planning is linked to the plan and later outcome event.
- Checkpoint reload restores model, EMA, optimizer/scheduler, replay, mapping, RNG, normalizers, novelty memory, relation memory, role schedule and collector boundary state.
- Enabling debug logging does not change selected actions or model outputs under the same deterministic test conditions.
- Frozen-transfer mode cannot write new relation evidence or update model weights.

## 29. Final completion criteria — revised

The project must report independent pass/fail statuses. Do not merge them into one “AEON understands Crafter” claim.

- `ENVIRONMENT_INTEGRITY_PASS`
- `JEPA_PREDICTION_PASS`
- `ACTION_CONDITIONING_PASS`
- `REPRESENTATION_HEALTH_PASS`
- `TRUE_HELDOUT_GENERALIZATION_PASS`
- `EXPLORATION_SIGNAL_PASS`
- `CHECKPOINT_RESUME_PASS`
- `VISUAL_STATE_TRACKING_PASS`
- `RELATION_DISCOVERY_PASS`
- `RELATION_MEMORY_PASS`
- `CROSS_EPISODE_TRANSFER_PASS`
- `COMPOSITION_PASS`
- `PLANNING_REAL_WORLD_UTILITY_PASS`
- `DEBUG_TRACE_PASS`
- `CRAFTER_BENCHMARK_PASS`

A failed gate blocks only the dependent phase, not the recording of the result. Every failure report must include the first failing test/event, the relevant checkpoint, all configuration and seed details, and the trace IDs needed for reproduction.

**Revised definition of done:** AEON is not complete because the source files exist, tests are written, training loss falls, or one episode succeeds. Completion requires reliable held-out action-conditioned prediction, valid visual state estimates, independently supported relationships, persistent retrieval, frozen transfer to new worlds, successful composition of multiple relations, real-environment utility, complete decision/memory traces, reproducible checkpoints, and fair comparison against matched ablations and previous baselines.

---

## Revision 1.2 source files reviewed

The implementation status above was checked against these committed source paths:

- `AEON/jepa-crafter/src/models/jepa_world_model.py`
- `AEON/jepa-crafter/src/learning/learner.py`
- `AEON/jepa-crafter/src/training/train_agent.py`
- `AEON/jepa-crafter/src/training/collect.py`
- `AEON/jepa-crafter/src/env/crafter_adapter.py`
- `AEON/jepa-crafter/src/env/opaque_actions.py`
- `AEON/jepa-crafter/src/exploration/action_selector.py`
- `AEON/jepa-crafter/src/exploration/novelty.py`
- `AEON/jepa-crafter/src/exploration/disagreement.py`
- `AEON/jepa-crafter/src/exploration/latent_memory.py`
- `AEON/jepa-crafter/src/evaluation/action_conditioning_tests.py`
- `AEON/jepa-crafter/src/evaluation/heldout_prediction.py`
- `AEON/jepa-crafter/src/evaluation/relation_diagnostics.py`
- `AEON/jepa-crafter/src/planning/cem_planner.py`
- `AEON/jepa-crafter/src/data/checkpoint_store.py`
- `AEON/jepa-crafter/configs/crafter_t4.yaml`
- `AEON/jepa-crafter/notebooks/aeon_colab_runner.ipynb`

The Crafter mechanics reference remains human-side ground truth for scoring and diagnosis only: `docs/CRAFTER_MECHANICS_REFERENCE.md`. It must not be imported by the learner.


---

# Revision 1.3 — Comprehensive observability, forensic debugging, and root-cause specification

**Revision date:** 3 November 2026  
**Supersedes:** Revision 1.2 wherever this section adds a more precise requirement or resolves ambiguity.  
**Purpose:** Make the system diagnosable from recorded evidence after a run. The implementation agent must not infer undocumented behaviour, silently choose missing thresholds, or claim a root cause without evidence.  
**Implementation rule:** A feature is not considered observable merely because a metric is printed. The evidence must identify the run, episode, transition, model version, decision, data sample, and memory mutation that produced it.

## 30. Non-negotiable debugging objective

AEON must be built so that, after any run, an investigator can answer the following without relying on the original Python process or on the implementer's memory:

1. What exact observation and history did AEON receive?
2. What did the perception and state-tracking components represent from that observation?
3. What action candidates were considered?
4. What did each candidate's score consist of, before and after normalization?
5. Which policy branch selected the final action, and what random draw or deterministic tie-break was involved?
6. What did JEPA predict for that action, and what was actually observed?
7. Which model/checkpoint version generated the prediction?
8. What replay data was sampled for each optimizer update, and how was it aligned?
9. Which parameters changed, by how much, and did the EMA target update correctly?
10. Why was a relationship hypothesis created, strengthened, weakened, rejected, merged, or retrieved?
11. What exact evidence supports and contradicts that relationship?
12. Which relation-memory entries were retrieved for a plan, and which were ignored?
13. What goal/subgoal and dependency chain caused a planner to choose a sequence?
14. Did the real environment confirm the predicted effect?
15. Where did the first measurable divergence occur when an episode failed?
16. Can another person replay the stored transition sequence and reproduce the recorded decision to the documented determinism level?
17. If exact reproduction is impossible, is the reason documented and is the original evidence still sufficient to analyze the failure?

**A root-cause conclusion must be traceable to event IDs, data samples, checkpoints, configuration values, and reproduction steps.** “The model probably did not understand the inventory” is not an acceptable diagnosis. “At transition T, the state tracker assigned the previously tracked quantity slot to a different anonymous visual slot; the slot-assignment confidence was 0.31, and the next relation candidate used that mismatched track” is an acceptable evidence-backed diagnosis if the trace proves it.

## 31. Observability principles

The debugging implementation must follow these principles.

### 31.1 Observe without changing behaviour

- Debug instrumentation must not alter action scores, sample order, random-number generator state, model gradients, optimizer updates, replay contents, relation confidence, planner scores, or action selection.
- Do not use debug-only labels or evaluator information to influence training, planning, relation discovery, or memory retrieval.
- A debug run and a non-debug run with the same deterministic inputs must produce the same model outputs and selected actions, within the declared numerical determinism tolerance.
- Any diagnostic that requires extra model inference must use a cloned model or a read-only inference path and a separately seeded diagnostic RNG. It must not consume the policy RNG.
- Any extra environment branch used to assess counterfactual actions must be evaluator-only. It must not be confused with the trajectory the agent actually experienced.

### 31.2 Preserve raw evidence and derived interpretation separately

Keep these distinct:

- **Raw evidence:** RGB observations, action tokens, scalar rewards, episode flags, exact sampled indices, model outputs, optimizer/gradient measurements, and trace events.
- **Derived diagnostics:** novelty scores, relation confidence, candidate scores, failure classifications, visual-slot associations, calibration metrics and summary reports.
- **Human interpretation:** optional mapping from anonymous visual/action concepts to game terms for a researcher reading a report. This belongs only in evaluator-side annotations and must never be imported into learner-facing code.

Never overwrite raw evidence with normalized values, labels, compressed summaries, or human corrections. Derived files must point back to the raw event IDs and raw tensor/observation artifacts.

### 31.3 Distinguish “not observed” from “observed to be false”

Every field that can be unavailable must have an explicit status. Use values such as `observed`, `not_applicable`, `not_recorded`, `unsupported`, `dropped_due_to_budget`, and `unknown`. Do not serialize missing values as zero unless zero is the actual observed value. Do not turn failed perception into “object absent”; use `unknown` or low-confidence detection.

### 31.4 Do not assert causality from one transition

A before/after change following an action creates a candidate effect, not a verified causal relationship. Relationship confidence must account for repeated support, independent episodes, controls, state/context differences, contradictions, and uncertainty. Use the language `candidate`, `supported`, `provisionally_confirmed`, `rejected`, and `unknown`; do not label a relation `causal` solely because two events were adjacent in time.

### 31.5 Every metric must have a definition

For every metric, document:

- Formula or exact algorithm.
- Input event fields and source code function.
- Units and value range.
- Aggregation window.
- Whether it is raw, normalized, clipped, or calibrated.
- Update timing (per environment step, batch, optimizer update, episode, or evaluation run).
- Missing-data handling.
- Expected direction, if any.
- Known limitations.
- Threshold provenance: fixed engineering gate, statistically calibrated threshold, or descriptive-only metric.

No undocumented scalar called `score`, `confidence`, `progress`, `novelty`, or `success` may be used as a decision signal.

## 32. Trace identity, versioning, and event envelope

### 32.1 Required identifiers

Every artifact/event must be joinable by these identifiers where applicable:

```text
experiment_id       unique experiment campaign identifier
run_id              unique training/evaluation run identifier
seed_id             independent statistical run seed label
worker_id           environment worker identifier
episode_id          monotonically unique within run; never reused
world_seed           actual environment reset seed, evaluator-side metadata
step_id             monotonically increasing real environment transition index
transition_id       stable identifier for one (obs, action, next_obs) transition
decision_id          one policy decision; normally one per real transition
prediction_id        one prediction query/rollout
planner_call_id      one CEM or random-shooting invocation
candidate_id         one candidate action sequence within a planner call
optimizer_update_id  monotonically increasing successful learner update identifier
batch_id             one sampled training batch
sample_id            one sequence sample within a batch
relation_id          immutable relation identity; revisions have separate revision IDs
relation_revision_id one version of a relation record
memory_event_id      one create/update/reject/merge/retrieve/evict event
checkpoint_id        immutable checkpoint identity
artifact_id          immutable artifact identity
trace_event_id       globally unique event identifier
parent_event_id      causal/derivation parent event where applicable
```

Identifiers must not be derived only from wall-clock timestamps. They must remain unique if multiple workers emit events in the same millisecond. A UUID or run-scoped monotonic sequence is acceptable. Use `episode_id`, not only `world_seed`, to identify episodes because a seed may be intentionally reused in a test.

### 32.2 Global event envelope

All JSONL events must conform to a common envelope. Implement a versioned schema and validate it before writing.

```json
{
  "schema_version": "1.0",
  "event_type": "action_decision",
  "trace_event_id": "...",
  "experiment_id": "...",
  "run_id": "...",
  "seed_id": "...",
  "worker_id": "env-0",
  "episode_id": 12,
  "step_id": 840,
  "transition_id": "...",
  "decision_id": "...",
  "parent_event_id": null,
  "monotonic_time_ns": 0,
  "utc_time": "ISO-8601 timestamp",
  "model_version": "checkpoint-id-or-update-id",
  "config_hash": "sha256:...",
  "code_revision": "git-sha",
  "payload": {}
}
```

The literal example values above are placeholders. Use actual values at runtime. `monotonic_time_ns` is for within-process ordering and latency; `utc_time` is for human-readable chronology. Do not compare monotonic clocks across machines.

### 32.3 Run manifest

Write `run_manifest.json` before collecting the first transition. It must include:

- Repository URL, branch, commit SHA, dirty-tree status if available, and code version.
- Config snapshot and its SHA-256 hash.
- Experiment ID, run ID, seed ID, training seed, environment seed range, validation seed range, final evaluation seed range, and whether the sets are disjoint.
- Exact opaque action-token permutation, stored in a restricted adapter/evaluator manifest. The learner must not receive it. Include a hash in learner-visible logs rather than the permutation itself.
- Environment ID, Crafter package version and source path, Gym/Gymnasium versions, Python, PyTorch, NumPy, CUDA, driver, GPU, CPU, RAM, OS, and relevant dependency versions.
- Model architecture configuration and trainable parameter count.
- Replay configuration and actual memory allocation.
- Debug mode, trace sampling rate, image/latent capture policy, compression, and storage cap.
- Enabled modules/feature flags and their effective values.
- Random-number generator stream names and initial seeds.
- Start time and declared experiment purpose/hypothesis.
- Any deviations from this plan, with reason and approval recorded in `deviations.md`.

If Git metadata is unavailable, record `unknown`; do not invent a commit SHA. If dependencies cannot be queried, record `not_recorded` and block benchmark claims until resolved.

### 32.4 Version every mutable schema

Events, relation records, visual-state records, checkpoints and reports must carry schema versions. Provide explicit migrations for old checkpoints. If migration cannot be performed losslessly, refuse to resume and provide a clear conversion report; never silently discard unknown fields.

## 33. Debug levels and their exact behaviour

Implement three levels. The default diagnostic pilot should use `full` for a short run with storage caps, then use `standard` for longer runs.

| Level | Must capture | Intended use |
|---|---|---|
| `off` | Run manifest, essential aggregate metrics, checkpoints, errors and gate results. | Throughput-only experiments after instrumentation is validated. |
| `standard` | Every real action decision and outcome; every memory mutation; all optimizer updates; batch/sample IDs; score components; compact latent statistics; periodic observations and artifacts. | Normal development training. |
| `full` | Standard plus raw observations or lossless image references for every transition; all candidate sequences and per-step rollout scores; full relation evidence; selected latent tensors/activations/gradients on configured diagnostic updates; detailed replay indices; deterministic replay package. | Short forensic pilots and root-cause studies. |

Rules:

1. Even in `off`, preserve enough data to identify run configuration, checkpoint, environment steps, optimizer updates, final gates and exceptions.
2. Every real action must have a decision event and an outcome event in `standard` and `full`.
3. Every relation-memory mutation must have an event in all debug levels whenever the relation system is enabled.
4. Candidate predictions can be expensive to store. In `standard`, record all scalar candidate scores and chosen sequence; in `full`, store each candidate sequence and each per-step scalar component. Save full latent tensors only for selected decisions, diagnostic triggers, and configured sampling, not by default for every candidate.
5. Debug level and storage caps must be fixed in the run manifest. Never silently reduce capture after a storage warning. Emit an explicit `trace_capture_degraded` event and mark the run's observability status as degraded.

## 34. Mandatory event catalogue

At minimum, implement the following `event_type` values. If one event type does not apply to a module, record a typed `not_applicable` status rather than fabricating data.

### 34.1 Environment and transition events

- `environment_reset_requested`
- `environment_reset_completed`
- `environment_step_requested`
- `environment_step_completed`
- `environment_exception`
- `transition_committed_to_replay`
- `episode_ended`
- `episode_aborted`
- `observation_artifact_saved`

Record reset seed on the evaluator side and a one-way correlation reference for the learner-side episode. Do not expose `info`, hidden inventory, achievements, coordinates, object IDs or other privileged state in learner events. Evaluator information must be stored in a separate directory and access path.

### 34.2 Perception and state-tracking events

- `visual_encoding_completed`
- `visual_slot_proposals`
- `visual_slot_association`
- `visual_attribute_estimate`
- `visual_state_snapshot`
- `visual_state_delta`
- `visual_state_uncertain`
- `visual_tracker_reset`
- `visual_tracker_drift_warning`

### 34.3 JEPA and learning events

- `context_encoded`
- `latent_prediction_requested`
- `latent_prediction_completed`
- `prediction_vs_observation_scored`
- `training_batch_sampled`
- `training_batch_validated`
- `learner_forward_completed`
- `loss_computed`
- `backward_completed`
- `optimizer_step_completed`
- `ema_update_completed`
- `gradient_anomaly`
- `representation_health_snapshot`
- `learning_gate_evaluated`

### 34.4 Action and planner events

- `action_candidates_scored`
- `action_decision`
- `action_outcome`
- `random_exploration_triggered`
- `planner_call_started`
- `planner_iteration_completed`
- `planner_candidate_scored`
- `planner_decision`
- `planner_outcome_evaluated`
- `planner_disabled_by_gate`
- `planner_risk_triggered`

### 34.5 Relationship and memory events

- `relation_candidate_created`
- `relation_candidate_updated`
- `relation_evidence_added`
- `relation_confidence_updated`
- `relation_contradiction_added`
- `relation_status_changed`
- `relation_merged`
- `relation_split`
- `relation_rejected`
- `relation_retrieval_query`
- `relation_retrieval_result`
- `relation_used_in_subgoal`
- `relation_outcome_verified`
- `relation_memory_checkpointed`
- `relation_memory_evicted`

### 34.6 Checkpoint, runtime and failure events

- `checkpoint_save_started`
- `checkpoint_save_completed`
- `checkpoint_save_failed`
- `checkpoint_load_started`
- `checkpoint_load_completed`
- `checkpoint_load_failed`
- `resume_consistency_checked`
- `runtime_health_snapshot`
- `storage_limit_warning`
- `trace_capture_degraded`
- `assertion_failed`
- `test_failed`
- `run_interrupted`
- `run_completed`

## 35. Exact action decision and action outcome tracing

This section is mandatory. It resolves the requirement “why did AEON take this action?” into fields that can be analyzed later.

### 35.1 Decision event

One `action_decision` event must be emitted before every real environment action. It must contain:

```text
observation_ref and observation_hash
current_history_transition_ids
current_state_snapshot_id, if tracker is enabled
current_latent_artifact_ref or latent_summary
context_state_artifact_ref or context_summary
policy_mode: warmup | random_floor | curiosity | mpc | relation_plan | fallback
role_id (policy metadata only; never a dynamics feature)
model_checkpoint_id and optimizer_update_id
planner_call_id, if applicable
relationship query/result IDs, if applicable
candidate action token list
for each token: raw novelty, raw disagreement, raw progress, predicted reward,
  predicted termination, uncertainty, normalized values, weighted components,
  final score, score status
normalizer snapshot/version used for the full candidate set
random_action_floor configured probability and the actual RNG draw
exploration temperature and categorical probabilities
selected token
selection reason enum
selection RNG stream ID and pre/post RNG state hashes
policy latency and each subcomponent latency
```

`selection_reason` must be a controlled enum, for example `uniform_warmup`, `random_floor`, `curiosity_sample`, `mpc_best_sequence`, `relation_subgoal_first_action`, `safety_fallback`, or `error_fallback`. Add a new enum only with a documented code path and test.

If a candidate cannot be scored, its status must be `not_scored` plus a reason. Do not substitute zero. If a policy branch does not calculate novelty or reward, set the field to `not_applicable`.

### 35.2 Candidate score calculation

For every action-selection method, preserve component scores before summing them. For a curiosity policy, store:

```text
score_raw = w_n * novelty_raw + w_d * disagreement_raw + w_p * progress_raw
score_used = w_n * novelty_normalized + w_d * disagreement_normalized
             + w_p * progress_normalized
```

If the implementation adds entropy bonuses, penalties, masks, risk terms or other components, each must be separately named and logged. Do not store only `score_used`.

Normalization must use one frozen normalizer snapshot for all candidates in a decision. Store the snapshot/version and values. Update the running normalizers once after the real transition has been observed, using the observed-data statistic specified in the config. Never update a normalizer while iterating over hypothetical action candidates. Candidate-order permutation tests must prove ranking invariance within numerical tolerance.

### 35.3 Action outcome event

After the environment step, emit exactly one `action_outcome` event linked to `decision_id`. Record:

- Actual opaque token executed.
- Adapter mapping hash and adapter event ID (not semantic mapping in learner-facing trace).
- Actual next observation reference and hash.
- Scalar reward, terminated, truncated.
- Wall-clock step latency and environment-step latency.
- Actual next visual-state snapshot/delta IDs, where available.
- Predicted next latent reference and actual EMA target latent reference, where available.
- Predicted-vs-actual latent distance and its exact metric definition/version.
- Predicted reward versus actual scalar reward.
- Predicted termination probability versus actual end flag.
- Any change-detection status and uncertainty.
- Whether the transition was committed to replay and the resulting replay index/sequence ID.
- Any error, retry, auto-reset or truncation detail.

Every decision must have one outcome, even if the environment errors or the process stops. In that case emit `action_outcome` with `outcome_status` `environment_error`, `process_interrupted`, or another declared status and link an exception event. Do not leave unmatched decisions unexplained.

### 35.4 Action-distribution diagnostics

At every episode and every 1,000 real steps, record:

- Count and percentage for each opaque token.
- Token entropy and effective number of actions `exp(entropy)`.
- Longest consecutive repetition of a token.
- Transition-conditioned token usage where the visual context can be grouped without semantic labels.
- Policy mode distribution and random-floor rate.
- Score mean/std/min/max by component and token.
- Frequency of safety/error fallbacks.

A low action entropy is not automatically a bug; it is an alert requiring context. Report the action histogram alongside reward, state coverage and task metrics.

## 36. Visual perception and state-tracker forensics

### 36.1 What the tracker is allowed to do

The tracker must learn from RGB pixels. It may create anonymous visual slots/prototypes and temporal associations. It must not use Crafter's internal inventory, map, entities, object names, source code, OCR output, hand-labelled object categories, or a handcrafted recipe table.

The implementation may use learned attention, object-centric representations, spatial feature clustering or learned patch tracking. The chosen method must be documented and ablated; the plan does not mandate a specific research algorithm. Do not hard-code item names or crafting semantics into its feature extractor.

### 36.2 Per-observation state snapshot

For each observation, store:

- Observation hash and artifact reference.
- Encoder/checkpoint version.
- Number of proposed anonymous slots/prototypes.
- For each slot: anonymous slot ID, feature-vector artifact reference, spatial attention/region reference if used, confidence, estimated persistence, estimated quantity/attribute outputs, and all uncertainty values.
- Association from previous slot IDs to current slot IDs, with match score and unmatched/new status.
- Tracker state before and after update.
- Any tracker reset, drift, occlusion or ambiguous-match warning.
- Time/compute used by each perception stage.

Slot IDs are temporary tracking IDs, not semantic object identities. A slot may not be called `wood`, `table`, `stone`, `pickaxe`, or another game term in learner code or training traces.

### 36.3 Visual state delta

For every consecutive pair of observations, create a structured delta:

```text
previous_snapshot_id
current_snapshot_id
matched_slots: [{previous_slot_id, current_slot_id, match_confidence}]
unmatched_previous_slots
new_current_slots
per-slot feature_delta_ref
per-slot attribute_delta estimates
quantity_change estimates with uncertainty
spatial_change estimates with uncertainty
global_scene_change summary
tracker_confidence and ambiguity status
```

A quantity estimator must retain a distribution or uncertainty estimate where possible, not just an integer. If it predicts `2 -> 0`, record raw predicted distributions and confidence. If the pixels are ambiguous, report unknown. Do not force a precise count from insufficient visual evidence.

### 36.4 Distinguish world changes from camera/view changes

The player and viewport can move. The tracker must diagnose whether a changed image is due to camera motion, player movement, lighting/day-night, animation, object interaction or an uncertain cause. Do not treat every pixel delta as an object creation/destruction event.

Required controls/diagnostics:

- Action-independent consecutive-frame baseline.
- Repeated-frame/no-op comparison where applicable.
- Global motion or scene-shift estimate and its confidence.
- Persistence test: does a candidate slot remain after subsequent observations?
- Occlusion/disappearance status distinct from destruction.
- Lighting/colour-shift sensitivity check.
- Tracker association quality on synthetic transformations that preserve unknown semantics.

If the tracker cannot separate these effects, it must reduce confidence and avoid creating a strong relation candidate from the ambiguous delta.

### 36.5 Perception-specific root-cause tests

- Same frame twice: slot and quantity estimates should be stable within the declared tolerance.
- Small visual perturbation: tracker stability must be measured, not assumed.
- Camera/viewport movement: anonymous association should not invent a new object for every shifted tile.
- Night/day visual tint: track continuity should be measured across lighting changes.
- Occlusion and reappearance: distinguish uncertainty from permanent disappearance.
- Repeated identical inventory artwork at different counts: quantity predictions must respond to the visible digit change.
- Held-out worlds: tracker outputs should not rely on memorized coordinates or one seed's layout.

All synthetic test transformations must be documented and must not be used as semantic labels in the learner.

## 37. JEPA inference, training, and representation diagnostics

### 37.1 Inference trace

For every prediction sampled for standard/full debugging, store:

- Input observation and history references.
- Exact opaque action token sequence.
- Context mask and padding pattern.
- Online encoder and EMA target checkpoint IDs.
- Latent shape, dtype, device, normalization convention and artifact reference.
- Per-ensemble-member prediction summary and optional tensor artifact.
- Ensemble mean, disagreement definition, and disagreement value.
- Predicted reward/termination outputs and calibration version.
- Prediction horizon and whether teacher-forced or free-running.
- Any non-finite value, saturation, clipping or exception.

### 37.2 Training update trace

Every optimizer update must record:

- `optimizer_update_id` and environment step range since prior update.
- `batch_id`, sample IDs and replay indices/transition IDs.
- Batch sequence metadata and integrity-check result.
- Batch action-token histogram.
- Loss per predictor member and per horizon.
- Target-token mask statistics and bootstrap mask statistics.
- JEPA, reward and termination loss separately; total loss separately.
- Learning rate for every parameter group.
- Gradient scaler scale and skipped/overflow status.
- Global gradient norm before clipping and after clipping (where measured).
- Per-module gradient norm and percentage of parameters with missing/non-finite/zero gradients.
- Parameter norm and parameter update norm by module.
- EMA momentum, target-online parameter distance, and actual EMA update confirmation.
- Batch processing time, data wait time, forward/backward time, optimizer time and peak GPU memory.

Do not store every gradient tensor by default. Store per-module statistics every update, and full named gradients/activations only on configured diagnostic updates or when an anomaly trigger fires. If a trigger fires, save the batch and checkpoint needed to reproduce it.

### 37.3 Loss diagnostics

A decreasing total loss can conceal one broken head or one bad horizon. Therefore:

- Plot/log each loss separately by horizon, predictor member and auxiliary head.
- Log per-sample loss quantiles, not only the mean.
- Record whether one member dominates the ensemble loss.
- Record predicted-target cosine by token and action-sensitive context group when valid.
- Compare training, validation and genuinely held-out loss on separate stores.
- Track error versus sample age and episode seed.
- Detect repeated/duplicate frames and repeated sequences in sampled batches.
- Report loss scale before and after any clipping or masking.

### 37.4 Representation health

Retain existing collapse metrics and add:

- Per-channel mean/std and feature norm quantiles.
- Effective rank and eigenvalue spectrum, not only a scalar participation ratio.
- Pairwise cosine distributions across same-frame augmentations and distinct frames.
- Latent variance across time within an episode and across episodes.
- Per-spatial-token variance to detect spatial collapse.
- Target-online representation drift.
- Predictor member diversity and pairwise disagreement distribution.
- Dead/saturated channel fraction.
- Feature health on a fixed, versioned diagnostic frame set and on fresh held-out frames.

All diagnostic frame sets must be stored as evaluator-only observation references. Never use the evaluator's hidden annotations as training labels.

### 37.5 Action-conditioning diagnostics

The paired test must pass the exact training action mapping to every branch that intends to execute the same opaque token. Record:

- Training mapping hash and evaluator adapter mapping hash.
- Token and actual mapped environment action index in evaluator-only files.
- Branch root-state reproducibility status.
- Initial observation hash equality across branches.
- Actual per-action next-frame and target-latent references.
- Conditioned, no-action, shuffled-action and random-predictor errors.
- Whether action changes are visually distinguishable in the branch outcome.

If the same root state cannot be reproduced, mark the branch as unpaired and do not use it in the paired action-conditioning gate. If exact environment state cloning is not supported by the installed Crafter version, do not claim paired counterfactual testing; use the closest documented seed-and-prefix protocol and disclose its limitation.

## 38. Replay, sampler, and dataset provenance diagnostics

### 38.1 Transition provenance

Each transition must retain stable references to:

- Current observation artifact/hash.
- Opaque action token.
- Next observation artifact/hash.
- Scalar reward and end flags.
- Episode/step/worker IDs for bookkeeping.
- Replay insertion event and ring-buffer slot/version.
- Action decision and outcome event IDs.

A replay slot may be overwritten. Therefore sample records must refer to a stable transition ID and a generation/version counter, not only an integer ring index.

### 38.2 Sequence sampling trace

For every sampled batch (standard/full), record:

- Batch ID, sample IDs, chosen anchor transition IDs, episode IDs, start/end timesteps.
- Context transition IDs and target observation IDs for offsets 1/2/4/8.
- Future action token sequence.
- Context padding mask and previous-action alignment.
- Every `terminated`/`truncated` boundary encountered or checked.
- Ring-buffer generation counters used to ensure frames were not overwritten.
- Sampling source: uniform, recent, prioritized, or other named strategy.
- Sampling probabilities/weights if non-uniform sampling is used.
- Whether the sample was accepted or rejected, and rejection reason.

### 38.3 Data-integrity checks

Required checks include:

- Exact action-to-next-observation alignment.
- Correct target frame at offsets 1, 2, 4 and 8.
- No sequence crosses episode boundaries or terminal flags.
- No stale ring-buffer indices after wrap-around.
- No observation hash collision with mismatched pixels (hash equality must imply byte equality in the verification path).
- No accidental train/held-out/final-evaluation seed overlap.
- No held-out observation inserted into training replay.
- No duplicate sequence leakage across split boundaries.
- Correct `uint8` storage and `[0,1]` preprocessing.
- Correct shape, finite scalar reward, valid token range and boolean end flags.
- No `info` or evaluator metadata in replay samples or model batches.

### 38.4 Dataset fingerprint

At each checkpoint and final report, write a dataset manifest with counts by split, environment ID, episode count, transition count, action-token histogram, reward statistics, episode length statistics, termination/truncation statistics, observation hashes or hash-tree root, replay capacity/occupancy and sampler version. The manifest must not expose hidden game state to the learner.

## 39. Relationship-discovery evidence and memory forensics

### 39.1 Relationship candidate is a hypothesis, not a fact

A candidate record must contain:

```text
relation_id
relation_revision_id
candidate_created_event_id
context_signature_ref
precondition_feature_refs
action_token
predicted_effect_feature_refs
observed_effect_feature_refs
quantity/attribute deltas with uncertainty
candidate_generation_method and version
supporting_evidence_event_ids
contradicting_evidence_event_ids
control_evidence_event_ids
independent_episode_count
independent_world_count (where defined)
confidence value and confidence method/version
status: candidate | supported | provisional | rejected | stale | superseded
last_tested_event_id
next_experiment_rationale
```

Fields must be anonymous learned feature references or visual-state IDs. Human-friendly names may be attached only in a separate evaluator annotation file.

### 39.2 Evidence ledger

Evidence must be append-only. Do not edit old evidence to make a hypothesis look stronger. Every new observation creates a new evidence record with:

- Source transition ID and decision ID.
- Before/after visual-state snapshots and deltas.
- Action token and model version at decision time.
- Tracker confidence and ambiguity status.
- Candidate precondition match score.
- Predicted effect and actual observed effect.
- Whether this is supporting, contradictory, neutral, or inconclusive evidence.
- Whether the evidence is from a new episode/world or correlated with earlier evidence.
- Any confounder flags (camera movement, lighting, occlusion, damage, terminal event, uncertain tracking, concurrent state changes).
- The rule/version used to score the evidence.

### 39.3 Confidence must be decomposable

Do not save an unexplained confidence number. Record the components used to calculate it, such as:

- Number of support cases.
- Number of contradictory cases.
- Number of independent episodes/worlds.
- Visual-state match quality.
- Effect consistency and variance.
- Precondition specificity.
- Control comparison result.
- Uncertainty in feature/quantity estimates.
- Data quality/confounder penalties.
- Confidence calibration method and version.

If a Bayesian, frequentist, learned-calibration or heuristic confidence method is selected, document its assumptions, formula, prior/threshold source, calibration data and limitations. Do not invent a confidence value from counts without explaining dependence and correlation.

### 39.4 Contradiction and forgetting diagnostics

When a new observation contradicts a stored relation:

1. Preserve the old relation revision.
2. Append the contradicting evidence.
3. Record whether preconditions appeared to match and how confidently.
4. Check for tracker mismatch, action-mapping mismatch, changed visual context, stochastic outcome, damaged/absent station, or previous evidence correlation.
5. Update status/confidence using the declared method.
6. Do not silently delete the relationship.
7. If evidence suggests multiple contexts were incorrectly merged, allow a documented relation split into more specific hypotheses.

Track false positive and false negative relation candidates using evaluator-only ground truth after the experiment, not as learner feedback. Report precision/recall only where evaluator annotations are reliable and the evaluation set was not used to create the relation.

### 39.5 Memory lifecycle events

Log every create, update, strengthen, weaken, merge, split, reject, retrieve, use, evict, archive and reload operation. Each mutation must include:

- Relation ID and old/new revision ID.
- Exact evidence event IDs used.
- Old and new status/confidence components.
- Algorithm/config version.
- Reason code and human-readable diagnostic explanation.
- Checkpoint ID containing the mutation.

A relationship may not change confidence/status without an evidence reference or an explicitly documented maintenance operation (for example, staleness/decay). Maintenance operations must be distinct from evidence-driven updates.

### 39.6 Retrieval observability

For each retrieval query, record:

- Query visual-state snapshot and context signature.
- Number of relations searched, filtered, retrieved and rejected.
- Match feature distances/similarities and thresholds.
- Confidence and status of every returned relation.
- Why each candidate was included/excluded.
- Retrieval top-k, cutoff and deterministic tie-break.
- Memory index/version and checkpoint ID.
- Whether a retrieved relation was actually used in a subgoal or action plan.

A relation that exists but is not retrieved is a retrieval failure, not a discovery failure. A relation retrieved but not selected by planning is a planning/policy issue. These must remain separately diagnosable.

## 40. Subgoal, dependency graph, and compositional planning traces

### 40.1 Planning layers must remain separable

Keep distinct logs for:

1. **Goal proposal:** what target state or learned effect AEON is trying to reach.
2. **Relation retrieval:** which previously supported relations are relevant.
3. **Dependency composition:** how outputs of one relation are hypothesized to satisfy preconditions of another.
4. **Subgoal planning:** ordering, feasibility and resource constraints inferred from visual evidence.
5. **Low-level action planning:** JEPA predictions and CEM action-token sequences.
6. **Execution monitoring:** predicted versus observed effect, subgoal progress and replanning.

Do not attribute a failure in one layer to another without trace evidence.

### 40.2 Plan record

Each plan must have a `plan_id` and record:

- Current visual-state snapshot ID and model/relation-memory versions.
- Proposed goal representation and how it was proposed.
- Subgoal IDs and dependency edges.
- Relation IDs/revisions supporting each edge.
- Preconditions, effects and uncertainty references.
- Resource/quantity estimates and confidence.
- Plan feasibility score and rejected alternatives.
- Expected sequence length/cost/risk and predicted outcomes.
- First action token and its linked action decision ID.
- Termination/replanning condition.

If the planner cannot find a valid dependency chain, record `no_supported_plan` and fall back to exploration or a documented policy. Do not fabricate a recipe or insert a hand-coded fallback sequence.

### 40.3 Plan execution trace

After every executed action, record:

- Active plan/subgoal IDs.
- Expected observation/effect.
- Observed visual state/delta.
- Subgoal progress estimate and uncertainty.
- Whether the expected effect occurred.
- Whether plan was continued, revised, abandoned or completed.
- Exact reason and evidence event IDs.

### 40.4 Planning metrics

Measure:

- Goal proposal rate and acceptance by evaluator-only diagnostic.
- Supported-plan rate.
- Subgoal completion rate.
- Relation-use rate and relation-use success rate.
- Plan length and environment steps per completed subgoal.
- Unnecessary repeated experiments after relation retrieval.
- Invalid plan rate due to stale/mismatched preconditions.
- Predicted-versus-observed subgoal effect accuracy.
- Planner fallback frequency.
- Time/latency per planning stage.

## 41. Checkpoint and resume forensics

### 41.1 Complete checkpoint contents

Extend the existing checkpoint specification. A resumable training checkpoint must include:

- Online model and all target/EMA parameters and buffers.
- Optimizer, LR scheduler, gradient scaler and learner update counter.
- Replay arrays, pointers, generation counters, episode/timestep metadata and dataset manifest hash.
- Opaque action mapping and its hash.
- Python, NumPy, CPU Torch and CUDA RNG states, plus separate RNG streams for environment workers, policy, sampler, CEM and diagnostics.
- Novelty latent memory, normalizer state, disagreement normalizer state and learning-progress state.
- Relationship memory, evidence ledger cursor, relation index/version and memory configuration.
- Visual tracker parameters/state, slot association state and tracker version.
- Current exploration role and role scheduler state.
- Collector state and per-worker current episode status, including uncommitted trajectory buffers if a mid-episode resume is supported.
- Planner settings and any stateful planner distribution/cache.
- Validation gate results and the IDs/hashes of the held-out dataset used to calculate them.
- Current environment-step count, optimizer-update count, completed episodes and wall-clock accounting.
- Full config snapshot, config hash, repository commit, package manifest and checkpoint schema version.
- Last fully committed trace-event ID and event-log offsets/checksums.

If an environment instance's internal state cannot be restored, the checkpoint must declare the limitation. It may resume only at a clean episode boundary, or must explicitly reset and log a new episode. It must not pretend to continue the same episode.

### 41.2 Atomic checkpoint protocol

1. Pause collection at a safe boundary or mark the exact collector state being captured.
2. Flush all event writers and artifact indexes.
3. Create a temporary checkpoint in the target directory.
4. Save all required state and manifests.
5. Reopen and validate the temporary file.
6. Verify required keys, tensor shapes, mapping hash, replay counters, memory versions, RNG streams and event-log cursor.
7. Calculate checksum.
8. Atomically rename to immutable checkpoint filename.
9. Update the `latest` pointer only after successful validation.
10. Emit `checkpoint_save_completed` with checksum and file size.

On failure, keep the previous valid checkpoint and emit `checkpoint_save_failed`; do not overwrite the last good checkpoint.

### 41.3 Resume verification

A resume test must compare uninterrupted and interrupted/resumed runs under deterministic settings:

- Run A: N transitions without interruption.
- Run B: save at K, close process, reload, continue to N.
- Compare action-token sequence, environment observations, replay state, sampled batch IDs, optimizer updates, model parameters, EMA parameters, relation memory and final metrics.
- Report exact equality or measured numerical divergence for each category.
- If the environment cannot be restored mid-episode, conduct the test at episode boundaries and state that exact mid-episode resume is unsupported.

Do not label resume `PASS` if only the model weights load. The entire state machine must meet the documented resume contract.

## 42. Deterministic replay and reproducibility levels

### 42.1 Three different replay concepts

Use precise names:

1. **Trace inspection:** read stored events, observations and artifacts; no environment is rerun.
2. **Transition replay:** re-feed stored transitions/batches to offline diagnostic functions or a cloned model to reproduce predictions/losses. This does not rerun Crafter.
3. **Environment re-execution:** instantiate Crafter and reapply the recorded reset seed and action-token sequence to attempt to reproduce observations. This is only exact if the installed environment and its RNG/state behavior are deterministic under that procedure.

Do not call trace inspection “deterministic environment replay.”

### 42.2 Record replay inputs

For every episode, save:

- Environment package/version and env ID.
- Reset seed and mapping hash.
- Ordered opaque action tokens.
- Observation hashes at every step.
- Termination/truncation outcomes.
- Any evaluator-only seed/state metadata in a separate restricted file.
- RNG state snapshots if supported and permitted for the environment/evaluator.

### 42.3 Replay verification

On environment re-execution, compare the new observation hash to the original after each step. Stop at the first mismatch and emit:

- First divergent step.
- Original/replayed observation hashes and image references.
- Action token and adapter mapping hash.
- Package/config/code versions.
- Previous step match status.
- Environment determinism support status.

If exact reproducibility is not achieved, keep original observations as the authoritative record and use transition replay for model debugging. Never alter original observations to match replay.

### 42.4 Determinism labels

Every run must report one of:

- `bitwise_deterministic`
- `numerically_reproducible_within_tolerance`
- `environment_replay_only`
- `trace_replay_only`
- `non_reproducible_with_reason`

The label must be based on an executed reproducibility test, not a config flag.

## 43. Failure detection, exception capture, and first-divergence analysis

### 43.1 Failure taxonomy

Use a controlled taxonomy with at least these top-level categories:

- `ENVIRONMENT`: reset/step/API/environment crash or unexplained transition.
- `ADAPTER`: action mapping, observation conversion, reward/end-flag normalization.
- `DATA`: replay corruption, sequence misalignment, leakage, split contamination.
- `PERCEPTION`: encoder/tracker/association/quantity estimate failure.
- `WORLD_MODEL`: JEPA prediction, action conditioning, rollout drift or collapse.
- `OPTIMIZATION`: gradients, optimizer, EMA, precision overflow or scheduler.
- `EXPLORATION`: novelty/disagreement/progress/action score/normalization failure.
- `RELATION_DISCOVERY`: candidate generation, condition/effect extraction or evidence evaluation.
- `RELATION_MEMORY`: consolidation, persistence, contradiction, retrieval or versioning.
- `PLANNING`: goal proposal, dependency graph, subgoal ordering, CEM or scoring.
- `EXECUTION`: action issued correctly but expected state transition not achieved.
- `CHECKPOINT`: save/load/resume inconsistency.
- `OBSERVABILITY`: missing/corrupt trace, storage degradation, event ordering or schema issue.
- `EVALUATION`: invalid held-out split, wrong mapping, score computation or metric leakage.
- `RESOURCE`: CPU/GPU/RAM/disk/time/worker queue bottleneck.
- `UNKNOWN`: insufficient evidence or multiple unresolved explanations.

A failure can have multiple contributing categories. Record a primary category only if evidence supports it; otherwise use `UNKNOWN` and list candidate explanations without ranking them as facts.

### 43.2 Component boundary health checks

Every boundary must validate input/output contracts and emit a result:

```text
Environment -> Adapter
Adapter -> Policy
Observation -> Encoder
Encoder -> Context
Context + Action -> Predictor
Predictor -> Planner/Policy
ActionSelector -> Adapter
Adapter -> Replay
Replay -> SequenceSampler
SequenceSampler -> Learner
Learner -> Optimizer/EMA
Observation/Delta -> Relationship Candidate Generator
Candidate -> Evidence Ledger
Evidence Ledger -> Relation Memory
Relation Memory -> Retrieval
Retrieval -> Subgoal Planner
Subgoal Planner -> CEM/Action Policy
Checkpoint -> Restored Runtime
Trace Writer -> Durable Artifact Store
```

For each boundary, record shape, dtype, device, finite-value status, range checks, ID references and timing. On validation failure, preserve the failing payload/artifact and stop the dependent stage.

### 43.3 First-divergence workflow

For every failed episode or gate:

1. Identify the failing goal/metric and the earliest relevant event.
2. Reconstruct the exact observation/history and action decision.
3. Verify adapter action-token mapping.
4. Verify that the transition was stored with correct action/next-frame alignment.
5. Verify the state tracker correctly associated relevant visual evidence.
6. Verify JEPA prediction against the actual next observation.
7. Verify relation candidate/evidence logic if applicable.
8. Verify memory mutation and subsequent retrieval.
9. Verify dependency composition and subgoal selection.
10. Verify low-level action selection/planner scoring.
11. Verify environment execution and actual result.
12. Identify the earliest stage whose output violates its own contract or is inconsistent with the next trusted observation.
13. Create a root-cause report with evidence, alternative hypotheses, tests run and confidence in diagnosis.

The earliest observed divergence is a *candidate root cause*, not automatically the sole cause. Check downstream contributing errors and interacting faults.

### 43.4 Root-cause report template

Every failed gate/episode with a material failure must generate `root_cause_report.md` and machine-readable `root_cause_report.json`:

```text
Failure ID:
Run/checkpoint/config/code revision:
Episode/step/decision/transition IDs:
Observed symptom:
Expected behaviour (testable definition):
Actual behaviour:
First failing boundary:
Primary failure category:
Supporting event/artifact IDs:
Reproduction command/procedure:
Reproduction result:
Candidate causes:
Tests used to confirm/eliminate each candidate:
Confirmed cause(s):
Unresolved alternatives:
Contributing factors:
Fix proposed:
Regression test required:
Owner/status:
```

Use `confirmed cause: none established` where evidence is insufficient. The plan explicitly forbids filling this section with a guess.

## 44. Failure injection and fault-isolation tests

Do not wait for accidental failures to test the debug system. Create controlled test doubles/failure injections that corrupt one component at a time and verify that the correct boundary detects and localizes the issue.

Mandatory injections:

1. Invalid action token outside `[0,16]`.
2. Wrong opaque mapping in a validation branch.
3. Missing or duplicated action/observation transition.
4. Replay ring-buffer wrap-around stale index.
5. Cross-episode sample attempt.
6. Corrupted observation shape/dtype/range.
7. Non-finite latent or predictor output.
8. Target encoder accidentally requires gradients.
9. EMA update skipped or performed twice.
10. One predictor member stops receiving gradients.
11. Artificial representation collapse.
12. Action token replaced or shuffled in a training batch.
13. Held-out episode intentionally inserted into training replay.
14. Candidate score normalizer updated during candidate loop.
15. Novelty memory update omitted or applied multiple times per observation.
16. Learning-progress update omitted or receives the wrong error.
17. Tracker slot identity deliberately swapped.
18. Quantity estimate deliberately marked uncertain.
19. One relationship support event duplicated to test correlated-evidence handling.
20. A contradictory evidence event injected.
21. Relation checkpoint missing one revision/evidence reference.
22. Relation retrieval returns a stale revision.
23. Dependency edge points to an unsupported relation.
24. Planner reward head deliberately overpredicts a candidate.
25. Planner termination probability deliberately underpredicts risk.
26. Checkpoint file truncated/corrupted.
27. RNG state omitted at checkpoint.
28. Event log disk write failure/storage full.
29. Worker process crashes after decision but before outcome.
30. Debug mode enabled/disabled under same deterministic input.

For each injection, assert:

- The intended fault was actually injected.
- The expected guard/test detects it.
- The emitted event includes the relevant ID and boundary.
- The run is blocked or marked degraded as specified.
- No false `PASS` is emitted.
- A regression test prevents recurrence.

Do not inject faults into live user data or normal benchmark runs. Use isolated test environments and synthetic inputs.

## 45. Calibration, uncertainty, and confidence diagnostics

### 45.1 Predictive uncertainty

Ensemble disagreement is a heuristic epistemic-uncertainty proxy, not a calibrated probability. Record raw disagreement, ensemble member outputs and any calibration procedure separately. Do not label raw disagreement “probability of error.”

### 45.2 Prediction calibration

On held-out data, evaluate:

- Latent prediction error distribution by horizon.
- Error versus ensemble disagreement bins.
- Error versus visual-state tracker confidence.
- Error versus context novelty.
- Error versus action-token frequency.
- Error versus relation retrieval confidence.
- Coverage of uncertainty intervals, if the model emits intervals.

A calibration method must be fitted on a development/calibration split separate from final evaluation. Report calibration error and sample counts. Do not tune a threshold on the final test set.

### 45.3 Relation confidence calibration

If relation confidence is used for planning, test whether confidence predicts actual relation correctness on an independent annotated evaluation set. Report reliability diagrams, Brier/log loss where probability semantics are justified, and precision/coverage at decision thresholds. If confidence is only a heuristic rank, call it a score, not probability.

### 45.4 Unknown and abstention behaviour

The agent must be able to abstain from using a weak relation. Track:

- How often it reports unknown/insufficient evidence.
- How often it requests further experimentation.
- False confident relation use.
- Missed opportunities due to excessive caution.

A system that always makes a guess is not better than one that recognizes insufficient evidence.

## 46. Evaluation integrity and statistical analysis

### 46.1 Split discipline

Maintain separate stores for:

- Training data/replay.
- Development validation episodes used for gate decisions and debugging.
- Calibration data used to fit confidence/calibration transforms.
- Final frozen evaluation episodes used only once for final claims.

Store split manifests and hashes. Never update model parameters, relation memory, thresholds, normalizers or planner weights using final evaluation episodes. If any final evaluation data is used to fix a bug, reclassify it as development data and collect a fresh final split.

### 46.2 Matched comparisons

For every ablation/comparison, match:

- Total environment steps and reset/episode protocol.
- Training seed count and seed list where a paired design is possible.
- Action-token permutation protocol.
- Hardware/software and training budget.
- Model capacity except for the intentionally removed component.
- Evaluation seeds and stopping rules.
- Online learning/frozen-policy status.

Report any unmatched condition. Do not compare a 50k model against a 100k model as if the comparison isolated one module.

### 46.3 Statistical reporting

For every behavioural metric, report sample counts and uncertainty:

- Per-seed results, mean and standard deviation.
- Confidence intervals with the method stated (for example, bootstrap over independent episodes or seeds, with the resampling unit specified).
- Per-achievement rates and the aggregate Crafter score computed according to the benchmark specification.
- Distribution across episodes, not just a maximum or best episode.
- Failure/interrupted runs and exclusion reasons.

Do not treat multiple episodes from one trained seed as fully independent model replications. Report seed-level variation separately from episode-level variation.

### 46.4 Transfer test must isolate memory benefit

For frozen transfer, define exactly what is frozen:

- Model weights frozen or continuing to update?
- Relationship memory frozen or updating?
- Novelty memory frozen or updating?
- Normalizers frozen or updating?
- Policy/planner learning frozen or updating?

Primary relation-transfer test: freeze model weights, relation memory, confidence calibration, thresholds and planner/policy parameters after discovery. Permit only ordinary within-episode context tracking. Do not allow new relation evidence or model updates in the evaluation worlds. A separate online-adaptation experiment may be run, but must not be described as frozen transfer.

Use the same action-token mapping across worlds within one training/evaluation run. For independent runs, mappings may differ; do not compare raw token IDs across runs as if they had the same environment meaning.

## 47. Runtime, resource, and throughput diagnostics

At startup and every configured interval, record:

- CPU and RAM utilization.
- GPU name, utilization, allocated/reserved/peak memory, temperature if available.
- Environment steps/second per worker and aggregate.
- Learner updates/second.
- Environment wait time, queue depth and worker idle time.
- Replay insertion and sampling latency.
- Image/latent encoding latency.
- Planner/CEM latency, candidates evaluated per second and time per candidate.
- Relationship retrieval and graph-planning latency.
- Trace serialization/write latency and pending queue depth.
- Checkpoint duration and bytes written.
- Disk free space, storage cap and artifact counts.
- Data-loader/batch preparation wait time.

Set configurable warning thresholds with values in config. A warning threshold is a resource alert, not an algorithmic failure threshold. Record the exact measurement and configured threshold. Do not silently reduce workers, debug level, replay size or candidate count under resource pressure; either stop with a clear error or apply an explicitly configured adaptive policy and log the change.

## 48. Artifact retention, storage budgets, and integrity

### 48.1 Required run directory

Use a structure equivalent to:

```text
runs/<experiment_id>/<run_id>/
  run_manifest.json
  config.yaml
  package_manifest.json
  action_mapping_restricted.json
  seeds/
    train_seeds.json
    validation_seeds.json
    calibration_seeds.json
    final_eval_seeds.json
  logs/
    events.jsonl
    metrics_step.jsonl
    metrics_episode.jsonl
    metrics_update.jsonl
    errors.jsonl
    trace_index.jsonl
  data/
    replay_manifest.json
    episode_manifest.jsonl
    split_manifests/
  observations/
    index.jsonl
    frames/                 # PNG/WebP/lossless source as configured
  latents/
    index.jsonl
    selected_tensors/       # FP16/FP32 with dtype recorded
  decisions/
    action_decisions.jsonl
    action_outcomes.jsonl
    planner_calls.jsonl
  relationships/
    relation_events.jsonl
    relation_snapshots/
    evidence_index.jsonl
    dependency_graphs/
  checkpoints/
    manifest.jsonl
    checkpoint_*.pt
  evaluation/
    heldout_prediction.json
    action_conditioning.json
    relation_discovery.json
    relation_transfer.json
    planning_utility.json
    crafter_score.json
    per_achievement.csv
  diagnostics/
    representation/
    calibration/
    replay_integrity/
    root_causes/
    fault_injection/
  videos/
  reports/
    debug_report.md
    final_report.md
```

The exact directory names may vary only if an equivalent manifest maps them. Do not store the restricted action mapping in learner-accessible training data.

### 48.2 Storage controls

- Calculate expected disk usage before a full-debug run.
- Use a configured maximum run artifact size and per-category quotas.
- Keep event JSONL append-only; rotate only with an index and checksum.
- Compress raw screenshots losslessly or with a documented quality setting. Keep a content hash of the original bytes before lossy conversion.
- Store latent tensors with explicit dtype/shape/version metadata.
- Keep all training-critical checkpoint/replay/relationship-memory files separate from disposable visualizations.
- Never delete raw evidence automatically to make room without emitting an event and updating the run's observability status.
- When disk is low, stop or switch only according to an explicitly configured and logged policy. Never mark a degraded trace as complete.

### 48.3 Integrity checksums

Create SHA-256 hashes for config, event-log segments, observation artifacts, dataset manifests, checkpoints and final report. Maintain a hash index. On report generation, verify every referenced artifact exists and matches its checksum. List missing/corrupt artifacts explicitly.

## 49. Human-facing diagnostic report and inspection tools

The project must provide a CLI/report workflow. A full interactive dashboard is optional; machine-readable, reproducible evidence is mandatory.

### 49.1 Required commands (or equivalent documented commands)

```bash
# Validate repository/config/environment before a run
python run.py --preflight --config configs/crafter_t4.yaml

# Run a short fully traced pilot
python run.py --config configs/crafter_t4.yaml --max-steps 5000 --debug full

# Inspect one run without modifying it
python tools/inspect_run.py --run-dir runs/<run_id>

# Reconstruct one decision and its outcome
python tools/inspect_decision.py --run-dir runs/<run_id> --decision-id <id>

# Inspect one relationship and its complete evidence history
python tools/inspect_relation.py --run-dir runs/<run_id> --relation-id <id>

# Compare predicted and actual outcomes for one transition
python tools/inspect_transition.py --run-dir runs/<run_id> --transition-id <id>

# Replay stored transition data through a saved model (not environment re-execution)
python tools/replay_transition.py --run-dir runs/<run_id> --transition-id <id>

# Verify checkpoint and event/data checksums
python tools/verify_run.py --run-dir runs/<run_id>

# Generate forensic report
python tools/generate_debug_report.py --run-dir runs/<run_id>
```

These are required command capabilities. If names or argument formats differ, the README must provide exact equivalents. Do not leave placeholders in the actual README after implementation.

### 49.2 Required views in `debug_report.md`

1. Run identity, configuration, code commit, hardware and dependency manifest.
2. Test/preflight status and all deviations.
3. Environment steps, episodes, optimizer updates, throughput and interruptions.
4. Action-token distribution, policy-mode distribution and action entropy over time.
5. Example decisions with all 17 candidate scores and selected-action rationale.
6. JEPA prediction errors by horizon/member and comparison with baselines.
7. Representation-health plots and collapse alerts.
8. Reward/termination calibration and imagination gap.
9. Replay integrity, sample coverage and train/validation split proof.
10. Visual-state tracker confidence, association failures and example before/after frames.
11. Relationship lifecycle table: candidates, confirmed/provisional/rejected, contradictions and evidence counts.
12. Memory retrieval/use statistics and examples of failed retrieval.
13. Dependency/subgoal plan traces and completion/failure outcomes.
14. Root-cause reports and unresolved failures.
15. Checkpoint/resume/replay determinism results.
16. Storage/trace completeness and missing artifact list.
17. Final gate statuses, with explicit `not_run` and `not_verified` values.

### 49.3 One-click episode dossier

For any selected episode, produce a dossier containing:

- Episode video or ordered observations.
- Action timeline with policy mode and reason.
- Candidate scores and planner sequence snapshots.
- Visual state snapshots/deltas.
- JEPA predicted-versus-actual plots.
- Relevant relation candidates/evidence/retrievals.
- Subgoal timeline.
- Rewards/end flags and evaluator-only achievement summary in a separate panel/file.
- The first divergence and root-cause report, if any.

Learner-visible information and evaluator-only annotations must be visibly and physically separated in the dossier. The report generator must not merge evaluator annotations back into data files used for training.

## 50. Expanded failure-mode map and diagnostic response

Use the table below as an investigation guide, not as an automatic root-cause oracle. Each row lists evidence to collect before assigning a cause.

| Symptom | First evidence to inspect | Candidate explanations to test | Required conclusion rule |
|---|---|---|---|
| Agent repeats one token | Action histogram, decision scores, RNG/policy mode, state coverage | Normalization bug, action mapping, score collapse, planner tie-break, legitimate stable optimum | Do not call it policy collapse until candidate scoring and mapping are verified. |
| Agent moves but never interacts | Token mapping, state/action trace, environment outcome, action-sensitive test | Adapter mapping mismatch, visual state not recognized, interaction action not sampled, predictor ignores actions | Identify the earliest boundary with evidence. |
| Visual tracker misses inventory change | Original image, slot proposals, association, quantity distributions | Digit ambiguity, small HUD resolution, changed slot association, encoding collapse | Preserve image and raw prediction; do not hand-correct learner data. |
| False relationship created | Candidate evidence and confounder flags | Camera motion, correlated repeated frames, stochastic effect, state mismatch, wrong action mapping | Require independent controls/episodes before confirmation. |
| True relationship not discovered | State deltas, candidate-generation recall, hypothesis threshold | Perception missed effect, effect below threshold, candidate generation incomplete, insufficient experiments | Identify whether missing evidence began in perception or candidate generation. |
| Relationship confidence too high | Confidence components, duplicate evidence, calibration set | Correlated samples counted as independent, bad prior, miscalibration | Mark confidence unreliable until calibrated. |
| Relationship is learned but not reused | Retrieval query, index version, filters, plan trace | Retrieval mismatch, low confidence, stale relation, planner ignores result | Separate retrieval failure from planner failure. |
| Agent rediscovers known relation | Memory snapshot, cross-episode retrieval, visual matching | Checkpoint omission, mapping changed, perception drift, retrieval threshold too strict | Verify memory actually loaded and query matched before calling forgetting. |
| Relation used in wrong context | Preconditions, match scores, context uncertainty | Overgeneralized relation, tracker association, insufficient precondition specificity | Check false-positive use and add discriminating evidence. |
| Plan is invalid | Dependency graph and relation evidence | Unsupported edge, resource estimate uncertainty, precondition mismatch | A plan must cite every edge; unsupported edge is a planning defect. |
| JEPA loss falls but game behaviour does not improve | Held-out prediction, action conditioning, task coverage | Predictive objective ignores task-relevant variables, policy/planner issue, poor exploration | Do not assume a better latent loss means better control. |
| JEPA held-out score worse than train | Split manifests, duplicate hashes, seed overlap, sample distribution | Overfit, distribution shift, corrupted split | Verify split integrity before attributing to overfitting. |
| Prediction good one step, poor at 8 | Horizon errors, rollout drift, context update, action alignment | Error accumulation, compounding latent drift, inadequate training horizon | Limit MPC horizon until free-running results justify it. |
| Planner imagined outcome is good but real outcome bad | Per-candidate trace, uncertainty, actual transition, model version | Model exploitation, action mismatch, perception shift, predictor calibration | Find first imagined/real mismatch; shorten/disable planning if persistent. |
| Reward head inaccurate | Scalar reward trace, head calibration, class/target alignment | Reward scaling, sparse target, data imbalance, batch misalignment | Do not use reward head for planner scoring until calibration gate passes. |
| Termination head misses death | Termination labels, episode boundary semantics, calibration | End-flag normalization, rare positive examples, target mismatch | Block risk-sensitive MPC if confidently wrong. |
| Training loss spikes | Batch IDs, gradients, LR, scaler, replay samples | Bad batch, numerical overflow, corrupted observation, LR instability | Re-run exact batch/checkpoint offline before changing hyperparameters. |
| One ensemble member collapses | Per-member gradients/loss/parameters | Bootstrap mask bug, optimizer grouping, dead member | Keep member-level evidence; ensemble average alone is insufficient. |
| EMA drifts or stops updating | Online/target norms, EMA event counters | update not called, momentum schedule bug, parameter/buffer omission | Verify exact EMA equation and call count. |
| Resume changes actions | RNG, mapping, model/EMA, memory and policy state hashes | incomplete checkpoint, seed reset, worker restart | Do not call same run resumed until exact differences are explained. |
| Results vary drastically across seeds | Per-seed traces, maps, episode counts, action distribution | environment stochasticity, unstable learning, too few seeds | Report variance; do not cherry-pick. |
| Debugging itself changes results | paired debug on/off trace | RNG consumption, extra model calls, changed batching/order | Instrumentation is invalid until action/model equivalence test passes. |
| Missing trace events | event counters, disk/queue health, writer exceptions | process crash, buffer flush failure, storage full | Mark run observability incomplete; no forensic claim beyond retained evidence. |

## 51. Mandatory observability and debugging acceptance suite

Add these tests in addition to all tests in the original plan and Revision 1.2. The exact filenames may differ only if every test is mapped in a test manifest and executed in CI/Colab preflight.

### 51.1 Event and trace tests

- `test_event_schema_validation`
- `test_event_ids_unique_across_workers`
- `test_event_ordering_and_parent_links`
- `test_every_decision_has_outcome`
- `test_every_memory_mutation_has_evidence_reference`
- `test_trace_writer_flushes_on_checkpoint`
- `test_trace_writer_recovers_after_interruption`
- `test_trace_integrity_checksums`
- `test_trace_capture_degradation_is_visible`
- `test_debug_mode_does_not_change_rng_or_policy`

### 51.2 Action-selection tests

- `test_all_17_action_candidate_scores_recorded`
- `test_action_score_components_reconstruct_final_score`
- `test_normalizer_snapshot_fixed_within_decision`
- `test_candidate_iteration_order_does_not_change_ranking`
- `test_random_floor_draw_is_logged_and_replayable`
- `test_selected_token_matches_adapter_execution_token`
- `test_action_outcome_links_to_correct_decision`
- `test_action_distribution_metrics_are_correct`

### 51.3 Perception tests

- `test_visual_snapshot_trace_complete`
- `test_slot_association_emits_match_confidence`
- `test_uncertain_visual_state_does_not_become_absent`
- `test_quantity_estimate_retains_uncertainty`
- `test_camera_shift_does_not_force_new_slot_identity`
- `test_day_night_tint_association_stability`
- `test_occlusion_is_not_mislabelled_as_destruction`
- `test_tracker_debug_artifact_references_observation`

### 51.4 JEPA and optimizer tests

- `test_prediction_trace_includes_model_version_and_action_sequence`
- `test_training_update_trace_has_batch_and_sample_ids`
- `test_per_horizon_and_member_losses_recorded`
- `test_gradient_anomaly_creates_reproduction_bundle`
- `test_optimizer_update_counter_matches_successful_steps`
- `test_ema_update_counter_matches_optimizer_steps`
- `test_representation_health_snapshot_versioned`
- `test_heldout_metrics_use_independent_episode_store`

### 51.5 Relationship and planner tests

- `test_relation_candidate_records_origin_transition`
- `test_evidence_ledger_is_append_only`
- `test_duplicate_evidence_is_not_counted_as_independent`
- `test_contradiction_never_silently_deletes_prior_evidence`
- `test_relation_confidence_components_reconstruct_score`
- `test_memory_retrieval_trace_lists_rejected_candidates`
- `test_plan_edges_reference_supported_relations`
- `test_plan_execution_links_actions_and_outcomes`
- `test_stale_relation_is_not_used_without_warning`
- `test_frozen_transfer_forbids_memory_and_model_updates`

### 51.6 Replay, checkpoint, runtime and fault-injection tests

- `test_replay_sample_provenance_complete`
- `test_replay_generation_prevents_stale_ring_indices`
- `test_data_split_manifest_disjointness`
- `test_checkpoint_contains_all_stateful_components`
- `test_uninterrupted_vs_resumed_equivalence`
- `test_event_cursor_checkpoint_consistency`
- `test_first_divergence_report_identifies_injected_fault`
- `test_corrupt_checkpoint_preserves_previous_latest`
- `test_disk_full_marks_trace_degraded`
- `test_worker_crash_leaves_unmatched_decision_diagnosable`
- `test_resource_metrics_do_not_change_agent_policy`

For every test, store: test name, code revision, config hash, result, duration, stdout/stderr, failure artifact references and test-run ID. A test file that exists but has not been executed is `not_run`, not `PASS`.

## 52. Preflight and stage gates — exact blocking behaviour

Before any training run, execute a preflight that verifies:

1. Git commit and config snapshot are captured.
2. All configuration keys are parsed; unknown keys fail unless explicitly allowed.
3. Required packages and environment interface are available.
4. Observation/action/reward contracts pass.
5. Opaque mapping is generated once and round-trips.
6. Evaluator information is not available to learner/replay/planner objects.
7. Trace writer can create, flush, reopen and checksum a test event.
8. Artifact directory has sufficient free space for the selected debug level.
9. Checkpoint save/load round-trip works.
10. A synthetic decision event plus outcome can be recorded and reconstructed.
11. A small replay sequence passes alignment and no-boundary checks.
12. The independent held-out store can be created and is not referenced by the training sampler.

Blocking rules:

- Contract, data leakage, action mapping, replay integrity, checkpoint consistency or trace corruption failures block all training beyond the smoke test.
- JEPA prediction/action-conditioning/collapse failures block MPC but do not prevent a labelled world-model diagnostic run.
- Perception/state-tracker failures block relationship candidate confirmation.
- Relationship/evidence/memory failures block relation-aware planning and frozen-transfer claims.
- Planner real-world utility failures block default MPC use; continue with the best validated non-MPC policy.
- Incomplete trace capture blocks root-cause conclusions for the missing interval and blocks any claim that observability passed.
- A failed gate must never be bypassed by changing its threshold in the same run. Create a new versioned experiment/config and explain why.

Every gate returns a structured object:

```json
{
  "gate_id": "ACTION_CONDITIONING_PASS",
  "status": "PASS | FAIL | NOT_RUN | NOT_VERIFIED | BLOCKED | DEGRADED",
  "test_ids": [],
  "metrics_ref": "...",
  "data_split_hash": "...",
  "checkpoint_id": "...",
  "threshold_version": "...",
  "failure_event_ids": [],
  "notes": ""
}
```

## 53. Revised implementation order with observability first

This section is the execution authority if any earlier run schedule appears to imply that a long uninstrumented run should happen first.

### Stage 0 — Repository and runtime preflight

- Verify the exact commit and working tree.
- Run the existing test suite; do not assume README's expected result is actual evidence.
- Confirm config parsing and report unused keys.
- Confirm environment and learner input boundary.
- Produce run manifest and artifact directory.

**Exit:** all integrity tests pass or failures are documented and block the relevant run.

### Stage 1 — Minimal observability foundation

Implement event envelope/schema, IDs, run manifest, decision/outcome tracing, trace writer/flush/checksum, exception events, standard debug report skeleton and one synthetic end-to-end test.

**Exit:** a synthetic episode with at least ten transitions can be reconstructed from the artifacts; every decision has an outcome; debug on/off does not alter actions under deterministic test conditions.

### Stage 2 — Correct current AEON baseline integration

Fix exact action mapping in validation; create independent held-out episode collection/storage; wire novelty insertion to actual observations exactly once; wire learning-progress updates to actual measured prediction errors; freeze normalizers while scoring candidate sets; fix resume/orchestration; honour `initial_random_steps`, worker count and other config keys; instrument all current components.

**Exit:** all tests for mapping, held-out split, novelty/progress update, candidate-order invariance and complete resume pass.

### Stage 3 — Smoke and baseline pilot

- Run 500 steps with full debug capture.
- Run 5k–10k steps with one environment worker and standard/full capture as storage permits.
- Verify action traces, replay provenance, JEPA update traces, held-out isolation, checkpoint reload, runtime metrics and root-cause tools.
- Do not enable MPC unless its gates pass on genuine held-out data.

**Exit:** complete pilot report, no unexplained trace gaps, no integrity failure, and all pass/fail values backed by data.

### Stage 4 — Corrected JEPA baseline

Run the planned 50k development pilot, with independent held-out seeds, one role, controlled action mapping, declared debug policy and all existing JEPA gates. Save the best checkpoint by a predeclared validation criterion, not by final evaluation results.

**Exit:** reproducible baseline report. A failed JEPA gate is an informative result; it blocks dependent modules but does not get hidden.

### Stage 5 — Visual state tracking

Build anonymous slot/state tracking, state deltas, quantity uncertainty, camera/light/occlusion diagnostics and perception forensics. Keep human labels evaluator-only.

**Exit:** perception tests and held-out visual tracking diagnostics pass at declared error/coverage levels. Do not proceed to confirmed relation memory if the tracker cannot provide sufficiently stable evidence.

### Stage 6 — Relationship candidate and evidence system

Generate hypotheses from observed state/action/state changes. Add evidence ledger, controls, contradictions, confidence components, hypothesis lifecycle and relation diagnostics. The system must distinguish support from causal confirmation.

**Exit:** synthetic known-transition tests, fault injections, false-positive/negative evaluation and evidence replay pass.

### Stage 7 — Persistent relationship memory and retrieval

Add versioned snapshots, checkpoint persistence, indexing, retrieval explanations, staleness handling, contradiction processing and reload tests.

**Exit:** relation memory round-trip, retrieval tests and checkpoint/resume test pass. A learned relation is demonstrably retrieved in a later episode.

### Stage 8 — Compositional planning and controlled experiments

Create subgoals from learned relation dependencies; ensure every dependency edge references supported evidence. Execute one step at a time and update plans from actual observations. Integrate with CEM only after low-level world-model validation.

**Exit:** micro-world composition and real Crafter tests show reliable plan creation, execution and recovery from failed preconditions.

### Stage 9 — Frozen transfer and matched ablations

Freeze specified components and test on unseen worlds. Compare relation memory enabled/disabled with matched discovery budgets. Include no-relation, shuffled-relation, no-retrieval and reset-memory controls.

**Exit:** predeclared transfer metrics across multiple seeds; report whether memory provides measurable benefit.

### Stage 10 — Scale-up

Only after all prior gates pass: increase environment steps/workers, test multiple exploration roles, longer planning horizons and 1M-step benchmark-style runs. Each expansion is a separate experiment, not an invisible config change.

## 54. Experiment protocol for debugging and component attribution

Every experiment must test a named hypothesis. Use one principal change per experiment whenever feasible.

### 54.1 Required experiment record

```text
experiment_id:
question/hypothesis:
component_changed:
component_held_constant:
baseline_run_ids:
training_seed_list:
validation_seed_list:
calibration_seed_list:
final_evaluation_seed_list:
environment-step budget:
optimizer-update budget/actual:
wall-clock budget/actual:
debug level and storage policy:
predeclared primary metric:
secondary metrics:
acceptance threshold and source:
known confounders:
result:
interpretation:
next experiment:
```

### 54.2 Ablations required to identify relation-memory value

Compare, at matched data budgets:

1. Corrected JEPA baseline without relationship system.
2. JEPA + visual state tracker only.
3. JEPA + relation candidate generator, without persistence/retrieval.
4. JEPA + evidence ledger and confirmation.
5. JEPA + persistent relation memory, retrieval disabled.
6. JEPA + relation memory and retrieval, composition disabled.
7. Full relation-aware composition.
8. Full system with relation memory randomly shuffled or mismatched as a negative control, used only in an isolated test.

These isolate perception, discovery, persistence, retrieval and composition. Do not jump directly from baseline to the fully integrated system and then attribute any gain to “memory.”

### 54.3 Debugging experiments must retain failed cases

Store successful, typical, low-performing and failed episodes. Select episodes using predeclared sampling rules (for example, random sample plus worst-performing sample), not only attractive videos. Keep the selection algorithm and IDs in the report.

## 55. Expanded final completion and reporting contract

Do not produce one overall “AI understood Crafter” label. Produce independent statuses:

```text
ENVIRONMENT_INTEGRITY_PASS
ACTION_MAPPING_INTEGRITY_PASS
DATASET_SPLIT_INTEGRITY_PASS
TRACE_INTEGRITY_PASS
CHECKPOINT_RESUME_PASS
DETERMINISTIC_REPLAY_LEVEL
JEPA_PREDICTION_PASS
ACTION_CONDITIONING_PASS
REPRESENTATION_HEALTH_PASS
TRUE_HELDOUT_GENERALIZATION_PASS
EXPLORATION_SIGNAL_PASS
VISUAL_STATE_TRACKING_PASS
RELATION_CANDIDATE_QUALITY_PASS
RELATION_EVIDENCE_QUALITY_PASS
RELATION_MEMORY_PERSISTENCE_PASS
RELATION_RETRIEVAL_PASS
CROSS_EPISODE_TRANSFER_PASS
FROZEN_CROSS_WORLD_TRANSFER_PASS
COMPOSITION_PASS
PLANNING_REAL_WORLD_UTILITY_PASS
CALIBRATION_PASS
DEBUG_TRACE_PASS
ROOT_CAUSE_TOOLING_PASS
CRAFTER_BENCHMARK_PASS
```

Every status must allow `PASS`, `FAIL`, `NOT_RUN`, `NOT_VERIFIED`, `BLOCKED`, and `DEGRADED` where appropriate. List the test IDs, checkpoint IDs, metric files and data split hash that support it.

The final report must include:

- What was implemented versus the plan.
- Exact code revision and config.
- What was run and what was not run.
- All test statuses and logs.
- All training/evaluation seeds and interaction budgets.
- JEPA prediction/conditioning/representation results.
- Action-selection rationale examples and action distributions.
- Perception/state-tracking quality and uncertainty.
- Relation candidate quality, evidence, contradiction and confidence calibration.
- Memory persistence/retrieval and transfer results.
- Compositional plan traces and subgoal outcomes.
- MPC predicted-versus-realized performance.
- Crafter score and all 22 achievement rates, with the evaluation protocol clearly labelled.
- Ablations and negative controls.
- Runtime/resource/throughput accounting.
- Checkpoint/resume and replay-determinism levels.
- Trace completeness, missing artifacts and storage degradation.
- Root-cause reports for material failures.
- Limitations and open questions.

## 56. Exact implementation-agent handoff rules

The next agent implementing this plan must:

1. Read the entire plan, including the original JEPA specification and Revisions 1.2 and 1.3, before changing code.
2. Inspect the current repository commit and working tree; the audited SHA in Revision 1.2 may no longer be the head.
3. Produce a file-by-file implementation matrix with `implemented`, `partial`, `missing`, `not verified`, and links/line references before coding.
4. Reconcile any code changes made after the audited commit. Never assume Revision 1.2 source findings remain current without rechecking them.
5. Preserve existing AEON modules and previous experiment artifacts unless a documented test shows they are invalid or a replacement is justified.
6. Do not overwrite the original plan, old checkpoints, old results or old experiment reports. Create a new config/run/version.
7. Implement the minimal observability foundation before any long training run.
8. Add tests with each module change. Do not add a large subsystem without proving its component contracts.
9. Keep all Crafter mechanics labels and privileged evaluator information outside learner-facing modules.
10. Never create synthetic training success evidence or report test outcomes that were not actually executed.
11. Never change acceptance thresholds after observing a result without creating a new versioned experiment and documenting the reason.
12. If an API, library feature, environment state-cloning capability, deterministic behavior or metric definition is unknown, inspect the installed version/source and write a compatibility test. If still unknown, report `unknown` and block only the dependent claim; do not guess.
13. At the end of each implementation stage, provide changed files, tests executed, test results, known limitations, checkpoint/config IDs and the exact next stage.
14. Treat this plan as an engineering specification, not a guarantee that the architecture will achieve a particular Crafter score or human-like reasoning.

## 57. Final pre-run debugging checklist

Before a diagnostic pilot:

- [ ] Run manifest and config hash created.
- [ ] Code revision and dependency versions captured.
- [ ] Train/validation/calibration/final splits are defined and disjoint.
- [ ] Action mapping generated once and mapping hash recorded.
- [ ] Every decision/outcome trace schema validated.
- [ ] Trace writer flush/checksum/reopen test passed.
- [ ] Observation artifacts indexed and checksummed.
- [ ] Replay provenance and sample alignment tests passed.
- [ ] Training update trace includes batch, loss, gradient and EMA information.
- [ ] Normalizers do not mutate during candidate ranking.
- [ ] Novelty memory and learning-progress update timing specified and tested.
- [ ] Checkpoint includes all stateful components and passes resume test.
- [ ] Error/exception capture and failure taxonomy implemented.
- [ ] Debug mode does not change policy output under deterministic conditions.
- [ ] Storage budget and degradation policy set.
- [ ] Predeclared primary metrics and gate thresholds saved.
- [ ] No unsupported claim is made from unexecuted tests or empty artifacts.

## 58. Final forensic-analysis checklist

Before concluding what the agent learned or why it failed:

- [ ] Verify run/checkpoint/config/code identities.
- [ ] Verify trace completeness and artifact hashes.
- [ ] Verify actual action token and adapter mapping.
- [ ] Reconstruct selected decisions and score components.
- [ ] Check real observed transitions and replay provenance.
- [ ] Check visual tracker association/confidence and state deltas.
- [ ] Compare JEPA predictions with actual next observations at multiple horizons.
- [ ] Check optimizer/EMA and representation health around the failure.
- [ ] Inspect relation candidate and append-only evidence history.
- [ ] Inspect relation retrieval decisions and rejected candidates.
- [ ] Inspect plan dependencies, subgoals and low-level action execution.
- [ ] Compare to baseline/ablation using matched budgets and valid splits.
- [ ] Run a reproduction or transition replay where possible.
- [ ] List alternative explanations and tests used to eliminate them.
- [ ] Mark unresolved cause as `UNKNOWN` when evidence is insufficient.
- [ ] Add a regression test for every confirmed defect.

**Revision 1.3 definition of done for debugging:** A person who did not write AEON must be able to select a run, locate an episode, reconstruct why each action was selected, inspect the evidence behind each memory mutation and retrieval, compare predicted and actual outcomes, identify the earliest failing component with trace references, and reproduce the diagnosis or clearly state why exact environment replay is unsupported. The system must never manufacture certainty where the logs do not support it.

---

# Revision 1.3 change summary

Revision 1.3 preserves the original model objective, Crafter integrity boundary, JEPA design, and Revision 1.2 relationship-learning architecture. It adds a much more detailed forensic-debugging specification: versioned event envelopes and IDs, mandatory event types, action candidate/decision/outcome traces, visual-slot and state-delta traces, optimizer/gradient/EMA diagnostics, replay and dataset provenance, append-only relationship evidence, retrieval and plan traces, complete checkpoint/resume forensics, distinct replay modes, root-cause taxonomy and workflow, controlled failure injection, uncertainty calibration, evaluation split discipline, runtime/storage monitoring, human-facing inspection commands and reports, explicit stage gates, ablations, and agent handoff rules.

**No source code implementation, training run, test execution, or GitHub commit is claimed by this document revision.**

---

## 59. Required observability configuration contract

This section resolves the older Revision 1.2 example that used `debug.level: summary`. For Revision 1.3, the only supported debug levels are `off`, `standard`, and `full`. `summary` is deprecated and must cause a config validation error rather than silently choosing a mode.

The following is the required shape of the configuration. The example values are conservative initial engineering defaults for development, not empirically optimized values. Every key must be parsed, validated, printed in the run manifest and covered by at least one config test.

```yaml
debug:
  enabled: true
  level: standard                 # off | standard | full
  event_schema_version: "1.0"
  emit_decision_events: true
  emit_outcome_events: true
  emit_training_update_events: true
  emit_relation_memory_events: true
  emit_runtime_health_events: true
  runtime_health_interval_steps: 1000
  action_distribution_interval_steps: 1000
  representation_diagnostics_interval_updates: 500
  full_candidate_scores: true
  full_cem_candidate_sequences: false
  save_observation_artifacts: selected  # none | selected | every_step
  save_latent_artifacts: selected        # none | selected | every_step
  save_activation_artifacts: on_anomaly  # none | selected | on_anomaly
  save_gradient_artifacts: on_anomaly    # none | selected | on_anomaly
  save_full_replay_sample_indices: true
  save_memory_snapshots: true
  save_episode_dossiers: true
  deterministic_transition_replay: true
  attempt_environment_reexecution: true
  verify_artifact_checksums: true
  flush_events_every_n_events: 100
  flush_events_on_episode_end: true
  flush_events_on_checkpoint: true
  max_run_artifact_gb: 10
  max_observation_artifacts_gb: 4
  max_latent_artifacts_gb: 2
  max_video_artifacts_gb: 2
  low_disk_warning_gb: 2
  on_storage_limit: stop_and_mark_degraded  # stop_and_mark_degraded | configured_degrade
  configured_degrade_level: standard
  allow_policy_rng_consumption: false
  require_debug_on_off_equivalence_test: true

observability:
  hash_algorithm: sha256
  preserve_raw_observation_bytes: true
  event_log_format: jsonl
  metrics_log_format: jsonl
  artifact_index_format: jsonl
  event_writer_queue_size: 10000
  fail_on_event_schema_error: true
  fail_on_missing_decision_outcome_pair: true
  fail_on_unresolved_artifact_reference: true
  include_latent_tensor_dtype_and_shape: true
  include_model_and_optimizer_version_ids: true
  include_config_hash: true
  include_code_revision: true

failure_handling:
  capture_python_tracebacks: true
  capture_component_boundary_inputs: true
  capture_component_boundary_outputs: true
  save_reproduction_bundle_on_anomaly: true
  stop_on_data_integrity_failure: true
  stop_on_info_leak: true
  stop_on_checkpoint_corruption: true
  allow_unknown_root_cause: true
```

Configuration validation rules:

- Reject unrecognized enum values, negative intervals, non-positive queue sizes, invalid storage caps, and incompatible combinations.
- If `debug.enabled` is false, event capture may be reduced only as defined by `off`; essential run identity, checkpoint, error, gate and integrity records remain mandatory.
- If `level=full`, `save_observation_artifacts=every_step` is recommended for short forensic pilots, but the run must pass a disk-budget preflight first.
- If `on_storage_limit=configured_degrade`, the exact downgrade must be declared before the run and logged at the transition where it occurs. A degraded run cannot receive `DEBUG_TRACE_PASS` for the interval that lost required events.
- If `allow_policy_rng_consumption=false`, diagnostic code must use independent RNG streams or deterministic transforms. The equivalence test must verify this.
- `attempt_environment_reexecution=true` is a request to test whether it is supported; it is not a claim that Crafter re-execution is deterministic.
- The values shown for artifact GB caps may need adjustment to the actual mounted disk/Drive quota. Preflight must inspect available space rather than assuming these caps fit.

## 60. Canonical event payload examples

These examples define the minimum structure and meaning. Implementers may add fields, but may not remove required fields or change units without a schema-version increment and migration.

### 60.1 Action decision payload

```json
{
  "decision_status": "completed",
  "observation_ref": "artifact-id",
  "observation_sha256": "sha256:...",
  "history_transition_ids": ["transition-id-1", "transition-id-2"],
  "state_snapshot_id": "state-snapshot-id-or-null",
  "latent_ref": "latent-artifact-id-or-null",
  "context_ref": "context-artifact-id-or-null",
  "policy_mode": "curiosity",
  "role_id": "explorer",
  "model_checkpoint_id": "checkpoint-id",
  "optimizer_update_id": 1234,
  "normalizer_snapshot_id": "normalizer-snapshot-id",
  "random_floor": {
    "configured_probability": 0.1,
    "rng_stream": "policy",
    "draw": 0.73,
    "triggered": false
  },
  "temperature": 1.0,
  "candidates": [
    {
      "action_token": 0,
      "status": "scored",
      "novelty_raw": 0.0,
      "novelty_normalized": 0.0,
      "disagreement_raw": 0.0,
      "disagreement_normalized": 0.0,
      "progress_raw": 0.0,
      "progress_normalized": 0.0,
      "weighted_components": {},
      "final_score": 0.0,
      "selection_probability": 0.0,
      "prediction_id": "prediction-id-or-null"
    }
  ],
  "selected_action_token": 0,
  "selection_reason": "curiosity_sample",
  "selected_candidate_id": null,
  "policy_rng_state_hash_before": "sha256:...",
  "policy_rng_state_hash_after": "sha256:...",
  "latency_ms": {
    "perception": 0.0,
    "context": 0.0,
    "candidate_prediction": 0.0,
    "scoring": 0.0,
    "sampling": 0.0,
    "total": 0.0
  }
}
```

The numeric values in this example are placeholders illustrating types, not target measurements. In actual events, populate all 17 candidate entries if the policy evaluates all 17 tokens. For warm-up/random-floor decisions, record the branch and random draw; do not claim that unscored candidates had score zero.

### 60.2 Action outcome payload

```json
{
  "outcome_status": "observed",
  "executed_action_token": 0,
  "adapter_mapping_hash": "sha256:...",
  "next_observation_ref": "artifact-id",
  "next_observation_sha256": "sha256:...",
  "reward": 0.0,
  "terminated": false,
  "truncated": false,
  "transition_id": "transition-id",
  "replay_inserted": true,
  "replay_generation": 4,
  "predicted_next_latent_ref": "latent-ref-or-null",
  "actual_target_latent_ref": "latent-ref-or-null",
  "latent_error_metric": "cosine_distance_v1_or_not_applicable",
  "latent_error": null,
  "predicted_reward": null,
  "reward_error": null,
  "predicted_termination_probability": null,
  "termination_brier_component": null,
  "visual_state_delta_id": "delta-id-or-null",
  "environment_latency_ms": 0.0
}
```

### 60.3 Relationship evidence payload

```json
{
  "relation_id": "relation-id",
  "relation_revision_id": "relation-revision-id",
  "evidence_id": "evidence-id",
  "source_transition_id": "transition-id",
  "source_decision_id": "decision-id",
  "before_state_snapshot_id": "state-id",
  "after_state_snapshot_id": "state-id",
  "action_token": 0,
  "precondition_feature_refs": ["anonymous-feature-ref"],
  "observed_effect_feature_refs": ["anonymous-feature-ref"],
  "delta_artifact_ref": "delta-artifact-id",
  "tracker_confidence": null,
  "confounders": [],
  "evidence_classification": "supporting | contradicting | neutral | inconclusive",
  "independent_episode": true,
  "independent_world": true,
  "confidence_components": {},
  "confidence_method_version": "method-version",
  "new_status": "candidate",
  "reason_code": "observed_transition_requires_testing"
}
```

The `independent_episode` and `independent_world` flags must be computed from event metadata and split manifests, not manually set based on a desired confidence result. When independence cannot be established, use `unknown`.

### 60.4 Training update payload

Each `optimizer_step_completed` event must include batch and sample IDs, per-horizon/per-member loss, auxiliary losses, LR per parameter group, gradient scaler status, gradient norms, clipped norm, parameter update norms, EMA momentum/update result, update duration and model version after the update. If an optimizer step was skipped due to overflow, record `optimizer_step_status=skipped_overflow`; do not increment the successful-update counter.

### 60.5 Gate result payload

Each gate result must include the exact dataset split hash, seed list hash, checkpoint ID, code/config hashes, threshold version, test IDs, sample count, metric artifact reference, status and failure event IDs. A gate without data or test references cannot be `PASS`.

## 61. Do not fabricate explanations for neural decisions

A debugging system must explain the *observable computation path*, not invent a natural-language rationale for hidden neural computations.

- For a curiosity policy, the explanation can say which measured score components and configured weights led to a token's probability.
- For CEM, it can say which candidate sequence received the highest logged objective and list that objective's measured components.
- For relation-aware planning, it can list retrieved relation revisions, evidence counts, precondition matches and dependency edges.
- For a neural representation, it may report feature/attention/slot diagnostics, but it must not claim that a specific latent dimension “means wood” without an independent, separately reported diagnostic.
- If an action is produced by a model whose internal computation is not causally interpretable, record `selection_reason=neural_policy_output` and save the relevant logits/scores/inputs. Do not generate a plausible story and present it as the model's actual reasoning.
- If an agent-facing language explanation is added later, label it `post_hoc_explanation`, link it to the trace, and never treat that text as ground truth.

## 62. Agent handoff deliverables at every milestone

At the end of each stage, the implementing agent must return a milestone packet containing:

1. **Repository delta:** commit SHA, changed files, added/deleted modules and a concise diff summary.
2. **Implementation matrix:** each relevant plan item mapped to a file/function/test and status.
3. **Commands executed:** exact shell commands or notebook cells, runtime and config.
4. **Test evidence:** test names, pass/fail/not-run status, durations and artifact paths.
5. **Run identity:** run ID, seed IDs, checkpoint IDs, environment steps and optimizer updates.
6. **Trace audit:** missing event counts, unmatched decisions/outcomes, unresolved artifact references and checksum status.
7. **Model results:** training and held-out metrics with data split hashes and baselines.
8. **Debug examples:** at least one reconstructed decision and one transition prediction-vs-actual analysis when relevant.
9. **Failure analysis:** root-cause reports or explicit `no material failure observed`; do not say “no problems” if traces are incomplete.
10. **Open issues:** unresolved questions with the evidence needed to answer them.
11. **Next stage recommendation:** the single next stage supported by the gate results.

A milestone is not complete when code is pushed. It is complete when the implementation, tests, run artifacts and report meet that stage's acceptance criteria.
