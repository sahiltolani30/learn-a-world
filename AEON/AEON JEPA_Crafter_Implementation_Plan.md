# JEPA Crafter Agent — Complete Implementation Plan

**Document type:** Engineering implementation specification  
**Status:** Build plan — implementation has not been performed  
**Primary environment:** Crafter (`CrafterReward-v1` and `CrafterNoReward-v1`)  
**Target hardware:** Free Google Colab, one NVIDIA T4 where available  
**Primary objective:** Build and validate a from-scratch, action-conditioned JEPA world model that learns from raw visual interaction, retains experience across episodes, and uses its learned model to improve its behaviour.

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
