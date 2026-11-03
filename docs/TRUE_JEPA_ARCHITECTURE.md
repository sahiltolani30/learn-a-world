# True JEPA Self-Play Architecture

## Decision

The new implementation lives in `true_jepa/` and is independent of the
previous model, planner, replay, and training packages.  It is a genuine
action-conditioned JEPA world model:

```text
pixels + previous reward
          |
          v
  online visual encoder E_theta -----> recurrent belief b_t
          |                                  |
          |                                  v
  EMA target encoder E_xi             action-conditioned predictor P_phi
          |                                  |
          +---- target latent y_{t+1} <-----+---- predicted latent z_hat_{t+1}
                                                        |
                                                        v
                                      closed-loop imagination and policy
```

The action path is strictly:

```python
action = agent.act(obs_uint8, reward_prev, step)
```

No inventory dictionary, achievement array, coordinate, semantic map,
environment `info`, or recipe knowledge crosses that boundary.

## What makes this a JEPA

For an observation sequence `o_0:T` and actions `a_0:T-1`:

```text
x_t       = E_theta(o_t)
y_{t+1}   = stop_gradient(E_xi(o_{t+1}))
b_t       = B(x_t, a_{t-1}, reward_{t-1}, b_{t-1})
x_hat_t+1 = P_phi(x_t, a_t, b_t)
```

The primary loss is embedding-space prediction:

```text
L_JEPA = ||x_hat_t+1.tokens - y_t+1.tokens||²
       + ||mean(x_hat_t+1.tokens) - mean(y_t+1.tokens)||²
```

There is no pixel decoder and no pixel reconstruction loss.  The target
encoder is a permanently frozen, evaluation-mode copy updated by EMA only
after an optimizer step.  Variance and covariance regularization are collapse
guards, not substitutes for the JEPA objective.

The predictor is action-conditioned and closed-loop during training.  After
the first prediction, predicted tokens—not teacher-forced target tokens—are
fed into the next belief and prediction step.  This is necessary for a model
that can be used for imagined action sequences rather than only one-step
regression.

## How relations are learned

`true_jepa/relations.py` stores anonymous evidence:

```text
(context_latent, action_id, next_latent - context_latent,
 scalar_reward, episode_id, step)
```

The bank clusters effects by learned action/effect similarity, keeps separate
success and failure context statistics, and exposes a hypothesis only after
support from multiple transitions and independent episodes.  It then tests
whether an effect direction from one anonymous relation aligns with the
learned success-vs-failure direction of another relation.

This is the mechanism by which a dependency can emerge:

1. The visual encoder learns stable factors because they predict future
   observations and reward-relevant consequences.
2. A transition changes those factors; the memory stores the observed latent
   delta and its magnitude.
3. Repeated successes and matched failures separate the context in which the
   transition is effective.
4. A later transition is predicted to be more successful when the earlier
   effect has moved the latent state toward its learned success context.
5. The edge becomes usable only after statistical support; a single success
   never creates a gate.

For the Crafter validation, the post-episode audit may align anonymous latent
   factors to the environment's named unlock log.  That alignment is for
   evaluation only.  The action policy must operate on the anonymous vectors.
   If a crafting quantity is visually represented, its learned effect scale
   is retained as a continuous value, so repeated evidence can distinguish
   different resource changes without a hand-written quantity table.

If an environment does not expose an unlock through pixels or reward, the
agent cannot honestly identify that unlock online.  The correct response is to
add a visual/reward learning signal or report the limitation—not to inject the
achievement label into control.

## Self-play loop

`true_jepa/self_play.py` alternates these operations:

1. Collect episodes through the pixel-only environment interface.
2. Select actions from the current policy, a frozen snapshot produced by the
   same run, or uniform exploration.
3. Store only pixels, actions, rewards, termination flags, and reset metadata.
4. Add latent transition evidence to the anonymous relation bank.
5. Train the EMA-targeted JEPA on boundary-safe sequence windows.
6. Train the policy and value heads in closed-loop latent imagination while
   keeping world-model dynamics detached from policy gradients.
7. Periodically snapshot the current model and repeat.

Historical snapshots are data diversity, not privileged opponents.  This is
self-play for a single-agent world: the policy generates its own curriculum,
and the world model learns from the resulting consequences.

## Google Colab free T4 profile

Start with the following budget in `TrainConfig` and `JEPAConfig`:

| Setting | T4 start point |
|---|---:|
| observation | 64x64 RGB uint8 |
| token grid | 8x8 patches |
| token dimension | 192 |
| belief dimension | 256 |
| predictor layers | 2 |
| replay resident frames | 120,000 |
| batch size | 8 |
| sequence length | 16 |
| imagination horizon | 8 |
| ensemble members | 3 |
| AMP | FP16 on CUDA |

Run data collection in short blocks and checkpoint the model, target encoder,
optimizer state, replay shards, relation state, RNG state, and configuration.
Keep replay shards under `/content` while training and copy completed shards to
Drive.  Never mix training and locked evaluation seeds.

## Promotion gates

Do not promote a checkpoint based only on a lower training loss.  Require:

### Integrity

- AST audit finds no semantic recipe/action mapping in `true_jepa/`.
- `act` receives exactly three arguments after `self`.
- A poisoned `info` object is never constructed or accessed by the policy.
- Training batches contain only observable stream fields.
- Model actions are invariant when hidden environment state is changed while
  pixels and reward are held fixed.

### Representation and dynamics

- Target encoder is restored separately and remains different from the online
  encoder after resumed training.
- Latent variance stays above the configured collapse floor.
- Free-running latent error beats persistence at horizons 1, 4, and 8.
- Shuffling actions increases held-out prediction error.
- Reward and continuation heads beat their constant baselines.

### Relation learning

- Dependencies require repeated successes across independent episodes.
- Removing the transitions that support an edge removes the edge.
- Shuffling outcomes destroys the edge.
- Permuting action IDs changes the learned relation IDs but not the behavior.
- The bank reports effect scale and uncertainty, never a human-authored item
  name or required-material count.

### Behavior

Report extrinsic reward, episode length, action entropy, frames/sec,
updates/sec, GPU memory, one-step and multi-step latent error, and locked
achievement score.  Named achievement counts are read only after an episode
for scoring and audit.

## Research inspiration

The design follows the key ideas in the public JEPA work rather than copying
an implementation:

- [I-JEPA](https://arxiv.org/abs/2301.08243): predict abstract target
  representations with an EMA target encoder instead of reconstructing pixels.
- [V-JEPA](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/):
  learn predictable visual structure in latent space.
- [V-JEPA 2](https://arxiv.org/abs/2506.09985): separate visual predictive
  pretraining from a later action-conditioned latent predictor used for
  planning.
- [H-JEPA notes](https://doi.org/10.48550/arxiv.2306.02572): use multiple
  temporal abstraction levels when one horizon cannot represent both local
  effects and long dependencies.

The repository implementation is deliberately smaller than those large-scale
systems so it can be measured and falsified on a free T4.

## Execution phases

### Phase A — representation health

Collect random and lightly exploratory pixel episodes. Train only the JEPA,
reward, continuation, and ensemble heads. Reject collapse and action-ignoring
predictors before policy learning.

### Phase B — action-conditioned consequences

Increase current-policy collection, retain a small historical snapshot stream,
and validate held-out counterfactual action ranking. Do not use the relation
bank to gate actions yet.

### Phase C — anonymous relation discovery

Enable relation hypotheses only after the evidence gates pass. Use them as
uncertain priors in latent planning, never as deterministic rules.

### Phase D — long-horizon self-play

Increase horizon and population diversity gradually. Promote only on locked
seeds with a frozen checkpoint and a complete provenance report.

## Colab entry point

From the repository root in a Colab cell:

```bash
python -m true_jepa.colab_run \
  --run-dir /content/true_jepa_run \
  --episodes 128 \
  --updates 256 \
  --device cuda
```

The run writes `state.pt`, `replay.npz`, and `config.json`.  Copy those files
to Drive after each short run block; resume only after restoring the same
configuration and replay seed split.
