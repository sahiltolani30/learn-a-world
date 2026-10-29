# Project ATLAS — Associative Topological LATent Skills

**Status:** COMPLETE IMPLEMENTATION SPEC. Any agent can implement from this document alone.
**Date:** 2026-10-29. Author: ZCode agent (new line; zero edits to any pre-existing file).

**Execution environments (operator policy):**
- **Local (Mac, MPS)**: smoke tests only — unit tests, 1–2 tiny episodes, short head-training dry runs. NEVER used for real collection/training scale.
- **Colab (T4/L4)**: all real runs — collection, training, screening. Google Drive is mounted at `/content/drive/MyDrive/learn-a-world/` (the repo lives at `/content/learn-a-world/` on Colab; artifacts sync to Drive).

---

## 1. Mission

A true-JEPA agent that **reuses the already-trained T2 world model frozen** and adds small
self-learned cognitive layers around it. No from-scratch retraining, no hardcoding, no cheating.
Every new mechanism learns from the agent's own play.

**Baselines (locked protocol: seeds 2050–2099, 50 eps, ≤1000 steps, `compute_crafter_score`):**
honest champion 4.31% (`results/screen_v3_ablation_c_active_50ep.json`), latest arms 3.87%/3.69%,
random floor 1.77%. ATLAS ladder: ≥4.5% (A3) → ≥6% (A4) → ≥8.6% (A5).

## 2. Reuse (read-only imports — the ONLY old modules ATLAS may touch)

| Asset | Import | Use |
|---|---|---|
| `dual_store.jepa_v3.JEPAv3` | read-only | model class; loads encoder (z 128), GRU core (h 256), latent predictor, event head, health head, Q ensemble (3×51 bins [0,12]), behavior-cloned policy prior, normalizer; auto-attaches calibrator |
| `envs.crafter_wrapper.CrafterEnv` | read-only | deterministic env, `snapshot_state()`/`restore_state()` |
| `eval.metrics` (`compute_crafter_score`, `CRAFTER_ACHIEVEMENTS`) | read-only | official scoring + the 22-name list for SCORING and nothing else |

`models/`, `planning/`, `training/`, `scripts/`, `dual_store/*` (except `jepa_v3.py`), `zephyr/`,
`godfather_jepa/` are **forbidden imports**. Enforced by an AST-isolation unit test.

**Checkpoint selection** (`atlas/config.py::_default_checkpoint`): env `ATLAS_CHECKPOINT` →
`checkpoints/jepa_crafter_v3_state_readout_v2.pt` (Drive champion `fc6cd2cb…`, if operator copies
it into repo `checkpoints/`) → `checkpoints/jepa_crafter_v3_phase2.pt` (local, step 5000).
SHA-256 of whatever loads goes into every result JSON and every checkpoint manifest.

**Why frozen:** Autopsy Attempt 7 — retraining the encoder drifted latents 7.05 and destroyed all
heads (4.55%→3.40%). All backbone forward passes are under `torch.no_grad()`; params
`requires_grad_(False)`.

## 3. Repository layout (everything new lives here)

```
atlas/
  __init__.py            # exports AtlasConfig, FrozenBackbone, AtlasAgent
  PLAN.md                # this document
  config.py              # ALL hyperparameters as dataclasses (single source of truth)
  backbone.py            # FrozenBackbone: load, freeze, provenance, inference facade
  belief.py              # AchievementBelief: strict m_bin/m_soft tracking
  placemap.py            # TopoMap: visual keyframe memory (cosine latent matching)
  skills.py              # OptionNet (goal-conditioned policy) + ReachabilityNet + HER loss
  epistemic.py           # EpistemicNet: K bootstrap residual predictors + kNN novelty
  safety.py              # ContinueNet: P(alive) head + risk queries
  readout.py             # VitalsReadout: 16-d self-state estimate (training targets only)
  goals.py               # GoalBank + predecessor-lift mining + goal scoring
  arbiter.py             # AtlasAgent: NAVIGATE / PROBE / FALLBACK controller + safety override
  collect.py             # EpisodeCollector (live) + ShardLoader (Drive/bootstrap npz shards)
  train.py               # HeadTrainer: trains all small heads from shards; latent caching
  selflearn.py           # SelfLearnLoop: bootstrap → [collect→train→mine]×N, resumable
  screen.py              # locked 50-ep strict screening + protocol JSON writer
  colab/
    setup_atlas.sh       # pip installs + Drive mount checks
    run_atlas.py         # Colab entrypoint: --phase {bootstrap,selflearn,screen} --iters N
  tests/
    test_atlas_isolation.py    # AST import audit + git cleanliness of old files
    test_atlas_backbone.py     # load, freeze, no-drift determinism, provenance SHA
    test_atlas_belief.py       # strict updates; refuses env info; fallback rule
    test_atlas_memory.py       # TopoMap add/match/persist; GoalBank mine/score/lift
    test_atlas_skills.py       # HER loss decreases; reachability separates pos/neg
    test_atlas_epistemic.py    # disagreement high for unseen pairs; trains down
    test_atlas_safety.py       # focal BCE learns rare deaths; risk query shapes
    test_atlas_arbiter.py      # mode transitions; safety override fires; strict inputs only
    test_atlas_train_collect.py# shard round-trip; cache correctness; trainer step decreases loss
    test_atlas_screen.py       # protocol JSON has required provenance fields; strict flags on
```

## 4. Canonical dimensions & conventions

| Symbol | Dim | Meaning |
|---|---|---|
| `z` | 128 | encoder latent (LayerNorm'd geometry from checkpoint) |
| `h` | 256 | GRU belief state |
| `s = [z; h]` | 384 | composite state; input to all heads |
| `a` | 17 | discrete action index (int64) |
| `m_bin / m_soft` | 22 | believed-unlocked mask (binary / running-max soft) |
| `x̂` | 16 | readout self-state: 12 item counts (log1p) + health,food,drink,energy each /16 |
| goal key `g` | 128 | a stored z (view) |

- **obs normalization**: uint8 HWC `[64,64,3]` → float `/255` → CHW `[B,3,64,64]` (backbone does this).
- **Devices**: `resolve_device()` = mps → cuda → cpu. All tensors created on backbone device.
- **Determinism**: seeds recorded per episode; env uses `CrafterEnv(seed=…)` with the repo's
  determinism patch.
- **Strictness invariant** (everywhere, tested): heads consume ONLY `(s or z, action, m)`. Engine
  inventory/achievements/coordinates appear exclusively (a) as training targets in shards,
  (b) inside `compute_crafter_score`.

## 5. Module specifications (interfaces + algorithms)

### 5.1 `backbone.py` — FrozenBackbone
```python
class FrozenBackbone:
    def __init__(self, config: AtlasConfig, device: str | None = None)
    provenance: dict  # checkpoint path, sha256, step, loaded_component_keys, frozen
    def to_tensor(obs: np.ndarray) -> torch.Tensor            # static, HWC→CHW /255
    @no_grad def encode(obs) -> z[B,128]                       # on self.device
    @no_grad def recurrent_step(z, a, h_prev) -> h[B,256]
    def composite(z, h) -> s[B,384]
    @no_grad def predict_next(z, h, a) -> (z', h', s')         # model.predict_next_step
    @no_grad def event_probs(s, a, m) -> p[B,22]               # calibrated predict_proba
    @no_grad def event_logits_all_actions(s, m) -> p[B,17,22]  # PROBE ranking (batched over a)
    @no_grad def health(s, a) -> r_vital[B]                    # health_head
    @no_grad def q_mean(s, a) -> q[B]
    @no_grad def prior_logits(s, m) -> logits[B,17]            # policy_prior
    def assert_frozen() -> None                                # raises if trainable/grad
```
`event_logits_all_actions` must call the event head **once per action** (17 forwards of batch B)
or as one flat batch of `B*17` — no loop over achievements, no per-name logic.

### 5.2 `belief.py` — AchievementBelief (fresh implementation; do NOT import planning/)
```python
class AchievementBelief:
    def __init__(self, unlock_threshold=0.25, reward_fallback=0.9)
    m_bin: np.ndarray[22]   # binary, persistent within episode
    m_soft: np.ndarray[22]  # running max of predicted p
    def reset()
    def update(event_probs: np.ndarray[22], reward: float) -> list[int]:
        # 1) m_soft = max(m_soft, p)
        # 2) for each dim with m_bin<0.5: if p > threshold -> m_bin=1, record newly_unlocked
        # 3) fallback: if reward >= 0.9 and no new unlock -> pin argmax p among locked dims
        #    (sets m_bin=m_soft=1 for that dim; records provenance last_fallback=[dim])
    def get_mask_tensor(device) / get_soft_mask_tensor(device)
```
The ONLY caller-supplied signals are the model's own event probs and the scalar reward.
`update` must raise if anything shaped like env info is passed (test: passing a 22-dim
`info["achievements"]` vector is forbidden by construction — the method has no parameter for it).

### 5.3 `placemap.py` — TopoMap
```python
class Keyframe:  z: np.ndarray[128];  context: np.ndarray[256];  step: int;  visits: int
class TopoMap:
    def __init__(cfg: TopoMapConfig)
    def reset_episode()                       # clears episode-local ring, keeps persistent
    def observe(z, context, step) -> Keyframe | None:
        # add keyframe if cos_dist(z, nearest) > scene_change_threshold AND step-last>=min_interval
        # also mirror the keyframe into the persistent store (capacity ring by oldest)
    def nearest(z, k=1) -> list[(Keyframe, cos_sim)]
    def match(z) -> (Keyframe, cos) | None    # persistent+episode union; None if best < match_threshold
    def save(path) / load(path)               # .npz: z[N,128], ctx[N,256], step[N], visits[N]
```
Pure cosine matching in latent space. No coordinates, no motion integration anywhere.

### 5.4 `skills.py` — OptionNet + ReachabilityNet
```python
class OptionNet(nn.Module):
    # goal encoder: Linear(128 -> 128) + LayerNorm + GELU
    # policy trunk: MLP(384 + 128 -> hidden 256 -> 256) -> logits[17]
    def act(s[B,384], g[B,128], greedy=True) -> a[B]
    def logits(s, g) -> [B,17]
class ReachabilityNet(nn.Module):
    # MLP(384 + 128 -> 256 -> 256 -> 1 sigmoid)
    def forward(s, g) -> psi[B]
def her_batch(batch, horizons=(5,10,20), negatives_per_pos=1):
    # batch: episode tensors (z[T,128], h[T,256], a[T], done[T])
    # for each start t and k in horizons with t+k < T_ep:
    #   g = z[t+k]; positives: (s_tau, g) -> label a_tau for tau in [t, t+k)
    #   reach label = 1.0
    # negatives: g' = z[u] sampled from a DIFFERENT episode; reach label = 0.0;
    #   (no action loss on negatives)
    # returns dict of tensors for joint loss:
    #   CE(policy_logits, taken_action)  +  BCE(reachability, label)   (CE weight 1.0, BCE 0.5)
```
Option execution (in arbiter): re-encode each step, arrival when `cos(z_t, g) >= arrival_cosine`
(0.93); abandon when `psi` has not increased for `stagnation_patience` (8) steps or
`max_option_steps` (25) reached.

### 5.5 `epistemic.py` — EpistemicNet
```python
class EpistemicNet(nn.Module):
    # K=4 heads: MLP(384+32(action emb) -> 128 -> 128 -> 128) each predicting a residual
    # dz_k(s,a); predicted next-latent_k = frozen_predictor(z,h,a) + dz_k
    def disagreement(z, h, a) -> d[B]           # std_k of predicted next latents, mean over dims
def knn_novelty(h, memory: deque[np.ndarray], k=4) -> float  # mean dist to k nearest in episode mem
```
Training: MSE per head against true `z_{t+1} - frozen_pred` on replay (each head trained on the
full batch; diversity comes from independent init + dropout 0.1 in trunk). Intrinsic reward at
play time: `r_int = disagreement + knn_beta * knn_novelty` (β scheduled by selflearn).
Rejection-sampled sanity: disagreement on (s, a) pairs sampled from training data must DROP
during training (test asserts decreasing loss and lower in-distribution disagreement than
random-pair disagreement).

### 5.6 `safety.py` — ContinueNet
```python
class ContinueNet(nn.Module):
    # MLP(384+32 -> 256 -> 256 -> 1)
    def alive_prob(s, a) -> [B]                  # sigmoid
    def risk(s, a) -> [B]                        # 1 - alive_prob
def focal_bce(logits, targets, weight=25.0, gamma=2.0)   # positive class weight w on done
```
Labels: real death only (`discount == 0` from env info; truncation at max_steps is NOT a death).
At play time the arbiter queries risk for the intended action and all 17 actions; override rule:
if `risk(s, a_intended) > risk_threshold (0.5)` and `min_a risk(s,a) < risk_intended − risk_margin (0.15)`
→ take `argmin_a risk` (ties → higher q_mean). Class weight 25 because deaths are ~2% of steps.

### 5.7 `readout.py` — VitalsReadout
```python
class VitalsReadout(nn.Module):
    # trunk MLP(384->256->256) -> mean[16] and logvar[16] (softplus+1e-4)
    def forward(s) -> (x_hat[B,16], unc[B,16])
```
Targets (training-time only): `log1p(count)` for the 12 item slots, `vital/16` for
health/food/drink/energy. Loss: masked MSE (slots engine always provides). Consumers: goal
scoring (vital-restoring context detection), diagnostics, logging. **Never an input to
OptionNet/ContinueNet/epistemic/policy heads.** If the Drive readout-v2 checkpoint is present,
ATLAS may OPTIONALLY load it via JEPAv3's own `state_readout_head` and skip local training —
record which path was used in provenance (`readout_source: drive_v2 | trained | none`).

### 5.8 `goals.py` — GoalBank + predecessor lift
```python
@dataclass
class GoalEntry:
    key: np.ndarray[128]        # visual key (z just BEFORE the event fired / at the resource)
    context: np.ndarray[256]    # h at that moment
    action_profile: np.ndarray[17,22]  # event head p(a,i) at that keyframe (from event_logits_all_actions)
    event_dims: list[int]       # dims the belief locked near this keyframe (empty for frontier/vital)
    kind: str                   # "event" | "vital" | "frontier"
    n_try: int; n_succ: int
    born_iter: int; last_used_iter: int
class GoalBank:
    def add(entry)                          # dedupe: if cos(key, existing) > 0.97 -> merge stats
    def select(z, h, m_soft, psi_fn, danger_fn, novelty_fn, w: GoalsConfig) -> (GoalEntry, score)
    def mark_used(key, success: bool)
    def save/load(path)                     # npz + json meta
class PredecessorLift:
    lift: np.ndarray[22,22]                 # lift[d,e]; ci_low: np.ndarray[22,22]
    def mine(episodes_events: list[np.ndarray[T,22]], window=30):
        # P(d in window before e | e) / P(d) per episode pair, Wilson CI at n_e successes
        # computed ONLY over the 22 event dims (Autopsy Rule 6: vitals never enter this)
    def prior_for(target_dims: list[int], m_soft) -> dict[goal_kind/dim hints -> weight]
    def save/load(path)
```
`select` scoring (all signals learned/mined; no names, no indices):
```
score(g) = w_unlock * max_{i: m_soft_i<0.5} action_profile[:,i].max()   # what could unlock there
         * (1 + 0.25 * sum_d lift[d, argmax_i] for d with m_soft_d<0.5)  # predecessor prior
         + w_reach * psi(s, g.key)
         + w_explor * (1 if g.kind=="frontier" else 0) * beta_t          # decayed curiosity
         + 0.1 * recency_bonus
         - w_danger * danger_along_route_estimate(s, g.key)
```
Goal sources mined per self-learn iteration (from this agent's own shards):
1. **event**: belief `newly_unlocked` at step t → store (z_{t-1}, h_{t-1}) keyframe with the
   action_profile computed there.
2. **vital**: health_head positive spike (`Δr_vital ≥ vital_spike 0.15`) with no unlock → store
   (z_t, h_t) as vital goal (drink/food/rest places emerge from reward decomposition, no names).
3. **frontier**: states in the top `frontier_quantile` (0.85) of disagreement in the episode.

### 5.9 `arbiter.py` — AtlasAgent (the only decision-maker)
```python
class AtlasAgent:
    def __init__(backbone, belief, topomap, goals, option_net, reach, epistemic, safety, readout, cfg)
    def reset_episode(seed)
    def act(obs_uint8, reward_prev, step) -> (action:int, telemetry:dict)
    def observe_result(obs_uint8, reward, done, discount)   # called after env.step
    telemetry keys: mode, goal_score, arrival_cos, psi, risk_intended, override_fired, r_int
```
Per-step algorithm:
1. `z = encode(obs)`; `h = recurrent_core(z, a_prev, h_prev)`; `s = [z;h]`; `topomap.observe`.
2. Update belief with the event head's own prediction for the action ABOUT to be chosen is not
   possible yet → belief is updated with the PREVIOUS step's (p, reward) at the start of `act`
   (ordering: belief.update happens first using last step's model prediction + realized reward).
3. **Safety override** (wraps whatever is chosen below): compute intended action first, then
   `risk`; apply the §5.6 rule; log `override_fired`.
4. Mode machine:
   - **PROBE** (entered on arrival or when goal is `kind="event"` and already at key): for
     `probe_steps (6)` steps choose `argmax_a [ max_{i: m_soft_i<0.5} p(a,i) + probe_entropy *
     softmax(disagreement(s,a)) ]` (event head ranking = learned affordance; no indices).
     Count probe outcomes into the GoalEntry stats (`mark_used`).
   - **NAVIGATE**: active goal g*; `a = option_net.act(s, g*.key)`; check arrival
     (`cos(z, g*) >= 0.93` → enter PROBE), stagnation (ψ flat 8 steps), timeout (25) → reselect.
   - **FALLBACK / explore**: `a ~ softmax(prior_logits + beta_t * disagreement(s,·))` with ε
     (scheduled) uniform tail.
5. Reselect goal when idle: `goals.select(...)`; if bank empty → FALLBACK.
6. Record (z, h, a, reward, p, m_soft, m_bin) into the in-progress episode buffer for the
   collector; `a_prev = a`.

Episode buffer → shard on done/max_steps (schema §6).

### 5.10 `collect.py` — EpisodeCollector + ShardLoader
```python
class EpisodeCollector:
    def __init__(agent, out_dir, cfg: CollectConfig)
    def run_episode(seed) -> dict      # full episode record (see schema)
    def run_episodes(n, seed_base) -> shard_path
    # epsilon/beta schedules: eps = lerp(start,end, iter/decay_iters); beta likewise
class ShardLoader:
    # reads npz shards (own format AND the existing data/sequences/*.npz T2 format:
    # obs, actions, rewards, dones, events, mask_before, episode_lengths)
    # T2 shards: events/mask_before provide supervision; inventory absent -> readout skipped for them
```
Shard npz schema (one file per self-learn iteration, `data/atlas/shard_<iter>_<n>.npz`):
```
obs: uint8 [T,64,64,3]        actions: int64 [T]      rewards: float32 [T]
dones: bool [T] (real deaths) truncated: bool [T]      discounts: float32 [T]
events: bool [T,22] (env-provided; TRAINING/SCORING TARGET ONLY)
mask_before: bool [T,22]      inventory: float32 [T,16] (targets; zeros where engine absent)
episode_lengths: int64 [E]    seeds: int64 [E]          source: "atlas_live"
```
**Drive paths on Colab**: read `learn-a-world/data/sequences/*.npz` (bootstrap) and the 666k
dataset under `learn-a-world/data/` (whatever the readout training audit used: 3832 eps) →
bootstrap shards. Write outputs to `learn-a-world/atlas_runs/<run_id>/…`; mirror result JSONs +
small checkpoints back into the repo.

### 5.11 `train.py` — HeadTrainer
1. **Latent caching** (mandatory): for each shard, encode all obs with the frozen backbone in
   batches of 512; run the GRU over each episode to get h (reset at episode starts); save
   sidecar `shard_*.latents.npz` with `z[T,128], h[T,256]`. Backbone frozen ⇒ cache is exact;
   training after this touches NO backbone forward. (Local smoke may run uncached.)
2. Datasets from cached latents:
   - OptionNet: HER batches (§5.4), horizons (5,10,20), 1 negative per positive, CE 1.0 + 0.5 BCE.
   - EpistemicNet: MSE residuals, dropout 0.1, Adam lr 3e-4.
   - ContinueNet: focal BCE (w=25, γ=2.0) on death labels; report class balance.
   - VitalsReadout: masked MSE on inventory targets (atlas shards only).
3. Adam, `epochs_per_iter` (6), batch 256, grad-clip 1.0. All losses logged to
   `atlas_runs/<run_id>/train_log.jsonl`.
4. Gate checks after each iteration (cheap, from val split 10% by episode):
   - reachability AUROC > 0.7; continue-head death recall > 0.5 (from trained data);
   - readout beats persistence (predict previous target) on held-out if Drive readout not used;
   - HER policy CE decreasing. Failures are logged loudly, never silently skipped.

### 5.12 `selflearn.py` — SelfLearnLoop
```
python -m atlas.colab.run_atlas --phase bootstrap --iters 1     # offline pass on Drive shards
python -m atlas.colab.run_atlas --phase selflearn --iters 40
python -m atlas.colab.run_atlas --phase screen   --tag atlas_v1
```
Loop body (per iteration): collect `episodes_per_iter` (8) → write shard → cache latents →
train heads (§5.11) → mine GoalBank + PredecessorLift (cumulative over all shards so far,
recomputed each iteration; cheap) → decay ε/β → every 5 iterations run dev screen
(seeds 6150–6159, 10 eps) → save state.
**Resumability**: every iteration, atomic-save
`atlas_runs/<run_id>/state.pt` (option/reach/epistemic/safety/readout state_dicts + optimizers,
GoalBank, PredecessorLift, TopoMap persistent, iter counter, RNG states, config dict). Resume
restores everything; seeds continue from manifest. 30-minute cadence on Colab.

### 5.13 `screen.py` — locked screening
Runs AtlasAgent with ε=0, β=0, greedy PROBE, no learning, no bank mutation. Seeds 2050–2099,
50 eps, ≤1000 steps. Output `results/atlas_screen_<tag>.json`:
```
protocol_id: "strict_visual_atlas_clean_v1", git_commit, timestamp,
backbone: {path, sha256, step}, heads_state_sha256, run_id,
strict_inference: true, privileged_inventory_used: false,
privileged_achievements_used_for_action: false, achievements_used_for_scoring_only: true,
seeds, num_episodes, crafter_score, mean/std survival, mean/std reward,
unlock_percentages (22), per-mode telemetry means (probe/nav/fallback fraction, override rate)
```
No screen JSON may be produced without every provenance field present (test enforces).

## 6. Colab runner contract (`atlas/colab/run_atlas.py`)
1. Repo at `/content/learn-a-world` (git pull), `pip install -e .` + deps (crafter, gymnasium).
2. Drive root `/content/drive/MyDrive/learn-a-world` — mounts, reads bootstrap shards, writes
   `atlas_runs/<run_id>/` + copies final `results/atlas_*.json` and screen checkpoints back.
3. Phases as §5.12; `--iters`, `--minutes` (wall-clock budget → graceful stop at boundary,
   state saved), `--run-id` (resume). T4-validated throughput target: ≥ 2 eps/min collection
   with frozen backbone + ≤ 90 s/iteration training on cached latents.

## 7. Tests (pytest, all runnable locally on MPS/CPU, each < 60 s)
- **isolation**: AST-scan `atlas/` imports ⊆ {allowed old modules ∪ atlas ∪ stdlib/3p}; git
  status shows no modified tracked files outside `atlas/`, `tests/`, `results/`, `data/`.
- **backbone**: loads; `assert_frozen()` passes; encode same obs twice → bitwise-equal latents;
  provenance SHA matches file.
- **belief**: threshold/fallback logic; refuses extra inputs; reset clears.
- **memory**: TopoMap dedupe/match thresholds; persistence across reset_episode; GoalBank
  merge-on-duplicate; lift matrix correct on a synthetic 3-episode pattern (wood→table→pickaxe
  ordering yields lift[wood,table] > 1).
- **skills**: HER batch shapes; 200 gradient steps on synthetic cyclic trajectories reduce CE >
  50%; reachability separates same-episode vs cross-episode goals (AUROC > 0.9 synthetic).
- **epistemic**: disagreement(random pairs) > disagreement(train pairs) after training.
- **safety**: with 5% synthetic deaths, focal BCE reaches recall ≥ 0.5.
- **arbiter**: mode transitions unit-tested with stub nets; override fires when ContinueNet
  stubbed to predict death for the intended action; telemetry keys present; act() reads no env
  info (signature has no info parameter).
- **train/collect**: shard round-trip; latent cache equals recomputation; one trainer step on
  tiny batch decreases loss.
- **screen**: JSON schema/fields; strict flags true; seed coverage exact.

## 8. E2E smoke (local, the "works" bar)
`pytest atlas/tests -q` green, then:
```
python -m atlas.smoke   # 2 episodes seeds 6150/6151, 200 steps, tiny nets, asserts:
                        # episode completes, telemetry populated, belief monotone, shard written
```

## 9. Blocker → mechanism map
| Autopsy blocker | ATLAS mechanism |
|---|---|
| 1 post-table return (87.5% drop-off) | TopoMap visual keyframe + OptionNet navigate-back (no dead-reckoning) |
| 2 nightfall mortality (93.3% deaths) | ContinueNet per-step reactive override + vital-place goals |
| 3 stone affordance (0%) | EpistemicNet disagreement + frontier goals; Drive-bootstrap stone/coal keyframes accelerate first contact |
| 4 horizon compounding (H≤10) | closed-loop options (effective 25-step horizon, zero latent rollout); imagination capped at H=1 residuals |
| 5 catalyst duality | PROBE mode: event head ranks actions at the arrived view; the table lives in the latent as visual precondition |

Iron rules: attribution only when the event head itself believed the context (p > threshold at
keyframe); readiness per immediate hop (arrive→probe each goal); no CEM beyond imagination H=1;
no unconditional action bonuses; predecessor lift over 22 event dims only; options + spatial
anchors + reactive survival are exactly the Rule-7 prescription.

## 10. Milestones & gates
| # | Scope | Gate | Status |
|---|---|---|---|
| A0 | package, backbone, belief, isolation tests | local pytest | **DONE** (config+backbone written) |
| A1 | memory/skills/epistemic/safety/readout/goals | unit tests | implement |
| A2 | arbiter + collect + train + selflearn + smoke | e2e smoke green | implement |
| A3 | Colab bootstrap (Drive shards incl. rare stone/coal) + 20-iter selflearn | gates §5.11 pass | run |
| A4 | locked 50-ep screen | ≥ 4.5% (> honest 4.31%) | run |
| A5 | scale (100+ iters, β decay), ablations (exploration off, PROBE off) | ≥ 6% | run |

Ablations (each ships with the claim it supports): A-expl (epistemic off → stone discovery
dies), A-probe (PROBE off → table→pickaxe drop-off returns), A-her (OptionNet off → return
navigation collapses), A-safe (ContinueNet off → nightfall deaths dominate). Report all four in
the A5 JSON.

## 11. Honesty contract
- Rare-event bootstrap from Drive shards is declared in provenance (`bootstrap_sources` +
  SHA-256s); it is T2-collected strict-visual experience, never engine-privileged data.
- Every number in reports carries VERIFIED / OBSERVED / PLANNED / HYPOTHESIS labels.
- No silent skips: gate failures are logged loudly and appear in the result JSON `gates` field.
