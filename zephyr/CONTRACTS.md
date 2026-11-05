# Zephyr internal contracts (pinned interfaces)

Every module conforms to these signatures. If a signature must change, change it here first
and in every conforming module in the same commit. Shapes are annotated with `[B, ...]`
etc.; `T` = sequence length, `H` = imagination horizon.

## Tensor conventions

- observations: `uint8 [B, 64, 64, 3]` (HWC) entering any API; models convert internally to
  float `CHW` in `[0,1]`.
- actions: `int64 [B]` or `[B, T]`, values `0..16`.
- belief: `h = float32 [B, 512]`; state for heads: `s = concat(z, h) → [B, 1024]`.
- inventory vector: `float32 [B, 16]` in `config.INVENTORY_KEYS` order (vitals scaled /9,
  counts raw); **training target only, never an inference input**.
- achievements: `bool [B, 22]` in `config.ACHIEVEMENTS` order.

## zephyr/losses.py

```python
def symlog(x: Tensor) -> Tensor
def inv_symlog(x: Tensor) -> Tensor

class TwoHotCoder:
    def __init__(self, symlog_range: float = 20.0, num_bins: int = 65)
    def encode(self, values: Tensor) -> Tensor            # [...] -> [..., bins] two-hot float
    def loss(self, logits: Tensor, values: Tensor) -> Tensor   # scalar mean CE
    def decode(self, logits: Tensor) -> Tensor            # [..., bins] -> [...]

def lambda_returns(rewards, continues, values, bootstrap, gamma=0.997, td_lambda=0.95) -> Tensor
    # rewards/continues/values: [T, B]; bootstrap: [B] value at t=T; returns [T, B]

def vicreg_var_loss(z: Tensor, gamma: float = 1.0) -> Tensor   # mean(relu(gamma - std(z))) over flattened [N, D]
def vicreg_cov_loss(z: Tensor) -> Tensor                       # mean off-diagonal cov^2 over flattened [N, D]
def focal_bce_with_logits(logits, targets, pos_weight=25.0, gamma=2.0) -> Tensor  # scalar
def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor  # mask broadcastable; 0-mask excluded
```

## zephyr/wm.py

```python
class ZephyrEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig)
    def forward(self, obs_uint8: Tensor) -> Tensor   # [N,64,64,3] u8 -> [N,512] float (LayerNorm'd)

class ZephyrGRU(nn.Module):
    def __init__(self, cfg: WorldModelConfig)
    def forward(self, z: Tensor, action: Tensor, h: Tensor) -> Tensor
        # z [B,512] (or [B,T,512] w/ action [B,T], h [B,512] broadcast) -> h_next same shape as z
    def init_state(self, batch: int, device) -> Tensor            # zeros [B,512]
    def initial_input_dim -> int  # latent_dim + action_embed_dim

class ZephyrPredictor(nn.Module):
    def __init__(self, cfg: WorldModelConfig)
    def forward(self, z: Tensor, action: Tensor, h: Tensor) -> Tensor  # -> z_hat [.., 512], LayerNorm'd

class ZephyrWM(nn.Module):
    def __init__(self, cfg: ZephyrConfig)
    def encode(self, obs_u8: Tensor) -> Tensor                    # online encoder (grad)
    def ema_encode(self, obs_u8: Tensor) -> Tensor                # target encoder, no_grad
    def belief_step(self, z: Tensor, action: Tensor, h: Tensor) -> Tensor
    def predict(self, z: Tensor, action: Tensor, h: Tensor) -> Tensor
    def init_state(self, batch: int, device) -> Tensor
    def update_ema(self, step: int) -> None                       # decay ramp by cfg
    def save_checkpoint(self, path, step: int, extra: Optional[dict] = None) -> None
    @classmethod
    def load_checkpoint(cls, path, device="cpu", strict=True) -> Tuple["ZephyrWM", dict]
        # returns (model, meta); meta has step/extra; format key "zephyr_v1"
```

`ZephyrWM` holds `encoder`, `ema_encoder` (deepcopy), `gru`, `predictor` as attributes.
Multi-step free-running rollouts are composed by callers via `predict` + `belief_step`:
the predicted z-hat feeds `belief_step` for the next step (closed-loop latent rollout).

## zephyr/heads.py

```python
class RewardHead(nn.Module):   # s [B,1024] -> logits [B, bins]; coder from HeadsConfig
class ContinueHead(nn.Module): # s -> logit [B]
class ReadoutHead(nn.Module):  # s -> [B,16] predicted inventory vector (linear head)
class EffectHead(nn.Module):   # (s [B,1024], action) -> [B,16] predicted inventory delta
class CriticHead(nn.Module):   # s -> logits [B, critic_bins]
class ActorHead(nn.Module):    # s -> logits [B, 17]
class DynamicsEnsemble(nn.Module):  # (z [B,512], action, h [B,512]) -> per-member z_hat [K, B, 512]
class ZephyrHeads(nn.Module):  # container: .reward .contin .readout .effect .critic .actor .ensemble
    def __init__(self, cfg: ZephyrConfig)
    def save_checkpoint(path, step, extra=None) / load_checkpoint(path, device) -> (heads, meta)
    # head input builder: heads.state_input(z, h) -> s
```

## zephyr/vec_env.py

```python
class ZephyrVecEnv:
    def __init__(self, num_envs: int, base_seed: int = 7000, size: int = 64,
                 max_steps: int = 1000, in_process: bool = False)
        # 'spawn' multiprocessing; in_process=True forces single-env loop (tests/mac)
    def reset(self) -> Tuple[np.ndarray /*[N,64,64,3] u8*/, List[dict]]
    def step(self, actions: np.ndarray /*[N] int*/) ->
        Tuple[np.ndarray /*obs*/, np.ndarray /*rewards f32 [N]*/, np.ndarray /*done bool [N]*/,
              List[dict]]
        # info dict keys: 'zephyr_inventory' f32[16], 'zephyr_achievements_cum' bool[22],
        #                 'episode_seed' int, 'truncated' bool
        # auto-reset on done: returned obs is the NEW episode's first obs
    def close(self) -> None
```

Uses `envs.crafter_wrapper.CrafterEnv` (read-only whitelist import).

## zephyr/buffer.py

```python
@dataclass
class EpisodeRecord:
    obs: np.ndarray            # uint8 [T,64,64,3]
    actions: np.ndarray        # int16 [T]
    rewards: np.ndarray        # float32 [T]  (extrinsic env reward)
    rewards_int: np.ndarray    # float32 [T]  (intrinsic, 0.0 for S0 data)
    done: np.ndarray           # bool [T]
    inventory: np.ndarray      # float32 [T,16]
    achievements_cum: np.ndarray  # bool [T,22]
    achievements_new: np.ndarray  # bool [T,22]
    seed: int
    source: str                # "random" | "actor" | ...

class ZephyrReplay:
    def add_episode(self, rec: EpisodeRecord) -> None          # FIFO up to capacity
    def __len__(self) -> int                                   # frames
    @property episodes -> int
    def sample(self, batch: int, seq_len: int, rng: np.random.Generator,
               prioritize_achievement_frac: float = 0.1) -> Dict[str, Tensor]
        # windows never cross episode boundaries (clipped+mask-padded inside episode):
        #   obs      uint8 [B, seq_len, 64,64,3]   (steps t .. t+seq_len-1)
        #   obs_next uint8 [B, seq_len, 64,64,3]   (steps t+1 .. t+seq_len; last obs of episode repeats)
        #   actions  int64 [B, seq_len], rewards f32, rewards_int f32, done bool, valid bool [B, seq_len]
        #   inventory f32 [B, seq_len,16], achievements_new bool [B, seq_len,22]
    def achievement_counts(self) -> Dict[str, int]
    def labeled_fraction(self) -> float
    def split(self, val_ratio: float, seed: int = 4242) -> Tuple["ZephyrReplay", "ZephyrReplay"]
    def save_shard(self, path: str) -> None                    # .npz, all episodes
    @classmethod
    def load_shard(cls, path: str) -> "ZephyrReplay"
```

## zephyr/ac.py, zephyr/explore.py, zephyr/planner.py, zephyr/screen.py

```python
# ac.py
def imagine_trajectory(wm, heads, z0: Tensor, h0: Tensor, horizon: int,
                       cfg: ZephyrConfig, deterministic: bool = False) -> dict
    # closed-loop latent rollout; actor samples actions (unimix categorical)
    # returns dict(z=[H,B,512], h=[H,B,512], actions=[H,B] int64,
    #              log_prob=[H,B], entropy=[H,B], reward_logits=[H,B,bins], continue_logits=[H,B])
def actor_lambda_return_loss(lam_ret: Tensor, log_prob: Tensor, entropy: Tensor,
                             cfg: ActorCriticConfig) -> Tensor     # scalar
def critic_twohot_loss(critic_logits: Tensor, lam_ret: Tensor, coder: TwoHotCoder) -> Tensor

# explore.py
class EpisodicNovelty:      # kNN over h within episode
    def __init__(self, cfg: ExplorationConfig)
    def reset(self) -> None
    def score(self, h: np.ndarray /*[512]*/) -> float   # mean dist to k-NN (normalized ~[0,1])
    def add(self, h: np.ndarray) -> None

# planner.py
class ZephyrAgent:          # MODE A — strict actor-greedy
    def __init__(self, wm: ZephyrWM, heads: ZephyrHeads, cfg: ZephyrConfig, device="cpu")
    def reset(self) -> None
    def act(self, obs_uint8: np.ndarray /*[64,64,3]*/) -> int    # NO other inputs, ever
class ZephyrCEMAgent:       # MODE B — ablation only
    def __init__(..., horizon=15, candidates=64, iterations=2, ...)
    def reset(self) / def act(self, obs_uint8) -> int

# screen.py
def run_screen(checkpoint_wm: str, checkpoint_heads: str, out_path: str,
               num_episodes: int = 50, seed_start: int = 2050, max_steps: int = 1000,
               agent_mode: str = "actor", device: str = "cpu",
               drive_mirror: Optional[str] = None) -> dict
    # protocol_id "strict_visual_zephyr_clean_v1"; scores via eval.metrics.compute_crafter_score
    # JSON: git commit, checkpoint sha256, config, strict flags, provenance labels,
    #       per-episode logs, per-achievement unlock %, score
```

## zephyr/train_wm.py, zephyr/train_loop.py

```python
# train_wm.py
def wm_joint_losses(wm, heads, batch, cfg, device) -> Tuple[Tensor, Dict[str, float]]
    # JEPA (free-running multi-step, horizon-decay weights, var/cov anti-collapse)
    # + reward CE + continue focal + readout MSE + effect MSE; returns (total, metrics)
def build_optimizers(wm, heads, cfg) -> List[torch.optim.Optimizer]  # param groups by lr class
def train_world_model(wm, heads, replay: ZephyrReplay, cfg, device, steps,
                      log_every=100, ckpt_path=None, on_checkpoint=None) -> Dict[str, float]
def evaluate_gates(wm, heads, val_replay: ZephyrReplay, cfg, device) -> dict
    # ZG1 collapse/latent-std, ZG2 free-run vs persistence @1/5/10/15,
    # ZG3 readout vs persistence, ZG5 continue recall/precision/AUC vs majority baseline

# train_loop.py
class ZephyrTrainer:      # S1 joint loop (WM + actor-critic + exploration), resumable
    def __init__(self, cfg, device, run_dir)
    def collect(self, frames: int, beta_intrinsic: float) -> None   # actor plays, episodes -> replay + shards
    def train(self, steps: int) -> Dict[str, float]                 # WM loss + AC losses on same batches
    def save_run(self, path: str) -> None
    @classmethod
    def load_run(cls, path: str, device) -> "ZephyrTrainer"
```

---

## Implementation notes (as built 2026-10-28) — deltas from the pinned spec above

These are the verified deviations/concretizations; the code is the source of truth.

1. **vec_env**: on auto-reset steps, info additionally carries `next_inventory`
   (float32 [16]) and `next_achievements_cum` (bool [22]) describing the NEW
   episode's starting state — no hardcoded starting-vitals constant anywhere.
2. **buffer**: capacity is in FRAMES (`ZephyrReplay(capacity_frames=...)`,
   `cfg.data.replay_capacity_frames`); `sample()` additionally returns
   `inventory_next [B, T, 16]`. Window boundary rule: the window's last
   next-frame is the true next frame mid-episode and the episode's own final
   frame (bootstrap) when the window reaches the episode end.
3. **wm**: multi-step rollouts are composed as documented in `ac.py`:
   `z_hat_{t+1} = predict(z_t, a_t, h_t)`; `h_{t+1} = belief_step(z_{t+1}, a_t, h_t)`
   (posterior uses the encoded real next latent; imagination/prior uses the
   predicted latent). `_posterior_states` / `_prior_rollout` in train_wm and
   `imagine_trajectory` in ac implement exactly this.
4. **losses**: `roc_auc` added (rank-based, tie-aware); `masked_mse` expands the
   mask over trailing dims. TwoHotCoder attribute is `bin_centers`.
5. **ac**: `trajectory_lambda_returns(imagination, critic_coder, cfg,
   intrinsic_weight=0.0, slow_value_logits=None)`; plus `normalize_returns`
   (per-timestep standardization) and `ensemble_disagreement([K,...,D]) -> [...]`.
6. **train_wm**: `wm_joint_losses(..., return_states=False)` optionally returns
   `{"z_now", "h_pre", "s_all"}` (posterior tensors, grad attached) for AC reuse;
   `train_world_model(..., heads_ckpt_path=None)` saves the WM/heads pair with
   explicit paths; `evaluate_gates` returns `zg1` (latent-std collapse, threshold
   0.05 on LayerNorm'd latents), `zg2` (free-run vs persistence @1/5/10/15),
   `zg3` (readout vs lag-1 persistence, ≥30% gate), `zg5` (recall/precision@0.5
   + AUC vs majority baseline).
7. **screen**: `run_screen(checkpoint_wm, checkpoint_heads, out_path, ...)` —
   takes BOTH checkpoint paths; asserts `config.ACHIEVEMENTS` order equals
   `eval.metrics.CRAFTER_ACHIEVEMENTS` before scoring.
8. **train_loop**: `ZephyrTrainer(cfg, device, run_dir, load_checkpoints=None,
   env_in_process=False)` with `collect / train / save_run / save_checkpoints /
   load_run / load_shards / run_dev_screen`. Collector generations offset env
   seeds by 1e9 per construction so resumed runs never replay seed streams.
   Intrinsic weight in imagination = `cfg.explore.beta(env_frames) *
   disagreement_weight` (episodic novelty is a real-state-only signal).
9. **Gates threshold note**: ZG1's plan text said "latent std ≥ 1.0"; as built the
   threshold is 0.05 mean per-dim batch std because latents are LayerNorm'd
   (healthy ≈ 0.1–0.5, collapsed ≈ 0). Recorded in every gate report.
