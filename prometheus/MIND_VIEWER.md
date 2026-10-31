# MIND_VIEWER.md — See the Agent's Map, Understanding, and Imagination

**Version:** 1.0 — 2026-10-31
**Purpose:** a read-only diagnostic that visualizes HOW the v3.1 agent views the world, what its token map encodes, and what its predictions look like — tile by tile. No training, no risk. Runs on any existing checkpoint (v3.1-w1a lineage: checkpoints_v4/v5 state.pt).
**Why now:** the W1a-0 reconstruction head (decode_tokens) makes the agent's internal map *decodable into pictures* for the first time. This viewer is also the direct verification of the W1a scene-collapse fix: a healthy brain shows scenes in its map; a collapsed brain shows grass everywhere.
**Runtime:** ~2 minutes on GPU. Read-only — touch nothing in checkpoints.

---

## THE 6 VIEWS (what each answers)

| View | Question it answers | Healthy signature | Broken signature |
|---|---|---|---|
| **V1 What it sees** | Does the token map carry the scene? | Reconstruction shows player, objects, HUD | Reconstruction = grass blur, no player/HUD |
| **V2 Its map** | Which tiles does the brain write strongly? | Player-center + object tiles + HUD row light up | All tiles uniform |
| **V3 How it predicts, tile by tile** | Does prediction error land where change happens? | Error heatmap aligns with the change heatmap | Uniform error everywhere |
| **V4 Its imagination** | What do 8 dreamed steps look like? | Player moves, scene evolves, blur grows *gradually* | All dream frames identical static grass |
| **V5 Action sensitivity** | Does the model know what each action does? | `do` changes the faced tile; `move_*` shift the view; 17 distinct maps | All 17 maps identical |
| **V6 What it thinks this is** | What does the agent associate with the current state? | Neighbors share the current context | Neighbors are random |

---

## THE SCRIPT (complete — copy as `src/mind_viewer.py`, run `python3 -m src.mind_viewer --checkpoint <state.pt>`)

```python
import argparse, torch, numpy as np, matplotlib.pyplot as plt
from src.agent.agent import JEPAAgent
from src.env.wrapper import make_crafter_env

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    dev = torch.device(args.device)

    agent = JEPAAgent(latent_dim=256, num_actions=17, device=dev)
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    agent.load_state_dict(ck["components"]["agent"].state_dict())
    agent.eval()

    env = make_crafter_env(seed=args.seed)
    obs = env.reset()
    h = torch.zeros(1, 256, device=dev)

    # ---- collect one real episode: frames, tokens, actions, real next tokens ----
    frames, tokens_seq, pooled_seq, actions, target_next = [], [], [], [], []
    for t in range(args.steps):
        x = obs.unsqueeze(0).to(dev)
        tokens, pooled = agent.encoder.forward_tokens(x)          # (1,64,64), (1,256)
        readout = agent.get_readout(pooled, h)
        probs = agent.actor.get_probs(readout)
        action = int(torch.multinomial(probs, 1).item())
        nxt, r, term, trunc, info = env.step(action)
        done = term or trunc
        nxt_dev = nxt.unsqueeze(0).to(dev)
        t_next, _ = agent.encoder.forward_tokens(nxt_dev)          # real next tokens (answer key)
        frames.append(obs.clone()); tokens_seq.append(tokens[0].detach().cpu())
        pooled_seq.append(pooled[0].detach().cpu()); actions.append(action)
        target_next.append(t_next[0].detach().cpu())
        h = agent.gru(pooled, h)
        obs = nxt if not done else env.reset()
        if done: h = torch.zeros_like(h)
    T = len(frames)
    tokens_all = torch.stack(tokens_seq).to(dev)                   # (T,64,64)
    target_next = torch.stack(target_next).to(dev)                 # (T,64,64)

    # ================= V1: WHAT IT SEES (reconstruction) =================
    idxs = np.linspace(0, T-1, 5).astype(int)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for col, i in enumerate(idxs):
        axes[0, col].imshow(frames[i].permute(1, 2, 0).cpu().numpy()); axes[0, col].set_title(f"REAL t={i}"); axes[0, col].axis("off")
        rec = agent.encoder.decode_tokens(tokens_all[i:i+1].to(dev))[0].cpu().clamp(0, 1)
        axes[1, col].imshow(rec.permute(1, 2, 0).numpy()); axes[1, col].set_title("AGENT'S MAP DECODED"); axes[1, col].axis("off")
    fig.suptitle("V1: WHAT THE AGENT SEES (real frame vs its own token map decoded)")
    fig.savefig("mind_v1_recon.png", dpi=130); plt.close(fig)

    # ================= V2: ITS MAP (per-token strength + variation) =================
    tok_norm = tokens_all.norm(dim=-1)                             # (T, 64)
    tok_var = tokens_all.std(dim=0).norm(dim=-1)                   # (64,) how much each POSITION varies over time
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im0 = axes[0].imshow(tok_norm.mean(0).cpu().view(8, 8), cmap="viridis"); axes[0].set_title("Mean token strength (agent's map)")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(tok_var.cpu().view(8, 8), cmap="magma"); axes[1].set_title("Which tiles CHANGE over time (scene info)")
    plt.colorbar(im1, ax=axes[1])
    fig.savefig("mind_v2_map.png", dpi=130); plt.close(fig)
    print("[V2] scene-info tiles (top-8 positions):", torch.topk(tok_var, 8).indices.cpu().tolist())
    print("     Expect: center (player) + bottom rows (HUD) + edges (entering objects). Uniform = collapse.")

    # ================= V3: PREDICTION, TILE BY TILE =================
    per_tile_cos = np.zeros((T, 64)); per_tile_change = np.zeros((T, 64))
    for t in range(T-1):
        a = torch.tensor([actions[t]], device=dev)
        pred_mean, _ = agent.predictor.forward_tokens(tokens_all[t:t+1].to(dev), a, h=None)
        tgt = target_next[t+1].unsqueeze(0)
        cur = tokens_all[t:t+1].to(dev)
        cos = torch.nn.functional.cosine_similarity(pred_mean[0].cpu(), tgt[0].cpu(), dim=-1)   # (64,)
        chg = (tgt[0].cpu() - tokens_seq[t].to(dev)).norm(dim=-1)                                # (64,) real change
        per_tile_cos[t] = cos.numpy(); per_tile_change[t] = chg.numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im0 = axes[0].imshow(per_tile_cos.mean(0).reshape(8, 8), cmap="RdYlGn", vmin=0.5, vmax=1.0)
    axes[0].set_title("Mean PREDICTION accuracy per tile"); plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(per_tile_change.mean(0).reshape(8, 8), cmap="hot")
    axes[1].set_title("Where the world actually CHANGES"); plt.colorbar(im1, ax=axes[1])
    fig.suptitle("V3: prediction accuracy vs real change — do errors land on dynamic tiles?")
    fig.savefig("mind_v3_predict.png", dpi=130); plt.close(fig)

    # ================= V4: THE IMAGINATION (dream filmstrip) =================
    start = T // 2
    tokens_hat = tokens_seq[start].unsqueeze(0).to(dev)
    h_dream = torch.zeros(1, 256, device=dev)
    dream = [agent.encoder.decode_tokens(tokens_hat)[0].cpu().clamp(0, 1)]
    real = [frames[start].cpu().clamp(0, 1)]
    for k in range(8):
        pooled_hat = agent.encoder.pool_tokens(tokens_hat)
        h_dream = agent.gru(pooled_hat, h_dream)
        a = torch.tensor([actions[min(start + k, T - 1)]], device=dev)
        tokens_hat, _ = agent.predictor.forward_tokens(tokens_hat, a, h=h_dream)
        dream.append(agent.encoder.decode_tokens(tokens_hat)[0].cpu().clamp(0, 1))
        if start + k + 1 < T: real.append(frames[start + k + 1].cpu().clamp(0, 1))
    fig, axes = plt.subplots(2, 9, figsize=(22, 5))
    for k in range(9):
        axes[0, k].imshow(dream[k].permute(1, 2, 0).numpy()); axes[0, k].set_title(f"IMAGINED +{k}"); axes[0, k].axis("off")
        if start + k < len(frames):
            axes[1, k].imshow(frames[start + k].permute(1, 2, 0).cpu().numpy()); axes[1, k].set_title(f"REAL +{k}")
        axes[1, k].axis("off")
    fig.suptitle("V4: THE AGENT'S IMAGINATION vs reality (8 steps of pure self-prediction)")
    fig.savefig("mind_v4_dream.png", dpi=130); plt.close(fig)

    # ================= V5: ACTION SENSITIVITY (17 'what-if' maps) =================
    base = tokens_seq[start].unsqueeze(0).to(dev)
    preds = []
    for a in range(17):
        p, _ = agent.predictor.forward_tokens(base, torch.tensor([a], device=dev), h=None)
        preds.append(p[0][0].cpu())                                 # (64, token_dim)
    preds = torch.stack(preds)                                      # (17, 64, token_dim)
    base_pooled = agent.encoder.pool_tokens(base).cpu()
    deltas = (preds - agent.encoder.pool_tokens(preds).mean(0).unsqueeze(0)).norm(dim=-1)  # (17, 64) per-action per-tile deviation
    fig, axes = plt.subplots(2, 9, figsize=(22, 6))
    for a in range(17):
        ax = axes[a // 9, a % 9]
        im = ax.imshow(deltas[a].view(8, 8).numpy(), cmap="hot", vmin=0, vmax=float(deltas.max()))
        ax.set_title(f"{a}:{['noop','mL','mR','mU','mD','do','slp','pSt','pTb','pFu','pPl','mWP','mSP','mIP','mWS','mSS','mIS'][a]}", fontsize=7)
        ax.axis("off")
    fig.suptitle("V5: WHAT EACH ACTION CHANGES in the agent's prediction (per-tile deviation)")
    fig.savefig("mind_v5_actions.png", dpi=130); plt.close(fig)
    print("[V5] mean per-action deviation (17 numbers):")
    print(np.round(deltas.mean(dim=1).numpy(), 4).tolist())
    print("     Healthy: 'do'/'move_*' deviate clearly; noop ~0. Action-blind: ALL ~equal.")

    # ================= V6: WHAT IT THINKS THIS IS (nearest neighbors) =================
    q = tokens_seq[start].to(dev).flatten()
    sims = torch.tensor([torch.nn.functional.cosine_similarity(q, tokens_all[i].to(dev).flatten(), dim=0).item() for i in range(T)])
    top = torch.topk(sims, 6).indices.tolist()
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    for col, i in enumerate(top):
        axes[col].imshow(frames[i].permute(1, 2, 0).cpu().numpy()); axes[col].set_title(f"sim={sims[i]:.3f} t={i}"); axes[col].axis("off")
    fig.suptitle("V6: the 6 frames the agent's map considers most similar to now")
    fig.savefig("mind_v6_knn.png", dpi=130); plt.close(fig)

    print("[+] Mind viewer complete: mind_v1..v6 saved.")

if __name__ == "__main__":
    main()
```

---

## INTERPRETATION GUIDE (read the images against these)

**V1 — What it sees.** The recon decodes the agent's own map back to a picture. Healthy: the player sprite, trees/objects, and the HUD strip are recognizable in the decoded frame. Collapsed: a grass-colored blur with no player and no HUD digits — the map carries no scene.

**V2 — Its map.** "Which tiles change over time" should light up: the **center** (the player is always egocentric-center), the **bottom rows** (HUD digits), and the **edges** (objects entering the view). If the heatmap is uniform, every tile carries the same (near-zero) scene information — the collapse signature.

**V3 — Prediction vs change.** The key overlay: prediction error should sit *on top of* real change (the player tile, the faced tile, HUD). If accuracy is uniformly ~1.0 while change happens at the center/edges, the model predicts "nothing changes" and happens to be right on static tiles — the action-blindness made visible.

**V4 — The dream.** The imagined frames show what the agent's world model actually believes. Healthy: the player shifts with the moves, blur grows gradually. Broken: every imagined frame is the same grass. **This is the drift test made visual** — and it directly previews whether dreaming (U4) would train on truth or fantasy.

**V5 — Action sensitivity.** Seventeen "what changes if I do this" maps. Healthy: `do` deviates on the faced tile, `move_*` shift the whole frame, `make_*`/`place_*` deviate near the player. Broken: seventeen identical near-zero maps — the model literally does not know actions exist.

**V6 — Nearest neighbors.** Shows what the agent's memory considers "the same situation." Sanity check for the episodic memory that drives ε.

---

## RUN NOTES

- Works on any v3.1-w1a checkpoint (checkpoints_v4/v5 state.pt). No training, no risk, read-only.
- `weights_only=False` is required (own trusted checkpoint — same P0.1 convention).
- If V1 reconstruction is poor, first check `l_recon` in the training telemetry — the head needs its 0.3 weight to have been trained (it was, per config verification).
- Expected runtime: ~2 min GPU for `--steps 60`.

---

## WHAT TO SEND BACK

The six PNGs + the two printed lines ([V2] top-8 positions, [V5] 17 deviations). With those, we can SEE whether the W1a fix gave the agent sight — before spending any more GPU on training.
