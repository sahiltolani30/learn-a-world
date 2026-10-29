"""Visual Memory Inspector & Knowledge Graph Generator for ATLAS.

Produces visual plots and interactive HTML dashboards of the agent's:
1. PredecessorLift Causal Network Graph (Directed Acyclic Graph of dependencies)
2. GoalBank Visual Memory Affordances (What action triggers what achievement at each goal)
3. TopoMap 2D Spatial Territory Map (PCA projection of 4,096 keyframes)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from atlas.config import CRAFTER_ACTION_NAMES, REPO_ROOT, resolve_drive_root
from eval.metrics import CRAFTER_ACHIEVEMENTS


def plot_causal_graph(lift_mat: np.ndarray, out_path: Path) -> Path:
    """Generates a visual network graph of learned causal dependencies."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import networkx as nx
    except ImportError:
        return out_path

    G = nx.DiGraph()

    # Find significant causal edges
    for i in range(len(CRAFTER_ACHIEVEMENTS)):
        for j in range(len(CRAFTER_ACHIEVEMENTS)):
            if i != j and lift_mat[i, j] > 2.0:
                src = CRAFTER_ACHIEVEMENTS[i]
                dst = CRAFTER_ACHIEVEMENTS[j]
                weight = float(lift_mat[i, j])
                G.add_edge(src, dst, weight=weight)

    if len(G.nodes) == 0:
        return out_path

    plt.figure(figsize=(14, 9), facecolor="#0d1117")
    ax = plt.gca()
    ax.set_facecolor("#0d1117")

    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Node colors by category
    node_colors = []
    for node in G.nodes():
        if "wood" in node or "table" in node or "pickaxe" in node:
            node_colors.append("#d29922")  # Gold/Orange: Technology
        elif "cow" in node or "drink" in node or "plant" in node:
            node_colors.append("#3fb950")  # Green: Sustenance / Agriculture
        elif "zombie" in node or "skeleton" in node:
            node_colors.append("#f85149")  # Red: Combat
        else:
            node_colors.append("#58a6ff")  # Blue: General

    # Draw edges with varying thickness
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    norm_widths = [max(1.5, min(5.0, (w / max_w) * 5.0)) for w in weights]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3200, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="#8b949e", width=norm_widths,
        arrows=True, arrowsize=20, arrowstyle="-|>",
        connectionstyle="arc3,rad=0.15", ax=ax,
    )
    nx.draw_networkx_labels(
        G, pos, font_size=10, font_weight="bold", font_color="#ffffff",
        font_family="sans-serif", ax=ax,
    )

    edge_labels = {(u, v): f"+{d['weight']:.0f}x" for u, v, d in G.edges(data=True) if d['weight'] >= 25.0}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="#e3b341", ax=ax)

    plt.title("ATLAS Learned Causal Graph (PredecessorLift Memory)\nArrows indicate: Action A causes statistical unlock of Action B", color="#ffffff", fontsize=15, pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    return out_path


def plot_goal_affordances(meta: List[Dict[str, Any]], profiles: np.ndarray, out_path: Path) -> Path:
    """Plots action affordances for key representative achievement goals."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pick representative goals for key achievements
    target_achs = ["collect_sapling", "place_plant", "eat_cow", "defeat_zombie", "collect_wood"]
    selected_indices = []
    selected_labels = []

    for ach in target_achs:
        if ach in CRAFTER_ACHIEVEMENTS:
            dim = CRAFTER_ACHIEVEMENTS.index(ach)
            for idx, m in enumerate(meta):
                if dim in m.get("event_dims", []) and idx not in selected_indices:
                    selected_indices.append(idx)
                    selected_labels.append(f"Goal #{idx}\n({ach})")
                    break

    if not selected_indices:
        return out_path

    # Extract profiles for selected goals [N_goals, 17 actions, target_dim]
    fig, axes = plt.subplots(1, len(selected_indices), figsize=(4 * len(selected_indices), 6), facecolor="#0d1117")
    if len(selected_indices) == 1:
        axes = [axes]

    for ax_idx, (g_idx, label) in enumerate(zip(selected_indices, selected_labels)):
        ax = axes[ax_idx]
        ax.set_facecolor("#161b22")

        # Sum of predicted achievement probabilities across all achievements for each action
        action_scores = np.max(profiles[g_idx], axis=-1)  # [17]
        best_a = int(np.argmax(action_scores))

        colors = ["#3fb950" if a == best_a else "#30363d" for a in range(len(CRAFTER_ACTION_NAMES))]
        bars = ax.barh(range(len(CRAFTER_ACTION_NAMES)), action_scores, color=colors)

        ax.set_yticks(range(len(CRAFTER_ACTION_NAMES)))
        ax.set_yticklabels(CRAFTER_ACTION_NAMES, color="#c9d1d9", fontsize=9)
        ax.set_xlabel("Predicted Affordance", color="#8b949e", fontsize=10)
        ax.set_title(label, color="#ffffff", fontsize=11, fontweight="bold", pad=10)
        ax.tick_params(colors="#8b949e")
        ax.invert_yaxis()

    fig.suptitle("Visual GoalBank Memories: Memorized Action Affordances\n(Green bar = Agent's chosen optimal action when seeing this goal)", color="#ffffff", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    return out_path


def plot_spatial_topomap(z_coords: np.ndarray, visits: np.ndarray, out_path: Path) -> Path:
    """Projects 4,096 spatial keyframes into 2D PCA space showing the agent's world map."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(z_coords) < 10:
        return out_path

    # Simple 2D PCA projection
    z_mean = np.mean(z_coords, axis=0)
    z_centered = z_coords - z_mean
    u, s, vt = np.linalg.svd(z_centered, full_matrices=False)
    z_2d = np.dot(z_centered, vt[:2].T)

    plt.figure(figsize=(10, 8), facecolor="#0d1117")
    ax = plt.gca()
    ax.set_facecolor("#161b22")

    v_log = np.log1p(visits)
    scatter = ax.scatter(
        z_2d[:, 0], z_2d[:, 1],
        c=v_log, cmap="viridis", s=25, alpha=0.75, edgecolors="none"
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Visit Count (log scale)", color="#c9d1d9")
    cbar.ax.yaxis.set_tick_params(color="#c9d1d9")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#c9d1d9")

    plt.title(f"ATLAS TopoMap: Spatial Territory Memory ({len(z_coords)} Locations Mapped)\nPoints represent distinct physical landmarks discovered across the world", color="#ffffff", fontsize=13, pad=15)
    ax.tick_params(colors="#8b949e")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    return out_path


def generate_interactive_memory_dashboard(
    run_dir: Path,
    out_path: Path,
) -> Path:
    """Creates a self-contained interactive HTML Memory Browser."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lift_file = run_dir / "lift.npz"
    gb_json = run_dir / "goal_bank.json"
    gb_npz = run_dir / "goal_bank.npz"
    topo_file = run_dir / "topomap.npz"

    causal_rules = []
    if lift_file.exists():
        lift_data = np.load(lift_file)
        mat = lift_data["lift"]
        for i in range(len(CRAFTER_ACHIEVEMENTS)):
            for j in range(len(CRAFTER_ACHIEVEMENTS)):
                if i != j and mat[i, j] > 1.5:
                    causal_rules.append({
                        "from": CRAFTER_ACHIEVEMENTS[i],
                        "to": CRAFTER_ACHIEVEMENTS[j],
                        "lift": round(float(mat[i, j]), 1),
                    })
    causal_rules.sort(key=lambda x: -x["lift"])

    goals_meta = []
    if gb_json.exists():
        with open(gb_json) as f:
            raw_meta = json.load(f)
        for idx, m in enumerate(raw_meta):
            dim_names = [CRAFTER_ACHIEVEMENTS[d] for d in m.get("event_dims", []) if d < len(CRAFTER_ACHIEVEMENTS)]
            goals_meta.append({
                "id": idx,
                "achievements": dim_names or ["General Landmark"],
                "tries": m.get("n_try", 0),
                "successes": m.get("n_succ", 0),
                "born_iter": m.get("born_iter", 0),
            })

    topo_count = 0
    max_visits = 0
    if topo_file.exists():
        topo_data = np.load(topo_file)
        topo_count = len(topo_data.get("z", []))
        visits = topo_data.get("visits", [])
        max_visits = int(np.max(visits)) if len(visits) > 0 else 0

    rules_json = json.dumps(causal_rules)
    goals_json = json.dumps(goals_meta)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ATLAS Proper Memory Explorer</title>
  <style>
    :root {{
      --bg: #0d1117;
      --card: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --text-dim: #8b949e;
      --blue: #58a6ff;
      --green: #3fb950;
      --gold: #d29922;
      --red: #f85149;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 24px;
    }}
    h1 {{ color: #ffffff; margin-bottom: 6px; }}
    .stats-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin: 20px 0;
    }}
    .stat-box {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      text-align: center;
    }}
    .stat-number {{ font-size: 32px; font-weight: bold; color: var(--blue); }}
    .stat-desc {{ font-size: 13px; color: var(--text-dim); margin-top: 4px; }}
    .section-title {{
      font-size: 20px;
      color: #ffffff;
      margin-top: 32px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }}
    .rule-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 600;
    }}
    .badge-green {{ background: rgba(63, 185, 80, 0.2); color: var(--green); }}
    .badge-gold {{ background: rgba(210, 153, 34, 0.2); color: var(--gold); }}
    .badge-red {{ background: rgba(248, 81, 73, 0.2); color: var(--red); }}
    .badge-blue {{ background: rgba(88, 166, 255, 0.2); color: var(--blue); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 10px 14px;
      text-align: left;
    }}
    th {{ background: #21262d; color: #ffffff; }}
    input {{
      background: var(--card);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px 14px;
      border-radius: 6px;
      width: 280px;
    }}
  </style>
</head>
<body>
  <h1>🧠 ATLAS Memory Browser</h1>
  <p style="color: var(--text-dim);">Inspection of the 3 interconnected memory systems constructed by the agent across 99 iterations.</p>

  <div class="stats-row">
    <div class="stat-box">
      <div class="stat-number">{len(causal_rules)}</div>
      <div class="stat-desc">Learned Causal Rules (PredecessorLift)</div>
    </div>
    <div class="stat-box">
      <div class="stat-number">{len(goals_meta)}</div>
      <div class="stat-desc">Visual Goal Landmarks (GoalBank)</div>
    </div>
    <div class="stat-box">
      <div class="stat-number">{topo_count}</div>
      <div class="stat-desc">Spatial Keyframe Nodes (TopoMap)</div>
    </div>
    <div class="stat-box">
      <div class="stat-number">{max_visits}</div>
      <div class="stat-desc">Peak Landmark Visits</div>
    </div>
  </div>

  <div class="section-title">🔗 1. Causal Memory (PredecessorLift): Cause & Effect Relationships</div>
  <p style="color: var(--text-dim);">Statistical causality mined by ATLAS: Performing Event A increases probability of Event B.</p>
  <div class="card-grid" id="causalGrid"></div>

  <div class="section-title">🎯 2. Visual Goal Memory (GoalBank): Stored Visual Targets</div>
  <p style="color: var(--text-dim);">128-d visual latent vectors stored in memory that the agent navigates toward.</p>
  <input type="text" id="goalSearch" placeholder="Search by achievement (e.g. wood, cow)..." oninput="renderGoals()">
  <table id="goalsTable">
    <thead>
      <tr>
        <th>Goal ID</th>
        <th>Associated Milestone / Item</th>
        <th>Discovered Iteration</th>
        <th>Navigation Attempts</th>
        <th>Success Rate</th>
      </tr>
    </thead>
    <tbody id="goalsBody"></tbody>
  </table>

  <script>
    const rules = {rules_json};
    const goals = {goals_json};

    function renderCausal() {{
      const grid = document.getElementById('causalGrid');
      grid.innerHTML = '';
      rules.slice(0, 12).forEach(r => {{
        const card = document.createElement('div');
        card.className = 'rule-card';
        card.innerHTML = `
          <div>
            <span class="badge badge-green">${{r.from}}</span>
            <span style="margin: 0 8px; color: var(--text-dim);">──▶</span>
            <span class="badge badge-gold">${{r.to}}</span>
          </div>
          <div style="font-weight: bold; color: var(--gold); font-size: 15px;">+${{r.lift}}x</div>
        `;
        grid.appendChild(card);
      }});
    }}

    function renderGoals() {{
      const q = document.getElementById('goalSearch').value.toLowerCase();
      const body = document.getElementById('goalsBody');
      body.innerHTML = '';

      const filtered = goals.filter(g => g.achievements.join(' ').toLowerCase().includes(q));
      filtered.slice(0, 50).forEach(g => {{
        const tr = document.createElement('tr');
        const achs = g.achievements.map(a => `<span class="badge badge-blue">${{a}}</span>`).join(' ');
        const rate = g.tries > 0 ? ((g.successes / g.tries) * 100).toFixed(0) + '%' : '0%';
        tr.innerHTML = `
          <td><strong>Goal #${{g.id}}</strong></td>
          <td>${{achs}}</td>
          <td>Iter ${{g.born_iter}}</td>
          <td>${{g.tries}} times</td>
          <td>${{g.successes}} / ${{g.tries}} (${{rate}})</td>
        `;
        body.appendChild(tr);
      }});
    }}

    renderCausal();
    renderGoals();
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="ATLAS Visual Memory Inspector")
    parser.add_argument("--run-dir", type=str, default=None, help="Path to run directory")
    parser.add_argument("--out-dir", type=str, default="/content/memory_visuals", help="Output directory for plots & dashboard")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else (resolve_drive_root() / "atlas" / "runs" / "atlas_v1")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Reading ATLAS memories from: {run_dir}")

    # 1. Causal plot
    lift_file = run_dir / "lift.npz"
    if lift_file.exists():
        lift_data = np.load(lift_file)
        causal_png = plot_causal_graph(lift_data["lift"], out_dir / "atlas_causal_network.png")
        print(f"   ✅ Exported Causal Graph: {causal_png}")

    # 2. Goal Affordances plot
    gb_json = run_dir / "goal_bank.json"
    gb_npz = run_dir / "goal_bank.npz"
    if gb_json.exists() and gb_npz.exists():
        with open(gb_json) as f:
            meta = json.load(f)
        npz = np.load(gb_npz)
        aff_png = plot_goal_affordances(meta, npz["profiles"], out_dir / "atlas_goal_affordances.png")
        print(f"   ✅ Exported Goal Affordance Matrix: {aff_png}")

    # 3. Spatial TopoMap plot
    topo_file = run_dir / "topomap.npz"
    if topo_file.exists():
        topo = np.load(topo_file)
        if "z" in topo and len(topo["z"]) > 0:
            topo_png = plot_spatial_topomap(topo["z"], topo["visits"], out_dir / "atlas_spatial_territory.png")
            print(f"   ✅ Exported Spatial Territory Map: {topo_png}")

    # 4. Interactive HTML Dashboard
    dash_html = generate_interactive_memory_dashboard(run_dir, out_dir / "atlas_memory_browser.html")
    print(f"   ✅ Exported Interactive Memory Dashboard: {dash_html}")


if __name__ == "__main__":
    main()
