"""ATLAS Game Journal Inspector & Debugger.

Inspects step-by-step decision telemetry recorded during ATLAS self-learning.
Allows finding best & worst games, tracking agent modes (NAV / PROBE / OVERRIDE),
and exporting an interactive HTML replay dashboard.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from atlas.config import REPO_ROOT, resolve_drive_root


def find_default_journal() -> Optional[Path]:
    """Finds game_journal.jsonl across known Drive and repo locations."""
    candidates = [
        REPO_ROOT / "results" / "game_journal.jsonl",
        REPO_ROOT / "atlas" / "runs" / "atlas_v1" / "game_journal.jsonl",
    ]
    drive_root = resolve_drive_root()
    if drive_root is not None:
        candidates.extend([
            drive_root / "atlas" / "runs" / "atlas_v1" / "game_journal.jsonl",
            drive_root / "results" / "game_journal.jsonl",
        ])
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c
    return None


def load_journal(path: Path) -> List[Dict[str, Any]]:
    """Loads all game records from a JSONL journal."""
    if not path.exists():
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def print_summary(games: List[Dict[str, Any]]) -> None:
    """Prints a high-level summary of all games played."""
    if not games:
        print("❌ No games found in journal.")
        return

    n = len(games)
    rewards = [g.get("total_reward", 0.0) for g in games]
    steps = [g.get("length", 0) for g in games]
    deaths = sum(1 for g in games if g.get("died", False))
    overrides = sum(g.get("overrides_count", 0) for g in games)

    # Count achievement frequencies
    ach_counts: Dict[str, int] = {}
    for g in games:
        for ach in g.get("achievements", []):
            ach_counts[ach] = ach_counts.get(ach, 0) + 1

    print("=" * 75)
    print(f"📊 ATLAS GAME JOURNAL SUMMARY ({n} Total Games Recorded)")
    print("=" * 75)
    print(f"  • Survival Steps   : Mean = {sum(steps)/n:.1f} | Max = {max(steps)} | Min = {min(steps)}")
    print(f"  • Rewards          : Mean = {sum(rewards)/n:.2f} | Max = {max(rewards):.1f} | Min = {min(rewards):.1f}")
    print(f"  • Deaths Recorded  : {deaths} / {n} ({deaths/n*100:.1f}%)")
    print(f"  • Safety Overrides : {overrides} total ({overrides/max(1, sum(steps))*100:.2f}% of all actions)")

    print("\n🏆 Achievements Unlocked Across All Games:")
    for ach, count in sorted(ach_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(count / n * 30)
        print(f"   └── {ach:<20}: {count:>4}/{n} ({count/n*100:>5.1f}%) | {bar}")

    # Top 5 Best Games (by reward, then achievements, then length)
    best_games = sorted(
        games,
        key=lambda g: (g.get("total_reward", 0.0), g.get("num_achievements", 0), g.get("length", 0)),
        reverse=True,
    )[:5]

    print("\n🥇 Top 5 BEST Games:")
    for i, g in enumerate(best_games, 1):
        achs = ", ".join(g.get("achievements", [])) or "none"
        print(
            f"   #{i} Game {g.get('game_id')} [Iter {g.get('iter')}] -> "
            f"Reward: {g.get('total_reward')} | Steps: {g.get('length')} | "
            f"Achievements ({g.get('num_achievements')}): {achs}"
        )

    # Top 5 Worst Games (fastest deaths / lowest rewards)
    worst_games = sorted(
        games,
        key=lambda g: (g.get("total_reward", 0.0), g.get("length", 0)),
    )[:5]

    print("\n💀 Top 5 WORST Games (Early Deaths):")
    for i, g in enumerate(worst_games, 1):
        achs = ", ".join(g.get("achievements", [])) or "none"
        print(
            f"   #{i} Game {g.get('game_id')} [Iter {g.get('iter')}] -> "
            f"Reward: {g.get('total_reward')} | Steps: {g.get('length')} | "
            f"Died: {g.get('died')} | Achievements: {achs}"
        )
    print("=" * 75)


def show_game_trace(game: Dict[str, Any], max_steps: Optional[int] = None) -> None:
    """Prints the step-by-step decision telemetry for a single game."""
    print("=" * 80)
    print(f"🔍 STEP-BY-STEP TRACE: {game.get('game_id')} (Iter {game.get('iter')})")
    print(f"Total Reward: {game.get('total_reward')} | Steps: {game.get('length')} | Died: {game.get('died')}")
    print(f"Achievements: {', '.join(game.get('achievements', []))}")
    print("=" * 80)

    steps = game.get("steps", [])
    if max_steps and len(steps) > max_steps:
        steps_to_show = steps[:max_steps]
        truncated = True
    else:
        steps_to_show = steps
        truncated = False

    print(f"{'Step':<6} | {'Action':<15} | {'Mode':<12} | {'Risk':<7} | {'Override':<10} | {'Reward':<7} | {'Events / Notes'}")
    print("-" * 80)

    for st in steps_to_show:
        t = st.get("t", 0)
        a = st.get("a", "")
        mode = st.get("mode", "")
        risk = st.get("risk", 0.0)
        ovr = "⚠️ YES" if st.get("override", False) else "no"
        r = st.get("r", 0.0)
        unlocked = st.get("unlocked", [])
        notes = f"🎉 UNLOCKED: {', '.join(unlocked)}" if unlocked else ""

        print(f"{t+1:<6} | {a:<15} | {mode:<12} | {risk:<7.2f} | {ovr:<10} | {r:<7.1f} | {notes}")

    if truncated:
        print(f"... [Truncated, total steps: {len(steps)}]")
    print("=" * 80)


def export_html_report(games: List[Dict[str, Any]], out_path: Path) -> Path:
    """Exports an interactive standalone HTML dashboard with dark mode and search."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    games_json = json.dumps(games)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ATLAS Game Journal Dashboard</title>
  <style>
    :root {{
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --text-dim: #8b949e;
      --accent: #58a6ff;
      --green: #3fb950;
      --red: #f85149;
      --yellow: #d29922;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 24px;
    }}
    h1 {{ color: #fff; margin-bottom: 8px; }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin: 20px 0;
    }}
    .stat-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      text-align: center;
    }}
    .stat-val {{ font-size: 28px; font-weight: bold; color: var(--accent); }}
    .stat-lbl {{ font-size: 13px; color: var(--text-dim); margin-top: 4px; }}
    .controls {{
      margin: 20px 0;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }}
    input, select, button {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 14px;
    }}
    button {{ cursor: pointer; background: #21262d; }}
    button:hover {{ background: #30363d; }}
    .game-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 12px;
      padding: 14px 18px;
      cursor: pointer;
      transition: border-color 0.15s ease;
    }}
    .game-card:hover {{ border-color: var(--accent); }}
    .game-header {{ display: flex; justify-content: space-between; align-items: center; }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 600;
      margin-right: 6px;
    }}
    .badge-green {{ background: rgba(63, 185, 80, 0.2); color: var(--green); }}
    .badge-red {{ background: rgba(248, 81, 73, 0.2); color: var(--red); }}
    .badge-yellow {{ background: rgba(210, 153, 34, 0.2); color: var(--yellow); }}
    .trace-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-size: 12px;
      display: none;
    }}
    .trace-table th, .trace-table td {{
      border: 1px solid var(--border);
      padding: 6px 10px;
      text-align: left;
    }}
    .trace-table th {{ background: #21262d; }}
    .override-row {{ background: rgba(248, 81, 73, 0.1); }}
    .unlock-row {{ background: rgba(63, 185, 80, 0.15); font-weight: bold; }}
  </style>
</head>
<body>
  <h1>🧭 ATLAS Game Journal Dashboard</h1>
  <p style="color: var(--text-dim);">Live decision traces and behavioral telemetry recorded during self-learning.</p>

  <div class="stats-grid">
    <div class="stat-card"><div class="stat-val" id="totalGames">0</div><div class="stat-lbl">Total Games Played</div></div>
    <div class="stat-card"><div class="stat-val" id="avgReward">0.0</div><div class="stat-lbl">Average Reward</div></div>
    <div class="stat-card"><div class="stat-val" id="avgSteps">0</div><div class="stat-lbl">Average Steps Survived</div></div>
    <div class="stat-card"><div class="stat-val" id="uniqueAchs">0</div><div class="stat-lbl">Unique Achievements</div></div>
  </div>

  <div class="controls">
    <input type="text" id="searchInput" placeholder="Search game, iter, achievement..." oninput="renderGames()">
    <select id="sortSelect" onchange="renderGames()">
      <option value="reward_desc">🏆 Highest Reward (Best)</option>
      <option value="reward_asc">💀 Lowest Reward (Worst)</option>
      <option value="steps_desc">⏳ Longest Survival</option>
      <option value="steps_asc">⚡ Shortest Survival (Early Death)</option>
      <option value="iter_desc">🕒 Newest Iteration</option>
    </select>
    <button onclick="filterAchievement('wood')">🪵 Has Wood</button>
    <button onclick="filterAchievement('cow')">🥩 Has Cow</button>
    <button onclick="filterAchievement('zombie')">🧟 Has Zombie</button>
    <button onclick="resetFilters()">🔄 Reset</button>
  </div>

  <div id="gamesList"></div>

  <script>
    const allGames = {games_json};

    function init() {{
      document.getElementById('totalGames').innerText = allGames.length;
      if (allGames.length > 0) {{
        const totalR = allGames.reduce((acc, g) => acc + (g.total_reward || 0), 0);
        const totalS = allGames.reduce((acc, g) => acc + (g.length || 0), 0);
        document.getElementById('avgReward').innerText = (totalR / allGames.length).toFixed(2);
        document.getElementById('avgSteps').innerText = (totalS / allGames.length).toFixed(0);

        const allAchs = new Set();
        allGames.forEach(g => (g.achievements || []).forEach(a => allAchs.add(a)));
        document.getElementById('uniqueAchs').innerText = allAchs.size;
      }}
      renderGames();
    }}

    function filterAchievement(term) {{
      document.getElementById('searchInput').value = term;
      renderGames();
    }}

    function resetFilters() {{
      document.getElementById('searchInput').value = '';
      renderGames();
    }}

    function renderGames() {{
      const query = document.getElementById('searchInput').value.toLowerCase();
      const sort = document.getElementById('sortSelect').value;

      let filtered = allGames.filter(g => {{
        const text = `${{g.game_id}} iter${{g.iter}} ${{ (g.achievements || []).join(' ') }}`.toLowerCase();
        return text.includes(query);
      }});

      if (sort === 'reward_desc') {{
        filtered.sort((a, b) => (b.total_reward || 0) - (a.total_reward || 0));
      }} else if (sort === 'reward_asc') {{
        filtered.sort((a, b) => (a.total_reward || 0) - (b.total_reward || 0));
      }} else if (sort === 'steps_desc') {{
        filtered.sort((a, b) => (b.length || 0) - (a.length || 0));
      }} else if (sort === 'steps_asc') {{
        filtered.sort((a, b) => (a.length || 0) - (b.length || 0));
      }} else if (sort === 'iter_desc') {{
        filtered.sort((a, b) => (b.iter || 0) - (a.iter || 0));
      }}

      const container = document.getElementById('gamesList');
      container.innerHTML = '';

      filtered.forEach((g, idx) => {{
        const card = document.createElement('div');
        card.className = 'game-card';

        const achBadges = (g.achievements || []).map(a => `<span class="badge badge-green">${{a}}</span>`).join(' ') || '<span style="color:var(--text-dim)">None</span>';
        const deathBadge = g.died ? '<span class="badge badge-red">Died</span>' : '<span class="badge badge-yellow">Truncated</span>';
        const overrideBadge = g.overrides_count > 0 ? `<span class="badge badge-yellow">⚠️ ${{g.overrides_count}} Overrides</span>` : '';

        card.innerHTML = `
          <div class="game-header" onclick="toggleTrace('trace_${{idx}}')">
            <div>
              <strong>${{g.game_id}}</strong> (Iter ${{g.iter}})
              <span style="margin-left: 14px;">${{deathBadge}} ${{overrideBadge}}</span>
              <div style="margin-top: 6px; font-size: 13px;">Achievements: ${{achBadges}}</div>
            </div>
            <div style="text-align: right;">
              <span style="font-size: 18px; font-weight: bold; color: var(--accent);">+${{g.total_reward}}</span>
              <div style="font-size: 12px; color: var(--text-dim);">${{g.length}} steps</div>
            </div>
          </div>
          <table class="trace-table" id="trace_${{idx}}">
            <thead>
              <tr>
                <th>Step</th>
                <th>Action</th>
                <th>Mode</th>
                <th>Risk</th>
                <th>Override</th>
                <th>Reward</th>
                <th>Unlocked</th>
              </tr>
            </thead>
            <tbody>
              ${{ (g.steps || []).map(st => `
                <tr class="${{st.override ? 'override-row' : ''}} ${{st.unlocked && st.unlocked.length ? 'unlock-row' : ''}}">
                  <td>${{st.t + 1}}</td>
                  <td><code>${{st.a}}</code></td>
                  <td>${{st.mode}}</td>
                  <td>${{st.risk}}</td>
                  <td>${{st.override ? '⚠️ YES' : 'no'}}</td>
                  <td>${{st.r}}</td>
                  <td>${{st.unlocked ? st.unlocked.join(', ') : ''}}</td>
                </tr>
              `).join('') }}
            </tbody>
          </table>
        `;
        container.appendChild(card);
      }});
    }}

    function toggleTrace(id) {{
      const table = document.getElementById(id);
      if (table.style.display === 'table') {{
        table.style.display = 'none';
      }} else {{
        table.style.display = 'table';
      }}
    }}

    init();
  </script>
</body>
</html>
"""
    out_path.write_text(html_content, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="ATLAS Game Journal Inspector")
    parser.add_argument("--journal", type=str, default=None, help="Path to game_journal.jsonl")
    parser.add_argument("--game", type=str, default=None, help="Game ID to inspect step-by-step")
    parser.add_argument("--best", type=int, default=None, help="Show top N best games")
    parser.add_argument("--worst", type=int, default=None, help="Show top N worst games")
    parser.add_argument("--achievement", type=str, default=None, help="Filter games unlocking this achievement")
    parser.add_argument("--html", type=str, default=None, help="Export interactive HTML dashboard")
    args = parser.parse_args()

    journal_path = Path(args.journal) if args.journal else find_default_journal()
    if journal_path is None or not journal_path.exists():
        print("❌ Could not find game_journal.jsonl. Provide explicit path with --journal <path>.")
        return

    print(f"📖 Reading journal: {journal_path}")
    games = load_journal(journal_path)
    if not games:
        print("❌ Journal is empty.")
        return

    if args.game:
        matching = [g for g in games if g.get("game_id") == args.game]
        if matching:
            show_game_trace(matching[0])
        else:
            print(f"❌ Game {args.game} not found in journal.")
        return

    if args.achievement:
        filtered = [g for g in games if args.achievement in g.get("achievements", [])]
        print(f"\n🔍 Games unlocking '{args.achievement}' ({len(filtered)} / {len(games)}):")
        for g in filtered:
            print(f"   • {g.get('game_id')} [Iter {g.get('iter')}] -> Steps: {g.get('length')} | Reward: {g.get('total_reward')}")
        return

    print_summary(games)

    if args.html:
        out = export_html_report(games, Path(args.html))
        print(f"\n✅ Exported HTML dashboard to: {out.resolve()}")


if __name__ == "__main__":
    main()
