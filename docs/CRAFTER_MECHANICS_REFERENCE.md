# Crafter Engine Mechanics Reference (ground truth)

> Verified against the installed engine (`.venv/lib/python3.11/site-packages/crafter/` — env.py, engine.py, objects.py, worldgen.py, data.yaml) plus `envs/crafter_wrapper.py` and `eval/metrics.py`. This is the audit basis for every design decision. The agent itself never reads any of this; this document exists so HUMANS can check whether what the agent learned matches reality.

## 1. World

- 64×64 tiles; materials: water, grass, stone, path, sand, tree, lava, coal, iron, diamond, table, furnace. Walkable: grass, path, sand (player may step into lava → instant death).
- Generation: OpenSimplex noise; grass guaranteed near spawn (32,32). Trees ≈ 20% of grassland. Diamond: 0.6% of eligible mountain tiles. Lava deep in mountains; caves → path.
- Initial objects: cows p=1.5% (dist>3), zombies p=0.7% (dist>10), skeletons p=5% on path tiles.
- **Day/night cycle = 300 steps**; daylight = 1−|cos(π·((step/300)%1+0.3))|³. Night (daylight<0.5) ≈ **steps 148–272** of each cycle. Night changes rendering (blue tint, vignette noise; grayscale while asleep) and monster spawns — nothing else.

## 2. Player

- Start center (32,32); vitals health/food/drink/energy = 9/9/9/9 (max 9); all items max 9.
- Decay (awake → asleep): food −1 every 26 → 51 steps; drink −1 every 21 → 41; energy −1 every 31 awake, asleep +1 per 11 steps. Full→empty awake: ~234 / ~189 / ~279 steps.
- Health: regen +1 per 26 steps (13 asleep) iff food>0 AND drink>0 AND (energy>0 or sleeping); otherwise −1 per 16 steps. **All starvation/dehydration/exhaustion deaths act through this single channel.** Death = health ≤ 0. Lava = instant health=0.
- Damage: zombie melee 2 (**7 vs sleeping player**), arrow 2. No fall damage, no drowning.
- **Sleep is irrevocable** (only energy=9 → `wake_up`, or damage wakes): halves hunger/thirst accrual, doubles regen, but a ~99-step commitment at 7 dmg/night-hit risk.

## 3. Items, recipes, tools

- Collect (`do` on faced material): tree→wood 1; grass→sapling **10% chance**; water→drink +1 **and thirst counter reset — free, repeatable**; stone/coal need wood_pickaxe; iron needs stone_pickaxe; diamond needs iron_pickaxe.
- Place: table {wood:2} on empty grass/sand/path; furnace {stone:4}; plant {sapling:1} grass only; **stone bridges water/lava**.
- Make (3×3 must contain `nearby`): wood_pickaxe {wood:1, table}; stone_pickaxe {wood:1,stone:1,table}; iron tools {wood:1,coal:1,iron:1, **table+furnace**}; wood_sword {wood:1,table}; stone_sword {wood:1,stone:1,table}.
- **Tools NEVER break or wear** — one pickaxe of a tier unlocks unlimited mining; swords only change damage (1/2/3/5 for none/wood/stone/iron).

## 4. Actions (17)

`noop; move_left/right/up/down; do; sleep; place_stone/table/furnace/plant; make_wood/stone/iron_pickaxe; make_wood/stone/iron_sword`.
`do` is overloaded: attack object if faced tile has one (fists 1, swords 2/3/5), else collect material, else drink if water. Failed moves still turn the player (facing persists). Objects update only within **L1 distance < 18** of the player.

## 5. Objects & monsters

| Object | HP | Facts |
|---|---|---|
| Tree | – | → 1 wood, tile becomes grass. **No regrowth anywhere.** |
| Grass | – | → sapling 10%; grass itself never consumed. |
| Cow | 3 | random walk; killed by `do` → +6 food, resets hunger. |
| Zombie | 5 | chases if dist ≤8 (90%/step); melee 2, cooldown 5; spawns **at night** (chunk target 3.5−3·daylight, checked every 10 steps); despawn possible even adjacent. |
| Skeleton | 3 | **spawns day AND night** (1–2 per path-chunk, dist ≥7); shoots arrows (p=0.5, reload 4) from dist ≤5; flees if dist ≤3. |
| Arrow | – | 2 dmg; **converts table/furnace tiles to path** (can destroy your crafting station). |
| Plant | 1 | ripens after ~301 *nearby* updates (proximity-gated); ripe → +4 food, re-ripenable; trampled by adjacent monsters/cows. |

## 6. Rewards, achievements, observation

- reward = (health − last_health)/10 **+1.0 once per episode per achievement** (all 22 reset each episode). No survival bonus.
- Obs 64×64×3: 9×7-tile egocentric viewport + a **2-tile inventory strip** (icon + digit for every item/vital ≥ 1) — the pixel source the state readout learns counts from. Night tint/vignette; sleeping = grayscale.
- Engine termination: health ≤ 0 or 10000 steps; the eval loop cuts at its own `max_steps` (1000).

## 7. Subtleties that matter to a learning agent

1. Drinking is free and repeatable — maintenance, not achievement economics.
2. Tools never break — tier unlocks are permanent within the episode.
3. Gating chain: wood → table → wood pickaxe → stone/coal → stone pickaxe + furnace → iron → iron pickaxe → diamond. Iron also needs coal.
4. Sleep is a commitment; zombies hit 7 vs sleepers. Energy forces sleep ~2–3 times per 1000-step episode.
5. Plants only ripen while the agent stays within L1 < 18 — "nurture by proximity."
6. Skeleton arrows destroy tables/furnaces — crafting stations are destructible.
7. Cow kills = 3 `do` hits (fists); `eat_cow` resets hunger.
8. Simultaneous unlocks still pay +1 total once.
