# Online Causal Memory Bank Review Report

**Source Bank File**: `dual_store_memory_bank_online.json`  
**Total Prototypes**: 447  
**Online-Created Cards**: 184  
**Online-Reinforced Cards**: 222  
**Causal Edges Involving Online Cards**: 5706  

---

## 1. Discovered Online Cards (Hypotheses & Confirmed)

| Card ID | Action | Event Achievement | Status | Wilson Conf | Online Succ / Fail | Needs | Produces | Consumes |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| `card_a5_e2_o0` | `do` (5) | **collect_drink** | `hypothesis` | 0.051 | **1.0** / 1.0 | — | 0.6 drink | — |
| `card_a3_e21_o0` | `move_up` (3) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.3 drink |
| `card_a5_e15_o0` | `do` (5) | **make_wood_pickaxe** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 0.9 wood | — |
| `card_a8_e20_o0` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 0.5 health | — |
| `card_a10_e18_o0` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 0.4 drink | 0.3 sapling, 1.9 health |
| `card_a5_e2_o1` | `do` (5) | **collect_drink** | `hypothesis` | 0.050 | **1.0** / 2.0 | — | 0.5 food, 0.7 drink | — |
| `card_a5_e6_o0` | `do` (5) | **collect_wood** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 1.0 wood, 0.3 sapling | — |
| `card_a6_e21_o0` | `sleep` (6) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | — |
| `card_a8_e20_o1` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.6 wood | 0.3 food | 1.6 wood |
| `card_a5_e8_o0` | `do` (5) | **defeat_zombie** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | — | 1.8 health, 0.3 drink |
| `card_a10_e18_o1` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.164 | **2.0** / 0.0 | 1.0 sapling | — | 1.0 sapling |
| `card_a5_e6_o1` | `do` (5) | **collect_wood** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.9 wood | — |
| `card_a8_e20_o2` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.5 wood | — | 1.5 wood |
| `card_a5_e2_o2` | `do` (5) | **collect_drink** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 1.0 wood | 0.3 drink |
| `card_a10_e21_o0` | `place_plant` (10) | **wake_up** | `hypothesis` | 0.050 | **1.0** / 2.0 | 1.2 sapling | 0.4 health | 1.2 sapling, 0.5 food |
| `card_a8_e15_o0` | `place_table` (8) | **make_wood_pickaxe** | `hypothesis` | 0.062 | **0.5** / 0.0 | 2.1 wood | — | 2.1 wood |
| `card_a5_e6_o2` | `do` (5) | **collect_wood** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.9 wood | — |
| `card_a13_e21_o0` | `make_iron_pickaxe` (13) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.4 sapling, 1.0 food, 1.2 drink |
| `card_a5_e6_o3` | `do` (5) | **collect_wood** | `hypothesis` | 0.051 | **1.0** / 1.0 | — | 0.9 wood, 0.4 drink | — |
| `card_a10_e21_o1` | `place_plant` (10) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.4 food, 0.5 drink |
| `card_a10_e18_o2` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.230 | **4.0** / 0.0 | 1.2 sapling | — | 1.2 sapling |
| `card_a8_e16_o0` | `place_table` (8) | **make_wood_sword** | `hypothesis` | 0.062 | **0.5** / 0.0 | 1.7 wood | — | 1.7 wood, 0.3 health, 0.3 drink |
| `card_a5_e4_o1` | `do` (5) | **collect_sapling** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.8 sapling | 0.4 food, 0.4 drink |
| `card_a10_e18_o3` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.106 | **1.0** / 0.0 | 0.8 sapling | — | 0.8 sapling |
| `card_a16_e21_o0` | `make_iron_sword` (16) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | — |
| `card_a5_e6_o4` | `do` (5) | **collect_wood** | `hypothesis` | 0.050 | **1.0** / 6.0 | — | 0.9 wood | — |
| `card_a5_e4_o2` | `do` (5) | **collect_sapling** | `hypothesis` | 0.050 | **0.5** / 2.0 | — | 1.2 sapling | — |
| `card_a2_e21_o0` | `move_right` (2) | **wake_up** | `hypothesis` | 0.051 | **1.0** / 1.0 | — | — | 0.4 drink |
| `card_a10_e18_o4` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.051 | **1.0** / 1.0 | 0.9 sapling | — | 0.9 sapling |
| `card_a14_e16_o0` | `make_wood_sword` (14) | **make_wood_sword** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.0 wood | — | 1.0 wood, 0.3 health, 0.4 food, 1.2 energy |
| `card_a5_e5_o0` | `do` (5) | **collect_stone** | `hypothesis` | 0.050 | **0.5** / 1.0 | — | 0.3 food, 0.7 drink | — |
| `card_a13_e21_o1` | `make_iron_pickaxe` (13) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.5 food, 0.3 energy |
| `card_a11_e15_o0` | `make_wood_pickaxe` (11) | **make_wood_pickaxe** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.5 wood, 1.2 health, 0.7 food, 0.6 drink |
| `card_a10_e21_o2` | `place_plant` (10) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.5 drink |
| `card_a10_e18_o5` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.1 sapling | — | 1.1 sapling |
| `card_a5_e5_o1` | `do` (5) | **collect_stone** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.9 wood, 0.3 food | — |
| `card_a8_e20_o3` | `place_table` (8) | **place_table** | `hypothesis` | 0.203 | **3.0** / 0.0 | 2.2 wood | — | 2.2 wood, 0.3 health |
| `card_a16_e21_o1` | `make_iron_sword` (16) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.5 sapling, 0.6 food, 0.3 drink |
| `card_a5_e5_o2` | `do` (5) | **collect_stone** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 1.1 wood, 0.5 sapling | — |
| `card_a8_e20_o4` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.8 wood | 0.4 sapling, 0.5 food | 1.8 wood |
| `card_a5_e2_o3` | `do` (5) | **collect_drink** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 0.6 drink | — |
| `card_a5_e4_o3` | `do` (5) | **collect_sapling** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.9 wood | — |
| `card_a8_e20_o5` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.8 wood | 0.4 food | 1.8 wood |
| `card_a5_e16_o0` | `do` (5) | **make_wood_sword** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 1.7 sapling | 0.3 health, 0.3 drink, 0.3 energy |
| `card_a10_e18_o6` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.4 sapling, 0.6 drink |
| `card_a2_e21_o1` | `move_right` (2) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.3 wood, 0.3 sapling |
| `card_a11_e15_o1` | `make_wood_pickaxe` (11) | **make_wood_pickaxe** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 1.3 sapling, 0.6 food | 0.5 wood, 0.5 health, 0.4 drink, 0.5 energy |
| `card_a5_e6_o5` | `do` (5) | **collect_wood** | `hypothesis` | 0.086 | **4.0** / 6.0 | — | 0.8 wood | — |
| `card_a4_e21_o0` | `move_down` (4) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.3 food, 0.5 drink |
| `card_a8_e20_o6` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 2.4 wood | 0.5 sapling, 0.3 food | 2.4 wood |
| `card_a10_e18_o7` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.106 | **1.0** / 0.0 | 0.8 sapling | — | 0.8 sapling |
| `card_a5_e21_o0` | `do` (5) | **wake_up** | `hypothesis` | 0.062 | **2.0** / 3.0 | — | 1.1 sapling | 0.3 food, 0.4 drink |
| `card_a8_e20_o7` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.0 wood | 0.5 food | 1.0 wood |
| `card_a14_e16_o1` | `make_wood_sword` (14) | **make_wood_sword** | `hypothesis` | 0.062 | **0.5** / 0.0 | 1.2 wood, 0.9 sapling | 0.5 food | 1.2 wood, 0.9 sapling, 0.4 energy |
| `card_a13_e21_o2` | `make_iron_pickaxe` (13) | **wake_up** | `hypothesis` | 0.051 | **1.0** / 1.0 | — | — | 0.3 drink |
| `card_a8_e20_o8` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.7 wood | 0.4 food, 0.4 drink | 1.7 wood |
| `card_a5_e6_o6` | `do` (5) | **collect_wood** | `hypothesis` | 0.203 | **3.0** / 0.0 | — | 0.9 wood | 0.3 energy |
| `card_a11_e21_o0` | `make_wood_pickaxe` (11) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | — |
| `card_a5_e2_o4` | `do` (5) | **collect_drink** | `hypothesis` | 0.051 | **1.0** / 1.0 | — | 0.3 food, 0.7 drink | — |
| `card_a8_e15_o1` | `place_table` (8) | **make_wood_pickaxe** | `hypothesis` | 0.062 | **0.5** / 0.0 | 2.2 wood | 0.4 food | 2.2 wood |
| `card_a5_e6_o7` | `do` (5) | **collect_wood** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.8 drink | — |
| `card_a13_e21_o3` | `make_iron_pickaxe` (13) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.7 food, 0.8 drink |
| `card_a5_e15_o1` | `do` (5) | **make_wood_pickaxe** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 1.0 wood | — |
| `card_a5_e9_o0` | `do` (5) | **eat_cow** | `hypothesis` | 0.050 | **0.5** / 1.0 | — | 0.8 wood | — |
| `card_a5_e15_o2` | `do` (5) | **make_wood_pickaxe** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | — | — |
| `card_a5_e4_o4` | `do` (5) | **collect_sapling** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.7 wood | — |
| `card_a5_e2_o5` | `do` (5) | **collect_drink** | `hypothesis` | 0.164 | **2.0** / 0.0 | — | 0.3 wood, 0.6 drink | 0.4 energy |
| `card_a16_e21_o2` | `make_iron_sword` (16) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | — |
| `card_a5_e15_o3` | `do` (5) | **make_wood_pickaxe** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 1.5 sapling | — |
| `card_a5_e6_o8` | `do` (5) | **collect_wood** | `hypothesis` | 0.051 | **1.0** / 1.0 | — | 0.8 wood, 0.3 drink | 0.6 sapling |
| `card_a8_e20_o9` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 2.3 wood | 0.4 sapling | 2.3 wood |
| `card_a5_e6_o9` | `do` (5) | **collect_wood** | `hypothesis` | 0.203 | **3.0** / 0.0 | — | 0.9 wood | — |
| `card_a11_e21_o1` | `make_wood_pickaxe` (11) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | — |
| `card_a8_e20_o10` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 2.1 wood | 0.5 food | 2.1 wood |
| `card_a5_e6_o10` | `do` (5) | **collect_wood** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 1.0 wood | — |
| `card_a16_e21_o3` | `make_iron_sword` (16) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.3 drink |
| `card_a5_e4_o5` | `do` (5) | **collect_sapling** | `hypothesis` | 0.050 | **0.5** / 6.0 | — | 0.9 sapling, 0.3 food | — |
| `card_a8_e20_o11` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.8 wood | 0.3 food | 1.8 wood |
| `card_a11_e15_o2` | `make_wood_pickaxe` (11) | **make_wood_pickaxe** | `hypothesis` | 0.106 | **1.0** / 0.0 | 0.9 wood | — | 0.9 wood, 2.2 health, 0.8 food, 0.5 drink, 0.6 energy |
| `card_a5_e16_o1` | `do` (5) | **make_wood_sword** | `hypothesis` | 0.050 | **0.5** / 1.0 | — | — | 0.4 sapling, 2.1 health, 0.5 drink, 0.6 energy |
| `card_a5_e6_o11` | `do` (5) | **collect_wood** | `hypothesis` | 0.125 | **4.0** / 3.0 | — | 0.7 wood, 0.3 sapling | — |
| `card_a8_e20_o12` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 2.0 wood | 0.4 food | 2.0 wood |
| `card_a5_e4_o6` | `do` (5) | **collect_sapling** | `hypothesis` | 0.050 | **0.5** / 10.0 | — | 0.8 sapling | — |
| `card_a10_e18_o8` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.106 | **1.0** / 0.0 | 0.8 sapling | — | 0.8 sapling |
| `card_a13_e21_o4` | `make_iron_pickaxe` (13) | **wake_up** | `hypothesis` | 0.205 | **5.0** / 1.0 | — | — | — |
| `card_a8_e20_o13` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.5 wood | 0.5 food, 0.3 drink | 1.5 wood |
| `card_a10_e18_o9` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.106 | **1.0** / 0.0 | 0.8 sapling | 0.7 drink | 0.8 sapling, 0.6 energy |
| `card_a8_e20_o14` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.6 wood | 0.3 food | 1.6 wood |
| `card_a10_e21_o3` | `place_plant` (10) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.2 sapling | — | 1.2 sapling, 1.0 food, 0.5 drink |
| `card_a5_e4_o8` | `do` (5) | **collect_sapling** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.8 wood, 0.3 food | — |
| `card_a15_e21_o0` | `make_stone_sword` (15) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | — |
| `card_a5_e6_o12` | `do` (5) | **collect_wood** | `hypothesis` | 0.050 | **1.0** / 3.0 | — | 1.0 sapling | — |
| `card_a5_e4_o9` | `do` (5) | **collect_sapling** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.9 wood, 0.3 food | — |
| `card_a5_e8_o1` | `do` (5) | **defeat_zombie** | `hypothesis` | 0.050 | **0.5** / 1.0 | — | — | 0.7 energy |
| `card_a5_e6_o13` | `do` (5) | **collect_wood** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 0.9 wood | — |
| `card_a4_e21_o1` | `move_down` (4) | **wake_up** | `hypothesis` | 0.106 | **2.0** / 1.0 | — | — | 0.3 drink |
| `card_a5_e4_o10` | `do` (5) | **collect_sapling** | `hypothesis` | 0.050 | **0.5** / 2.0 | — | 1.3 sapling | — |
| `card_a5_e2_o6` | `do` (5) | **collect_drink** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 0.3 food, 1.2 drink | — |
| `card_a5_e15_o4` | `do` (5) | **make_wood_pickaxe** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 1.0 wood | — |
| `card_a8_e20_o15` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 2.0 wood | 0.6 sapling, 0.3 food | 2.0 wood, 0.9 drink |
| `card_a11_e16_o0` | `make_wood_pickaxe` (11) | **make_wood_sword** | `hypothesis` | 0.062 | **0.5** / 0.0 | 1.3 wood | 0.8 sapling, 0.6 food | 1.3 wood |
| `card_a5_e5_o3` | `do` (5) | **collect_stone** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.5 wood, 1.4 sapling, 1.0 drink | 0.3 health, 0.6 energy |
| `card_a5_e6_o14` | `do` (5) | **collect_wood** | `hypothesis` | 0.164 | **2.0** / 0.0 | — | 0.8 wood | — |
| `card_a8_e20_o16` | `place_table` (8) | **place_table** | `hypothesis` | 0.203 | **3.0** / 0.0 | 1.6 wood | — | 1.6 wood |
| `card_a5_e7_o0` | `do` (5) | **defeat_skeleton** | `hypothesis` | 0.050 | **0.5** / 1.0 | — | 0.6 health, 1.6 food, 1.3 drink, 0.3 energy | — |
| `card_a5_e6_o15` | `do` (5) | **collect_wood** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 0.8 wood | — |
| `card_a5_e4_o11` | `do` (5) | **collect_sapling** | `hypothesis` | 0.050 | **0.5** / 3.0 | — | 1.5 sapling | — |
| `card_a10_e18_o10` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.5 sapling | — | 1.5 sapling |
| `card_a8_e20_o17` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.7 wood | — | 1.7 wood |
| `card_a9_e21_o0` | `place_furnace` (9) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.3 drink |
| `card_a5_e15_o5` | `do` (5) | **make_wood_pickaxe** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 1.0 wood | — |
| `card_a8_e20_o18` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 2.1 wood | 0.6 sapling | 2.1 wood, 0.7 health, 0.3 food, 0.4 drink, 0.4 energy |
| `card_a5_e6_o16` | `do` (5) | **collect_wood** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 0.8 wood | — |
| `card_a12_e21_o0` | `make_stone_pickaxe` (12) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.3 drink |
| `card_a8_e20_o19` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.4 wood | — | 1.4 wood |
| `card_a5_e4_o12` | `do` (5) | **collect_sapling** | `hypothesis` | 0.050 | **0.5** / 9.0 | — | 0.8 sapling | — |
| `card_a10_e18_o11` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.106 | **1.0** / 0.0 | 0.8 sapling | — | 0.8 sapling |
| `card_a11_e15_o3` | `make_wood_pickaxe` (11) | **make_wood_pickaxe** | `hypothesis` | 0.062 | **0.5** / 0.0 | 2.1 wood | 0.5 sapling | 2.1 wood, 5.4 health, 2.8 food, 2.5 drink, 3.3 energy |
| `card_a5_e5_o4` | `do` (5) | **collect_stone** | `hypothesis` | 0.050 | **0.5** / 3.0 | — | 1.3 health, 2.5 food, 0.9 energy | 1.0 drink |
| `card_a5_e6_o17` | `do` (5) | **collect_wood** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 0.8 wood | — |
| `card_a5_e4_o13` | `do` (5) | **collect_sapling** | `hypothesis` | 0.050 | **0.5** / 2.0 | — | 0.9 sapling | — |
| `card_a10_e18_o12` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.106 | **1.0** / 0.0 | 0.9 sapling | — | 0.9 sapling |
| `card_a8_e20_o20` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.9 wood | 0.4 food | 1.9 wood |
| `card_a13_e21_o5` | `make_iron_pickaxe` (13) | **wake_up** | `hypothesis` | 0.164 | **2.0** / 0.0 | — | — | — |
| `card_a11_e16_o1` | `make_wood_pickaxe` (11) | **make_wood_sword** | `hypothesis` | 0.062 | **0.5** / 0.0 | 1.0 wood | 0.5 sapling, 0.5 food | 1.0 wood |
| `card_a5_e5_o5` | `do` (5) | **collect_stone** | `hypothesis` | 0.050 | **0.5** / 4.0 | — | — | 0.3 sapling, 2.6 health, 1.1 food, 0.9 drink, 1.7 energy |
| `card_a5_e6_o18` | `do` (5) | **collect_wood** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 0.8 wood | — |
| `card_a8_e20_o21` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.5 wood | — | 1.5 wood |
| `card_a5_e4_o14` | `do` (5) | **collect_sapling** | `hypothesis` | 0.050 | **0.5** / 2.0 | — | 0.9 sapling, 0.4 food | — |
| `card_a10_e18_o13` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.2 sapling | — | 1.2 sapling |
| `card_a8_e21_o0` | `place_table` (8) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.9 drink |
| `card_a5_e2_o7` | `do` (5) | **collect_drink** | `hypothesis` | 0.051 | **1.0** / 1.0 | — | 0.3 food, 0.6 drink | — |
| `card_a5_e4_o15` | `do` (5) | **collect_sapling** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.9 sapling | — |
| `card_a5_e15_o6` | `do` (5) | **make_wood_pickaxe** | `hypothesis` | 0.050 | **0.5** / 1.0 | — | 0.9 wood | 0.3 drink |
| `card_a10_e19_o0` | `place_plant` (10) | **place_stone** | `hypothesis` | 0.062 | **0.5** / 0.0 | 0.9 sapling | — | 0.9 sapling, 0.5 food, 0.3 drink |
| `card_a8_e20_o22` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 2.1 wood | 0.4 sapling | 2.1 wood, 1.2 health, 0.4 food, 0.3 energy |
| `card_a14_e21_o0` | `make_wood_sword` (14) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 0.5 health | 0.8 food, 0.6 drink |
| `card_a8_e20_o23` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 2.1 wood | 0.3 food | 2.1 wood |
| `card_a5_e4_o16` | `do` (5) | **collect_sapling** | `hypothesis` | 0.050 | **0.5** / 1.0 | — | 0.7 sapling | — |
| `card_a5_e6_o19` | `do` (5) | **collect_wood** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 0.8 wood, 0.3 sapling | — |
| `card_a8_e20_o24` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 2.2 wood | 0.6 food | 2.2 wood |
| `card_a5_e2_o8` | `do` (5) | **collect_drink** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 1.1 drink | — |
| `card_a5_e4_o17` | `do` (5) | **collect_sapling** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.3 wood, 0.5 sapling | 0.3 energy |
| `card_a8_e20_o25` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 2.0 wood | 0.3 sapling, 0.5 food | 2.0 wood |
| `card_a5_e2_o9` | `do` (5) | **collect_drink** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 0.6 food, 0.8 drink | — |
| `card_a13_e21_o6` | `make_iron_pickaxe` (13) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | — |
| `card_a11_e15_o4` | `make_wood_pickaxe` (11) | **make_wood_pickaxe** | `hypothesis` | 0.062 | **0.5** / 0.0 | 1.1 wood | 0.5 food, 0.3 drink | 1.1 wood |
| `card_a10_e12_o0` | `place_plant` (10) | **make_iron_sword** | `hypothesis` | 0.062 | **0.5** / 0.0 | 1.4 sapling | — | 1.4 sapling, 0.3 drink |
| `card_a5_e0_o0` | `do` (5) | **collect_coal** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.7 health, 0.4 food | 0.5 drink |
| `card_a5_e4_o18` | `do` (5) | **collect_sapling** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.9 sapling | — |
| `card_a6_e21_o1` | `sleep` (6) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 1.0 food, 1.1 drink |
| `card_a8_e20_o26` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 2.0 wood | 0.6 sapling, 0.6 food | 2.0 wood |
| `card_a14_e16_o2` | `make_wood_sword` (14) | **make_wood_sword** | `hypothesis` | 0.062 | **0.5** / 0.0 | 0.8 wood | — | 0.8 wood, 0.3 drink, 0.4 energy |
| `card_a5_e15_o7` | `do` (5) | **make_wood_pickaxe** | `hypothesis` | 0.050 | **0.5** / 3.0 | — | — | 0.5 health, 0.7 drink |
| `card_a10_e18_o14` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.2 sapling | — | 1.2 sapling |
| `card_a14_e21_o1` | `make_wood_sword` (14) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.5 sapling, 0.6 food, 1.0 drink |
| `card_a5_e6_o20` | `do` (5) | **collect_wood** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 1.1 wood | — |
| `card_a5_e6_o21` | `do` (5) | **collect_wood** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 1.0 wood | — |
| `card_a5_e4_o19` | `do` (5) | **collect_sapling** | `hypothesis` | 0.050 | **0.5** / 4.0 | — | 0.7 sapling | 0.5 health |
| `card_a10_e18_o15` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.1 sapling | 0.7 health | 1.1 sapling |
| `card_a5_e21_o1` | `do` (5) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.3 food, 0.4 drink |
| `card_a8_e20_o27` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.7 wood | — | 1.7 wood |
| `card_a1_e21_o0` | `move_left` (1) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | — |
| `card_a8_e20_o28` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.6 wood | — | 1.6 wood, 0.8 drink |
| `card_a5_e15_o8` | `do` (5) | **make_wood_pickaxe** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.5 drink | — |
| `card_a10_e15_o0` | `place_plant` (10) | **make_wood_pickaxe** | `hypothesis` | 0.062 | **0.5** / 0.0 | 1.1 sapling | — | 1.1 sapling |
| `card_a6_e21_o2` | `sleep` (6) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | — | 0.4 sapling, 0.7 food, 0.7 drink |
| `card_a5_e4_o20` | `do` (5) | **collect_sapling** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.9 wood, 0.5 food | — |
| `card_a5_e4_o21` | `do` (5) | **collect_sapling** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.9 wood | — |
| `card_a5_e15_o9` | `do` (5) | **make_wood_pickaxe** | `hypothesis` | 0.050 | **0.5** / 5.0 | — | 0.8 sapling | 0.9 energy |
| `card_a10_e18_o16` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.106 | **1.0** / 0.0 | 0.8 sapling | — | 0.8 sapling, 0.3 food |
| `card_a8_e20_o29` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.4 wood | 0.5 food, 0.6 drink | 1.4 wood |
| `card_a5_e8_o2` | `do` (5) | **defeat_zombie** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | — | 0.4 food |
| `card_a8_e20_o30` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 1.6 wood | — | 1.6 wood |
| `card_a5_e2_o10` | `do` (5) | **collect_drink** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 0.3 food, 0.8 drink | — |
| `card_a5_e4_o22` | `do` (5) | **collect_sapling** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 0.7 wood | 0.7 energy |
| `card_a5_e15_o10` | `do` (5) | **make_wood_pickaxe** | `hypothesis` | 0.050 | **0.5** / 13.0 | — | 0.9 sapling | — |
| `card_a10_e18_o17` | `place_plant` (10) | **place_plant** | `hypothesis` | 0.106 | **1.0** / 0.0 | 0.9 sapling | — | 0.9 sapling |
| `card_a8_e12_o0` | `place_table` (8) | **make_iron_sword** | `hypothesis` | 0.062 | **0.5** / 0.0 | 1.7 wood | 0.4 food | 1.7 wood |
| `card_a5_e2_o11` | `do` (5) | **collect_drink** | `hypothesis` | 0.106 | **1.0** / 0.0 | — | 0.5 food, 1.1 drink | — |
| `card_a5_e8_o3` | `do` (5) | **defeat_zombie** | `hypothesis` | 0.050 | **1.0** / 3.0 | — | — | 0.5 health |
| `card_a5_e16_o2` | `do` (5) | **make_wood_sword** | `hypothesis` | 0.062 | **0.5** / 0.0 | — | 1.0 food, 0.8 drink | — |
| `card_a8_e20_o31` | `place_table` (8) | **place_table** | `hypothesis` | 0.106 | **1.0** / 0.0 | 2 wood | 0.9 sapling | 2 wood, 0.4 health, 0.4 food, 0.9 drink, 0.3 energy |
| `card_a8_e21_o1` | `place_table` (8) | **wake_up** | `hypothesis` | 0.106 | **1.0** / 0.0 | 2 wood | 0.9 sapling | 2 wood, 0.4 health, 0.4 food, 0.9 drink, 0.3 energy |

---

## 2. Offline Cards Reinforced During Gameplay

Cards originally formed offline whose Bayesian confidence was sharpened by live online confirmations or near-miss denials:

| Card ID | Action | Event | Offline S/F | Online S/F | New Confidence | Needs | Produces |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `card_a2_e21_c0` | `move_right` | wake_up | 2/4 | **+0.0** / **-1.0** | **0.125** | — | — |
| `card_a3_enone_c8` | `move_up` | None | 2/46 | **+0.0** / **-1.0** | **0.050** | — | 0.3 drink |
| `card_a5_e4_c0` | `do` | collect_sapling | 184/2873 | **+5.5** / **-162.0** | **0.053** | — | 0.9 sapling, 0.3 food |
| `card_a5_e4_c1` | `do` | collect_sapling | 20/1968 | **+0.0** / **-11.0** | **0.050** | — | 0.8 sapling |
| `card_a5_e4_c2` | `do` | collect_sapling | 4/66 | **+0.0** / **-6.0** | **0.050** | — | 0.9 sapling |
| `card_a5_e4_c4` | `do` | collect_sapling | 4/114 | **+0.0** / **-2.0** | **0.050** | — | 1.0 sapling, 0.5 food |
| `card_a5_e4_c6` | `do` | collect_sapling | 3/109 | **+0.0** / **-1.0** | **0.050** | — | 0.8 sapling, 0.3 food |
| `card_a5_e4_c8` | `do` | collect_sapling | 23/2857 | **+1.5** / **-27.0** | **0.050** | — | 1.0 sapling, 0.3 food |
| `card_a5_e4_c9` | `do` | collect_sapling | 3/42 | **+0.5** / **-6.0** | **0.050** | — | 1.0 sapling |
| `card_a5_e4_c10` | `do` | collect_sapling | 16/2820 | **+2.5** / **-31.0** | **0.050** | — | 0.9 sapling, 0.3 food |
| `card_a5_e4_c11` | `do` | collect_sapling | 3/71 | **+0.0** / **-1.0** | **0.050** | — | 0.7 sapling |
| `card_a5_e4_c12` | `do` | collect_sapling | 3/17 | **+0.5** / **-4.0** | **0.068** | — | 0.5 sapling, 0.5 health |
| `card_a5_e4_c13` | `do` | collect_sapling | 9/2713 | **+0.5** / **-17.0** | **0.050** | — | 1.0 sapling |
| `card_a5_e4_c15` | `do` | collect_sapling | 5/130 | **+0.0** / **-6.0** | **0.050** | — | 0.8 sapling |
| `card_a5_e4_c16` | `do` | collect_sapling | 2/57 | **+1.0** / **-1.0** | **0.050** | — | 0.9 sapling |
| `card_a5_enone_c0` | `do` | None | 124/3639 | **+0.0** / **-1.0** | **0.050** | — | 0.6 sapling |
| `card_a5_enone_c7` | `do` | None | 11/975 | **+0.0** / **-5.0** | **0.050** | — | 0.3 food, 0.4 drink |
| `card_a5_enone_c9` | `do` | None | 11/2826 | **+0.0** / **-2.0** | **0.050** | — | 0.5 sapling, 0.3 drink |
| `card_a5_enone_c17` | `do` | None | 20/2910 | **+0.0** / **-1.0** | **0.050** | — | 0.3 sapling |
| `card_a5_enone_c21` | `do` | None | 13/2863 | **+0.0** / **-5.0** | **0.050** | — | 0.5 sapling |
| `card_a5_enone_c30` | `do` | None | 12/2351 | **+0.0** / **-1.0** | **0.050** | — | 0.4 sapling |
| `card_a5_enone_c34` | `do` | None | 4/454 | **+0.0** / **-2.0** | **0.050** | — | 0.4 sapling |
| `card_a5_enone_c40` | `do` | None | 5/130 | **+0.0** / **-8.0** | **0.050** | — | — |
| `card_a5_enone_c54` | `do` | None | 3/52 | **+0.0** / **-1.0** | **0.050** | — | 0.4 sapling, 0.7 drink |
| `card_a5_enone_c56` | `do` | None | 5/1154 | **+0.0** / **-1.0** | **0.050** | — | — |
| ... | ... | ... | ... | ... | ... | *(+13 more reinforced cards)* | ... |

---

## 3. Causal Dependency Graph (Online Relations)

| Source Card (Enabler / Producer) | -> | Target Card (Consumer / Milestone) | Relation Type | Resource / Lift |
| :--- | :---: | :--- | :--- | :--- |
| `card_a5_e6_c5` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e15_o0` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e6_o0` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e6_o1` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e2_o2` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e6_o2` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e6_o3` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e6_o4` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e5_o1` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e5_o2` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e4_o3` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e6_o5` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e6_o6` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e15_o1` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e9_o0` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e4_o4` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e2_o5` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e6_o8` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e6_o9` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e6_o10` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e6_o11` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e4_o8` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e4_o9` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e6_o13` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e15_o4` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e5_o3` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e6_o14` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e6_o15` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e15_o5` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| `card_a5_e6_o16` | ➔ | `card_a8_e20_c0` | `produces_need` | Resource: **wood** |
| ... | ➔ | ... | ... | *(+5676 more causal edges)* |