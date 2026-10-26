# JEPA Crafter Project: Google Drive Data & Checkpoint Audit Manifest
**Generated**: 2026-10-26 23:35:46 | **Root**: `/content/drive/.shortcut-targets-by-id/18tW0Bzv4yPHA5kCgtSjS-ytI4IDUYJaa/learn-a-world`

## 1. System, GPU & Storage Status
- **Drive Free Storage**: **63.46 GB** free (Total: 112.64 GB, Used: 49.18 GB)
- **Compute Device**: Tesla T4 (CUDA Available: True, Total VRAM: 14.56 GB)
- **Feasibility for Collection**: **CONFIRMED YES** (Storage and GPU are fully sufficient for multi-shard collection runs).

## 2. Checkpoint Audit & Latest Verified Model
- **Official Verified Champion Baseline**: `jepa_crafter_v3_t2_final.pt` (Used for T2 screening & Phase 0/1 baseline).
- **Latest Phase 1 State-Delta Checkpoint**: `jepa_crafter_v3_phase1_transitions.pt` (Trained with active `StateDeltaHead`).

| Checkpoint File | Size | Step | Version | Has State Delta Head? | Modified |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `clean_optimal_memory_bank.pt` | 0.79 MB | None | v1/v2 | False | 2026-10-26 22:31:57 |
| `curiosity_t2.pt` | 7.15 MB | None | v1/v2 | False | 2026-10-24 19:46:07 |
| `jepa_crafter_v2.pt` | 19.71 MB | 5000 | v1/v2 | False | 2026-10-22 14:30:07 |
| `jepa_crafter_v2_best.pt` | 14.36 MB | 3000 | v1/v2 | False | 2026-10-22 17:25:08 |
| `jepa_crafter_v2_r1.pt` | 14.36 MB | 1000 | v1/v2 | False | 2026-10-22 15:20:11 |
| `jepa_crafter_v2_r2.pt` | 14.36 MB | 2000 | v1/v2 | False | 2026-10-22 16:15:33 |
| `jepa_crafter_v2_r3.pt` | 14.36 MB | 3000 | v1/v2 | False | 2026-10-22 17:11:01 |
| `jepa_crafter_v3_phase1_transitions.pt` | 39.83 MB | 1000 | v3 | True | 2026-10-26 23:16:24 |
| `jepa_crafter_v3_phase2.pt` | 18.1 MB | 5000 | v3 | False | 2026-10-22 20:20:41 |
| `jepa_crafter_v3_q_longhorizon.pt` | 18.77 MB | 0 | v3 | False | 2026-10-25 22:25:11 |
| `jepa_crafter_v3_step1_phase_a.pt` | 18.77 MB | 0 | v3 | False | 2026-10-24 23:52:47 |
| `jepa_crafter_v3_step1_stone.pt` | 18.77 MB | 0 | v3 | False | 2026-10-25 00:22:43 |
| `jepa_crafter_v3_step1_value.pt` | 14.36 MB | 10000 | v1/v2 | False | 2026-10-22 20:12:28 |
| `jepa_crafter_v3_t1.pt` | 18.1 MB | 15000 | v3 | False | 2026-10-23 17:35:11 |
| `jepa_crafter_v3_t1_joint.pt` | 40.72 MB | 15000 | v3 | False | 2026-10-23 19:30:16 |
| `jepa_crafter_v3_t1_joint_best.pt` | 40.73 MB | 4000 | v3 | False | 2026-10-23 19:14:51 |
| `jepa_crafter_v3_t2_final.pt` | 18.11 MB | 15000 | v3 | False | 2026-10-24 20:50:23 |
| `jepa_crafter_v3_t2_joint.pt` | 18.11 MB | 15000 | v3 | False | 2026-10-24 20:50:23 |
| `jepa_crafter_v3_t2_valuefix.pt` | 18.11 MB | 5000 | v3 | False | 2026-10-24 20:36:01 |
| `jepa_crafter_v3_t3_selfplay.pt` | 18.77 MB | 0 | v3 | False | 2026-10-26 17:38:42 |
| `semantic_memory_bank.pt` | 1.56 MB | None | v1/v2 | False | 2026-10-26 19:14:47 |
| `tabula_rasa_memory_bank.pt` | 0.73 MB | None | v1/v2 | False | 2026-10-26 21:00:48 |
| `train_state_t01_step0151000.pt` | 25.27 MB | None | unknown | False | 2026-10-23 15:38:07 |
| `train_state_t01_step0152000.pt` | 25.27 MB | None | unknown | False | 2026-10-23 15:39:36 |
| `train_state_t01_step0001000.pt` | 27.27 MB | None | unknown | False | 2026-10-26 23:12:17 |
| `train_state_t01_step0002000.pt` | 27.27 MB | None | unknown | False | 2026-10-26 23:14:15 |

## 3. Dataset Directories & Active Transitions
- **Total Shards Found**: 291 (Active Unique: 253, Superseded Numbered Duplicates: 38)
- **Active Deduplicated Transitions**: **860,314 transitions** across 4,925 episodes.
- **Advanced Chain Trajectories Exist?**: **YES**

### Key Causal Achievement Event Counts (Active Deduplicated Data):
| Achievement | Event Count | Chain Milestone Status |
| :--- | :---: | :--- |
| `collect_wood` | **4,022** | Abundant baseline supervisory evidence |
| `place_table` | **687** | Abundant supervisory evidence |
| `make_wood_pickaxe` | **223** | **Present in dataset** (rare positive demonstrations) |
| `collect_stone` | **87** | **Present in dataset** (frontier milestone) |
| `make_stone_pickaxe` | **0** | Zero (unreached frontier) |
| `collect_coal` | **10** | Rare (10) |
| `collect_iron` | **0** | Zero (unreached frontier) |
| `collect_diamond` | **0** | Zero (unreached frontier) |

## 4. Cumulative Duplicate Shard Identification
- **Found 38 numbered snapshots superseded by cumulative `_final.npz` shards**:
  * `shard_structured_random_0000.npz` (2000 trans) is superseded by `shard_structured_random_final.npz` in `combined_shards` (omit to avoid double-training).
  * `shard_teacher_hybrid_0000.npz` (2000 trans) is superseded by `shard_teacher_hybrid_final.npz` in `combined_shards` (omit to avoid double-training).
  * `shard_structured_random_0000.npz` (10122 trans) is superseded by `shard_structured_random_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_structured_random_0001.npz` (20108 trans) is superseded by `shard_structured_random_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_structured_random_0002.npz` (30129 trans) is superseded by `shard_structured_random_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_structured_random_0003.npz` (40175 trans) is superseded by `shard_structured_random_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_structured_random_0004.npz` (50105 trans) is superseded by `shard_structured_random_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_structured_random_0005.npz` (60126 trans) is superseded by `shard_structured_random_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_structured_random_0006.npz` (70052 trans) is superseded by `shard_structured_random_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_structured_random_0007.npz` (80004 trans) is superseded by `shard_structured_random_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_structured_random_0008.npz` (90148 trans) is superseded by `shard_structured_random_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_structured_random_0009.npz` (100142 trans) is superseded by `shard_structured_random_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_structured_random_0010.npz` (110146 trans) is superseded by `shard_structured_random_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_structured_random_0011.npz` (120138 trans) is superseded by `shard_structured_random_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_structured_random_0012.npz` (130203 trans) is superseded by `shard_structured_random_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_structured_random_0013.npz` (140049 trans) is superseded by `shard_structured_random_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_structured_random_0014.npz` (150000 trans) is superseded by `shard_structured_random_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_teacher_hybrid_0000.npz` (10065 trans) is superseded by `shard_teacher_hybrid_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_teacher_hybrid_0001.npz` (20139 trans) is superseded by `shard_teacher_hybrid_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_teacher_hybrid_0002.npz` (30108 trans) is superseded by `shard_teacher_hybrid_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_teacher_hybrid_0003.npz` (40037 trans) is superseded by `shard_teacher_hybrid_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_teacher_hybrid_0004.npz` (50119 trans) is superseded by `shard_teacher_hybrid_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_teacher_hybrid_0005.npz` (60136 trans) is superseded by `shard_teacher_hybrid_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_teacher_hybrid_0006.npz` (70004 trans) is superseded by `shard_teacher_hybrid_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_teacher_hybrid_0007.npz` (80009 trans) is superseded by `shard_teacher_hybrid_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_teacher_hybrid_0008.npz` (90109 trans) is superseded by `shard_teacher_hybrid_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_teacher_hybrid_0009.npz` (100078 trans) is superseded by `shard_teacher_hybrid_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_teacher_hybrid_0010.npz` (110044 trans) is superseded by `shard_teacher_hybrid_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_teacher_hybrid_0011.npz` (120049 trans) is superseded by `shard_teacher_hybrid_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_teacher_hybrid_0012.npz` (130009 trans) is superseded by `shard_teacher_hybrid_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_teacher_hybrid_0013.npz` (140093 trans) is superseded by `shard_teacher_hybrid_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_teacher_hybrid_0014.npz` (150000 trans) is superseded by `shard_teacher_hybrid_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_planner_v2_0000.npz` (5047 trans) is superseded by `shard_planner_v2_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_planner_v2_0001.npz` (10097 trans) is superseded by `shard_planner_v2_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_planner_v2_0002.npz` (15012 trans) is superseded by `shard_planner_v2_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_planner_v2_0003.npz` (20131 trans) is superseded by `shard_planner_v2_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_planner_v2_0004.npz` (25120 trans) is superseded by `shard_planner_v2_final.npz` in `sequences` (omit to avoid double-training).
  * `shard_planner_v2_0005.npz` (30000 trans) is superseded by `shard_planner_v2_final.npz` in `sequences` (omit to avoid double-training).

## 5. Detailed Shard Directory Breakdown
| Directory | Shard Filename | Size | Transitions | Episodes | Policy Source | State Delta? | Valid Deltas | Events? |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| `combined_shards` | `shard_curiosity_v3_t01_0000.npz` | 2.94 MB | 2006 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0001.npz` | 3.24 MB | 2286 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0002.npz` | 2.59 MB | 2131 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0003.npz` | 3.11 MB | 2031 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0004.npz` | 2.95 MB | 2041 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0005.npz` | 2.86 MB | 2121 | 13 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0006.npz` | 3.31 MB | 2071 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0007.npz` | 3.19 MB | 2097 | 13 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0008.npz` | 3.1 MB | 2105 | 13 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0009.npz` | 3.34 MB | 2080 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0010.npz` | 2.37 MB | 2066 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0011.npz` | 3.13 MB | 2119 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0012.npz` | 3.37 MB | 2217 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0013.npz` | 3.68 MB | 2118 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0014.npz` | 3.14 MB | 2139 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0015.npz` | 2.96 MB | 2155 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0016.npz` | 2.78 MB | 2152 | 13 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0017.npz` | 3.04 MB | 2115 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0018.npz` | 3.47 MB | 2067 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0019.npz` | 2.94 MB | 2108 | 13 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0020.npz` | 2.62 MB | 2115 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0021.npz` | 3.81 MB | 2180 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0022.npz` | 3.51 MB | 2066 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0023.npz` | 3.5 MB | 2181 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0024.npz` | 2.77 MB | 2185 | 14 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0025.npz` | 4.34 MB | 2163 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0026.npz` | 3.36 MB | 2147 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0027.npz` | 3.74 MB | 2015 | 10 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0028.npz` | 3.84 MB | 2248 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0029.npz` | 4.2 MB | 2075 | 9 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0030.npz` | 3.5 MB | 2064 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0031.npz` | 4.1 MB | 2220 | 10 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0032.npz` | 3.33 MB | 2009 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0033.npz` | 3.1 MB | 2003 | 10 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0034.npz` | 3.29 MB | 2182 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0035.npz` | 2.54 MB | 2021 | 14 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0036.npz` | 3.08 MB | 2153 | 13 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0037.npz` | 3.15 MB | 2036 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0038.npz` | 3.63 MB | 2059 | 10 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0039.npz` | 3.32 MB | 2027 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0040.npz` | 3.31 MB | 2062 | 10 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0041.npz` | 3.98 MB | 2138 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0042.npz` | 2.66 MB | 2035 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0043.npz` | 3.38 MB | 2018 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0044.npz` | 3.3 MB | 2056 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0045.npz` | 2.75 MB | 2008 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0046.npz` | 3.34 MB | 2071 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0047.npz` | 3.27 MB | 2150 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0048.npz` | 3.45 MB | 2220 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0049.npz` | 3.73 MB | 2228 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0050.npz` | 3.14 MB | 2079 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0051.npz` | 2.75 MB | 2064 | 14 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0052.npz` | 3.25 MB | 2112 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0053.npz` | 2.87 MB | 2047 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0054.npz` | 3.22 MB | 2006 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0055.npz` | 3.19 MB | 2013 | 10 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0056.npz` | 2.93 MB | 2151 | 13 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0057.npz` | 3.68 MB | 2143 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0058.npz` | 3.48 MB | 2047 | 10 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0059.npz` | 3.39 MB | 2362 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0060.npz` | 2.96 MB | 2127 | 13 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0061.npz` | 3.38 MB | 2018 | 10 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0062.npz` | 3.88 MB | 2069 | 10 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0063.npz` | 3.7 MB | 2068 | 10 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0064.npz` | 2.45 MB | 2125 | 13 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0065.npz` | 2.97 MB | 2075 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0066.npz` | 3.89 MB | 2092 | 10 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0067.npz` | 3.32 MB | 2144 | 12 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0068.npz` | 3.29 MB | 2092 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0069.npz` | 4.47 MB | 2141 | 10 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0070.npz` | 2.7 MB | 2150 | 13 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0071.npz` | 4.02 MB | 2170 | 10 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0072.npz` | 3.41 MB | 2023 | 10 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0073.npz` | 4.21 MB | 2093 | 10 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0074.npz` | 1.79 MB | 2091 | 14 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_curiosity_v3_t01_0075.npz` | 3.02 MB | 2114 | 11 | curiosity_v3 | False | 0 | True |
| `combined_shards` | `shard_structured_random_0000.npz` | 2.81 MB | 2000 | 13 | structured_random | False | 0 | True |
| `combined_shards` | `shard_structured_random_final.npz` | 2.81 MB | 2000 | 13 | structured_random | False | 0 | True |
| `combined_shards` | `shard_teacher_hybrid_0000.npz` | 2.94 MB | 2000 | 11 | teacher_hybrid | False | 0 | True |
| `combined_shards` | `shard_teacher_hybrid_final.npz` | 2.94 MB | 2000 | 11 | teacher_hybrid | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0000.npz` | 2.94 MB | 2006 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0001.npz` | 3.24 MB | 2286 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0002.npz` | 2.59 MB | 2131 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0003.npz` | 3.11 MB | 2031 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0004.npz` | 2.95 MB | 2041 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0005.npz` | 2.86 MB | 2121 | 13 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0006.npz` | 3.31 MB | 2071 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0007.npz` | 3.19 MB | 2097 | 13 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0008.npz` | 3.1 MB | 2105 | 13 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0009.npz` | 3.34 MB | 2080 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0010.npz` | 2.37 MB | 2066 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0011.npz` | 3.13 MB | 2119 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0012.npz` | 3.37 MB | 2217 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0013.npz` | 3.68 MB | 2118 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0014.npz` | 3.14 MB | 2139 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0015.npz` | 2.96 MB | 2155 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0016.npz` | 2.78 MB | 2152 | 13 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0017.npz` | 3.04 MB | 2115 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0018.npz` | 3.47 MB | 2067 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0019.npz` | 2.94 MB | 2108 | 13 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0020.npz` | 2.62 MB | 2115 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0021.npz` | 3.81 MB | 2180 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0022.npz` | 3.51 MB | 2066 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0023.npz` | 3.5 MB | 2181 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0024.npz` | 2.77 MB | 2185 | 14 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0025.npz` | 4.34 MB | 2163 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0026.npz` | 3.36 MB | 2147 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0027.npz` | 3.74 MB | 2015 | 10 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0028.npz` | 3.84 MB | 2248 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0029.npz` | 4.2 MB | 2075 | 9 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0030.npz` | 3.5 MB | 2064 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0031.npz` | 4.1 MB | 2220 | 10 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0032.npz` | 3.33 MB | 2009 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0033.npz` | 3.1 MB | 2003 | 10 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0034.npz` | 3.29 MB | 2182 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0035.npz` | 2.54 MB | 2021 | 14 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0036.npz` | 3.08 MB | 2153 | 13 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0037.npz` | 3.15 MB | 2036 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0038.npz` | 3.63 MB | 2059 | 10 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0039.npz` | 3.32 MB | 2027 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0040.npz` | 3.31 MB | 2062 | 10 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0041.npz` | 3.98 MB | 2138 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0042.npz` | 2.66 MB | 2035 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0043.npz` | 3.38 MB | 2018 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0044.npz` | 3.3 MB | 2056 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0045.npz` | 2.75 MB | 2008 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0046.npz` | 3.34 MB | 2071 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0047.npz` | 3.27 MB | 2150 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0048.npz` | 3.45 MB | 2220 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0049.npz` | 3.73 MB | 2228 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0050.npz` | 3.14 MB | 2079 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0051.npz` | 2.75 MB | 2064 | 14 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0052.npz` | 3.25 MB | 2112 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0053.npz` | 2.87 MB | 2047 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0054.npz` | 3.22 MB | 2006 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0055.npz` | 3.19 MB | 2013 | 10 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0056.npz` | 2.93 MB | 2151 | 13 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0057.npz` | 3.68 MB | 2143 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0058.npz` | 3.48 MB | 2047 | 10 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0059.npz` | 3.39 MB | 2362 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0060.npz` | 2.96 MB | 2127 | 13 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0061.npz` | 3.38 MB | 2018 | 10 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0062.npz` | 3.88 MB | 2069 | 10 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0063.npz` | 3.7 MB | 2068 | 10 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0064.npz` | 2.45 MB | 2125 | 13 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0065.npz` | 2.97 MB | 2075 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0066.npz` | 3.89 MB | 2092 | 10 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0067.npz` | 3.32 MB | 2144 | 12 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0068.npz` | 3.29 MB | 2092 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0069.npz` | 4.47 MB | 2141 | 10 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0070.npz` | 2.7 MB | 2150 | 13 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0071.npz` | 4.02 MB | 2170 | 10 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0072.npz` | 3.41 MB | 2023 | 10 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0073.npz` | 4.21 MB | 2093 | 10 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0074.npz` | 1.79 MB | 2091 | 14 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t01_0075.npz` | 3.02 MB | 2114 | 11 | curiosity_v3 | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0000.npz` | 2.21 MB | 2004 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0001.npz` | 1.99 MB | 2093 | 13 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0002.npz` | 3.0 MB | 2081 | 11 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0003.npz` | 2.61 MB | 2013 | 11 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0004.npz` | 2.47 MB | 2159 | 13 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0005.npz` | 2.72 MB | 2004 | 11 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0006.npz` | 2.7 MB | 2168 | 14 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0007.npz` | 2.71 MB | 2058 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0008.npz` | 4.31 MB | 2208 | 13 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0009.npz` | 3.87 MB | 2189 | 14 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0010.npz` | 4.88 MB | 2144 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0011.npz` | 4.03 MB | 2029 | 14 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0012.npz` | 2.99 MB | 2058 | 14 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0013.npz` | 3.62 MB | 2129 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0014.npz` | 3.82 MB | 2088 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0015.npz` | 2.37 MB | 2004 | 18 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0016.npz` | 3.91 MB | 2152 | 16 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0017.npz` | 2.97 MB | 2128 | 14 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0018.npz` | 2.9 MB | 2058 | 14 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0019.npz` | 4.29 MB | 2140 | 17 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0020.npz` | 2.93 MB | 2021 | 13 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0021.npz` | 3.64 MB | 2146 | 13 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0022.npz` | 4.89 MB | 2102 | 17 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0023.npz` | 3.42 MB | 2022 | 13 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0024.npz` | 3.53 MB | 2105 | 17 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0025.npz` | 3.52 MB | 2088 | 15 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0026.npz` | 4.17 MB | 2112 | 16 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0027.npz` | 5.22 MB | 2096 | 16 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0028.npz` | 4.26 MB | 2053 | 15 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0029.npz` | 3.39 MB | 2137 | 17 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0030.npz` | 3.72 MB | 2002 | 16 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0031.npz` | 3.3 MB | 2092 | 13 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0032.npz` | 3.44 MB | 2049 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0033.npz` | 3.75 MB | 2082 | 17 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0034.npz` | 4.63 MB | 2027 | 14 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0035.npz` | 3.92 MB | 2008 | 14 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0036.npz` | 4.08 MB | 2120 | 15 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0037.npz` | 3.37 MB | 2059 | 17 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0038.npz` | 3.05 MB | 2076 | 17 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0039.npz` | 4.04 MB | 2069 | 13 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0040.npz` | 4.07 MB | 2093 | 15 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0043.npz` | 3.46 MB | 2140 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0044.npz` | 2.77 MB | 2045 | 11 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0045.npz` | 3.19 MB | 2183 | 13 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0046.npz` | 3.06 MB | 2204 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0047.npz` | 2.41 MB | 2136 | 14 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0048.npz` | 1.92 MB | 2047 | 13 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0049.npz` | 2.38 MB | 2106 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0050.npz` | 3.85 MB | 2107 | 10 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0051.npz` | 3.53 MB | 2066 | 10 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0052.npz` | 2.98 MB | 2107 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0053.npz` | 2.64 MB | 2158 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0054.npz` | 2.65 MB | 2062 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0055.npz` | 2.47 MB | 2009 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0056.npz` | 3.16 MB | 2123 | 11 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0057.npz` | 3.24 MB | 2157 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0058.npz` | 2.64 MB | 2049 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0059.npz` | 2.27 MB | 2080 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0060.npz` | 1.94 MB | 2029 | 14 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0061.npz` | 3.7 MB | 2043 | 9 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0062.npz` | 3.45 MB | 2015 | 10 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0063.npz` | 2.89 MB | 2136 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0064.npz` | 2.71 MB | 2194 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0065.npz` | 2.8 MB | 2060 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0066.npz` | 2.85 MB | 2144 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0067.npz` | 3.13 MB | 2109 | 10 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0068.npz` | 2.95 MB | 2042 | 11 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0069.npz` | 3.38 MB | 2172 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0070.npz` | 3.26 MB | 2032 | 11 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0071.npz` | 2.14 MB | 2150 | 13 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0072.npz` | 3.47 MB | 2103 | 11 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0073.npz` | 2.8 MB | 2102 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0074.npz` | 2.6 MB | 2181 | 13 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0075.npz` | 3.46 MB | 2127 | 11 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0076.npz` | 2.13 MB | 2145 | 15 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0077.npz` | 3.29 MB | 2075 | 11 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0078.npz` | 2.63 MB | 2082 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0079.npz` | 1.7 MB | 2023 | 13 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0080.npz` | 2.8 MB | 2182 | 14 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0081.npz` | 2.72 MB | 2020 | 11 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0082.npz` | 2.55 MB | 2048 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0083.npz` | 2.84 MB | 2117 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0084.npz` | 2.22 MB | 2111 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0085.npz` | 2.41 MB | 2045 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0086.npz` | 2.6 MB | 2108 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0087.npz` | 2.7 MB | 2053 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0088.npz` | 2.71 MB | 2090 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0089.npz` | 1.92 MB | 2043 | 13 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0090.npz` | 3.7 MB | 2055 | 11 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0091.npz` | 2.66 MB | 2043 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0092.npz` | 3.69 MB | 2155 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0093.npz` | 3.81 MB | 2263 | 10 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0094.npz` | 2.76 MB | 2096 | 12 | curiosity_planner | False | 0 | True |
| `curiosity_shards` | `shard_curiosity_v3_t02_0095.npz` | 0.91 MB | 855 | 3 | curiosity_planner | False | 0 | True |
| `phase1_transitions` | `shard_curiosity_v3_t01_0000.npz` | 1.91 MB | 2000 | 13 | curiosity_v3 | True | 2000 | True |
| `self_play_shards` | `shard_self_play_5000_5049.npz` | 12.25 MB | 8869 | 50 | self_play_mpc | False | 0 | True |
| `sequences` | `shard_structured_random_0000.npz` | 11.09 MB | 10122 | 59 | structured_random | False | 0 | True |
| `sequences` | `shard_structured_random_0001.npz` | 23.06 MB | 20108 | 114 | structured_random | False | 0 | True |
| `sequences` | `shard_structured_random_0002.npz` | 34.88 MB | 30129 | 173 | structured_random | False | 0 | True |
| `sequences` | `shard_structured_random_0003.npz` | 47.36 MB | 40175 | 228 | structured_random | False | 0 | True |
| `sequences` | `shard_structured_random_0004.npz` | 59.17 MB | 50105 | 285 | structured_random | False | 0 | True |
| `sequences` | `shard_structured_random_0005.npz` | 71.16 MB | 60126 | 341 | structured_random | False | 0 | True |
| `sequences` | `shard_structured_random_0006.npz` | 82.96 MB | 70052 | 397 | structured_random | False | 0 | True |
| `sequences` | `shard_structured_random_0007.npz` | 94.74 MB | 80004 | 456 | structured_random | False | 0 | True |
| `sequences` | `shard_structured_random_0008.npz` | 108.1 MB | 90148 | 511 | structured_random | False | 0 | True |
| `sequences` | `shard_structured_random_0009.npz` | 120.83 MB | 100142 | 568 | structured_random | False | 0 | True |
| `sequences` | `shard_structured_random_0010.npz` | 131.32 MB | 110146 | 629 | structured_random | False | 0 | True |
| `sequences` | `shard_structured_random_0011.npz` | 144.76 MB | 120138 | 685 | structured_random | False | 0 | True |
| `sequences` | `shard_structured_random_0012.npz` | 157.98 MB | 130203 | 742 | structured_random | False | 0 | True |
| `sequences` | `shard_structured_random_0013.npz` | 171.28 MB | 140049 | 796 | structured_random | False | 0 | True |
| `sequences` | `shard_structured_random_0014.npz` | 184.66 MB | 150000 | 849 | structured_random | False | 0 | True |
| `sequences` | `shard_structured_random_final.npz` | 184.66 MB | 150000 | 849 | structured_random | False | 0 | True |
| `sequences` | `shard_teacher_hybrid_0000.npz` | 12.96 MB | 10065 | 57 | teacher_hybrid | False | 0 | True |
| `sequences` | `shard_teacher_hybrid_0001.npz` | 25.94 MB | 20139 | 113 | teacher_hybrid | False | 0 | True |
| `sequences` | `shard_teacher_hybrid_0002.npz` | 39.22 MB | 30108 | 168 | teacher_hybrid | False | 0 | True |
| `sequences` | `shard_teacher_hybrid_0003.npz` | 50.82 MB | 40037 | 227 | teacher_hybrid | False | 0 | True |
| `sequences` | `shard_teacher_hybrid_0004.npz` | 62.83 MB | 50119 | 286 | teacher_hybrid | False | 0 | True |
| `sequences` | `shard_teacher_hybrid_0005.npz` | 76.41 MB | 60136 | 342 | teacher_hybrid | False | 0 | True |
| `sequences` | `shard_teacher_hybrid_0006.npz` | 88.07 MB | 70004 | 399 | teacher_hybrid | False | 0 | True |
| `sequences` | `shard_teacher_hybrid_0007.npz` | 101.44 MB | 80009 | 455 | teacher_hybrid | False | 0 | True |
| `sequences` | `shard_teacher_hybrid_0008.npz` | 114.96 MB | 90109 | 511 | teacher_hybrid | False | 0 | True |
| `sequences` | `shard_teacher_hybrid_0009.npz` | 129.79 MB | 100078 | 564 | teacher_hybrid | False | 0 | True |
| `sequences` | `shard_teacher_hybrid_0010.npz` | 143.74 MB | 110044 | 618 | teacher_hybrid | False | 0 | True |
| `sequences` | `shard_teacher_hybrid_0011.npz` | 157.21 MB | 120049 | 674 | teacher_hybrid | False | 0 | True |
| `sequences` | `shard_teacher_hybrid_0012.npz` | 171.18 MB | 130009 | 728 | teacher_hybrid | False | 0 | True |
| `sequences` | `shard_teacher_hybrid_0013.npz` | 183.98 MB | 140093 | 784 | teacher_hybrid | False | 0 | True |
| `sequences` | `shard_teacher_hybrid_0014.npz` | 197.4 MB | 150000 | 839 | teacher_hybrid | False | 0 | True |
| `sequences` | `shard_teacher_hybrid_final.npz` | 197.4 MB | 150000 | 839 | teacher_hybrid | False | 0 | True |
| `sequences` | `shard_planner_v2_0000.npz` | 5.99 MB | 5047 | 30 | planner_v2 | False | 0 | True |
| `sequences` | `shard_planner_v2_0001.npz` | 10.03 MB | 10097 | 64 | planner_v2 | False | 0 | True |
| `sequences` | `shard_planner_v2_0002.npz` | 15.16 MB | 15012 | 96 | planner_v2 | False | 0 | True |
| `sequences` | `shard_planner_v2_0003.npz` | 19.61 MB | 20131 | 129 | planner_v2 | False | 0 | True |
| `sequences` | `shard_planner_v2_0004.npz` | 24.51 MB | 25120 | 160 | planner_v2 | False | 0 | True |
| `sequences` | `shard_planner_v2_0005.npz` | 29.48 MB | 30000 | 192 | planner_v2 | False | 0 | True |
| `sequences` | `shard_planner_v2_final.npz` | 29.48 MB | 30000 | 192 | planner_v2 | False | 0 | True |