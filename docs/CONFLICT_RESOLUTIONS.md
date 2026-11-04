# Conflict resolutions

Resolved 2026-11-03 by separate agents, each with a narrow brief and without the others' conclusions. These verdicts replace the conflicting claims in `UPSTREAM_FINDINGS.md`, `STARTING_POINT.md`, `PERFORMANCE_BOTTLENECKS.md` and `RAM_MAP_REPORT.md`. Those reports are otherwise unchanged. Scripts and raw results are in `docs/evidence/`; their paths point at the temporary folder they first ran in.

## Verdicts

| # | Conflict | Verdict | Status |
|---|---|---|---|
| 1 | Is the emulator the bottleneck? | Both reports were wrong as general statements. It depends on the configuration | Resolved |
| 2 | Does a Squirtle + Pokédex state exist? | Yes: `has_pokedex.state` and `has_pokedex_nballs.state` | Resolved |
| 3 | Is replay deterministic? | Yes for current-format saves. Old-format saves depend on emulator history | Resolved |
| 4 | 311 or 320 event-flag bytes? | 320. A 311-byte read drops 8 defined flags | Resolved |
| 5 | Keep the level reward? | Keep it. Neither source establishes its claim | Default set; only training can settle it |
| 6 | Keep the heal reward? | Depends on the starter. Not the paper's repeatable form; test a latched nurse heal first | Default set; only training can settle it |
| 7 | 1,000 or 2,500 steps/s? | Mostly configuration. SB3 itself costs about 12% | Resolved |
| 8 | Does most of an A100 sit idle? | **Yes, with a Nature CNN on the host tested:** 4–7% utilization | Resolved for that host |
| 9 | MPS or CPU inference? | Both, for different things. Not applicable to the A100 until measured | Resolved for Mac |

## 1. Bottleneck and 7. throughput

All runs used the same env, the same 1.06M-parameter Nature CNN (SB3's dict-observation row used a 1.11M-parameter one), 3 epochs, and minibatch 512 unless noted. Each timed window was 60 s or more, with each configuration run at least twice. Runs with background load above about 1 core are flagged and left out when a clean run exists; the 12x1 `fast_mode` row has no clean run.

| Configuration | Steps/s | Emulator | Sync/IPC | Rollout policy | Update |
|---|---|---|---|---|---|
| SB3, 10 envs, dict obs, T=2048, minibatch 8,192 | 1,268 | 28% | 14% | 45% | 13% |
| SB3, 10 envs | 1,484 | 32% | 15% | 40% | 12% |
| SB3, 12 envs | 1,557 | 32% | 17% | 37% | 13% |
| SB3, 10 envs, `fast_mode` | 2,042 | 18% | 17% | 47% | 17% |
| Custom, 12 workers × 1 env | 1,739 | 31% | 17% | 38% | 14% |
| Custom, 12 × 1, `fast_mode` (flagged) | 2,468 | 18% | 14% | 46% | 20% |
| **Custom, 12 × 8** | **2,667** | **46%** | **16%** | **16%** | **21%** |
| **Custom, 12 × 8, `fast_mode`** | **4,782** | **29%** | **13%** | **20%** | **37%** |

- **"The emulator limits training, not the network"** is true only at 12 × 8, and not with `fast_mode`, where the update is the largest part.
- **"About 80% of time is outside the emulator"** overstated it. The measured figure is about 68% at SB3 with 10 envs.
- **"`fast_mode` alone gives about 14%"** is wrong. It gave +38% under SB3 and +79% at 12 × 8. This speed result says nothing about whether `fast_mode` plays the same game.
- **Where the throughput gap came from:** SB3 to custom is ×1.12, then 12 × 1 to 12 × 8 is ×1.53, the largest step. The reported 1,000 steps/s could not be reproduced; its script and load data are gone, so its cause is unproven.

**Decision:** the custom loop, with several envs per worker. Its measured advantage over SB3 comes more from that batching than from the loop itself.

Other corrections:

- **Memory:** each extra env adds about 5 MB; the rest is per-process baseline. A worker with 1 env and no torch is about 76 MB, with 8 envs about 100 MB. So 96 envs in 12 workers used about 1.2 GB, not 6.8 GB.
- **Observation cost:** an integer block mean with stacking costs about 23 µs, against 172 µs for the float path, so the pixel-discarding slice isn't needed for speed. Reading 320 flag bytes for rewards costs about 18 µs per step.
- **`uint8` on device:** converting on the device vs on the CPU gained 0–3% on the Mac. CUDA is unmeasured.

## 2. Start state

Confirmed by playing the state: the Start menu has POKéDEX, the Pokédex shows Squirtle at No. 007 with 1 owned, and the party holds one Squirtle. SQUIRTLE is `$B1` in `pret/pokered`. Game memory in the converted state is identical to the source save.

**The state has battle animations on.** `wOptions` (`D355`) is `00000001`: fast text, with bit 7 clear. So it is not the paper's Fast setup, despite `STARTING_POINT.md:27`.

## 3. Replay determinism

| Condition | Result |
|---|---|
| Current-format state: repeated, fresh vs reused, reloaded after other play | Identical |
| Current-format state: save and restore mid-run, in the same or another emulator | Identical to an uninterrupted run |
| Old-format state: fresh vs reused emulator | **Diverges from step 1** |
| Old-format state: reused after different histories | **Diverges from step 1** |

- **Cause:** version 7 saves omit the LCD timing fields, so `load_state` keeps whatever the emulator already had. Copying those 5 fields reproduces the other trajectory exactly. LCD phase shifts `DIV`, which feeds the game's RNG.
- **Correction:** the earlier "step 983" came from comparing WRAM only. With HRAM included, the RNG bytes `FFD3` and `FFD4` differ from step 1. On `has_pokedex`, the player's position first differs at step 794.
- **Blank screen:** after a restore in a fresh emulator, frame 1 is the saved image, frame 2 is blank, and frame 3 onward is normal. Ticking at least 3 frames is still right.

## 4. Event flags

320 bytes (`NUM_EVENTS` is `$A00` = 2,560 bits), from `D747` through `D886`. The source is `constants/event_constants.asm:776-777` and two separately obtained symbol files. Upstream's v2 misses `D87E` through `D886`, which includes `EVENT_BEAT_ARTICUNO`. Bits number lowest first: event n is byte n/8, bit n%8.

## 5. Level reward

New primary source: the authors' public W&B project `thatguy11325/pokemon`, with 1,790 runs and 41 that beat the Champion. I checked those counts against the pulled data.

| Claim | Evidence | Assessment |
|---|---|---|
| Paper: removal is the "best ablation" | 4.60 ± 0.7 vs 4.19 ± 0.2 milestones, 5 seeds; p ≈ 0.27 | Highest mean only. Not separable. The only clear effect is catching fewer Pokémon (party 1.58 vs 3.48) |
| Rubinstein: "Any attempt to remove the reward led to a failed experiment" | No numbers. Removal periods were confounded by other changes | Unproven. All 41 winners use the term, at weight 0.93–1.13, which shows use, not need |

Separating the paper's gap would take about 25 seeds per arm (rough estimate).

**Default: keep it on.**

**The cap:** the `LEVEL_CAP_BY_BADGES` numbers match `parties.asm` exactly. No source supports or forbids a hard cap. A flat cap at level 15 was tried and quickly removed with no reason given; the runs hit the same wall either way. Winners' estimated level totals stayed below our cap almost everywhere, while a non-winning run went past it at 6 badges (estimated from averages, and winners used `perfect_ivs`).

**Known limits of the cap:** rival fights are not covered (Cerulean rival totals 65 against a cap of 39), the index counts badges rather than which gym was beaten, and the total doesn't measure team strength. Both sources' soft knee is the natural comparison arm.

## 6. Heal reward

| Start | Milestones, no heal vs heal | Events, no heal vs heal |
|---|---|---|
| Squirtle | 3.94 vs 4.19 (p ≈ 0.08) | 43.2 vs 51.8 (p < 0.001) |
| Fast (Squirtle) | 3.77 vs 4.40 | 41.8 vs 57.2 (p ≈ 0.002) |
| Charmander | 3.73 vs 3.03 | 45.3 vs 46.5 |
| Bulbasaur | 3.18 vs 2.89 | 36.6 vs 30.3 (p ≈ 0.004) |

**This matters for our start: removing heal cost Squirtle agents events.** The 399-heal farming figure is Bulbasaur only; the paper says Squirtle "typically avoids healing exploits".

All 41 winners used a latched form: `pokecenter_heal` pays at most once per mini-episode, set by the `AnimateHealingMachine` hook, weight about 0.49. In pret, a nurse visit also sets the blackout respawn point, which is the mechanism the paper credits. Why it works is an inference.

**Default:** no heal term yet. **First reward experiment:** a latched nurse heal, which meets the no-farming rule.

## 8. GPU use

Measured on the M4 Pro at 12 × 8. Capacity is the share of the GPU's tight-loop throughput actually used.

| Model | Parameters | Steps/s | Capacity used |
|---|---|---|---|
| Nature | 1.06M | 2,612–2,677 | About 25% |
| IMPALA ×1 | 0.84M | 1,111–1,149 | About 60% |
| IMPALA ×4 | 4.50M | 270–281 | 86–90% |

Before the pod run below, with op-level data suggesting an A100 is about 1–5× this GPU for this kind of work, the projection for a Nature CNN at 32 vCPUs is 4–55% busy. At up to 16 vCPUs it stays under 50% in every case. So the claim leans true, but rests on unmeasured inputs.

What raises GPU use, measured on the Mac:

| Knob | Steps/s | Effect on GPU use |
|---|---|---|
| More envs per worker (k = 1 → 32) | +75% | **Slightly lowers** busy %. Worth doing for throughput, not utilization |
| Larger model (IMPALA ×1, ×2, ×4) | −57%, −75%, −90% | Raises it. Only training shows whether it learns better |
| 10 epochs | −32% | Raises it, with no new data. Benefit unproven |
| 30 epochs | −66% | Treat as utilization only unless a learning test says otherwise |
| Minibatch 128 | −16% | Mostly dispatch overhead |
| Minibatch 2,048 | No change | Not a utilization knob |

**Catalog read 2026-11-03, live and changing:** A100 PCIe Secure $1.79/h, 12 vCPU minimum, availability LOW. A100 SXM Community $1.39/h, 32 vCPU minimum, availability LOW. A100 SXM Secure read NONE. `minVcpuCountPerGpu` is a placement filter, so the real count is known only after creating a pod.

**Errors in the earlier projection:** it divided every model by one inference peak (5.1 TFLOP/s, hence 6% for Nature rather than about 25%), scaled only the update by GPU speed, gave IMPALA ×4 Nature's inference cost, fitted sync overhead at k=1 but used it at k=8, and used W = V. The 5 ms emulator time and g=20 have no source.

### Measured on a pod, 2026-11-04

One A100 SXM 80GB, Community Cloud, $1.39/h, AMD EPYC 7742, torch 2.13, about 37 minutes, about $0.87 by minutes × rate (billing had no record yet). Raw results: `docs/evidence/gpu/pod_results/`.

- **The pod's real CPU limit is 27.2 vCPUs.** The API reports 32, but the cgroup quota is `2720000 100000`.
- **Each emulator step takes about 6.4 ms**, against about 2.1–2.6 ms on the M4 Pro. Free-running scaling tops out at about 4,150 steps/s.
- **Pinning each worker to its own physical core on the GPU's NUMA node was the largest gain:** +56% at k=8 and +20% at k=32. It gave nothing at k=1.

| Config, 26 workers | Steps/s | Env % | Update % | GPU utilization | Power |
|---|---|---|---|---|---|
| Nature, k=8 | 1,795 | 89 | 8 | 4.3% | 81 W |
| Nature, k=32 | 2,434–2,464 | 86 | 12 | 5.5% | 81 W |
| **Nature, k=8, pinned** | **2,793** | 80 | 15 | 6.6% | 80 W |
| **Nature, k=32, pinned** | **2,966** | 82 | 16 | 6.5% | 80 W |
| Nature, k=8, 10 epochs | 1,689 | 67 | 30 | 11.2% | 85 W |
| IMPALA ×1, k=32 | 1,870 | 70 | 27 | 24% | 129 W |
| IMPALA ×4, k=8, T=64 | 1,243 | 50 | 43 | 45% | 211 W |

Idle power is 68 W. Update speed relative to the M4 Pro: Nature 1.7×, IMPALA ×1 3.7×, IMPALA ×4 8.7×. The earlier 4–55% projection is replaced by these measurements.

**So this pod trains about 11% faster than the M4 Pro** (2,966 vs 2,667 steps/s), because its emulator steps are about 3× slower. At $1.39/h, 400M steps is about 37.5 hours and $52. The pinned configs ran once each, only 2–4 PPO cycles per window, and on one host; other A100 machines may differ. `fast_mode` was not measured on the pod.

## 9. Inference device

At 12 envs, CPU inference beats MPS by 29% (2,243 vs 1,739 steps/s). At 96 envs, MPS beats CPU by 37% (2,667 vs 1,952). The update belongs on MPS either way: a CPU update takes 66% of wall time and costs about 12× more per sample.

## Smaller mismatches

| Item | Answer |
|---|---|
| LSTM size | 512. All 1,397 runs that log a size use it, including the winners; the docs' 128 is wrong |
| Level formula | `15 + (sum-15)/4`, from code and confirmed by W&B values; the docs' `30 + ...` is wrong |
| Paper title | Both are real: v1 "Playing Pokémon Red via Deep Reinforcement Learning", v2 "Pokemon Red via Reinforcement Learning" |
| Paper's level sum in code | `Σ max(lvl-2, 0) - 4`, so Eq. 2's knee at 22 sits at about raw level 28 for one Pokémon |
| Winning config | `baseline.ObjectRewardRequiredEventsMapIds`, Bulbasaur, `perfect_ivs: True`, infinite money and field-move scripts; not today's `config.yaml` defaults |
| "The paper capped the level reward" (`UPSTREAM_FINDINGS.md`) | Wrong: it is a soft knee with no cap |

## What only training can settle

- Whether the level reward, a latched heal or the level cap helps from our start state.
- Whether a larger model learns better.
- Whether `fast_mode` is safe late in the game. It matched for 20,000 steps, covering only early maps and 317 battle steps.
