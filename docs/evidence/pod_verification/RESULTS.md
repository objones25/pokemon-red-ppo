# Pod verification, 2026-11-04

Hosts: A100 80GB PCIe, Secure Cloud, CA-MTL-3, $1.79/h, AMD EPYC 7763 as a VM (252 CPUs visible, one NUMA node, no SMT visible),
cgroup v1 quota 26.35 vCPUs, torch 2.13.0+cu130 from the image, PyBoy 2.9.0. Other tenants shared the host, and `/proc/stat` and the
load average show them. Raw files are in `results/`; the code that made them is in `tools/`. No ROM, states, screenshots or addresses are kept.

## Scripts on a real pod

| Item | Result |
|---|---|
| `setup.sh` | Mount check passed. Venv on the container disk (302 MB), uv cache in `/root/.cache`, volume 37 MB. Torch comes from the image's `dist-packages`, with no second copy in the venv. CUDA available. 8 s |
| Worker count | **Bug, fixed:** the host has no `cpu.max`. It would have run 251 workers. Now reads `cpu.cfs_quota_us`: 25 |
| Pinning | 25 workers each on one CPU, on the GPU's NUMA node (node 0, the only one), main on its own CPU (`c_pincheck.json`). The VM hides SMT, so distinct physical cores can't be confirmed from sysfs. Pairs of adjacent CPUs ran at full speed (`cores.jsonl`), which is indirect evidence |
| Pinning choice | **Bug, fixed:** pinning to the lowest CPUs gave 1,881 steps/s, because CPUs 0-25 carried other tenants' load. Unpinned 3,095-3,226, pinned elsewhere 2,997-3,338, and pinned to the quietest cores 3,083 (`loopcost.jsonl`) |
| GPU use | Real, 5.8% utilization at k=8 T=512, 111 W against 53 W idle, 6.2 GB |
| Checkpoints | On the volume at `/workspace/pokered/runs/main/checkpoints` |
| SIGTERM | Saved at 819,200 steps and exited in 3 s with code 1 and a traceback, no workers left (`c_train.log`). The next run read "resumed from 819200 env steps" (`d_train.log`) |
| Self-delete | `--terminate-pod-on-exit` with the pod-scoped key from `/proc/1/environ` deleted the pod: GET 200, DELETE 204, and from outside the API read 404 (`d_train.log`). That key can't list volumes (403) |
| Confirming the delete | **Bug, fixed:** after the DELETE the same key reads 403, because it dies with its pod. `terminate()` raised `TerminateFailed` and logged a false "may still be billing". A refusal after a successful DELETE now counts as gone |
| W&B with no key | **Bug, fixed:** online mode raised `UsageError` at start-up and leaked the workers and shared memory; with self-delete on, that would have deleted the pod. `train.sh` now logs offline when there is no key |

| Second pod, same volume | Same host and limit. `setup.sh` rebuilt the venv in 8 s. `train.sh` read "resumed from 1024000 env steps", logged 1,126,400 and 1,228,800 next (no step repeated), and the config check passed (`e_train.log`). A worker count that differs on resume is covered by a unit test only, because both pods had 25 |
| `pokered-evaluate` | On the 1,228,800-step checkpoint: exit 0, 50 episodes, 67.6 tiles, 0.08 events, 10,404 steps (`f_eval_main.json`). This early policy hasn't left the start area |
| Self-delete with the fix | Second pod: GET 200, DELETE 204, GET 401, then the connection closed with no traceback in the log, and the API read 404 (`g_final_selfdelete.log`) |

Offline W&B ran. Online W&B was not tested: no key was available.

`chmod` has no effect on this network volume: `.env` stays `rw-rw-rw-` whatever `setup.sh` asks for. The volume is only reachable from this account's pods, but a real key there is readable by any process on them.

Time and cost: pod `x4dv4gnmzsazoz` 150.8 min, pod `qj5mh9fcd206dj` 13.5 min, 164.3 min at $1.79/h, $4.90 by minutes x rate (billing had recorded $3.43 of it when read), plus a 10 GB volume for about 3 hours. A100s were out of stock in most reads: a single create failed on stock twice before each success.

## Throughput (Part 2)

Real `pokered-train` through `train.sh`, 25 workers pinned to quiet cores, `perf/steps_per_second` per update with the first dropped.

| Config | Batch | Updates used | Steps/s (SD) | Env seconds per update | Update seconds | Update share |
|---|---|---|---|---|---|---|
| k=8, T=128 | 25,600 | 15 | 2,840 (119) | 8.72 | 0.31 | 3.5% |
| k=8, T=512 | 102,400 | 5 | 3,002 (122) | 33.01 | 1.16 | 3.4% |
| k=8, T=2048 | 409,600 | 3 | 2,896 (100) | 136.96 | 4.65 | 3.3% |
| k=32, T=64 | 51,200 | 7 | 3,161 (75) | 15.64 | 0.57 | 3.5% |
| k=8, T=512, 1 torch thread | 102,400 | 3 | 3,168 (69) | 31.17 | 1.16 | 3.6% |
| k=8, T=128, fast | 25,600 | 15 | 5,652 (386) | 4.24 | 0.31 | 6.8% |
| k=8, T=512, fast | 102,400 | 5 | 6,303 (279) | 15.10 | 1.18 | 7.2% |
| k=32, T=64, fast | 51,200 | 7 | 7,232 (274) | 6.52 | 0.57 | 8.0% |

T changes speed little because the update is 3-4% of the time here. T=2048 was within one SD of T=512, so the default is now T=2048, the length the paper preferred. Its batch is 6.25 times the paper's 65,536. Whether that batch, or T=512 at 102,400, learns better needs training. A stop now loses up to about 2.3 minutes of rollout rather than 34 s. k=32 is about 5% faster, but changes the batch and is a separate question. These ran from the start state on a host whose load average moved between 22 and 90, and only 3-15 updates each, so a few percent means little.

## fast_mode (Part 3)

**Memory.** Compared after every 24-frame decision: WRAM C000-DFFF, HRAM FF80-FFFE (RNG at FFD3/FFD4), map/x/y, battle byte, event flags, rewards, tiles. No difference in any byte, in any run, so there is no first divergence.

| Set | Steps | Battle steps | Map changes | Maps | WRAM+HRAM differences |
|---|---|---|---|---|---|
| Scripted scenarios | 45,533 (and again open-loop) | 18,504 | 170 | 41 | 0 |
| Policy, 10 trajectories of 12,000 | 120,000 | 7,962 | 1,486 | 6 | 0 |
| Policy into Viridian Forest, 12 of 15,000 | 180,000 | 55,546 | 102 | 9 | 0 |
| Controls, accurate vs accurate | 15,044 | 5,347 | 101 | 25 | 0, and no screen differences |

Scenarios: 60 forced wild battles at levels 2-7 and 40, 24 trainer battles, every Start menu item, 6,000 steps of A presses around people in Oak's lab and beyond, flights to all 11 cities with 600 steps each, 6 Pokemon Center heals from 1 HP back to 22 of 22 in both modes, and 13 of 14 later maps entered by rewriting warps (Viridian Forest, Mt Moon, Rock Tunnel, S.S. Anne, Pokemon Tower, Safari, Silph Co, Mansion, Victory Road, Seafoam, Rocket Hideout, Power Plant, Celadon Mart; Pewter Gym wasn't entered). The forced battles and later maps are reached by writing memory, the same way in both emulators, not by playing there.

At decision boundaries DIV and LY also matched in a short local check, and after 100-400 steps a save state differed in 4-148 of 178,108 bytes, mostly a 4-byte field (`statediff.py`).

Not compared: frames between decisions, cartridge RAM directly, and anything beyond the maps above with real play. Sound is off in both modes.

**Screens.** Differences touch 1.2-7.8% of steps by set, and never over half a frame.

| Set | Screen-differing steps | Overworld | Window up | Battle |
|---|---|---|---|---|
| Scenarios | 7.8% | 5.4% | 7.5% | 10.0% |
| Policy, Pallet and Route 1 | 4.8% | 4.2% | 15.0% | 12.6% |
| Policy, into the forest | 1.2% | 0.25% | 6.2% | 2.9% |

Replaying each decision frame by frame, in 3,300 classified differences (at most 300 per run) fast's image was identical to accurate's image one frame later in 2,862 (87%). The source shows why: `FastLCD` renders the whole frame at the end of frame 153 from current memory, so an OAM or tile change made in VBlank shows immediately, where accurate draws scanlines before it. So fast shows the same picture one frame sooner, not a different one. In the other 438, the closest match in a sample was the same frame with about 100-280 pixels differing, during fades and battle slides (`results/neither/`).

**Policy consequence.** Same state, accurate vs fast observation, on steps where any observation differs.

| Policy | Share of steps | Mean total variation | Worst run's p95 | Max | Most-likely action changes | Mean value gap |
|---|---|---|---|---|---|---|
| Fast-trained, 5M | 12.0% | 0.010 | 0.047 | 0.79 | 153 of 14,419 | 0.0045 |
| Accurate-trained, 5M | 2.6% | 0.014 | 0.079 | 0.24 | 34 of 4,668 | 0.0076 |

**Cross-mode evaluation**, with the real `pokered-evaluate`, 48 episodes, seed 7. Only means are recorded, so no significance is claimed.

| Policy | In accurate: tiles, events, forest | In fast: tiles, events, forest |
|---|---|---|
| Accurate-trained, seed 0, 5M | 243.2, 4.00, 100% | 245.4, 4.00, 100% |
| Fast-trained, seed 0, 5M | 364.5, 3.98, 0% | 359.4, 3.92, 0% |
| Accurate-trained, seed 1, 3M | 187.9, 3.88, 0% | 191.8, 3.85, 0% |
| Fast-trained, seed 1, 3M | 420.6, 3.52, 79% | 442.7, 3.58, 85% |

No policy got worse in the mode it wasn't trained in.

**Save states** saved in either mode and loaded in the other ran 3,000 random steps with no WRAM+HRAM divergence from the original, in Oak's lab and Viridian (`fid_interchange.jsonl`). The screen differed on 281-323 steps, as usual.

**Learning**, same seed and steps, 88 envs, T=256, budgets 2,048 + 512 per event (`learning_curves.txt`).

| Run | Reached Viridian | Forest | Late tiles | Events |
|---|---|---|---|---|
| Seed 0, accurate, 5M | 100% by about 4M | 99-100% | About 240 | 3.9-4.0 |
| Seed 0, fast, 5M | Never | Never | 260-370 | 4.0 |
| Seed 1, accurate, 3M | 4% | Never | About 190 | 3.8 |
| Seed 1, fast, 3M | 42%, then 100% | 18%, then 50% | 375-390 | 3.8-4.0 |

Which mode got further flips with the seed. Two seeds can't rule out a small effect in either direction, but there is no sign of one. Same-steps wall time: fast took 2.3 times less at equal shares of the CPU (3,685 vs 1,578 and 3,328 vs 1,531 steps/s).

**Verdict: safe to train with** on these maps, because game state stayed identical in everything compared. What changes is the picture, one frame sooner on a few percent of steps, and policies didn't care. Expect about 2.0-2.3 times the throughput on a host like this one (+99%, +110%, +129% across three configs). Conditions: rerun `tools/fidelity.py` if PyBoy changes from 2.9.0, because this rests on that version's `FastLCD`; and keep evaluation on one mode so comparisons across runs stay clean. What this can't settle: late-game maps under real play, and a small learning effect, which needs more seeds.
