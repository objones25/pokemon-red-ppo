# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Goal

Train a PPO agent to play Pokemon Red on the PyBoy emulator, as a rebuild and extension of
[PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments), and measure whether it can beat that
baseline. Progress means **exploration and story events**. Battling and training Pokemon are rewarded only as far as
they serve that progress: the agent must not be able to earn reward by grinding.

## Status

Built and tested: start state, Gymnasium env (battle animations off), parallel workers with CPU pinning, model, custom
PPO, atomic checkpoints with resume, W&B logging, pod shutdown, and the pod scripts (`launch_pod.sh`, `start.sh`, `setup.sh`,
`train.sh`). Pods train with `fast_mode`; `EnvConfig` still defaults to accurate, and `pokered-evaluate` uses whichever mode the
checkpoint was trained in.

A 3M-step local run (one seed, `docs/evidence/learning/`) reached 693 unique tiles and 4 events per episode against 73 and
0 for random actions, though its episodes ran twice as long because events extend the budget, and it gained no levels.

`scripts/setup.sh`, `scripts/train.sh`, pinning, SIGTERM saves and self-deletion with the pod-scoped key have run on a real
A100 pod (`docs/evidence/pod_verification/RESULTS.md`). `launch_pod.sh` and `start.sh` have too. One full 400M-step run finished on an A100 in 17.6 hours, online to W&B, under the old level cap and metrics: events plateaued near 35 from about 200M, Misty reached 72% of episodes at best, and no episode got the S.S. Ticket. The final checkpoint, frozen, beat Misty in only 25 of 96 first episodes on an A100 (`docs/evidence/first_run/a100/`). Not the cause, as tested: hardware, snapshots, resets, old versus new code, number precision, the evaluator. The cause: from the same weights, PPO updates during the episode took Misty from 39.5% to 84% of first episodes, against frozen weights (`docs/evidence/first_run/update_test/`, `--init-from`, `--no-update-policy`). Training-time rates measure the policy plus its ongoing learning.
A second 400M run (run2, per-place level cap, `docs/evidence/run2/AUDIT.md`) did worse against Misty (27% at best, 8% at the end) and ended with 54% of its steps stalled in Cerulean Gym: with its starter fainted it kept a level-5 Magikarp out and never took a turn. run3 (those fixes) reached 37-42 events but never fought Misty and never met Bill. Of run2's 320 stall states, 318 had never spoken to a nurse. run4 added a nurse heal paid once per center and a latched move term: it beat Misty in 95-99% of episodes from 120M and reached about 52 events, with the respawn point at Cerulean in 74 of 84 late stall states (run3: Pallet Town in all 320), and Bite on 90% of starters at level 24+ (run3: 10%). It never used Bill's Cell Separator (`docs/evidence/run4/`). Since then the move term is signed, so losing a move costs its score, and a hidden-event term pays once per PC, sign or poster pressed. run5 (both, 258M steps, ended when the account ran out of credit) reached 60-64 events, got the S.S. Ticket in 90-98% of episodes from 215M and boarded the S.S. Anne in at most 3%, then ended 89-94% of late episodes in the Pokemon Fan Club, still pressing into the table where it got the Bike Voucher (`docs/evidence/run5/`). run6 (the map term, from run5's weights, 400M) boarded the S.S. Anne in a median 97% of episodes from 45M but got HM01 in a median 1%: it cleared every cabin before the rival and arrived at a median 14% HP, and replaying its 15 rival fights fully healed won 119 of 120 against 11 of 120 as saved (`docs/evidence/run6/ANALYSIS.md`). The S.S. Anne rival sets no event flag, so beating him paid nothing. Since then the Cut chain pays: beating that rival, Poke Balls held, a party Pokemon that can learn Cut, knowing Cut, and each tree cut. Not trained yet.
Keep this section and **Commands** in step with the code, and do not document commands that do not exist.

## Requirements

- **Start state:** `states/has_pokedex_squirtle.state`: Squirtle chosen, Pokedex received, in Oak's lab.
- **No farmable reward:** every term pays at most once for the same thing, so each has a fixed lifetime maximum per
  episode, and repeating any sequence of frames pays nothing more. Never add a repeatable term (heals, XP, money,
  battles, menu use). Only a term declared `signed` may pay a negative reward. See **Reward rules**.
- **GPU training on Runpod** (an A100-class pod), working from a network volume mounted at `/workspace`, with scripts
  that install dependencies, set up `.env`, run training and shut the pod down on completion.
- **Resume and checkpointing:** training resumes from the latest checkpoint on the volume, with W&B logging on the
  same run.

## How to work here

- **Don't assume library behavior.** Check current docs with `npx ctx7@latest` and the installed version with
  `inspect` before relying on a signature, default or number. PyBoy, Gymnasium, W&B and Runpod all change.
- **TDD, DRY, SOLID.** Failing test first, and watch it fail for the right reason. One source of truth for each
  address and config value. `src/pokered/ram_map.py` owns game addresses; don't repeat them.
- **Negative space programming is required.** Write the checks with the code: preconditions, postconditions and what
  must never be true, using `pokered.negative_space` (`require`, `expect`, `check_shape`, `check_finite`, `bounded`),
  never bare `assert` in `src`. Bugs crash; operating errors (bad CLI input, missing file, network) raise typed
  exceptions and are handled. Bound every loop. Never swallow an exception. Fix type errors at the cause rather than
  with `cast` or `# type: ignore`.
- **`scripts/check.sh` must pass before any work is called done.** It runs pyright (what Pylance runs) over `src`, `tests`
  and `scripts`, ruff, both audits, and the whole test suite. Passing tests alone is not done.
- **Skills:** `negative-space-programming`, `pytest-expert` (gates, mutation checks), `pytorch` (training loop, checkpoints),
  `robot-learning-expert` (reward and PPO correctness, evaluation), `design-patterns-expert` (structure).
- **The user's Mac must stay usable.** It's their daily machine. Long training runs on rented pods, and local jobs run at
  low priority (`nice`, `OMP_NUM_THREADS=1`) with a time limit. Check for leftover processes afterward.
- **Other Claude shells may run here at the same time.** Benchmarks taken while they run are contaminated: check the
  load average and report it. Don't overwrite files you didn't write.

## Commands

Python 3.12, uv only.

```bash
uv sync                                           # install, including the dev group
uv run pytest                                     # fast tier, with coverage floor
uv run pytest -m ""                               # everything, including slow tests
uv run pytest tests/test_rewards.py::test_new_tile_pays_nothing_on_revisit --no-cov -v   # one test
uv run pytest -m "" -p randomly -n auto           # random order, in parallel
scripts/check.sh                                  # every gate: pyright, ruff, both audits, all tests
uv run python scripts/audit_tests.py tests/       # must print "clean: no findings"
uv run python scripts/audit_negative_space.py src/pokered --min-assertions 1
uv run ruff check .
uv run pyright

# Prove tests can fail: each survivor needs a new test or a one-line reason it can't be killed.
# Use .venv/bin/python, not `uv run`: a timeout only kills the direct child, and a hung mutant under uv lives on.
OMP_NUM_THREADS=1 nice -n 15 python3 ~/.claude/skills/pytest-expert/scripts/mutate_check.py src/pokered/rewards.py -- \
  .venv/bin/python -m pytest -x -q -p no:randomly --no-cov tests/test_rewards.py

# Re-save a state in the installed PyBoy's format, without advancing the game.
uv run python scripts/convert_state.py "Pokemon Red.gb" SOURCE.state DESTINATION.state
```

To train on a pod, from the Mac, with the pod's public IP and external SSH port:

```bash
POKERED_RUN_NAME=run3 scripts/launch_pod.sh HOST PORT   # repo and ROM onto the volume, then start.sh in a terminal; rerun to reattach
# Always set POKERED_RUN_NAME: it is the W&B run id and run directory. Pod env vars don't reach SSH, and unset means "main", the first run.
```

On the pod, with the repo on the `/workspace` volume and the ROM at `/workspace/rom/Pokemon Red.gb`:

```bash
scripts/start.sh                                  # setup, then train detached and show the log; rerun to reattach
scripts/setup.sh                                  # needs a mounted volume and the right ROM; asks for a W&B key if none works
scripts/train.sh                                  # foreground, fast_mode: resumes, logs to W&B (offline without a key), deletes the pod at the end
```

Locally: `uv run pokered-train --env.rom "Pokemon Red.gb" --wandb-mode offline`.

Record one episode of a checkpoint to an MP4 (needs ffmpeg), with a `.json` timeline of each milestone's step and video time.
One frame per agent step, so 60 fps plays at 24x game speed; `--stop-after` ends the video `--tail-steps` after a milestone:

```bash
uv run pokered-record --rom "Pokemon Red.gb" --checkpoints RUN_DIR/checkpoints --output video/seed0.mp4 --seed 0 --stop-after knows_cut
```

ROM tests skip when `Pokemon Red.gb` is missing from the repo root. Its SHA-1 must be
`ea9bcae617fdf159b045185467ae58b2e4a48b9a`.

## Architecture

The core is pure: emulator memory is parsed once into a frozen `GameFrame` (`src/pokered/frame.py`), and rewards
(`src/pokered/rewards.py`) see only that, so they can be tested without a ROM or emulator. `src/pokered/ram_map.py` holds addresses and typed
readers.

Rewards are a list of `RewardTerm` objects run by `EpisodeReward`, which returns each term's reward by name. Adding a term shouldn't require changing the env.

`pokered.telemetry` records what each episode did and `pokered.report` turns a window of episodes into scalars, tables and
stuck-game snapshots. Neither can affect a reward. `docs/METRICS.md` says what's logged and why: keep it in step, and
don't add a metric that doesn't answer a question there.

`pokered.vec` runs W workers x k envs in lockstep with observations in shared memory, and keeps each truncated env's
final observation so `pokered.ppo` can bootstrap its value: running out of budget never counts as the world ending. The one terminal ending is `stall_steps` (10,240) with no reward but exploration: run2 spent 54% of its late steps stalled in Cerulean Gym, because with only non-negative rewards and bootstrapped time limits, stalling cost nothing. The share of that limit used is in the `party` observation, so the deadline is visible. Observations
stay `uint8` until `pokered.model` converts them on the device. `pokered.train` is the only composition root: `train()`
takes its env, logger and store as arguments, so tests run it on fakes. Workers never import torch.

## Reward rules

| Term | Pays | Limit |
|---|---|---|
| `NewTile` | Once per `(map, x, y)`, outside battle | Tiles in the game |
| `EventFlagGain` | Once per flag, even if the game clears and re-sets it; flags already set at reset and the museum ticket pay nothing | About 491 of the 2,560 bits can be set and pay |
| `BadgeGain` | Once per badge | 8 |
| `LevelGain` | Per level gained, for each of 6 party places ranked by level, up to 5 levels over the next gym leader's ace (`LEVEL_CAP_BY_BADGES`: 19 before Brock, 26 before Misty). Each place keeps its highest level, so storing costs nothing and swapping earns nothing. A Pokemon pays nothing for the level it joins at (a bought level-5 Magikarp paid 5 levels in run2), only for levels it gains after. | 6 x 67 levels |
| `NewSpecies` | Once per species owned (Pokedex owned bits); species owned at reset pay nothing, so a second of the same species pays nothing | 151 |
| `CenterHeal` | Once per respawn point: a nurse heal that takes a hurt party to full HP, in a Pokemon Center whose respawn point (`D719`) is now this one. A heal at full HP pays nothing and leaves the center unpaid; blackouts heal on the battle map, so they never pay | One per center (12) |
| `MoveGain` (signed) | The change in the party's total move score: learning a stronger move pays, losing one costs, storing a Pokemon costs its moves, reordering pays nothing. A move scores power x accuracy (x1.5 for its user's type), half for moves that charge or recharge (Solar Beam, Skull Bash, Hyper Beam) but not Fly or Dig, and nothing for status, fixed-damage, one-hit-KO or self-fainting moves. A joining Pokemon is credited without pay | 6 Pokemon x 4 moves |
| `HiddenEventGain` | Once per hidden event the game matches when A is pressed (a PC, sign, poster or trash can with no sprite; Bill's Cell Separator is one), read by a PyBoy hook on `CheckForHiddenEvent.foundMatchingEvent`, because its RAM is shared with other variables. Exploration, not progress: it does not reset the stall count | 212 |
| `NewMap` | Once per map id entered; the map the episode starts on pays nothing. Progress: it resets the stall count, so each unentered door is worth leaving for | 248 maps |
| `FirstTime` "ss_anne_rival" | Once, when `wSSAnne2FCurScript` (`D665`) reaches 3: the S.S. Anne rival is beaten, the one story fight with no event flag (a loss resets the byte to 0) | 1 |
| `FirstTime` "cut_learner" | Once, when a party Pokemon can learn Cut (HM01 bit of its base stats, read from the ROM); a Pokemon in the PC doesn't count, since it can't be taught | 1 |
| `FirstTime` "knows_cut" | Once, when a party Pokemon knows Cut | 1 |
| `BallStock` | Per ball, for the most balls of any kind held at once, up to 5: five Poke Balls catch a full-HP Oddish 87% of the time. Throwing and buying back pays nothing | 5 balls |
| `TreeCut` | Once per tree cut (outdoor tile `3D` or gym tile `50`, not grass), told apart by where the player stood and faced. Read by a PyBoy hook on `ReplaceTreeTileBlock`, since a cut sets no flag and the tree grows back when the map reloads | Trees in the game |

- Terms return this step's reward directly. Paying the change in a running total is what made upstream's stuck penalty
  refund itself. `MoveGain` does pay a change, on purpose: run4 swapped Tackle for Bide once Bite had been paid, because
  a latched term charged nothing for it. Its payments add up to the final total minus the first, minus join credits that
  are never negative, so a replayed loop pays at most nothing, and Gen 1 has no move relearner and single-use TMs.
- The heal term is the latched form all 41 winning Rubinstein runs used. The paper's repeatable form is farmable, though its
  399-heal figure is Bulbasaur only, and removing it cost Squirtle agents events (43.2 vs 51.8). Its first job is to
  move the respawn point forward, so a blackout no longer sends the agent back to Pallet Town.
- The level cap and the weights are untested design choices. The cap doesn't account for rival fights, and the level
  term stays on because neither source's claim about removing it holds up. Total party level was dropped as a cap: it let one overlevelled starter collect the whole team's allowance.

## Traps already verified

- **The screen isn't valid straight after a restore.** Frame 1 is the saved image and frame 2 is blank, so tick at least
  3 frames after a reset before the first observation.
- **Old savestates break replay.** Upstream's `.state` files are an older format that omits LCD timing, so results
  depend on the emulator's history, from step 1. Run them through `scripts/convert_state.py` in a fresh emulator first.
- **Compare HRAM as well as WRAM in determinism tests.** The RNG bytes `FFD3` and `FFD4` diverge long before WRAM does.
- **The start state has battle animations on** (`D355` bit 7 clear) and fast text.
- **Event flags are 320 bytes** (`D747` through `D886`). Upstream's v2 reads 311.
- **Event bits are numbered lowest first.** Upstream's `f"{val:08b}"` lookup reverses names.
- **Fake observations must be nonzero.** On all-zero inputs the ReLU body outputs zero, and only the actor's bias can
  learn.
- **Battle animations off saves little by itself.** It cut only one decision (24 frames) from the first attack, and the
  start state already has fast text.
- **Blackouts:** `D057` reads `FF`, then HP fully restores and money halves. Treat as battle, never as healing. The game clears `D057` when a battle ends and only then writes `FF`, so the marker can arrive after the fight looks over: `EpisodeTelemetry` credits it to the last fight.
- **Envs that reset together stay together.** The first episode keeps a random fraction of its whole budget, event
  extensions included. Shortening only the base budget left `--init-from` runs finishing episodes in waves of about 150,
  so every batch held one part of the game.
- **Use `pyboy.stop(save=False)`.** The default writes a `.ram` file next to the ROM, and a later run would load it.
- **Don't commit the ROM.** It is gitignored, so a pod needs its own copy on the volume.
- **Evaluate with sampled actions.** The emulator is deterministic, so a deterministic policy repeats one episode.
- **`fast_mode=True` is safe on PyBoy 2.9.0, as far as tested.** About 2-2.3x the throughput on a pod. WRAM, HRAM, map, events
  and rewards matched at every decision over 345,000 steps, into Viridian Forest and through battles, menus and heals. The
  screen differs on 1-8% of steps, mostly the same picture one frame sooner. Re-test if PyBoy changes.

## GPU and Runpod

PyBoy runs on the CPU, so a GPU only speeds up the policy. Before sizing workers, pinning, or changing the model for GPU
utilization, load the `pod-training` skill (`.claude/skills/pod-training/SKILL.md`): it holds the measured throughput and the
parts of `massively-parallel-rl` to follow.

## Background documents

`docs/CONFLICT_RESOLUTIONS.md` holds evidence-based verdicts on where the reports below disagree, and wins over them. Raw
evidence is in `docs/evidence/`.

| File | Use it for |
|---|---|
| `UPSTREAM_FINDINGS.md` | Upstream defects, ranked experiments, design rationale |
| `STARTING_POINT.md` | Paper baseline, PPO settings, build order (SB3 parts don't apply: PPO is custom) |
| `PERFORMANCE_BOTTLENECKS.md` | Measured costs on the M4 Pro. Its GPU section is a projection with known errors |
| `RAM_MAP_REPORT.md` | Addresses and memory traps |

The authors' public W&B project `thatguy11325/pokemon` holds the configs of the runs that beat the game. Today's
`pokemonred_puffer` defaults are not those configs. That repo has no license file, so use its ideas but don't copy its
code. Upstream is MIT.
