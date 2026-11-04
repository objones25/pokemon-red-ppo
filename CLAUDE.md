# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Goal

Train a PPO agent to play Pokemon Red on the PyBoy emulator, as a rebuild and extension of
[PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments), and measure whether it can beat that
baseline. Progress means **exploration and story events**. Battling and training Pokemon are rewarded only as far as
they serve that progress: the agent must not be able to earn reward by grinding.

## Status

Built and tested: start state, Gymnasium env (battle animations off), parallel workers with CPU pinning, model, custom
PPO, atomic checkpoints with resume, W&B logging, pod shutdown, `scripts/setup.sh` and `scripts/train.sh`.

A 3M-step local run (one seed, `docs/evidence/learning/`) reached 693 unique tiles and 4 events per episode against 73 and
0 for random actions, though its episodes ran twice as long because events extend the budget, and it gained no levels.

Not built yet: a fixed evaluator and milestone tracking. `scripts/setup.sh`, `scripts/train.sh` and CPU pinning have not
run on a real pod. No full training has run. Keep this section and **Commands** in step with the code, and do not
document commands that do not exist.

## Requirements

- **Start state:** `states/has_pokedex_squirtle.state`: Squirtle chosen, Pokedex received, in Oak's lab.
- **No farmable reward:** every term pays at most once for the same thing, so each has a fixed lifetime maximum per
  episode. Never add a repeatable term (heals, XP, money, battles, menu use). See **Reward rules**.
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

On a pod, with the repo on the `/workspace` volume and the ROM at `/workspace/rom/Pokemon Red.gb`:

```bash
scripts/setup.sh                                  # needs a mounted volume and the right ROM; asks for keys
scripts/train.sh                                  # resumes, logs to W&B, deletes the pod when training ends
```

Locally: `uv run pokered-train --env.rom "Pokemon Red.gb" --wandb-mode offline`.

ROM tests skip when `Pokemon Red.gb` is missing from the repo root. Its SHA-1 must be
`ea9bcae617fdf159b045185467ae58b2e4a48b9a`.

## Architecture

The core is pure: emulator memory is parsed once into a frozen `GameFrame` (`src/pokered/frame.py`), and rewards
(`src/pokered/rewards.py`) see only that, so they can be tested without a ROM or emulator. `src/pokered/ram_map.py` holds addresses and typed
readers.

Rewards are a list of `RewardTerm` objects run by `EpisodeReward`, which returns each term's reward by name. Log those
per-term values. Adding a term shouldn't require changing the env.

`pokered.vec` runs W workers x k envs in lockstep with observations in shared memory, and keeps each truncated env's
final observation so `pokered.ppo` can bootstrap its value: a time limit never counts as the world ending. Observations
stay `uint8` until `pokered.model` converts them on the device. `pokered.train` is the only composition root: `train()`
takes its env, logger and store as arguments, so tests run it on fakes. Workers never import torch.

## Reward rules

| Term | Pays | Limit |
|---|---|---|
| `NewTile` | Once per `(map, x, y)`, outside battle | Tiles in the game |
| `EventFlagGain` | Once per flag, even if the game clears and re-sets it; flags already set at reset and the museum ticket pay nothing | 2,560 flags |
| `BadgeGain` | Once per badge | 8 |
| `LevelGain` | Per party level, up to the next gym leader's team total (`LEVEL_CAP_BY_BADGES`) | 292 total levels |

- Terms return this step's reward directly. Paying the change in a running total is what made upstream's stuck penalty
  refund itself.
- There is no heal term yet. The paper's repeatable form is farmable, though its 399-heal figure is Bulbasaur only, and
  removing it cost Squirtle agents events (43.2 vs 51.8). The first reward experiment is a latched nurse heal, the form
  all 41 winning Rubinstein runs used.
- The level cap and the weights are untested design choices. The cap doesn't account for rival fights, and the level
  term stays on because neither source's claim about removing it holds up.

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
- **Blackouts:** `D057` reads `FF`, then HP fully restores and money halves. Treat as battle, never as healing.
- **Use `pyboy.stop(save=False)`.** The default writes a `.ram` file next to the ROM, and a later run would load it.
- **Don't commit the ROM.** It is gitignored, so a pod needs its own copy on the volume.
- **Evaluate with sampled actions.** The emulator is deterministic, so a deterministic policy repeats one episode.
- **`fast_mode=True` is unproven.** It is about 2.85x faster and game memory matched for 20,000 steps, but only around
  Pallet Town and Route 1, and some screen pixels differ. Validate it on real trajectories before using it.

## GPU and Runpod

PyBoy runs on the CPU, so **a GPU only speeds up the policy**. Measured on an A100 SXM with an AMD EPYC 7742
(`docs/CONFLICT_RESOLUTIONS.md` §8): a Nature CNN reaches 4–7% GPU utilization and at best 2,966 steps/s, about 11% above
the M4 Pro, because each emulator step takes about 6.4 ms against about 2.1–2.6 ms. A larger model or more epochs raise
utilization but cost throughput and have no proven learning benefit, so don't add them for utilization alone.

On a pod, read the real CPU limit from the cgroup rather than the API (32 vCPUs reported, 27.2 usable on that host), use
one worker per usable core at most (26 there), and pin workers to physical cores on the GPU's NUMA node (`--pin-workers`).

The user's separate `massively-parallel-rl` project (`~/Desktop/CS291K/massively-parallel-rl`) is the working example. Follow these parts of it:

- `scripts/setup.sh`, `scripts/train.sh`, `docs/runbook.md`: the volume-mount check, a venv and uv cache on the
  container disk (never the volume), and `.env` prompts.
- `src/mprl/runpod.py`: `PodTerminator` (read, delete, confirm gone).
- `src/mprl/checkpoints.py`, `run_state.py`: atomic checkpoints and resume state.
- `src/mprl/wandb_writer.py`: a fixed run id with `resume="allow"`. Its notes record that `resume_from` is a private
  preview, and that resume must re-randomize episode lengths or all envs reset together.
- Don't copy its versions (torch 2.8.0, its base image) without re-checking them.

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
