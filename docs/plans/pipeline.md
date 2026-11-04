# Training pipeline plan

Goal: train from `states/has_pokedex_squirtle.state`, battle animations off, on a Runpod GPU, resuming from checkpoints on
the `/workspace` volume and logging to W&B. Every module is written test-first.

| Stage | Module | Done when |
|---|---|---|
| 1 | `emulator.py`: only `pyboy` import; open, restore with options and settle | Restored screen is valid; animation bit set |
| 1 | `observations.py`, `frame.py`: uint8 screens, visited map, party and flags | Shapes, values and crop edges pinned |
| 1 | `env.py`: Gymnasium env, step budget, staggered first episode | Time limit truncates, never terminates |
| 2 | `vec.py`: W workers x k envs, shared memory, final observation kept for bootstrap | Same results as stepping envs directly |
| 3 | `model.py`: two Nature CNNs plus state MLP, raw logits, uint8 converted on device | Overfits one batch |
| 3 | `ppo.py`: rollout, GAE, clipped update, health metrics | GAE matches hand values and the `rl_math.py` oracle |
| 4 | `checkpoints.py`: atomic files, CRC fallback, full RNG and counters | Kill-and-resume restores everything |
| 4 | `wandb_logger.py`: fixed id, `resume="allow"`, offline fallback | Offline round trip |
| 4 | `train.py`, CLI, `config.py` | Local run beats a random policy on unique tiles |
| 5 | `evaluate.py`: fixed start, sampled actions, milestones | Same evaluator for every checkpoint |
| 5 | `scripts/setup.sh`, `scripts/train.sh`, pod shutdown, Runpod volume checks | Shellcheck clean, fake-tool tests pass |

Stages 1-4 and the stage 5 scripts are built. The evaluator isn't, and the scripts haven't run on a real pod.

Defaults come from Pleines et al. Table I and `docs/CONFLICT_RESOLUTIONS.md`: gamma 0.997, lambda 0.95, clip 0.2, value
clipping on, AdamW lr 3e-4, entropy 0, grad norm 0.5, T 2048, 3 epochs, minibatch 8192, 8 envs per worker. Worker and env
counts follow the pod benchmark.
