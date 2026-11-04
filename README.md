# pokemon-red-ppo

A PPO agent that learns Pokémon Red on the [PyBoy](https://github.com/Baekalfen/PyBoy) emulator, rebuilt from
[PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) to see whether it can beat that baseline. The
reward pays only for first-time progress (new tiles, story events, badges and capped levels), so grinding can't earn it.

**The ROM isn't included.** You need your own legally obtained copy of Pokémon Red, with SHA-1
`ea9bcae617fdf159b045185467ae58b2e4a48b9a`, at the repo root as `Pokemon Red.gb`.

## Run it

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
scripts/check.sh                                                     # type checks, lint, audits and tests
uv run pokered-train --env.rom "Pokemon Red.gb" --wandb-mode offline  # train; resumes from the latest checkpoint
uv run pokered-evaluate --rom "Pokemon Red.gb" \
  --checkpoints runs/default/checkpoints --output eval.json           # score the latest checkpoint
```

To train on a Runpod GPU pod, create a pod (image `runpod/pytorch:1.6.0-cu1300-torch2130-ubuntu2404`) with a network volume
at `/workspace` and 22/tcp exposed, then run this from your machine, with the pod's public IP and external SSH port:

```bash
scripts/launch_pod.sh HOST PORT
```

It clones the repo onto the volume, uploads the ROM if the volume has none, installs everything, and asks once for your
W&B key, which it checks and stores. Training then runs detached, so you can close the connection or sleep your machine.
It resumes from the newest checkpoint on the volume, and the pod deletes itself when training ends, while the volume and
its checkpoints stay. Run the same command again to watch a running pod. Already on the pod? `scripts/start.sh` does the
same from there.

## Status

The pipeline trains and resumes, but no full training run has finished. A 3M-step local run reached 693 unique tiles and 4
story events per episode, against 73 and 0 for random actions, on one seed, with its episodes running twice as long.

`CLAUDE.md` describes the architecture, and `docs/CONFLICT_RESOLUTIONS.md` holds the measurements behind the design.

## Credits

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

MIT: see [LICENSE](LICENSE). The ROM isn't covered and isn't distributed here.
