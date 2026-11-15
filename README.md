# pokemon-red-ppo

A PPO agent that learns Pokémon Red on the [PyBoy](https://github.com/Baekalfen/PyBoy) emulator, rebuilt from
[PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) to see whether it can beat that baseline. The
reward pays only for first-time progress (new tiles and maps, story events, badges, capped levels, new species, one
nurse heal per center, and the steps toward Cut), so grinding can't earn it. `CLAUDE.md` lists every term and its limit.

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
uv run pokered-record --rom "Pokemon Red.gb" \
  --checkpoints runs/default/checkpoints --output video/seed0.mp4 \
  --seed 0 --stop-after got_hm01                                      # one episode to MP4 (needs ffmpeg)
```

`pokered-record` writes one video frame per agent step (24 game frames), so 60 fps plays at 24x game speed, and puts a
`.json` timeline of each milestone's step and video time next to the video. Actions are sampled and the emulator is
deterministic, so a seed always replays the same episode.

To train on a Runpod GPU pod, create a pod (image `runpod/pytorch:1.6.0-cu1300-torch2130-ubuntu2404`) with a network volume
at `/workspace` and 22/tcp exposed, then run this from your machine, with the pod's public IP and external SSH port:

```bash
POKERED_RUN_NAME=run1 scripts/launch_pod.sh HOST PORT     # add --init-from DIR to start from another run's weights
```

`POKERED_RUN_NAME` is the W&B run id and the run directory on the volume; unset, it is `main`. Options after `PORT` go to
`pokered-train`.

It clones the repo onto the volume, uploads the ROM if the volume has none, installs everything, and asks once for your
W&B key, which it checks and stores. Training then runs detached, so you can close the connection or sleep your machine.
It resumes from the newest checkpoint on the volume, and the pod deletes itself when training ends, while the volume and
its checkpoints stay. Run the same command again to watch a running pod. Already on the pod? `scripts/start.sh` does the
same from there.

## Status

Nine training runs on one A100 each, one seed apiece, most of them 400M steps (run5 ended at 258M when the account ran
out of credit, and run7 was stopped after 2M). The best policy (run9, started from run6's weights)
beats Brock and Misty, gets the S.S. Ticket, beats the rival on the S.S. Anne and gets HM01 in 99% of training episodes
and 63 of 64 frozen ones, and buys Poke Balls. It never catches a Pokemon that can learn Cut: it meets an Oddish or Paras
with balls in hand in most episodes but almost never throws one, so it never cuts the tree in front of Vermilion Gym.
Each run's analysis is in `docs/evidence/`; `CLAUDE.md` summarizes them.

`CLAUDE.md` describes the architecture, and `docs/CONFLICT_RESOLUTIONS.md` holds the measurements behind the design.

## Credits

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

MIT: see [LICENSE](LICENSE). The ROM isn't covered and isn't distributed here.
