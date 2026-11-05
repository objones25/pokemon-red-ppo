# Frozen versus updating policy, from the first run's final checkpoint

Question: training reported Misty in about 74% of episodes near 400M steps, but the same weights, frozen, beat her in
25 of 96 first episodes (`../a100/`). Does training's score depend on the weights changing during an episode?

## Setup

- Both arms start from `checkpoint_000400015360.pt` (weights and AdamW state) with `--init-from`, seed 0, the same
  training collection code (`collect_rollout`, bf16 autocast, sampled actions), 30M steps each.
- Only difference: `--no-update-policy` skips `ppo_update` in the frozen arm.
- RTX 4090 pod, Ryzen 9 7950X, 13.6 usable CPUs: 12 workers x 13 envs = 156 envs, batch 319,488 (training: 152 envs,
  311,296). About 9,800 steps/s; 52 minutes per arm; pod self-deleted. W&B runs `test-frozen` and `test-updating`;
  reports in `/workspace/pokered/runs/test-*/reports/` on the volume.

## Result

Per 5M report window (episodes that finished in it):

| Window ends | Frozen: episodes, Misty | Updating: episodes, Misty |
|---|---|---|
| 10.2M | 8, 0% | 1, 0% |
| 15.0M | 147, **39.5%** | 162, **84.0%** |
| 20.1M | 6, 0% | 0, - |
| 25.2M | 144, 25.0% | 128, 30.5% |
| 30.0M | 8, 75% | 28, 89% |
| **All** | **314, 31.8%** (100) | **319, 62.7%** (200) |

The 15.0M window holds each env's first episode, all started from the same weights: updates during the episode
(about 25 updates by the median Misty step, about 40,000) took Misty from 39.5% to 84%. That gap is far outside
sampling noise at these counts. The updating arm's mean approx KL was 0.015 and clip fraction 0.10 over 94 updates,
ordinary values: no update was broken.

The frozen arm's 31.8% agrees with the frozen evaluator's 26% (`evaluate.py`, 96 episodes), so the evaluator is not
the cause.

## Caveats

- Both arms' episodes finish in waves (147, 6, 144 per window), because every env started from reset at once and
  the first-episode stagger spreads only up to `base_budget` (10,240) of about 80,000 steps. The first run did not:
  from 340M to 400M, 1 to 9 episodes finished at every one of 188 updates. So all envs in an arm meet Misty at about
  the same time and within a few updates of each other, which training's mixed batches never do. The updating arm's
  second wave (30.5%) is unexplained and may come from that.
- One seed per arm.
- The new level reward (per place, capped at the gym ace) applied to both arms, and changes the updating arm's
  gradient slightly from training's.

## What it means

Training-time rates measure the policy and its ongoing learning together, and a fixed snapshot of the weights is
much weaker. The policy has no recurrent state: it sees three screens, a visited-tile map around the player, the party
and the event flags, so nothing records that it already lost to Misty five times or walked a loop on an already-visited
route. A frozen policy that falls into such a loop (Route 22, Viridian, Misty's gym) can stay in it, while the updates,
driven by that same experience, can push it out. That explanation fits the stall locations in `../a100/` but is not
tested.
