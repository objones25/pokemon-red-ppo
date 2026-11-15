# run9: analysis

run9 started from run6's 400M-step weights (`--init-from`) with the Cut-chain rewards (`FirstTime` ss_anne_rival,
cut_learner and knows_cut at 5.0 each; `BallStock` at 0.4 per ball up to 5; `TreeCut` at 5.0) and `ent_coef` 0. It
trained 400M steps on one A100 80GB PCIe in 20.5 hours (about 5,430 steps/s on average, $2.09/hr, about $43) and ended
normally; the pod deleted itself. One seed. It was the last training run.

Sources, all in this folder unless noted:

- W&B history (977 rows, 80 reports) and every report's tables, downloaded to `runs/analysis/run9/` (gitignored). The
  400M tables are copied here (`*_400M.table.json`).
- The 320 stall states the run saved, read from RAM: `stall_states.json` (`read_stall_states.py`). At most 4 per report,
  one per map, so they show variety, not frequency.
- 64 episodes of the final 400M checkpoint, frozen, sampled actions, seeds 0-63, traced step by step on the Mac:
  `traces_400M_64_episodes.tar.gz` (`trace_episode.py`, the run6 script unchanged).
- The deliverable video: seed 0, `pokered-record --stop-after got_hm01 --tail-steps 2500`. The trace and the recording of
  seed 0 reach HM01 at the same step (14,111), so a seed reproduces its episode.

## Headline

| | run6 (start weights) | run9, training at 400M | run9, frozen, 64 episodes |
|---|---|---|---|
| Events per episode | about 71 | 82 (peak 92 at 160-175M) | median 80.5 rewarded flags |
| Boarded the S.S. Anne | median 97% | 99% | 63/64 |
| Beat the S.S. Anne rival | 1/15 fights frozen | 99% of episodes | 63 of 66 fights won |
| **HM01** | median 1% | **99%** (93-100% from 155M) | **63/64**, median step 14,545 (12,357-24,448) |
| Bicycle | 87% at 400M | 98% | 63/64 |
| **Holds Poke Balls** | 0 | **98%** (91-99% from 215M) | 63/64 buy them, median 18 per episode, at most 53 |
| Threw a ball at a Cut learner | - | - | **0** (14 throws in 64 episodes, none at a catchable Cut learner) |
| Cut learner in party, knows Cut, tree cut, Vermilion Gym | 0 | 0 | 0 |

The one frozen episode without HM01 (seed 22) walked into Pewter Gym at step 1,238 and stayed there, without fighting
Brock, until the stall limit ended it at 11,419.

## The rival and HM01

| Steps | S.S. Anne rival fights per episode | Rival wins (party blacks out) | HM01 |
|---|---|---|---|
| 0-145M | 0.14-0.28 | 92-100% | 0-6% |
| 150-205M | 1.5-1.7 | 36-44% | 75% at 150M, 91% at 155M, then 93-99% |
| 225-400M | 1.04-1.12 | 8-12% | 95-100% |

For 145M steps run9 behaved like run6: it cleared the cabins first and met the rival rarely and weak. Then, within 10M
steps, it started fighting him more than once per episode, and HM01 followed. This is the pattern the run6 analysis
predicted: the policy could beat a healthy rival (119 of 120 replays), and run6 had nothing paying for the fight. The
latched rival term (5.0, once) is the only change to the reward between run6 and run9 on this part of the game, so it
is the likely cause; with one seed and no ablation, that is not proven.

Training rates and frozen rates agree here (99% and 63/64), as they did for run6.

## Poke Balls: bought, hoarded, almost never thrown

- **Where:** the first ball comes from the Cerulean Mart in 49 of 63 episodes, the Viridian Mart in 12 (before Brock) and
  the Vermilion Mart in 2. Training history: balls held in under 13% of episodes until 200M, 43% at 210M, 91% at 215M.
  (The trace's `first_enemy` field is read at the start of a fight and can still hold the previous fight's Pokemon:
  Brock's fight lists a Weedle first. Encounter counts below are given with and without the fights whose first enemy
  equals the previous fight's last one.)
- **How:** the agent talks to the clerk and keeps pressing A. Poke Ball is first on every mart's list and every prompt
  defaults to buying one, so each loop buys one ball, about every 9 steps (seed 0: 9 balls in 73 steps at the Cerulean
  Mart, money 10,682 to 9,082).
- **How many:** the term pays for up to 5. The policy keeps buying: a median of 18 per frozen episode, at most 53, and up
  to 77 held in late stall states (median 19). Extra balls cost money and pay nothing; nothing in the reward rules
  forbids it, and money has no other use to this agent.
- **Where they sit in the bag:** in late stall states the ball stack is in slot 10 or 11 of a 13-item bag in 108 of 147
  cases, under the key items (Town Map, Helix Fossil, S.S. Ticket, HM01, Nugget, three TMs, Bicycle). In battle,
  throwing one is ITEM, then Down about 9-10 times, then A.
- **Throws:** 14 in 64 frozen episodes, in 6 episodes. 8 were in trainer battles (Brock, a Route 3 Bug Catcher and
  Youngster, a Route 6 Jr. Trainer), where a ball cannot catch anything. 2 were at a wild Kakuna in Viridian Forest; both failed. 4 happened
  outside any fight the trace recorded; the cause is not established.
- **Targets were there:** with balls in the bag, the agent met 2,289 wild Pokemon (35.8 per episode). 138 of those were
  Pokemon that can learn Cut: Oddish on Route 6 (60), Route 24 (20), Route 5 (18) and Route 25 (4), and Paras in Mt. Moon
  (36). Dropping the 23 whose first enemy could be left over from the previous fight leaves 115, still in 45 of 64
  episodes, all on maps whose wild tables have these species (`docs/evidence/run6/RESEARCH_CUT.md`). It threw at none
  of them.
- **Catches:** 4 of 164 late stall states (from 200M) have a third Pokemon: Weedle three times, plus a Pikachu once.
  Neither can learn Cut (Weedle's evolution Beedrill can, two evolutions later). `reward/cut_learner` never paid in
  training.

**Why the chain stopped at the throw:** every earlier step of the chain was reachable by actions the policy already took
or that the game defaults made easy (fight the rival, press A at a clerk). Throwing a ball needs a precise in-battle menu
sequence that pays nothing on its own, then a 34% chance per ball on a full-HP Oddish, and only then does a reward
arrive (the species term, then `cut_learner`). With `ent_coef` 0 and a policy already settled on its battle habits,
that sequence was never found in a way PPO could reinforce.

## What the agent does after HM01

The policy leaves the ship with HM01 at a median step of about 14,500 and plays on for a median of 14,600 more steps
before the stall limit. In nearly every frozen episode it walks the S.S. Anne decks once more, then goes to Vermilion,
the Fan Club (61/64), Route 6 and the Underground Path back to Cerulean for the bicycle, and east to Route 11 (62/64)
and Route 12 (52/64). It never enters Vermilion Gym: the tree blocks it, and cutting it needs the Pokemon it never caught.

Where episodes ended in training at 400M: Cerulean Gym 31%, Cerulean City 28%, Vermilion City 11%, Route 12 7%. Frozen:
Cerulean City 19/64, Route 12 10, Route 11 8, Vermilion City 8, Cerulean Gym 5.

## Training phases (where most episodes ended)

| Steps | Ended mostly in | What changed |
|---|---|---|
| 0-25M | Cerulean City, Bike Shop, Route 25 | The new terms reshuffle run6's late game; HM01 0-4% |
| 30-125M | Route 25, Route 4, Cerulean Mart (71% at 80M), Route 5 | Explores around Cerulean; events about 74 |
| 125-130M | Diglett's Cave (72%) | A long detour through Diglett's Cave, 108 wild fights per episode at 130M |
| 150-155M | S.S. Anne 2F/3F/bow/B1F | Starts fighting the rival; HM01 75-91% |
| 175-180M | Vermilion City (96%) | Parks right after HM01: the last reward |
| 200-400M | Route 5, Cerulean, Route 11, Route 24/25, Cerulean Gym | Buys balls from 205M; HM01 and balls both at 95-99% |

## Blackouts

9.2 per frozen episode (median 9), mostly early: `YOUNGSTER#1` on Route 3 (90 in 64 episodes), Brock (66), Misty (43),
the Lass on Route 4 (37). As in run6, losing costs only money, which the agent spends only on balls.

## Run-by-run summary of the Cut chain

| Step | run6 | run8 (from scratch) | run9 (from run6) |
|---|---|---|---|
| S.S. Ticket | 92-100% | 0 | 99% |
| Beat the S.S. Anne rival | 1/15 frozen fights | 0 | 99% |
| HM01 | median 1% | 0 | 99% (63/64 frozen) |
| Poke Balls | 0 | 92-99% from 325M | 98% |
| Catch a Cut learner | 0 | 0 | 0 |
| Teach Cut, cut the tree | 0 | 0 | 0 |
