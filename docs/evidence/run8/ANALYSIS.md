# run8: analysis

run8 trained from scratch (random weights, no `--init-from`) with the Cut-chain rewards (`FirstTime` ss_anne_rival,
cut_learner and knows_cut; `BallStock`; `TreeCut`) and `ent_coef` 0. It ran 400M steps on one A100 in 15.3 hours (about
7,270 steps/s on average) and ended normally. One seed. run7, a warm start from run6's weights, was stopped after about
2M steps and is not analyzed.

Sources: W&B history (977 rows, 80 reports) and all report tables, downloaded to `runs/analysis/run8/` (gitignored); the
400M tables are copied here (`*_400M.table.json`); the 400M stuck screens; the 320 stall states, read from RAM
(`stall_states.json`, `read_stall_states.py`). No episodes were traced.

## Headline

| | run6 (from run5's weights, 400M) | run8 (from scratch, 400M) |
|---|---|---|
| Events per episode at the end | about 71 | 47 (at most 53) |
| Misty | about 100% | 94% at 400M, first above 50% at 185M |
| S.S. Ticket | 92-100% | 0 in every report |
| Boarded the S.S. Anne | median 97% | 0 |
| HM01 | median 1% | 0 |
| Holds a Poke Ball | 0 | 92-99% from 325M |
| Caught a Pokemon | no | no: party is Wartortle and a level-5 Magikarp |
| Cut learner, knows Cut, tree | 0 | 0 (none of those reward terms ever paid) |
| Where episodes end at 400M | Bike Shop 78% | Bill's house 50%, Route 4 18%, Cerulean Gym 11% |

run8 got as far as Bill's house in the last few reports and never got past it.

## Progress over training

| Steps | What happened | Where most episodes ended |
|---|---|---|
| 0-45M | Pewter; Brock first beaten at about 45M; Poke Balls held by 96% at 45M, then dropped to 1-9% | Viridian, Route 2, Pewter |
| 65-155M | Brock about 100%; Mt. Moon; events 18-28 | Pewter Museum, Mt. Moon, Route 4 |
| 165-205M | Misty first beaten, 13% at 165M, 76% at 185M; events peak at 41 then fall back to 32 | Cerulean Gym 76-94% |
| 210-305M | Misty 89-100%; events flat at about 32 | **Cerulean Gym 94-100%**, entropy there 0.03-0.12: the agent beats Misty and parks in her gym |
| 320-365M | Leaves the gym; Nugget Bridge, Route 24 Rocket, Route 25's trainers; buys balls (about 5 held, the cap); events 44-49 | Cerulean City, Route 4, Cerulean Mart |
| 395-400M | Bill's house becomes the last stop | **Bill's house 50%** |

## Why it stopped at Bill

- In Bill's house it sets **none** of Bill's flags. run6 set `0x55E`, `0x55B`, `0x550`, `0x55D`, `0x55C` and `0x55F` in 92%
  of episodes within 100 steps of each other, at a median step of about 11,900. In run8, `0x55E` (the first of them)
  appears in at most 0.8% of episodes in any report, and the others never do.
- The stall at 400M is on Bill's-house tile (0,4) with the party's "ABLE / NOT ABLE" item menu open (stuck screen), with
  high entropy there (1.16, against 0.3 in the Route 4 stall). The policy is wandering and opening menus, not talking to Bill.
- Talking to Bill pays event flags directly, so the reward is there; the frontier opened only around 395M, and the run ended
  before the policy learned it. The run5/run6 lineage took most of run5 (S.S. Ticket at 90-98% only from 215M) to learn
  the same thing from run4's weights.

## The Cut chain

None of the new terms that sit beyond Bill could pay: `ss_anne_rival`, `cut_learner`, `knows_cut` and `trees` never
appear in the logged rewards. The ball term worked as intended: about 4.7 balls held (reward 1.87 at weight 0.4) from
325M, bought at the Cerulean Mart. They were never used to catch anything: of the 164 stall states from 200M, 100 have a second Pokemon and it is a
Magikarp in 98 (in the other 2 the Magikarp leads and Wartortle is second); none has a third. 100 of the 164 hold
balls, up to 34. The species reward (1.9) is the evolution plus the Magikarp.

## Other observations

- **210-305M, a 95M-step park in Cerulean Gym.** The same habit as run5's Fan Club and run6's Fan Club: once the next
  reward is out of reach, the policy stops where it was last paid. With `ent_coef` 0 its entropy in the gym fell to 0.03.
  It left the gym only at about 320M.
- Blackouts 7-9 per episode from 85M; the most fought is Misty (2.3 fights per episode at 400M, lost 60%).
- `Holds a Poke Ball` at 96% at 45M and then 1-9% until 325M: an early ball the policy picked up and later stopped
  picking up. Not investigated (no traces).
