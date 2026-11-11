# run6: analysis

run6 started from run5's 255M-step weights (`--init-from`) and added `NewMap` (once per map id, resets the stall count).
It trained 400M steps on one A100 80GB PCIe in 19.2 hours (about 5,800 steps/s on average, fast mode, 29 workers x 8
envs, $1.79/hr, about $34) and ended normally; the pod deleted itself. One seed.

Sources, all in this folder unless noted:

- W&B history (842 rows) and all 80 reports' tables, downloaded to `runs/analysis/run6/` (gitignored). The 400M tables are
  copied here (`*_400M.table.json`).
- The 320 stall states the run saved, read from RAM: `stall_states.json` (`read_stall_states.py`). The volume keeps 4 per
  report, at most one per map, so they show variety, not frequency. Use `maps` `end_share` for frequency.
- 64 episodes of the final 400M checkpoint, frozen, sampled actions, seeds 0-63, traced step by step:
  `traces_400M_64_episodes.tar.gz` (`trace_episode.py`). Each trace lists every map change, fight (opponent, party HP and
  PP before and after, blackout), event flag, reward payment and bag change. Frozen rates are not training-time rates
  (CLAUDE.md, first run), but here they agree: training at 400M had S.S. Anne 91% and HM01 2%; frozen, 60/64 and 1/64.
- `rival_fight_test.py` and `rival_fight_test.json`: the 15 saved rival-fight starts, each replayed 8 times as saved and 8
  times with the party fully healed.
- Game facts: `RESEARCH_CUT.md` (a research agent's notes from pret/pokered, the symbols file, pokemonred_puffer and
  PokemonRedExperiments). Claims used below were re-checked against the pret source or by experiment; the ones that were
  re-checked are marked "checked".

## Headline

| | run5 (255M) | run6 (400M) |
|---|---|---|
| Events per episode | 60-64 | median 71 from 45M (49-76) |
| S.S. Ticket | 90-98% | 92-100% |
| Boarded the S.S. Anne | at most 3% | median 97% from 45M (62-100%) |
| HM01 | about 0 | 26% at 35M; median 1% from 45M, at most 16% (75M); 2% at 400M |
| Knows Cut | 0 | 0 |
| Bicycle | 0 | 0 until 390M, 87% at 400M |
| Where late episodes end | Fan Club 89-94% | Fan Club, median 94% of episode ends from 55M to 355M; Bike Shop 78% at 400M |

The map term got the agent onto the ship. It did not get it to HM01, and it did not remove the habit of parking where the
last reward was paid.

## One episode, step by step (the frozen policy's typical run)

Medians over the 64 traced episodes; the counts say how many episodes reached each point.

| Step (median) | Reached | What happens |
|---|---|---|
| 9 | 64 | Leaves Oak's lab |
| 2,048 | 64 | Beats Brock. Squirtle fights him at level 7-8 (his Onix is 14) and wins 64 of 145 attempts (44%) |
| about 3,800 | 59 | Squirtle evolves into Wartortle on Route 3; evolution sets a Pokedex owned bit, which pays the species term |
| about 4,300 | 55 | Buys a Magikarp at the Mt. Moon Pokemon Center (Route 4), the second species payment. It stays level 5 all episode |
| 8,328 | 64 | Reaches Cerulean |
| 9,789 | 62 | Beats Misty, Wartortle level 22-23, 62 of 135 attempts (46%) |
| 10,183 | 62 | Beats the rival on the bridge north of Cerulean |
| 10,339 | 62 | First Route 24 trainer (Nugget Bridge), then Route 25's trainers on the way to Bill |
| 12,037 / 12,162 | 62 | Bill's house (no fight), S.S. Ticket |
| 12,435 | 62 | Beats the Rocket in the trashed house |
| 13,491 / 13,749 | 61 / 60 | Vermilion, boards the S.S. Anne |
| 15,735 | 60 | First blackout aboard, after a median of 13 cabin-trainer fights (5-14) |
| 15,901 | 61 | Fan Club: Pikachu fan, Seel fan, Bike Voucher |
| 17,636 | 54 | Back to Cerulean through the Underground Path, beats the Route 24/25 trainers it skipped, gets the bicycle |
| about 27,000-30,000 | | The stall limit (10,240 steps with no progress) ends the episode: Bike Shop 38/64, Cerulean City 17/64 |

Blackouts: 10.5 per episode, 9.5 of them before reaching Vermilion. The most frequent: `YOUNGSTER#1` on Route 3 (1.81 per
episode, lost 116 of 180 fights), Brock (1.27), Misty (1.14). A blackout returns the party fully healed to the last
center and halves the money, which the agent never spends.

The S.S. Anne in detail:

- Each cabin trainer sets one event flag (2.0). The ship has no Pokemon Center, and the agent never walks back to
  Vermilion's to heal between fights.
- All 60 boarding episodes black out aboard. The first blackout is to a cabin trainer in 47 and to the rival in 13.
  Wartortle starts the losing fight at a median of 14% HP; in 46 of the 60, Bite is at 0 PP (Water Gun and Bubble still
  have PP). Wartortle faints, then Magikarp is sent out and faints.
- The rival (2F; the fight starts on tile (36,8) in 14 fights, (37,8) in 1, checked against `scripts/SSAnne2F.asm`) was
  fought 15 times in 64 episodes and beaten once. The losses started with Wartortle at 3-60% HP.
- After waking up at the Vermilion Pokemon Center, 59 of the 60 go to the Fan Club next. Only 3 of 60 ever board again.

The one episode that got HM01 (seed 3): blacked out aboard at 14/84 HP; spent about 6,000 steps pacing Route 6 and the
Underground Path and walking in and out of the Fan Club; went back aboard at full HP (86/86), cleared only the 2F rooms
and the bow, met the rival at 45/89 HP, won in 5 turns, and got HM01 350 steps later. Nothing pays after HM01, so it
wandered 2F until the stall limit.

## Why so few episodes got HM01

**The policy can beat the rival; it arrives too weak.** The rival's team for a Squirtle player is Pidgeotto 19, Raticate 16,
Kadabra 18, Ivysaur 20 (checked: `data/trainers/parties.asm`, Rival2 team 2, chosen in `scripts/SSAnne2F.asm` by
`wRivalStarter`). Replaying the 15 saved rival-fight starts with the frozen 400M policy, 8 times each:

| Party at the start of the fight | Wins |
|---|---|
| As saved | 11 / 120 (9%) |
| Fully healed (HP and PP to maximum, status cleared) | 119 / 120 (99%) |

Why it arrives weak:

1. **Losing costs nothing but money.** A blackout is a free full heal and a teleport to the center. The policy treats
   fights as retries: Brock at level 7, every cabin on the ship at 14% HP.
2. **The rival fight pays nothing; every cabin trainer pays 2.0.** Beating the S.S. Anne rival sets no event flag (checked:
   `scripts/SSAnne2F.asm` has no `SetEvent`; a win only advances `wSSAnne2FCurScript` and hides the rival sprite). The
   only reward behind him is the captain's two flags (4.0), 350 steps later in seed 3. From about 45M the agent clears
   every cabin block before the rival.
3. **After the blackout, the ship looks worse than the Fan Club.** Healed, it would win 99% of the time, but the rival is a
   walk away and pays nothing directly, while three easy events sit next door. 3 of 60 go back.

Training-time history agrees. At 25-40M the rival was fought 0.4-1.0 times per episode and lost 48-72% of the time, and
HM01 reached 26% at 35M. Cabin fights per episode rose from 3.6 at 25M to about 7 at 35-45M and 11-12 from about 190M; the
rival loss rate stayed mostly at 80-100% from 45M, and HM01 stayed at a median of 1%.

## Where it got stuck, and why

| Steps | Where most episodes ended | Why |
|---|---|---|
| 10-65M | Fan Club (46-95%), Captain's Room (up to 41% at 30M) | The Fan Club pays 3 events, then nothing nearby. In the Captain's Room, HM01 is the last reward on the trail |
| 55-355M | Fan Club (median 94%; below 85% in 8 of 61 reports, lowest 33% at 70M when 53% ended in Vermilion City) | After the ship blackout: Fan Club, then nothing |
| 360-390M | Route 6, Route 24, Route 25, Cerulean | Entropy rose from about 0.3 to 0.8 and the agent started walking north after the Fan Club, picking up the leftover Route 24/25 trainers |
| 395-400M | Bike Shop (56%, then 78%) | The bicycle is the new last reward |

The pattern is the same as run5's: once the next reward is out of reach, the policy stops where it was last paid and the
stall limit ends the episode. The map term moved that point from Vermilion to the ship and back to the Fan Club, but every
map near the end of the trail is already visited.

## Metric oddities

- **`lost_fights_mean` equals `blackouts_mean` in every row.** By design: telemetry counts a fight as lost only when the
  blackout marker follows it.
- **Dip at 175M** (events 59.5, Misty 0.80) and **dip at 255M** (events 48.8, Misty 0.62, party size 1.56, max level 23):
  both are early-game regressions with the S.S. Anne untouched. At 255M, 24% of episodes ended in Pewter Museum 2F and 6%
  in Pewter Gym; at 175M, Museum 2F and Pewter City took 6% each. Both coincide with entropy rising for 5-10M steps
  (0.30 to 0.54 at 175M, 0.27 to 0.65 at 249-253M) and settling back. PPO was otherwise stable through both: approx KL
  0.005-0.007, clip fraction 0.05-0.08, explained variance 0.99, gradient norm 0.25-0.35. The largest single-update KL
  (`approx_kl_max`) was 0.171 at 264M, once, after the dip had already recovered.
- **Rival fights rose to 0.5-0.7 per episode at 340-375M** while the rival loss rate stayed above 90%: more returns to the
  ship in that window, not better fighting.
- **Bicycle 0 to 87% in the last 10M steps**: the 360M behavior change above.
- **Steps per second dipped to about 4,800 around 105M.** Cause not established (host load is the likely one); no visible
  effect on learning.

## BubbleBeam

Stall states knowing BubbleBeam (TM11, from Misty): run4 75/320 (23%), run5 0/204, run6 9/320 (3%) and 0/64 frozen
episodes, while 173/320 run6 states still carry TM11. The change coincides with the move term becoming signed between run4
and run5: teaching a TM to a four-move Wartortle forces it to forget one, and forgetting Bite or Water Gun is charged, so
fumbling the menu often costs more than it pays. One seed per run, so this is a likely cause, not a proven one.

The agent does use the bag: in 64 episodes a Potion was added 99 times and removed 96 times, 51 of them during a fight.
The in-battle item menu is the one that throws a Poke Ball.

## The Cut chain, from what run6 shows

| Goal | run6 | What stands in the way |
|---|---|---|
| Beat the S.S. Anne rival, get HM01 | 1/64 frozen, 2% training | Arrives at the rival weak; the rival pays nothing (above) |
| Poke Balls | none bought or found in 64 episodes | Nothing pays for them. Every mart before Vermilion sells them for 200 (`RESEARCH_CUT.md`); the agent has 800-9,000. Oak gives 5 only to a player who has beaten the optional Route 22 rival, owns fewer than 2 species and has no ball (checked: `scripts/OaksLab.asm`); evolving into Wartortle on Route 3 makes 2 |
| Catch a Pokemon that can learn Cut | never | Needs balls. Checked against the base stats' HM01 bit and Red's wild tables: Oddish is 35% of grass encounters on Routes 5 and 6 (levels 13-16) and at least 25% on Routes 24 and 25 (12-14); Paras is 5-15% in Mt. Moon; Beedrill can learn Cut (Weedle in Viridian Forest). Squirtle, Wartortle and Magikarp cannot. A Poke Ball catches a full-HP Oddish 86/256 = 33.6% of the time (Gen 1 formula, catch rate 255) |
| Teach Cut | never | START, ITEM, HM01, USE, pick the Pokemon. A newly caught Oddish has fewer than 4 moves, so no forget prompt. `MoveGain` would pay about 0.5 |
| Cut the tree before Vermilion Gym | never | Sets no flag (`RESEARCH_CUT.md` section 5); detectable with a hook on `ReplaceTreeTileBlock`. The tree's coordinates and the hook are not yet checked in the emulator |
