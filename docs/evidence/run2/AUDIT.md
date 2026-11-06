# Audit of run2 (400M steps, A100, 2026-11-05 to 11-06)

W&B run `objones25/pokemon-red-ppo/run2`. The sources are all 977 history rows, all 80 report windows (80 each of
`tables/fights`, `tables/flags`, `tables/maps` and `tables/reward_phases`), all 320 stall screens, `output.log`,
`wandb-metadata.json`, the config, and the first run (`main`) for comparison. Map and event names come from pret/pokered
`constants/map_constants.asm` and `constants/event_constants.asm`. The parsed IDs match the repo's own constants (Brock
0x077, Misty 0x0BF, Mt Moon 0x3B, S.S. Anne 0x5F).

Labels: **verified** means it is read straight from the logged data, the saved stall states' RAM, or pret. **Inferred**
means it is the best explanation of the data but not checked directly.

The stall `.state` files from the network volume were added after the first draft. They corrected two claims: finding 3
(the Magikarp is the active Pokémon in the stalled fights, not a fainted starter waiting to be replaced) and finding 5
(the level term was not at its maximum, only the starter's share of it).

## Findings

1. **The run ended stuck in Cerulean Gym.** From 380M on, 90–97% of episodes end there, 53–56% of all steps are spent
   there, and 84–93% of stalls happen there. Every late stall snapshot in the gym is at **(5,2), the tile beside Misty**,
   and most are in battle. (verified)
2. **Misty is the wall, and this run did much worse against her than the first run.** Misty's rate peaked at 27% (360M) and ended
   at 8%. The first run held 59–78% over its last 200M. (verified)
3. **Most Misty fights end in neither a win nor a loss.** From 375M on, per episode: about 1.3 Misty fights, 0.35 losses and
   0.1 wins, so about 0.8 fights per episode are unresolved. Every other trainer shows fought − lost ≈ 1.00, meaning beaten
   once, so this is specific to Misty. The saved states show how (see **Stall states, from RAM**): the starter has fainted
   and the **Magikarp is already the active Pokémon**, at full HP. The agent opens the party menu and picks the fainted
   starter (*"There's no will to fight!"*), opens its summary, or presses RUN, none of which uses a turn. It never picks
   FIGHT, where Splash is the only move and would end in a blackout. The battle is stalled, not stuck, and the episode
   runs out of budget in the middle of it. The same thing happens with the Swimmer's Horsea and Shellder (RUN: *"No!
   There's no running from a trainer battle!"*). In 14 of the 77 in-battle stall states, the Magikarp is out against a
   trainer with the starter fainted. Without the Magikarp, the starter fainting would end the battle in a blackout
   immediately. (verified in RAM for these states; that this accounts for all 0.8 unresolved fights is inferred)
4. **The bought Magikarp is a liability that the reward pays for.** Nothing is caught (Oak's free Poké Balls are taken in
   0–2% of episodes, and party size stays at 2), so the only second Pokémon is the Magikarp sold at the Mt Moon Pokémon Center (flag 0x3FF, 95–100% of episodes from 60M). Buying
   it pays an event (2.0) and 5 levels in party place 2 (2.5). In battle it can only use Splash. It appears in the RUN loops
   against the Swimmer's Horsea, in the TM-on-Magikarp menus, and in the "no will to fight" loop. (verified)
5. **The new level cap stopped paying exactly where Misty needs more.** `GYM_ACE_LEVEL_BY_BADGES[1] = 21`, which is
   Starmie's level. `reward/level` reached about 10.0 per episode by 75M and stayed there. That is 20 levels at weight 0.5:
   the starter's place from 6 to 21 (15) plus the Magikarp's arrival at 5 (5). **It is not the term's maximum.** After one
   badge each of the six places pays up to 21, so levelling the Magikarp to 21 would have paid 8.0 more (and evolved it into
   Gyarados at 20, which learns Bite at 20, per pret), and four empty places another 42.0 through catches. What stopped
   paying at 75M was the **starter** getting stronger. Levelling anything else still paid, and the agent never did it.
   Party top level averaged 23.9 at the end of training. The first run's comparable training-time number isn't logged,
   but its party level total averaged about 33 over its last 200M (`episode/level_sum_mean`, level reward about 13.4), which
   with a level-5 Magikarp puts its starter near 28. The 26.2 elsewhere in this report is the first run's final 16-episode
   report, and 26.3 its frozen A100 evaluation; neither is a like-for-like comparison with 23.9. Under the old party-total
   cap of 39, the starter could earn reward up to 34, and the first run stopped short of that, at about 28. (Numbers
   verified; that the lower cap caused the Misty drop is inferred, with one seed per run)
6. **The starter almost always declines Bite, and often cancels its evolution.** Squirtle learns Bite at 22 and Wartortle
   at 24 (pret). Across the 320 stall states, only 5 of 148 Squirtles at level 22 or above and 1 of 108 Wartortles at 24
   or above know Bite. The usual moveset is Tackle, Tail Whip, Bubble and Water Gun, sometimes with Bide (TM34) in place of
   one. Against Misty's Water Pokémon, Bubble and Water Gun are resisted; in Gen 1, Bite is a 60-power Normal move,
   neutral against them, and Tackle has 35 power. Squirtle evolves at 16, yet from 75M on, **about 55% of stall states
   still have a Squirtle at level 16–30**. Evolution can only be refused by pressing B during it, so the agent cancels it,
   and keeps cancelling at each level-up. (verified in RAM; stall states are a sample of stalled episodes, not of all
   episodes)
7. **The policy found a better branch and then lost it.** The Nugget Bridge and Route 25 chain (the Cerulean rival, 6
   bridge trainers, the Nugget, the Route 24 Rocket and 9 Route 25 trainers, about 15 flags) rose from 0% at 300M to 30–60%
   at 340–345M. Events per episode peaked at 38.6, the run's best. **By 355M the whole chain was at 0%**, and it never
   returned. Misty became the policy's goal instead. Episode return fell from about 125 (345M) to about 109 (400M). In the
   same window the fossil choice flipped, from Dome (97% until 345M) to Helix (97% from 375M). That is a large policy
   change, with no PPO metric out of line. (verified)
8. **The loss pattern never improved after 50M.** Route 3's Youngster #1 (Rattata and Ekans, about level 11) was lost in
   45–50% of fights from 50M to 400M, at a party top level of about 14. Mt Moon's Lass #5 (Grass Oddish and Bellsprout at 11,
   super-effective on Water) and Rocket #4 (Raticate at 16): 40–57%. Each loss is
   a blackout (7.8 per episode at the end), and the agent then retraces the route. That fits 22 wild fights per episode on
   Route 1, which is short. (verified; the retracing is inferred)
9. **PPO itself looks healthy.** Explained variance was 0.99 from 40M, clip fraction 0.08–0.14, and gradient norm about
   0.18. Two single-update KL spikes occurred (`approx_kl_max` 17.6 at 95.0M and 8.0 at 130.3M), each back to normal on the
   next update with no later damage. Things to watch: weight norm grew without stopping (47 → 121) despite weight decay,
   feature rank fell from about 360 to 272 over the last 50M, and entropy has no bonus (`ent_coef=0`). (verified)

## Run facts

| | |
|---|---|
| Commit | `9dcdea6` (current HEAD: first-episode stagger over the whole budget, per-place level cap at the gym ace, new telemetry) |
| Hardware | A100 80GB PCIe, AMD EPYC 7763 (252 logical CPUs visible), NUMA 0, workers pinned |
| Layout | 25 workers × 8 envs = 200 envs, T = 2048: 409,600 steps per update, 977 updates |
| PPO | lr 3e-4, 3 epochs, minibatch 8192, clip 0.2, γ 0.997, λ 0.95, vf 0.5, **ent 0**, wd 0.01, grad clip 0.5 |
| Env | fast_mode, base budget 10,240 + 2,048 per event, stall 10,240, weights: tile 0.005, event 2, badge 5, level 0.5 |
| Duration | 20.3 h (23:30 to 19:49 UTC), no restarts (no runtime gap over 200 s), pod self-terminated |
| Throughput | 5,472 steps/s on average. 6,040 at 20–40M, about 5,650 mid-run, **about 4,920 over the last 40M** (minimum 4,367 at 374.8M). Update time steady at 4.5 s |
| Log | Only 4 warnings, all transient OpenTelemetry "Service Unavailable" retries. No errors, tracebacks or NaNs |

The final episodes ran about 77,800 steps (10,240 + 2,048 × 33). The mean `dry_steps` at the end was 44,445: **57% of each
episode came after its last event or badge**.

## How progress changed over time

One row per selected report window. Rates are the share of episodes that finished in that window.

| Steps | Episodes | Events (max) | Tiles | Brock | Mt Moon | Cerulean | Rival | R24 Rocket | Misty | Top level | Party | Blackouts | Fights | Dry steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5M | 531 | 0.1 (4) | 54 | 0 | 0 | 0 | 0 | 0 | 0 | 6.0 | 1.00 | 0.1 | 1 | 8,083 |
| 25M | 285 | 4.1 (6) | 1,069 | 0 | 0 | 0 | 0 | 0 | 0 | 6.5 | 1.00 | 1.3 | 15 | 16,754 |
| 40M | 153 | 8.7 (17) | 2,811 | 0.49 | 0 | 0 | 0 | 0 | 0 | 15.6 | 1.00 | 2.8 | 71 | 14,972 |
| 50M | 117 | 16.0 (20) | 3,250 | 0.97 | 0.34 | 0 | 0 | 0 | 0 | 19.2 | 1.00 | 3.5 | 81 | 20,547 |
| 75M | 93 | 21.9 (26) | 3,838 | 1 | 1 | 0 | 0 | 0 | 0 | 23.0 | 1.94 | 5.9 | 130 | 15,073 |
| 100M | 73 | 24.5 (27) | 4,267 | 1 | 1 | 0 | 0 | 0 | 0 | 23.8 | 1.99 | 6.6 | 203 | 23,034 |
| 125M | 92 | 27.1 (32) | 4,348 | 0.95 | 0.95 | 0.07 | 0 | 0 | 0 | 23.6 | 1.93 | 7.0 | 254 | 27,178 |
| 150M | 82 | 29.2 (33) | 5,006 | 1 | 1 | 0.38 | 0 | 0 | 0 | 25.1 | 1.99 | 7.7 | 318 | 35,265 |
| 180M | 74 | 33.3 (34) | 5,962 | 1 | 1 | 0.95 | 0 | 0 | 0 | 26.1 | 1.96 | 7.6 | 294 | 31,502 |
| 200M | 62 | 31.9 (34) | 5,624 | 0.98 | 0.98 | 0.77 | 0 | 0 | 0 | 24.8 | 1.95 | 8.5 | 262 | 32,950 |
| 250M | 53 | 31.5 (34) | 5,253 | 1 | 0.98 | 0.89 | 0 | 0 | 0 | 23.7 | 1.91 | 7.6 | 221 | 34,427 |
| 300M | 68 | 32.6 (34) | 6,102 | 1 | 1 | 0.97 | 0 | 0 | 0 | 24.2 | 1.97 | 8.5 | 264 | 38,998 |
| 320M | 70 | 33.5 (46) | 6,074 | 1 | 1 | 0.96 | 0.14 | 0.04 | 0 | 24.1 | 1.97 | 9.1 | 268 | 35,708 |
| 335M | 50 | 36.4 (51) | 6,484 | 1 | 1 | 1 | 0.36 | 0.20 | 0.04 | 25.4 | 2.00 | 9.2 | 282 | 37,628 |
| **340M** | 73 | **38.4** (52) | 6,549 | 1 | 1 | 0.99 | **0.60** | **0.33** | 0.01 | 26.1 | 1.97 | 9.5 | 351 | 30,329 |
| 345M | 64 | 38.6 (52) | 6,612 | 1 | 1 | 1 | 0.48 | 0.31 | 0.02 | 26.2 | 2.00 | 9.5 | 373 | 37,679 |
| 350M | 58 | 35.1 (52) | 6,470 | 1 | 1 | 1 | 0.16 | 0.09 | 0.09 | 25.1 | 2.00 | 9.8 | 296 | 34,467 |
| 355M | 70 | 33.3 (36) | 6,449 | 1 | 1 | 0.97 | **0** | **0** | 0.14 | 24.5 | 2.00 | 10.0 | 325 | 37,125 |
| **360M** | 62 | 33.4 (36) | 6,274 | 1 | 1 | 0.95 | 0 | 0 | **0.27** | 24.4 | 1.98 | 9.4 | 343 | 42,527 |
| 380M | 57 | 33.3 (36) | 5,481 | 1 | 1 | 0.98 | 0 | 0 | 0.11 | 24.0 | 1.98 | 8.0 | 218 | 37,201 |
| 400M | 59 | 33.2 (36) | 5,389 | 1 | 1 | 0.97 | 0 | 0 | 0.08 | 23.9 | 2.00 | 7.8 | 190 | 44,445 |

All 80 windows are in `report_scalars.csv`. Nothing after the Cerulean rival was ever reached: Bill, S.S. Ticket,
trashed house and every later milestone stayed at 0%. `badges_max` was 2 throughout.

### Phases

| Phase | What happened |
|---|---|
| 0–20M | Leaving Oak's lab. At 5M, 91% of steps and 97% of stalls are in Oak's lab. Pallet flags (town map, Blue's house, potion) reach 100% |
| 20–50M | Viridian Forest trainers, Pewter Gym and Brock (0 → 97%). The Route 22 rival is lost 83% of the time, the run's top loss early on. The first Route 22 rival flag peaks at 5%, then becomes unreachable once Brock falls |
| 50–75M | Route 3 trainers, then Mt Moon. **The Magikarp purchase appears (0% at 50M → 99% at 75M)** and party size goes 1 → 2. 17% of steps are in the Mt Moon Pokémon Center at 75M. The level reward saturates |
| 75–125M | Working through Mt Moon's floors: B2F trainers, the Super Nerd and the fossil, 0% → 39% by 105M, 97% by 300M |
| 125–180M | Cerulean City 7% → 95%. Cerulean Gym's two trainers are beaten, but Misty is fought only about 1% of the time. Rival fights start (up to 0.38 per episode) and are lost 85–100% of the time |
| 180–250M | **Plateau**: about 32 events, Cerulean City 77–97%. Cerulean Gym holds 15–32% of steps without fighting Misty (0–4%). Stalls are at the gym's trainers and on Route 4 |
| 250–300M | Misty fights begin (0.4 → 0.9 per episode, lost 90%). RUN loops against the Swimmer's Horsea with the Magikarp out (screens) |
| 300–350M | **Rival branch**: rival fights 0.2 → 2.4 per episode, beat rate up to 60%. Nugget Bridge, Route 24 Rocket and Route 25 trainers come up together. Misty fights drop to 0.04–0.15 per episode |
| 350–400M | **Collapse of the rival branch and lock-in at Misty**: rival 0% from 355M, Misty fights 1.1 → 2.4 per episode, then the gym absorbs 90%+ of episode ends. Helix replaces Dome |

## Event graphs

Events per episode by 20M bin: 4.0 · 8.7 · 18.8 · 22.3 · 24.5 · 25.8 · 29.0 · 29.9 · 33.3 · 31.9 · 32.1 · 32.0 · 32.8 ·
32.6 · 32.6 · 33.5 · **38.4** · 33.4 · 33.3 · 33.2.
There are three regimes: fast growth to about 22 by 75M; slow growth to about 33 by 175M; then flat at 32–33 for 225M,
broken only by the 335–350M rival bump. The maximum per episode jumped to 43–52 whenever the rival branch was alive (145M,
170M, 225M, 310–350M), so the policy had the higher-paying route in reach several times before 340M.

Flag rates over time, by group. The first column is the 5M window in which the flag first appears; the rest are rates.

| Flag group | First seen | 50M | 100M | 150M | 200M | 300M | 340M | 355M | 375M | 400M | Median step at 400M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Pallet (town map, Blue's house, Daisy, potion) | 5M | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 29–91 |
| Viridian Forest trainers 0–2 | 25M | ≥0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 671–3,512 |
| Museum ticket (pays nothing) | 25M | 0.69 | 0.97 | 0.95 | 0.87 | 0.94 | 1.00 | 0.99 | 0.94 | 0.95 | 18,896 |
| Pewter Gym trainer, TM34, Brock | 30–35M | 0.97 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1,856 |
| Route 3 trainers 0–7 | 40–45M | 0.50–0.91 | ≥0.98 | ≥0.98 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 4,397–11,022 |
| Bought Magikarp | 55M | 0.00 | 0.99 | 0.99 | 0.95 | 0.97 | 0.97 | 1.00 | 0.95 | 1.00 | 8,508 |
| Mt Moon 1F trainers | 45–75M | 0.00–0.07 | 0.11–0.99 | 0.52–1.00 | 0.92–0.98 | 0.96–1.00 | 1.00 | ≥0.97 | 1.00 | ≥0.93 | 8,888–16,941 |
| Mt Moon B2F trainers and Super Nerd | 50–105M | 0.00–0.01 | 0.00–0.81 | 0.39–0.89 | 0.77–0.98 | 0.60–0.99 | 0.73–1.00 | 0.69–0.97 | 0.66–0.98 | 0.71–0.97 | 12,308–24,419 |
| **Dome Fossil** | 120M | 0 | 0 | 0.39 | 0.77 | 0.97 | 0.99 | 0.97 | **0.25** | **0.00** | – |
| **Helix Fossil** | 245M | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0.74** | **0.97** | 24,430 |
| Cerulean Gym trainers 0, 1 | 145M | 0 | 0 | 0.00–0.05 | 0.60–0.74 | 0.87–0.94 | 0.93–0.96 | 0.96–0.97 | 0.95–0.98 | 0.95–0.97 | 28,418–31,301 |
| Misty, TM11 | 255M | 0 | 0 | 0 | 0 | 0 | 0.01 | 0.14 | 0.14 | 0.08 | 41,102 |
| **Cerulean rival** | 145M | 0 | 0 | 0 | 0 | 0 | **0.60** | **0** | 0 | 0 | – |
| **Nugget Bridge (Route 24 trainers 0–5, Nugget, Rocket)** | 145M | 0 | 0 | 0 | 0 | 0 | 0.11–0.60 | 0 | 0 | 0 | – |
| **Route 25 trainers 0–8** | 145M | 0 | 0 | 0 | 0 | 0 | 0.05–0.30 | 0 | 0 | 0 | – |
| Oak's Poké Balls, Pallet after balls | 40M | 0.01 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | – |
| First Route 22 rival battle | 35M | 0.04 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | – |

60 distinct flags were reached all run. Things to note:

- **Mt Moon B2F trainers 2 and 3 are never reliable** (66–89%). The other required Mt Moon flags are at 97% or more.
- **The museum ticket** (`EVENT_BOUGHT_MUSEUM_TICKET`) is bought in about 95% of episodes from 100M, at a median step of
  about 19,000, despite paying nothing. It probably comes from walking into the museum and pressing A. It costs a detour
  through MUSEUM_1F (7% of steps at 150M).
- **The fossil flip** coincides with the rival collapse. Either fossil pays one flag, so the reward is indifferent between them:
  the flip is a sign of a wider policy change, not something the agent was optimizing.

### Reward over time

| Steps | event | explore | level | badge | Total |
|---|---|---|---|---|---|
| 50M | 32.1 | 16.3 | 6.6 | 4.8 | 59.7 |
| 100M | 48.9 | 21.3 | 10.0 | 5.0 | 85.2 |
| 200M | 63.8 | 28.1 | 9.8 | 4.9 | 106.6 |
| 300M | 65.2 | 30.5 | 9.9 | 5.0 | 110.6 |
| **345M** | **77.2** | **33.1** | 10.1 | 5.1 | **125.4** |
| 400M | 66.3 | 27.0 | 10.1 | 5.4 | 108.8 |

Rewards by episode phase (`tables/reward_phases`, 400M): 80% of event reward comes in the first 20,480 steps (39.5 + 14.6
of 66.3), and almost nothing after 40,960. Level reward is 8.3 in the first 10,240 steps, 1.5 in the next 10,240, and about
0 after that. Exploration also falls to about 0 after 50k. The second half of every episode pays almost nothing.

## Stuck states

### Where episodes end and stall, through the run

| Window | Top end maps (share of episodes) | Top stall maps (share of stalls) |
|---|---|---|
| 5M | Oak's Lab 0.88 | Oak's Lab 0.97 |
| 50M | Route 22 0.42, Pewter 0.37 | Route 22 0.36, Pewter 0.19, Pewter Gym 0.09 |
| 75M | Mt Moon 1F 0.30, Route 4 0.27, Route 3 0.14 | Mt Moon 1F 0.24, Route 4 0.18, Route 3 0.16 |
| 105M | Mt Moon B2F 0.32, 1F 0.25, Viridian City 0.23 | Mt Moon B2F 0.37, 1F 0.30 |
| 155M | Cerulean Gym 0.24, Viridian Forest 0.12 | Viridian Forest 0.26, Cerulean Gym 0.17, Mt Moon B1F 0.16 |
| 180M | **Cerulean Gym 0.61**, Route 4 0.18 | Cerulean Gym 0.51, Cerulean City 0.18 |
| 250M | Cerulean Gym 0.55, Route 4 0.13 | Cerulean Gym 0.58, Route 4 0.13 |
| 300M | Cerulean Gym 0.32, Cerulean City 0.15 | Cerulean Gym 0.38 |
| 340M | Mt Moon B2F 0.23, Viridian City 0.16, Mt Moon B1F 0.11, Route 24 0.08 | **Viridian City 0.36**, Mt Moon B2F 0.15 |
| 360M | Cerulean Gym 0.34, Viridian City 0.21 | Cerulean Gym 0.31, Viridian City 0.15 |
| 375M | Cerulean Gym 0.74 | Cerulean Gym 0.62 |
| 380–400M | **Cerulean Gym 0.90–0.97** | **Cerulean Gym 0.84–0.93** |

Cerulean Gym's share of all steps went 9% (155M) → 23% (180M) → 32% (250M) → 6% (340M) → **54% (400M)**.

### Cerulean Gym: the main stuck state

`cerulean_gym_stalls.png` shows every gym stall snapshot of the run, in order. The positions show how the gym problem changed:

| Steps | Position | What the screens show |
|---|---|---|
| 155–235M | (3,1), (0,1), (5,7), (6,1), (2,6), (7,5) | Walking around the gym, or in menus (Start menu, a TOSS/USE prompt on TM34, the party menu). Misty is barely fought. The 200M screen says "RED blacked out!" against a Goldeen |
| 240–280M | (5,7) mostly, in battle | **The Magikarp is out against the Swimmer's Horsea L16. The agent spams RUN: "No! There's no running from a trainer battle!"** Five of six snapshots here |
| 290–300M | (4,3) | "MISTY used X DEFEND" against the Wartortle |
| 355–365M, 400M | (5,2) | Two badges in RAM. "TM11 teaches BUBBLEBEAM!": **stalled after beating Misty**, still talking to her (she repeats the TM11 line) |
| 370–395M | (5,2), in battle | Starmie L21 against the starter at 4–17 HP (a Squirtle in RAM at 370M and 390M, unevolved). At 375M, 380M and 385M the starter is fainted and **the Magikarp is the active Pokémon at full HP**: the screens are the fainted starter's summary and "There's no will to fight!" |

The report keeps only one stall snapshot per map per window (`_pick_snapshots`). So the 4 screens per window
**understate** the concentration: at 395M, 93% of stalls were in the gym, but only one of the 4 screens is.

### Other recurring stalls

| Position | Times seen | When | Notes |
|---|---|---|---|
| Viridian City (6–8, 4–6) | 18 | 330–400M | Top stall map at 335–345M (33–42% of stalls). The cause is not visible on the screens (one shows the Start menu) |
| Route 4 (63,10), (85–88,10–11) | 11 | 180–390M | The Route 4 ledge east of Mt Moon. Up to 32% of stalls at 205M |
| Route 22 (4–7,15), (16,4), (29,4) | 9 | 50M, 330–375M | West of Viridian. Early on, the rival fight there |
| Mt Moon B2F (22–23,31–32), (3,4) | 8 | 300–400M | Often in battle; (3,4) is next to a Rocket |
| Mt Moon 1F, B1F | many | 75–150M | Moving between floors |

Two snapshots (160M and 170M) are **blank white screens**, taken during a screen transition. That is a telemetry artifact, not a game state.

## Trainers lost to

Losses per episode, weighting each window by its episode count. A loss is a blackout credited to the last fight. For every
trainer except Misty, fought − lost ≈ 1.00 per episode: every episode beats them eventually, and each loss means a rematch.

| Phase | Losses/ep | Fights/ep | Top losses (lost/ep, loss rate, party top level at fight start) |
|---|---|---|---|
| 0–50M | 0.93 | 19.6 | **Route 22 rival** 0.28 (83%, L6.7) · Pewter Gym Jr. Trainer 0.13 (64%) · Viridian Forest wild 0.10 · Bug Catchers 0.07–0.09 |
| 50–125M | 5.83 | 186 | **Route 3 Youngster #1** 0.80 (45%, L14.8) · Mt Moon Lass #5 0.49 (85%, L21.1) · Mt Moon B2F wild 0.43 · Pewter Jr. Trainer 0.38 · Route 3 Bug Catcher #4 0.37 |
| 125–250M | 8.03 | 295 | **Mt Moon Lass #5** 1.05 (57%, L21.6) · **Mt Moon Rocket #4** 0.98 (55%) · Route 3 Youngster #1 0.95 (49%) · Route 3 Lass #1 0.78 (44%) |
| 250–330M | 8.12 | 250 | Route 3 Youngster #1 0.99 (50%) · Lass #5 0.86 · Route 3 Lass #1 0.70 · Rocket #4 0.65 · Cerulean Swimmer 0.46 · **Misty 0.43 (89%, L23.9)** |
| 330–350M | 9.30 | 319 | **Cerulean rival 1.64 (77%, L23.7)** · Route 3 Youngster #1 1.02 · Route 3 Lass #1 0.66 · Rocket #4 0.51 |
| 350–400M | 8.63 | 259 | **Misty 1.03 (63%, L23.8)** · Route 3 Youngster #1 0.92 (48%) · Lass #5 0.86 (46%) · Route 3 Lass #1 0.72 · Swimmer 0.50 · Rocket #4 0.49 |

At the end (400M window), per episode: Lass #5 lost 1.32 times, Route 3 Bug Catcher #4 0.76, Youngster #1 0.71, Lass #1
0.68, Rocket #1 0.49, Swimmer 0.47, Jr. Trainer F 0.42, Rocket #4 0.42.

What the loss data says:

- **Easy early trainers stay costly forever.** Route 3 Youngster #1 is lost in half of all fights from 50M to 400M, at a
  party top level of about 14. That is a young Squirtle (no Water Gun until 15) that probably has not healed since its last fights.
  The losses did not decline in 350M steps. Each one is a blackout and a walk back.
- **Mt Moon Lass #5 is lost at L20–21** in 46–85% of fights, though her Oddish and Bellsprout are level 11 (pret). Both
  are Grass types and hit the Wartortle super-effectively. The 400M stall shows "ODDISH L11 … It's super effective!" with
  the Wartortle at 0/52. Most of her losses are type matchup and probably entering the fight already damaged, since there
  is no heal reward. The rival's team has a Bulbasaur in both fights, Route 22 (Pidgey 9, Bulbasaur 8) and Cerulean
  (Pidgeotto 18, Abra 15, Rattata 15, Bulbasaur 17): Grass again.
- Other teams (pret): Route 3 Youngster #1 is Rattata and Ekans at 11, Route 3 Lass #1 is two Pidgey at 9, and Mt Moon
  Rocket #4 is a Raticate at 16.
- **Misty: 1.03 losses plus about 0.8 unresolved fights per episode.**

| Steps | Misty fought/ep | Lost | Not lost | Misty rate |
|---|---|---|---|---|
| 255M | 0.40 | 0.37 | 0.03 | 0.03 |
| 275M | 0.90 | 0.88 | 0.01 | 0.00 |
| 290M | 0.72 | 0.46 | 0.26 | 0.00 |
| 355M | 2.27 | 2.10 | 0.17 | 0.14 |
| 360M | 2.44 | 2.10 | 0.34 | 0.27 |
| 375M | 1.49 | 0.75 | 0.74 | 0.14 |
| 380M | 1.35 | 0.42 | 0.93 | 0.11 |
| 395M | 1.37 | 0.42 | 0.95 | 0.15 |
| 400M | 1.15 | 0.25 | 0.90 | 0.08 |

  From 375M, losses fell while unresolved fights rose to about 0.9, and the beat rate did not improve: the policy swapped
  losing to Misty for stalling in her fight until the budget ran out. A stalled fight costs nothing, while a blackout only
  sends the agent back without penalty either, so neither is punished. The fight table can't tell a truncated fight from a
  won one; see the telemetry notes.
- **The Cerulean rival** was lost 77–100% of the time before 330M. At the 340M peak the agent beat him in 60% of episodes
  after about 1.6 losses per episode.

## Party and levels

- **Party size** was 1 until 60M, then 2 (the Magikarp) for the rest of the run. Max level climbed to about 24 by 65M and
  then **never rose**: it stayed at 22.7–26.8 for 335M steps.
- **Level reward** reached about 10.0 per episode by 75M: starter 6 → 21 is 15 levels and the Magikarp's arrival is 5
  levels, so 20 × 0.5. That is the starter's cap with one badge, not the term's: the Magikarp's place could still pay 8.0
  (to 21) and the empty places 42.0. The starter's levels 22–24 pay nothing; they come from fights it fights anyway.
- The Magikarp leads the party in some snapshots (levels `[5, 26]`, `[5, 23]`, `[5, 25]`), so the agent sometimes switches
  the order or swaps out in battle. It then sends the Magikarp against an Onix (Route 25) or a Horsea.
- **No other catch.** Party size stays at about 2 with the Magikarp flag at 95–100%, so nothing else joins. Oak's free
  Poké Balls are taken in 0–2% of episodes, but balls are also sold at the Viridian Mart, so the agent could catch; it
  doesn't. Whether it ever buys balls isn't logged.

## Training data (PPO health)

Means per 20M bin:

| Steps | Entropy | Overworld | Battle | KL | KL max | Clip | Expl. var | Value loss | Grad norm | Weight norm | Steps/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0–20M | 1.720 | 1.698 | 1.724 | 0.008 | 0.012 | 0.092 | 0.690 | 0.0066 | 0.200 | 52 | 5,276 |
| 40–60M | 0.963 | 0.970 | 0.859 | 0.014 | 0.029 | 0.136 | 0.982 | 0.0096 | 0.238 | 68 | 5,906 |
| 80–100M | 0.584 | 0.562 | 0.628 | 0.017 | 0.403 | 0.116 | 0.990 | 0.0079 | 0.224 | 79 | 5,621 |
| 100–120M | **0.431** | 0.415 | 0.456 | 0.013 | 0.029 | 0.091 | 0.991 | 0.0069 | 0.210 | 84 | 5,437 |
| 180–200M | 0.480 | 0.373 | 0.748 | 0.011 | 0.023 | 0.093 | 0.992 | 0.0065 | 0.191 | 98 | 5,624 |
| 280–300M | 0.449 | 0.426 | 0.520 | 0.011 | 0.025 | 0.094 | 0.993 | 0.0058 | 0.186 | 111 | 5,708 |
| 320–340M | 0.609 | 0.564 | 0.709 | 0.012 | 0.043 | 0.109 | 0.991 | 0.0061 | 0.189 | 115 | 5,446 |
| 360–380M | 0.618 | 0.494 | 0.760 | 0.015 | 0.090 | 0.093 | 0.991 | 0.0069 | 0.179 | 119 | 4,961 |
| 380–400M | 0.564 | 0.476 | 0.626 | 0.013 | 0.050 | 0.082 | 0.992 | 0.0063 | 0.178 | 120 | 4,924 |

- **Entropy** fell fastest in the first 100M (1.72 → 0.43), with no entropy bonus. After that it stayed at 0.43–0.62 and
  rose slightly in the rival and Misty period. The lowest battle entropy, 0.31, is at 399.4M. In the final window,
  Cerulean Gym entropy is 0.31 and its value error is 0.005, the lowest of any busy map. **The critic is confident in the gym
  loop**: the loop is settled behavior, not confusion.
- **KL spikes**: 42 updates had `approx_kl_max` > 0.1. The large ones: 17.6 (95.0M, mean KL 0.214), 8.0 (130.3M, mean
  0.230), 1.27 (69.6M), and mean KL 0.295 at 226.1M (max 0.58). Every one returned to about 0.01 on the next update, with no
  change in entropy, explained variance or events. Clip fraction never exceeded 0.172. Each looks like one minibatch of rare
  states. They are harmless here, but **there is no KL early stop**, so a worse one would go through.
- **Explained variance** was 0.98–0.99 from 40M to the end. Overworld value error rose at the end (0.023 → 0.039), led by
  Route 3 (0.118) and Mt Moon (0.076–0.078): the early maps, which the agent now crosses again after blackouts.
- **Network capacity**: weight norm grew without stopping, 47 → 121, despite weight decay 0.01. It was still growing at
  400M. Feature rank was about 300–370 for most of the run and then fell to **272 at 400M** (306 at 375M, 292 at 380M). The
  dead fraction fell from 0.145 to 0.07. There's no collapse, but the falling rank lines up with the gym lock-in.
- **Actions**: UP rose from about 0.22 (100–350M) to **0.40** at 400M, while DOWN fell to 0.08 and RIGHT to 0.12. That fits
  the agent spending half its time facing Misty at (5,2) and moving the cursor in battle menus. START stayed at 1–3% after 25M.
- **Throughput** fell about 12% in the last 40M (5,650 → 4,920 steps/s) while update time stayed at 4.5 s, so the slowdown
  is in collection. One possible cause is the larger share of battle and menu steps. Host contention can't be ruled out:
  the run logged no host-level metrics.

## Comparison with the first run (`main`)

The first run used the same PPO and env settings except: party-total level cap, no new telemetry, an earlier first-episode
stagger (base budget only), and 19 workers on NUMA 1.

| 20M bin ending | main: events, Misty | run2: events, Misty |
|---|---|---|
| 60M | 8.5, 0 (Brock 48%) | 18.8, 0 (Brock 100%) |
| 100M | 24.7, 0 | 24.5, 0 |
| 160M | 31.9, 0.23 | 29.9, 0 |
| 200M | 34.0, **0.61** | 31.9, 0 |
| 300M | 34.3, 0.63 | 32.6, 0 |
| 360M | 33.9, 0.50 | 33.4, 0.27 |
| 400M | 34.8, **0.78** | 33.2, **0.08** |

The first run's numbers are means over updates, which are noisier than run2's reports, and its event counts are close to
run2's. run2 beat Brock about 20M steps sooner. It is far worse against Misty and gained the rival branch only briefly. Both
runs had the Magikarp (the first run's evaluation shows party size 1.94). The main difference that bears on Misty is the
level term. Under the old party-total cap of 39, the starter could earn reward up to level 34. During training the first
run's party level total averaged about 33 over its last 200M (starter near 28, with a level-5 Magikarp), against a top
level of about 24 here. Its frozen A100 evaluation had a top level of 26.3. This is one seed per run, so the link is a
hypothesis, not a result.

## Telemetry and tooling notes

- **The fights table can't tell a truncated fight from a win.** Misty's "fought minus lost" counts truncations as
  non-losses. Recording an `unfinished` flag per fight (episode ended in battle) would separate them.
- **One stall snapshot per map per window** hides concentration. When one map holds 90% of stalls, the screens show it
  once. Consider weighting snapshots by stall share, or logging each window's end snapshots too.
- **Blank snapshots** (160M, 170M): a stall that lands on a transition frame saves a white screen.
- **The museum ticket** is bought 95% of the time. It pays nothing, but the detour costs time.
- **W&B**: `run.scan_history()` hung for over 12 minutes on this host. `run.history(samples=100000, pandas=False)`
  returned all 977 rows in seconds.
- **The stall states were copied from the volume** (`/workspace/pokered/runs/run2/{stuck,reports,checkpoints}`, 130 MB)
  through a 10-minute pod, then read in PyBoy without advancing a frame. See **Stall states, from RAM**.

## Stall states, from RAM

All 320 stall `.state` files were loaded with `pyboy.load_state` and read with `ram_map.read_state`, with no frames run,
so each one is exactly the saved moment. Names come from pret `pokemon_constants.asm`, `move_constants.asm` and
`item_constants.asm`.

| Question | Answer |
|---|---|
| Who is in the party? | Squirtle or Wartortle plus a Magikarp in 248 states, the starter alone in 72. Never a third Pokémon, never a Gyarados |
| Does the Magikarp level? | Level 5 in 241 states, 6 in 7. It never gets past 6 |
| Is the starter evolved? | From 75M, about 55% of states have a Squirtle at level 16 or above (up to 30): evolution cancelled |
| Does it learn Bite? | 5 of 148 Squirtles at 22+, 1 of 108 Wartortles at 24+ |
| Magikarp out with the starter fainted, in a trainer battle | 14 of 77 in-battle states: 8 against the Swimmer's Horsea or Shellder (RUN spam), 3 against Misty's Starmie (party menu and summary, "no will to fight"), 1 Jr. Trainer's Goldeen (blacked out), 1 Route 25 Hiker's Onix, 1 Cerulean rival's Pidgeotto |
| Can it catch? | Poké Balls are in the bag in 18 of 320 states, so sometimes yes; it never does |
| Unused free levels | A Rare Candy (Mt Moon) is in the bag in 57 states. Used on the Magikarp it would pay a level |
| Late gym stalls after winning | 4 of the 10 gym stalls from 355M have two badges: the agent beat Misty and kept talking to her |
| Money | Median $540–$1,066 from 50M; it halves at every blackout |

## What to test next

In order of evidence:

1. **Done: the stall states** are read in **Stall states, from RAM**. They changed finding 3 (the Magikarp is out,
   not stuck in a forced switch) and widened finding 6 (Bite declined and evolution cancelled, across the run).
2. **The Misty level cap**: raise the per-place cap after one badge (for example to 26–28), or pay the starter's place
   beyond the ace, and compare Misty rate against run2 using `--init-from` on run2's checkpoint, with a frozen control as in
   `docs/evidence/first_run/update_test/`.
3. **Remove the Magikarp incentive**: a catch that can't fight still pays 2.5 in levels plus a flag. Options include paying
   party level only for Pokémon that have won a fight, or not paying for the Magikarp purchase flag. The trade-off: the
   Magikarp sometimes absorbs the last hit and changes how a blackout happens.
4. **A latched heal** (already the planned first reward experiment in CLAUDE.md). The 45–57% loss rates against low-level
   trainers late in episodes are the cost of never healing.
5. **The rival branch**: the policy reached 60% and lost it in 15M steps with no PPO metric out of line. Several seeds
   would show whether that is forgetting or noise between competing branches.

## Files

- `report_scalars.csv`: every scalar from all 80 reports.
- `cerulean_gym_stalls.png`: every Cerulean Gym stall snapshot of the run.
- `stalls_375M_400M.png`, `stalls_330M_350M.png`: all stall snapshots in those windows.
- `stall_states.json`: RAM of all 320 stall states (party, moves, PP, bag, money, battle). Made by
  `read_stall_states.py`, which expects the states in `vol/stuck/` and pret constants in `pret/` beside it (written for the
  audit scratchpad, not the repo's scripts).
