# What a run logs, and why

Each metric has to answer a question someone will actually ask of a run. `ppo/*`, `perf/*` come every update. Everything
else comes in a report every `report_every_steps` (5M by default, about 60 episodes at 81,000 steps each), and at the end of a run. Single-update
episode numbers were dropped because about four episodes finish per update, too few to read.

## Questions and where they're answered

| Question | Where |
|---|---|
| How far does it get, and how reliably? | `story/<milestone>_rate`, `story/<milestone>_median_step` |
| Which exact flags does it reach? | `tables/flags` (`reports/report_<step>.json`): flag id, address, bit, rate, median step |
| Where does it get stuck? | `tables/maps` (`step_share`, `end_share`, `stall_share`), `progress/dry_steps_mean`, `stuck/screens` |
| What did the game look like at that point? | `stuck/screens` in W&B; `stuck/<step>_<i>_stall_map<id>.png` and `.state` on the volume (the state reloads in PyBoy) |
| Who does it lose to? | `tables/fights`: per trainer, or per map for wild fights: fights and losses per episode, loss rate, mean party max level; `progress/lost_fights_mean`, `progress/blackouts_mean` |
| How strong is the party? | `progress/max_level_mean`, `progress/party_size_mean`, and `mean_party_max_level` per fight |
| Which rewards drive it, and when? | `reward/<term>` per episode; `tables/reward_phases`: per term, per 10,240 steps of episode |
| What does it press? | `actions/<button>`: share of presses |
| Has exploration collapsed? | `ppo/entropy_overworld`, `ppo/entropy_battle`; `entropy` per map in `tables/maps` |
| Did an update go wrong? | `ppo/approx_kl_max`, `ppo/approx_kl`, `ppo/clip_fraction`, `ppo/update_ratio`, `ppo/grad_norm` |
| Does the critic understand this part of the game? | `ppo/explained_variance`, `ppo/value_error_overworld`, `ppo/value_error_battle`; `value_error` per map in `tables/maps` |
| Is the network running out of room? | `net/feature_rank`, `net/dead_fraction`, `net/feature_norm`, `ppo/weight_norm` |
| Is the hardware keeping up? | `perf/steps_per_second`, `perf/update_seconds` |

A stall is `stall_steps` (10,240) with no new event or badge. New tiles don't end one, because an agent can wander and keep collecting them. Each episode keeps the snapshot from its last stall, plus one at its end.

## Removed from the first run's log

| Metric | Why |
|---|---|
| `episode/length_*` | Exactly `base_budget + budget_per_event * events` in the first run, so it repeated `events` |
| `episode/event_*`, `badge_*`, `explore_*`, `level_*` | Fixed multiples of the counts, or better shown as `reward/<term>` |
| `episode/level_sum_*` | Total level says little about strength. A level 20 starter and four level 5s sum the same. Replaced by `max_level` and `party_size` |
| `*_max` of milestones and step counts | A boolean's max only says it ever happened in a handful of episodes, and the slowest route isn't useful. `tables/flags` shows rare hits. Max stays for events, tiles, badges and max level |
| `episode/steps_to_*_mean` | Skewed by slow outliers. Replaced by the median |
| `episode/count`, `rollout/truncations` | Same number. Replaced by `progress/episodes` |
| `rollout/reward_per_step` | Misleading: flat from about 80M because reward and episode length both grow with events |
| `ppo/policy_loss` | With normalized advantages its size says little: at the 53-58M near-collapse it stayed under 0.02 while approx KL reached 3.3 |
| `perf/collect_seconds` | Follows from `steps_per_second` and `update_seconds` |

## Milestones

The first run tracked 13, with nothing between Misty and the S.S. Ticket, so it couldn't show whether the agent reached the rival. There are now 36, in the required order from `docs/evidence/events/EVENT_PROGRESSION.md`, including steps the game sets no flag for: `knows_cut` (the move in the party) and `has_silph_scope` (the item in the bag). The S.S. Anne rival has no flag either; find it as `RIVAL` in `tables/fights`.
