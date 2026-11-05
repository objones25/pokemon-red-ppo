# Event progression in Pokemon Red, from the start state to the Hall of Fame

Which progress steps have an event flag, which don't, and what `EventFlagGain` pays along the way. Research only: nothing in
`src/` or `tests/` changed, and no training or pod was touched.

**No data from any run is used here.** Nothing below says which flags the live run has hit.

## Short answers

- **The user's order for Misty to Vermilion is right, with three refinements:** rival, Nugget Bridge, Bill,
  S.S. Ticket, back to Cerulean, Cerulean Rocket, then Route 5, the Underground Path and Route 6.
  - Misty isn't a prerequisite of any of this. Her badge is needed later, for Cut.
  - The thing that blocks the way is a guard at (27,12), standing in front of the trashed house door. Taking the ticket moves him.
    The Rocket stands in the yard behind the house, outdoors.
  - Route 5 doesn't reach Saffron without a drink. Vermilion is reached through the Underground Path.
- **This stretch isn't a reward desert overall.** Its shortest route has 21 forced flags. The dry parts are three walks of
  115 to about 150 moves.
- **The larger stall risk is just after it: Squirtle can't learn Cut.** Catching a Cut-capable Pokemon, teaching Cut and using it
  have no flag, and Vermilion Gym, Route 9 and everything after need Cut.
- **Required steps with no flag include** the S.S. Anne rival, the Silph Scope, and the drink for the Saffron guards. There is no Tea
  in Red.
- **The ROM agrees with pret on every set and reset site.** See "ROM check".

## How things were verified

| Label | Meaning |
|---|---|
| **pret** | Read in pret/pokered at the pinned commit |
| **both** | pret, and the code or trainer header was found in the ROM by a scan |
| **model** | Static reachability over pret's map data (see "Reachability model"), not played in an emulator |
| **state** | Read from `states/has_pokedex_squirtle.state` through the repo's `open_emulator` and `restore` |
| **unverified** | Not independently shown; says what it rests on |

`Pokemon Red.gb` SHA-1 is `ea9bcae617fdf159b045185467ae58b2e4a48b9a`, which is also pret's `roms.sha1` entry for `pokered.gbc`, so
pret's source claims to build this exact ROM.

**Addresses:**

| Symbol | Address | How verified |
|---|---|---|
| `wEventFlags` | `D747` up to `wLinkEnemyTrainerName` at `D887`: 320 bytes | pret symbols; every flag address below is `0xD747 + id//8`, bit `id%8`; ROM scan hit the expected flag under the expected routine at 306 of 324 sites unaided |
| `wToggleableObjectFlags` (older name `wMissableObjectFlags`) | `D5A6` to `D5C6`: 32 bytes, bit set means hidden | pret symbols; state: Cerulean rival and Guard 1 are hidden, Rocket and Guard 2 are shown, matching `toggleable_objects.asm` |
| `wBagItems`, `wNumBagItems` | `D31E`, `D31D` | pret symbols; matches `ram_map.py` |
| `wObtainedBadges` | `D356` | pret symbols; matches `ram_map.py` |
| `wStatusFlags1` | `D728` | pret symbols; the state reads 0 here, as expected before any drink. The set itself not observed: unverified at runtime |
| `wSSAnne2FCurScript` | `D665` | pret symbols; not observed at runtime: unverified |

## Progress table, in required order

**Req:** R required by the game's scripts. R-model required in the reachability model, because the trainer's line of sight covers
the shortest route with the fewest fights, which is a modelling result and not proof. R-walkthrough rests on guides. C one of an
alternative set. O optional. Auto is set by a script without a separate player action. Bulk flags are set together by one macro, marked `*` in the "Set by"
column. Order is required order where it matters; many neighbours can swap.

### Already done in the start state

| Step | Req | Constant | id | addr.bit | Set by (pret) | Pret and ROM | Cleared by game | Notes |
|---|---|---|---|---|---|---|---|---|
| Rival in Oak's lab | done | `EVENT_BATTLED_RIVAL_IN_OAKS_LAB` **(set at start)** | $023 | D74B.3 | OaksLab.asm:433 | both: 1 set site(s) | no | Pays nothing |
| Oak's Parcel picked up | done | `EVENT_GOT_OAKS_PARCEL` **(set at start)** | $039 | D74E.1 | ViridianMart.asm:58 | both: 1 set site(s) | no | Pays nothing |
| Parcel delivered | done | `EVENT_OAK_GOT_PARCEL` **(set at start)** | $038 | D74E.0 | OaksLab.asm:601 | both: 1 set site(s) | no | Pays nothing |
| Pokedex | done | `EVENT_GOT_POKEDEX` **(set at start)** | $025 | D74B.5 | OaksLab.asm:600 | both: 1 set site(s) | no | Pret's second set site is debug-only (`engine/debug`), absent from this ROM |

### Start to Brock

| Step | Req | Constant | id | addr.bit | Set by (pret) | Pret and ROM | Cleared by game | Notes |
|---|---|---|---|---|---|---|---|---|
| Poke Balls from Oak | O, needed in practice | `EVENT_GOT_POKEBALLS_FROM_OAK` | $024 | D74B.4 | OaksLab.asm:1023 | both: 1 set site(s) | no | Bag is empty at start; a catch needs balls (Oak or a mart) |
| Back in Pallet after balls | auto, O | `EVENT_PALLET_AFTER_GETTING_POKEBALLS` | $006 | D747.6 | PalletTown.asm:4 | both: 1 set site(s) | no | Set on loading Pallet Town |
| Back in Pallet after balls | auto, O | `EVENT_PALLET_AFTER_GETTING_POKEBALLS_2` | $026 | D74B.6 | PalletTown.asm:148 | both: 1 set site(s) | no |  |
| Enter Blue's house | O, trivial | `EVENT_ENTERED_BLUES_HOUSE` | $019 | D74A.1 | BluesHouse.asm:13 | both: 1 set site(s) | no | Walk in |
| Town Map from Daisy | O | `EVENT_GOT_TOWN_MAP` | $018 | D74A.0 | BluesHouse.asm:48 | both: 1 set site(s) | no | Item TOWN_MAP $05 too |
| Daisy walks | auto, O | `EVENT_DAISY_WALKING` | $01A | D74A.2 | PalletTown.asm:138 | both: 1 set site(s) | no | Follows the two above |
| Route 1 potion | O, trivial | `EVENT_GOT_POTION_SAMPLE` | $3C0 | D7BF.0 | Route1.asm:12 | both: 1 set site(s) | no | Talk to the mart employee |
| Route 22 rival, first | O, missable | `EVENT_BEAT_ROUTE22_RIVAL_1ST_BATTLE` | $525 | D7EB.5 | Route22.asm:167 | both: 1 set site(s) | no | Rival enabled at start; beating Brock hides him and clears both enabling flags, so this flag is unreachable afterwards |
| Viridian Forest sight | R-model | `EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_2` | $564 | D7F3.4 | ViridianForest.asm `trainer` | both: trainer header x1 | no | Trainers 0 and 1 are avoidable in the model |
| Brock | R | `EVENT_BEAT_BROCK` | $077 | D755.7 | PewterGym.asm:50 | both: 1 set site(s) | no | Pewter's east exit is blocked by a script at (35,17),(36,17),(37,18),(37,19) until this is set. BADGES `D356` bit 0 |
| TM34 | needs bag slot | `EVENT_GOT_TM34` | $076 | D755.6 | PewterGym.asm:57 | both: 1 set site(s) | no | Skipped when the bag is full |
| Pewter Gym trainer | bulk with Brock | `EVENT_BEAT_PEWTER_GYM_TRAINER_0` | $072 | D755.2 | PewterGym.asm `trainer`; also bulk: PewterGym.asm | both: trainer header x1 | no | Paid at the Brock step if not fought |

### Brock to Misty

| Step | Req | Constant | id | addr.bit | Set by (pret) | Pret and ROM | Cleared by game | Notes |
|---|---|---|---|---|---|---|---|---|
| Route 3 sight | R-model | `EVENT_BEAT_ROUTE_3_TRAINER_0` | $3E2 | D7C3.2 | Route3.asm `trainer` | both: trainer header x1 | no |  |
| Route 3 sight | R-model | `EVENT_BEAT_ROUTE_3_TRAINER_1` | $3E3 | D7C3.3 | Route3.asm `trainer` | both: trainer header x1 | no |  |
| Route 3 sight | R-model | `EVENT_BEAT_ROUTE_3_TRAINER_3` | $3E5 | D7C3.5 | Route3.asm `trainer` | both: trainer header x1 | no |  |
| Route 3 sight | R-model | `EVENT_BEAT_ROUTE_3_TRAINER_6` | $3E8 | D7C4.0 | Route3.asm `trainer` | both: trainer header x1 | no |  |
| Mt. Moon B2F sight | R-model | `EVENT_BEAT_MT_MOON_3_TRAINER_0` | $57A | D7F6.2 | MtMoonB2F.asm `trainer` | both: trainer header x1 | no |  |
| Mt. Moon Super Nerd | R | `EVENT_BEAT_MT_MOON_EXIT_SUPER_NERD` | $579 | D7F6.1 | MtMoonB2F.asm:82 | both: 1 set site(s) | no | Trigger at (13,8) |
| Dome Fossil | C: either | `EVENT_GOT_DOME_FOSSIL` | $57E | D7F6.6 | MtMoonB2F.asm:256 | both: 1 set site(s) | no | Taking a fossil is **required**: the two fossil sprites fill the 2-wide gap at (12,6),(13,6) on the only way out. Only one can be had, so one of these two bits can never pay |
| Helix Fossil | C: either | `EVENT_GOT_HELIX_FOSSIL` | $57F | D7F6.7 | MtMoonB2F.asm:284 | both: 1 set site(s) | no |  |
| Misty | R for Cut and 8 badges, not for Bill | `EVENT_BEAT_MISTY` | $0BF | D75E.7 | CeruleanGym.asm:50 | both: 1 set site(s) | no | Can come before or after Bill. BADGES bit 1 |
| TM11 | needs bag slot | `EVENT_GOT_TM11` | $0BE | D75E.6 | CeruleanGym.asm:57 | both: 1 set site(s) | no |  |
| Cerulean Gym trainer | bulk with Misty | `EVENT_BEAT_CERULEAN_GYM_TRAINER_0` | $0BA | D75E.2 | CeruleanGym.asm `trainer`; also bulk: CeruleanGym.asm | both: trainer header x1 | no | SetEvents |
| Cerulean Gym trainer | bulk with Misty | `EVENT_BEAT_CERULEAN_GYM_TRAINER_1` | $0BB | D75E.3 | CeruleanGym.asm `trainer`; also bulk: CeruleanGym.asm | both: trainer header x1 | no | SetEvents |

### Cerulean to Vermilion (the stretch in question)

| Step | Req | Constant | id | addr.bit | Set by (pret) | Pret and ROM | Cleared by game | Notes |
|---|---|---|---|---|---|---|---|---|
| Rival at the north exit | R | `EVENT_BEAT_CERULEAN_RIVAL` | $098 | D75A.0 | CeruleanCity.asm:175 | both: 1 set site(s) | no | Trigger at (20,6),(21,6), the whole width of the bridge approach. Walled either side (model, script) |
| Nugget Bridge 5 | R-model | `EVENT_BEAT_ROUTE_24_TRAINER_5` | $547 | D7EF.7 | Route24.asm `trainer` | both: trainer header x1 | no | (11,31), sight 1 |
| Nugget Bridge 4 | R-model | `EVENT_BEAT_ROUTE_24_TRAINER_4` | $546 | D7EF.6 | Route24.asm `trainer` | both: trainer header x1 | no | (10,28), sight 1 |
| Nugget Bridge 3 | R-model | `EVENT_BEAT_ROUTE_24_TRAINER_3` | $545 | D7EF.5 | Route24.asm `trainer` | both: trainer header x1 | no | (11,25), sight 1 |
| Nugget Bridge 2 | R-model | `EVENT_BEAT_ROUTE_24_TRAINER_2` | $544 | D7EF.4 | Route24.asm `trainer` | both: trainer header x1 | no | (10,22), sight 1 |
| Nugget Bridge 1 | R-model | `EVENT_BEAT_ROUTE_24_TRAINER_1` | $543 | D7EF.3 | Route24.asm `trainer` | both: trainer header x1 | no | (11,19), sight 1 |
| Route 24, off the bridge | O | `EVENT_BEAT_ROUTE_24_TRAINER_0` | $542 | D7EF.2 | Route24.asm `trainer` | both: trainer header x1 | no | (5,20) facing up. Not on the bridge; avoidable. This is why guides disagree on 5 or 6 |
| Nugget | R, needs bag slot | `EVENT_GOT_NUGGET` | $540 | D7EF.0 | Route24.asm:117 | both: 1 set site(s) | no | Forced at (10,15). Once set, that trigger is off for good, so a loss to this Rocket still opens the road (`Route24DefaultScript`) |
| Route 24 Rocket | effectively R | `EVENT_BEAT_ROUTE24_ROCKET` | $541 | D7EF.1 | Route24.asm:69 | both: 1 set site(s) | no | The battle starts straight after the Nugget. Losing means a blackout but does not re-gate the road |
| Nugget, bag full | O | `EVENT_NUGGET_REWARD_AVAILABLE` | $549 | D7F0.1 | Route24.asm:145 | both: 1 set site(s) | yes: Route24.asm | Bag-full branch only; pays once for being unable to take the Nugget; pushes the player back |
| Route 25 sight | R-model | `EVENT_BEAT_ROUTE_25_TRAINER_6` | $557 | D7F1.7 | Route25.asm `trainer` | both: trainer header x1 | no | (8,4) |
| Route 25 sight | R-model | `EVENT_BEAT_ROUTE_25_TRAINER_3` | $554 | D7F1.4 | Route25.asm `trainer` | both: trainer header x1 | no | (18,8) |
| Route 25 sight | R-model | `EVENT_BEAT_ROUTE_25_TRAINER_2` | $553 | D7F1.3 | Route25.asm `trainer` | both: trainer header x1 | no | (24,4) |
| Route 25 sight | R-model | `EVENT_BEAT_ROUTE_25_TRAINER_5` | $556 | D7F1.6 | Route25.asm `trainer` | both: trainer header x1 | no | (37,4) |
| Talk to Bill, say yes | R | `EVENT_BILL_SAID_USE_CELL_SEPARATOR` | $55E | D7F2.6 | BillsHouse.asm:55 | both: 1 set site(s) | yes: Route25.asm | Cleared on Route 25 if the player leaves before the PC step |
| Use Bill's PC | R | `EVENT_USED_CELL_SEPARATOR_ON_BILL` | $55B | D7F2.3 | bills_house_pc.asm:39 | both: 1 set site(s) | no | `hidden_events/bills_house_pc.asm` |
| Bill restored | auto, R | `EVENT_MET_BILL_2` | $55D | D7F2.5 | BillsHouse.asm:105 | both: 1 set site(s) | no | Set in the same frame as the next one, so 2 flags for one step |
| Bill restored | auto, R | `EVENT_MET_BILL` | $550 | D7F1.0 | BillsHouse.asm:106 | both: 1 set site(s) | no |  |
| Talk to Bill | R, needs bag slot | `EVENT_GOT_SS_TICKET` | $55C | D7F2.4 | BillsHouse.asm:173 | both: 1 set site(s) | no | **This is what opens the way.** Same routine shows Guard 1 at (28,12) and hides Guard 2 at (27,12), who stands on the only tile in front of the trashed house door. Item S_S_TICKET $3F too |
| Leave Bill's house | auto, R | `EVENT_LEFT_BILLS_HOUSE_AFTER_HELPING` | $55F | D7F2.7 | Route25.asm:27 | both: 1 via label check (opcode in pret-labelled routine) | no | Set on loading Route 25. Last flag for 115 moves |
| Walk back; front door; wall hole | R | none: walking through CERULEAN_TRASHED_HOUSE | - | - | - | - | - | Map id, or `wToggleableObjectFlags` D5A7 bit 1 (Guard 2 hidden). New tiles only |
| Cerulean Rocket | R (no Cut yet) | `EVENT_BEAT_CERULEAN_ROCKET_THIEF` | $0A7 | D75B.7 | CeruleanCity.asm:29 | both: 1 set site(s) | no | He is outdoors in the yard behind the house at (30,8), not in it. Triggers at (30,7) and (30,9), so the yard's width is covered. Flag is set after the win, before TM28 |
| TM28 Dig | O, needs bag slot | none: no flag | - | - | - | - | - | Bag item TM_DIG $E4, or D5A6 bit 6 (Rocket hidden; only when the TM fitted). If the bag is full he stays at (30,8), but the flag is set so (30,7) and (30,9) are free |
| Route 5, Underground Path | R | none: no flag on any of these maps | - | - | - | - | - | Map ids. Route 5 has 0 trainers, the underground maps 0. Route 5 Gate to Saffron is closed; the small building is the way |
| Route 6 sight | R-model | `EVENT_BEAT_ROUTE_6_TRAINER_3` | $414 | D7C9.4 | Route6.asm `trainer` | both: trainer header x1 | no | (11,31), sight 3 |
| Route 6 sight | R-model | `EVENT_BEAT_ROUTE_6_TRAINER_4` | $415 | D7C9.5 | Route6.asm `trainer` | both: trainer header x1 | no | (11,30), sight 3 |

### Vermilion to Surge

| Step | Req | Constant | id | addr.bit | Set by (pret) | Pret and ROM | Cleared by game | Notes |
|---|---|---|---|---|---|---|---|---|
| Bike Voucher | O | `EVENT_GOT_BIKE_VOUCHER` | $151 | D771.1 | PokemonFanClub.asm:117 | both: 1 set site(s) | no | Talk to the Fan Club chairman. Item $2D |
| Fan boast | O, trivial | `EVENT_SEEL_FAN_BOAST` | $156 | D771.6 | PokemonFanClub.asm:31 | both: 1 set site(s) | yes: PokemonFanClub.asm | Alternates with the next; talk twice for 2 flags, then nothing |
| Fan boast | O, trivial | `EVENT_PIKACHU_FAN_BOAST` | $157 | D771.7 | PokemonFanClub.asm:54 | both: 1 set site(s) | yes: PokemonFanClub.asm |  |
| Bicycle | O | `EVENT_GOT_BICYCLE` | $0C0 | D75F.0 | BikeShop.asm:29 | both: 1 set site(s) | no | Back in Cerulean. Item $06. Required only for Cycling Road |
| Dock ticket check | R | none: no flag | - | - | - | - | - | Item S_S_TICKET; map id VERMILION_DOCK. Trigger at (18,30), which covers the walk to the (18,31) warp when facing down |
| S.S. Anne rival | R | none: **no flag** | - | - | - | - | - | `wSSAnne2FCurScript` D665 leaves its default value; TOGGLE $71 D5B4 bit 1 is hidden both before and after, so it can't be used alone. Trigger (36,8),(37,8). Model: captain unreachable if blocked |
| Rub the captain's back | R | `EVENT_RUBBED_CAPTAINS_BACK` | $5E1 | D803.1 | SSAnneCaptainsRoom.asm:66 | both: 1 set site(s) | no | Same talk as HM01 |
| HM01 Cut | R, needs bag slot | `EVENT_GOT_HM01` | $5E0 | D803.0 | SSAnneCaptainsRoom.asm:31 | both: 1 set site(s) | no | Item $C4. Usable outside battle only with the Cascade Badge (`start_sub_menus.asm`) |
| Ship leaves | auto | `EVENT_SS_ANNE_LEFT` | $5E2 | D803.2 | VermilionDock.asm:40 | both: 1 via label check (opcode in pret-labelled routine) | no | 4 flags fire from leaving the ship with HM01 in hand |
| Ship leaves | auto | `EVENT_STARTED_WALKING_OUT_OF_DOCK` | $5E4 | D803.4 | VermilionDock.asm:12 | both: 1 set site(s) | no |  |
| Ship leaves | auto | `EVENT_WALKED_OUT_OF_DOCK` | $5E5 | D803.5 | VermilionDock.asm:36 | both: 1 via label check (opcode in pret-labelled routine) | no |  |
| Ship leaves | auto | `EVENT_WALKED_PAST_GUARD_AFTER_SS_ANNE_LEFT` | $5E3 | D803.3 | VermilionCity.asm:27 | both: 1 set site(s) | no |  |
| A Pokemon that can learn Cut | **R** | none: no flag | - | - | - | - | - | Party species and moves; party count at `D163`. **Squirtle's, Wartortle's and Blastoise's TM/HM lists have no Cut** (`data/pokemon/base_stats`). Grass catches on the way in Red: Paras (Mt. Moon), Oddish (Routes 24, 25, 5, 6). `LevelGain` pays a catch through the level sum, while under the cap |
| Teach Cut | **R** | none: no flag | - | - | - | - | - | Party move list holds CUT. Menu only |
| Cut the Vermilion tree | **R** | none: no flag | - | - | - | - | - | Map id VERMILION_GYM. Gym door unreachable without Cut (model) |
| Trash can 1 | R | `EVENT_1ST_LOCK_OPENED` | $161 | D773.1 | vermilion_gym_trash.asm:35 | both: 1 set site(s) | yes: vermilion_gym_trash.asm | Cleared on a wrong second can. Pays once |
| Trash can 2 | R | `EVENT_2ND_LOCK_OPENED` | $160 | D773.0 | vermilion_gym_trash.asm:98 | both: 1 set site(s) | no |  |
| Lt. Surge | R | `EVENT_BEAT_LT_SURGE` | $167 | D773.7 | VermilionGym.asm:69 | both: 1 set site(s) | no | BADGES bit 2 |
| TM24 | needs bag slot | `EVENT_GOT_TM24` | $166 | D773.6 | VermilionGym.asm:76 | both: 1 set site(s) | no |  |
| Vermilion Gym trainers | bulk | `EVENT_BEAT_VERMILION_GYM_TRAINER_0` | $162 | D773.2 | VermilionGym.asm `trainer`; also bulk: VermilionGym.asm | both: trainer header x1 | no | SetEventRange, 3 flags |

### Surge to the Silph Scope

| Step | Req | Constant | id | addr.bit | Set by (pret) | Pret and ROM | Cleared by game | Notes |
|---|---|---|---|---|---|---|---|---|
| HM05 Flash | O | `EVENT_GOT_HM05` | $3D8 | D7C2.0 | Route2Gate.asm:27 | both: 1 set site(s) | no | Route 2 Gate aide; needs 10 Pokemon owned; Cut tree first. Rock Tunnel can be walked dark |
| Cerulean Cut, Route 9, Rock Tunnel, Route 10 | R | none: no story flag | - | - | - | - | - | Map ids; route trainers pay. Forced sights on these maps not computed |
| Buy a drink | R | none: no flag | - | - | - | - | - | Bag FRESH_WATER $3C, SODA_POP $3D, or LEMONADE $3E. **There is no Tea in Red** (FireRed only). `constants/item_constants.asm` has no such item. Celadon Mart roof vending machines |
| Give a drink to a Saffron guard | R | none: no event flag | - | - | - | - | - | `wStatusFlags1` `D728` bit 6 (`BIT_GAVE_SAFFRON_GUARDS_DRINK`). Any of the 4 gates (`Route5Gate.asm` etc.) |
| Erika | R | `EVENT_BEAT_ERIKA` | $1A9 | D77C.1 | CeladonGym.asm:50 | both: 1 set site(s) | no | BADGES bit 3. Gym needs Cut |
| TM21 | needs bag slot | `EVENT_GOT_TM21` | $1A8 | D77C.0 | CeladonGym.asm:57 | both: 1 set site(s) | no |  |
| Celadon Gym trainers | bulk | `EVENT_BEAT_CELADON_GYM_TRAINER_0` | $1AA | D77C.2 | CeladonGym.asm `trainer`; also bulk: CeladonGym.asm | both: trainer header x1 | no | SetEventRange, 7 flags |
| Poster switch | R | `EVENT_FOUND_ROCKET_HIDEOUT` | $1B9 | D77E.1 | GameCorner.asm:465 | both: 1 set site(s) | no | The Rocket by the poster has no flag: TOGGLE $46 D5AE bit 6 hides him |
| Hideout B4F trainers | R | `EVENT_BEAT_ROCKET_HIDEOUT_4_TRAINER_0` | $6A2 | D81B.2 | RocketHideoutB4F.asm `trainer` | both: trainer header x1 | no | Both beaten unlocks the door |
| Hideout B4F trainers | R | `EVENT_BEAT_ROCKET_HIDEOUT_4_TRAINER_1` | $6A3 | D81B.3 | RocketHideoutB4F.asm `trainer` | both: trainer header x1 | no |  |
| Door unlocks | auto | `EVENT_ROCKET_HIDEOUT_4_DOOR_UNLOCKED` | $6A5 | D81B.5 | RocketHideoutB4F.asm:25 | both: 1 set site(s) | no |  |
| Rocket drops Lift Key | R | `EVENT_ROCKET_DROPPED_LIFT_KEY` | $6A6 | D81B.6 | RocketHideoutB4F.asm:193 | both: 1 set site(s) | no | Set on talking to him after the fight |
| Pick up Lift Key | R | none: no flag | - | - | - | - | - | Bag LIFT_KEY $4A. Toggle $88 D5B7 bit 0 goes hidden, shown, hidden. The elevator checks the bag |
| Hideout Giovanni | R | `EVENT_BEAT_ROCKET_HIDEOUT_GIOVANNI` | $6A7 | D81B.7 | RocketHideoutB4F.asm:54 | both: 1 set site(s) | no | Only reveals the Scope |
| **Silph Scope** | R | none: **no flag** | - | - | - | - | - | Bag SILPH_SCOPE $48. Toggle $87 D5B6 bit 7 goes hidden, shown, hidden. Item ball appears after Giovanni. His flag is the nearest proxy but doesn't mean the ball was taken. `engine/battle/core.asm` checks the bag on Tower maps |

### Tower, Fuji, Silph, Sabrina

| Step | Req | Constant | id | addr.bit | Set by (pret) | Pret and ROM | Cleared by game | Notes |
|---|---|---|---|---|---|---|---|---|
| Tower rival | R (model) | `EVENT_BEAT_POKEMON_TOWER_RIVAL` | $0EF | D764.7 | PokemonTower2F.asm:72 | both: 1 set site(s) | no | Trigger (15,5),(14,6); model: 3F unreachable if blocked |
| Which side | auto | `EVENT_POKEMON_TOWER_RIVAL_ON_LEFT` | $0EE | D764.6 | PokemonTower2F.asm:43 | both: 1 set site(s) | yes: PokemonTower2F.asm | Side-dependent, cleared on map load; may never pay |
| Purified zone | auto, O | `EVENT_IN_PURIFIED_ZONE` | $107 | D767.7 | PokemonTower5F.asm:25 | both: 1 set site(s) | yes: PokemonTower5F.asm | Walk on the safe square |
| Ghost Marowak | R | `EVENT_BEAT_GHOST_MAROWAK` | $10F | D768.7 | PokemonTower6F.asm:64 | both: 1 set site(s) | no | Model: 7F unreachable if the trigger is blocked |
| Tower 7F Rocket | R-unverified | `EVENT_BEAT_POKEMONTOWER_7_TRAINER_0` | $111 | D769.1 | PokemonTower7F.asm `trainer` | both: trainer header x1 | no |  |
| Tower 7F Rocket | R-unverified | `EVENT_BEAT_POKEMONTOWER_7_TRAINER_1` | $112 | D769.2 | PokemonTower7F.asm `trainer` | both: trainer header x1 | no |  |
| Tower 7F Rocket | R-unverified | `EVENT_BEAT_POKEMONTOWER_7_TRAINER_2` | $113 | D769.3 | PokemonTower7F.asm `trainer` | both: trainer header x1 | no |  |
| Rescue Mr. Fuji | R | `EVENT_RESCUED_MR_FUJI` | $4CF | D7E0.7 | PokemonTower7F.asm:229 | both: 1 set site(s) | no | Also moves the Rocket from Silph's door (D5A8 bit 7 hidden, D5A9 bit 0 shown). 2 flags for one talk |
| Rescue Mr. Fuji | R | `EVENT_RESCUED_MR_FUJI_2` | $117 | D769.7 | PokemonTower7F.asm:230 | both: 1 set site(s) | no |  |
| Poke Flute | R | `EVENT_GOT_POKE_FLUTE` | $128 | D76C.0 | MrFujisHouse.asm:81 | both: 1 set site(s) | no | Item $49 |
| Pick up Card Key | R-walkthrough | none: no flag | - | - | - | - | - | Bag CARD_KEY $30. Toggle $9D D5B9 bit 5. Door blocks aren't in the reachability model, so 'required' rests on walkthroughs. 20 door flags exist; none individually verified as required |
| Silph rival | unverified | `EVENT_BEAT_SILPH_CO_RIVAL` | $740 | D82F.0 | SilphCo7F.asm:212 | both: 1 set site(s) | no | Trigger (3,2),(3,3). With doors open the model reaches 11F without him; real door states not modelled |
| Silph Giovanni | R | `EVENT_BEAT_SILPH_CO_GIOVANNI` | $78F | D838.7 | SilphCo11F.asm:233 | both: 1 set site(s) | no | Rockets leave, including the one at (34,4) in front of Saffron Gym's door (D5A7 bit 4) |
| Master Ball | O | `EVENT_GOT_MASTER_BALL` | $78D | D838.5 | SilphCo11F.asm:304 | both: 1 set site(s) | no | Item $01 |
| Sabrina | R | `EVENT_BEAT_SABRINA` | $361 | D7B3.1 | SaffronGym.asm:50 | both: 1 set site(s) | no | BADGES bit 5 |
| TM46 | needs bag slot | `EVENT_GOT_TM46` | $360 | D7B3.0 | SaffronGym.asm:57 | both: 1 set site(s) | no |  |
| Saffron Gym trainers | bulk | `EVENT_BEAT_SAFFRON_GYM_TRAINER_0` | $362 | D7B3.2 | SaffronGym.asm `trainer`; also bulk: SaffronGym.asm | both: trainer header x1 | no | SetEventRange, 7 flags |

### Snorlax, Fuchsia, Cinnabar

| Step | Req | Constant | id | addr.bit | Set by (pret) | Pret and ROM | Cleared by game | Notes |
|---|---|---|---|---|---|---|---|---|
| Flute at Route 12 Snorlax | C: either | `EVENT_FIGHT_ROUTE12_SNORLAX` | $48E | D7D8.6 | engine/items/item_effects.asm:1688 | both: 1 set site(s) | yes: Route12.asm | Pays on the flute playing, before the fight. Cleared when the fight starts |
| Route 12 Snorlax | C: either | `EVENT_BEAT_ROUTE12_SNORLAX` | $48F | D7D8.7 | Route12.asm:57 | both: 1 set site(s) | no | Set on win, catch or flee. Model: Fuchsia is unreachable with both Snorlaxes in place, before Surf |
| Flute at Route 16 Snorlax | C: either | `EVENT_FIGHT_ROUTE16_SNORLAX` | $4C8 | D7E0.0 | engine/items/item_effects.asm:1701 | both: 1 set site(s) | yes: Route16.asm |  |
| Route 16 Snorlax | C: either | `EVENT_BEAT_ROUTE16_SNORLAX` | $4C9 | D7E0.1 | Route16.asm:58 | both: 1 set site(s) | no | Cycling Road then needs the Bicycle (`Route16Gate1F`) |
| HM02 Fly | O | `EVENT_GOT_HM02` | $4CE | D7E0.6 | Route16FlyHouse.asm:19 | both: 1 set site(s) | no | Route 16 house. Item $C5. Needs Thunder Badge to use |
| Koga | R | `EVENT_BEAT_KOGA` | $259 | D792.1 | FuchsiaGym.asm:52 | both: 1 set site(s) | no | BADGES bit 4. Surf needs this badge outside battle, so Koga precedes Cinnabar |
| TM06 | needs bag slot | `EVENT_GOT_TM06` | $258 | D792.0 | FuchsiaGym.asm:59 | both: 1 set site(s) | no |  |
| Fuchsia Gym trainers | bulk | `EVENT_BEAT_FUCHSIA_GYM_TRAINER_0` | $25A | D792.2 | FuchsiaGym.asm `trainer`; also bulk: FuchsiaGym.asm | both: trainer header x1 | no | SetEventRange, 6 flags |
| Enter Safari Zone | R | `EVENT_IN_SAFARI_ZONE` | $24F | D790.7 | SafariZoneGate.asm:195 | both: 1 set site(s) | yes: SafariZoneGate.asm; engine/items/item_effects.asm | Costs money. Cleared on leaving; pays once |
| Safari ends | auto, O | `EVENT_SAFARI_GAME_OVER` | $24E | D790.6 | safari_game.asm:54 | both: 1 set site(s) | yes: SafariZoneGate.asm | Step or ball limit |
| Gold Teeth | R | none: no flag | - | - | - | - | - | Bag GOLD_TEETH $40. Toggle $CB D5BF bit 3. Item ball in Safari West |
| HM03 Surf | R | `EVENT_GOT_HM03` | $880 | D857.0 | SafariZoneSecretHouse.asm:19 | both: 1 set site(s) | no | Secret House. Item $C6. Squirtle can learn it |
| Give Gold Teeth | R | `EVENT_GAVE_GOLD_TEETH` | $239 | D78E.1 | WardensHouse.asm:38 | both: 1 set site(s) | no |  |
| HM04 Strength | R | `EVENT_GOT_HM04` | $238 | D78E.0 | WardensHouse.asm:47 | both: 1 set site(s) | no | Item $C7. Squirtle can learn it. Needs Rainbow Badge to use |
| Teach Surf, teach Strength | R | none: no flag | - | - | - | - | - | Party move list. Menu only |
| Mansion switch | R-walkthrough | `EVENT_MANSION_SWITCH_ON` | $278 | D796.0 | PokemonMansion1F.asm:111; PokemonMansion2F.asm:116 | both: 2 set site(s) | yes: CinnabarIsland.asm; PokemonMansion1F.asm; PokemonMansion2F.asm | Toggled by player; cleared on entering Cinnabar; pays once |
| Secret Key | R | none: no flag | - | - | - | - | - | Bag SECRET_KEY $2B. Toggle $C1 D5BE bit 1. Cinnabar Gym door checks the bag (`CinnabarIsland.asm`) |
| Blaine | R | `EVENT_BEAT_BLAINE` | $299 | D79A.1 | CinnabarGym.asm:150 | both: 1 set site(s) | no | BADGES bit 6 |
| TM38 | needs bag slot | `EVENT_GOT_TM38` | $298 | D79A.0 | CinnabarGym.asm:157 | both: 1 set site(s) | no |  |
| Cinnabar Gym trainers | bulk | `EVENT_BEAT_CINNABAR_GYM_TRAINER_0` | $29A | D79A.2 | CinnabarGym.asm:170* | both: 1 set site(s) | no | SetEventRange, 7 flags |
| Cinnabar quiz gate | O or R | `EVENT_CINNABAR_GYM_GATE0_UNLOCKED` | $2A8 | D79C.0 | scripts/CinnabarGym.asm, hidden_events/cinnabar_gym_quiz.asm (FlagAction) | both: `ld hl,base` + FlagAction (bit is runtime) | no | 7 gates, FlagAction with a runtime bit |

### Viridian to the Hall of Fame

| Step | Req | Constant | id | addr.bit | Set by (pret) | Pret and ROM | Cleared by game | Notes |
|---|---|---|---|---|---|---|---|---|
| Viridian Gym opens | auto, R | `EVENT_VIRIDIAN_GYM_OPEN` | $028 | D74C.0 | ViridianCity.asm:24 | both: 1 set site(s) | no | Set on Viridian City when badges are exactly the first 7 |
| Giovanni | R | `EVENT_BEAT_VIRIDIAN_GYM_GIOVANNI` | $051 | D751.1 | ViridianGym.asm:142 | both: 1 set site(s) | no | BADGES bit 7 |
| TM27 | needs bag slot | `EVENT_GOT_TM27` | $050 | D751.0 | ViridianGym.asm:149 | both: 1 set site(s) | no |  |
| Viridian Gym trainers | bulk | `EVENT_BEAT_VIRIDIAN_GYM_TRAINER_0` | $052 | D751.2 | ViridianGym.asm `trainer`; also bulk: ViridianGym.asm | both: trainer header x1 | no | SetEventRange, 8 flags |
| Route 22 rival enabled | auto | `EVENT_2ND_ROUTE22_RIVAL_BATTLE` | $521 | D7EB.1 | ViridianGym.asm:167* | both: 1 set site(s) | yes: OaksLab.asm; Route22.asm | Set by the Giovanni win |
| Route 22 rival, second | R (model) | `EVENT_BEAT_ROUTE22_RIVAL_2ND_BATTLE` | $526 | D7EB.6 | Route22.asm:326 | both: 1 set site(s) | no | Trigger (29,4),(29,5); model: gate unreachable if blocked |
| Route 22 Gate guard | R | none: no flag | - | - | - | - | - | BADGES bit 0. Trigger (4,2),(5,2) in the gate |
| Badge check 1 | R | `EVENT_PASSED_CASCADEBADGE_CHECK` | $530 | D7ED.0 | scripts/Route23.asm (FlagAction) | both: `ld hl,base` + FlagAction (bit is runtime) | no | 7 flags for 7 guards, one per badge |
| Badge check 7 | R | `EVENT_PASSED_EARTHBADGE_CHECK` | $536 | D7ED.6 | scripts/Route23.asm (FlagAction) | both: `ld hl,base` + FlagAction (bit is runtime) | no |  |
| Victory Road 1F switch | R-walkthrough | `EVENT_VICTORY_ROAD_1_BOULDER_ON_SWITCH` | $917 | D869.7 | VictoryRoad1F.asm:35 | both: 1 set site(s) | yes: IndigoPlateauLobby.asm; VictoryRoad2F.asm | Individual need not modelled. Cleared at the Lobby and on 2F |
| Victory Road 2F switch | R-walkthrough | `EVENT_VICTORY_ROAD_2_BOULDER_ON_SWITCH1` | $538 | D7EE.0 | VictoryRoad2F.asm:54 | both: 1 set site(s) | yes: Route23.asm | Cleared by Route 23 |
| Victory Road 2F switch | R-walkthrough | `EVENT_VICTORY_ROAD_2_BOULDER_ON_SWITCH2` | $53F | D7EE.7 | VictoryRoad2F.asm:59 | both: 1 via label check (opcode in pret-labelled routine) | yes: Route23.asm |  |
| Victory Road 3F switch | R-walkthrough | `EVENT_VICTORY_ROAD_3_BOULDER_ON_SWITCH1` | $660 | D813.0 | VictoryRoad3F.asm:42 | both: 1 set site(s) | yes: Route23.asm |  |
| Victory Road 3F switch | R-walkthrough | `EVENT_VICTORY_ROAD_3_BOULDER_ON_SWITCH2` | $666 | D813.6 | VictoryRoad3F.asm:45 | both: 1 set site(s) | yes: Route23.asm |  |
| Lorelei | R | `EVENT_BEAT_LORELEIS_ROOM_TRAINER_0` | $8E1 | D863.1 | LoreleisRoom.asm `trainer` | both: trainer header x1 | yes: HallOfFame.asm; IndigoPlateauLobby.asm | Cleared if the player restarts the Elite Four |
| Lorelei's room autowalk | auto | `EVENT_AUTOWALKED_INTO_LORELEIS_ROOM` | $8E6 | D863.6 | LoreleisRoom.asm:76 | both: 1 set site(s) | yes: HallOfFame.asm; IndigoPlateauLobby.asm |  |
| Bruno | R | `EVENT_BEAT_BRUNOS_ROOM_TRAINER_0` | $8E9 | D864.1 | BrunosRoom.asm `trainer` | both: trainer header x1 | yes: HallOfFame.asm; IndigoPlateauLobby.asm |  |
| Bruno's room autowalk | auto | `EVENT_AUTOWALKED_INTO_BRUNOS_ROOM` | $8EE | D864.6 | BrunosRoom.asm:74 | both: 1 set site(s) | yes: HallOfFame.asm; IndigoPlateauLobby.asm |  |
| Agatha | R | `EVENT_BEAT_AGATHAS_ROOM_TRAINER_0` | $8F1 | D865.1 | AgathasRoom.asm `trainer` | both: trainer header x1 | yes: HallOfFame.asm; IndigoPlateauLobby.asm |  |
| Agatha's room autowalk | auto | `EVENT_AUTOWALKED_INTO_AGATHAS_ROOM` | $8F6 | D865.6 | AgathasRoom.asm:74 | both: 1 set site(s) | yes: HallOfFame.asm; IndigoPlateauLobby.asm |  |
| Lance | R | `EVENT_BEAT_LANCES_ROOM_TRAINER_0` | $8F9 | D866.1 | LancesRoom.asm `trainer` | both: trainer header x1 | yes: HallOfFame.asm; IndigoPlateauLobby.asm |  |
| Lance | R | `EVENT_BEAT_LANCE` | $8FE | D866.6 | LancesRoom.asm:156 | both: 1 set site(s) | yes: HallOfFame.asm; IndigoPlateauLobby.asm |  |
| Lance's door locks | auto | `EVENT_LANCES_ROOM_LOCK_DOOR` | $8FF | D866.7 | LancesRoom.asm:71 | both: 1 set site(s) | yes: HallOfFame.asm; IndigoPlateauLobby.asm |  |
| Champion | R | `EVENT_BEAT_CHAMPION_RIVAL` | $901 | D867.1 | ChampionsRoom.asm:98 | both: 1 set site(s) | yes: HallOfFame.asm | Cleared by the Hall of Fame range reset |
| Hall of Fame stats screen | R, the end | `EVENT_HALL_OF_FAME_DEX_RATING` | $003 | D747.3 | engine/movie/hall_of_fame.asm:210 | both: 1 set site(s) | yes: engine/events/pokedex_rating.asm | Set in `HoFDisplayPlayerStats` and cleared moments later in `DisplayDexRating`, so `milestones.py`'s Champion flag is the stabler signal |

Named flags not listed above are optional, or are the remaining members of a listed group: route trainers, Silph Co. and Tower
trainers, Silph doors, Seafoam boulders, gift TMs and similar. All are counted under "Noise".

## Required steps with no event flag

| Step | Why it's required | How to detect it instead |
|---|---|---|
| Walking through the trashed house | Only way to the yard before Cut | Map id; Guard 2 hidden, `D5A7` bit 1 |
| Route 5, Underground Path | Only way south before Cut | Map ids |
| Dock ticket check | Scripted at (18,30) | Bag `S_S_TICKET` `$3F`, plus map `VERMILION_DOCK` |
| **S.S. Anne rival** | Trigger on the way to the captain (model) | `wSSAnne2FCurScript` `D665`, unverified at runtime |
| **Catch a Cut-capable Pokemon** | Squirtle's line can't learn Cut | Party species; `LevelGain` pays the added levels while under the cap |
| **Teach Cut, use Cut** | Vermilion Gym and Route 9 are behind trees | Party moves, then map ids. Trees leave no flag |
| Buy a drink | Saffron gates | Bag `$3C`, `$3D` or `$3E` |
| Give a drink | Saffron gates | `wStatusFlags1` `D728` bit 6 |
| Pick up Lift Key | Hideout elevator checks the bag | Bag `$4A`. Toggle `D5B7` bit 0 goes hidden, shown, hidden |
| **Silph Scope** | Ghost Marowak can't be fought without it (`core.asm`) | Bag `$48`. Toggle `D5B6` bit 7 goes hidden, shown, hidden. **No flag; `EVENT_BEAT_ROCKET_HIDEOUT_GIOVANNI` only reveals the ball** |
| Pick up Card Key | Walkthroughs; doors not modelled | Bag `$30`. Toggle `D5B9` bit 5 |
| Gold Teeth | Warden gives Strength for them | Bag `$40`. Toggle `D5BF` bit 3 |
| Teach Surf, teach Strength | Cinnabar, Victory Road | Party moves |
| Secret Key | Cinnabar Gym door checks the bag | Bag `$2B`. Toggle `D5BE` bit 1 |
| Route 22 Gate guard | Boulder Badge check | `D356` bit 0 |

Item ids are from `constants/item_constants.asm`. A toggle bit set means hidden: for an item ball that starts visible it means taken, and
for the Scope and the Lift Key, which start hidden, it needs reading alongside the flag that reveals them.

Bag space is a hidden requirement. The Nugget, S.S. Ticket and HM01 all use `GiveItem`, and each one's flag stays unset if the bag is full,
with the 20-slot limit at `ram_map.BAG_CAPACITY`.

Items that do have a flag: S.S. Ticket, Poke Flute, Bicycle, Bike Voucher, Town Map, all HMs, Old Amber, the fossils, Coin Case.

## Noise

| Count | Value | Source |
|---|---|---|
| Bit range | 2,560 | `NUM_EVENTS` = `$A00` |
| Named constants | 507 | `event_constants.asm` |
| Unnamed ids | 2053 | The same file |
| Unnamed ids any code touches | 30, all inside the Indigo Plateau reset ranges, and never set | pret |
| Bits the game can set | 503 | pret, ROM |
| Named but never set (dead) | 4: `EVENT_1B8`, `EVENT_1BF`, `EVENT_67F`, `EVENT_ENTERED_ROCKET_HIDEOUT` (checked, never set) | pret; no set site in the ROM |
| Already set in the start state | 11 | state |
| Excluded by `MUSEUM_TICKET` | 1 | `rewards.py` |
| **Payable at most** | **491**, so at most 491 x 2.0 of event reward; less in practice, since Dome/Helix and Hitmonlee/Hitmonchan are either-or | Computed |
| Named flags the game also clears | 40 | pret, ROM |

**The 2,560 in CLAUDE.md's reward table is the bit range. About 491 can pay.**

**What the 503 settable bits are:**

| Kind | Count |
|---|---|
| Trainer-defeat flags | 310 |
| Static-battle flags (legendaries, Power Plant Voltorbs) | 12 |
| Doors and locks | 31 |
| Boulder switches and holes | 13 |
| Gifts and item hand-overs | 52 |
| Scripted fights (gym leaders, rivals, Snorlax and others) | 31 |
| Script progress and other | 54 |

So trainers are about two thirds of what can pay. In the stretches modelled, most were avoidable.

**Bulk pay.** Winning a gym sets the leader flag, the TM flag and, by macro, the flags of all gym trainers not yet fought. At current
weights, if no gym trainer was fought and the TM fit:

| Leader | Bulk flags | Flags | Event reward | Badge | Budget steps added |
|---|---|---|---|---|---|
| Brock | 1 trainer flags | 3 | 6.0 | 5.0 | 6,144 |
| Misty | 2 trainer flags | 4 | 8.0 | 5.0 | 8,192 |
| Surge | 3 trainer flags | 5 | 10.0 | 5.0 | 10,240 |
| Erika | 7 trainer flags | 9 | 18.0 | 5.0 | 18,432 |
| Koga | 6 trainer flags | 8 | 16.0 | 5.0 | 16,384 |
| Sabrina | 7 trainer flags | 9 | 18.0 | 5.0 | 18,432 |
| Blaine | 7 trainer flags | 9 | 18.0 | 5.0 | 18,432 |
| Giovanni | 8 trainer flags + `EVENT_2ND_ROUTE22_RIVAL_BATTLE` | 11 | 22.0 | 5.0 | 22,528 |

The Fighting Dojo is similar: `SetEventRange EVENT_BEAT_KARATE_MASTER..TRAINER_3` is up to 5 flags, then 3 more for the prize Pokemon, and it's optional.
Fighting a gym's trainers first pays the same flags earlier, plus their levels, so the bulk total is an upper bound on a single step. Flat
pay means Erika, Sabrina or Blaine can pay 9 flags at once, against 3 for Brock.

**Other unevenness:**

- **Two flags for one action:** Bill restored (`MET_BILL` and `MET_BILL_2`), Mr. Fuji rescued, fossil handed in (`SetEvents`), Hitmonlee or
  Hitmonchan (`SetEvents`), ship leaving (4 flags across two map loads).
- **Trivial, near the start:** `EVENT_ENTERED_BLUES_HOUSE` (walk in), `EVENT_GOT_POTION_SAMPLE` (talk), `EVENT_GOT_TOWN_MAP` (talk),
  `EVENT_GOT_POKEBALLS_FROM_OAK` (talk), which then fires the two `EVENT_PALLET_AFTER_GETTING_POKEBALLS` flags on the next Pallet load.
- **Trivial elsewhere:** the two fan boasts, `EVENT_IN_PURIFIED_ZONE`, `EVENT_IN_SEAFOAM_ISLANDS`, `EVENT_IN_SAFARI_ZONE`, the coin gifts, `EVENT_MANSION_SWITCH_ON`.
- **State flags, not progress:** autowalk flags, `EVENT_POKEMON_TOWER_RIVAL_ON_LEFT` (depends on the side), `EVENT_2A7` (set after any battle in Cinnabar Gym), Snorlax `FIGHT` flags.
- **Paid for failing:** `EVENT_NUGGET_REWARD_AVAILABLE` needs a full bag.

**Cleared or re-set by the game.** `EventFlagGain` latches, so each pays once however often it re-sets. Outside the Indigo Plateau range,
the named ones are:

`EVENT_HALL_OF_FAME_DEX_RATING`, `EVENT_BOUGHT_MUSEUM_TICKET`, `EVENT_POKEMON_TOWER_RIVAL_ON_LEFT`, `EVENT_IN_PURIFIED_ZONE`, `EVENT_SEEL_FAN_BOAST`, `EVENT_PIKACHU_FAN_BOAST`, `EVENT_1ST_LOCK_OPENED`, `EVENT_1B8`, `EVENT_1BF`, `EVENT_SAFARI_GAME_OVER`, `EVENT_IN_SAFARI_ZONE`, `EVENT_MANSION_SWITCH_ON`, `EVENT_2A7`, `EVENT_GAVE_FOSSIL_TO_LAB`, `EVENT_LAB_STILL_REVIVING_FOSSIL`, `EVENT_LAB_HANDING_OVER_FOSSIL_MON`, `EVENT_FIGHT_ROUTE12_SNORLAX`, `EVENT_FIGHT_ROUTE16_SNORLAX`, `EVENT_IN_SEAFOAM_ISLANDS`, `EVENT_1ST_ROUTE22_RIVAL_BATTLE`, `EVENT_2ND_ROUTE22_RIVAL_BATTLE`, `EVENT_ROUTE22_RIVAL_WANTS_BATTLE`, `EVENT_VICTORY_ROAD_2_BOULDER_ON_SWITCH1`, `EVENT_VICTORY_ROAD_2_BOULDER_ON_SWITCH2`, `EVENT_NUGGET_REWARD_AVAILABLE`, `EVENT_BILL_SAID_USE_CELL_SEPARATOR`, `EVENT_VICTORY_ROAD_3_BOULDER_ON_SWITCH1`, `EVENT_VICTORY_ROAD_3_BOULDER_ON_SWITCH2`, `EVENT_67F`, `EVENT_VICTORY_ROAD_1_BOULDER_ON_SWITCH`

`IndigoPlateauLobby_Script` clears `$8E0` through `EVENT_LANCES_ROOM_LOCK_DOOR` when the player restarts the Elite Four, and
`HallOfFameResetEventsAndSaveScript` clears `$8E0` through `$907`, which includes `EVENT_BEAT_CHAMPION_RIVAL`.

## Stall analysis: Misty to Vermilion

Weights and budget from `src/pokered/config.py`, assuming the live run uses defaults (`scripts/train.sh` passes no reward or budget options, but
the live run's actual config wasn't checked): `event_weight` 2.0, `tile_weight` 0.005, `badge_weight` 5.0, `level_weight` 0.5,
`base_budget` 10,240, `budget_per_event` 2,048. An episode ends only at the budget.

Moves are the model's shortest route with the fewest fights: single-tile steps, ignoring text, menus, turning and battles. An env step is one button press in a
24-frame decision (`buttons.py`). Steps per tile weren't measured. Real policies wander, so read these as lower bounds.

| Leg | Moves | Forced flags | Event reward | Budget added |
|---|---|---|---|---|
| Gym door to the rival | 47 | 1: rival | 2.0 | 2,048 |
| Rival to the end of the bridge | 29 | 5: bridge trainers | 10.0 | 10,240 |
| Nugget and Route 24 Rocket | few | 2 | 4.0 | 4,096 |
| Route 24 to Bill's door | 79 | 4: Route 25 sights | 8.0 | 8,192 |
| Inside Bill's house, and leaving | few | 6 | 12.0 | 12,288 |
| **Bill's door to the Cerulean Rocket** | **115** | 1: Rocket, at the end | 2.0 | 2,048 |
| **Rocket to Vermilion** | **155** | 2: Route 6 sights, near the end | 4.0 | 4,096 |
| **Total** | **about 425** | **21** | **42.0** | **43,008** |

**Other pay in the stretch, all optional, 24 flags and 48.0 reward at most:** `EVENT_BEAT_ROUTE_24_TRAINER_0`; `EVENT_BEAT_ROUTE_25_TRAINER_` 0, 1, 4, 7 and 8; `EVENT_BEAT_ROUTE_6_TRAINER_` 0, 1, 2 and 5; `EVENT_BEAT_ROUTE_11_TRAINER_` 0 to 9; `EVENT_GOT_BIKE_VOUCHER`, both fan boasts, and `EVENT_GOT_BICYCLE`. A full bag adds `EVENT_NUGGET_REWARD_AVAILABLE`. Route 5 and the underground maps have none.

**Tiles.** Reachable tiles new at each stage in the model: rival 49, Route 24 Rocket 633, ticket 60, Cerulean Rocket 3,339, which is 4,081 tiles or 20.4 reward at most.
The required path can collect at most about 425 x 0.005 = 2.1. Revisits pay nothing.

**Levels.** With 2 badges the level cap is 63 (`LEVEL_CAP_BY_BADGES[2]`), so `LevelGain` keeps paying up to 63 total party levels, at 0.5 each. A catch adds its level to the sum.

**Budget.** The minimum flags to reach Misty under the model is 15, which would make the budget at least 40,960 by then. Any forced flag adds 2,048 steps, far more than the longest
forced gap of about 150 moves. So on a straight route the budget doesn't end an episode in this stretch; a wandering policy can still use it up. How much was left at Misty isn't known, as no run data was used.

### Largest gaps, by distinct required actions with no pay

| Rank | Gap | Moves | Required actions, none paying a flag | Only pay in the gap |
|---|---|---|---|---|
| 1 | Vermilion arrival to HM01 | 141 | Reach the dock; pass the ticket check; board; find the 2F stairs; **fight the S.S. Anne rival**; enter the captain's room; talk | New tiles; levels; 16 optional S.S. Anne trainer flags |
| 2 | Cerulean Rocket to the Route 6 sights | About 150 | Leave the yard south; walk Route 5; pick the small building over the closed gate; cross the underground passage; exit to Route 6 | New tiles; wild-battle levels. Route 6's other 4 sights are avoidable, and one of those is met at 137 moves |
| 3 | Bill's house to the Cerulean Rocket | 115 | Walk back through Routes 25 and 24; enter the house at the door the guard just left; cross it; exit by the wall hole; walk the yard | Very few: about 60 reachable new tiles, 0.30 reward |

Gap 3 has the least pay per move, and it depends on a change the agent has to notice: nothing on the route says the guard moved. The `events` observation does carry the ticket bit.

For scale, `gamma` 0.997 discounts by 0.71 over 115 steps, 0.63 over 155 and 0.65 over 141, on a perfect route. Longer in practice.

### Just past the stretch: the Cut gap

From the ship to Surge's first trash can, forced actions pay no flag: catch a Cut-capable Pokemon if the party has none, open the menu, teach HM01, walk 55 moves to the tree, use Cut, enter the
gym. **The start state's party is one Squirtle, whose line can't learn Cut.** The catch needs Poke Balls, and the start bag is empty. `LevelGain` pays the catch's levels if the sum is under the cap of 63 (or 82 after Surge). Nothing pays the teach or the Cut.
This gap has more distinct actions, and they're of a kind the agent hasn't been paid to do (menu and item use). Whether it stalls here is a question for run data.

## Contradictions and corrections

| Claim | Finding | Evidence |
|---|---|---|
| User: Rival, bridge, Bill, ticket, return, Rocket, in that order | Correct | Scripts, model |
| User: Rocket blocks the way to Route 5 | Correct, but only before Cut. **The guard at (27,12) is what forces Bill first:** with the Rocket gone and no ticket, Route 5 is still unreachable; with the ticket but the Rocket undefeated, it's also unreachable | Model counterfactuals |
| User's list: Tea | **Doesn't exist in Red.** Drinks are Fresh Water, Soda Pop, Lemonade | `item_constants.asm`; FireRed-only is from general knowledge, unverified here |
| User: stalls at Bill's chain | Bill's chain pays well: 6 flags in the house. **The poorly paid parts are around it**, and the Cut step after | This table |
| One guide summary: Rocket "inside" the robbed house | He's in the yard behind it, at (30,8) on the city map | `CeruleanCity.asm` |
| Guides: 5 or 6 bridge trainers | 5 on the bridge, 1 off it on Route 24, plus the Rocket | Objects, model |
| `CONFLICT_RESOLUTIONS.md`: Cerulean rival 65 "against a cap of 39" | True only at 1 badge. With Misty beaten the cap is 63. His team does total 65 (18+15+15+17) | `rewards.py`, `parties.asm` |
| CLAUDE.md: `EventFlagGain` limit "2,560 flags" | Upper bound only. About 491 can pay | "Noise" |
| `milestones.py`: `EVENT_GOT_SS_TICKET`, "the end of Bill's quest" | Nearly: `EVENT_LEFT_BILLS_HOUSE_AFTER_HELPING` follows. All milestone addresses in that file are right | Computed |
| `rewards.py`: museum ticket `D754` bit 0 | Right: id `$68` | Computed |

## ROM check

pret/pokered compiles events as in `macros/scripts/events.asm`:

| Macro | Compiles to |
|---|---|
| `SetEvent`, `ResetEvent`, `CheckEventHL` | `21 lo hi`, then `CB C6+8b`, `CB 86+8b`, or `CB 46+8b` |
| `...ReuseHL` | Only the `CB` op, when the byte matches |
| `CheckEvent` | `FA lo hi`, then `CB 47+8b`; with a second argument, `rrca` x (b+1), or `add a` for bit 7 |
| `CheckAndSetEventA` | `FA`, `bit`, `set`, `EA` |
| `SetEventRange`, `ResetEventRange` | `or`/`and` mask on partial bytes; whole bytes by `ld [hl],$FF`, or `ld [hli],a` runs |
| `EventFlagAddress` | Just `21 lo hi`; the bit comes at runtime through `FlagAction` |
| `trainer` | Data: `db k; db view<<4; dw wEventFlags+(id-k)/8; dw a,b,c,c`. `id` is `8*(addr-D747)+k` |

| Result | Count |
|---|---|
| pret set and reset sites, retail code, counted per flag per site | 324 |
| Confirmed by a linear scan under the same pret-labelled routine | 306 |
| Confirmed by finding the opcode in that routine (the 18 are `AfterBranch` and `ForceReuse` forms, whose `ld hl` is outside the linear scan's reach) | 18 |
| **Unconfirmed** | **0** |
| Trainer headers in pret / matched in ROM by the data pattern | 322 / 322 |
| Pointer-based flags | 14, checked at `ld hl,D7ED` and `ld hl,D79C` (2 sites each); the bit itself is computed at runtime, so that part is pret only |

**Differences:**

- **pret's second set site for `EVENT_GOT_POKEDEX`** is in `engine/debug/debug_party.asm`, debug builds only. The retail ROM has one set site. Not a real disagreement.
- **Unaided counts differ from pret on 39 ids.** Of those, 1 is the debug site, 18 are the `AfterBranch`, `ForceReuse` or range sites confirmed by the label check, and 20 are check sites only: the scanner doesn't decode all compiled check forms (`CheckBothEventsSet`, `and` masks) and it also matches data as code.
  One false hit is `$548` under `UpdateMovingBgTiles`, which pret never uses as an event. Checks don't pay reward, so these weren't chased down.
- **No other disagreement found.** Set sites were matched against pret's label positions, so this check isn't fully independent of pret: it shows the ROM and the source agree, which the SHA-1 already implies.

## Reachability model, and its limits

BFS over pret's data: map sizes, `.blk` and blockset data, tileset collision lists, ledges, pair collisions, warps, connections, sprite positions, and the start state's toggle bits. It models Cut, Surf and Strength as options. Script gates were added by hand from each script's coordinates.

**Checks that it behaves:** Route 9 stops at its tree and Vermilion Gym needs Cut, which match the Wikibooks guide. A bug in the model (a `Copy` map overriding the real one) was found and fixed along the way. Separately, it puts Diglett's Cave's Route 2 entrance behind Cut trees, which no fetched guide states.

**It doesn't model:** walking NPCs, spinner tiles, door blocks changed by scripts (so Silph doors count as open), boulder pushing, sight lines stopped by walls, or anything in battle. Sights are straight lines to the stated range. It was not run in the emulator. Results marked R-model are the fewest-fight route under those limits.

**Stage results, new maps each unlocks:**

| Stage | Unlocks |
|---|---|
| Start | Through Pewter and the Viridian maps. Not Route 3 |
| Brock | Routes 3 and 4, Mt. Moon |
| A fossil taken | Route 4 east, Cerulean and its buildings |
| Cerulean rival | Route 24, up to the Rocket's trigger |
| Past the Route 24 Rocket | Route 25, Bill's house |
| Ticket | Trashed house and the yard's west side |
| Cerulean Rocket | Route 5, the Underground Path, Route 6, Vermilion, Route 11, Diglett's Cave, part of Route 12 |
| Ticket check | S.S. Anne |
| Cut | Vermilion Gym, Route 9 onward, Lavender, Tower, Route 8, Celadon, Hideout, Route 16 |
| Drink | Saffron, except the Silph door and Gym door, which Rockets block |
| Fuji, Silph | Silph Co., then Saffron Gym |
| A Snorlax | Fuchsia, Safari Zone, Routes 13 to 19 |

**Counterfactuals:** Rocket gone and no ticket: no Route 5. Ticket and Rocket undefeated: no Route 5. With Cut, the tree at (19,28) reaches Route 5 regardless, but HM01 is behind Vermilion, so this can't happen on a first pass.

## Proposals (not verified, not recommendations of fact)

These follow from the findings and have not been tested on anything. Any weight here would be a guess.

1. A latched bonus for a required no-flag step in `RewardTerm` form, for example first time a Cut-capable species or the CUT move is in the party, first S.S. Anne rival win, first Silph Scope in the bag. All pay at most once, so they fit the "no farmable reward" rule.
2. A narrower event term, paying required flags more than optional ones. The required list here would need checking for every R-model row first.
3. Logging per-flag first-hit steps from the live run, to replace this analysis's assumptions about where it stalls. `milestones.py` only tracks 13.

## Sources

- pret/pokered at commit `0dbf95cd12f5b1511c62ce69efe283000f780a8f` (2026-10-31): `constants/event_constants.asm`, `constants/item_constants.asm`, `constants/toggle_constants.asm`, `macros/scripts/events.asm`, `macros/scripts/maps.asm`, `scripts/`, `engine/`, `data/`. <https://github.com/pret/pokered/tree/0dbf95cd12f5b1511c62ce69efe283000f780a8f>
- pret/pokered `symbols` branch at `bd77855a9f6407f47d61d94c5eefff7ac83c05a5` (2026-10-24), `pokered.sym`. This is a little older than the source commit; symbols match the ROM's expected build. <https://github.com/pret/pokered/tree/symbols>
- Wikibooks, Red and Blue, Thunder Badge section (Bill, burglar "must" be beaten, Route 5, Underground Path, S.S. Anne, Cut, Surge). <https://en.wikibooks.org/wiki/Pok%C3%A9mon_Red_and_Blue/Thunder_Badge>
- PokeDream Red/Blue walkthrough, Cerulean page (rival, Nugget Bridge, Bill, ticket). <https://pokedream.com/games/redblue/walkthrough/04-cerulean.php>
- NationalDex.io, Red/Green/Blue/Yellow walkthrough (whole game order, overview only). <https://nationaldex.io/guide/pokemon-red-green-blue-yellow/walkthrough>
- Bulbapedia, Red and Blue walkthrough: **returned HTTP 403 to fetches; not read.** Only a search-result summary was seen. <https://bulbapedia.bulbagarden.net/wiki/Red_and_Blue_walkthrough>

Late-game requirements rest on pret's scripts and the model more than on these guides, since the fetched guides are thin past Vermilion. Where that matters, rows say R-walkthrough or unverified.
