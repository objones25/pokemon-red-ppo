# Research: S.S. Anne -> HM01 -> Cut (Pokemon Red, pret/pokered master)

Local copies of fetched pret files: scratchpad/pret/ and scratchpad/repo/ (full tarball). Symbols: scratchpad/pokered.sym
(same ROM as repo: `CheckForHiddenEvent.foundMatchingEvent` = 11:69f0 matches ram_map.HIDDEN_EVENT_MATCH_*).
Paths below are pret/pokered master. Everything not marked "unconfirmed" was read from source.

## 1. S.S. Anne -> HM01

Map ids (constants/map_constants.asm): VERMILION_CITY 0x05, VERMILION_DOCK 0x5E, SS_ANNE_1F 0x5F, SS_ANNE_2F 0x60,
SS_ANNE_3F 0x61, SS_ANNE_B1F 0x62, SS_ANNE_CAPTAINS_ROOM 0x65, VERMILION_GYM 0x5C, VERMILION_TRADE_HOUSE 0xC4.

Sequence:
1. Vermilion City guard at (19,30) checks the S.S. Ticket when the player stands at (18,30) facing down
   (scripts/VermilionCity.asm:45-72, `SSAnneTicketCheckCoords`). Warps (18,31)/(19,31) -> Dock; Dock -> SS_ANNE_1F (26/27,0)
   (data/maps/objects/SSAnne1F.asm warps).
2. 1F: stairs warp at (2,6) -> 2F warp 7 which is (2,4) (SSAnne1F.asm / SSAnne2F.asm).
3. 2F corridor east. The rival is a map object at (36,4) (hidden until triggered; TOGGLE_SS_ANNE_2F_RIVAL). The captain's
   room door is warp (36,4) -> SS_ANNE_CAPTAINS_ROOM warp 1 (data/maps/objects/SSAnne2F.asm).
4. Rival trigger: scripts/SSAnne2F.asm:24-71 `SSAnne2FDefaultScript`: when player coords are in
   `.PlayerCoordinatesArray` = (36,8) or (37,8) (dbmapcoord x,y), the rival is shown, walks down 3 (player at x=37) or 4 (x=36),
   then `SSAnne2FRivalStartBattleScript` (line 89) starts OPP_RIVAL2. No event flag gates the trigger: it is `wSSAnne2FCurScript`
   (wram d665) state 0 -> 1 -> 2 (after battle) -> 3 (exit) -> 4 (noop). Losing resets the script to 0 (line 124-127), so it
   retriggers. (puffer reads `wSSAnne2FCurScript == 4` as a "rival3" reward - see section 7.)
5. Rival team for a Squirtle player. OaksLab.asm ~L796-823: choosing Squirtle sets rival starter = STARTER3 = Bulbasaur.
   SSAnne2F.asm L104-117: Squirtle->trainerNo 1, Bulbasaur->2, Charmander->3, and the rival's starter is compared:
   rival starter STARTER3 (Bulbasaur) -> trainerNo 2. data/trainers/parties.asm L675-678 Rival2Data, row 2:
   **Pidgeotto L19, Raticate L16, Kadabra L18, Ivysaur L20.** (Row 1 = Wartortle team, row 3 = Charmeleon team.)
   A level ~30 Wartortle should win easily; a lead Magikarp L5 would not.
6. After the win: `SSAnne2FRivalAfterBattleScript` (L124): "Cut master" text, rival walks away, script 3 -> 4, rival hidden.
7. Captain's room (map 0x65): arrive at warp (0,7). Captain object (4,2) facing UP; trash (4,1), book (1,2)
   (data/maps/objects/SSAnneCaptainsRoom.asm). Player must stand at (4,3) facing up and press A.
   scripts/SSAnneCaptainsRoom.asm:18-43: if EVENT_GOT_HM01 clear -> "rub back" text (sets EVENT_RUBBED_CAPTAINS_BACK,
   L66, after the PkmnHealed jingle) -> "I feel much better" -> `GiveItem HM_CUT, 1`; success: print received text and
   `SetEvent EVENT_GOT_HM01`. **Bag-full trap**: `GiveItem` fails if the bag has 20 distinct item slots
   (BAG_ITEM_CAPACITY = 20, constants/menu_constants.asm:1) -> "no room" text, GOT_HM01 NOT set, but RUBBED is set; talking
   again retries (HM01 is not lost). No other gating: no rival-win requirement for the captain itself, only that the
   rival fight sits on the path (36,8)/(37,8) before the door at (36,4)... the corridor from 2F west to the door passes y=8.
   (Whether another route to (36,4) avoids (36,8)/(37,8): unconfirmed; the layout implies the approach is through them.)
8. Flags (computed from constants/event_constants.asm with const_skip/const_next; base wEventFlags = 0xD747):
   EVENT_GOT_HM01 index 0x5E0 (byte 0xD803 bit 0), EVENT_RUBBED_CAPTAINS_BACK 0x5E1 (bit 1), EVENT_SS_ANNE_LEFT 0x5E2 (bit 2),
   EVENT_WALKED_PAST_GUARD_AFTER_SS_ANNE_LEFT 0x5E3. EVENT_GOT_SS_TICKET 0x55C, EVENT_BEAT_LT_SURGE 0x167, GOT_BIKE_VOUCHER 0x151
   (the repo's milestones use `_event(0x5E0)` for got_hm01 - matches). Item ids: HM_CUT is HM01 (item 0xC4 standard; not re-verified
   against item_constants numerically), S_S_TICKET 0x3F, BIKE_VOUCHER 0x2D, POKE_BALL 0x04.
9. **Ship departure** (scripts/VermilionDock.asm:1-12, 39-): Dock map script returns immediately unless EVENT_GOT_HM01 is set.
   If set and `wDestinationWarpID == 1` (arriving from Vermilion City side) and SS_ANNE_LEFT clear, the ship leaves
   (`SetEventForceReuseHL EVENT_SS_ANNE_LEFT`). So the ship **never leaves before HM01**; there is no timer and no lockout
   before HM01. After HM01, leaving the ship to the city and re-entering the dock departs it - you cannot go back
   (the guard then lets you through only as a "ship departed" text, VermilionCity.asm:54-61).
   So HM01 is the last thing to do on the ship, and nothing on the ship (trainers, rooms, items) is available afterwards.

## 2. Pokemon that can learn Cut, obtainable before Vermilion Gym (Red version)

Cut learners (grep `CUT` in data/pokemon/base_stats/*.asm, all 151 files fetched): Bulbasaur line, Charmander line, Beedrill,
Sandshrew/Sandslash, Oddish/Gloom/Vileplume, Paras/Parasect, Bellsprout/Weepinbell/Victreebel, Krabby/Kingler, Tentacool/Tentacruel,
Exeggcute, Lickitung, Pinsir, Scyther, Tangela, Farfetch'd, Mew. **Squirtle, Wartortle, Blastoise and Magikarp cannot**
(squirtle/wartortle/blastoise.asm tmhm lists have no CUT; magikarp tmhm is empty).

Red-version wild data (data/wild/maps/*.asm, `IF DEF(_RED)`); slot chances (data/wild/probabilities.asm):
slots 0-9 = 51,51,39,25,25,25,13,13,11,3 of 256. Encounter rate = grass byte/256 per eligible step.

| Source | Cut-capable | Levels | Chance per encounter | Step rate |
|---|---|---|---|---|
| Route 5, Route 6 | Oddish (slots 0,5,6) | 13, 15, 16 | 89/256 = 34.8% | 15/256 |
| Route 7 (past Vermilion's west side) | Oddish | 19, 22 | 64/256 | 15 |
| Route 24 | Oddish (slots 3,4,6) | 12, 13, 14 | 63/256 = 24.6% | 25/256 |
| Route 25 | Oddish (3,4,6) | 12, 13, 14 | 63/256 | 15/256 |
| Mt Moon 1F | Paras (slot 7) | 8 | 13/256 = 5.1% | 10/256 |
| Mt Moon B1F / B2F | Paras (slot 5 / slots 5,6) | 10 / 10, 12 | 25/256 / 38/256 | 10 |
| Route 4, 3, 9, 10, 11, 2, 1, 8 (Red) | none (Ekans, Spearow, Mankey, ...) | | | |
| Viridian Forest | Weedle->Kakuna->Beedrill (evolves L7, L10) | 3-5 | Weedle slots; Beedrill needs L10 | 8/256 |
| Route 2 / 24 / 25 | Weedle (also Beedrill path) | | | |
| Sandshrew | **Blue only** (Route 4/9/10/11/8) | | not in Red | |
| Bellsprout | **Blue only** (Routes 5,6,7,24,25) | | not in Red | |
| Diglett's Cave | Diglett/Dugtrio only: no Cut | | | |
| Trade: Vermilion Trade House | Spearow -> Farfetch'd ("DUX") | Farfetch'd arrives at the Spearow's level | | |

Notes:
- Trade: data/events/trades.asm: `npctrade SPEAROW, FARFETCHD, TRADE_DIALOGSET_HAPPY, "DUX" ; VERMILION_TRADE_HOUSE`. Exists in Red.
  Spearow is catchable (Route 3 L5-8, Route 4 L8-12, Route 9, 11, 22). Farfetch'd is a Cut learner. Trade-house is in Vermilion
  (map 0xC4, door warp (15,13) on the city map), so it is available before the gym.
- Other trades (Route 2 Abra->Mr. Mime, Route 11 gate Nidorino->Nidorina, Underground Path Nidoran) give no Cut learner.
- Other Cut-learner sources before the gym: Red has no gift Cut-capable Pokemon before Vermilion (starters are not available:
  the start state already has Squirtle; Bulbasaur/Charmander pokeballs are gone). Krabby is a fishing encounter (Super Rod / Good Rod
  location unconfirmed here).
- Easiest: Oddish on Route 5/6, directly on the path to Vermilion (Route 6 runs into the city), or Routes 24/25 (Cerulean north).
  Catch rate 255 (oddish.asm). Paras catch rate 190, Farfetch'd 45.
- Obedience (traded or high-level mon vs badges) and HM move use in the field: field Cut needs only the Cascade Badge
  (see section 5), not obedience; obedience matters only in battle. Farfetch'd level-based disobedience: unconfirmed, irrelevant to field use.

## 3. Poke Balls and catching

Marts (data/items/marts.asm): Viridian, Pewter, Cerulean and Vermilion marts all list POKE_BALL first. Price 200
(data/items/prices.asm:6). Viridian Mart: `ViridianMartCheckParcelDeliveredScript` switches to the normal clerk text only when
EVENT_OAK_GOT_PARCEL is set (scripts/ViridianMart.asm:8-17). The start state already is after the Pokedex (parcel delivered), so it
should be set - NOT verified by reading the state. Danger: `ViridianMartDefaultScript` (cur script 0) runs the "came from Pallet Town"
parcel scene on first entry and `GiveItem OAKS_PARCEL` + SetEvent EVENT_GOT_OAKS_PARCEL (L36-60) regardless; harmless but it consumes a
bag slot.

Free balls:
- Oak, in his lab, gives 5 Poke Balls once (`CheckAndSetEvent EVENT_GOT_POKEBALLS_FROM_OAK`, OaksLab.asm:1022-1026) **only if**
  EVENT_BEAT_ROUTE22_RIVAL_1ST_BATTLE is set, no Poke Ball is in the bag, and NOT (Pokedex owned count >= 2 with EVENT_GOT_POKEDEX)
  (OaksLab.asm:970-995). An agent that owns Magikarp+Squirtle (2 species) gets the Pokedex rating instead, never balls.
- One ground Poke Ball in Viridian Forest at (1,31) (data/maps/objects/ViridianForest.asm:37). `object_event ... POKE_BALL` is the only
  Poke Ball item-ball before Vermilion (grep of all objects).
- Route 1 youngster gives a Potion only (the "also Poke Balls" text is flavor).

Catch mechanics (engine/items/item_effects.asm:104-:363+, ItemUseBall):
- Balls only in a wild battle (wIsInBattle == 1). Rand1 for Poke Ball 0-255. Status: sleep/freeze subtract 25, burn/par/poison 12; if
  Rand1 < Status, caught.
- Else if Rand1 - Status > CatchRate: fail. Else W = (MaxHP*255/12) / max(HP/4,1); if W > 255 caught; else caught if Rand2 <= W.
- Full HP, no status, catch rate 255 (Oddish): W = 85 -> ~33.6% per ball. At HP <= 1/4: W >= 255 -> always (given Rand1 <= 255). Paras (190)
  also needs Rand1 <= 190 (74.6%). Farfetch'd (45): Rand1 <= 45 (18%). So weakening is not required for Oddish (about 3 balls on average)
  but helps greatly, and status makes it near-certain. Bound: Poke Ball price 200; 5 balls ~ 1000.
- Full party (6) with a full box: ball refused. Party count < 6: caught mon goes to party; else to the box.
- Detection of a catch: `wPokedexOwned` (0xD2F7, 19 bytes; ram_map.POKEDEX_OWNED) gains the species bit; `wPartyCount` (0xD163) +1
  or `wBoxCount` (0xDA80) +1; `wCapturedMonSpecies` (0xD11C) set. Hook: ItemUseBall.captured = bank 03, addr 0x578b (sym), runs
  for a successful capture (Master Ball path and success path both reach it). Already-owned species do not change Pokedex bits.
- Ghost Marowak and Pokemon Tower ghosts cannot be caught (irrelevant here).

## 4. Teaching HM01

Engine: ItemUseTMHM (item_effects.asm:2154-2249; sym 03:6479). Not in battle. Path: START > ITEM > HM01 > USE > "Booted up an HM!
It contained CUT! Teach CUT to a Pokemon?" YES > party menu (TMHM_PARTY_MENU, shows ABLE / NOT ABLE) > pick the mon. A mon that
cannot learn it: "cannot learn" text and the party menu reopens (loop `.chooseMon`); cancel returns. Already known: reopens.
LearnMove (sym 01:6e43): with 4 moves it asks which move to forget; HM moves in that mon cannot be forgotten, a normal move can. The
HM is not consumed (IsItemHM). Prefer a mon with < 4 moves (Oddish L12-16 has Absorb + a couple of level-up moves, so likely < 4;
unconfirmed exact count).
Detect: any party mon's 4 move bytes == 0x0F (move_constants CUT = 15). Moves at wPartyMon1Moves 0xD173, +0x2C per mon
(party mon struct stride 44, standard; the repo's frame.py already reads moves).

## 5. Using Cut in the field

Menu: START > POKEMON > select the mon with Cut > CUT (field move menu lists moves the mon knows).
engine/menus/start_sub_menus.asm L152-159: requires **BIT_CASCADEBADGE** (Misty's badge; the agent already beats Misty) else
"newBadgeRequired" text. Then `predef UsedCut` (engine/overworld/cut.asm:1; sym 03:6f54).
UsedCut: overworld tileset needs `wTileInFrontOfPlayer` (0xCFC6) == 0x3D (cut tree) or 0x52 (**grass**); gym tileset needs 0x50. Else
"nothing to cut" and no effect. **Cutting grass also succeeds** in the overworld (hook consequence: filter on wCutTile == 0x3D).
On success: `wCutTile` (0xCD4D) = the tile, `wActionResultOrTookBattleTurn` = 1, `ReplaceTreeTileBlock` (cut.asm:~178; sym 03:709f)
rewrites the **block in the in-RAM block map** using data/tilesets/cut_tree_blocks.asm (block $32->$6D, $33->$6C, $34->$6F, $35->$4C,
$60->$6E, $0B->$0A, $3C->$35, $3F->$35, $3D->$36). `wOverworldMap` = 0xC6E8 (block map; width = wCurMapWidth + 6 border
per row). **No event flag is set; nothing persistent**: the map is reloaded from the ROM .blk on every map load, so trees regrow
when you leave and re-enter the map (inference from the code: no write anywhere else; the engine has no other cut state).
Also after AnimCut the player can walk through (cut block tiles 0x2c are in Overworld_Coll passable list, data/tilesets/collision_tile_ids.asm).

Detection of "a tree was cut":
- PyBoy hook (the repo already uses bank/addr hooks): `ReplaceTreeTileBlock` bank 03, addr 0x709F (from the symbols file; this
  routine is only reached when the cut proceeds), then read wCutTile (0xCD4D) == 0x3D (tree; 0x52 is grass, 0x50 gym tree) and the
  map id 0xD35E. Or `UsedCut.canCut` bank 03 addr 0x6F82 (note: the instruction at that label is the one storing wCutTile, so read
  `wTileInFrontOfPlayer` instead, 0xCFC6, as pokemonred_puffer does). `UsedCut.nothingToCut` is 03:6f77 (failed attempts).
- RAM-only alternative: compare block ids in wOverworldMap at the tree block ($35 vs $4C for Vermilion). Block-read details
  unverified against the repo's frame.py.
- Symbols file: https://raw.githubusercontent.com/pret/pokered/symbols/pokered.sym ; lines are `BANK:ADDR Label`
  (e.g. `03:6f54 UsedCut`, `03:709f ReplaceTreeTileBlock`, `03:578b ItemUseBall.captured`, `03:6479 ItemUseTMHM`). Same version as the repo's
  hidden-event hook (11:69f0 matches).

Vermilion tree: maps/VermilionCity.blk is 20x18 blocks (map_constants L31). Block (x=7, y=9) is block id $35, which the cut table
swaps to $4C. From gfx/blocksets/overworld.bst, block $35's 16 tiles are `2c 2c 2d 2e / 2c 2c 3d 3e / 40 41 2c 2c / 50 51 2c 2c`,
so the tree (tiles 3d/3e, `wTileInFrontOfPlayer`=0x3d) is the **top-right 2x2 tile step of that block = step (x=15, y=18)**
(each block is 2x2 steps; map step coords = block*2). Approach: stand at (15,17) facing down (block (7,8) is $31, open path) or
(14,18)? facing right (that step is tree-free; reachability unverified). Gym door: warp (12,19) -> VERMILION_GYM warp; the region
of blocks x 3-7, y 10-11 behind the tree holds the door (block row 9 at x=4..6 is the gym building $10 $11 $12). Whether any
ledge or other route reaches the door without the tree: unconfirmed; community knowledge says the tree is the only way.
Because trees regrow on map reload, the agent must Cut again each time it re-enters Vermilion after leaving it.

## 6. Lt. Surge's gym

Entering the gym: only the tree (and the badge/Cut from section 5). Inside (scripts/VermilionGym.asm; objects): door at warps
(4,17)/(5,17). Lt. Surge object at (5,1); trainers Gentleman (9,6), Super Nerd (3,8), Sailor (0,10). The electric-lock puzzle:
`VermilionGymSetDoorTile` shows the double door tile $24 until EVENT_2ND_LOCK_OPENED; two hidden trash-can switches
(first can random: `wFirstLockTrashCanIndex` set from Random at map load, scripts/VermilionCity... L16-21 is the city; gym
version in the gym engine - detail unconfirmed) open it. Trainers are not required to be beaten to reach Surge once the lock is open.
Fighting Surge: Thunder Badge, TM24 (needs a bag slot; bag-full variant exists, L88-94). Trash can presses are hidden events, so the repo's
HiddenEventGain already covers them.

## 7. What others did

- **drubinstein/pokemonred_puffer** (no license; ideas only; fetched main, grep'd): config.yaml L36-58 has `auto_teach_cut`,
  `auto_use_cut`, `auto_use_strength`, `auto_use_surf`, `auto_teach_surf`, `auto_teach_strength`, `auto_solve_strength_puzzles`,
  `auto_pokeflute`, `auto_next_elevator_floor` (comments: "teach cut if possible", "use cut when standing next to a tree").
  `teach_hm` (environment.py ~L892) **writes RAM**: finds a party mon whose species is in `CUT_SPECIES_IDS` and overwrites the first
  non-HM move slot with the Cut move id and PP. `cut_if_next` (L1013) is a scripted menu sequence that presses START > POKEMON >
  mon > CUT when facing a tree. Both are scripting, defaulted False for cut in config.yaml (surf/strength True).
  Also an episode **reset** if the party has no Cut-capable mon (environment.py ~L803: `reset=True`, reward `-0.5*total`) -
  a penalty/termination, not latched reward.
  Hooks: `pyboy.hook_register(None, "UsedCut.canCut", cut_hook, context=True)` and `"UsedCut.nothingToCut"` (L287-289); the hook
  records tile-in-front and map coords into dicts `valid_cut_coords` (tile 0x3D or 0x50), `invalid_cut_coords`, `cut_tiles`.
  Rewards (rewards/baseline.py; config.yaml L279-): `cut_coords` and `cut_tiles` (set-based => latched per coordinate/tile),
  `hm_count: 10` (HMs in bag), `taught_cut: 10`, `obtained_move_ids`, `caught_pokemon: 4`, `got_hm01: 5`, `rubbed_captains_back: 5`,
  `ss_ticket: 5`, `rival3` (reads `wSSAnne2FCurScript == 4`, i.e. the rival-beaten script state, latched by the game state),
  `explore_npcs`, `explore_hidden_objs` (latched). Also gating menu rewards on `taught_cut` (`start_menu`,`pokemon_menu`, `stats_menu`,
  `bag_menu` multiplied by `int(taught_cut)`) - the "seen_*_menu" flags are sticky (latched) but weights are small. The sweep config
  (sweep-config.yaml / cut-config.yaml) tunes `cut_coords`, `valid_cut_coords` (1.5-4.8 in tuned configs), `invalid_cut_coords` (~0.0001-0.001).
  `taught_cut` is a function of current party state (not latched; the party can lose/regain) - not farmable by itself, but a reward of
  `10*int(taught_cut)` re-pays if cut is lost and re-learned only if the term is a running total (it is not a delta).
  How well it worked: unconfirmed; their README claims runs beat the game with the scripted `auto_*` options, no numbers I verified.
- **PWhiddy/PokemonRedExperiments**: grep for cut/hm/teach in all .py found nothing; the v2 reward is level, events, badges,
  exploration, healing, opponent level - no Cut or HM handling. (Matches CLAUDE.md.)
- **Pleines et al. "Playing Pokemon Red via Deep RL"** (arXiv 2502.19920, https://arxiv.org/pdf/2502.19920): baseline reaches Cerulean City
  gym-and-beyond, no Cut-specific reward found in the summaries. Cut mentioned only as a map obstacle. **PokeRL** (arXiv 2604.10812):
  anti-loop systems, per-map visited masks, reaches the second gym; no Cut. Seen in search summaries only, not read fully.
- Latched vs repeatable summary: latched = event-flag based (got_hm01, rubbed_captains_back, ss_ticket), set-of-coordinates (cut_coords,
  cut_tiles, hidden objs, npcs), seen-menu sticky flags; non-latched = `hm_count`, `taught_cut`, `caught_pokemon` computed from current state (risk
  only if the quantity can go down and back up: a stored/released Pokemon and re-caught would re-pay a running count).

## Recommended latched reward design for this repo (suggestions, nothing implemented)

1. HM01: the repo's EventFlagGain already pays EVENT_GOT_HM01 once if the flag is within the ~491 payable bits (check it is, and RUBBED).
2. Poke Balls: pay once when the bag first contains POKE_BALL count > 0? That is latched if "first time owned" but buying then selling
   cannot re-pay if latched. Alternatively NewSpecies already pays a catch (Pokedex owned bit), with no extra term needed except Cut learner
   bias.
3. Cut learner: latched `OwnedCutLearner` - pay once the first time a party/box species in the Cut set is owned (Pokedex bits for Oddish 43 /
   Paras 46 / Farfetch'd 83 / Beedrill 15 etc.; dex numbers: verify in pokemon_constants.asm before use). Pays once per episode.
4. Cut taught: latched `KnowsCut`, once when any party mon has move 0x0F (MoveGain already pays for stronger moves - Cut is power 50 acc 95,
   so MoveGain already pays slightly; unconfirmed that it is not 0 under the repo's table).
5. Tree cut: hook `ReplaceTreeTileBlock` 03:709F, require wCutTile == 0x3D, pay once per (map, tile coords) - or just once per map, or once
   total at Vermilion. A repeated cut of regrown trees pays nothing if keyed by (map, x, y) in an episode-long set.
6. The user's own milestone list already has `knows_cut` and `got_hm01`; reaching the S.S. Anne is 91% but HM01 is 2%, so the likely
   blockers are on 2F: the rival at (36,8)/(37,8), then (4,3)+UP+A in the captain's room, then a free bag slot. Check these first in a stuck-state
   dump (map 0x60 / 0x65, coords).

Unconfirmed items (flag them): item id of HM01 numerically; Viridian Mart parcel state in the start state; gym trash-can mechanics; exact tree
approach tile reachability; how well the puffer scripted options performed.
