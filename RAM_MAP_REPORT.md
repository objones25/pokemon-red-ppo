# Pokemon Red RAM map for PPO training

> Some claims here were later corrected; see `docs/CONFLICT_RESOLUTIONS.md`.

This report maps the RAM and ROM data useful for training a PPO agent on Pokemon Red. `src/pokered/ram_map.py` provides typed readers for most of it.

**ROM:** `POKEMON RED`, international release, 1 MB. Header checksums are valid.
**SHA-1:** `ea9bcae617fdf159b045185467ae58b2e4a48b9a`

These addresses only apply to this revision. `ram_map.RomTables` refuses any other ROM.

## How this was verified

Candidate addresses came from memory of the community `pokered` disassembly, then were checked in two ways:

1. **Code:** a small SM83 disassembler cross-referenced every WRAM access in the ROM. It is a linear sweep that decodes some data as code, so it is not a full annotated disassembly.
2. **Running game:** in PyBoy, distinct values were injected into each field and read back from the game's own screens (party, status, bag, Pokedex, trainer card). Real wild and trainer battles, a warp and a blackout were then logged.

Everything below was confirmed in the running game unless it is listed under [Not fully tested](#not-fully-tested).

## Memory map

### Player and world

| Address | Size | Meaning |
|---|---|---|
| `D347` | 3 | Money, as BCD (`12 34 56` displays ¥123456) |
| `D356` | 1 | Badges, one bit each; bit 0 is Boulder, bit 7 is Earth |
| `D35E` | 1 | Current map ID |
| `D361` / `D362` | 1 each | Y / X position |
| `C109` | 1 | Facing: `0` down, `4` up, `8` left, `C` right |
| `D2F7` / `D30A` | 19 each | Pokedex owned / seen bitfields |
| `D31D`, `D31E` | 1, then pairs | Bag item count, then `(item, qty)` pairs ending in `FF` |
| `D53A` | 1, then pairs | PC item count, in the same format |
| `D158` / `D34A` | 11 each | Player / rival name |
| `D359` | 2 | Player ID, used for the obedience check |
| `DA41`, `DA43`-`DA45` | | Play time: hours, minutes, seconds, frames (only seconds and frames seen changing) |
| `C3A0` | 360 | Screen tilemap, 20x18; `tile(12,0)==0x79` means a text box is open |
| `CC26` | 1 | Menu cursor |

### Party

- The count is at `D163`, the species list at `D164`, and the 6 Pokemon structs start at `D16B`, 44 bytes each.
- OT names start at `D273` and nicknames at `D2B5`, 11 bytes each.
- Multi-byte values are big-endian.

| Offset | Size | Field |
|---|---|---|
| +0 | 1 | Species (internal ID) |
| +1 | 2 | Current HP |
| +4 | 1 | Status: bits 0-2 sleep counter, 3 PSN, 4 BRN, 5 FRZ, 6 PAR |
| +5, +6 | 1 each | Types |
| +8 | 4 | Moves |
| +12 | 2 | OT ID |
| +14 | 3 | EXP |
| +29 | 4 | PP: low 6 bits are current PP, top 2 bits are PP Ups |
| +33 | 1 | Level (use this one) |
| +34 | 2 | Max HP |
| +36 | 8 | Attack, Defense, Speed, Special (2 bytes each) |

### Battle

| Address | Meaning |
|---|---|
| `D057` | `0` none, `1` wild, `2` trainer, `FF` lost |
| `D059`, `D127` | Opponent, enemy level (`>=200` means trainer class + 200) |
| `D05D`, `D031` | Trainer number, trainer class |
| `CCD5`, `CCDC`, `CCDD` | Turn count, player's selected move, enemy's selected move |
| `CD1A` / `CD2E` | Player / enemy stat stages: 6 bytes, 7 is neutral. Observed order is Attack, Defense, Speed, Special, then two more. |
| `D89C`, `D89D`, `D8A4` | Enemy party count, species list, 44-byte structs |

| Field | Player active | Enemy active |
|---|---|---|
| Species | `D014` | `CFE5` |
| HP (2) | `D015` | `CFE6` |
| Status | `D018` | `CFE9` |
| Types (2) | `D019` | `CFEA` |
| Moves (4) | `D01C` | `CFED` |
| Level | `D022` | `CFF3` |
| Max HP (2) | `D023` | `CFF4` |
| Stats (8) | `D025` | `CFF6` |
| PP (4) | `D02D` | not read |

### ROM tables

| File offset | Contents |
|---|---|
| `0x383DE` | Base stats, 28 bytes x 150, indexed by Pokedex number. Fields start: dex, HP, Attack, Defense, Speed, Special, type 1, type 2, catch rate, EXP yield. |
| `0x425B` | Mew's base stats, stored separately |
| `0x38000` | Moves, 6 bytes x 165: ID, effect, power, type, accuracy (out of 255), PP |
| `0x3E474` | Type chart: `(attacker, defender, multiplier x 10)`, 82 entries, ends in `FF` |
| `0x41024` | Internal ID to Pokedex number, indexed by `id-1`; `0` marks an unused slot |

## Traps for training

1. **Species IDs aren't Pokedex numbers.** Bulbasaur is `0x99` and Charmander is `0xB0`.
2. **Selected isn't executed.** A traded Pokemon over the obedience limit ignored orders, spent no PP and hit itself, with all status bytes at 0. Read `CCDC` with care; a PP drop is better evidence the move ran.
3. **Party HP lags battle HP.** The party value updated about 85 frames later, so use `D015` during battle.
4. **Battle stats differ from party stats.** Badges boosted Attack from 101 to 113. It later reached 159 with its stage still neutral, which fits the known Gen 1 badge-boost stacking bug.
5. **Levels can glitch briefly.** `CFF3` and `D022` swapped for one frame mid-turn, so sample only at decision points such as `at_fight_menu`.
6. **Battle data goes stale.** Enemy data, stages and turn count kept their old values after battle, so gate them on `D057`. `read_battle` returns `None` outside battle.
7. **Blackouts can distort rewards.** `D057` stays at `FF` for about 100 frames, then the map changes, money halves and HP fully restores. A reward on HP gain would pay out for losing, so use this as an episode or penalty signal.
8. **The move cursor is 1-based.** The main battle menu's `FIGHT` cursor is 0, but in the move menu slot N is `CC26 == N+1`.
9. **Level-ups lag EXP.** EXP changed at frame 184 and level at frame 377, while evolution showed up by the end of the battle (frame 1560).
10. **Raw PP bytes include PP Ups.** A byte of `0x54` displayed as 20/47, so mask with `0x3F`. The prediction was 48; the 47 fits a recalled +7 cap per PP Up, which was not tested separately.

## Not fully tested

- **Code references only:** event flags at `D747` are set and tested by game scripts. The 320-byte length is confirmed by the pokered symbol file (`wEventFlags` at `D747`, next symbol at `D887`).
- **Recalled, not tested:** coins `D5A4`, walk/bike/surf `D700`, Hall of Fame count `D5A2`, the PC box at `DA80`, and the bag and PC item capacities (20 and 50).
- **Unclear:**
  - `D719` (last blackout map) read 0 and the blackout landed on map 0, but it was never varied.
  - `D365` (last map) stayed 0 through an indoor warp, so its meaning is unconfirmed.
- **Fields not checked:** DVs, stat EXP, catch rate and the +3 level byte in the party struct, and non-zero enemy status.
- **Injected test data:** after one battle, party HP read 65485. That came from unrealistic injected stats (max HP 231 at level 37), not a game bug. `hp_fraction` clamps to 1.0 for this reason.

## Suggested inputs for PPO

**Observations:**

- Screen pixels
- Party HP fractions, levels and status
- Badge bits
- Battle state, gated on `D057`
- Active Pokemon types and moves, with power and type looked up from the ROM tables
- Stat stages

**Reward signals:**

- New `(map, x, y)` tiles visited
- Badge count
- Party level sum
- Event flag count

**Training shortcut:** writing `D059` and `D127` from the overworld starts a battle within a few frames, which is handy for battle-focused training. With `D059 = 201` and `D05D = 1`, a trainer battle started with a two-Pokemon party.

## Using `ram_map.py`

```python
from pyboy import PyBoy
import ram_map

pyboy = PyBoy("Pokemon Red.gb", window="null")
tables = ram_map.RomTables("Pokemon Red.gb")

# After reaching a state past the intro:
state = ram_map.read_state(pyboy.memory)
print(state.map_id, state.money, ram_map.badge_count(pyboy.memory))

if state.battle is not None and ram_map.at_fight_menu(pyboy.memory):
    enemy = state.battle.enemy
    for move_id in state.battle.player.moves:
        if move_id:
            move = tables.moves[move_id]
            print(move.power, tables.type_multiplier(move.type, enemy.types))
```

A default `pyboy.stop()` writes a `.ram` save file next to the ROM. Use `pyboy.stop(save=False)` to avoid it.

Battles pause on text prompts such as the opening "Wild ... appeared!", so `at_fight_menu` only becomes true once something presses A or B to advance them.
