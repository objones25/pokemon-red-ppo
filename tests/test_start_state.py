"""The training start state: Squirtle chosen, Pokedex received, nothing else in progress."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from support import ROM, emulator, requires_rom

from pokered import ram_map
from pokered.buttons import press
from pokered.start_state import START_STATE

GOT_POKEDEX = (0xD74B, 5)  # events.json key 0xD74B-5
OAKS_LAB_MAP = 40
SQUIRTLE_DEX = 7

pytestmark = requires_rom


def _load(pyboy) -> None:
    with START_STATE.open("rb") as handle:
        pyboy.load_state(handle)


def test_start_state_starter_is_squirtle():
    tables = ram_map.RomTables(ROM)
    with emulator() as pyboy:
        _load(pyboy)
        starter = ram_map.read_party(pyboy.memory)[0]
        assert tables.dex_number(starter.species) == SQUIRTLE_DEX


def test_start_state_party_is_only_the_starter():
    with emulator() as pyboy:
        _load(pyboy)
        assert pyboy.memory[ram_map.PARTY_COUNT] == 1


def test_start_state_has_received_pokedex():
    address, bit = GOT_POKEDEX
    with emulator() as pyboy:
        _load(pyboy)
        assert pyboy.memory[address] >> bit & 1 == 1


def test_start_state_is_in_oaks_lab():
    with emulator() as pyboy:
        _load(pyboy)
        assert pyboy.memory[ram_map.MAP_ID] == OAKS_LAB_MAP


def test_start_state_has_no_badges():
    with emulator() as pyboy:
        _load(pyboy)
        assert ram_map.badge_count(pyboy.memory) == 0


def test_start_state_is_not_in_battle():
    with emulator() as pyboy:
        _load(pyboy)
        assert pyboy.memory[ram_map.BATTLE_TYPE] == ram_map.BATTLE_NONE


def _game_memory(pyboy) -> bytes:
    # HRAM holds the RNG bytes (FFD3, FFD4), where a timing difference shows up first.
    return bytes(pyboy.memory[0xC000:0xE000]) + bytes(pyboy.memory[0xFF80:0xFFFF])


def _replay(pyboy, actions) -> list[bytes]:
    _load(pyboy)
    digests = []
    for action in actions:
        press(pyboy, int(action))
        digests.append(hashlib.md5(_game_memory(pyboy), usedforsecurity=False).digest())
    return digests


@pytest.mark.slow
@pytest.mark.parametrize("ticks_before_reload", [0, 1, 7, 1000])
def test_start_state_replay_ignores_emulator_history(ticks_before_reload):
    # Old-format saves depend on the emulator's LCD timing history (docs/CONFLICT_RESOLUTIONS.md).
    actions = np.random.default_rng(0).integers(0, 7, 500)
    with emulator() as fresh:
        expected = _replay(fresh, actions)
    with emulator() as reused:
        _load(reused)
        reused.tick(ticks_before_reload + 1, False, False)
        assert _replay(reused, actions) == expected
