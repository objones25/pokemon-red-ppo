"""Static game data read from the ROM."""

from __future__ import annotations

import pytest
from support import ROM, requires_rom

from pokered.ram_map import HM01, RomTables

pytestmark = requires_rom

# Dex numbers of every species whose base stats list CUT among their machines (pret data/pokemon/base_stats/*.asm).
CUT_LEARNERS = {
    1, 2, 3, 4, 5, 6, 15, 27, 28, 43, 44, 45, 46, 47, 69, 70, 71, 72, 73, 83, 98, 99, 108, 114, 123, 127, 151,
}  # fmt: skip


def test_the_species_that_can_learn_cut_match_the_disassembly():
    tables = RomTables(ROM)
    assert {tables.dex_number(species) for species in tables.learners(HM01)} == CUT_LEARNERS


@pytest.mark.parametrize("dex", [7, 8, 9, 129])  # the Squirtle line and Magikarp
def test_the_starter_line_and_magikarp_cannot_learn_cut(dex):
    tables = RomTables(ROM)
    assert dex not in {tables.dex_number(species) for species in tables.learners(HM01)}


@pytest.mark.parametrize("machine", [0, 56])
def test_learners_refuses_a_machine_that_does_not_exist(machine):
    with pytest.raises(AssertionError, match="machine"):
        RomTables(ROM).learners(machine)
