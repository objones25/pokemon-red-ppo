"""Observations are uint8 arrays; the GPU side converts them."""

from __future__ import annotations

import numpy as np
import pytest

from pokered.frame import EVENT_FLAG_BYTES, GameFrame
from pokered.observations import (
    OBSERVATION_SHAPES,
    FrameStack,
    VisitedMap,
    downscale_screen,
    global_tile,
    party_and_badges,
)


def _frame(**overrides) -> GameFrame:
    values = {
        "map_id": 0, "x": 0, "y": 0, "in_battle": False, "event_flags": bytes(EVENT_FLAG_BYTES),
        "badges": 0, "party_levels": (6,), "party_hp": (0.5,),
    }  # fmt: skip
    return GameFrame(**(values | overrides))


def test_downscale_screen_averages_each_two_by_two_block():
    screen = np.zeros((144, 160), np.uint8)
    screen[0:2, 0:2] = np.array([[0, 85], [153, 255]], np.uint8)
    out = downscale_screen(screen)
    assert out.shape == (72, 80)
    assert out[0, 0] == (0 + 85 + 153 + 255) // 4


def test_downscale_screen_returns_uint8():
    assert downscale_screen(np.full((144, 160), 255, np.uint8)).dtype == np.uint8


def test_downscale_screen_does_not_overflow_on_white():
    assert downscale_screen(np.full((144, 160), 255, np.uint8))[10, 10] == 255


def test_frame_stack_fills_with_first_frame_on_reset():
    stack = FrameStack()
    stack.reset(np.full((72, 80), 7, np.uint8))
    assert stack.array.tolist() == np.full((3, 72, 80), 7, np.uint8).tolist()


def test_frame_stack_puts_newest_frame_last():
    stack = FrameStack()
    stack.reset(np.full((72, 80), 1, np.uint8))
    stack.push(np.full((72, 80), 2, np.uint8))
    stack.push(np.full((72, 80), 3, np.uint8))
    assert [int(stack.array[i, 0, 0]) for i in range(3)] == [1, 2, 3]


def test_global_tile_adds_map_origin_and_padding():
    # Pallet Town's origin in map_data.json is (64, 318); padding is 20 on each axis.
    assert global_tile(map_id=0, x=3, y=4) == (4 + 318 + 20, 3 + 64 + 20)


def test_global_tile_is_none_for_unknown_map():
    assert global_tile(map_id=250, x=0, y=0) is None


@pytest.mark.parametrize(("y", "valid"), [(133, True), (134, False), (255, False)])
def test_global_tile_rejects_positions_too_close_to_the_edge_to_crop(y, valid):
    assert (global_tile(map_id=0, x=0, y=y) is not None) is valid


def test_visited_map_crop_keeps_its_shape_at_the_lowest_valid_row():
    assert VisitedMap().crop(0, 0, 133).shape == OBSERVATION_SHAPES["visited"]


def test_visited_map_marks_current_tile_at_crop_center():
    visited = VisitedMap()
    visited.visit(0, 5, 5)
    crop = visited.crop(0, 5, 5)
    assert crop.shape == OBSERVATION_SHAPES["visited"]
    assert crop[0, 24, 24] == 255


def test_visited_map_scales_each_tile_to_two_pixels():
    visited = VisitedMap()
    visited.visit(0, 5, 5)
    assert visited.crop(0, 5, 5).sum() == 255 * 4


def test_visited_map_shows_a_neighbor_one_tile_away():
    visited = VisitedMap()
    visited.visit(0, 6, 5)
    assert visited.crop(0, 5, 5)[0, 24, 26] == 255


def test_visited_map_clears_on_reset():
    visited = VisitedMap()
    visited.visit(0, 5, 5)
    visited.reset()
    assert visited.crop(0, 5, 5).sum() == 0


def test_visited_map_crop_is_empty_for_unknown_map():
    visited = VisitedMap()
    visited.visit(250, 1, 1)
    assert visited.crop(250, 1, 1).sum() == 0


def test_party_and_badges_encodes_hp_and_level_per_slot():
    out = party_and_badges(_frame(party_levels=(6, 50), party_hp=(0.5, 1.0)))
    assert out.tolist()[:4] == [127, 6, 255, 50]


def test_party_and_badges_leaves_empty_slots_zero():
    out = party_and_badges(_frame())
    assert out.tolist()[2:12] == [0] * 10


def test_party_and_badges_appends_badge_byte():
    assert party_and_badges(_frame(badges=0b101)).tolist()[12] == 0b101


def test_party_and_badges_has_declared_shape():
    assert party_and_badges(_frame()).shape == OBSERVATION_SHAPES["party"]


@pytest.mark.parametrize(("key", "shape"), [
    ("screens", (3, 72, 80)), ("visited", (1, 48, 48)), ("party", (13,)), ("events", (320,)),
])  # fmt: skip
def test_observation_shapes(key, shape):
    assert OBSERVATION_SHAPES[key] == shape
