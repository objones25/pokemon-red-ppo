"""Observation pieces: all uint8, so rollout storage stays small and the GPU side converts them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from pokered import ram_map
from pokered.frame import EVENT_FLAG_BYTES, GameFrame
from pokered.negative_space import check_shape, require

Image = NDArray[np.uint8]

OBSERVATION_SHAPES: Final[dict[str, tuple[int, ...]]] = {
    "screens": (3, 72, 80),
    "visited": (1, 48, 48),
    "party": (2 * ram_map.MAX_PARTY + 1,),
    "events": (EVENT_FLAG_BYTES,),
}

_MAP_DATA: Final = Path(__file__).parent / "data" / "map_data.json"
_PADDING: Final[int] = 20
_MAP_ROWS: Final[int] = 444 + 2 * _PADDING  # map_data.json's "Kanto" region is 436 x 444 tiles
_MAP_COLUMNS: Final[int] = 436 + 2 * _PADDING
_CROP_RADIUS: Final[int] = 12  # tiles either side of the player; each tile becomes 2 x 2 pixels
_VISITED: Final[int] = 255


def _load_origins() -> dict[int, tuple[int, int]]:
    regions = json.loads(_MAP_DATA.read_text())["regions"]
    origins = {int(region["id"]): (int(region["coordinates"][0]), int(region["coordinates"][1])) for region in regions if int(region["id"]) >= 0}
    require(origins, f"{_MAP_DATA} holds no maps")
    return origins


_ORIGINS: Final = _load_origins()


def downscale_screen(screen: Image) -> Image:
    """144 x 160 to 72 x 80 by integer 2 x 2 block mean, about 15x cheaper than a float mean."""
    check_shape(screen, (144, 160), "screen")
    wide = screen.astype(np.uint16)
    return ((wide[::2, ::2] + wide[1::2, ::2] + wide[::2, 1::2] + wide[1::2, 1::2]) >> 2).astype(np.uint8)


class FrameStack:
    """The last three screens, oldest first."""

    def __init__(self) -> None:
        self.array: Image = np.zeros(OBSERVATION_SHAPES["screens"], np.uint8)

    def reset(self, frame: Image) -> None:
        check_shape(frame, OBSERVATION_SHAPES["screens"][1:], "frame")
        self.array[:] = frame

    def push(self, frame: Image) -> None:
        check_shape(frame, OBSERVATION_SHAPES["screens"][1:], "frame")
        self.array[:2] = self.array[1:]
        self.array[2] = frame


def global_tile(map_id: int, x: int, y: int) -> tuple[int, int] | None:
    """(row, column) on the Kanto map, or None for a map or position the crop can't represent."""
    require(0 <= map_id <= 0xFF, f"map id must fit one byte, got {map_id}")
    origin = _ORIGINS.get(map_id)
    if origin is None:
        return None
    row, column = y + origin[1] + _PADDING, x + origin[0] + _PADDING
    in_rows = _CROP_RADIUS <= row < _MAP_ROWS - _CROP_RADIUS
    in_columns = _CROP_RADIUS <= column < _MAP_COLUMNS - _CROP_RADIUS
    return (row, column) if in_rows and in_columns else None


class VisitedMap:
    """Tiles visited this episode. The new-tile reward depends on history, so the critic must be able to see it."""

    def __init__(self) -> None:
        self._tiles: Image = np.zeros((_MAP_ROWS, _MAP_COLUMNS), np.uint8)

    def reset(self) -> None:
        self._tiles[:] = 0

    def visit(self, map_id: int, x: int, y: int) -> None:  # nsp: ignore NSP001 global_tile validates; None is a legal unknown map
        tile = global_tile(map_id, x, y)
        if tile is not None:
            self._tiles[tile] = _VISITED

    def crop(self, map_id: int, x: int, y: int) -> Image:
        tile = global_tile(map_id, x, y)
        if tile is None:
            return np.zeros(OBSERVATION_SHAPES["visited"], np.uint8)
        row, column = tile
        window = self._tiles[row - _CROP_RADIUS : row + _CROP_RADIUS, column - _CROP_RADIUS : column + _CROP_RADIUS]
        crop = window.repeat(2, axis=0).repeat(2, axis=1)[None]
        check_shape(crop, OBSERVATION_SHAPES["visited"], "visited crop")
        return crop


def party_and_badges(frame: GameFrame) -> Image:
    """HP (scaled to 0-255) and level per slot, then the badge byte."""
    require(len(frame.party_levels) <= ram_map.MAX_PARTY, f"party of {len(frame.party_levels)} does not fit the slots")
    out = np.zeros(OBSERVATION_SHAPES["party"], np.uint8)
    for slot, (hp, level) in enumerate(zip(frame.party_hp, frame.party_levels, strict=True)):
        out[2 * slot] = int(hp * 255)
        out[2 * slot + 1] = level
    out[-1] = frame.badges
    return out
