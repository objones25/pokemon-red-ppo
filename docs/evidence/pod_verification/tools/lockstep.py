"""Lockstep comparison of accurate and fast_mode PyBoy through the real PokemonRedEnv. Pod-only evidence code."""
from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from pokered.config import EnvConfig
from pokered.env import PokemonRedEnv

WRAM = (0xC000, 0xE000)
HRAM = (0xFF80, 0xFFFF)
RNG = {0xFFD3, 0xFFD4}
OAM = (0xFE00, 0xFEA0)
VRAM = (0x8000, 0xA000)
SRAM_NOTE = "cartridge RAM is not compared: only the Save menu writes it"
MAP_ID, X_POS, Y_POS, BATTLE = 0xD35E, 0xD362, 0xD361, 0xD057


def mem(pyboy, span) -> bytes:
    return bytes(pyboy.memory[span[0]:span[1]])


def diff_addresses(a: bytes, b: bytes, base: int) -> list[int]:
    x, y = np.frombuffer(a, np.uint8), np.frombuffer(b, np.uint8)
    return (np.nonzero(x != y)[0] + base).tolist()


def save(pyboy) -> bytes:
    buffer = io.BytesIO()
    pyboy.save_state(buffer)
    return buffer.getvalue()


def load(pyboy, state: bytes) -> None:
    pyboy.load_state(io.BytesIO(state))


def place(pyboy) -> tuple[int, int, int, int]:
    m = pyboy.memory
    return (m[MAP_ID], m[X_POS], m[Y_POS], m[BATTLE])


def sprite_mask(pyboy) -> np.ndarray:
    """True where a visible OAM sprite covers the 144 x 160 screen."""
    mask = np.zeros((144, 160), bool)
    height = 16 if pyboy.memory[0xFF40] & 4 else 8
    for i in range(40):
        y, x = pyboy.memory[0xFE00 + 4 * i] - 16, pyboy.memory[0xFE00 + 4 * i + 1] - 8
        if -height < y < 144 and -8 < x < 160:
            mask[max(y, 0):y + height, max(x, 0):x + 8] = True
    return mask


class Pair:
    """Two real envs fed the same actions. ``fast=False`` gives the accurate-vs-accurate control."""

    def __init__(self, rom: Path, state: Path, *, fast: bool = True) -> None:
        base = EnvConfig(rom=rom, state=state, base_budget=10**9, stagger_first_episode=False)
        self.a = PokemonRedEnv(base)
        self.b = PokemonRedEnv(replace(base, fast_mode=fast))
        self.obs_a, _ = self.a.reset(seed=0)
        self.obs_b, _ = self.b.reset(seed=0)
        self.reward_a = self.reward_b = 0.0

    @property
    def emus(self):
        return (self.a.pyboy, self.b.pyboy)

    def write(self, address: int, value: int) -> None:
        for emu in self.emus:
            emu.memory[address] = value

    def step(self, action: int) -> None:
        self.obs_a, ra, *_ = self.a.step(action)
        self.obs_b, rb, *_ = self.b.step(action)
        self.reward_a += ra
        self.reward_b += rb

    def compare(self) -> dict:
        a, b = self.a.pyboy, self.b.pyboy
        wram = diff_addresses(mem(a, WRAM), mem(b, WRAM), WRAM[0])
        hram = diff_addresses(mem(a, HRAM), mem(b, HRAM), HRAM[0])
        oam = diff_addresses(mem(a, OAM), mem(b, OAM), OAM[0])
        vram = diff_addresses(mem(a, VRAM), mem(b, VRAM), VRAM[0])
        sa, sb = a.screen.ndarray[:, :, 0], b.screen.ndarray[:, :, 0]
        differ = sa != sb
        pixels = int(differ.sum())
        out = {
            "wram": wram,
            "hram": hram,
            "oam": len(oam),
            "vram": len(vram),
            "place_equal": place(a) == place(b),
            "place_a": place(a),
            "place_b": place(b),
            "events_equal": mem(a, (0xD747, 0xD887)) == mem(b, (0xD747, 0xD887)),
            "pixels": pixels,
            "obs_screen_differs": bool((self.obs_a["screens"][-1] != self.obs_b["screens"][-1]).any()),
            "obs_other_differs": any(bool((self.obs_a[k] != self.obs_b[k]).any()) for k in ("visited", "party", "events")),
            "ly": (a.memory[0xFF44], b.memory[0xFF44]),
        }
        if pixels:
            rows, columns = np.nonzero(differ)
            mask = sprite_mask(a) | sprite_mask(b)
            out["screen"] = {
                "bbox": [int(rows.min()), int(rows.max()), int(columns.min()), int(columns.max())],
                "in_sprites": float(mask[differ].mean()),
            }
        return out

    def close(self) -> None:
        self.a.close()
        self.b.close()


def as_line(record: dict) -> str:
    return json.dumps(record, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))


def write_png(path: Path, gray: np.ndarray) -> None:
    """A grayscale PNG, without needing Pillow on the pod."""
    import struct
    import zlib

    height, width = gray.shape

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = b"".join(b"\0" + gray[row].astype(np.uint8).tobytes() for row in range(height))
    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
