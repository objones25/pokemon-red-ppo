"""Turns a window of finished episodes into the numbers and tables worth reading."""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from PIL import Image

from pokered import ram_map
from pokered.buttons import BUTTONS
from pokered.negative_space import check_finite, require
from pokered.telemetry import PHASE_STEPS, EpisodeReport, Fight, Snapshot

_MILESTONE: Final[str] = "milestone_"
_STEPS_TO: Final[str] = "steps_to_"
# Where the best episode matters as much as the mean. Anything else's max is noise.
_KEEP_MAX: Final[frozenset[str]] = frozenset({"events", "tiles", "badges", "max_level"})
_BITS_PER_BYTE: Final[int] = 8
_MAX_SNAPSHOT_STATE_BYTES: Final[int] = 1 << 22


@dataclass(frozen=True, slots=True)
class PlaceTotals:
    """Sums over rollout steps on one map."""

    steps: int
    entropy: float
    value_error: float


@dataclass(frozen=True, slots=True)
class Report:
    scalars: dict[str, float]
    tables: dict[str, tuple[tuple[str, ...], list[tuple[Any, ...]]]]
    snapshots: list[Snapshot]


def build_report(episodes: Sequence[EpisodeReport], *, max_snapshots: int = 4, places: Mapping[int, PlaceTotals] | None = None) -> Report:
    require(episodes, "no episodes to report")
    require(max_snapshots >= 0, f"max_snapshots cannot be negative, got {max_snapshots}")
    names = episodes[0].summary.keys()
    require(all(episode.summary.keys() == names for episode in episodes), "episodes must report the same statistics")
    scalars = {
        "progress/episodes": float(len(episodes)),
        **_summary_scalars(episodes),
        **_action_shares(episodes),
        **_reward_by_term(episodes),
        "progress/lost_fights_mean": sum(fight.lost for episode in episodes for fight in episode.detail.fights) / len(episodes),
    }
    for name, value in scalars.items():
        check_finite(value, name)
    tables = {
        "flags": (("flag", "address", "bit", "rate", "median_step"), _flag_rows(episodes)),
        "fights": (("opponent", "map_id", "fought_per_episode", "lost_per_episode", "loss_rate", "mean_party_max_level"), _fight_rows(episodes)),
        "maps": (("map_id", "step_share", "end_share", "stall_share", "entropy", "value_error"), _map_rows(episodes, places or {})),
        "reward_phases": (("term", "phase_start_step", "per_episode"), _phase_rows(episodes)),
    }
    return Report(scalars=scalars, tables=tables, snapshots=_pick_snapshots(episodes, max_snapshots))


def _summary_scalars(episodes: Sequence[EpisodeReport]) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for name in episodes[0].summary:
        values = [episode.summary[name] for episode in episodes]
        finite = [value for value in values if math.isfinite(value)]  # NaN: a milestone this episode never reached.
        if name.startswith(_MILESTONE):
            scalars[f"story/{name.removeprefix(_MILESTONE)}_rate"] = sum(values) / len(values)
        elif name.startswith(_STEPS_TO):
            if finite:
                scalars[f"story/{name.removeprefix(_STEPS_TO)}_median_step"] = statistics.median(finite)
        else:
            require(len(finite) == len(values), f"progress statistic {name} must always have a value")
            scalars[f"progress/{name}_mean"] = sum(values) / len(values)
            if name in _KEEP_MAX:
                scalars[f"progress/{name}_max"] = max(values)
    return scalars


def _action_shares(episodes: Sequence[EpisodeReport]) -> dict[str, float]:
    counts = [sum(episode.detail.action_counts[i] for episode in episodes) for i in range(len(BUTTONS))]
    total = sum(counts)
    require(total > 0, "episodes pressed no buttons")
    return {f"actions/{button}": count / total for button, count in zip(BUTTONS, counts, strict=True)}


def _reward_by_term(episodes: Sequence[EpisodeReport]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for episode in episodes:
        for (term, _), value in episode.detail.phase_rewards.items():
            totals[term] = totals.get(term, 0.0) + value
    require(all(value > 0.0 for value in totals.values()), f"terms record only rewards they paid, got {totals}")
    return {f"reward/{term}": value / len(episodes) for term, value in sorted(totals.items())}


def _flag_rows(episodes: Sequence[EpisodeReport]) -> list[tuple[Any, ...]]:
    steps: dict[int, list[int]] = {}
    for episode in episodes:
        for flag, step in episode.detail.flag_steps.items():
            steps.setdefault(flag, []).append(step)
    rows = [
        (
            f"0x{flag:03X}",
            f"0x{ram_map.EVENT_FLAGS_START + flag // _BITS_PER_BYTE:04X}",
            flag % _BITS_PER_BYTE,
            len(hits) / len(episodes),
            float(statistics.median(hits)),
        )
        for flag, hits in steps.items()
    ]
    require(all(row[3] <= 1.0 for row in rows), "a flag can be first reached only once per episode")
    return sorted(rows, key=lambda row: (-row[3], row[4], row[0]))


def _fight_key(fight: Fight) -> str:
    return f"wild@{fight.map_id}" if fight.opponent == "wild" else fight.opponent


def _fight_rows(episodes: Sequence[EpisodeReport]) -> list[tuple[Any, ...]]:
    groups: dict[str, list[Fight]] = {}
    for episode in episodes:
        for fight in episode.detail.fights:
            groups.setdefault(_fight_key(fight), []).append(fight)
    rows = []
    for key, fights in groups.items():
        lost = sum(fight.lost for fight in fights)
        rows.append(
            (
                key,
                fights[0].map_id,
                len(fights) / len(episodes),
                lost / len(episodes),
                lost / len(fights),
                sum(fight.party_max_level for fight in fights) / len(fights),
            )
        )
    require(all(row[4] <= 1.0 for row in rows), "losses cannot outnumber fights")
    return sorted(rows, key=lambda row: (-row[3], -row[2], row[0]))


def _map_rows(episodes: Sequence[EpisodeReport], places: Mapping[int, PlaceTotals]) -> list[tuple[Any, ...]]:
    steps: Counter[int] = Counter()
    for episode in episodes:
        steps.update(episode.detail.map_steps)
    total = sum(steps.values())
    require(total > 0, "episodes spent no steps on any map")
    ends = Counter(episode.detail.end.map_id for episode in episodes)
    stalls = Counter(episode.detail.last_stall.map_id for episode in episodes if episode.detail.last_stall is not None)
    rows = []
    for map_id in sorted(steps | ends | stalls, key=lambda item: (-steps[item], item)):
        place = places.get(map_id)
        rows.append(
            (
                map_id,
                steps[map_id] / total,
                ends[map_id] / len(episodes),
                stalls[map_id] / max(sum(stalls.values()), 1),
                None if place is None else place.entropy / place.steps,
                None if place is None else place.value_error / place.steps,
            )
        )
    return rows


def _phase_rows(episodes: Sequence[EpisodeReport]) -> list[tuple[Any, ...]]:
    totals: dict[tuple[str, int], float] = {}
    for episode in episodes:
        for key, value in episode.detail.phase_rewards.items():
            totals[key] = totals.get(key, 0.0) + value
    require(all(phase >= 0 for _, phase in totals), "a phase cannot be before the episode")
    return [(term, phase * PHASE_STEPS, value / len(episodes)) for (term, phase), value in sorted(totals.items())]


def _pick_snapshots(episodes: Sequence[EpisodeReport], limit: int) -> list[Snapshot]:
    """Stalls first, one per map from the most common map down, then again for second examples, up to ``limit``."""
    by_map: dict[int, list[Snapshot]] = {}
    for episode in episodes:
        stall = episode.detail.last_stall
        if stall is not None:
            by_map.setdefault(stall.map_id, []).append(stall)
    queues = sorted(by_map.values(), key=lambda items: (-len(items), items[0].map_id))
    picked: list[Snapshot] = []
    for depth in range(max((len(queue) for queue in queues), default=0)):
        picked.extend(queue[depth] for queue in queues if depth < len(queue))
    require(all(snapshot.kind == "stall" for snapshot in picked), "only stall snapshots are picked")
    return picked[:limit]


def _atomic_write(path: Path, data: bytes) -> None:
    require(data, f"refusing to write an empty {path.name}")
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def snapshot_stem(step: int, index: int, snapshot: Snapshot) -> str:
    require(step >= 0, f"step cannot be negative, got {step}")
    return f"{step:012d}_{index}_{snapshot.kind}_map{snapshot.map_id}"


def save_report(report: Report, directory: Path, *, step: int) -> None:
    """Write the report as strict JSON, and each snapshot as a picture and a state that reloads in PyBoy."""
    reports, stuck = directory / "reports", directory / "stuck"
    reports.mkdir(parents=True, exist_ok=True)
    stuck.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "scalars": report.scalars,
        "tables": {name: {"columns": columns, "rows": rows} for name, (columns, rows) in report.tables.items()},
        "snapshots": [
            {
                "file": snapshot_stem(step, index, snapshot), "kind": snapshot.kind, "step": snapshot.step,
                "dry_steps": snapshot.dry_steps, "map_id": snapshot.map_id, "x": snapshot.x, "y": snapshot.y,
                "in_battle": snapshot.in_battle, "party_levels": snapshot.party_levels,
            }
            for index, snapshot in enumerate(report.snapshots)
        ],
    }  # fmt: skip
    _atomic_write(reports / f"report_{step:012d}.json", json.dumps(payload, allow_nan=False, indent=2).encode())
    for index, snapshot in enumerate(report.snapshots):
        require(len(snapshot.state) <= _MAX_SNAPSHOT_STATE_BYTES, f"a saved state of {len(snapshot.state)} bytes is not a game state")
        stem = snapshot_stem(step, index, snapshot)
        _atomic_write(stuck / f"{stem}.state", snapshot.state)
        Image.fromarray(snapshot.screen).save(stuck / f"{stem}.png")
