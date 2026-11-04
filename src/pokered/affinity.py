"""Pinning workers to quiet physical cores on the GPU's NUMA node.

On the benchmarked EPYC pod this raised throughput by 56% at 8 envs per worker, restoring the free-running emulator
speed (docs/CONFLICT_RESOLUTIONS.md). Unpinned workers share cores and cross NUMA nodes. On a shared host the lowest
CPUs can be the busiest, and pinning to them cost 40%, so cores are chosen by measured load.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pokered.negative_space import require

_SYSFS: Final[Path] = Path("/sys/devices/system")
_LOAD_SAMPLE_S: Final[float] = 3.0


@dataclass(frozen=True, slots=True)
class CorePlan:
    workers: list[int]
    main: list[int]


def parse_cpu_list(text: str) -> list[int]:
    """Parse the kernel's cpu-list format, such as ``0-63,128-191``."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        first, _, last = part.partition("-")
        if not first.isdigit() or (last and not last.isdigit()):
            raise ValueError(f"not a cpu list: {text!r}")
        cpus.extend(range(int(first), int(last or first) + 1))
    return cpus


def _cpu_counters(stat: str) -> dict[int, tuple[int, int]]:
    """(idle + iowait, total) jiffies for each CPU line of /proc/stat."""
    counters: dict[int, tuple[int, int]] = {}
    for line in stat.splitlines():
        name, *fields = line.split()
        if name.startswith("cpu") and name[3:].isdigit():
            values = [int(field) for field in fields]
            counters[int(name[3:])] = (values[3] + values[4], sum(values))
    require(counters, "/proc/stat lists no CPUs")
    return counters


def busy_between(before: str, after: str) -> dict[int, float]:
    """Each CPU's busy share between two readings of /proc/stat, which shows the whole host, other tenants included."""
    first, second = _cpu_counters(before), _cpu_counters(after)
    require(first.keys() == second.keys(), "/proc/stat lists different CPUs between readings")
    busy: dict[int, float] = {}
    for cpu, (idle, total) in second.items():
        elapsed = total - first[cpu][1]
        require(elapsed > 0, f"cpu{cpu}'s counters did not advance")
        busy[cpu] = 1.0 - (idle - first[cpu][0]) / elapsed
    return busy


def plan_cores(
    workers: int,
    *,
    node_cpus: Sequence[int],
    siblings: Mapping[int, Sequence[int]],
    allowed: Collection[int],
    busy: Mapping[int, float] | None = None,
) -> CorePlan:
    """One hardware thread per physical core for each worker, quietest core first, then one more for the main process.

    A pinned worker can't move off a CPU another tenant is using, and in lockstep the slowest worker sets the pace.
    """
    require(workers >= 1, f"need at least one worker, got {workers}")
    load = busy or {}
    chosen: list[int] = []
    used: set[int] = set()
    for cpu in sorted(node_cpus):
        if cpu in allowed and cpu not in used:
            chosen.append(cpu)
            used.update(siblings[cpu])
    chosen.sort(key=lambda cpu: sum(load.get(sibling, 0.0) for sibling in siblings[cpu]))
    if len(chosen) < workers + 1:
        raise ValueError(f"{workers} workers plus the main process need {workers + 1} physical cores; {len(chosen)} are usable")
    plan = CorePlan(workers=chosen[:workers], main=chosen[workers : workers + 1])
    require(not set(plan.workers) & set(plan.main), "the main process must not share a core with a worker")
    return plan


def plan_for_this_machine(workers: int, numa_node: int) -> CorePlan:
    """Read the topology from sysfs. Linux only; elsewhere pinning can't be requested."""
    if sys.platform != "linux":
        raise ValueError("--pin-workers needs Linux")
    node_cpus = parse_cpu_list((_SYSFS / "node" / f"node{numa_node}" / "cpulist").read_text())
    siblings = {
        cpu: parse_cpu_list((_SYSFS / "cpu" / f"cpu{cpu}" / "topology" / "thread_siblings_list").read_text())
        for cpu in node_cpus
    }
    stat = Path("/proc/stat")
    before = stat.read_text()
    time.sleep(_LOAD_SAMPLE_S)
    busy = busy_between(before, stat.read_text())
    return plan_cores(workers, node_cpus=node_cpus, siblings=siblings, allowed=os.sched_getaffinity(0), busy=busy)


def pin_process(cores: Collection[int]) -> None:
    """Restrict the calling process to ``cores``."""
    require(cores, "cannot pin to an empty set of cores")
    if sys.platform != "linux":
        raise ValueError("pinning needs Linux")
    os.sched_setaffinity(0, cores)
