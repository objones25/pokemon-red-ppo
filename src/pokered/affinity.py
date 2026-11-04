"""Pinning workers to physical cores on the GPU's NUMA node.

On the benchmarked EPYC pod this raised throughput by 56% at 8 envs per worker, restoring the free-running emulator
speed (docs/CONFLICT_RESOLUTIONS.md). Unpinned workers share cores and cross NUMA nodes.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pokered.negative_space import require

_SYSFS: Final[Path] = Path("/sys/devices/system")


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


def plan_cores(
    workers: int,
    *,
    node_cpus: Sequence[int],
    siblings: Mapping[int, Sequence[int]],
    allowed: Collection[int],
) -> CorePlan:
    """One hardware thread per physical core for each worker, then one more core for the main process."""
    require(workers >= 1, f"need at least one worker, got {workers}")
    chosen: list[int] = []
    used: set[int] = set()
    for cpu in sorted(node_cpus):
        if cpu in allowed and cpu not in used:
            chosen.append(cpu)
            used.update(siblings[cpu])
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
    return plan_cores(workers, node_cpus=node_cpus, siblings=siblings, allowed=os.sched_getaffinity(0))


def pin_process(cores: Collection[int]) -> None:
    """Restrict the calling process to ``cores``."""
    require(cores, "cannot pin to an empty set of cores")
    if sys.platform != "linux":
        raise ValueError("pinning needs Linux")
    os.sched_setaffinity(0, cores)
