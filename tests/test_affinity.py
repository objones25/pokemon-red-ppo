"""Choosing physical cores on the GPU's NUMA node for workers."""

from __future__ import annotations

import pytest

from pokered.affinity import CorePlan, parse_cpu_list, plan_cores
from pokered.negative_space import CheckFailed


@pytest.mark.parametrize(("text", "expected"), [("0-3", [0, 1, 2, 3]), ("0-1,128-129", [0, 1, 128, 129]), ("5", [5]), ("0-1\n", [0, 1])])
def test_parse_cpu_list(text, expected):
    assert parse_cpu_list(text) == expected


def test_parse_cpu_list_refuses_garbage():
    with pytest.raises(ValueError, match="cpu list"):
        parse_cpu_list("zero")


# Node 0 of the benchmarked EPYC: cores 0-3, with hyperthread siblings 128-131.
NODE = [0, 1, 2, 3, 128, 129, 130, 131]
SIBLINGS = {0: [0, 128], 1: [1, 129], 2: [2, 130], 3: [3, 131], 128: [0, 128], 129: [1, 129], 130: [2, 130], 131: [3, 131]}


def test_workers_get_one_thread_per_physical_core():
    assert plan_cores(2, node_cpus=NODE, siblings=SIBLINGS, allowed=set(NODE)).workers == [0, 1]


def test_main_process_gets_the_next_physical_core():
    assert plan_cores(2, node_cpus=NODE, siblings=SIBLINGS, allowed=set(NODE)).main == [2]


def test_workers_skip_cpus_the_process_may_not_use():
    allowed = {1, 2, 3, 129, 130, 131}
    assert plan_cores(2, node_cpus=NODE, siblings=SIBLINGS, allowed=allowed).workers == [1, 2]


def test_too_many_workers_is_an_operating_error():
    with pytest.raises(ValueError, match="need 5 physical cores; 4 are usable"):
        plan_cores(4, node_cpus=NODE, siblings=SIBLINGS, allowed=set(NODE))


def test_plan_never_shares_a_physical_core():
    plan = plan_cores(3, node_cpus=NODE, siblings=SIBLINGS, allowed=set(NODE))
    assert plan == CorePlan(workers=[0, 1, 2], main=[3])


def test_plan_refuses_zero_workers():
    with pytest.raises(CheckFailed, match="at least one worker"):
        plan_cores(0, node_cpus=NODE, siblings=SIBLINGS, allowed=set(NODE))


def test_plan_accepts_one_worker():
    assert plan_cores(1, node_cpus=NODE, siblings=SIBLINGS, allowed=set(NODE)).workers == [0]
