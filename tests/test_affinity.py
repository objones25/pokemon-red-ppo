"""Choosing physical cores on the GPU's NUMA node for workers."""

from __future__ import annotations

import pytest

from pokered.affinity import CorePlan, busy_between, parse_cpu_list, plan_cores
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


def test_workers_avoid_a_core_other_tenants_keep_busy():
    """Measured on a shared A100 host: CPUs 0-25 carried other load, and pinning there cost 40% against pinning elsewhere."""
    plan = plan_cores(2, node_cpus=NODE, siblings=SIBLINGS, allowed=set(NODE), busy={0: 0.9, 1: 0.0, 2: 0.0, 3: 0.0})
    assert plan.workers == [1, 2]


def test_a_core_counts_as_busy_when_its_hyperthread_sibling_is():
    plan = plan_cores(2, node_cpus=NODE, siblings=SIBLINGS, allowed=set(NODE), busy={128: 0.9})
    assert plan.workers == [1, 2]


def test_the_main_process_gets_the_next_quietest_core():
    plan = plan_cores(1, node_cpus=NODE, siblings=SIBLINGS, allowed=set(NODE), busy={0: 0.9, 1: 0.5, 2: 0.0, 3: 0.2})
    assert plan == CorePlan(workers=[2], main=[3])


def test_equally_quiet_cores_keep_their_order():
    plan = plan_cores(2, node_cpus=NODE, siblings=SIBLINGS, allowed=set(NODE), busy=dict.fromkeys(NODE, 0.0))
    assert plan.workers == [0, 1]


STAT_BEFORE = "cpu  1 2 3 4 5 6 7 8 9 10\ncpu0 10 0 0 90 0 0 0 0 0 0\ncpu1 10 0 0 90 0 0 0 0 0 0\nintr 1\n"
STAT_AFTER = "cpu  1 2 3 4 5 6 7 8 9 10\ncpu0 60 0 0 140 0 0 0 0 0 0\ncpu1 10 0 0 190 0 0 0 0 0 0\nintr 1\n"


def test_busy_share_comes_from_the_change_in_each_cpus_counters():
    assert busy_between(STAT_BEFORE, STAT_AFTER) == {0: 0.5, 1: 0.0}


def test_busy_counts_iowait_as_idle():
    before = "cpu0 0 0 0 0 0 0 0 0 0 0\n"
    after = "cpu0 0 0 0 50 50 0 0 0 0 0\n"
    assert busy_between(before, after) == {0: 0.0}


def test_busy_refuses_counters_that_did_not_advance():
    with pytest.raises(CheckFailed, match="did not advance"):
        busy_between(STAT_BEFORE, STAT_BEFORE)


def test_busy_refuses_a_reading_with_no_cpus():
    with pytest.raises(CheckFailed, match="no CPUs"):
        busy_between("intr 1\n", "intr 1\n")
