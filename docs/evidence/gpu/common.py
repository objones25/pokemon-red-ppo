"""Helpers shared by both benchmark loops. Does not import torch at module level."""
from __future__ import annotations

import json
import time

import psutil


def worker_mem(pids: list[int]) -> dict:
    rss, uss = [], []
    for pid in pids:
        p = psutil.Process(pid)
        rss.append(p.memory_info().rss)
        try:
            uss.append(p.memory_full_info().uss)
        except (psutil.AccessDenied, AttributeError):
            pass
    return {
        "rss_mb_mean": sum(rss) / len(rss) / 1e6,
        "rss_mb_each": [round(x / 1e6, 1) for x in rss],
        "uss_mb_mean": (sum(uss) / len(uss) / 1e6) if uss else None,
    }


def emit(out: str, rec: dict) -> None:
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "mem"}), flush=True)


def now() -> float:
    return time.perf_counter()
