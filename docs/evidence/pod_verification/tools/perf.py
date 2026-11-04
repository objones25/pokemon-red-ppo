"""Summarize perf/* from offline runs: drop the first update (includes start-up), report the rest."""
import json
import sys
from pathlib import Path
import numpy as np
from read_wandb import history
import glob

def summarize(root):
    rows = []
    for path in sorted(glob.glob(root + "/wandb/offline-run-*/run-*.wandb")):
        rows += [r for r in history(path) if "perf/steps_per_second" in r]
    used = rows[1:]
    if not used:
        return {"root": root, "updates": len(rows)}
    sps = np.array([r["perf/steps_per_second"] for r in used])
    collect = np.array([r["perf/collect_seconds"] for r in used])
    update = np.array([r["perf/update_seconds"] for r in used])
    steps = np.array([r["_step"] for r in rows])
    batch = int(steps[1] - steps[0]) if len(steps) > 1 else None
    return {"root": root, "updates_total": len(rows), "updates_used": len(used), "batch": batch,
            "steps_per_second_mean": float(sps.mean()), "sd": float(sps.std()), "min": float(sps.min()), "max": float(sps.max()),
            "harmonic": float(len(sps) / (1 / sps).sum()), "collect_s_mean": float(collect.mean()), "update_s_mean": float(update.mean()),
            "update_share": float(update.sum() / (update.sum() + collect.sum()))}

for root in sys.argv[1:]:
    print(json.dumps(summarize(root)))
