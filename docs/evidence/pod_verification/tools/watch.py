"""Latest metrics of offline runs: steps, steps/s, tiles, events."""
import glob, json, sys
from read_wandb import history
for root in sys.argv[1:]:
    rows = []
    for path in sorted(glob.glob(root + "/wandb/offline-run-*/run-*.wandb")):
        rows += [r for r in history(path)]
    if not rows:
        print(root, "no rows"); continue
    last = rows[-12:]
    episodes = [r for r in last if "episode/tiles_mean" in r]
    def mean(k):
        v = [r[k] for r in episodes if k in r]
        return round(sum(v) / len(v), 1) if v else None
    print(root.split("/")[-1], "steps", rows[-1]["_step"], "sps", round(sum(r["perf/steps_per_second"] for r in last) / len(last)), "tiles", mean("episode/tiles_mean"), "events", mean("episode/events_mean"), "tiles_max", max([r.get("episode/tiles_max", 0) for r in episodes], default=None), "ev_max", max([r.get("episode/events_max", 0) for r in episodes], default=None), "viridian", mean("episode/milestone_viridian_city_mean"), "forest", mean("episode/milestone_viridian_forest_mean"), "pewter", mean("episode/milestone_pewter_city_mean"))
