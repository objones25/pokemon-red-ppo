"""Totals over the fidelity files. Open-loop scenario repeats are counted apart, so no step is counted twice."""
import glob, json, sys
import numpy as np
def load(pattern):
    return [json.loads(l) for p in sorted(glob.glob(pattern)) for l in open(p)]
groups = {"scenarios": [r for r in load(sys.argv[1] + "/fid_scen_*.jsonl") if r["resync"]],
          "scenarios_open_loop": [r for r in load(sys.argv[1] + "/fid_scen_*.jsonl") if not r["resync"]],
          "policy_fast_trained": load(sys.argv[1] + "/fid_policy_fasttrained.jsonl"),
          "policy_accurate_trained": load(sys.argv[1] + "/fid_policy_acctrained.jsonl"),
          "controls": load(sys.argv[1] + "/fid_control.jsonl") + load(sys.argv[1] + "/fid_policy_*_control.jsonl")}
for name, rows in groups.items():
    steps = sum(r["steps"] for r in rows)
    out = {"runs": len(rows), "steps": steps, "battle_steps": sum(r["battle_steps"] for r in rows), "window_steps": sum(r["window_steps"] for r in rows), "map_changes": sum(r["map_changes"] for r in rows),
           "maps": len({m for r in rows for m in r["maps_visited"]}), "wram_hram_divergence_steps": sum(len(r["first"]) > 0 for r in rows), "any_first": [r["first"] for r in rows if r["first"]][:3],
           "place_diff": sum(r["place_diff_steps"] for r in rows), "event_diff": sum(r["event_diff_steps"] for r in rows), "vram_diff": sum(r["vram_diff_steps"] for r in rows), "oam_diff": sum(r["oam_diff_steps"] for r in rows),
           "reward_equal": all(abs(r["reward"][0] - r["reward"][1]) < 1e-9 for r in rows), "tiles_equal": all(r["tiles"][0] == r["tiles"][1] for r in rows), "events_equal": all(r["events"][0] == r["events"][1] for r in rows),
           "final_equal": all(r["final_wram_equal"] and r["final_hram_equal"] for r in rows),
           "screen_diff_steps": sum(r["screen"]["diff_steps"] for r in rows), "screen_share": sum(r["screen"]["diff_steps"] for r in rows) / max(steps, 1), "over_half_frame": sum(r["screen"]["steps_over_half_frame"] for r in rows)}
    classes = {}
    for r in rows:
        for k, v in r["screen"]["classes"].items():
            classes[k] = classes.get(k, 0) + v
    out["classes"] = classes
    kinds, pix = {}, {}
    for r in rows:
        for k, v in r["step_kinds"].items():
            kinds[k] = kinds.get(k, 0) + v
        for k, v in r["screen"]["by_kind"].items():
            pix[k] = pix.get(k, 0) + v
    out["screen_share_by_kind"] = {k: round(pix.get(k, 0) / v, 4) for k, v in kinds.items()}
    if any(r["policy"]["compared_steps"] for r in rows):
        n = np.array([r["policy"]["compared_steps"] for r in rows])
        out["policy"] = {"compared_steps": int(n.sum()), "share_of_steps": float(n.sum() / steps), "tv_mean": float((n * np.array([r["policy"]["tv_mean"] for r in rows])).sum() / n.sum()),
                         "tv_p95_worst_run": max(r["policy"]["tv_p95"] for r in rows), "tv_max": max(r["policy"]["tv_max"] for r in rows), "argmax_flips": sum(r["policy"]["argmax_flips"] for r in rows),
                         "value_gap_mean": float((n * np.array([r["policy"]["value_gap_mean"] for r in rows])).sum() / n.sum())}
    print(name, json.dumps(out, indent=None))
