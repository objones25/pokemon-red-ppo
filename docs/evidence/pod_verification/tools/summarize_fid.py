"""One table from the fidelity JSONL files."""
import glob, json, sys
rows = []
for path in sorted(sum((glob.glob(p) for p in sys.argv[1:]), [])):
    rows += [json.loads(l) for l in open(path)]
total = {}
for r in rows:
    s = r["screen"]
    print(f'{r["name"]:18} {"ctrl" if r.get("control") else "fast"} {"sync" if r["resync"] else "open"} steps {r["steps"]:6} battle {r["battle_steps"]:5} win {r["window_steps"]:5} maps {len(r["maps_visited"]):3} chg {r["map_changes"]:3} | '
          f'div {r["divergence_steps"]} nonrng {r["non_rng_divergence_steps"]} first {({k: v.get("step") for k, v in r["first"].items()})} | place {r["place_diff_steps"]} ev {r["event_diff_steps"]} oam {r["oam_diff_steps"]} vram {r["vram_diff_steps"]} | '
          f'scr {s["diff_steps"]} ({100*s["share"]:.1f}%) med {s["pixels_median"]:.0f} p95 {s["pixels_p95"]:.0f} max {s["pixels_max"]} >half {s["steps_over_half_frame"]} sprites {s["mean_share_inside_sprites"]:.2f} {s["classes"]} | final eq {r["final_wram_equal"]} {r["final_hram_equal"]} {r["final_place"]} tiles {r["tiles"]} ev {r["events"]} rew {r["reward"][0]:.3f}/{r["reward"][1]:.3f}')
    if r.get("policy", {}).get("compared_steps"):
        print("   policy", r["policy"])
    for k in ("heals", "started", "flights", "entered"):
        if r.get(k) is not None:
            print("  ", k, r[k])
    if r["top_diff_addresses"]:
        print("   top", r["top_diff_addresses"][:10])
        if r["first_divergences"]:
            print("   firstdiv", json.dumps(r["first_divergences"][0])[:400])
