"""What are the screen differences that match no neighbouring accurate frame? Log their pixel counts and location against each accurate frame near the decision's end."""
import json, sys
from pathlib import Path
import numpy as np, torch
from fidelity import Frames
from lockstep import Pair, save, write_png
from pokered.checkpoints import CheckpointStore
from pokered.buttons import FRAMES_PER_ACTION
from pokered.model import ActorCritic
from pokered.start_state import START_STATE
rom, ck, out, seed, steps = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
out.mkdir(parents=True, exist_ok=True)
payload = CheckpointStore(ck).load_latest(); m = ActorCritic(); m.load_state_dict(payload["model"]); m.eval(); torch.set_num_threads(1)
g = torch.Generator().manual_seed(seed); pair = Pair(rom, START_STATE); fr = Frames(rom); a, b = pair.emus
records = []; shots = 0; last = FRAMES_PER_ACTION - 1
for s in range(steps):
    with torch.inference_mode():
        logits, _ = m({k: torch.from_numpy(v[None]) for k, v in pair.obs_a.items()})
    action = int(torch.multinomial(logits.softmax(1)[0], 1, generator=g)); prev = (save(a), save(b))
    pair.step(action)
    sa, sb = a.screen.ndarray[:, :, 0].copy(), b.screen.ndarray[:, :, 0].copy()
    if (sa != sb).any():
        fa, fb = fr.run(fr.a, prev[0], action), fr.run(fr.b, prev[1], action)
        counts = {off: int((fb[last] != fa[last + off]).sum()) for off in (-2, -1, 0, 1, 2)}
        if counts[1] and counts[2] and counts[0]:
            rows = np.nonzero((fb[last] != fa[last + 1]).any(axis=1))[0]; cols = np.nonzero((fb[last] != fa[last + 1]).any(axis=0))[0]
            rec = {"step": s, "action": action, "counts_by_offset": counts, "rows": [int(rows.min()), int(rows.max())], "cols": [int(cols.min()), int(cols.max())],
                   "battle": int(a.memory[0xD057]), "map": int(a.memory[0xD35E]), "acc_changes_each_frame": [int((fa[i] != fa[i + 1]).sum()) for i in range(last - 3, last + 1)]}
            records.append(rec)
            if shots < 6:
                write_png(out / f"neither_{shots}_acc.png", fa[last]); write_png(out / f"neither_{shots}_acc_next.png", fa[last + 1]); write_png(out / f"neither_{shots}_fast.png", fb[last]); shots += 1
json.dump(records, open(out / f"neither_{seed}.json", "w"))
print(len(records), "of", steps)
for r in records[:25]:
    print(r)
