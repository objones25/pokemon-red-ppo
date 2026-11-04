"""After identical inputs, which bytes of a save state differ between modes? The mismatch is located, not assumed."""
import io, sys
from pathlib import Path
import numpy as np
from lockstep import Pair, save
from pokered.start_state import START_STATE
p = Pair(Path(sys.argv[1]), START_STATE); g = np.random.default_rng(0)
for s in range(int(sys.argv[2])):
    p.step(int(g.integers(0, 7)))
    if s % 100 == 99:
        a, b = np.frombuffer(save(p.a.pyboy), np.uint8), np.frombuffer(save(p.b.pyboy), np.uint8)
        where = np.nonzero(a != b)[0]
        ly = (p.a.pyboy.memory[0xFF44], p.b.pyboy.memory[0xFF44]); stat = (p.a.pyboy.memory[0xFF41], p.b.pyboy.memory[0xFF41])
        print(s, "bytes", len(a), "differ", len(where), "first", where[:12].tolist(), "spans", (where.min(), where.max()) if len(where) else None, "LY", ly, "STAT", stat, "DIV", p.a.pyboy.memory[0xFF04], p.b.pyboy.memory[0xFF04])
