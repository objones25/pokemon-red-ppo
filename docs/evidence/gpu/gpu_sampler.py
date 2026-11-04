"""Sample AGXAccelerator PerformanceStatistics via ioreg (no sudo) to a JSONL file until killed."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

PAT = {k: re.compile(rf'"{k} Utilization %"=(\d+)') for k in ("Device", "Renderer", "Tiler")}


def read() -> dict[str, int]:
    s = subprocess.run(
        ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"], capture_output=True, text=True, timeout=5
    ).stdout
    return {k: int(m.group(1)) for k, p in PAT.items() if (m := p.search(s))}


def main() -> None:
    out, period = sys.argv[1], float(sys.argv[2])
    with open(out, "a") as f:
        while True:
            r = read()
            r["t"] = time.time()
            r["load1"] = os.getloadavg()[0]
            f.write(json.dumps(r) + "\n")
            f.flush()
            time.sleep(period)


if __name__ == "__main__":
    main()
