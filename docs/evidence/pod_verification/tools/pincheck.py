"""On a running pokered-train: is each worker on its own physical core, on the GPU's NUMA node? Reads /proc and sysfs only."""
import json
import os
import subprocess
from pathlib import Path

def parse(text):
    out = []
    for part in text.strip().split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out

def allowed(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("Cpus_allowed_list"):
            return parse(line.split()[1])
    raise RuntimeError(pid)

pids = []
for entry in Path("/proc").iterdir():
    if entry.name.isdigit():
        try:
            argv = entry.joinpath("cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if len(argv) > 1 and argv[1].endswith(b"bin/pokered-train"):
            pids.append(int(entry.name))
kids = {}
for pid in pids:
    try:
        kids[pid] = [int(x) for x in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except OSError:
        kids[pid] = []
child_ids = {k for v in kids.values() for k in v}
mains = [p for p in pids if p not in child_ids]
out = {"mains": mains}
bus = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().lower()
sysfs = Path("/sys/bus/pci/devices") / bus.removeprefix("0000")
out["gpu_bus"], out["gpu_numa_sysfs"] = bus, sysfs.joinpath("numa_node").read_text().strip() if sysfs.exists() else "missing"
node = max(int(out["gpu_numa_sysfs"]), 0) if out["gpu_numa_sysfs"].lstrip("-").isdigit() else 0
node_cpus = set(parse(Path(f"/sys/devices/system/node/node{node}/cpulist").read_text()))
cpu_max = Path("/sys/fs/cgroup/cpu.max").read_text().strip() if Path("/sys/fs/cgroup/cpu.max").exists() else "v1 quota " + Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip()
out["cgroup_cpu_max"] = cpu_max
for main in mains:
    workers = []
    for pid in kids.get(main, []):
        cmd = Path(f"/proc/{pid}/cmdline").read_text().replace("\0", " ")
        cores = allowed(pid)
        if "spawn_main" not in cmd:
            continue
        worker = cores[0] if len(cores) == 1 else None
        siblings = Path(f"/sys/devices/system/cpu/cpu{cores[0]}/topology/thread_siblings_list").read_text().strip()
        package = Path(f"/sys/devices/system/cpu/cpu{cores[0]}/topology/core_id").read_text().strip()
        stat = Path(f"/proc/{pid}/stat").read_text().split()
        workers.append({"pid": pid, "cores": cores if len(cores) < 5 else f"{len(cores)} cores", "single": worker is not None, "siblings": siblings, "core_id": package, "last_ran_on": int(stat[38]), "on_gpu_node": set(cores) <= node_cpus})
    physical = [w["siblings"] for w in workers]
    main_cores = allowed(main)
    out[str(main)] = {"worker_count": len(workers), "all_single_core": all(w["single"] for w in workers), "distinct_physical": len(set(physical)) == len(physical),
                      "all_on_gpu_node": all(w["on_gpu_node"] for w in workers), "ran_where_pinned": all(w["last_ran_on"] in (w["cores"] if isinstance(w["cores"], list) else [w["last_ran_on"]]) for w in workers),
                      "main_cores": main_cores, "main_shares_with_worker": bool({str(main_cores)} & {str(w["cores"]) for w in workers}),
                      "workers": workers}
print(json.dumps(out, indent=1))
