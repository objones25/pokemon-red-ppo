#!/bin/bash
# Scenario fidelity: one process per scenario, resync (with frame classification) and open-loop; then controls and save-state interchange.
cd /workspace/podverify; export PYTHONPATH=/workspace/podverify
R=/workspace/podverify/results; ROM="/workspace/rom/Pokemon Red.gb"; PY=/opt/venv/bin/python
SHOTS=$R/screens; mkdir -p $SHOTS
for name in wild trainer menus text cities heal dungeons; do
  $PY fidelity.py scenarios --rom "$ROM" --out $R/fid_scen_$name.jsonl --names $name --both --frames --screens $SHOTS > $R/fid_scen_$name.log 2>&1 &
done
$PY fidelity.py scenarios --rom "$ROM" --out $R/fid_control.jsonl --names wild menus cities heal --scale 0.34 --resync --control > $R/fid_control.log 2>&1 &
$PY fidelity.py interchange --rom "$ROM" --out $R/fid_interchange.jsonl > $R/fid_interchange.log 2>&1 &
wait
echo done $(date -u +%T) >> $R/fid_scen.done
