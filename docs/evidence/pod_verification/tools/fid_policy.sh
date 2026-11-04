#!/bin/bash
# usage: fid_policy.sh <label> <checkpoints> <procs> <steps> [cpu range]
# Trained policy samples on accurate; the same actions replay on fast. Resync is on so every divergence is counted.
cd /workspace/podverify; export PYTHONPATH=/workspace/podverify OMP_NUM_THREADS=1
R=/workspace/podverify/results; ROM="/workspace/rom/Pokemon Red.gb"; PY=/opt/venv/bin/python
label=$1; ck=$2; procs=$3; steps=$4
for i in $(seq 0 $((procs - 1))); do
  extra=""; [ "$i" -lt 3 ] && extra="--frames"
  $PY fidelity.py policy --rom "$ROM" --checkpoints "$ck" --out $R/fid_policy_$label.jsonl --seed $((1000 + i)) --steps "$steps" --resync $extra > $R/fid_policy_${label}_$i.log 2>&1 &
done
$PY fidelity.py policy --rom "$ROM" --checkpoints "$ck" --out $R/fid_policy_${label}_control.jsonl --seed 1000 --steps $((steps / 3)) --resync --control > $R/fid_policy_${label}_control.log 2>&1 &
wait
echo "done $label $(date -u +%T)" >> $R/fid_policy.done
