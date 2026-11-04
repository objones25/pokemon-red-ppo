#!/bin/bash
# usage: crosseval.sh <label> <run>: the real pokered-evaluate on the newest checkpoint of a run, in accurate and fast mode.
cd /workspace/podverify; export PYTHONPATH=/workspace/podverify
R=/workspace/podverify/results; ROM="/workspace/rom/Pokemon Red.gb"; PY=/opt/venv/bin/python
label=$1; run=$2
for mode in accurate fast; do
  $PY flip_mode.py /workspace/pokered/runs/$run/checkpoints /workspace/podverify/flip/$label-$mode/ $mode 2>&1 | grep -v SDL
  /opt/venv/bin/pokered-evaluate --rom "$ROM" --checkpoints /workspace/podverify/flip/$label-$mode --output $R/eval_${label}_in_$mode.json --workers 12 --envs-per-worker 4 --seed 7 > $R/eval_${label}_in_$mode.log 2>&1
  echo "$label $mode exit $? $(date -u +%T)" >> $R/crosseval.done
done
