#!/bin/bash
# Part 2 and 3b: real pokered-train throughput through train.sh, one config at a time, with GPU samples.
cd /workspace/pokemon-red-ppo
R=/workspace/podverify/results
run() { # name k T updates extra...
  local name=$1 k=$2 t=$3 updates=$4; shift 4
  local batch=$((25 * k * t))
  echo "START $name $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $R/matrix.log
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -l 2 > $R/gpu_$name.csv 2>/dev/null &
  local gm=$!
  WANDB_SILENT=true POKERED_RUN_NAME=$name scripts/train.sh --wandb-mode offline --no-terminate-pod-on-exit --envs-per-worker "$k" --ppo.rollout-steps "$t" --total-steps $((batch * updates)) "$@" > $R/log_$name.txt 2>&1
  echo "END $name exit $? $(date -u +%T)" >> $R/matrix.log
  kill $gm 2>/dev/null
}
run k8T512 8 512 6
run k32T64 32 64 8
run k8T128 8 128 16
run k8T512fast 8 512 6 --env.fast-mode
run k8T128fast 8 128 16 --env.fast-mode
run k32T64fast 32 64 8 --env.fast-mode
run k8T512th1 8 512 4 --torch-threads 1
run k8T2048 8 2048 4
