#!/bin/bash
# Part 3a/3e: same seed and steps, accurate vs fast, concurrently on disjoint quiet CPU ranges. Settings as the earlier 3M-step Mac run.
cd /workspace/pokemon-red-ppo
R=/workspace/podverify/results
STEPS=${1:-5000000}
common=(--wandb-mode offline --no-terminate-pod-on-exit --workers 11 --envs-per-worker 8 --ppo.rollout-steps 256 --ppo.minibatch-size 2048 --env.base-budget 2048 --env.budget-per-event 512 --seed 1 --total-steps "$STEPS" --checkpoint-every-steps 500000 --keep-checkpoints 40)
WANDB_SILENT=true POKERED_RUN_NAME=learn_acc_s1 taskset -c 100-135 scripts/train.sh "${common[@]}" > $R/learn_acc_s1.log 2>&1 &
sleep 8
WANDB_SILENT=true POKERED_RUN_NAME=learn_fast_s1 taskset -c 136-171 scripts/train.sh "${common[@]}" --env.fast-mode > $R/learn_fast_s1.log 2>&1 &
wait
echo "learn_s1.done $(date -u +%T)" >> $R/learn.done
