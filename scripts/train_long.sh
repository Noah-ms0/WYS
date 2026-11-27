#!/bin/bash
# usage: train_long.sh NUM_ENVS ITERS [hydra overrides...]
cd $(dirname "$(readlink -f "$0")")/../src
OUT=../runs/$(date +%m%d_%H%M%S)_$1x24_$2; mkdir -p $OUT
echo "OUT=$(realpath $OUT)"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm --format=csv,noheader,nounits -l 30 > $OUT/gpu.csv &
MON=$!
trap 'kill $MON' EXIT
OMNI_KIT_ACCEPT_EULA=YES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GIT_PYTHON_REFRESH=quiet PYTHONUNBUFFERED=1 timeout 24h ../.venv/bin/python scripts/train.py --headless --seed 1 --num_envs $1 --max_iterations $2 --run_name long$1_seed1 --logger tensorboard "${@:3}" > $OUT/train.log 2>&1
echo "python rc=$?"
grep "Learning iteration" $OUT/train.log | tail -1
grep -nE "Traceback|OutOfMemory" $OUT/train.log | head
