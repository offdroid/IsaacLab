#!/bin/bash

echo "Run training script..."

datetime="$1" # $1 refers to the first command-line argument

# check if datetime is empty. If so, set it to current datetime.
if [ -z "$datetime" ]; then
  datetime=$(date +"%Y-%m-%d_%H-%M-%S")
fi

seeds=(1)
for seed in "${seeds[@]}"; do
  ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-AMPShortStairs-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_amp_DRv5_slow+stairs2+stairs2fast_wildcard1_long_sparse1000_schedule" --logger wandb --log_project_name amp_4dr --max_iterations 40000 --num_envs 4096
done
