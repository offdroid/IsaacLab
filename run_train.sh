#!/bin/bash

echo "Run training script..."

datetime="$1" # $1 refers to the first command-line argument

# check if datetime is empty. If so, set it to current datetime.
if [ -z "$datetime" ]; then
  datetime=$(date +"%Y-%m-%d_%H-%M-%S")
fi

seeds=(2)
for seed in "${seeds[@]}"; do
  ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-AMPShortStairs-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_amp_DRv4_stairs2slowfast_slow" --logger wandb --log_project_name amp_4dr --max_iterations 20000 --num_envs 4096
done
seeds=(3)
for seed in "${seeds[@]}"; do
  ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-AMPShortStairs-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_amp_DRv4_stairs2fast_slow" --logger wandb --log_project_name amp_4dr --max_iterations 20000 --num_envs 4096
done
