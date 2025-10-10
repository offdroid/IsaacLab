#!/bin/bash

echo "Run training script..."

datetime="$1" # $1 refers to the first command-line argument

# check if datetime is empty. If so, set it to current datetime.
if [ -z "$datetime" ]; then
  datetime=$(date +"%Y-%m-%d_%H-%M-%S")
fi

seeds=(1 2 3)
seeds=(2)

for seed in "${seeds[@]}"; do
  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-AMPShortStairs-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_ampDRv2_shortstairs_test" --logger wandb --log_project_name amp_dr --max_iterations 5000 --num_envs 4096
  ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-AMPStairs-Alignment-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_alignment_fixed30cm" --logger wandb --log_project_name alignment --max_iterations 10000 --num_envs 4096
done
