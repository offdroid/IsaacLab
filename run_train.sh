#!/bin/bash

echo "Run training script..."

datetime="$1" # $1 refers to the first command-line argument

# check if datetime is empty. If so, set it to current datetime.
if [ -z "$datetime" ]; then
  datetime=$(date +"%Y-%m-%d_%H-%M-%S")
fi

seeds=(1 2 3)
for seed in "${seeds[@]}"; do
  ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-AMPShortStairs-Unitree-Go2-v0 --seed $seed --max_iterations 20000 --num_envs 4096 --log_dir "${datetime}_ampv6_roughstairs" --logger wandb --log_project_name ampv6_roughplatform --headless
  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-AMPShortStairs-Unitree-Go2-Play-v0 --seed $seed --max_iterations 2000 --num_envs 3 --log_dir debug
  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-AMPShortStairs-Unitree-Go2-Play-v0 --headless --seed $seed --log_dir "${datetime}_ampv6_roughplatform" --logger wandb --log_project_name ampv6_roughplatform --max_iterations 20000 --num_envs 4096
  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Stairs-ComplexReward-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_complex" --logger wandb --log_project_name complex_v2 --max_iterations 10000 --num_envs 4096
done
