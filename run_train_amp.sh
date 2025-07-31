#!/bin/bash

echo "Run training script..."

datetime="$1" # $1 refers to the first command-line argument

# check if datetime is empty. If so, set it to current datetime.
if [ -z "$datetime" ]; then
  datetime=$(date +"%Y-%m-%d_%H-%M-%S")
fi

seeds1=(1)
for seed in "${seeds1[@]}"; do
  ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Stairs-SimpleReward-Curriculum-Unitree-Go2-v0 --log_dir "${datetime}-paper_simple_curr" --logger wandb --log_project_name paper_simple_curr --num_envs 4096 --seed $seed --headless
done

seeds=(1 2 3)
for seed in "${seeds[@]}"; do
  ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/fromVision_motions_DepthCamStairs/*' agent.amp_motion_folder='datasets/fromVision_motions_DepthCamStairs/*' --task Isaac-Velocity-AMPStairs-Unitree-Go2-v0 --headless --num_envs 4096 --seed $seed --log_dir "${datetime}_stairs_first_try_curr" --logger wandb --log_project_name paper_amp
done
