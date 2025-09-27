#!/bin/bash

echo "Run training script..."

datetime="$1" # $1 refers to the first command-line argument

# check if datetime is empty. If so, set it to current datetime.
if [ -z "$datetime" ]; then
    datetime=$(date +"%Y-%m-%d_%H-%M-%S")
fi


seeds=(1 2)
 
for seed in "${seeds[@]}"; do
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/mocap_AMP_for_hardware/*' agent.amp_motion_folder='datasets/mocap_AMP_for_hardware/*' --task Isaac-Velocity-AMPStairs-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_paper_mocap" --logger wandb --log_project_name paper_mocap --max_iterations 25000
done
