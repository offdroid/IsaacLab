#!/bin/bash

echo "Run training script..."

datetime="$1" # $1 refers to the first command-line argument

# check if datetime is empty. If so, set it to current datetime.
if [ -z "$datetime" ]; then
  datetime=$(date +"%Y-%m-%d_%H-%M-%S")
fi

seeds=(1 2 3)
seeds=(1)

for seed in "${seeds[@]}"; do
  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/manuallyGenerated/*' agent.amp_motion_folder='datasets/manuallyGenerated/*' --task Isaac-Velocity-AMPFlat-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_manuallyGenerated"

  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/fromVision_motions_DepthCam_extended/*' agent.amp_motion_folder='datasets/fromVision_motions_DepthCam_extended/*' --task Isaac-Velocity-AMPFlat-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_fromVision_motions_DepthCam_extended_DR2"

  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-AMPStairs-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_stairs_0torqpen_newdataset_curr0-20_adjusteddataset_stairnoise_noiseparams_flexstepwidth_newobs_furtherback_-0.006doftorql2after100000" --logger wandb --log_project_name amp_dr --max_iterations 20000 --num_envs 4096
  ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-AMPStairs-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_stairs_curr0-20_dsslow_dist2stairsv1_stumble-5_slide-5_standstill-1_norelativeposobs_onlystairs_consttorqpen" --logger wandb --log_project_name amp_dr --max_iterations 20000 --num_envs 4096

  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/fromVision_motions_DepthCam_extendedWithoutReverse_feetZAmpl/*' agent.amp_motion_folder='datasets/fromVision_motions_DepthCam_extendedWithoutReverse_feetZAmpl/*' --task Isaac-Velocity-AMPNoisyFlat-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_fromVision_motions_DepthCam_extendedWithoutReverse_feetZAmpl_DR2"

  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/fromVision_motions_DepthCam_extendedWithoutReverse/*' agent.amp_motion_folder='datasets/fromVision_motions_DepthCam_extendedWithoutReverse/*' --task Isaac-Velocity-AMPFlat-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_fromVision_motions_DepthCam_extendedWithoutReverse"

  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/from_AnimalAvatar/*' agent.amp_motion_folder='datasets/from_AnimalAvatar/*' --task Isaac-Velocity-AMPFlat-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_from_AnimalAvatar"

  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/fromVision_motions_AlignedDepthAnything/*' agent.amp_motion_folder='datasets/fromVision_motions_AlignedDepthAnything/*' --task Isaac-Velocity-AMPFlat-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_fromVision_motions_AlignedDepthAnything"

  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/mocap_AMP_for_hardware/*' agent.amp_motion_folder='datasets/mocap_AMP_for_hardware/*' --task Isaac-Velocity-AMPFlat-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_mocap_AMP_for_hardware"

  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Flat-SimpleReward-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_simpleReward_DR"

  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Flat-ComplexReward-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_complexReward_DR5"

  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Box-ComplexReward-Unitree-Go2-v0 --headless --seed 3 --log_dir "2025-07-23_15-13-01_ComplexRew_Curr"

  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Box-SimpleReward-Unitree-Go2-v0 --headless --seed 2 --log_dir "2025-07-23_15-13-01_SimpleRew_Curr"

  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Standing-SimpleReward-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_SimpleRew"

  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-AMPStanding-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_RSI"

  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-AMPBox-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_MoCapAMP_reduced_data"

  # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Standing-ComplexReward-Unitree-Go2-v0 --headless --seed $seed --log_dir "${datetime}_ComplexRew"
done

# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Flat-ComplexReward-Unitree-Go2-v0 --headless --seed 1 --log_dir "testtest"bash
