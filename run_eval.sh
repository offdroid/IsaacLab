#!/bin/bash

echo "Run eval script..."

datetime="$1" # $1 refers to the first command-line argument

seeds=(1 2 3)

for seed in "${seeds[@]}"; do
    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
    # --load_run ${datetime}_manuallyGenerated_SEED_${seed} \
    # --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
    # --load_run ${datetime}_mocap_AMP_for_hardware_SEED_${seed} \
    # --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
    # --load_run ${datetime}_fromVision_motions_DepthCam_SEED_${seed} \
    # --evaluate
    
    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
    # --load_run ${datetime}_fromVision_motions_AlignedDepthAnything_SEED_${seed} \
    # --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-Flat-SimpleReward-Unitree-Go2-Play-v0 \
    # --load_run ${datetime}_simpleReward_SEED_${seed} \
    # --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-Flat-ComplexReward-Unitree-Go2-Play-v0 \
    # --load_run ${datetime}_complexReward_SEED_${seed} \
    # --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
    # --load_run ${datetime}_fromVision_motions_DepthCam_extended_SEED_${seed} \
    # --evaluate

    # USE
    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
    # --load_run ${datetime}_fromVision_motions_DepthCam_extendedWithoutReverse_feetZAmpl_DR2_SEED_${seed} \
    # --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-AMPNoisyFlat-Unitree-Go2-Play-v0 \
    # --load_run ${datetime}_fromVision_motions_DepthCam_extendedWithoutReverse_feetZAmpl_DR2_SEED_${seed} \
    # --evaluate


    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Velocity-Box-ComplexReward-Unitree-Go2-Play-v0 --headless --load_run 2025-07-23_15-13-01_ComplexRew_Curr_SEED_3 --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Velocity-Box-SimpleReward-Unitree-Go2-Play-v0 --headless --load_run 2025-07-23_15-13-01_SimpleRew_Curr_SEED_2 --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Velocity-Standing-ComplexReward-Unitree-Go2-Play-v0 --headless --load_run 2025-08-01_14-22-59_ComplexRew_SEED_${seed} --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Velocity-Standing-SimpleReward-Unitree-Go2-Play-v0 --headless --load_run 2025-07-25_13-00-01_SimpleRew_SEED_${seed} --evaluate


    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Velocity-AMPStanding-Unitree-Go2-Play-v0 --headless --load_run 2025-07-26_17-52-05_RSI_SEED_${seed} --evaluate


done