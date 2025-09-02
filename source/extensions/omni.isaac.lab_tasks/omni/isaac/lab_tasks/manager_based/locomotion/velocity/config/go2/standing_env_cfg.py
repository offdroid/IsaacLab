# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)


from . import parameters

####################################################################
# Standing simple reward


@configclass
class UnitreeGo2StandingEnvCfgSimpleReward(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):

        # post init of parent
        super().__post_init__()

        self.terrain_type = "flat"
        
        # TODO DR: set timing for random pushes
        
        self.scene.num_envs = 4096  # with DR: 2 * 4096; without DR: 5480
        parameters.set_terrain(self)
        parameters.set_rewards_standing(self)
        parameters.set_standing_env_terminations(self)
        
        self.episode_length_s = 5.0
        
        
        # RSI        
        # rsi_params = {
        #     "reference_states": ["joints", "base"],
        # }
        # parameters.set_amp_settings(self, motion_folder = "datasets/fromVision_motions_DepthCam_standUp_feetZAmpl/stand_up_2431270000_amp.txt", **rsi_params)
        
        # self.is_amp_env = False


@configclass
class UnitreeGo2StandingEnvCfgSimpleReward_PLAY(UnitreeGo2StandingEnvCfgSimpleReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)


#######################################################################
# Standing complex reward


@configclass
class UnitreeGo2StandingEnvCfgComplexReward(UnitreeGo2StandingEnvCfgSimpleReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        
        parameters.set_rewards_standing_complex(self)


@configclass
class UnitreeGo2StandingEnvCfgComplexReward_PLAY(UnitreeGo2StandingEnvCfgComplexReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)


#######################################################################
# Standing AMP


@configclass
class AMPUnitreeGo2StandingEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terrain_type = "flat"
        parameters.set_terrain(self)
        parameters.set_rewards_standing_amp(self)
        parameters.set_standing_env_terminations(self)
        parameters.standing_domain_randomization(self)

        self.episode_length_s = 5.0 # This duration should be matched with the duration of the expert trajectory so that the style distributions can be equal. TODO reduce this to 3.0 again, and also reduce repeated frames at end of expert dataframes. See commit 6c0e6b10aa51ad72d0640e94c4c7158ce562aa06.
        self.scene.num_envs = 5480  # with DR: 2 * 4096; without DR: 5480

        self.events.push_robot.interval_range_s = (2.0, 4.0) # TODO reduce to episode length
        self.events.push_robot.params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}} # TODO increase
        self.events.reset_gravity.interval_range_s = (2.0, 4.0)
        
        rsi_params = {
            "reference_states": ["joints", "base"],
            "motion_files": [
                "datasets/fromVision_motions_DepthCam_standUp_feetZAmpl/stand_up_2431270000_amp.txt",
                "datasets/fromVision_motions_DepthCam_extendedWithoutReverse_feetZAmpl_minimal/slow_1313807000_amp.txt",
            ]
        }
        parameters.set_amp_settings(self, motion_folder = "datasets/fromVision_motions_DepthCam_standUp_feetZAmpl/*", **rsi_params)

    def update_motion_files(self):
        motion_files = glob.glob(self.amp_motion_folder)
        self.amp_motion_files = motion_files

        # assert (
        #     self.events.reference_state_initialization is not None
        # ), "Always expecting RSI. For evaluation, please use the same motion files as used for training."
        # self.events.reference_state_initialization.params["motion_files"] = motion_files


@configclass
class AMPUnitreeGo2StandingEnvCfg_PLAY(AMPUnitreeGo2StandingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)

        self.amp_motion_folder = "datasets/fromVision_motions_DepthCam_standUp_feetZAmpl/*" # required otherwise it wont start; it is recomended to use same motion files as used for training
