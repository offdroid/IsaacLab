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
# Flat simple reward


@configclass
class UnitreeGo2FlatEnvCfgSimpleReward(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):

        # post init of parent
        super().__post_init__()

        self.terrain_type = "flat"
        parameters.set_terrain(self)
        parameters.set_rewards_simple(self)


@configclass
class UnitreeGo2FlatEnvCfgSimpleReward_PLAY(UnitreeGo2FlatEnvCfgSimpleReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)


#######################################################################
# Flat complex reward


@configclass
class UnitreeGo2FlatEnvCfgComplexReward(UnitreeGo2FlatEnvCfgSimpleReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_rewards_complex(self)


@configclass
class UnitreeGo2FlatEnvCfgComplexReward_PLAY(UnitreeGo2FlatEnvCfgComplexReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)


#######################################################################
# Flat AMP


@configclass
class AMPUnitreeGo2FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terrain_type = "flat"
        parameters.set_terrain(self)

        parameters.set_velocity_rewards_amp(self)
        
        self.scene.num_envs = 2 * 4096  # with DR: 2 * 4096; without DR: 5480

        # style
        self.action_manager_class = "ActionManager"  # Default action manager
        
        
        parameters.set_amp_settings(self, motion_folder = "datasets/fromVision_motions_DepthCam_extendedWithoutReverse_feetZAmpl_minimal_feet_forward/*")


    def update_motion_files(self):
        motion_files = glob.glob(self.amp_motion_folder)
        self.amp_motion_files = motion_files

        assert (
            self.events.reference_state_initialization is not None
        ), "Always expecting RSI. For evaluation, please use the same motion files as used for training."
        self.events.reference_state_initialization.params["motion_files"] = motion_files


@configclass
class AMPUnitreeGo2FlatEnvCfg_PLAY(AMPUnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)

        self.amp_motion_folder = "datasets/fromVision_motions_DepthCam_extendedWithoutReverse_feetZAmpl_minimal_feet_forward/*"  # required otherwise it wont start; it is recomended to use same motion files as used for training
