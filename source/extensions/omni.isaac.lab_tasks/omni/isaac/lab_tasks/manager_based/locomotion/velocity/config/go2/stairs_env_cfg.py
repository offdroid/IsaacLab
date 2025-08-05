# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp


from . import parameters
from . import residual_rl_data

#######################################################################
# Stairs simple reward

@configclass
class UnitreeGo2StairsEnvCfgSimpleReward(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        
        # post init of parent
        super().__post_init__()
        
        self.terrain_type = "stairs"
        parameters.set_terrain(self)
        parameters.set_rewards_simple(self)
        parameters.set_stairs_env_cfg_cmds(self)
        parameters.set_stairs_env_cfg_reset_base(self)
        parameters.add_relative_position_on_stairs_observation(self)
        parameters.add_stair_parameters_observation(self)
        parameters.set_curriculum(self, enable=False)


@configclass
class UnitreeGo2StairsEnvCfgSimpleReward_PLAY(UnitreeGo2StairsEnvCfgSimpleReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)
        parameters.set_play_settings_rough(self)


@configclass
class UnitreeGo2StairsEnvCfgSimpleRewardCurriculum(UnitreeGo2StairsEnvCfgSimpleReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_curriculum(self, enable=True)

@configclass
class UnitreeGo2StairsEnvCfgSimpleRewardCurriculum_PLAY(UnitreeGo2StairsEnvCfgSimpleRewardCurriculum):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)
        parameters.set_play_settings_rough(self)


#######################################################################
# Stairs complex reward

@configclass
class UnitreeGo2StairsEnvCfgComplexReward(UnitreeGo2StairsEnvCfgSimpleReward):
    def __post_init__(self):
        
        # post init of parent
        super().__post_init__()

        parameters.set_rewards_complex(self)
        parameters.set_curriculum(self, enable=False)


@configclass
class UnitreeGo2StairsEnvCfgComplexReward_PLAY(UnitreeGo2StairsEnvCfgComplexReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)
        parameters.set_play_settings_rough(self)


@configclass
class UnitreeGo2StairsEnvCfgComplexRewardCurriculum(
    UnitreeGo2StairsEnvCfgComplexReward
):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_curriculum(self, enable=True)


@configclass
class UnitreeGo2StairsEnvCfgComplexRewardCurriculum_PLAY(
    UnitreeGo2StairsEnvCfgComplexRewardCurriculum
):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)
        parameters.set_play_settings_rough(self)


#######################################################################
# Stairs AMP

@configclass
class AMPUnitreeGo2StairsEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terrain_type = "stairs"
        parameters.set_terrain(self)
        parameters.set_stairs_env_cfg_cmds(self)
        parameters.set_stairs_env_cfg_reset_base(self)
        parameters.add_relative_position_on_stairs_observation(self)
        parameters.add_stair_parameters_observation(self)
        parameters.set_velocity_rewards_amp(self)

        enable_rsi = True
        parameters.set_curriculum(self, enable=not enable_rsi)
        if not enable_rsi:
            self.events.reference_state_initialization = None

        self.scene.num_envs = 2 * 4096  # 5480

        # style
        self.action_manager_class = "ActionManager"  # Default action manager

        parameters.set_amp_settings(self)
        # update motion files
        self.amp_motion_folder = "datasets/fromVision_motions_DepthCamStairs/*"
        self.amp_motion_files = glob.glob(self.amp_motion_folder)

    def update_motion_files(self):
        import os

        motion_files = glob.glob(self.amp_motion_folder)
        self.amp_motion_files = motion_files
        rsi_motion_files = [
            f
            for f in motion_files
            if os.path.basename(f)
            in {"stairs_2_5299211000_amp.txt", "walk_869488000_amp.txt"}
        ]

        # assert (
        #     self.events.reference_state_initialization is not None
        # ), "Always expecting RSI. For evaluation, please use the same motion files as used for training."
        self.events.reference_state_initialization.params["motion_files"] = (
            rsi_motion_files
        )


@configclass
class AMPUnitreeGo2StairsEnvCfg_PLAY(AMPUnitreeGo2StairsEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)
        parameters.set_play_settings_rough(self)

        self.amp_motion_folder = "datasets/dummy/*"  # required otherwise it wont start; it is recomended to use same motion files as used for training
