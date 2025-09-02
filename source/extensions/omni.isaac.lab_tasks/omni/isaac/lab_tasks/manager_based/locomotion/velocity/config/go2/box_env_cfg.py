# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import glob

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp

from . import parameters
# from . import residual_rl_data

#######################################################################
# Box simple reward

@configclass
class UnitreeGo2BoxEnvCfgSimpleReward(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        
        # post init of parent
        super().__post_init__()
        
        self.terrain_type = "box"
        
        # TODO DR: set timing for random pushes
        
        
        parameters.disable_domain_randomization(self)
        parameters.set_terrain(self)
        parameters.set_box_env_cfg_reset_base(self)
        parameters.add_relative_position_to_box_observation(self)
        parameters.add_box_parameters_observation(self)
        parameters.set_box_env_cfg_cmds(self) # calling this last is the savest way
        parameters.set_rewards_simple(self)
        
        self.scene.num_envs = 4096 # 4096  # 5480
        self.episode_length_s = 10.0

        # RSI        
        # rsi_params = {
        #     "reference_states": ["joints", "base"],
        # }
        # parameters.set_amp_settings(self, motion_folder = "datasets/fromVision_motions_DepthCam_obstacle", **rsi_params)
        
        # self.is_amp_env = False
        
        
        # ResidualRL
        # self.action_manager_class = "ResidualRLActionManager"

        # self.residual_rl_data = residual_rl_data.data[
        #     "datasets/fromVision_motions_DepthCam_obstacle/obstacle_2_3126098000_amp.txt"
        # ]
        # self.actions.joint_pos.scale = 0.15

        # self.observations.policy.phases = ObsTerm(func=mdp.phases, noise=None)

        # self.scene.robot.actuators["base_legs"].stiffness = 50

        # # we require the joint pos action offsets to be activated for ResRL
        # assert (
        #     self.actions.joint_pos.use_default_offset == True
        # ), "Require default offset to be activated for residual RL."
        
        


@configclass
class UnitreeGo2BoxEnvCfgSimpleReward_PLAY(UnitreeGo2BoxEnvCfgSimpleReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)
        # self.events.reference_state_initialization = None
        
#######################################################################
# Box complex reward

@configclass
class UnitreeGo2BoxEnvCfgComplexReward(UnitreeGo2BoxEnvCfgSimpleReward):
    def __post_init__(self):
        
        # post init of parent
        super().__post_init__()

        parameters.set_rewards_complex(self)
        # self.rewards.dof_torques_l2.weight = -0.0001 # use this to make it a bit worse
        
        
        # RSI
        # rsi_params = {
        #     "reference_states": ["joints", "base"],
        # }
        # parameters.set_amp_settings(self, motion_folder = "datasets/fromVision_motions_DepthCam_obstacle/*", **rsi_params)
        

@configclass
class UnitreeGo2BoxEnvCfgComplexReward_PLAY(UnitreeGo2BoxEnvCfgComplexReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        parameters.set_play_settings_flat(self)
        # self.events.reference_state_initialization = None

#######################################################################
# Box AMP

@configclass
class AMPUnitreeGo2BoxEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        
        # post init of parent
        super().__post_init__()
        
        self.terrain_type = "box"
        
        parameters.set_terrain(self)
        parameters.set_box_env_cfg_reset_base(self)
        parameters.add_relative_position_to_box_observation(self)
        parameters.add_box_parameters_observation(self)
        parameters.set_box_env_cfg_cmds(self) # calling this last is the savest way
        
        parameters.set_box_rewards_amp(self)

        self.scene.num_envs = 5480  # w/o DR: 5480; w/ DR: 2 * 4096
        self.episode_length_s = 3.5
        self.curriculum.terrain_levels.params = {"custom_required_distance_for_move_up": 1.3}
        
        
        self.events.push_robot.interval_range_s = (0.0, 3.0)
        self.events.push_robot.params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}}
        self.events.reset_gravity.interval_range_s = (2.0, 4.0)

        # RSI
        rsi_params = {"use_rsi": False}
        parameters.set_amp_settings(self, motion_folder = "datasets/fromVision_motions_DepthCam_box/*", **rsi_params)

    def update_motion_files(self):
        motion_files = glob.glob(self.amp_motion_folder)
        self.amp_motion_files = motion_files





@configclass
class AMPUnitreeGo2BoxEnvCfg_PLAY(AMPUnitreeGo2BoxEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)
        # parameters.set_play_settings_rough(self)

        self.amp_motion_folder = "datasets/fromVision_motions_DepthCam_box/*"  # required otherwise it wont start; it is recomended to use same motion files as used for training
        
        # self.events.reference_state_initialization = None
        
        