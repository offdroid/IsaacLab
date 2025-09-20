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
from omni.isaac.lab.managers import CurriculumTermCfg
from omni.isaac.lab.managers import ManagerTermBase
from omni.isaac.lab.envs import ManagerBasedRLEnv
from typing import Sequence


from . import parameters

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


@configclass
class UnitreeGo2StairsEnvCfgSimpleReward_PLAY(UnitreeGo2StairsEnvCfgSimpleReward):
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


@configclass
class UnitreeGo2StairsEnvCfgComplexReward_PLAY(UnitreeGo2StairsEnvCfgComplexReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)
        parameters.set_play_settings_rough(self)


#######################################################################
# Stairs AMP


class modify_reward_weight(ManagerTermBase):
    """Curriculum that modifies the reward weight based on a step-wise schedule."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # obtain term configuration
        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)

    def lerp(self, a: float, b: float, t: float) -> float:
        t = min(1.0, max(0.0, t))
        return (1 - t) * a + t * b

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        weight: float,
        num_steps: float,
        warmup_period: float | None = None,
    ) -> float:
        # update term settings
        t = env.common_step_counter / env.num_envs
        if t > num_steps:
            if warmup_period is None:
                _weight = weight
            else:
                _weight = self.lerp(0.0, weight, t=(t - num_steps) / warmup_period)

            self._term_cfg.weight = _weight
            env.reward_manager.set_term_cfg(term_name, self._term_cfg)

        return self._term_cfg.weight


@configclass
class AMPUnitreeGo2StairsEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terrain_type = "stairs"
        parameters.set_terrain(self)
        parameters.set_curriculum(self, enable=False)
        parameters.set_stairs_env_cfg_cmds(self)
        parameters.set_stairs_env_cfg_reset_base(self)
        parameters.add_relative_position_on_stairs_observation(self)
        # self.observations.policy.yaw = None
        # self.observations.policy.relative_position = None
        parameters.add_stair_parameters_observation(self)

        parameters.set_velocity_rewards_amp(self)

        self.rewards.dof_torques_l2.weight = 0
        self.rewards.torque_limits.weight = 0
        self.rewards.torque_limits_2.weight = 0
        self.rewards.feet_stumble.weight = -5
        self.rewards.feet_slide.weight = -5
        self.rewards.stand_still.weight = -10
        self.rewards.feet_air_time.weight = 100

        self.rewards.undesired_contacts_thigh.weight = -20
        self.rewards.undesired_contacts_calf.weight = -20

        self.terminations.bad_orientation = None

        #
        self.curriculum.dof_torques_l2_schedule = CurriculumTermCfg(
            func=modify_reward_weight,
            params={
                "term_name": "dof_torques_l2",
                "weight": -0.006 * 5,
                "num_steps": 25,
                "warmup_period": 15,
            },
        )
        self.curriculum.torque_limits_schedule = CurriculumTermCfg(
            func=modify_reward_weight,
            params={
                "term_name": "torque_limits",
                "weight": -35,
                "num_steps": 25,
                "warmup_period": 15,
            },
        )
        self.curriculum.torque_limits_2_schedule = CurriculumTermCfg(
            func=modify_reward_weight,
            params={
                "term_name": "torque_limits_2",
                "weight": -100,
                "num_steps": 25,
                "warmup_period": 15,
            },
        )
        self.curriculum.feet_stumble_schedule = CurriculumTermCfg(
            func=modify_reward_weight,
            params={
                "term_name": "feet_stumble",
                "weight": -20,
                "num_steps": 25,
                "warmup_period": 15,
            },
        )

        self.events.push_robot.params["velocity_range"] = {
            "x": (-1, 1),
            "y": (-1, 1),
            "roll": (-1, 1),
            "pitch": (-1, 1),
            "yaw": (-1, 1),
        }
        self.events.base_external_force_torque.params["torque_range"] = (-0.1, 0.1)
        self.events.base_external_force_torque.params["force_range"] = (-0.1, 0.1)
        # params={
        #     "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        #     "force_range": (0.0, 0.0),
        #     "torque_range": (-0.0, 0.0),
        # },
        # self.curriculum.feet_air_time_schedule = CurriculumTermCfg(
        #     func=modify_reward_weight,
        #     params={
        #         "term_name": "feet_air_time",
        #         "weight": 100,
        #         "num_steps": 30,
        #         "warmup_period": 15,
        #     }
        # )

        self.scene.num_envs = 2 * 4096  # 5480

        # style
        self.action_manager_class = "ActionManager"  # Default action manager

        parameters.set_amp_settings(self, use_rsi=False)
        # update motion files
        self.amp_motion_folder = (
            "datasets/fromVision_motions_DepthCam_stairs_feetZAmpl_minimal_slow/*"
        )
        self.amp_motion_files = glob.glob(self.amp_motion_folder)

    def update_motion_files(self):
        motion_files = glob.glob(self.amp_motion_folder)
        self.amp_motion_files = motion_files

        # assert (
        #     self.events.reference_state_initialization is not None
        # ), "Always expecting RSI. For evaluation, please use the same motion files as used for training."
        # self.events.reference_state_initialization.params["motion_files"] = motion_files


@configclass
class AMPUnitreeGo2StairsEnvCfg_PLAY(AMPUnitreeGo2StairsEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)
        parameters.set_play_settings_rough(self)

        self.amp_motion_folder = "datasets/dummy/*"  # required otherwise it wont start; it is recomended to use same motion files as used for training
