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

        self.terrain_type = "shortstairs"
        parameters.set_terrain(self)
        parameters.set_rewards_complex(self)
        parameters.set_curriculum(self, True)

        self.episode_length_s = 8.0
        self.scene.terrain.max_init_terrain_level = 0

        # Random force pushes on body
        self.events.push_robot.params["velocity_range"] = {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        }
        self.events.push_robot.interval_range_s = (2.0, 6.0)
        # Random feet pushes
        # self.events.push_feet.params["velocity_range"] = {
        #     "x": (-0.02, 0.02),
        #     "y": (-0.02, 0.02),
        # }
        # self.events.push_feet.interval_range_s = (1.0, 3.0)
        self.rewards.foot_clearance.weight = -5
        self.rewards.feet_air_time.weight = 1

        self.curriculum.terrain_levels.params[
            "custom_required_distance_for_move_up"
        ] = (((0.7 - 0.3) / 2 + 0.2) * 8 * 2 / 3)

        self.rewards.dof_torques_l2.weight = -0.006 / 5
        self.rewards.dof_acc_l2.weight = -2.5e-7 / 4
        self.rewards.torque_limits.weight = -1.0e-5 / 4
        self.rewards.dof_acc_l2.weight = -2.5e-7 / 5
        self.rewards.action_rate_l2.weight = -0.01 / 15

        num_steps = [0, 2, 8]
        warmup_period = 10
        data = {
            "undesired_contacts_thigh": {
                "order": 0,
                "weight": -1 / 5,
            },
            "undesired_contacts_calf": {
                "order": 0,
                "weight": -1 / 5,
            },
            # "dof_torques_l2": {"order": 1, "weight": -0.006 / 8},
            "dof_acc_l2": {"order": 1, "weight": -2.5e-7 / 5},
            # "torque_limits": {"order": 1, "weight": -1.0e-5 / 4},
            # "torque_limits_2": {"order": 1, "weight": -1.0e-5 / 4},
            "feet_stumble": {"order": 0, "weight": -0.2},
            "feet_slide": {"order": 2, "weight": -0.02},
            "feet_air_time": {"order": 2, "weight": 20},
            # "contact_forces": {"order": 2, "weight": -0.2},
            "ang_vel_xy_l2": {"order": 1, "weight": -0.05},
            "lin_vel_z_l2": {"order": 1, "weight": -0.01},
            # "foot_clearance": {"order": 0, "weight": -1.0},
            # "joint_deviation_l1_hip": {
            #     "order": 1,
            #     "weight": -0.5,
            # },
            # "joint_deviation_l1_calf_thigh": {
            #     "order": 1,
            #     "weight": -0.1,
            # },
            # "dof_acc_l2": {"order": 2, "weight": -2.5e-7 / 10},
            # "action_rate_l2": {"order": 2, "weight": -0.01 / 10},
        }
            for key, value in data.items():
                setattr(
                    self.curriculum,
                    f"{key}_schedule",
                    CurriculumTermCfg(
                        func=modify_reward_weight,
                        params={
                            "term_name": key,
                            "weight": value["weight"],
                            "initial_weight": getattr(self.rewards, key).weight,
                            "num_steps": num_steps[value["order"]],
                            "warmup_period": getattr(value, "warmup_period", warmup_period),
                        },
                    ),
                )

            import math

            self.events.reset_base.params["pose_range"] = {
                "x": (-0.5, 0.5),
                "y": (-0.9, 0.3),
                "z": (-0.08, -0.08),
                # "roll": (-math.radians(20), math.radians(20)),
                # "pitch": (-math.radians(20), math.radians(20)),
                "yaw": (math.pi / 2 - math.radians(15), math.pi / 2 + math.radians(15)),
            }


    @configclass
    class UnitreeGo2StairsEnvCfgComplexReward_PLAY(UnitreeGo2StairsEnvCfgComplexReward):
        def __post_init__(self):
            # post init of parent
            super().__post_init__()

            parameters.set_play_settings_flat(self)
            parameters.set_play_settings_rough(self)


#######################################################################
# Stairs AMP
    from typing import ClassVar


    class modify_env_param(ManagerTermBase):
        NO_CHANGE: ClassVar = object()
        """Special token to indicate no change in the value to be set.

        This token is used to signal that the `modify_fn` did not produce a new value. It can
        be returned by the `modify_fn` to indicate that the current value should remain unchanged.
        """

        def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
            super().__init__(cfg, env)
            # resolve term configuration
            if "address" not in cfg.params:
                raise ValueError(
                    "The 'address' parameter must be specified in the curriculum term configuration."
                )

            # store current address
            self._address: str = cfg.params["address"]
            # store accessor functions
            self._get_fn: callable = None
            self._set_fn: callable = None

        def __del__(self):
            """Destructor to clean up the compiled functions."""
            # clear the getter and setter functions
            self._get_fn = None
            self._set_fn = None
            self._container = None
            self._last_path = None

        """
        Operations.
        """

        def __call__(
            self,
            env: ManagerBasedRLEnv,
            env_ids: Sequence[int],
            address: str,
            modify_fn: callable,
            modify_params: dict | None = None,
        ):
            # fetch the getter and setter functions if not already compiled
            if not self._get_fn:
                self._get_fn, self._set_fn = self._process_accessors(
                    self._env, self._address
                )

            # resolve none type
            modify_params = {} if modify_params is None else modify_params

            # get the current value of the target attribute
            data = self._get_fn()
        # modify the value using the provided function
        new_val = modify_fn(self._env, env_ids, data, **modify_params)
        # set the modified value back to the target attribute
        # note: if the modify_fn return NO_CHANGE signal, we do not invoke self.set_fn
        if new_val is not self.NO_CHANGE:
            self._set_fn(new_val)

    """
    Helper functions.
    """

    def _process_accessors(
        self, root: ManagerBasedRLEnv, path: str
    ) -> tuple[callable, callable]:
        """Process and return the (getter, setter) functions for a dotted attribute path.

        This function resolves a dotted path string to an attribute in the given root object.
        The dotted path can include nested attributes, dictionary keys, and sequence indexing.

        For instance, the path "foo.bar[2].baz" would resolve to `root.foo.bar[2].baz`. This
        allows accessing attributes in a nested structure, such as a dictionary or a list.

        Args:
            root: The main object from which to resolve the attribute.
            path: Dotted path string to the attribute variable. For e.g., "foo.bar[2].baz".

        Returns:
            A tuple of two functions (getter, setter), where:
            the getter retrieves the current value of the attribute, and
            the setter writes a new value back to the attribute.
        """
        import re

        # Turn "a.b[2].c" into ["a", ("b", 2), "c"] and store in parts
        path_parts: list[str | tuple[str, int]] = []
        for part in path.split("."):
            m = re.compile(r"^(\w+)\[(\d+)\]$").match(part)
            if m:
                path_parts.append((m.group(1), int(m.group(2))))
            else:
                path_parts.append(part)

        # Traverse the parts to find the container
        container = root
        for container_path in path_parts[:-1]:
            if isinstance(container_path, tuple):
                # we are accessing a list element
                name, idx = container_path
                # find underlying attribute
                if isinstance(container_path, dict):
                    seq = container[name]  # type: ignore[assignment]
                else:
                    seq = getattr(container, name)
                # save the container for the next iteration
                container = seq[idx]
            else:
                # we are accessing a dictionary key or an attribute
                if isinstance(container, dict):
                    container = container[container_path]
                else:
                    container = getattr(container, container_path)

        # save the container and the last part of the path
        self._container = container
        self._last_path = path_parts[
            -1
        ]  # for "a.b[2].c", this is "c", while for "a.b[2]" it is 2

        # build the getter and setter
        if isinstance(self._container, tuple):
            get_value = lambda: self._container[self._last_path]  # noqa: E731

            def set_value(val):
                tuple_list = list(self._container)
                tuple_list[self._last_path] = val
                self._container = tuple(tuple_list)

        elif isinstance(self._container, (list, dict)):
            get_value = lambda: self._container[self._last_path]  # noqa: E731

            def set_value(val):
                self._container[self._last_path] = val

        elif isinstance(self._container, object):
            get_value = lambda: getattr(self._container, self._last_path)  # noqa: E731
            set_value = lambda val: setattr(
                self._container, self._last_path, val
            )  # noqa: E731
        else:
            raise TypeError(
                f"Unable to build accessors for address '{path}'. Unknown type found for access variable:"
                f" '{type(self._container)}'. Expected a list, dict, or object with attributes."
            )

        return get_value, set_value


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
        initial_weight: float | None = None,
    ) -> float:
        # update term settings
        t = env.common_step_counter / env.num_envs
        if t > num_steps:
            if warmup_period is None:
                _weight = weight
            else:
                _weight = self.lerp(
                    initial_weight if initial_weight is not None else 0.0,
                    weight,
                    t=(t - num_steps) / warmup_period,
                )

            self._term_cfg.weight = _weight
            env.reward_manager.set_term_cfg(term_name, self._term_cfg)

        return self._term_cfg.weight


class modify_term_cfg(modify_env_param):
    def __init__(self, cfg, env):
        # initialize the parent
        super().__init__(cfg, env)
        # overwrite the simplified address with the full manager path
        self._address = self._address.replace("s.", "_manager.cfg.", 1)


def override_command_range(env, env_ids, old_value, value, num_steps):
    if env.common_step_counter / env.num_envs > num_steps:
        return value
    return modify_term_cfg.NO_CHANGE


@configclass
class AMPUnitreeGo2StairsEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terrain_type = "stairs"
        parameters.set_terrain(self)
        parameters.set_curriculum(self, enable=True)
        parameters.set_stairs_env_cfg_cmds(self)
        parameters.set_stairs_env_cfg_reset_base(self)
        # parameters.add_relative_position_on_stairs_observation(self)
        # self.observations.policy.yaw = None
        # self.observations.policy.relative_position = None
        # parameters.add_stair_parameters_observation(self)

        parameters.set_velocity_rewards_amp(self)

        deployment = True
        if deployment:
            num_steps = [15, 25, 40]
            warmup_period = 20
            # self.rewards.stand_still.weight = -5
            self.rewards.feet_contact_without_cmd.weight = 0.1
            self.rewards.feet_air_time.weight = 100
            self.rewards.feet_on_step.weight = 0
            self.rewards.ang_vel_xy_l2.weight = 0
            self.rewards.ang_vel_x_l2.weight = 0
            self.rewards.sparse_end_of_stairs.weight = 0

            self.rewards.feet_on_step.params["distance_a"] = 0.02
            self.rewards.feet_on_step.params["distance_b"] = 0.02

            data = {
                "feet_on_step": {
                    "order": 0,
                    "weight": -40,
                },
                "undesired_contacts_thigh": {
                    "order": 0,
                    "weight": -20 * 0.5,
                },
                "undesired_contacts_calf": {
                    "order": 0,
                    "weight": -20 * 0.5,
                },
                "sparse_end_of_stairs": {
                    "order": 0,
                    "weight": 1500,
                    "warmup_period": 30,
                },
                "dof_torques_l2": {"order": 1, "weight": -0.006 * 3},
                "torque_limits": {"order": 1, "weight": -35 * 1},
                "torque_limits_2": {"order": 1, "weight": -100 * 1},
                "feet_stumble": {"order": 1, "weight": -50},
                "feet_slide": {"order": 1, "weight": -2.5},
                # "joint_deviation_l1_hip": {
                #     "order": 1,
                #     "weight": -0.5,
                # },
                # "joint_deviation_l1_calf_thigh": {
                #     "order": 1,
                #     "weight": -0.1,
                # },
                "ang_vel_xy_l2": {
                    "order": 1,
                    "weight": -10.0,
                },
                # "dof_acc_l2": {"order": 2, "weight": -2.5e-7 * 0.01},
                # "action_rate_l2": {"order": 2, "weight": -0.01 * 0.05},
            }
            for key, value in data.items():
                setattr(
                    self.curriculum,
                    f"{key}_schedule",
                    CurriculumTermCfg(
                        func=modify_reward_weight,
                        params={
                            "term_name": key,
                            "weight": value["weight"],
                            "initial_weight": getattr(self.rewards, key).weight,
                            "num_steps": num_steps[value["order"]],
                            "warmup_period": getattr(
                                value, "warmup_period", warmup_period
                            ),
                        },
                    ),
                )

        self.events.push_robot.params["velocity_range"] = {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        }
        self.events.push_robot.interval_range_s = (2.0, 6.0)
        # self.events.base_external_force_torque.params["torque_range"] = (-0.1, 0.1)
        # self.events.base_external_force_torque.params["force_range"] = (-0.1, 0.1)

        self.scene.num_envs = 2 * 4096  # 5480

        # style
        self.action_manager_class = "ActionManager"  # Default action manager

        parameters.set_amp_settings(self, use_rsi=False)
        # update motion files
        self.amp_motion_folder = "datasets/fromVision_motions_DepthCam_stairsv3fast_feetZAmpl_minimal_stairs2_slow/*"
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


@configclass
class AMPUnitreeGo2StairsAlignmentEnvCfg(AMPUnitreeGo2StairsEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terrain_type = "stairs"
        parameters.set_terrain(self)
        parameters.set_curriculum(self, False)
        self.scene.terrain.terrain_generator.sub_terrains[
            "stairs"
        ].step_height_range = (
            0.0,
            0.0,
        )

        self.rewards.feet_on_step.weight = 0
        self.rewards.feet_on_step.params["distance_a"] = 0.01
        self.rewards.feet_on_step.params["distance_b"] = 0.01
        self.curriculum.feet_on_step_schedule = CurriculumTermCfg(
            func=modify_reward_weight,
            params={
                "term_name": "feet_on_step",
                "weight": 20,
                "num_steps": 5,
                "warmup_period": 5,
                "initial_weight": self.rewards.feet_on_step.weight,
            },
        )

        self.episode_length_s = 5.0

        self.amp_motion_folder = "datasets/fromVision_motions_DepthCam_extendedWithoutReverse_feetZAmpl_minimal_feet_forward/*"
        self.amp_motion_files = glob.glob(self.amp_motion_folder)


@configclass
class AMPUnitreeGo2ShortStairsEnvCfg(AMPUnitreeGo2StairsEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terrain_type = "shortstairs"
        parameters.set_terrain(self)

        self.episode_length_s = 8.0

        # Random force pushes on body
        self.events.push_robot.params["velocity_range"] = {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        }
        self.events.push_robot.interval_range_s = (1.0, 5.0)
        # Random feet pushes
        self.events.push_feet.params["velocity_range"] = {
            "x": (-0.04, 0.04),
            "y": (-0.04, 0.04),
        }
        self.events.push_feet.interval_range_s = (1.0, 3.0)

        if self.curriculum.terrain_levels is not None:
            self.curriculum.terrain_levels.params[
                "custom_required_distance_for_move_up"
            ] = (((0.7 - 0.2) / 2 + 0.2) * 8 * 2 / 3)


@configclass
class AMPUnitreeGo2ShortStairsEnvCfg_PLAY(AMPUnitreeGo2ShortStairsEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)
        parameters.set_play_settings_rough(self)

        self.amp_motion_folder = "datasets/dummy/*"  # required otherwise it wont start; it is recomended to use same motion files as used for training


@configclass
class AMPUnitreeGo2ShortStairsBasicEnvCfg(AMPUnitreeGo2StairsEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terrain_type = "shortstairs"
        parameters.set_terrain(self)
        parameters.set_rewards_simple(self)

        from dataclasses import fields

        for field in fields(self.curriculum):
            if field.name == "terrain_levels":
                continue
            setattr(self.curriculum, field.name, None)

        self.curriculum.feet_on_step_schedule = None
        self.curriculum.joint_deviation_l1_schedule = None
        self.curriculum.dof_acc_l2_schedule = None
        self.curriculum.action_rate_l2_schedule = None
        self.curriculum.dof_torques_l2_schedule = None
        self.curriculum.torque_limits_schedule = None
        self.curriculum.torque_limits_2_schedule = None
        self.curriculum.feet_stumble_schedule = None
        self.curriculum.feet_slide_schedule = None
        self.curriculum.undesired_contacts_thigh_schedule = None
        self.curriculum.undesired_contacts_calf_schedule = None

        self.episode_length_s = 8.0

        # Random force pushes on body
        self.events.push_robot.params["velocity_range"] = {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        }
        self.events.push_robot.interval_range_s = (2.0, 6.0)
        # Random feet pushes
        self.events.push_feet.params["velocity_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
        }
        self.events.push_feet.interval_range_s = (1.0, 3.0)

        self.curriculum.terrain_levels.params[
            "custom_required_distance_for_move_up"
        ] = (((0.7 - 0.2) / 2 + 0.2) * 8 * 2 / 3)


@configclass
class AMPUnitreeGo2ShortStairsBasicEnvCfg_PLAY(AMPUnitreeGo2ShortStairsEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)
        parameters.set_play_settings_rough(self)

        self.amp_motion_folder = "datasets/dummy/*"  # required otherwise it wont start; it is recomended to use same motion files as used for training
