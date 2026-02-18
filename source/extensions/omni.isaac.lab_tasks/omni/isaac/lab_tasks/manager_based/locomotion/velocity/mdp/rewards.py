# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(
        torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1
    )
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return reward


def feet_air_time_positive_biped(
    env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return reward


def feet_slide(
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2
    )
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(
        yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]
        ),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2]
        - asset.data.root_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def joint_mirror(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if (
        not hasattr(env, "joint_mirror_joints_cache")
        or env.joint_mirror_joints_cache is None
    ):
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair]
            for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                asset.data.joint_pos[:, joint_pair[0][0]]
                - asset.data.joint_pos[:, joint_pair[1][0]]
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def action_mirror(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if (
        not hasattr(env, "action_mirror_joints_cache")
        or env.action_mirror_joints_cache is None
    ):
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair]
            for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def action_sync(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if (
        not hasattr(env, "action_sync_joint_cache")
        or env.action_sync_joint_cache is None
    ):
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group]
            for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [
                torch.abs(env.action_manager.action[:, joint[0]])
                for joint in joint_group
            ],
            dim=1,
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= (
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    )
    return reward


def trapezoid_step(x, a, b, L=0.1):
    """
    Generates a smooth trapezoidal step function.

    It is 0 for x <= a-L and x >= b+L, 1 for a+L <= x <= b-L,
    and linearly ramps over the transition length L=0.1.

    Args:
        x (float or np.ndarray): The input value(s).
        a (float): The start of the rise (where y starts increasing from 0).
        b (float): The end of the fall (where y finishes decreasing to 0).
        L (float): The length of the linear transition (default is 0.1).

    Returns:
        float or np.ndarray: The function value(s).
    """

    # 1. Rising ramp (from 0 to 1 between a and a+L)
    # The linear equation is y = (x - a) / L
    ramp_up = torch.clamp((x - a) / L, 0.0, 1.0)

    # 2. Falling ramp (from 1 to 0 between b-L and b)
    # The linear equation is y = 1 - (x - (b - L)) / L
    ramp_down = torch.clamp(1.0 - (x - (b - L)) / L, 0.0, 1.0)

    # The final function is the minimum of the two ramps.
    # This creates the flat top of 1.0 between the ramps.
    return torch.minimum(ramp_up, ramp_down)


def trapezoid_step_asymmetric(x, a, b, L_a=0.1, L_b=0.1):
    """
    Generates a smooth trapezoidal step function.
    """

    # 1. Rising ramp (from 0 to 1 between a and a+L)
    # The linear equation is y = (x - a) / L
    ramp_up = torch.clamp((x - a) / L_a, 0.0, 1.0)

    # 2. Falling ramp (from 1 to 0 between b-L and b)
    # The linear equation is y = 1 - (x - (b - L)) / L
    ramp_down = torch.clamp(1.0 - (x - (b - L_b)) / L_b, 0.0, 1.0)

    # The final function is the minimum of the two ramps.
    # This creates the flat top of 1.0 between the ramps.
    return torch.minimum(ramp_up, ramp_down)


def feet_on_step(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    distance_a: float = 0.05,
    distance_b: float = 0.05,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    rows = env.scene.terrain.terrain_levels
    cols = env.scene.terrain.terrain_types
    n = env.scene.terrain.terrain_params["num_steps"][rows, cols]
    b = env.scene.terrain.terrain_params["step_width"][rows, cols]

    feet_indices = asset.find_bodies(["FR_foot", "FL_foot", "RR_foot", "RL_foot"])[0]
    feet_pos_y = (
        asset.data.body_pos_w[:, feet_indices, 1]
        - env.scene.env_origins[:, 1].unsqueeze(1)
        + env.cfg.scene.terrain.terrain_generator.sub_terrains[
            "stairs"
        ].y_coordinate_origin_relative_to_first_stair_step
    )
    n = n.unsqueeze(-1)
    b = b.unsqueeze(-1)
    feet_pos_y_rel = torch.fmod(
        feet_pos_y + b,
        b,
    )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact: torch.Tensor = (
        contact_sensor.data.net_forces_w[
            :, torch.tensor([4, 8, 14, 18], device="cuda"), 2
        ]
        > 10
    )

    is_on_stairs: torch.Tensor = torch.logical_and(
        feet_pos_y + distance_b >= 0.0, feet_pos_y + distance_b <= n * b
    )
    reward: torch.Tensor = (
        (1 - trapezoid_step_asymmetric(feet_pos_y_rel, 0.0, b, distance_a, distance_b))
        * is_contact
        * is_on_stairs
    )

    return torch.sum(reward, dim=1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = -0.22,
):
    """
    Reward the swinging feet for clearing a specified height off the ground.

    Code adapted from SLR paper.
    """
    asset = env.scene[asset_cfg.name]
    rigid_body_states = asset.data.body_state_w
    base_quat = asset.data.root_quat_w
    root_states = asset.data.root_state_w

    num_envs = asset.num_instances

    feet_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    feet_indices = torch.zeros(
        len(feet_names), dtype=torch.long, device=asset.device, requires_grad=False
    )
    for i in range(len(feet_names)):
        feet_indices[i] = asset.find_bodies(feet_names[i])[0][0]
    # feet_indices = asset_cfg.body_ids

    feet_pos = rigid_body_states[:, feet_indices, 0:3]
    feet_vel = rigid_body_states[:, feet_indices, 7:10]

    cur_footpos_translated = feet_pos - root_states[:, 0:3].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(
        num_envs, len(feet_names), 3, device=asset.device
    )
    cur_footvel_translated = feet_vel - root_states[:, 7:10].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(
        num_envs, len(feet_names), 3, device=asset.device
    )
    for i in range(len(feet_names)):
        footpos_in_body_frame[:, i, :] = quat_rotate_inverse(
            base_quat, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = quat_rotate_inverse(
            base_quat, cur_footvel_translated[:, i, :]
        )

    height_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(
        num_envs, -1
    )
    foot_leteral_vel = torch.sqrt(
        torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)
    ).view(num_envs, -1)

    clearance_reward = height_error * foot_leteral_vel

    return torch.sum(clearance_reward, dim=1)


def sparse_end_of_stairs_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: RigidObject = env.scene[asset_cfg.name]

    rows = env.scene.terrain.terrain_levels
    cols = env.scene.terrain.terrain_types

    # step width for each robot's terrain
    step_width = env.scene.terrain.terrain_params["step_width"][rows, cols]
    num_steps = env.scene.terrain.terrain_params["num_steps"][rows, cols]

    root_pos_x_absolute = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    x_position_relative = (
        root_pos_x_absolute
        + env.cfg.scene.terrain.terrain_generator.sub_terrains[
            "stairs"
        ].x_coordinate_origin_relative_to_first_stair_step  # needs to be ADDED according to definition
    )

    on_stairs = torch.logical_and(
        x_position_relative >= 0.0,
        x_position_relative - (num_steps + 0.5) * step_width <= 0,
    )
    over_stairs = x_position_relative - num_steps * step_width >= 0.0

    # has_passed = over_stairs
    # is_first_pass = torch.logical_and(
    #     has_passed, torch.logical_not(env.has_passed_target) >= 1
    # )
    # if is_first_pass.any():
    #     print(is_first_pass)
    #
    # reward = torch.where(
    #     is_first_pass,
    #     torch.tensor(1.0, device=env.device),
    #     torch.tensor(0.0, device=env.device),
    # )
    # env.has_passed_target[is_first_pass] = 1

    passed_step = (x_position_relative // step_width) * on_stairs
    higher_target_reached = torch.logical_and(
        passed_step > env.has_passed_target, on_stairs
    )
    env.has_passed_target[higher_target_reached] += 1
    reward = torch.where(
        higher_target_reached,
        torch.tensor(1.0, device=env.device),
        torch.tensor(0.0, device=env.device),
    )

    return reward
