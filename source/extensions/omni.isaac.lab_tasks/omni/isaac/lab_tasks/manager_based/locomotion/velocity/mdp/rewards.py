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

from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
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
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
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
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


# def track_lin_vel_z_exp(
#     env,
#     std: float,
#     command_name: str,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     # extract the used quantities (to enable type-hinting)
#     asset = env.scene[asset_cfg.name]
#     rows = env.scene.terrain.terrain_levels
#     cols = env.scene.terrain.terrain_types
#     slope = (
#         env.scene.terrain.terrain_params["step_height"][rows, cols]
#         / env.scene.terrain.terrain_params["step_width"][rows, cols]
#     )
#
#     v_target_z = slope * torch.norm(
#         env.command_manager.get_command(command_name)[:, :2], p=2, dim=-1
#     )
#     lin_vel_error = torch.square(v_target_z - asset.data.root_lin_vel_w[:, 2])
#     return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_stairs_exp(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    rows = env.scene.terrain.terrain_levels
    cols = env.scene.terrain.terrain_types
    slope = (
        env.scene.terrain.terrain_params["step_height"][rows, cols]
        / env.scene.terrain.terrain_params["step_width"][rows, cols]
    )
    root_pos_y_absolute = asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    y_position_relative = (
        root_pos_y_absolute
        + env.cfg.scene.terrain.terrain_generator.sub_terrains[
            "stairs"
        ].y_coordinate_origin_relative_to_first_stair_step
    )
    stairs_end = (
        env.scene.terrain.terrain_params["num_steps"][rows, cols]
        * env.scene.terrain.terrain_params["step_width"][rows, cols]
        + env.cfg.scene.terrain.terrain_generator.sub_terrains[
            "stairs"
        ].y_coordinate_origin_relative_to_first_stair_step
    )
    target_height = 0.3
    is_on_stairs = torch.logical_and(
        -target_height / slope <= y_position_relative,  # ,
        y_position_relative <= stairs_end + target_height / slope,  # ,
    )

    # vel_yaw = quat_rotate_inverse(
    #     yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    # )
    command = env.command_manager.get_command(command_name)[:, :3]
    command[is_on_stairs, 2] = (slope * torch.norm(command[:, :2], p=2, dim=-1))[
        is_on_stairs
    ]
    v = asset.data.root_lin_vel_w[:, :3]
    command[:, 2] *= 3
    v[:, 2] *= 3
    lin_vel_error = torch.sum(
        torch.square(command[:, :3] - v),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def track_ang_vel_z_world_exp_3d(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 3] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def foot_clearance_reward_flat(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]

    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

def foot_clearance_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_height: float = -0.22):
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
    feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=asset.device, requires_grad=False)
    for i in range(len(feet_names)):
        feet_indices[i] = asset.find_bodies(feet_names[i])[0][0]
    feet_indices = asset_cfg.body_ids

    feet_pos = rigid_body_states[:, feet_indices, 0:3]
    feet_vel = rigid_body_states[:, feet_indices, 7:10]

    cur_footpos_translated = feet_pos - root_states[:, 0:3].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(num_envs, len(feet_indices), 3, device=asset.device)
    cur_footvel_translated = feet_vel - root_states[:, 7:10].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(num_envs, len(feet_indices), 3, device=asset.device)
    for i in range(len(feet_indices)):
        footpos_in_body_frame[:, i, :] = quat_rotate_inverse(base_quat, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_rotate_inverse(base_quat, cur_footvel_translated[:, i, :])

    height_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(num_envs, -1)
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(num_envs, -1)

    clearance_reward = height_error * foot_leteral_vel

    return torch.sum(clearance_reward, dim=1)


def base_height(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: RigidObject = env.scene[asset_cfg.name]
    z = asset.data.root_state_w[:, 2]
    return z


def base_z_vel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: RigidObject = env.scene[asset_cfg.name]
    z_vel = asset.data.root_state_w[:, 3 + 4 + 2]
    # TODO: Limit to either only be active during the stairs section
    # or in general to be within limits / around the necessary speed based on the target command
    return torch.clamp(
        z_vel,
        torch.zeros((), device=z_vel.device),
        torch.tensor(0.144, device=z_vel.device),
    )


def base_z_at_stairs(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    rows = env.scene.terrain.terrain_levels
    cols = env.scene.terrain.terrain_types
    stairs_end = (
        env.scene.terrain.terrain_params["num_steps"][rows, cols]
        * env.scene.terrain.terrain_params["step_width"][rows, cols]
        + env.cfg.scene.terrain.terrain_generator.sub_terrains[
            "stairs"
        ].y_coordinate_origin_relative_to_first_stair_step
    )

    slope = (
        env.scene.terrain.terrain_params["step_height"][rows, cols]
        / env.scene.terrain.terrain_params["step_width"][rows, cols]
    )
    root_pos_y_absolute = asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    y_position_relative = (
        root_pos_y_absolute
        + env.cfg.scene.terrain.terrain_generator.sub_terrains[
            "stairs"
        ].y_coordinate_origin_relative_to_first_stair_step
    )
    max_height = (
        env.scene.terrain.terrain_params["num_steps"][rows, cols]
        * env.scene.terrain.terrain_params["step_height"][rows, cols]
    )

    # Stairs start at 0 of y_position_relative
    y_target = (y_position_relative * slope) + target_height
    # print("y target", y_target)
    # print("y", asset.data.root_state_w[:, 2])
    # print()
    y_target = torch.clamp(
        y_target,
        torch.tensor(target_height, device=y_target.device),
        max_height + target_height,
    )

    # TODO: USE TANH SOMEHOW
    # torch.tanh(y_target, )

    # print("y_position_relative", y_position_relative)
    # print("offset", y_position_relative - target_height / slope)

    # Use the root of the new slope for the on stairs direction.
    # TODO: What if we interpolate linearly?
    is_on_stairs = torch.logical_and(
        y_position_relative - target_height / slope > 0,
        y_position_relative < stairs_end,
    )
    before_stairs = y_position_relative - target_height / slope > 0
    after_stairs = y_position_relative < stairs_end

    # Target speed in xy plane
    v_target_xy = env.command_manager.get_command(command_name)[:, :2]
    # Current velocity alongisde z-axis
    v_z = asset.data.root_state_w[:, 2]

    # y_target[~before_stairs] = target_height
    # y_target[~after_stairs] = (target_height + max_height)[~after_stairs]

    v_target_z = slope * torch.norm(v_target_xy, p=2, dim=-1)

    delta_1 = -torch.square(v_target_z - v_z)
    delta_2 = -2 * torch.square(v_z)

    return torch.where(v_z >= 0.0, delta_1, delta_2)
    # print(torch.max(error))
    # error[~is_on_stairs] = 0.03

    # Because of tanh's strict monotonicity greater values are ignored
    # target_error_for_99percent = 0.05  # [in m]
    # error = (
    #     torch.tanh(
    #         (y_target - asset.data.root_state_w[:, 2])
    #         * 2.646
    #         / target_error_for_99percent
    #     )
    #     * is_on_stairs
    # )
    return -error
