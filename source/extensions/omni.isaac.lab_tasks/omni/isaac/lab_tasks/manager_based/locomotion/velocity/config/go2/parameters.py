from dataclasses import fields
import glob
import math
import torch

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg

from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils.noise import (
    AdditiveUniformNoiseCfg as Unoise,
    UniformSinusodalPositionNoiseCfg as USinPosNoise,
    UniformAngleNoiseCfg as UAngleNoise,
    BinaryNoiseCfg as BinaryNoise,
)

from omni.isaac.lab_tasks.manager_based.navigation.mdp.rewards import position_command_error_tanh, heading_command_error_abs

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from omni.isaac.lab.terrains.config.flat_noisy import FLAT_TERRAINS_CFG  # isort: skip
from omni.isaac.lab.terrains.config.stairs import STAIRS_TERRAINS_CFG  # isort: skip

from omni.isaac.lab.terrains.config.box import BOX_TERRAINS_CFG  # isort: skip

def set_play_settings_flat(cfg):
    cfg.scene.num_envs = 100
    cfg.scene.env_spacing = 2.5
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None


def set_play_settings_rough(cfg):
    # reduce the number of terrains to save memory
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.curriculum = False

        cfg.scene.terrain.terrain_generator.sub_terrains["stairs"].step_height_range = (
            0.0,
            0.10,
        )


def set_curriculum(cfg, enable: bool):
    if not enable:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.curriculum.terrain_levels = None
        cfg.scene.terrain.max_init_terrain_level = None
        if cfg.curriculum is not None:
            cfg.curriculum.terrain_levels = None
        else:
            print("[WARN] Curriculum Manager is disabled.")

        cfg.scene.terrain.terrain_generator.sub_terrains["stairs"].step_height_range = (
            0.0,
            0.25,
        )
        cfg.scene.terrain.terrain_generator.sub_terrains["stairs"].step_width = None

        # cfg.curriculum.sparse_reward_schedule = CurrTerm(
        #     func=modify_reward_weight,
        #     params={
        #         "term_name": "sparse_reward",
        #         "num_steps": 10_000,
        #     },
        # )
        # cfg.curriculum = None
        # cfg.curriculum.epsiode_length = CurrTerm(
        #     func=mdp.modify_env_param,
        #     params={
        #         "address": "cfg.episode_length_s",
        #         "modify_fn": resample_epsiode_length,  # e.g., decrease or increase
        #     },
        # )
    else:
        cfg.scene.terrain.terrain_generator.curriculum = True
        cfg.scene.terrain.max_init_terrain_level = 0
        assert cfg.terrain_type == "stairs"
        cfg.curriculum.terrain_levels = CurrTerm(
            func=mdp.terrain_levels_vel,
            params={"custom_required_distance_for_move_up": 3},
        )
        assert "stairs" in cfg.scene.terrain.terrain_generator.sub_terrains
        cfg.scene.terrain.terrain_generator.sub_terrains["stairs"].step_height_range = (
            0.00,
            0.20,
        )
        cfg.scene.terrain.terrain_generator.sub_terrains["stairs"].step_width = None
        # cfg.scene.terrain.terrain_generator.sub_terrains["stairs"].step_width = 0.3


def set_terrain(cfg):
    if cfg.terrain_type == "flat":
        # change terrain to flat
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.curriculum.terrain_levels = None
        # no height scan
        cfg.scene.height_scanner = None
        cfg.observations.policy.height_scan = None
    elif cfg.terrain_type == "rough":
        assert (
            cfg.scene.terrain.terrain_generator == ROUGH_TERRAINS_CFG
        ), "Expected ROUGH_TERRAINS_CFG as default terrain generator."
        # scale down the terrains because the robot is small
        cfg.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (
            0.025,
            0.1,
        )
        cfg.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (
            0.01,
            0.06,
        )
        cfg.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = (
            0.01
        )
    elif cfg.terrain_type == "stairs":
        cfg.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG
        cfg.scene.height_scanner = None
        cfg.observations.policy.height_scan = None
        cfg.curriculum.terrain_levels = None
    elif cfg.terrain_type == "box":
        cfg.scene.terrain.terrain_generator = BOX_TERRAINS_CFG
        cfg.scene.height_scanner = None
        cfg.observations.policy.height_scan = None
    elif cfg.terrain_type == "flat_noisy":
        cfg.scene.terrain.terrain_generator = None
        cfg.curriculum.terrain_levels = None
        cfg.scene.height_scanner = None
        cfg.observations.policy.height_scan = None
        cfg.scene.terrain.terrain_generator = FLAT_TERRAINS_CFG
    else:
        raise ValueError(f"Unknown terrain type: {cfg.terrain_type}.")


def set_rewards_simple(cfg):
    # disable rewards
    for field in fields(cfg.rewards):
        reward_obj = getattr(cfg.rewards, field.name)
        reward_obj.weight = 0.0

    # set task reward: from AMP for hardware baseline
    # NOTE AMP for Hardware has std=1. Can be activated by commenting out the two following lines
    cfg.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
    cfg.rewards.track_ang_vel_z_exp.params["std"] = 0.5
    cfg.rewards.track_lin_vel_xy_exp.weight = 3.25  # was 1.5 before adding actuator delay; was 2.5 before increasing actuator delay 1 -> 4
    cfg.rewards.track_ang_vel_z_exp.weight = (
        1.5  # was 0.75 before adding actuator delay
    )

def set_rewards_standing(cfg):
    # disable rewards
    for field in fields(cfg.rewards):
        reward_obj = getattr(cfg.rewards, field.name)
        reward_obj.weight = 0.0
        
    cfg.rewards.head_height_l2.weight = 1.0
    cfg.rewards.feet_height_l2.weight = 1.0
    
    # Make robot stand still, i.e., avoid drift
    cfg.commands.base_velocity.ranges.lin_vel_x=(0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y=(0.0, 0.0)
    cfg.rewards.track_lin_vel_xy_exp.weight = 0.3

def set_rewards_standing_amp(cfg):
    # disable rewards
    for field in fields(cfg.rewards):
        reward_obj = getattr(cfg.rewards, field.name)
        reward_obj.weight = 0.0
        
    # cfg.rewards.base_height_l2.weight = 1.0
    # cfg.rewards.head_height_l2.weight = 45.0
    cfg.rewards.feet_height_l2.weight = 45.0
    
    # Make robot stand still, i.e., avoid drift
    cfg.commands.base_velocity.ranges.lin_vel_x=(0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y=(0.0, 0.0)
    cfg.rewards.track_lin_vel_xy_exp.weight = 20
    
    cfg.rewards.dof_torques_l2.weight = -0.006 # -2.8
    cfg.rewards.torque_limits.weight = -70 # -0.5
    cfg.rewards.torque_limits_2.weight = -200 # -0.005
    cfg.rewards.dof_acc_l2.weight = -5e-6


def set_velocity_rewards_amp(cfg):
    # disable rewards
    for field in fields(cfg.rewards):
        reward_obj = getattr(cfg.rewards, field.name)
        # we need to check this because some reward terms might have been deleted previously depending on the environment and task
        if reward_obj is not None:
            reward_obj.weight = 0.0

    # set only task reward
    cfg.rewards.track_lin_vel_xy_exp.weight = 60
    cfg.rewards.track_lin_vel_xy_exp.params["std"] = 0.22
    cfg.rewards.track_ang_vel_z_exp.weight = 20
    cfg.rewards.track_lin_vel_xy_exp.params["std"] = (
        0.22  # TODO should this be ang_vel?
    )

    # cfg.rewards.feet_air_time.weight = (
    #     100  # consider reducing this to 7.5 if performance on task reward is bad; do not use for ResRL
    # )
    # cfg.rewards.flat_orientation_l2.weight = -25
    # cfg.rewards.feet_slide.weight = -5.0
    cfg.rewards.dof_torques_l2.weight = -0.006
    cfg.rewards.torque_limits.weight = -35
    cfg.rewards.torque_limits_2.weight = -100
    # cfg.rewards.dof_acc_l2.weight = -5e-6

    # cfg.rewards.undesired_contacts_thigh.weight = -1.0
    # cfg.rewards.undesired_contacts_calf.weight = -1.0
    # cfg.rewards.contact_forces.weight = -1.0

def set_box_rewards_amp(cfg):
    # disable rewards
    for field in fields(cfg.rewards):
        reward_obj = getattr(cfg.rewards, field.name)
        # we need to check this because some reward terms might have been deleted previously depending on the environment and task
        if reward_obj is not None:
            reward_obj.weight = 0.0

    # set only task reward
    cfg.rewards.track_lin_vel_xy_exp.weight = 60
    cfg.rewards.track_lin_vel_xy_exp.params["std"] = 0.22
    cfg.rewards.track_ang_vel_z_exp.weight = 20
    cfg.rewards.track_lin_vel_xy_exp.params["std"] = (
        0.22  # TODO should this be ang_vel?
    )
    
    assert False, "Rewards"
    # cfg.rewards.feet_air_time.weight = (
    #     100  # consider reducing this to 7.5 if performance on task reward is bad; do not use for ResRL
    # )
    # # cfg.rewards.flat_orientation_l2.weight = -25
    # cfg.rewards.feet_slide.weight = -5.0
    cfg.rewards.dof_torques_l2.weight = -0.006
    cfg.rewards.torque_limits.weight = -35
    cfg.rewards.torque_limits_2.weight = -100
    # cfg.rewards.dof_acc_l2.weight = -5e-6
    
    # cfg.rewards.undesired_contacts_thigh.weight = -1.0
    # cfg.rewards.undesired_contacts_calf.weight = -1.0
    # cfg.rewards.contact_forces.weight = -1.0


def set_pose2d_rewards_amp(cfg):
    # disable rewards
    for field in fields(cfg.rewards):
        reward_obj = getattr(cfg.rewards, field.name)
        # we need to check this because some reward terms might have been deleted previously depending on the environment and task
        if reward_obj is not None:
            reward_obj.weight = 0.0

    # set only task reward
    cfg.rewards.orientation_tracking.weight = -20
    cfg.rewards.position_tracking.weight = 40
    cfg.rewards.position_tracking_fine_grained.weight = 30


def set_rewards_complex(cfg):
    cfg.rewards.lin_vel_z_l2.weight = -2.0
    cfg.rewards.ang_vel_xy_l2.weight = -0.05
    cfg.rewards.dof_torques_l2.weight = -0.0002
    cfg.rewards.dof_acc_l2.weight = -2.5e-7  # do not use for ResRL
    cfg.rewards.action_rate_l2.weight = -0.01
    cfg.rewards.feet_air_time.weight = (
        10  # consider reducing this to 7.5 if performance on task reward is bad; do not use for ResRL
    )
    cfg.rewards.undesired_contacts_thigh.weight = -1.0
    cfg.rewards.undesired_contacts_calf.weight = -1.0
    cfg.rewards.contact_forces.weight = -1.0
    cfg.rewards.flat_orientation_l2.weight = -0.01
    cfg.rewards.joint_pos_limits.weight = -10.0 # do not use for ResRL
    cfg.rewards.torque_limits.weight = -1.0e-5
    cfg.rewards.joint_deviation_l1.weight = (
        -0.75
    )  # consider reducing this in case performance on task reward is bad
    
    cfg.rewards.feet_slide.weight = -0.05
    # TODO: desired base height


def set_rewards_standing_complex(cfg):
    # cfg.rewards.lin_vel_z_l2.weight = -2.0
    # cfg.rewards.ang_vel_xy_l2.weight = -0.05
    cfg.rewards.dof_torques_l2.weight = 0.1 * -0.0002
    cfg.rewards.dof_acc_l2.weight =  0.02 * -2.5e-7  # do not use for ResRL
    cfg.rewards.action_rate_l2.weight = 0.1 * -0.01
    cfg.rewards.feet_air_time.weight = (
        0.1 * 10  # consider reducing this to 7.5 if performance on task reward is bad; do not use for ResRL
    )
    # # cfg.rewards.undesired_contacts_thigh.weight = -1.0
    # cfg.rewards.undesired_contacts_calf.weight = -1.0
    cfg.rewards.contact_forces.weight = 0.1 *-1.0
    # cfg.rewards.flat_orientation_l2.weight = -0.01
    cfg.rewards.joint_pos_limits.weight = 0.1 * -10.0 # do not use for ResRL
    cfg.rewards.torque_limits.weight = 0.1 * -1.0e-5
    cfg.rewards.joint_deviation_l1.weight = (
        0.1 * -0.25
    )  # consider reducing this in case performance on task reward is bad


def set_stairs_env_cfg_cmds(cfg):
    cfg.commands.base_velocity = mdp.Global3DUniformVelocityCommandCfg(
        # inherit parameters where possible
        asset_name=cfg.commands.base_velocity.asset_name,
        resampling_time_range=cfg.commands.base_velocity.resampling_time_range,
        rel_standing_envs=cfg.commands.base_velocity.rel_standing_envs,
        rel_heading_envs=cfg.commands.base_velocity.rel_heading_envs,
        heading_command=cfg.commands.base_velocity.heading_command,
        heading_control_stiffness=cfg.commands.base_velocity.heading_control_stiffness,
        debug_vis=cfg.commands.base_velocity.debug_vis,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(0.2, 0.7),
            ang_vel_z=(0, 0),
            heading=(
                math.pi / 2 - math.radians(20),
                math.pi / 2 + math.radians(20),
            ),  # global heading "up the stairs" is in y direction, which is math.pi/2
        ),
    )


def set_box_env_cfg_cmds(cfg):
    cfg.commands.base_velocity = mdp.Global3DUniformVelocityCommandCfg(
        # inherit parameters where possible
        asset_name=cfg.commands.base_velocity.asset_name,
        resampling_time_range=cfg.commands.base_velocity.resampling_time_range,
        rel_standing_envs=cfg.commands.base_velocity.rel_standing_envs,
        rel_heading_envs=cfg.commands.base_velocity.rel_heading_envs,
        heading_command=cfg.commands.base_velocity.heading_command,
        heading_control_stiffness=cfg.commands.base_velocity.heading_control_stiffness,
        debug_vis=cfg.commands.base_velocity.debug_vis,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(0.4, 0.8),
            ang_vel_z=(0, 0),
            heading=(
                math.pi / 2, #- math.radians(20),
                math.pi / 2, #+ math.radians(20),
            ),  # global heading "up the stairs" is in y direction, which is math.pi/2
        ),
    )

    # TODO this observation should be added
    # cfg.observations.policy.root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w, noise=Unoise(n_min=-0.1, n_max=0.1))

    # PoseTracking Command
    # command
    # cfg.commands.base_velocity = None
    # # NOTE could add z height to command to guidance during learning
    # cfg.commands.pose_command = mdp.UniformPose2dCommandCfg(
    #     asset_name="robot",
    #     simple_heading=True,
    #     resampling_time_range=(20.0, 20.0), # >= episode length = no resampling
    #     debug_vis=True,
    #     ranges=mdp.UniformPose2dCommandCfg.Ranges(
    #         pos_x=(0.0, 0.0),
    #         pos_y=(2.0, 2.0),
    #         heading=(
    #             math.pi / 2, # - math.radians(20),
    #             math.pi / 2, # + math.radians(20),
    #         ),  # global heading "up the stairs" is in y direction, which is math.pi/2
    #     ),
    # )

    # # observation
    # cfg.observations.policy.velocity_commands = None
    # cfg.observations.policy.pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

    # # reward; omit orientation tracking for now
    # cfg.rewards.track_lin_vel_xy_exp = None
    # cfg.rewards.track_ang_vel_z_exp = None
    # cfg.rewards.position_tracking = RewTerm(
    #     func=position_command_error_tanh,
    #     weight=30,
    #     params={"std": 2.0, "command_name": "pose_command"},
    # )
    # cfg.rewards.position_tracking_fine_grained = RewTerm(
    #     func=position_command_error_tanh,
    #     weight=30,
    #     params={"std": 0.2, "command_name": "pose_command"},
    # )
    # cfg.rewards.orientation_tracking = RewTerm(
    #     func=heading_command_error_abs,
    #     weight=-20,
    #     params={"command_name": "pose_command"},
    # )

    # assert cfg.curriculum.terrain_levels == None, "Curriculum not supported for box environment as it is velocity dependent in the base environment config (vel_env_cfg.py).


def set_stairs_env_cfg_reset_base(cfg):
    cfg.events.reset_base.params["pose_range"] = {
        "x": (-0.5, 0.5),
        "y": (-0.1, 0.1),
        "yaw": (math.pi / 2 - math.radians(20), math.pi / 2 + math.radians(20)),
    }

def set_box_env_cfg_reset_base(cfg):
    cfg.events.reset_base.params["pose_range"] = {
                # "x": (-0.0, 0.0),
                # "y": (0.04, 0.06),
                # "yaw": (math.pi / 2, math.pi / 2),
                "x": (-0.5, 0.5),
                "y": (-0.25, 0.15),
                "yaw": (math.pi / 2, math.pi / 2),
                # "yaw": (math.pi / 2 - math.radians(20), math.pi / 2 + math.radians(20)),
            }

def add_relative_position_on_stairs_observation(cfg):
    cfg.observations.policy.relative_position = ObsTerm(
        func=mdp.relative_position_on_stairs,
        noise=USinPosNoise(n_min=-0.1, n_max=0.1),
    )
    cfg.observations.policy.yaw = ObsTerm(
        func=mdp.yaw,
        noise=USinPosNoise(n_min=-0.05 * 2 * math.pi, n_max=0.05 * 2 * math.pi),
    )
    cfg.observations.policy.is_on_stairs = ObsTerm(
        func=mdp.distance_to_stairs,
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )

def add_relative_position_to_box_observation(cfg):
    cfg.observations.policy.relative_position_to_box = ObsTerm(func=mdp.relative_position_to_box, noise=Unoise(n_min=-0.02, n_max=0.02))

def add_stair_parameters_observation(cfg):
    cfg.observations.policy.stair_parameters = ObsTerm(
        func=mdp.stair_parameters,
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )

def add_box_parameters_observation(cfg):
    cfg.observations.policy.box_parameters = ObsTerm(func=mdp.box_parameters, noise=Unoise(n_min=-0.01, n_max=0.01))


def set_amp_settings(cfg, motion_folder="datasets/fromVision_motions_3/*", **kwargs):
    cfg.is_amp_env = True

    cfg.amp_motion_folder = motion_folder
    cfg.amp_motion_files = glob.glob(cfg.amp_motion_folder)
    

    use_rsi = kwargs.pop("use_rsi", True) # you can use this key to disable rsi
    params = {
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "device": cfg.sim.device,
        "time_between_frames": cfg.decimation * cfg.sim.dt,
        "motion_files": cfg.amp_motion_files, # by default RSI motion files are equal to AMP motion files. Overwrite this using the argument **kwargs!
    }
    params.update(kwargs)
    
    # use reference state initialization
    if use_rsi:
        cfg.events.reset_robot_joints = None
        cfg.events.reference_state_initialization = EventTerm(
            func=mdp.reference_state_initialization, mode="reset", params=params
        )


# NOTE this function keeps track of DR params that were used to train previous policies. Consider this function legacy. It should only be used if you know what you are doing.
def previous_domain_randomization_params(cfg):
    cfg.events.physics_material.params["static_friction_range"] = (0.6, 3.0)
    cfg.events.physics_material.params["dynamic_friction_range"] = (0.6, 1.2)
    cfg.events.physics_material.params["restitution_range"] = (0.0, 0.1)
    cfg.events.randomize_link_mass.params["mass_distribution_params"] = (0.9, 1.1)
    cfg.events.push_robot.params["velocity_range"] = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
    }
    cfg.events.base_com = None
    cfg.events.links_com = None
    cfg.events.reset_gravity = None
    cfg.events.actuator_gains.params["stiffness_distribution_params"] = (0.8, 1.2)
    cfg.events.actuator_gains.params["damping_distribution_params"] = (0.8, 1.2)

def standing_domain_randomization(cfg):
    cfg.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 2.0)
    cfg.events.base_com.params["com_range"]["z"] = (-0.02, 0.04) # robot false to the bag - so higher COM in z should help


def disable_domain_randomization(cfg):
    cfg.scene.robot.actuators["base_legs"].min_delay = 0
    cfg.scene.robot.actuators["base_legs"].max_delay = 0
    cfg.events.push_robot = None
    cfg.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 2.0)
    if cfg.events.reset_robot_joints is not None: # its None for AMP
        cfg.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
    cfg.events.reset_base.params = {
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0.0, 0.0),
            "pitch": (-0.0, 0.0),
            "yaw": (-0.0, 0.0),
        },
    }

    cfg.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
    cfg.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
    cfg.events.physics_material.params["restitution_range"] = (0.0, 0.0)
    cfg.events.randomize_link_mass = None
    cfg.events.actuator_gains = None
    cfg.events.joint_limits = None
    cfg.events.base_com = None
    cfg.events.links_com = None
    cfg.events.reset_gravity = None
    
    cfg.disable_domain_randomization = True

    print("[INFO] Domain Randomization disabled. Note that you can probably train with much lower max_iterations compared to when using Domain Randomization.")

    assert not cfg.terrain_type == "flat_noisy", "flat_noisy is only for Domain Randomization."

def zero_domain_randomization(cfg):
    cfg.scene.robot.actuators["base_legs"].min_delay = 0
    cfg.scene.robot.actuators["base_legs"].max_delay = 0
    cfg.events.push_robot = None
    cfg.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
    if cfg.events.reset_robot_joints is not None: # its None for AMP
        cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0.0, 0.0),
            "pitch": (-0.0, 0.0),
            "yaw": (-0.0, 0.0),
        },
    }

    cfg.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
    cfg.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
    cfg.events.physics_material.params["restitution_range"] = (0.0, 0.0)
    cfg.events.randomize_link_mass = None
    cfg.events.add_base_mass = None
    cfg.events.actuator_gains = None
    cfg.events.joint_limits = None
    cfg.events.base_com = None
    cfg.events.links_com = None
    cfg.events.reset_gravity = None
    
    cfg.disable_domain_randomization = True

    print("[INFO] Zero Domain Randomization. This setting is not meant for any DRL training.")

    assert not cfg.terrain_type == "flat_noisy", "flat_noisy is only for Domain Randomization."

def set_standing_env_terminations(cfg):
    cfg.terminations.bad_orientation = None
    cfg.terminations.head_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_upper", "Head_lower"]), "threshold": 1.0},
    )
    cfg.terminations.hip_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_hip", "FR_hip", "RL_hip", "RR_hip"]), "threshold": 1.0},
    )
    cfg.terminations.root_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.15},
    )
