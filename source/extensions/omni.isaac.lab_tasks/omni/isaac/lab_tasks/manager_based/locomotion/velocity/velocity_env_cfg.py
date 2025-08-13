# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
from dataclasses import MISSING

from omni.isaac.lab.envs.mdp.rewards import joint_deviation_l1, joint_pos_limits, applied_torque_limits
from omni.isaac.lab.envs.mdp.terminations import bad_orientation, root_height_below_minimum
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG # isort: skip
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0), # Always keep that exactly 10.0, otherwise metric computation will be wrong. (It is wrong anyways if episodes terminate prematurely!)
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        # training
        # ranges=mdp.UniformVelocityCommandCfg.Ranges(
        #     lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        # ),
        # https://arxiv.org/pdf/2203.15103 (AMP make good substitutes for reward function) uses (-1,2), (-0.3, 0.3), (-1.57, + 1.57)
        # NOTE below target values are from AMP for hardware
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), # AMP for hardware has here (-1.0, 2.0)
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.57, 1.57),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)

    # joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=10.0) # torque control; also change Go2 config actuator damping and stiffness to 0.0!


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        
        # cmd terms
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        
        # state terms
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        # phases
        phases = None


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # "static_friction_range": (0.4, 1.0),
            # "dynamic_friction_range": (0.2, 0.8),
            # "restitution_range": (0.0, 0.1),
            # "static_friction_range": (2.0, 2.0),
            # "dynamic_friction_range": (1.0, 1.0),
            # "restitution_range": (0.0, 0.0),
            # "static_friction_range": (0.8, 0.8),
            # "dynamic_friction_range": (0.6, 0.6),
            # "restitution_range": (0.0, 0.0),
            "static_friction_range": (0.6, 3.0),
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 256,
        },
    )
    
    randomize_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*thigh", ".*calf", ".*hip"]),
            "mass_distribution_params": (0.9, 1.1),  # Small random mass variation
            "operation": "scale",  # Scale the default mass
            "distribution": "uniform",
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        # Doing this in mode "reset" slows down training ~25%
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        # params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}, # if DR doesnt work, potentially increase this parameter. For AMP for hardware this is +-1.3m/s
    )
    
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    
    # Randomizing gravity using 'randomize_physics_scene_gravity': Gravity is set for all environments,and should be used with 'mode=startup"'. Thus, you most likely dont want to use randomized gravity.

    
    joint_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        # Doing this in mode "reset" slows down training ~25%
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "lower_limit_distribution_params": (0.95, 1.05),
            "upper_limit_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    
    # NOTE do not use, seems buggy: https://github.com/isaac-sim/IsaacLab/issues/676
    # joint_frictions = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     # Doing this in mode "reset" slows down training ~25%
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "friction_distribution_params": (0.0, 0.05),
    #         "operation": "add",
    #         "distribution": "uniform",
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # NOTE all rewards are deactivated here and need to be activated in inheriting confings
    
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp_3d, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.0)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-0.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts_thigh = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*thigh"), "threshold": 1.0},
    )
    undesired_contacts_calf = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*calf"), "threshold": 1.0},
    )
    
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"), "threshold": 100.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.0)

    joint_deviation_l1 = RewTerm(
        func=joint_deviation_l1,
        weight=-100.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_joint"])},
    )
    
    joint_pos_limits = RewTerm(
        func=joint_pos_limits,
        weight=-0.0,
    )
    
    torque_limits = RewTerm(
        func=applied_torque_limits,
        weight=-0.0,
    )

    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.0,
        params={
            "target_height": -0.22,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_height": 0.3,
        },
    )

    # residual_action_l2 = RewTerm(
    #     func=mdp.residual_action_l2,
    #     weight=0.0,#-0.04,
    # )
    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.25,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    #     },
    # )
    # freq_rate_l2 = RewTerm(
    #     func=mdp.freq_rate_l2,
    #     weight=-0.1,
    # )
    # amp_rate_l2 = RewTerm(
    #     func=mdp.amp_rate_l2,
    #     weight=-0.05,
    # )

    # add power penalty: "We define the mechanical COT as: Power / Weight×Velocity. P, where τ is the joint torque, ˙θ is the motor velocity.

    # styles
    # style_jpos = RewTerm(func=mdp.style_jpos, weight=0.0, params={"factor": -2.0})
    # style_jvel = RewTerm(func=mdp.style_jvel, weight=0.0, params={"factor": -0.1})
    # TODO add foot z-height style penalty
    
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-0.0, params={"target_height": 0.4})
    
    head_height_l2 = RewTerm(func=mdp.head_height_l2, weight=-0.0)
    feet_height_l2 = RewTerm(func=mdp.feet_height_l2, weight=-0.0)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    calf_contact1 = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*calf"), "threshold": 1.0},
    )
    tigh_contact= DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*thigh"), "threshold": 1.0},
    )
    lower_head_contact= DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Head_lower"), "threshold": 1.0},
    )
    hip_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_hip"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": torch.pi/2})
    # joint_pos_out_of_limits = DoneTerm(
    #     func=mdp.joint_pos_out_of_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot")}
    # )
    root_height_below_minimum = DoneTerm(
        func=root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.15}
    )
    # bad_orientation = DoneTerm(
    #     func=bad_orientation,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.4},
    # )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    
    action_manager_class: str = "ActionManager"

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=2*4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    is_eval_env: bool = False

    def __post_init__(self):
        """Post initialization."""
        
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
