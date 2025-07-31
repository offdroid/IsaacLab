# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script replays joint positions.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/quadrupeds.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates different legged robots."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
import json
import time

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.utils.math as math_utils


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation

from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.utils import utils

from omni.isaac.lab.terrains.config.stairs import STAIRS_TERRAINS_CFG  # isort:skip
from omni.isaac.lab.terrains.config.box import BOX_TERRAINS_CFG  # isort:skip
from omni.isaac.lab.terrains import TerrainImporter, TerrainImporterCfg

##
# Pre-defined configs
##

from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort:skip

UNITREE_GO2_CFG.spawn.rigid_props.disable_gravity = True

scene = "stairs"

# Recorded jpos path
recording_path = "datasets/fromVision_motions_DepthCamStairs/stairs_1_5199540000_amp.txt"  # "datasets/fromVision_motions/fromVision_amp.txt" || datasets/mocap_motions/trot2_amp.txt

freq = 0.2  # replay frequency in Hz for the recorded trajectory


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_cols = np.floor(np.sqrt(num_origins))
    num_rows = np.ceil(num_origins / num_cols)
    xx, yy = torch.meshgrid(
        torch.arange(num_rows), torch.arange(num_cols), indexing="xy"
    )
    env_origins[:, 0] = (
        spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    )
    env_origins[:, 1] = (
        spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    )
    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a mount and a robot on top of it
    origins = define_origins(num_origins=1, spacing=1.25)

    # Origin with Unitree Go2
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # -- Robot
    unitree_go2 = Articulation(
        UNITREE_GO2_CFG.replace(prim_path="/World/Origin1/Robot")
    )

    # return the scene information
    scene_entities = {
        "unitree_go2": unitree_go2,
    }
    return scene_entities, origins


def design_stairs_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    terrain_cfg = STAIRS_TERRAINS_CFG
    terrain_cfg.num_rows = 1
    terrain_cfg.num_cols = 1
    terrain_cfg.sub_terrains["stairs"].step_height_range = (0.14, 0.14)
    terrain_cfg.sub_terrains["stairs"].step_width = None

    # Handler for terrains importing
    terrain_importer_cfg = TerrainImporterCfg(
        num_envs=1,
        env_spacing=3.0,
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=terrain_cfg,
        debug_vis=True,
    )
    terrain_importer = TerrainImporter(terrain_importer_cfg)

    # Origin with Unitree Go2
    prim_utils.create_prim(
        "/World/Origin1", "Xform", translation=terrain_importer.env_origins[0]
    )
    # -- Robot
    unitree_go2 = Articulation(
        UNITREE_GO2_CFG.replace(prim_path="/World/Origin1/Robot")
    )

    # return the scene information
    scene_entities = {
        "unitree_go2": unitree_go2,
        "terrain": terrain_importer,
    }

    return scene_entities, terrain_importer.env_origins

def design_box_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    terrain_cfg = BOX_TERRAINS_CFG
    terrain_cfg.sub_terrains["box"].box_height_range = (box_height, box_height)
    terrain_cfg.num_rows = 1
    terrain_cfg.num_cols = 1


    # Handler for terrains importing
    terrain_importer_cfg = TerrainImporterCfg(
        num_envs=1,
        env_spacing=3.0,
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=terrain_cfg,
        debug_vis=True,
    )
    terrain_importer = TerrainImporter(terrain_importer_cfg)

    # Origin with Unitree Go2
    prim_utils.create_prim(
        "/World/Origin1", "Xform", translation=terrain_importer.env_origins[0]
    )
    # -- Robot
    unitree_go2 = Articulation(
        UNITREE_GO2_CFG.replace(prim_path="/World/Origin1/Robot")
    )

    # return the scene information
    scene_entities = {
        "unitree_go2": unitree_go2,
        "terrain": terrain_importer,
    }

    return scene_entities, terrain_importer.env_origins

def run_simulator(
    sim: sim_utils.SimulationContext,
    entities: dict[str, Articulation],
    origins: torch.Tensor,
    motion_data: torch.Tensor = None,
):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    global freq

    motion_data = torch.tensor(motion_data, device=sim.device)

    for index, robot in enumerate(entities.values()):
        if not isinstance(robot, Articulation):
            continue
        # root state
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += origins[index]
        robot.write_root_state_to_sim(root_state)
        # joint state
        joint_pos, joint_vel = (
            robot.data.default_joint_pos.clone(),
            robot.data.default_joint_vel.clone(),
        )
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        # reset the internal state
        robot.reset()

    freq = torch.tensor(freq, device=sim.device)
    phase = torch.tensor(0.0, device=sim.device)
    num_frames = torch.tensor(motion_data.shape[0], device=sim.device)

    # Simulate physics
    while simulation_app.is_running():
        start_time = time.time()

        # freq dependent phase
        phase = phase + sim_dt * 2 * torch.pi * freq

        # perform linear interpolation between two frames
        unscaled_index = (phase / (2 * torch.pi)) * num_frames
        idx0 = torch.floor(unscaled_index).long() % num_frames
        idx1 = (idx0 + 1) % num_frames
        
        # Check to not loop through from end to beginning of trajectory
        if not idx0 + 1 == idx1:
            assert idx0 == num_frames - 1
            idx1 = idx0
        
        
        alpha = (unscaled_index - torch.floor(unscaled_index)).unsqueeze(-1)

        pos_start = AMPLoader.get_root_pos(motion_data[idx0])
        pos_end = AMPLoader.get_root_pos(motion_data[idx1])
        rot_start = AMPLoader.get_root_rot(motion_data[idx0])
        rot_end = AMPLoader.get_root_rot(motion_data[idx1])
        jpos_start = AMPLoader.get_joint_pose(motion_data[idx0])
        jpos_end = AMPLoader.get_joint_pose(motion_data[idx1])
        

        pos_interpolated = AMPLoader.slerp(pos_start, pos_end, alpha)
        rot_interpolated = utils.quaternion_slerp(rot_start.clone(), rot_end.clone(), alpha)

        jpos_interpolated = AMPLoader.slerp(jpos_start, jpos_end, alpha)
        # pos_interpolated_relative_to_origin = pos_interpolated - AMPLoader.get_root_pos(
        #     motion_data[0]
        # )

        # apply states to the robot
        for index, robot in enumerate(entities.values()):
            if not isinstance(robot, Articulation):
                continue
            robot.write_joint_state_to_sim(
                jpos_interpolated.clone(), torch.zeros_like(jpos_interpolated).clone()
            )
            
                                # Combine new root pose
            root_state = torch.cat(
                [
                    pos_interpolated + origins[0],
                    rot_interpolated,
                    torch.zeros_like(pos_start), # base lin vel
                    torch.zeros_like(pos_start), # base ang vel
                ],
                dim=-1,
            ).unsqueeze(0)

            robot.write_root_state_to_sim(root_state)

        # perform step
        sim.step()
        
        


        # update buffers
        for robot in entities.values():
            if not isinstance(robot, Articulation):
                continue
            robot.update(sim_dt)

        # print(f"Frame: {count % num_frames}")
        sleep_duration = sim_dt - (time.time() - start_time)
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        count += 1


def main():
    """Main function."""

    global recording_path, robot_local_offset

    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # design scene
    if scene == "stairs":
        scene_entities, scene_origins = design_stairs_scene()
    elif scene == "flat":
        scene_entities, scene_origins = design_scene()
        scene_origins = torch.tensor(scene_origins, device=sim.device)
    elif scene == "box":
        scene_entities, scene_origins = design_box_scene()
        scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()

    with open(recording_path, "r") as f:
        motion_json = json.load(f)
        
    recording_dt = float(motion_json["FrameDuration"])
    assert (
        recording_dt == 0.03334 or recording_dt == 0.01667 or recording_dt == 0.021
    )  # should be 30Hz (video) or 60Hz (mocap)
    # recording_dt *= 5 if recording_dt == 0.01667 else 5 # slow down a little
    
    motion_data = AMPLoader("cuda", recording_dt, motion_files=[recording_path], transform_root_trajectory=True)
    
    assert len(motion_data.trajectories_full) == 1, "Only support one motion file for replay."
    motion_data = motion_data.trajectories_full[0]
    
    lin_vel = AMPLoader.get_linear_vel_batch(motion_data)
    lin_vel = torch.tensor(lin_vel, device=sim.device)
    mean_speed = torch.norm(lin_vel, dim=1).mean().item()
    print(f"[INFO]: Mean speed of the robot: {mean_speed:.2f} m/s")


    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins, motion_data)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
