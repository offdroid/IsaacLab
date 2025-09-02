# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script generates and replays joint positions for a quadruped using
open-loop oscillators and an analytical inverse kinematics solver.

The physics simulation is disabled to purely visualize the kinematic output.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/workflows/replay_oscillator_gait.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import time
import torch
from collections.abc import Sequence

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script replays a generated oscillator-based gait on the Unitree Go2 robot."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation

# Pre-defined configs
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort:skip


class OscillatorGaitGenerator:
    """
    Generates a rhythmic trotting gait using oscillators and analytical IK.
    """

    def __init__(self, num_envs: int, device: str = "cpu"):
        self.num_envs = num_envs
        self.device = device

        # --- Oscillator Parameters ---
        self.phase = torch.zeros(self.num_envs, 1, device=self.device)
        self.frequency = .1  # rad/s, controls the speed of the gait (reduced for smoother motion)

        # Phase shifts for a trotting gait (diagonal pairs FR/RL and FL/RR move together)
        self.phase_shifts = torch.tensor([[torch.pi, 0.0, 0.0, torch.pi]], device=self.device)

        # Hip positions relative to base frame (based on Unitree Go2 dimensions)
        self.hip_pos = torch.tensor(
            [
                [0.188, 0.05, 0.0],   # FL
                [0.188, -0.05, 0.0],  # FR
                [-0.188, 0.05, 0.0],  # RL
                [-0.188, -0.05, 0.0], # RR
            ],
            device=self.device,
        ).repeat(self.num_envs, 1, 1)

        # Offsets from the base frame to the neutral foot position (aligned under hips)
        self.default_foot_pos = self.hip_pos.clone()
        self.default_foot_pos[..., 2] = -0.3

        # Oscillator amplitudes for step motion
        self.step_length_amp = 0.0 # m (reduced for stability)
        self.step_height_amp = 0.1  # m (reduced for stability)

        # --- Kinematic Parameters ---
        self.thigh_length = 0.213  # meters
        self.calf_length = 0.213   # meters

    def step(self, dt: float) -> torch.Tensor:
        """
        Advances the gait by one time step `dt` and returns target joint positions.

        Args:
            dt: The simulation time step.

        Returns:
            A tensor of shape (num_envs, 12) containing the target joint angles.
        """
        # 1. Update the phase and compute target foot positions from the oscillator (in body frame)
        foot_positions = self._compute_foot_positions(dt)

        # 2. Convert to local positions relative to each hip
        local_foot_positions = foot_positions - self.hip_pos

        # 3. Compute target joint angles using Analytical Inverse Kinematics
        joint_positions = self._solve_analytical_ik(local_foot_positions)

        return joint_positions

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.phase[env_ids] = 0.0

    def _compute_foot_positions(self, dt: float) -> torch.Tensor:
        """Calculates target foot positions based on the current phase."""
        # Update the global phase
        self.phase += 2*torch.pi * self.frequency * dt

        # Calculate cycle phase for each foot (shape: [num_envs, 4])
        cycle_phase = (self.phase + self.phase_shifts) % (2 * torch.pi)

        # Determine if a foot is in swing or stance phase
        is_swing_phase = cycle_phase < torch.pi

        # Calculate target position deltas for swing phase
        x_swing = -self.step_length_amp * torch.cos(cycle_phase)
        z_swing = self.step_height_amp * torch.abs(torch.sin(cycle_phase))  # Always positive for lifting

        # Calculate target position deltas for stance phase
        x_stance = -self.step_length_amp * torch.cos(cycle_phase)
        z_stance = torch.zeros_like(x_stance)

        # Combine using the condition
        delta_x = torch.where(is_swing_phase, x_swing, x_stance)
        delta_z = torch.where(is_swing_phase, z_swing, z_stance)

        # Add the oscillator deltas to the default foot positions
        foot_positions = self.default_foot_pos.clone()
        foot_positions[..., 0] += delta_x  # Add to X-coordinate
        foot_positions[..., 2] += delta_z  # Add to Z-coordinate
        print(foot_positions[0])

        return foot_positions

    def _solve_analytical_ik(self, foot_positions: torch.Tensor) -> torch.Tensor:
        """Solves inverse kinematics for the Go2's legs."""
        # Extract foot positions relative to hip joints
        x = foot_positions[..., 0]  # Shape: (num_envs, 4)
        y = foot_positions[..., 1]  # Shape: (num_envs, 4) - lateral offset
        z = foot_positions[..., 2]  # Shape: (num_envs, 4)

        # --- Hip Abduction/Adduction Joint (HAA) ---
        # The HAA angle depends on the foot's position in the YZ plane.
        # The angle is computed from the negative z-axis towards the y-axis.
        hip_joint_angle = torch.deg2rad(torch.tensor([[5,-5,5,-5]], device="cuda")).repeat(x.shape[0], 1)

        # --- Knee and Thigh IK in the leg's sagittal plane ---
        # Use the true 3D distance from the hip to the foot (D) for the law of cosines.
        D_sq = x**2 + y**2 + z**2

        # Knee angle calculation using law of cosines
        cos_theta2_arg = torch.clamp(
            (self.thigh_length**2 + self.calf_length**2 - D_sq) / (2 * self.thigh_length * self.calf_length),
            -1.0,
            1.0,
        )
        theta2 = torch.acos(cos_theta2_arg)

        # The Go2 URDF defines the calf joint angle relative to a straight leg (theta2 = pi, joint angle = 0).
        calf_joint_angle = theta2 - torch.pi

        # Thigh angle calculation
        # Fix: Adjust alpha to ensure forward motion by using positive sqrt(y^2 + z^2).
        alpha = torch.atan2(torch.sqrt(y**2 + z**2), x)  # Changed -torch.sqrt to +torch.sqrt
        beta = torch.atan2(
            self.calf_length * torch.sin(theta2),
            self.thigh_length + self.calf_length * torch.cos(theta2)
        )
        thigh_joint_angle = alpha - beta # + torch.pi / 2 # Changed to alpha + beta to align with forward motion

        # Assemble the final joint angles tensor in Isaac Lab's breadth-first order
        target_joint_pos = torch.stack(
            [
                hip_joint_angle[:, 0],   # FL_HAA
                hip_joint_angle[:, 1],   # FR_HAA
                hip_joint_angle[:, 2],   # RL_HAA
                hip_joint_angle[:, 3],   # RR_HAA
                thigh_joint_angle[:, 0], # FL_HFE
                thigh_joint_angle[:, 1], # FR_HFE
                thigh_joint_angle[:, 2], # RL_HFE
                thigh_joint_angle[:, 3], # RR_HFE
                calf_joint_angle[:, 0],  # FL_KFE
                calf_joint_angle[:, 1],  # FR_KFE
                calf_joint_angle[:, 2],  # RL_KFE
                calf_joint_angle[:, 3],  # RR_KFE
            ],
            dim=1,
        )

        return target_joint_pos



def design_scene() -> dict:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robot
    # -- Disable gravity to have a pure kinematic replay
    UNITREE_GO2_CFG.spawn.rigid_props.disable_gravity = True
    unitree_go2 = Articulation(UNITREE_GO2_CFG.replace(prim_path="/World/Robot"))

    # Return the scene entities
    return {"unitree_go2": unitree_go2}


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    """Runs the simulation loop."""
    # Extract the robot from the entities
    robot = entities["unitree_go2"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Initialize the gait generator for 1 robot on the specified device
    gait_generator = OscillatorGaitGenerator(num_envs=1, device=sim.device)

    # Reset the robot to its default state
    robot.reset()
    # Place the robot at a stable starting height
    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] = 0.4  # Set a reasonable height
    robot.write_root_state_to_sim(root_state)
    default_joint_pos = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos))

    # Simulate
    while simulation_app.is_running():
        # Get start time
        start_time = time.time()

        # --- Generate and apply the gait ---
        # 1. Generate the target joint positions for this timestep
        target_joint_positions = gait_generator.step(sim_dt)

        # 2. Write the target joint positions directly to the robot
        #    We provide zero velocities as they are not needed for kinematic replay.
        robot.write_joint_state_to_sim(
            target_joint_positions, torch.zeros_like(target_joint_positions)
        )
        

        # --- Step the simulation ---
        # This step is primarily for rendering, as physics is not active.
        sim.step()

        # Update the robot's internal buffers
        robot.update(sim_dt)

        # --- Real-time playback ---
        # Sleep to maintain a real-time playback speed
        elapsed_time = time.time() - start_time
        sleep_duration = sim_dt - elapsed_time
        if sleep_duration > 0:
            time.sleep(sleep_duration)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0") # Use a slightly larger dt for smoother viz
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[2.0, 2.0, 1.0], target=[0.0, 0.0, 0.5])
    # Design scene
    scene_entities = design_scene()
    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete. Replaying generated oscillator gait...")
    # Run the simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
