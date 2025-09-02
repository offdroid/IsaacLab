# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from torch.distributions.exponential import Exponential
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.utils import DelayBuffer, LinearInterpolation
from omni.isaac.lab.utils.types import ArticulationActions

from .actuator_base import ActuatorBase

if TYPE_CHECKING:
    from .actuator_cfg import (
        DCMotorCfg,
        DelayedPDActuatorCfg,
        Delayed5GPDActuatorCfg,
        IdealPDActuatorCfg,
        ImplicitActuatorCfg,
        RemotizedPDActuatorCfg,
        DelayedDCMotorCfg,
    )


"""
Implicit Actuator Models.
"""


class ImplicitActuator(ActuatorBase):
    """Implicit actuator model that is handled by the simulation.

    This performs a similar function as the :class:`IdealPDActuator` class. However, the PD control is handled
    implicitly by the simulation which performs continuous-time integration of the PD control law. This is
    generally more accurate than the explicit PD control law used in :class:`IdealPDActuator` when the simulation
    time-step is large.

    .. note::

        The articulation class sets the stiffness and damping parameters from the configuration into the simulation.
        Thus, the parameters are not used in this class.

    .. caution::

        The class is only provided for consistency with the other actuator models. It does not implement any
        functionality and should not be used. All values should be set to the simulation directly.
    """

    cfg: ImplicitActuatorCfg
    """The configuration for the actuator model."""

    """
    Operations.
    """

    def reset(self, *args, **kwargs):
        # This is a no-op. There is no state to reset for implicit actuators.
        pass

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """Process the actuator group actions and compute the articulation actions.

        In case of implicit actuator, the control action is directly returned as the computed action.
        This function is a no-op and does not perform any computation on the input control action.
        However, it computes the approximate torques for the actuated joint since PhysX does not compute
        this quantity explicitly.

        Args:
            control_action: The joint action instance comprising of the desired joint positions, joint velocities
                and (feed-forward) joint efforts.
            joint_pos: The current joint positions of the joints in the group. Shape is (num_envs, num_joints).
            joint_vel: The current joint velocities of the joints in the group. Shape is (num_envs, num_joints).

        Returns:
            The computed desired joint positions, joint velocities and joint efforts.
        """
        # store approximate torques for reward computation
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)
        return control_action


"""
Explicit Actuator Models.
"""


class IdealPDActuator(ActuatorBase):
    r"""Ideal torque-controlled actuator model with a simple saturation model.

    It employs the following model for computing torques for the actuated joint :math:`j`:

    .. math::

        \tau_{j, computed} = k_p * (q - q_{des}) + k_d * (\dot{q} - \dot{q}_{des}) + \tau_{ff}

    where, :math:`k_p` and :math:`k_d` are joint stiffness and damping gains, :math:`q` and :math:`\dot{q}`
    are the current joint positions and velocities, :math:`q_{des}`, :math:`\dot{q}_{des}` and :math:`\tau_{ff}`
    are the desired joint positions, velocities and torques commands.

    The clipping model is based on the maximum torque applied by the motor. It is implemented as:

    .. math::

        \tau_{j, max} & = \gamma \times \tau_{motor, max} \\
        \tau_{j, applied} & = clip(\tau_{computed}, -\tau_{j, max}, \tau_{j, max})

    where the clipping function is defined as :math:`clip(x, x_{min}, x_{max}) = min(max(x, x_{min}), x_{max})`.
    The parameters :math:`\gamma` is the gear ratio of the gear box connecting the motor and the actuated joint ends,
    and :math:`\tau_{motor, max}` is the maximum motor effort possible. These parameters are read from
    the configuration instance passed to the class.
    """

    cfg: IdealPDActuatorCfg
    """The configuration for the actuator model."""

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int]):
        pass

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # compute errors
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        # calculate the desired joint torques
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)
        # set the computed actions back into the control action
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


class DCMotor(IdealPDActuator):
    r"""Direct control (DC) motor actuator model with velocity-based saturation model.

    It uses the same model as the :class:`IdealActuator` for computing the torques from input commands.
    However, it implements a saturation model defined by DC motor characteristics.

    A DC motor is a type of electric motor that is powered by direct current electricity. In most cases,
    the motor is connected to a constant source of voltage supply, and the current is controlled by a rheostat.
    Depending on various design factors such as windings and materials, the motor can draw a limited maximum power
    from the electronic source, which limits the produced motor torque and speed.

    A DC motor characteristics are defined by the following parameters:

    * Continuous-rated speed (:math:`\dot{q}_{motor, max}`) : The maximum-rated speed of the motor.
    * Continuous-stall torque (:math:`\tau_{motor, max}`): The maximum-rated torque produced at 0 speed.
    * Saturation torque (:math:`\tau_{motor, sat}`): The maximum torque that can be outputted for a short period.

    Based on these parameters, the instantaneous minimum and maximum torques are defined as follows:

    .. math::

        \tau_{j, max}(\dot{q}) & = clip \left (\tau_{j, sat} \times \left(1 -
            \frac{\dot{q}}{\dot{q}_{j, max}}\right), 0.0, \tau_{j, max} \right) \\
        \tau_{j, min}(\dot{q}) & = clip \left (\tau_{j, sat} \times \left( -1 -
            \frac{\dot{q}}{\dot{q}_{j, max}}\right), - \tau_{j, max}, 0.0 \right)

    where :math:`\gamma` is the gear ratio of the gear box connecting the motor and the actuated joint ends,
    :math:`\dot{q}_{j, max} = \gamma^{-1} \times  \dot{q}_{motor, max}`, :math:`\tau_{j, max} =
    \gamma \times \tau_{motor, max}` and :math:`\tau_{j, peak} = \gamma \times \tau_{motor, peak}`
    are the maximum joint velocity, maximum joint torque and peak torque, respectively. These parameters
    are read from the configuration instance passed to the class.

    Using these values, the computed torques are clipped to the minimum and maximum values based on the
    instantaneous joint velocity:

    .. math::

        \tau_{j, applied} = clip(\tau_{computed}, \tau_{j, min}(\dot{q}), \tau_{j, max}(\dot{q}))

    """

    cfg: DCMotorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # parse configuration
        if self.cfg.saturation_effort is not None:
            self._saturation_effort = self.cfg.saturation_effort
        else:
            self._saturation_effort = torch.inf
        self.clip_effort_factor = self.cfg.clip_effort_factor
        assert self.clip_effort_factor is None or self.clip_effort_factor > 0, "The clip effort factor must be greater than 0. You can use 'None' if you dont want to perform any clipping."
        # prepare joint vel buffer for max effort computation
        self._joint_vel = torch.zeros_like(self.computed_effort)
        # create buffer for zeros effort
        self._zeros_effort = torch.zeros_like(self.computed_effort)
        # check that quantities are provided
        if self.cfg.velocity_limit is None:
            raise ValueError("The velocity limit must be provided for the DC motor actuator model.")

        self.effort_log = []

    """
    Operations.
    """

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # save current joint vel
        self._joint_vel[:] = joint_vel
        # calculate the desired joint torques
        return super().compute(control_action, joint_pos, joint_vel)

    """
    Helper functions.
    """

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        if self.clip_effort_factor is None:
            # we try to not clip the effort explicitely, as this is also not done in DOOM
            return effort
        
        # compute torque limits
        # -- max limit
        max_effort = self.clip_effort_factor * self._saturation_effort * (1.0 - self._joint_vel / self.velocity_limit)
        max_effort = torch.clip(max_effort, min=self._zeros_effort, max=self.effort_limit)
        # -- min limit
        min_effort = self.clip_effort_factor * self._saturation_effort * (-1.0 - self._joint_vel / self.velocity_limit)
        min_effort = torch.clip(min_effort, min=-self.effort_limit, max=self._zeros_effort)

        # clip the torques based on the motor limits
        # self.effort_log.append(
        #     torch.clip(effort, min=min_effort, max=max_effort).cpu().numpy().tolist()
        # )

        # efforts = np.array(self.effort_log).squeeze(axis=1)  # shape: (523, 12)

        # joint_names = [
        #     'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
        #     'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
        #     'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'
        # ]

        # timesteps = np.arange(efforts.shape[0])
        # num_joints = efforts.shape[1]

        # fig, axes = plt.subplots(num_joints, 1, figsize=(10, 2 * num_joints), sharex=True)

        # for i in range(num_joints):
        #     axes[i].plot(timesteps, efforts[:, i])
        #     axes[i].set_ylabel(joint_names[i])
        #     axes[i].grid(True)

        # axes[-1].set_xlabel('Timestep')
        # fig.suptitle("Robot Joint Efforts Over Time (clipped)", fontsize=16)
        # fig.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for the title
        # plt.savefig("effortsClipped_ComplexRewardFlat.pdf")
        # effort_list = effort[0].cpu().tolist()
        # rounded_effort = [round(val) for val in effort_list]
        # print(rounded_effort)

        return torch.clip(effort, min=min_effort, max=max_effort)

class DelayedPDActuator(IdealPDActuator):
    """Ideal PD actuator with delayed command application.

    This class extends the :class:`IdealPDActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
    applied to the simulation is lagged by a certain number of physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
    to the class.
    """

    cfg: DelayedPDActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DelayedPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # all of the envs
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # compte actuator model
        return super().compute(control_action, joint_pos, joint_vel)
    
    
class Delayed5GPDActuator(IdealPDActuator):
    """Ideal PD actuator with per-command stochastic delay.

    This class simulates a network with variable latency. For each control command
    sent at every physics step, a new delay is sampled from a shifted exponential
    distribution. The command is then scheduled for application in a future step.

    This model handles out-of-order command arrivals by always applying the most
    recently issued command that is ready for execution.
    """

    cfg: Delayed5GPDActuatorCfg

    def __init__(self, cfg: Delayed5GPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        
        # -- Internal state for managing dynamic delays --

        # Calculate a safe maximum buffer size to handle the tail of the distribution.
        # This covers ~99.99% of samples.
        self.max_delay_steps = self.cfg.min_delay + int(9 / self.cfg.delay_rate)

        # Step counter for simulation time
        self.step_counter = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)

        # Buffers to store "in-flight" commands
        # The size is [max_delay, num_envs, num_actions]
        self._action_dim = 12
        self.positions_buffer = torch.zeros(self.max_delay_steps, self._num_envs, self._action_dim, device=self._device)
        self.velocities_buffer = torch.zeros(self.max_delay_steps, self._num_envs, self._action_dim, device=self._device)
        self.efforts_buffer = torch.zeros(self.max_delay_steps, self._num_envs, self._action_dim, device=self._device)

        # Buffer to store the simulation step when a command is scheduled to be released
        self.release_step_buffer = torch.full((self.max_delay_steps, self._num_envs), float('inf'), device=self._device)
        # Buffer to store the simulation step when a command was issued (to find the latest)
        self.issue_step_buffer = torch.full((self.max_delay_steps, self._num_envs), -1, device=self._device)

        # Store the last valid applied action, to use when no new command is ready
        self.last_applied_pos = torch.zeros(self._num_envs, self._action_dim, device=self._device)
        self.last_applied_vel = torch.zeros(self._num_envs, self._action_dim, device=self._device)
        self.last_applied_effort = torch.zeros(self._num_envs, self._action_dim, device=self._device)
        
        # Exponential distribution for sampling delays
        self.exp_dist = Exponential(rate=torch.tensor([self.cfg.delay_rate], device=self._device))

    def reset(self, env_ids: Sequence[int]):
        """Resets the internal state for the specified environments."""
        super().reset(env_ids)
        # Convert env_ids to a tensor for indexing
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self._num_envs, device=self._device)[env_ids]
        else:
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self._device)

        # Reset state for the selected environments
        self.step_counter[env_ids] = 0
        self.positions_buffer[:, env_ids] = 0
        self.velocities_buffer[:, env_ids] = 0
        self.efforts_buffer[:, env_ids] = 0
        self.release_step_buffer[:, env_ids] = float('inf')
        self.issue_step_buffer[:, env_ids] = -1
        self.last_applied_pos[env_ids] = 0
        self.last_applied_vel[env_ids] = 0
        self.last_applied_effort[env_ids] = 0


    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # Create a tensor [0, 1, 2, ..., num_envs-1] for vectorized indexing
        env_indices = torch.arange(self._num_envs, device=self._device)

        # -- 1. Schedule the new incoming command (Vectorized) --

        # Get the circular buffer index to write the new command to
        write_indices = self.step_counter % self.max_delay_steps

        # Sample additional delay from the exponential distribution
        additional_delays = self.exp_dist.sample(sample_shape=(self._num_envs,)).squeeze(-1)

        # Calculate total delay in steps and schedule the release time
        total_delays = (self.cfg.min_delay + additional_delays).round().long()
        release_steps = self.step_counter + total_delays

        # Store the new command and its metadata in the buffers using advanced indexing
        # This replaces the first for-loop by writing to all environments at once.
        self.positions_buffer[write_indices, env_indices] = control_action.joint_positions
        self.velocities_buffer[write_indices, env_indices] = control_action.joint_velocities
        self.efforts_buffer[write_indices, env_indices] = control_action.joint_efforts
        self.release_step_buffer[write_indices, env_indices] = release_steps.float()
        self.issue_step_buffer[write_indices, env_indices] = self.step_counter

        # -- 2. Find and select the command to apply for the current step (Vectorized) --

        # Find all commands in the buffer that are ready to be released
        is_ready_mask = self.release_step_buffer <= self.step_counter.unsqueeze(0)

        # To avoid applying old commands, we only want the *latest* ready command.
        # We set the issue step of not-ready commands to -1 so they are ignored by argmax.
        valid_issue_steps = torch.where(is_ready_mask, self.issue_step_buffer, -1)

        # Find the buffer index of the latest ready command for each environment
        latest_ready_indices = torch.argmax(valid_issue_steps, dim=0)

        # Check for which environments a valid command was found
        env_has_ready_cmd = torch.any(is_ready_mask, dim=0)

        # Gather the latest ready commands from the buffers using the found indices
        retrieved_pos = self.positions_buffer[latest_ready_indices, env_indices]
        retrieved_vel = self.velocities_buffer[latest_ready_indices, env_indices]
        retrieved_effort = self.efforts_buffer[latest_ready_indices, env_indices]

        # Use torch.where to select between the new command and the last applied one.
        # This is a vectorized equivalent of the if-statement inside the second for-loop.
        broadcast_mask = env_has_ready_cmd.unsqueeze(-1)
        final_pos = torch.where(broadcast_mask, retrieved_pos, self.last_applied_pos)
        final_vel = torch.where(broadcast_mask, retrieved_vel, self.last_applied_vel)
        final_effort = torch.where(broadcast_mask, retrieved_effort, self.last_applied_effort)

        # Invalidate the applied commands in the buffer to prevent re-application.
        # We only do this for environments that had a ready command.
        self.release_step_buffer[latest_ready_indices, env_indices] = torch.where(
            env_has_ready_cmd, float('inf'), self.release_step_buffer[latest_ready_indices, env_indices]
        )

        # Update the last applied action
        self.last_applied_pos = final_pos
        self.last_applied_vel = final_vel
        self.last_applied_effort = final_effort

        # Update the output action with the delayed commands
        control_action.joint_positions = final_pos
        control_action.joint_velocities = final_vel
        control_action.joint_efforts = final_effort

        # -- 3. Advance time and compute actuator physics --

        # Increment the step counter for the next iteration
        self.step_counter += 1

        # Compute the final actuator efforts using the (potentially delayed) action
        return super().compute(control_action, joint_pos, joint_vel)
    
class DelayedDCMotor(DCMotor):
    cfg: DelayedDCMotorCfg
    
    def __init__(self, cfg: DelayedDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # all of the envs
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)
        
    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)
        
    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # compte actuator model
        return super().compute(control_action, joint_pos, joint_vel)


class RemotizedPDActuator(DelayedPDActuator):
    """Ideal PD actuator with angle-dependent torque limits.

    This class extends the :class:`DelayedPDActuator` class by adding angle-dependent torque limits to the actuator.
    The torque limits are applied by querying a lookup table describing the relationship between the joint angle
    and the maximum output torque. The lookup table is provided in the configuration instance passed to the class.

    The torque limits are interpolated based on the current joint positions and applied to the actuator commands.
    """

    def __init__(
        self,
        cfg: RemotizedPDActuatorCfg,
        joint_names: list[str],
        joint_ids: Sequence[int],
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        armature: torch.Tensor | float = 0.0,
        friction: torch.Tensor | float = 0.0,
        effort_limit: torch.Tensor | float = torch.inf,
        velocity_limit: torch.Tensor | float = torch.inf,
    ):
        # remove effort and velocity box constraints from the base class
        cfg.effort_limit = torch.inf
        cfg.velocity_limit = torch.inf
        # call the base method and set default effort_limit and velocity_limit to inf
        super().__init__(
            cfg, joint_names, joint_ids, num_envs, device, stiffness, damping, armature, friction, torch.inf, torch.inf
        )
        self._joint_parameter_lookup = cfg.joint_parameter_lookup.to(device=device)
        # define remotized joint torque limit
        self._torque_limit = LinearInterpolation(self.angle_samples, self.max_torque_samples, device=device)

    """
    Properties.
    """

    @property
    def angle_samples(self) -> torch.Tensor:
        return self._joint_parameter_lookup[:, 0]

    @property
    def transmission_ratio_samples(self) -> torch.Tensor:
        return self._joint_parameter_lookup[:, 1]

    @property
    def max_torque_samples(self) -> torch.Tensor:
        return self._joint_parameter_lookup[:, 2]

    """
    Operations.
    """

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # call the base method
        control_action = super().compute(control_action, joint_pos, joint_vel)
        # compute the absolute torque limits for the current joint positions
        abs_torque_limits = self._torque_limit.compute(joint_pos)
        # apply the limits
        control_action.joint_efforts = torch.clamp(
            control_action.joint_efforts, min=-abs_torque_limits, max=abs_torque_limits
        )
        self.applied_effort = control_action.joint_efforts
        return control_action
