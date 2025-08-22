# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action manager for processing actions sent to the environment."""

from __future__ import annotations

import yaml
import json
import inspect
import torch
import numpy as np
import weakref
from abc import abstractmethod
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import omni.kit.app

from omni.isaac.lab.assets import AssetBase

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ActionTermCfg


from rsl_rl.datasets.motion_loader import AMPLoader

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


class ActionTerm(ManagerTermBase):
    """Base class for action terms.

    The action term is responsible for processing the raw actions sent to the environment
    and applying them to the asset managed by the term. The action term is comprised of two
    operations:

    * Processing of actions: This operation is performed once per **environment step** and
      is responsible for pre-processing the raw actions sent to the environment.
    * Applying actions: This operation is performed once per **simulation step** and is
      responsible for applying the processed actions to the asset managed by the term.
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        # call the base class constructor
        super().__init__(cfg, env)
        # parse config to obtain asset to which the term is applied
        self._asset: AssetBase = self._env.scene[self.cfg.asset_name]

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def __del__(self):
        """Unsubscribe from the callbacks."""
        if self._debug_vis_handle:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    """
    Properties.
    """

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action term."""
        raise NotImplementedError

    @property
    @abstractmethod
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the term."""
        raise NotImplementedError

    @property
    @abstractmethod
    def processed_actions(self) -> torch.Tensor:
        """The actions computed by the term after applying any processing."""
        raise NotImplementedError

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the action term has a debug visualization implemented."""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    """
    Operations.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the action term data.
        Args:
            debug_vis: Whether to visualize the action term data.
        Returns:
            Whether the debug visualization was successfully set. False if the action term does
            not support debug visualization.
        """
        # check if debug visualization is supported
        if not self.has_debug_vis_implementation:
            return False
        # toggle debug visualization objects
        self._set_debug_vis_impl(debug_vis)
        # toggle debug visualization handles
        if debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
        # return success
        return True

    @abstractmethod
    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function is called once per environment step by the manager.

        Args:
            actions: The actions to process.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_actions(self):
        """Applies the actions to the asset managed by the term.

        Note:
            This is called at every simulation step by the manager.
        """
        raise NotImplementedError

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.
        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.
        This function calls the visualization objects and sets the data to visualize into them.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")


class ActionManager(ManagerBase):
    """Manager for processing and applying actions for a given world.

    The action manager handles the interpretation and application of user-defined
    actions on a given world. It is comprised of different action terms that decide
    the dimension of the expected actions.

    The action manager performs operations at two stages:

    * processing of actions: It splits the input actions to each term and performs any
      pre-processing needed. This should be called once at every environment step.
    * apply actions: This operation typically sets the processed actions into the assets in the
      scene (such as robots). It should be called before every simulation step.
    """

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the action manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ActionTermCfg]``).
            env: The environment instance.

        Raises:
            ValueError: If the configuration is None.
        """
        # check if config is None
        if cfg is None:
            raise ValueError("Action manager configuration is None. Please provide a valid configuration.")

        # call the base class constructor (this prepares the terms)
        super().__init__(cfg, env)
        # create buffers to store actions
        self._action = torch.zeros((self.num_envs, self.total_action_dim), device=self.device)
        self._prev_action = torch.zeros_like(self._action)

        # check if any term has debug visualization implemented
        self.cfg.debug_vis = False
        for term in self._terms.values():
            self.cfg.debug_vis |= term.cfg.debug_vis

    def __str__(self) -> str:
        """Returns: A string representation for action manager."""
        msg = f"<ActionManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = f"Active Action Terms (shape: {self.total_action_dim})"
        table.field_names = ["Index", "Name", "Dimension"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Dimension"] = "r"
        # add info on each term
        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.action_dim])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def total_action_dim(self) -> int:
        """Total dimension of actions."""
        return sum(self.action_term_dim)

    @property
    def active_terms(self) -> list[str]:
        """Name of active action terms."""
        return self._term_names

    @property
    def action_term_dim(self) -> list[int]:
        """Shape of each action term."""
        return [term.action_dim for term in self._terms.values()]

    @property
    def action(self) -> torch.Tensor:
        """The actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._action

    @property
    def prev_action(self) -> torch.Tensor:
        """The previous actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._prev_action

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command terms have debug visualization implemented."""
        # check if function raises NotImplementedError
        has_debug_vis = False
        for term in self._terms.values():
            has_debug_vis |= term.has_debug_vis_implementation
        return has_debug_vis

    """
    Operations.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the action data.
        Args:
            debug_vis: Whether to visualize the action data.
        Returns:
            Whether the debug visualization was successfully set. False if the action
            does not support debug visualization.
        """
        for term in self._terms.values():
            term.set_debug_vis(debug_vis)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets the action history.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            An empty dictionary.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # reset the action history
        self._prev_action[env_ids] = 0.0
        self._action[env_ids] = 0.0
        # reset all action terms
        for term in self._terms.values():
            term.reset(env_ids=env_ids)
        # nothing to log here
        return {}

    def process_action(self, action: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
        """
        # check if action dimension is valid
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}.")
        # store the input actions
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    def apply_action(self) -> None:
        """Applies the actions to the environment/simulation.

        Note:
            This should be called at every simulation step.
        """
        for term in self._terms.values():
            term.apply_actions()

    def get_term(self, name: str) -> ActionTerm:
        """Returns the action term with the specified name.

        Args:
            name: The name of the action term.

        Returns:
            The action term with the specified name.
        """
        return self._terms[name]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # create buffers to parse and store terms
        self._term_names: list[str] = list()
        self._terms: dict[str, ActionTerm] = dict()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # parse action terms from the config
        for term_name, term_cfg in cfg_items:
            # check if term config is None
            if term_cfg is None:
                continue
            # check valid type
            if not isinstance(term_cfg, ActionTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ActionTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, ActionTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type ActionType.")
            # add term name and parameters
            self._term_names.append(term_name)
            self._terms[term_name] = term


torch.pi = torch.acos(torch.zeros(1)).item() * 2
import yaml
import matplotlib.pyplot as plt

constants_path = "source/constants.yaml"
with open(constants_path, "r") as file:
    constants = yaml.safe_load(file)
JOINT_UNITREE_TO_ISAAC_LAB_MAPPING = constants["JOINT_UNITREE_TO_ISAAC_LAB_MAPPING"]
JOINT_ISAAC_LAB_TO_UNITREE_MAPPING = constants["JOINT_ISAAC_LAB_TO_UNITREE_MAPPING"]
DEFAULT_JOINT_POS_ISAAC_LAB = constants["DEFAULT_JOINT_POS_ISAAC_LAB"]

class LegwiseLatentActionManager(ActionManager):

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        self.robot_action_dim = 12
        self.latent_action_dim = 1 + 4 * 2 # main freq, 4 legs with freq and amp each
        self.residual_action_weight = 0.1

        super().__init__(cfg, env)

        # action buffers
        self._action = torch.zeros(
            (self.num_envs, self.robot_action_dim), device=self.device
        )
        self._prev_action = torch.zeros_like(self._action)

        self._residual_action = torch.zeros(
            (self.num_envs, self.robot_action_dim), device=self.device
        )
        self._prev_residual_action = torch.zeros_like(
            self._residual_action
        )
        self._latent_action = torch.zeros(
            (self.num_envs, self.latent_action_dim), device=self.device
        )
        self._prev_latent_action = torch.zeros_like(
            self._latent_action
        )

        prior_suffix = "dog_retargeted_5" # "real_robot", "dog_retargeted"

        self.projector = torch.jit.load(f"expert_projectors/temporal_spatial_prior_{prior_suffix}.pt").to(self.device)

        with open(f"expert_projectors/output_{prior_suffix}.yaml", "r") as file:
            prior_data = yaml.safe_load(file)

        self.init_joint_pos_gait = (
            torch.tensor(prior_data["FIRST_JOINT_POS_GAIT_CYCLE"])
            - torch.tensor(DEFAULT_JOINT_POS_ISAAC_LAB)[
                JOINT_ISAAC_LAB_TO_UNITREE_MAPPING
            ]
        )

        self.init_joint_pos_gait = torch.zeros_like(self.init_joint_pos_gait)

        self.init_joint_pos_gait = self.init_joint_pos_gait.repeat(self.num_envs, 1).to(self.device)


        self.live_print = False
        self.live_plot = False
        if self.live_plot:
            plt.ion()
            self._values_to_plot = {}
            for i in range(4):
                    self._values_to_plot.setdefault(f"leg_{i}_freq", [])
                    self._values_to_plot.setdefault(f"leg_{i}_amp", [])
                    self._values_to_plot.setdefault(f"leg_{i}_phase", [])
                    self._values_to_plot.setdefault(f"main_freq", [])
            for i in range(12):
                self._values_to_plot.setdefault(f"joint_{i}_pos", [])
            num_plots = len(self._values_to_plot)
            num_rows = (num_plots + 1) // 2
            self.figure, self.axs = plt.subplots(num_rows, 2)
            self.figure.set_size_inches(10, 5 * num_rows)
            self.axs = self.axs.flatten()
            


        self.has_freq = False
        if self.projector.get_frequency() is not None: # temporal prior

            self.has_freq = True

            self._demo_freq = self.projector.get_frequency()
            self.mean_main_freq = 0.0

            self.mean_amp = 1.0

            self.range_main_freq = 2 * self._demo_freq # * 1.2  # 0.3 # 1.0
            self.range_leg_freq = 0.0 # 0.3 # 0.3
            self.range_amp = 0.1 # 0.2 # 0.3

            self.phases = torch.zeros((self.num_envs, 4), device=self.device) # phase for each leg
            # sin cos phase for observations
            self.sin_cos_phases = torch.cat(
                (
                    torch.sin(self.phases),
                    torch.cos(self.phases),
                ),
                dim=1,
            ) # sin cos phase for leg; required for observations
            
            self.one_hot_vector = torch.eye(4).repeat(self.num_envs, 1).to(self.device)

            # buffers
            self._freqs = torch.zeros((self.num_envs, 4), device=self.device) # freq for each leg
            self._prev_freqs = torch.zeros_like(self._freqs)
            self._amps = torch.zeros((self.num_envs, 4), device=self.device) # amp for each leg
            self._prev_amps = torch.zeros_like(self._amps)

            self.i = 0

        else:
            raise NotImplementedError("Only temporal prior is supported")

    @property
    def action_term_dim(self) -> list[int]:
        """Shape of each action term."""
        return [self.robot_action_dim + self.latent_action_dim] # This is queried by the policy to get output dimension. Its seems save to modify this variable.

    def process_action(self, residual_and_latent_action: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
        """
        self.i += 1

        self._prev_action[:] = self.action
        self._prev_freqs = self.freqs
        self._prev_amps = self.amps

        assert residual_and_latent_action.shape[1] == self.total_action_dim
        assert residual_and_latent_action.shape[1] - self.latent_action_dim == self.robot_action_dim, "Dims obtained from the actor policy do not match expected shape."

        # clip actions [-1,1]
        residual_and_latent_action = torch.clamp(residual_and_latent_action, -1.0, 1.0)

        # get residual actions (should be same for all projectors); these are the first 12 actions
        self._residual_action[:] = residual_and_latent_action[
            :, : self.robot_action_dim
        ].to(self.device)
        self._prev_residual_action[:] = self.residual_action

        # latent actions are remaining actions
        self._latent_action[:] = residual_and_latent_action[
            :, self.robot_action_dim :
        ].to(self.device)
        self._prev_latent_action[:] = self.latent_action

        # get projected actions
        if self.has_freq:  # temporal prior

            main_freq = (
                self.latent_action[:, 0] * self.range_main_freq + self.mean_main_freq
                # torch.where(
                #     self._env.unwrapped.command_manager.get_command('base_velocity')[:, 0] < 0,
                #     -self.mean_main_freq,
                #     self.mean_main_freq
                # )
            )
            # for each leg we have two parameters: freq and amp
            n = (self.latent_action.shape[1] - 1) // 2
            assert n == 4  # number of legs
            _params = self.latent_action[:, 1:].reshape(-1, 2, n)
            amps = _params[:, 0] * self.range_amp + self.mean_amp
            leg_freqs = _params[:, 1] * self.range_leg_freq

            self._freqs = main_freq.unsqueeze(1) + leg_freqs
            self._amps = amps

            self.phases = self.phases + self._env.step_dt * 2 * torch.pi * self.freqs

            if self.live_print:
                print(f"Residual Action Magnitude: {torch.mean(torch.abs(self.residual_action)):.5f}")
                for i in range(20):
                    leg_freqs_str = ", ".join(
                        [
                            f"Leg {j+1} freq: {self.freqs[i][j].cpu().numpy().tolist():.5f}"
                            for j in range(4)
                        ]
                    )
                    print(
                        f"Target velocity: {self._env.unwrapped.command_manager.get_command('base_velocity')[i].tolist()}, Main freq: {main_freq[i].cpu().numpy().tolist():.5f}, {leg_freqs_str}"
                    )
                print("###############")

            if self.live_plot:

                self._values_to_plot.setdefault(f"main_freq", []).append(main_freq[0].cpu().numpy())
                for i in range(self._freqs.shape[1]): # for each leg
                    self._values_to_plot.setdefault(f"leg_{i}_freq", []).append(self.freqs[0][i].cpu().numpy())
                    self._values_to_plot.setdefault(f"leg_{i}_amp", []).append(amps[0][i].cpu().numpy())
                    self._values_to_plot.setdefault(f"leg_{i}_phase", []).append(self.phases[0][i].cpu().numpy())

                for i in range(12):
                    self._values_to_plot.setdefault(f"joint_{i}_pos", []).append(self._env.unwrapped.scene["robot"].data.joint_pos.cpu().numpy()[0][i])

                for i, (key, value) in enumerate(self._values_to_plot.items()):
                    self.axs[i].clear()
                    self.axs[i].plot(
                        [x * self._env.step_dt for x in range(len(value))], value
                    )
                    self.axs[i].set_title(key)

                
                
                plt.pause(0.001)
                plt.show()



            # reset phases for envs where episode length is 0; possible not required, as its handled by reset method of this class
            self.phases[torch.where(self._env.episode_length_buf == 0, True, False)] = (
                0.0
            )

            # sin cos phase for observations
            self.sin_cos_phases = torch.cat(
                (torch.sin(self.phases), torch.cos(self.phases)),
                dim=1,
            )

            sin_phase = self.amps * torch.sin(self.phases)
            cos_phase = self.amps * torch.cos(self.phases)

            # sin cos phase for projector
            sin_cos_phase = torch.cat(
                (sin_phase.flatten().unsqueeze(1), cos_phase.flatten().unsqueeze(1)),
                dim=1,
            )
            inputs = torch.column_stack((sin_cos_phase, self.one_hot_vector))
            with torch.no_grad():
                projected_action = self.projector(inputs).reshape(self.num_envs, -1)
                assert projected_action.shape[1] == self.robot_action_dim
        else:
            raise NotImplementedError("Only temporal prior is supported")

        action = (
            1 - self.residual_action_weight
        ) * projected_action + self.residual_action_weight * self.residual_action

        # if self.i < 150:
        #     action = self.init_joint_pos_gait
        # else:
        #     self.mean_main_freq = self._demo_freq

        assert action.shape[1] == self.robot_action_dim

        # re-order joints
        action = action[:, JOINT_UNITREE_TO_ISAAC_LAB_MAPPING]


        self._action[:] = action.to(self.device)

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions) # rescaling, offset
            idx += term.action_dim


    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets the action history.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            An empty dictionary.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # reset the action history
        self._prev_action[env_ids] = 0.0
        self._action[env_ids] = 0.0
        self._prev_residual_action[env_ids] = 0.0
        self._residual_action[env_ids] = 0.0
        self._prev_latent_action[env_ids] = 0.0
        self._latent_action[env_ids] = 0.0
        self._prev_freqs[env_ids] = 0.0
        self._freqs[env_ids] = 0.0
        self._prev_amps[env_ids] = 0.0
        self._amps[env_ids] = 0.0
        if self.has_freq:
            self.phases[env_ids] = 0.0
            self.sin_cos_phases = torch.cat(
                (
                    torch.sin(self.phases),
                    torch.cos(self.phases),
                ),
                dim=1,
            )

        # reset all action terms
        for term in self._terms.values():
            term.reset(env_ids=env_ids)
        # nothing to log here
        return {}

    @property
    def latent_action(self) -> torch.Tensor:
        return self._latent_action

    @property
    def prev_latent_action(self) -> torch.Tensor:
        return self._prev_latent_action

    @property
    def residual_action(self) -> torch.Tensor:
        return self._residual_action

    @property
    def prev_residual_action(self) -> torch.Tensor:
        return self._prev_residual_action
    
    @property
    def freqs(self) -> torch.Tensor:
        return self._freqs
    
    @property
    def prev_freqs(self) -> torch.Tensor:
        return self._prev_freqs
    
    @property
    def amps(self) -> torch.Tensor:
        return self._amps
    
    @property
    def prev_amps(self) -> torch.Tensor:
        return self._prev_amps

class PhaseActionManager(ActionManager):
    """Extends the standard action manager with a phase. This is a misuse of the action manager, but convenient as the action manager is called every env step."""

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        self.robot_action_dim = 12
        
        super().__init__(cfg, env)

        self.phases = torch.zeros((self.num_envs, 1), device=self.device)
        self.sin_cos_phases = torch.cat(
            (
                torch.sin(self.phases),
                torch.cos(self.phases),
            ),
            dim=1,
        ) # sin cos phase for observations
            
        # buffers
        # You most likely want to overwrite freqs to be non-zero for your application
        self._freqs = torch.zeros((self.num_envs, 1), device=self.device) # freq

    def process_action(self, action: torch.Tensor):
        # Dont inherit from base class as it checks for action dimensionality, and this is not correct for inheriting classes that change the actions
        assert action.shape[1] == self.robot_action_dim
        
        # store the input actions
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

        self.phases = self.phases + self._env.step_dt * 2 * torch.pi * self._freqs

        # reset phases for envs where episode length is 0; possibly not required, as its handled by reset method of this class
        self.phases[torch.where(self._env.episode_length_buf == 0, True, False)] = (
            0.0
        )
        self.sin_cos_phases = torch.cat(
            (
                torch.sin(self.phases),
                torch.cos(self.phases),
            ),
            dim=1,
        ) # sin cos phase for observations

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim] 
            term.process_actions(term_actions)
            idx += term.action_dim


    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        # TODO Check if resetting the phase here yields the correct obs for the rl env
        # if RSI is used, phase reset will be handled there!
        if not hasattr(self._env.cfg.events, 'reference_state_initialization'):
            self.phases[env_ids] = 0.0
        self.sin_cos_phases = torch.cat(
            (
                torch.sin(self.phases),
                torch.cos(self.phases),
            ),
            dim=1,
        )

        return super().reset(env_ids)

    @property
    def freqs(self) -> torch.Tensor:
        return self._freqs

class ResidualRLActionManager(PhaseActionManager):
    def __init__(self, cfg: object, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        assert hasattr(env.cfg, "residual_rl_data"), "Expected a motion file to be provided for residual RL action manager."

        # buffers
        self._action = torch.zeros((self.num_envs,  13), device=self.device) # need to overwrite as it is set to self.action_term_dim by default
        self._prev_action = torch.zeros_like(self._action)

        with open(env.cfg.residual_rl_data["motion_file"], "r") as f:
            motion_json = json.load(f)
            motion_data = np.array(motion_json["Frames"])
            motion_data = AMPLoader.reorder_from_pybullet_to_isaac_lab(motion_data)
        self.reference_jpos = AMPLoader.get_joint_pose_batch(motion_data)[
            env.cfg.residual_rl_data["start_frame"] : env.cfg.residual_rl_data[
                "end_frame"
            ],
            :,
        ]
        self.reference_jpos = torch.tensor(self.reference_jpos, device=self.device)
        self.num_frames = self.reference_jpos.shape[0]

        self.mean_freq = torch.ones_like(self._freqs) * 0.75
        self.range_freq = torch.ones_like(self._freqs) * 1.0

        self._prev_freq = torch.zeros_like(self._freqs)

    def process_action(self, action: torch.Tensor):
        # Dont inherit from base class as it checks for action dimensionality, and this is not correct for inheriting classes that change the actions
        assert action.shape[1] == 13

        # store the input actions
        self._prev_action[:] = self._action
        self._prev_freq[:] = self._freqs
        self._action[:] = action.to(self.device)

        # frequency is last action
        self._freqs = self.mean_freq + self.range_freq * torch.clamp(
            action[:, 12].unsqueeze(-1), -1.0, 1.0
        )

        # freq dependent phase
        self.phases = self.phases + self._env.step_dt * 2 * torch.pi * self._freqs

        # perform linear interpolation between two frames
        unscaled_index = (self.phases / (2 * torch.pi)) * self.num_frames
        idx0 = torch.floor(unscaled_index).long() % self.num_frames
        idx1 = (idx0 + 1) % self.num_frames
        alpha = (unscaled_index - torch.floor(unscaled_index))

        idx0 = idx0.squeeze()
        idx1 = idx1.squeeze()

        pos0 = self.reference_jpos[idx0]
        pos1 = self.reference_jpos[idx1]
        interpolated_reference_jpos = AMPLoader.slerp(pos0, pos1, alpha)

        # sin cos phase for observations
        self.sin_cos_phases = torch.cat(
            (
                torch.sin(self.phases),
                torch.cos(self.phases),
            ),
            dim=1,
        ) 

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            assert list(self._terms.keys()) == ["joint_pos"], "Only joint_pos actions supported."
            term_actions = action[:, idx : idx + term.action_dim]
            # we treat residual RL as time-varying offset for actuator target commands; term.action_dim is 12 so no need to change this line
            term._offset = interpolated_reference_jpos
            term.process_actions(term_actions)
            idx += term.action_dim

    @property
    def prev_freq(self) -> torch.Tensor:
        return self._prev_freq

    # This is queried by the policy to get output dimension. Its seems save to modify this variable.
    @property
    def action_term_dim(self) -> list[int]:
        """Shape of each action term."""
        return [self.robot_action_dim + 1] 


class MotionBlendingActionManager(PhaseActionManager):
    def __init__(self, cfg: object, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._amp_loader = None

    def update(self, amp_loader, traj_ids, times):
        if self._amp_loader is None:
            self._amp_loader = amp_loader
        self.traj_ids = traj_ids
        self.times = times
        self.num_frames = self._amp_loader.trajectory_num_frames[traj_ids]

    def process_action(self, action: torch.Tensor):
        # Dont inherit from base class as it checks for action dimensionality, and this is not correct for inheriting classes that change the actions
        if self._amp_loader is not None:
            subst = (
                self._amp_loader.time_between_frames
                + self._amp_loader.trajectory_frame_durations[self.traj_ids]
            )
            eps_length = self._env.episode_length_buf.cpu().numpy() * self._env.step_dt
            times = np.minimum(
                self.times + eps_length,
                self._amp_loader.trajectory_lens[self.traj_ids] - subst,
            )

            reference = self._amp_loader.get_full_frame_at_time_batch(
                self.traj_ids, times
            )
            reference_jpos = self._amp_loader.__class__.get_joint_pose_batch(reference)
        else:
            print("[WARN] amp loader is none in motion blending action manager")

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            assert list(self._terms.keys()) == [
                "joint_pos"
            ], "Only joint_pos actions supported."
            term_actions = action[:, idx : idx + term.action_dim]
            # we treat residual RL as time-varying offset for actuator target commands; term.action_dim is 12 so no need to change this line
            if self._amp_loader is not None:

                def decay(t):
                    return np.exp(-1 * t)

                alpha = decay(
                    self._env.episode_length_buf.cpu().numpy() * self._env.step_dt
                )
                term._offset = AMPLoader.slerp(
                    term._asset.data.default_joint_pos[:, term._joint_ids],
                    torch.as_tensor(reference_jpos, device=self.device),
                    torch.as_tensor(alpha, device=self.device).unsqueeze(-1),
                )
            term.process_actions(term_actions)
            idx += term.action_dim


class LegwisePhaseActionManager(ActionManager):
    """Extends the standard action manager with a phase for each leg. This is a misuse of the action manager, but convenient as the action manager is called every env step."""

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        self.robot_action_dim = 12

        super().__init__(cfg, env)

        self.phases = torch.zeros((self.num_envs, 4), device=self.device)
        self.sin_cos_phases = torch.cat(
            (
                torch.sin(self.phases),
                torch.cos(self.phases),
            ),
            dim=1,
        ) # sin cos phase for observations

        # buffers
        # You most likely want to overwrite freqs to be non-zero for your application
        self._freqs = torch.zeros((self.num_envs, 4), device=self.device) # freq

    def process_action(self, action: torch.Tensor):
        # Dont inherit from base class as it checks for action dimensionality, and this is not correct for inheriting classes that change the actions
        assert action.shape[1] == 12, "Not expecting any latent actions."
        
        # store the input actions
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

        self.phases = self.phases + self._env.step_dt * 2 * torch.pi * self._freqs

        # reset phases for envs where episode length is 0; possible not required, as its handled by reset method of this class
        self.phases[torch.where(self._env.episode_length_buf == 0, True, False)] = (
            0.0
        )

        # sin cos phase for observations
        self.sin_cos_phases = torch.cat(
            (
                torch.sin(self.phases),
                torch.cos(self.phases),
            ),
            dim=1,
        ) 

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        self.phases[env_ids] = 0.0
        self.sin_cos_phases = torch.cat(
            (
                torch.sin(self.phases),
                torch.cos(self.phases),
            ),
            dim=1,
        )

        return super().reset(env_ids)

    @property
    def freqs(self) -> torch.Tensor:
        return self._freqs


class InterpolatedStyleActionManager(LegwisePhaseActionManager):
    """This action manager takes a (Legwise) projector as input. It provides the style to the current point in time using the phase. Using the projector, the expert style can be interpolated between the expert timestamps."""

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.projector = torch.jit.load(
            "expert_projectors/temporal_spatial_prior_dog_retargeted_5.pt"
        ).to(self.device)
        self.expert_freq = self.projector.get_frequency()

        # buffers
        self._freqs = torch.full(
            (self.num_envs, 1), self.expert_freq, device=self.device
        )  # set expert frequency

        # required for legwise projector
        self.one_hot_vector = torch.eye(4).repeat(self.num_envs, 1).to(self.device)

        # initialize the jpos reference for the first frame; required for jvel calculation in process_action
        self.get_current_jpos_reference()


    def get_current_jpos_reference(self):
        sin_phase = torch.sin(self.phases) # no amp so far
        cos_phase = torch.cos(self.phases) # no amp so far

        # sin cos phase for projector
        sin_cos_phase = torch.cat(
            (sin_phase.flatten().unsqueeze(1), cos_phase.flatten().unsqueeze(1)),
            dim=1,
        )

        inputs = torch.column_stack((sin_cos_phase, self.one_hot_vector))
        with torch.no_grad():
            self._jpos_ref = self.projector(inputs).reshape(self.num_envs, -1)

        
        self._jpos_ref = self._jpos_ref[:, JOINT_UNITREE_TO_ISAAC_LAB_MAPPING]
        self._jpos_ref = self._jpos_ref + torch.tensor(DEFAULT_JOINT_POS_ISAAC_LAB, device=self.device)
        
    def process_action(self, action: torch.Tensor):
        assert action.shape[1] == 12, "Not expecting latent actions, only 12 full actions."

        super().process_action(action)

        self._prev_jpos_ref = self._jpos_ref

        self.get_current_jpos_reference() 

        self._jvel_ref = (self._jpos_ref - self.prev_jpos_ref) / self._env.step_dt

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        self.phases[env_ids] = 0.0
        self.sin_cos_phases = torch.cat(
            (
                torch.sin(self.phases),
                torch.cos(self.phases),
            ),
            dim=1,
        )

        self.get_current_jpos_reference() # initialize the jpos reference for the first frame; required for jvel calculation in process_action
        return super().reset(env_ids)
    
    @property
    def prev_jpos_ref(self) -> torch.Tensor:
        return self._prev_jpos_ref
    
    @property
    def jpos_ref(self) -> torch.Tensor:
        return self._jpos_ref
    
    @property
    def jvel_ref(self) -> torch.Tensor:
        return self._jvel_ref

class FrequencyInterpolatedStyleActionManager(InterpolatedStyleActionManager):
    """This action manager extends the base class by making the frequency of the phase learnable."""

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        self.latent_action_dim = 1 # main freq

        super().__init__(cfg, env)

        # mean_main_freq = self.expert_freq, and range_main_freq = 0.5 also worked good. Range of frequency needs to be tuned in the future for a larger range of target velocities. Problematic might be that just frequency=0 is learned if 0 is included.
        self.mean_main_freq = self.expert_freq / 2
        self.range_main_freq = self.expert_freq / 2

       
        # buffers
        self._action = torch.zeros((self.num_envs,  self.robot_action_dim), device=self.device) # need to overwrite as it is set to self.action_term_dim by default
        self._prev_action = torch.zeros_like(self._action)

        self._latent_action = torch.zeros(
            (self.num_envs, self.latent_action_dim), device=self.device
        )
        self._prev_latent_action = torch.zeros_like(
            self._latent_action
        )
        self._prev_freqs = torch.zeros_like(self._freqs)


    def process_action(self, residual_and_latent_action: torch.Tensor):
        self._prev_freqs = self.freqs
        self._prev_latent_action = self.latent_action
        
        residual_action = residual_and_latent_action[:, :self.robot_action_dim]
        self._latent_action = torch.clamp(residual_and_latent_action[:, self.robot_action_dim:], -1.0, 1.0)

        assert residual_action.shape[1] == self.robot_action_dim
        assert self._latent_action.shape[1] == self.latent_action_dim

        self._freqs = self._latent_action * self.range_main_freq + self.mean_main_freq
        
        super().process_action(residual_action)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        # Do not reset self._freqs to zero here!
        self._latent_action[env_ids] = 0.0
        self._prev_latent_action[env_ids] = 0.0
        return super().reset(env_ids)

    @property
    def prev_freqs(self) -> torch.Tensor:
        return self._prev_freqs
    
    @property
    def latent_action(self) -> torch.Tensor:
        return self._latent_action

    @property
    def prev_latent_action(self) -> torch.Tensor:
        return self._prev_latent_action
    
    @property
    def action_term_dim(self) -> list[int]:
        """Shape of each action term."""
        return [self.robot_action_dim + self.latent_action_dim] # This is queried by the policy to get output dimension. Its seems save to modify this variable.


# class StyleActionManager(PhaseActionManager):
#     """Extends the PhaseActionManager to return a style to the current point in time. No interpolation implemented!"""

#     def __init__(self, cfg: object, env: ManagerBasedEnv):
#         raise NotImplementedError("This class is not yet implemented")

#         super().__init__(cfg, env)

#         expert_data = "gait_cycle_data_5.json"
#         expert_metadata = "output_dog_retargeted_5.yaml"
#         expert_projector = "temporal_spatial_prior_dog_retargeted_5.pt"

#         self.projector = torch.jit.load(f"expert_projectors/{expert_projector}").to(self.device)
#         with open(f"expert_projectors/{expert_data}", "r") as file:
#             expert_data = json.load(file)

#         with open(f"expert_projectors/{expert_metadata}", "r") as file:
#             expert_metadata = yaml.safe_load(file)

#         self._freqs = torch.full((self.num_envs, 1), expert_metadata["Learned frequency expert"],device=self.device) # set expert frequency


#     def process_action(self, action: torch.Tensor):
#         super().process_action(action)

#         self.phases = self.phases + self._env.step_dt * 2 * torch.pi * self._freqs

#         # reset phases for envs where episode length is 0; possible not required, as its handled by reset method of this class
#         self.phases[torch.where(self._env.episode_length_buf == 0, True, False)] = (
#             0.0
#         )
#         self.sin_cos_phases = torch.cat(
#             (
#                 torch.sin(self.phases),
#                 torch.cos(self.phases),
#             ),
#             dim=1,
#         ) # sin cos phase for observations


#     def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
#         self.phases[env_ids] = 0.0
#         self.sin_cos_phases = torch.cat(
#             (
#                 torch.sin(self.phases),
#                 torch.cos(self.phases),
#             ),
#             dim=1,
#         )

#         return super().reset(env_ids)

#     @property
#     def freqs(self) -> torch.Tensor:
#         return self._freqs
