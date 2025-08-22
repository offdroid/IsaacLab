# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from pyclbr import Class
from typing import Literal

from omni.isaac.lab.utils import configclass

from rsl_rl.runners import OnPolicyRunner, AMPOnPolicyRunner


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    vel_dependent_actor_latent_dim: int = 0
    """ Whether to use an actor that outputs something based only on the envs target velocity. Set zero to disable."""


@configclass
class RslRlRndCfg:
    weight: float = 0.0
    """initial weight of the RND reward"""
    weight_schedule: dict | None = None
    """note: this is a dictionary with a required key called "mode". Please check the RND module for more information"""
    reward_normalization: bool = False
    """whether to normalize RND reward"""
    learning_rate: float = 0.001
    """learning rate for RND"""
    num_outputs: int = 1
    """number of outputs of RND network. Note: if -1, then the network will use dimensions of the observation"""
    predictor_hidden_dims: list[int] = [-1]
    """hidden dimensions of predictor network"""
    target_hidden_dims: list[int] = [-1]
    """hidden dimensions of target network"""


@configclass
class RslRlSymmetryCfg:
    use_data_augmentation: bool = False
    """this adds symmetric trajectories to the batch"""
    use_mirror_loss: bool = False
    """this adds symmetry loss term to the loss function"""
    data_augmentation_func: str = (
        "omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp:get_symmetric_states"
    )
    """string containing the module and function name to import
    Example: "legged_gym.envs.locomotion.anymal_c.symmetry:get_symmetric_states"
        @torch.no_grad()
        def get_symmetric_states(
            obs: Optional[torch.Tensor] = None, actions: Optional[torch.Tensor] = None, cfg: "BaseEnvCfg" = None, obs_type: str = "policy"
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    mirror_loss_coeff: float = 0.0
    """coefficient for symmetry loss term. If 0, no symmetry loss is used"""


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    symmetry_cfg: RslRlSymmetryCfg = None
    """The symmetry configuration."""

    rnd_cfg: RslRlRndCfg = None
    """The Random Network Distillation configuration."""


@configclass
class RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

    runner_class: type = OnPolicyRunner
    
    ##
    # AMP
    ##
    
    amp_replay_buffer_size: int = MISSING
