# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import glob

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from rsl_rl.runners import OnPolicyRunner, AMPOnPolicyRunner


@configclass
class UnitreeGo2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500 # num_learning_iterations
    save_interval = 249
    experiment_name = "unitree_go2_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeGo2FlatPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = int(300 * 1.5 * 2 * 2 * 1.5) # +2 it for rough terrain
        self.experiment_name = "unitree_go2_flat"
        self.policy.actor_hidden_dims = [256, 256, 256] # with DR: 256; without DR: 128
        self.policy.critic_hidden_dims = [256, 256, 256] # with DR: 256; without DR: 128

@configclass
class UnitreeGo2AMPFlatPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.save_interval = 4999
        
        
        self.experiment_name = "unitree_go2_AMPflat" 

        self.policy_class_name = 'ActorCritic'
        self.max_iterations = 30_000 # with DR: 30_000 + 100 to make sure last policy is saved; without DR: 25_000

        self.amp_reward_coef = 2.0
        
        self.amp_motion_folder = 'datasets/dummy/*'
        self.amp_motion_files = glob.glob(self.amp_motion_folder)
        self.amp_num_preload_transitions = 2_000_000
        self.amp_task_reward_lerp = 0.5 # weighting factor of task reward (style reward is 1-task_reward_lerp)
        self.amp_discr_hidden_dims = [1024, 512]

        self.min_normalized_std = [0.05] * 4 + [0.02] * 4 +[0.05] * 4#  + [0.05] # for ResidualRL

        self.algorithm.amp_replay_buffer_size = 1_000_000
        self.algorithm.num_learning_epochs = 5
        self.algorithm.class_name = 'AMPPPO'
        self.algorithm.entropy_coef = 0.01
        self.algorithm.num_mini_batches = 6 # 4?
        
        self.runner_class = AMPOnPolicyRunner
        
    def update_motion_files(self):
        motion_files = glob.glob(self.amp_motion_folder)
        self.amp_motion_files = motion_files

# This class only exists to provide self.experiment_name for logging.
@configclass
class UnitreeGo2AMPNoisyFlatPPORunnerCfg(UnitreeGo2AMPFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_AMPNoisyflat" 

# This class only exists to provide self.experiment_name for logging.
@configclass
class UnitreeGo2AMPStairsPPORunnerCfg(UnitreeGo2AMPFlatPPORunnerCfg):
        def __post_init__(self):
            super().__post_init__()
            self.experiment_name = "unitree_go2_AMPStairs" 

# This class only exists to provide self.experiment_name for logging.
@configclass
class UnitreeGo2AMPBoxPPORunnerCfg(UnitreeGo2AMPFlatPPORunnerCfg):
        def __post_init__(self):
            super().__post_init__()
            self.experiment_name = "unitree_go2_AMPBox" 

# This class only exists to provide self.experiment_name for logging.
@configclass
class UnitreeGo2StairsPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_Stairs" 

# This class only exists to provide self.experiment_name for logging.
@configclass
class UnitreeGo2BoxPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_Box" 

# This class only exists to provide self.experiment_name for logging.
@configclass
class UnitreeGo2StairsResidualRLPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_Stairs_ResidualRL" 

# This class only exists to provide self.experiment_name for logging.
@configclass
class UnitreeGo2StandingPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_standing" 
        self.algorithm.entropy_coef = 0.0


# This class only exists to provide self.experiment_name for logging.
@configclass
class UnitreeGo2AMPStandingPPORunnerCfg(UnitreeGo2AMPFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.amp_task_reward_lerp = 0.5 # weighting factor of task reward (style reward is 1-task_reward_lerp)
        self.experiment_name = "unitree_go2_AMPstanding"
        
# This class only exists to provide self.experiment_name for logging.
@configclass
class UnitreeGo2TorquePPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = int(300 * 1.5 * 2 * 2 * 1.5*2)
        self.algorithm.entropy_coef = 0.002

