# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

import eval_configurator

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record videos during training.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Number of environments to simulate.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# Olivers additional args
parser.add_argument(
    "--evaluate",
    action="store_true",
    default=None,
    help="Use this to log evaluation metrics. It also enables additional logging and sets further flags programmatically below.",
)
parser.add_argument("--eval_config", type=str, default="DefaultEvalConfig", help="Here you can specify the name of a dataclass in eval_configurator.py to set some parameters of the evaluation (mostly impacts file saving for now).")

parser.add_argument(
    "--x_speed",
    type=float,
    default=None,
    help="Target x speed for evaluation.",
)
parser.add_argument(
    "--y_speed",
    type=float,
    default=None,
    help="Target y speed for evaluation.",
)
parser.add_argument(
    "--yaw",
    type=float,
    default=None,
    help="Target yaw (ang_vel_z, note that this is actually yaw rate) for evaluation.",
)
parser.add_argument(
    "--max_delay",
    type=float,
    default=None,
    help="Target max_delay for evaluation. For Huawei experiments.",
)
parser.add_argument(
    "--disable_rsi",
    action="store_true",
    help="Disable RSI. If set, RSI will be disabled.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# headless for evaluation
if args_cli.evaluate:
    # NOTE it would be more clean to create separate environment configs, but this many additional environments, all of which would share same configurations.
    eval_config_class = getattr(eval_configurator, args_cli.eval_config)
    eval_config = eval_config_class()
    
    args_cli.headless = True
    args_cli.num_envs = eval_config.num_envs

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import yaml
import gymnasium as gym
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from rsl_rl.runners import OnPolicyRunner, AMPOnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

from rsl_rl.storage import ObservationHistoryStorage

from actionManagerLatentActorMapping import (
    get_vel_dependent_actor_latent_dim_for_action_manager_class,
)

from rsl_rl_utils import interpolate_trajectory


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # for correct metrics calculation
    if args_cli.evaluate:
        PLAY_EPISODE_LENGTH = eval_config.play_episode_length  # s
        PLAY_EPISODES_PER_ENV = eval_config.play_episodes_per_env # int

        env_cfg = eval_config.set_env_cfg(env_cfg)

        # run some checks
        if (
            args_cli.x_speed is not None
            or args_cli.y_speed is not None
            or args_cli.yaw is not None
        ) and not args_cli.eval_config in [
            "TargetXYDistribution",
            "TargetXYawDistribution",
            "RecordJposEpisodeTargetVelocity",
        ]:
            raise ValueError("You most likely want to use target speed and yaw values with TargetSpeedDistribution eval_config.")

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli
    )

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    log_dir = os.path.dirname(resume_path)

    # Overwrite env_cfg and agent_cfg with args_cli
    if args_cli.x_speed is not None:
        env_cfg.commands.base_velocity.ranges.lin_vel_x = [args_cli.x_speed, args_cli.x_speed]
    if args_cli.y_speed is not None:
        env_cfg.commands.base_velocity.ranges.lin_vel_y = [args_cli.y_speed, args_cli.y_speed]
    if args_cli.yaw is not None:
        env_cfg.commands.base_velocity.ranges.ang_vel_z = [args_cli.yaw, args_cli.yaw]

    # overwrite actuator max_delay for HUAWEI experiments
    if args_cli.max_delay is not None:
        env_cfg.scene.robot.actuators['base_legs'].max_delay = int(args_cli.max_delay)
        print(env_cfg.scene.robot.actuators['base_legs'].max_delay)

    # Overwrite RSI
    if args_cli.disable_rsi:
        env_cfg.events.reference_state_initialization = None
        # TODO would need to replace by random initialization.

    # Load the stored agent config. We replace some parameters in agent_cfg with the stored values later in the code.
    f = open(os.path.join(log_dir, "params", "agent.yaml"))
    loaded_agent_cfg = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    f = open(os.path.join(log_dir, "params", "env.yaml"))
    loaded_env_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
    f.close()

    # Get curriculu

    # Previous policies have been trained with different configuration.
    agent_cfg.policy.actor_hidden_dims = loaded_agent_cfg["policy"]["actor_hidden_dims"]
    agent_cfg.policy.critic_hidden_dims = loaded_agent_cfg["policy"]["critic_hidden_dims"]

    if env_cfg.is_amp_env:
        # Load same motion files that were used during training. This is required, otherwise results might be different than in training due to RSI, and the agent_expert_distances gets calculated incorrectly.
        amp_motion_folder = loaded_agent_cfg["amp_motion_folder"]
        env_cfg.amp_motion_folder = amp_motion_folder
        agent_cfg.amp_motion_folder = amp_motion_folder
        print(f"Using the following AMP motion folder: {amp_motion_folder}")

        env_cfg.update_motion_files()
        agent_cfg.update_motion_files()

        print(
            f"Loaded the following AMP motion files: {env_cfg.amp_motion_files}"
        )

        assert (
            env_cfg.amp_motion_files == agent_cfg.amp_motion_files
        ), f"Motion files in env and agent config should be the same, but got {env_cfg.amp_motion_files} and {agent_cfg.amp_motion_files}."

    # specify directory for logging experiments
    env_cfg.seed = agent_cfg.seed

    agent_cfg.policy.vel_dependent_actor_latent_dim = (
        get_vel_dependent_actor_latent_dim_for_action_manager_class(
            env_cfg.action_manager_class
        )
    )

    if not agent_cfg.policy.vel_dependent_actor_latent_dim == 0:
        assert (
            list(env_cfg.observations.policy.__dict__.items())[4][0]
            == "velocity_commands"
            and list(env_cfg.observations.policy.__dict__.items())[2][0]
            == "base_lin_vel"
            and list(env_cfg.observations.policy.__dict__.items())[3][0]
            == "base_ang_vel"
        ), "Found obs terms at wrong position. It must be on the correct position for the ActorFreq!"

    if args_cli.evaluate:
        eval_config.run_checks(env_cfg=env_cfg, args_cli=args_cli)

        eval_metric_folder = os.path.join(log_dir, eval_config.eval_metric_subfolder)
        os.makedirs(eval_metric_folder, exist_ok=True)

    # create isaac environment
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = agent_cfg.runner_class(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )  # OnPolicyRunner, AMPOnPolicyRunner
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic,
        ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.pt",
    )
    # export_policy_as_onnx(
    #     ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # ) # doesnt work for current actor_critic model

    # thats how to import the model
    policy = torch.jit.load(os.path.join(export_model_dir, "policy.pt")).cuda()

    # reset environment
    obs, _ = env.get_observations()
    obs_history_storage = ObservationHistoryStorage(
        num_envs=args_cli.num_envs,
        num_obs=obs.shape[1],
        max_length=5,
        device=env.unwrapped.device,
    )

    if env_cfg.is_amp_env:

        # Expert trajectories are required to compute agent_expert_distances metrics
        # expert_trajectories = env.unwrapped.event_manager.get_term_cfg("reference_state_initialization").func.amp_loader.trajectories # this is a list of trajectories: [(n_frames, n_amp_obs),...]
        # Better retrieve the expert trajectories from the ppo_runner, as RSI might not be used in some cases.
        expert_trajectories = ppo_runner.alg.amp_data.trajectories

        # interpolate trajectories to calculate distances more precisely
        interpolated_expert_trajectories = []
        num_interpolations = 10
        for trajectory in expert_trajectories:
            interpolated_expert_trajectories.append(
                interpolate_trajectory(trajectory, num_interpolations)
            )

        interpolated_expert_trajectories_list = interpolated_expert_trajectories
        interpolated_expert_trajectories = torch.cat(
            interpolated_expert_trajectories, dim=0
        )

        agent_expert_distances_individual = torch.zeros(
            (
                int(env_cfg.episode_length_s / env.unwrapped.step_dt),
                len(interpolated_expert_trajectories_list),
            ),
            device=env.unwrapped.device,
        )

        # amp_obs = env.unwrapped.get_amp_observations().to(env.unwrapped.device)
        # amp_rewards_buffer = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
        agent_expert_distances = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)

    # It is very unclean to use a second episode_length_buf besides the one in the environment. However, the one in the environment gets reset to 0 before I can calculate the metrics in this script. So I need to keep track of the episode lengths myself. In the future I'd like to find a way to avoid introducing a second episode_length_buf here.
    episode_length_buf = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    early_termination_counter = 0
    obs_history_storage.add(obs)
    obs_history = obs_history_storage.get()

    simulated_step_time = env.unwrapped.step_dt

    ###### Debug correspondance of target speeds with freq ######
    # if hasattr(policy.actor, "actor_freq"):
    #     x_speeds = torch.arange(-1.0, 2.0 + 0.1, 0.1)
    #     target_cmds = torch.stack(
    #         [torch.tensor([x, 0.0, 0.0] * 5) for x in x_speeds]
    #     ).to(args_cli.device)
    #     freq = (
    #         torch.clamp(policy.actor.actor_freq(target_cmds)[:, 0], -1.0, 1.0)
    #         * env.unwrapped.action_manager.range_freq
    #         + env.unwrapped.action_manager.mean_freq
    #     )  # main_freq * scaling + offset
    #     for x_speed, frequency in zip(x_speeds, freq):
    #         print(f"x speed {x_speed:+.4f}: {frequency:+.4f} Hz")

    #     print("######")

    #     y_speeds = torch.arange(-1.0, 1.0 + 0.1, 0.1)
    #     target_cmds = torch.stack(
    #         [torch.tensor([0.0, y, 0.0] * 5) for y in y_speeds]
    #     ).to(args_cli.device)
    #     freq = (
    #         torch.clamp(policy.actor.actor_freq(target_cmds)[:, 0], -1.0, 1.0)
    #         * env.unwrapped.action_manager.range_freq
    #         + env.unwrapped.action_manager.mean_freq
    #     )  # main_freq * scaling + offset
    #     for x_speed, frequency in zip(y_speeds, freq):
    #         print(f"y speed {x_speed:+.4f}: {frequency:+.4f} Hz")
    #     print("######")
    ###### ###################### ######

    if args_cli.evaluate:
        eval_episode_metrics = dict()

        NUM_EVAL_STEPS = (
            PLAY_EPISODES_PER_ENV
            * env.unwrapped.num_envs
            * PLAY_EPISODE_LENGTH
            / env.unwrapped.step_dt
        )

        if args_cli.evaluate:
            jpos_log = torch.zeros(
                (
                    int(PLAY_EPISODE_LENGTH / env.unwrapped.step_dt),
                    int(env.unwrapped.num_envs),
                    12,
                )
            )

    timestep = 0
    total_num_steps = 0
    # simulate environment
    
    # action_log = []
    while simulation_app.is_running():
        # Record the start time of the current loop
        current_time = time.time()

        # Run everything in inference mode
        with torch.inference_mode():
            # Agent steppinp
            actions = policy(obs_history)
            # action_log.append(actions.cpu().numpy().tolist())
            
            # Environment stepping
            obs, _, dones, extras, *optional_values = env.step(actions)
            # d = env.unwrapped.observation_manager._group_obs_term_cfgs['policy'][-2].func(env.unwrapped).item()
            # print(
            #     f"relative_distance_to_box: {d:.4f}"
            # )
            if env_cfg.is_amp_env:

                amp_observations = env.unwrapped.get_amp_observations()
                agent_expert_distances += torch.cdist(amp_observations, interpolated_expert_trajectories).min(dim=1).values

                # for i, traj in enumerate(interpolated_expert_trajectories_list):
                #     assert amp_observations.shape[0] == 1, "Expect only one environment."
                #     agent_expert_distances_individual[timestep, i] = torch.cdist(amp_observations, traj).min(dim=1).values[0]

                # NOTE using the discriminator output to calculate style imitation is suboptimal. Its better to introduce the agent_expert_distances metric.
                # next_amp_obs_with_term = torch.clone(next_amp_obs)
                # next_amp_obs_with_term[rest_env_ids] = terminal_amp_states

                # rewards, _, amp_rewards_logging = ppo_runner.alg.discriminator.predict_amp_reward(
                #     amp_obs, next_amp_obs_with_term, rewards, normalizer=None)#ppo_runner.alg.amp_normalizer)

                # amp_obs = torch.clone(next_amp_obs)
                # amp_rewards_buffer += amp_rewards_logging

            if args_cli.evaluate and eval_config.record_episode_jpos:
                jpos_log[total_num_steps // env.unwrapped.num_envs] = (
                    env.unwrapped.scene["robot"].data.joint_pos
                )

            total_num_steps += env.unwrapped.num_envs
            episode_length_buf += 1

            assert (
                len(optional_values) == 2 or len(optional_values) == 0
            ), "Too many optional values returned by the environment"

            if dones.any():
                if args_cli.evaluate:
                    # command_manager metrics
                    for (
                        metric_name,
                        metric_value,
                    ) in env.unwrapped.command_manager._terms[
                        "base_velocity"
                    ].episode_metrics.items():
                        eval_episode_metrics.setdefault(metric_name, []).extend(metric_value[dones==1.0].cpu().tolist())
                    # Curriculum state
                    if hasattr(env.unwrapped, "curriculum_manager"):
                        if "terrain_levels" in env.unwrapped.curriculum_manager._curriculum_state:
                            eval_episode_metrics["curriculum_state"] = (
                                env.unwrapped.curriculum_manager._curriculum_state[
                                    "terrain_levels"
                                ]
                            )
                    # amp rewards
                    # eval_episode_metrics.setdefault("amp_rewards", []).extend(
                    #     (
                    #         amp_rewards_buffer[dones == 1.0]
                    #         / episode_length_buf[dones == 1.0]
                    #     )
                    #     .cpu()
                    #     .tolist()
                    # )
                    # amp_rewards_buffer[dones == 1.0] = 0.0
                    # episode_length_buf[dones == 1.0] = 0
                    # agent expert distances
                    if env_cfg.is_amp_env:
                        eval_episode_metrics.setdefault(
                            "agent_expert_distances", []
                        ).extend(
                            (
                                agent_expert_distances[dones == 1.0]
                                / episode_length_buf[dones == 1.0]
                            )
                            .cpu()
                            .tolist()
                        )
                        agent_expert_distances[dones == 1.0] = 0.0

                    # keep track of early-terminated episodes (i.e. non-success)
                    early_termination_counter += (
                        ((dones == 1) & (episode_length_buf < env.max_episode_length))
                        .sum()
                        .item()
                    )

                obs_history_storage.reset(dones)

            obs_history_storage.add(obs)
            obs_history = obs_history_storage.get()

        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if not args_cli.evaluate:
            # Calculate the elapsed real-world time for this loop iteration
            elapsed_real_time = time.time() - current_time

            # Sleep for the remaining time to match the simulated step time
            sleep_time = simulated_step_time - elapsed_real_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"WARNING: Simulation slower than real time for {sleep_time}s!")
                pass

        if args_cli.evaluate:
            if total_num_steps >= NUM_EVAL_STEPS:
                break

    # Assuming action_log is already defined and has shape (523, 1, 12)
    # Example: action_log = np.random.rand(523, 1, 12)
    # actions = np.array(action_log).squeeze(axis=1)  # shape: (523, 12)

    # joint_names = [
    #     'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
    #     'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
    #     'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'
    # ]

    # timesteps = np.arange(actions.shape[0])
    # num_joints = actions.shape[1]

    # fig, axes = plt.subplots(num_joints, 1, figsize=(10, 2 * num_joints), sharex=True)

    # for i in range(num_joints):
    #     axes[i].plot(timesteps, actions[:, i])
    #     axes[i].set_ylabel(joint_names[i])
    #     axes[i].grid(True)

    # axes[-1].set_xlabel('Timestep')
    # fig.suptitle("Robot Joint Actions Over Time", fontsize=16)
    # fig.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for the title
    # plt.savefig("actions_AMPflatVision_2025-08-06_08-47-51_flat_test_DR5_improved_minimal_motion_files_SEED_1.pdf")

    # store the metrics
    if args_cli.evaluate:
        total_episodes_real = len(
            eval_episode_metrics["episode_lengths"]
        ) # can use any metric here - len should be similar for all

        for key, value in eval_episode_metrics.items():
            eval_episode_metrics[key] = torch.mean(torch.tensor(value)).item()

        # get real values for curriculum
        if "curriculum_state" in eval_episode_metrics:
            if (
                "box"
                in loaded_env_cfg["scene"]["terrain"]["terrain_generator"][
                    "sub_terrains"
                ]
            ):
                curr_min_max = loaded_env_cfg["scene"]["terrain"]["terrain_generator"][
                    "sub_terrains"
                ]["box"]["box_height_range"]
            else:
                raise ValueError("Unknown terrain type to calculate real curriculum values.")
            eval_episode_metrics["real_curriculum_state"] = (
                (eval_episode_metrics["curriculum_state"] - 0) /
                loaded_env_cfg["scene"]["terrain"]["terrain_generator"]["num_rows"]
            ) * (curr_min_max[1] - curr_min_max[0]) + curr_min_max[0]

        # other stats
        eval_episode_metrics["num_eval_steps"] = NUM_EVAL_STEPS
        eval_episode_metrics["num_envs"] = env.unwrapped.num_envs
        eval_episode_metrics["episodes_per_env"] = PLAY_EPISODES_PER_ENV
        eval_episode_metrics["total_episodes (real)"] = total_episodes_real

        eval_episode_metrics["number_failed_episodes"] = early_termination_counter
        eval_episode_metrics["successrate"] = (total_episodes_real - early_termination_counter) / total_episodes_real

        eval_episode_metrics["episode length in s (target)"] = PLAY_EPISODE_LENGTH

        if env_cfg.is_amp_env:
            eval_episode_metrics["amp_motion_folder"] = env_cfg.amp_motion_folder

        eval_metric_file_name = eval(eval_config.eval_metric_filename) # eval: allows for dynamic file naming which is convenient for logging
        with open(os.path.join(eval_metric_folder, eval_metric_file_name), "w") as f:
            yaml.dump(eval_episode_metrics, f)
        print(f"Metrics: {eval_episode_metrics}")

        if eval_config.record_episode_jpos:
            jpos_log_path = os.path.join(eval_metric_folder, eval(eval_config.jpos_log_filename))
            torch.save(jpos_log, jpos_log_path)
            print(f"Joints positions log saved to: {jpos_log_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
