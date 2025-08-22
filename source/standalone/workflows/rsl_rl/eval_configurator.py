from dataclasses import dataclass


@dataclass
class DefaultEvalConfig:
    eval_metric_subfolder: str = ""  # empty for not using a subfolder
    eval_metric_filename: str = "f'metrics.yaml'"  # this expression will be EVALUATED eval(...) during runtime, ie you can use python code here and it must be an evaluatable string
    num_envs = 1000
    play_episode_length = 10.0  # s
    play_episodes_per_env = int(20) # for envs with curriculum this number needs to be higher so that equilibrium curriculum stage is reached
    rel_standing_envs = 0.0

    record_episode_jpos = False

    # run checks on env or agent cfg
    def run_checks(self, **kwargs):
        env_cfg = kwargs["env_cfg"]
        try:
            eval(self.eval_metric_filename)
        except:
            raise ValueError("eval_metric_filename must be an evaluatable string.")

    def set_env_cfg(self, env_cfg):
        env_cfg.commands.base_velocity.rel_standing_envs = self.rel_standing_envs
        env_cfg.commands.base_velocity.resampling_time_range = (
            self.play_episode_length,
            self.play_episode_length,
        )
        env_cfg.episode_length_s = self.play_episode_length
        env_cfg.is_eval_env = True  # enables additional logging

        return env_cfg


# This is meant as an abstract base class. Others should inherit.
@dataclass
class TargetDistribution(DefaultEvalConfig):
    eval_metric_filename: str = "f'x_{env_cfg.commands.base_velocity.ranges.lin_vel_x[0]}_y_{env_cfg.commands.base_velocity.ranges.lin_vel_y[0]}_yaw_{env_cfg.commands.base_velocity.ranges.ang_vel_z[0]}.yaml'"

    # require less evaluation because target velocities and yaws are fixed.
    play_episodes_per_env = int(1)
    num_envs = 5000

    def run_checks(self, **kwargs):
        super().run_checks(**kwargs)

        assert (
            kwargs["env_cfg"].commands.base_velocity.ranges.lin_vel_x[0]
            == kwargs["env_cfg"].commands.base_velocity.ranges.lin_vel_x[1]
        ), "Expected constant target speed."

        assert (
            kwargs["env_cfg"].commands.base_velocity.ranges.lin_vel_y[0]
            == kwargs["env_cfg"].commands.base_velocity.ranges.lin_vel_y[1]
        ), "Expected constant target speed for TargetSpeedDistributionEvaluation."

        assert (
            kwargs["env_cfg"].commands.base_velocity.ranges.heading[0]
            == kwargs["env_cfg"].commands.base_velocity.ranges.heading[1] or
            kwargs["env_cfg"].commands.base_velocity.ranges.ang_vel_z[0]
            == kwargs["env_cfg"].commands.base_velocity.ranges.ang_vel_z[1]
        ), "Expected constant target yaw or heading for TargetSpeedDistributionEvaluation."
        

        assert kwargs["args_cli"].x_speed is not None, (
            "You most likely want to set a fixed speed for evaluation."
        )
        assert kwargs["args_cli"].y_speed is not None, (
            "You most likely want to set a fixed speed for evaluation."
        )
        assert kwargs["args_cli"].yaw is not None, (
            "You most likely want to set a fixed yaw for evaluation."
        )


@dataclass
class TargetXYDistribution(TargetDistribution):
    eval_metric_subfolder: str = "TargetXYDistributionEvaluation"

    def run_checks(self, **kwargs):
        super().run_checks(**kwargs)

        assert kwargs["args_cli"].yaw == 0, (
            "You most likely want to set yaw to 0 for evaluation."
        )


@dataclass
class TargetXYawDistribution(TargetDistribution):
    eval_metric_subfolder: str = "TargetXYawDistributionEvaluation"
    

    def run_checks(self, **kwargs):
        super().run_checks(**kwargs)

        assert kwargs["args_cli"].y_speed == 0, (
            "You most likely want to set y_speed to 0 for evaluation."
        )
        
    # Because we train with heading target, but evaluate yaw rate targets, we need to modify the environment here
    def set_env_cfg(self, env_cfg):
        env_cfg = super().set_env_cfg(env_cfg)
        env_cfg.commands.base_velocity.heading_command = False
        return env_cfg


@dataclass
class RecordJposEpisodeTargetVelocity(TargetDistribution):
    eval_metric_subfolder: str = "RecordJposEpisodeTargetVelocityEvaluation"
    record_episode_jpos = True

    play_episodes_per_env = int(1)
    num_envs = 20
    jpos_log_filename: str = "f'x_{env_cfg.commands.base_velocity.ranges.lin_vel_x[0]}_y_{env_cfg.commands.base_velocity.ranges.lin_vel_y[0]}_yaw_{env_cfg.commands.base_velocity.ranges.ang_vel_z[0]}.th'"

    def run_checks(self, **kwargs):
        super().run_checks(**kwargs)

        assert kwargs["args_cli"].yaw == 0, (
            "You most likely want to set yaw to 0 for evaluation."
        )


@dataclass
class AgentExpertDistance(TargetDistribution):
    eval_metric_subfolder: str = "AgentExpertDistanceEvaluation"
    record_episode_jpos = False

    play_episodes_per_env = int(1)
    num_envs = 1
    # TODO: Use this for the agent expert distances
    jpos_log_filename: str = (
        "f'x_{env_cfg.commands.base_velocity.ranges.lin_vel_x[0]}_y_{env_cfg.commands.base_velocity.ranges.lin_vel_y[0]}_yaw_{env_cfg.commands.base_velocity.ranges.ang_vel_z[0]}.th'"
    )

    def run_checks(self, **kwargs):
        super().run_checks(**kwargs)

        assert (
            kwargs["args_cli"].yaw == 0
        ), "You most likely want to set yaw to 0 for evaluation."
