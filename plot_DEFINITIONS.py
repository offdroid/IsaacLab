from dataclasses import dataclass
import matplotlib.pyplot as plt


METRIC_FIELD_PLOT_TITLE_MAPPING = {
    "mean_mechanical_cot": "Cost of Transport [1]",
    "error_vel_xy": "Tracking Error Vel. [m/s]",
    "agent_expert_distances": "Imitation score ↓",
    "heading_error": "Heading Error [rad]",
    "error_vel_yaw": "Tracking Error Yaw [rad]",
    "real_curriculum_state": "Mean successful\nstep height [m]",  # TODO needs to be adjusted based on terrain
    "error_feet_height": "Feet height error [m]",
    "successrate": "Success rate",
}

XY_FIELD_XY_LABEL_MAPPING = {
    "target_velocity_x": "Target Vel. X [m/s]",
    "target_velocity_y": "Target Vel. Y [m/s]",
    "heading_target": "Target Heading [rad]",
    "target_yaw": "Target Ang. Vel. z [rad]",
}


@dataclass
class DataSourceNames:
    manual_trajectory = "Manual Trajectory"
    video_depth_cam = "Video w. Depth Camera"
    video_depth_model = "Video w. DepthAnythingV2"
    mocap = "MoCap"
    mocap2 = "MoCap2"
    mocap3 = "MoCap3"
    mocap4 = "MoCap4"


@dataclass
@dataclass
class ExperimentNames:
    manual_trajectory = "Manual Trajectory (AMP)"
    video_depth_cam = "Video w. Depth Camera (AMP)"
    video_depth_cam_extended = "Video w. Depth Camera (extended with reverse) (AMP)"
    video_depth_cam_extendedWithoutReverse = "Video w. Depth Camera (extended) (AMP)"
    video_depth_model = "Video w. DepthAnythingV2 (AMP)"
    mocap = "MoCap (AMP)"
    mocap2 = "MoCap2 (AMP)"
    mocap3 = "MoCap3 (AMP)"
    mocap4 = "MoCap4 (AMP)"
    drl_simple_reward = "Simple Reward (PPO)"
    drl_complex_reward = "Complex Reward (PPO)"
    animal_avatar = "Animal Avatar (AMP)"

    def map_experiment_dir_to_experiment_name(self, experiment_dir):
        # Currently this function is used by plot_errorOnTargetDistribution.py

        # NOTE order matters for this elif chain
        if "mocap_AMP_for_hardware" in experiment_dir:
            return self.mocap
        elif "manuallyGenerated" in experiment_dir:
            return self.manual_trajectory
        elif "fromVision_motions_DepthCam_extendedWithoutReverse" in experiment_dir:
            return self.video_depth_cam_extendedWithoutReverse
        elif "fromVision_motions_DepthCam_extended" in experiment_dir:
            return self.video_depth_cam_extended
        elif "fromVision_motions_AlignedDepthAnything" in experiment_dir:
            return self.video_depth_model
        elif "complexReward" in experiment_dir:
            return self.drl_complex_reward
        elif "simpleReward" in experiment_dir:
            return self.drl_simple_reward
        elif "fromVision_motions_DepthCam" in experiment_dir:
            return self.video_depth_cam
        else:
            raise ValueError(f"Unknown experiment dir: {experiment_dir}.")


def get_color_for_experiment_name(experiment_name):
    cmap = plt.cm.get_cmap("tab10", 10)
    if experiment_name == ExperimentNames.manual_trajectory:
        return cmap(0)
    elif experiment_name == ExperimentNames.video_depth_cam:
        return cmap(1)
    elif experiment_name == ExperimentNames.drl_complex_reward:
        return cmap(2)
    elif experiment_name == ExperimentNames.drl_simple_reward:
        return cmap(8)  # cmap(4) is red -> avoid
    elif experiment_name == ExperimentNames.animal_avatar:
        return cmap(4)
    elif experiment_name == ExperimentNames.mocap:
        return cmap(5)
    elif experiment_name == ExperimentNames.video_depth_cam_extendedWithoutReverse:
        return cmap(6)
    elif experiment_name == ExperimentNames.video_depth_model:
        return cmap(7)
    else:
        raise ValueError(
            f"Unknown experiment name for colormapping: {experiment_name}."
        )
