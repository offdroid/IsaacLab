import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
import plot_DEFINITIONS

# Set global plot styling
plt.rcParams.update(
    {
        "font.size": 40,  # Default font size
        "axes.labelsize": 40,  # Axes labels font size
        "xtick.labelsize": 24,  # X-tick label size
        "ytick.labelsize": 24,  # Y-tick label size
        "legend.fontsize": 24,  # Legend font size
        "axes.titlesize": 24,  # Axes titles font size
        "axes.titleweight": "bold",  # Axes titles font weight
    }
)


metrics_to_plot = [
    "error_vel_xy",
    # "error_vel_yaw",
    "mean_mechanical_cot",
    "real_curriculum_state",
    # "heading_error",
    # "agent_expert_distances",
]


def load_yaml_file(file_path):
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
            if data is None:
                data = {}
            return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def collect_metrics_per_run(runs_dict):
    """
    Aggregate metrics across seeds for each run.
    Returns a dictionary: {metric_name: [(run_name, mean, min, max)]}
    """
    metrics = defaultdict(list)

    for run_name, paths in runs_dict.items():
        all_metrics_per_seed = defaultdict(list)

        for path in paths:
            yaml_path = os.path.join(path, "metrics.yaml")
            data = load_yaml_file(yaml_path)
            for key, value in data.items():
                all_metrics_per_seed[key].append(value)

        for key, values in all_metrics_per_seed.items():
            if type(values[0]) == str:
                print(
                    f"Metric '{key}' is a string. Skipping. This is is only an info message: no action required."
                )
                continue
            arr = np.array(values)
            mean_val = np.mean(arr)
            min_val = np.min(arr)
            max_val = np.max(arr)
            metrics[key].append((run_name, mean_val, min_val, max_val))

    return metrics


def plot_metrics(metrics, runs, save_file_name):
    n_metrics = len(metrics_to_plot)
    if n_metrics == 0:
        print("No metrics to plot.")
        return

    cols = n_metrics  # int(np.ceil(np.sqrt(n_metrics)))
    rows = 1  # int(np.ceil(n_metrics / cols))

    fig = plt.figure(figsize=(5 * cols, 4 * rows))
    # fig.suptitle("Mean and Range Between Seeds", fontsize=16, y=0.995)

    all_run_names = list(runs.keys())

    for i, metric in enumerate(metrics_to_plot):
        if metric not in metrics:
            print(f"Metric '{metric}' not found in data. Skipping.")
            continue

        values = metrics[metric]
        row = i // cols
        col = i % cols
        ax = plt.subplot2grid((rows, cols), (row, col))

        run_names = [v[0] for v in values]
        means = [v[1] for v in values]
        mins = [v[2] for v in values]
        maxs = [v[3] for v in values]
        errors = [[mean - mn, mx - mean] for mean, mn, mx in zip(means, mins, maxs)]
        errors = np.array(errors).T  # shape (2, N)

        x_pos = np.arange(len(run_names))
        bars = ax.bar(
            x_pos,
            means,
            yerr=errors,
            capsize=5,
            color=[
                plot_DEFINITIONS.get_color_for_experiment_name(run) for run in run_names
            ],
            alpha=0.8,
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(run_names, rotation=45, ha="right", fontsize=8)
        ax.set_xticklabels([])
        ax.set_xticks([])

        # Set ymax, because value for AnimalAvatar is so large
        if metric == "error_vel_xy":
            y_max = 0.105
            ax.set_ylim(0, y_max)
            for x, mean, mn, mx in zip(x_pos, means, mins, maxs):
                # add value in plot
                if mx > y_max:
                    ax.text(
                        x,
                        y_max - 0.02,
                        f"{mx:.2f}\n+-{mean - mn + (mx - mean):.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                    )
        if metric == "mean_mechanical_cot":
            y_max = 5.0
            ax.set_ylim(0, y_max)
            for x, mean, mn, mx in zip(x_pos, means, mins, maxs):
                # add value in plot
                if mx > y_max:
                    ax.text(
                        x,
                        y_max - 1.5,
                        f"{mx:.2f}\n+-{mean - mn + (mx - mean):.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                    )

        ax.set_title(plot_DEFINITIONS.METRIC_FIELD_PLOT_TITLE_MAPPING[metric], y=1.07)
        # ax.set_ylabel(plot_DEFINITIONS.METRIC_FIELD_PLOT_TITLE_MAPPING[metric], fontsize=16)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, color=plot_DEFINITIONS.get_color_for_experiment_name(run)
        )
        for run in all_run_names
    ]
    fig.legend(
        handles,
        all_run_names,
        bbox_to_anchor=(0.51, 0.05),
        loc="upper center",
        ncol=2,
        frameon=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1 + 0.02 * (len(all_run_names) // 5))
    plt.savefig(f"plots/{save_file_name}", bbox_inches="tight")
    print(f"Saved figure with selected metrics as 'plots/{save_file_name}'")
    plt.close()


runs_box = {
    plot_DEFINITIONS.ExperimentNames.video_depth_cam: [
        "logs/rsl_rl/unitree_go2_AMPBox/2025-07-19_12-52-33_Curr_RSI_SEED_1",
        "logs/rsl_rl/unitree_go2_AMPBox/2025-07-19_12-52-33_Curr_RSI_SEED_2",
        "logs/rsl_rl/unitree_go2_AMPBox/2025-07-19_12-52-33_Curr_RSI_SEED_3",
    ],
    plot_DEFINITIONS.ExperimentNames.drl_simple_reward: [
        # "logs/rsl_rl/unitree_go2_Box/2025-07-23_15-13-01_SimpleRew_Curr_SEED_1",
        "logs/rsl_rl/unitree_go2_Box/2025-07-23_15-13-01_SimpleRew_Curr_SEED_2",
        "logs/rsl_rl/unitree_go2_Box/2025-07-23_15-13-01_SimpleRew_Curr_SEED_3",
    ],
    plot_DEFINITIONS.ExperimentNames.drl_complex_reward: [
        "logs/rsl_rl/unitree_go2_Box/2025-07-23_15-13-01_ComplexRew_Curr_SEED_1",
        "logs/rsl_rl/unitree_go2_Box/2025-07-23_15-13-01_ComplexRew_Curr_SEED_2",
        "logs/rsl_rl/unitree_go2_Box/2025-07-23_15-13-01_ComplexRew_Curr_SEED_3",
    ],
}

runs_stairs = {
    plot_DEFINITIONS.ExperimentNames.video_depth_cam: [
        "logs/rsl_rl/unitree_go2_AMPStairs/2025-07-25_17-14-21_stairs_first_try_curr_SEED_1/",
        "logs/rsl_rl/unitree_go2_AMPStairs/2025-07-27_23-21-03_stairs_first_try_curr_v2_SEED_2/",
    ],
    plot_DEFINITIONS.ExperimentNames.drl_simple_reward: [
        "logs/rsl_rl/unitree_go2_Stairs/2025-07-27_21-55-43-paper_simple_curr_v2_SEED_1/",
        "logs/rsl_rl/unitree_go2_Stairs/2025-07-27_21-55-43-paper_simple_curr_v2_SEED_2/",
        "logs/rsl_rl/unitree_go2_Stairs/2025-07-27_21-55-43-paper_simple_curr_v2_SEED_3/",
    ],
    plot_DEFINITIONS.ExperimentNames.drl_complex_reward: [
        "logs/rsl_rl/unitree_go2_Stairs/2025-07-27_17-01-20-paper_paper_curr_tunedv2_SEED_1/",
        "logs/rsl_rl/unitree_go2_Stairs/2025-07-27_17-01-20-paper_paper_curr_tunedv2_SEED_2/",
        "logs/rsl_rl/unitree_go2_Stairs/2025-07-27_17-01-20-paper_paper_curr_tunedv2_SEED_3/",
    ],
}

runs_flat = {
    plot_DEFINITIONS.ExperimentNames.manual_trajectory: [
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_manuallyGenerated_SEED_1",
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_manuallyGenerated_SEED_2",
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_manuallyGenerated_SEED_3",
    ],
    plot_DEFINITIONS.ExperimentNames.mocap: [
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_1",
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_2",
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_3",
    ],
    plot_DEFINITIONS.ExperimentNames.video_depth_cam: [
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_fromVision_motions_DepthCam_SEED_1",
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_fromVision_motions_DepthCam_SEED_2",
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_fromVision_motions_DepthCam_SEED_3",
    ],
    plot_DEFINITIONS.ExperimentNames.video_depth_cam_extendedWithoutReverse: [
        "logs/rsl_rl/unitree_go2_AMPflat/2025-06-06_15-35-34_fromVision_motions_DepthCam_extendedWithoutReverse_SEED_1",
        "logs/rsl_rl/unitree_go2_AMPflat/2025-06-06_15-35-34_fromVision_motions_DepthCam_extendedWithoutReverse_SEED_2",
        "logs/rsl_rl/unitree_go2_AMPflat/2025-06-06_15-35-34_fromVision_motions_DepthCam_extendedWithoutReverse_SEED_3",
    ],
    # plot_DEFINITIONS.ExperimentNames.video_depth_cam_extended: [
    #     "logs/rsl_rl/unitree_go2_AMPflat/2025-05-30_18-17-23_fromVision_motions_DepthCam_extended_SEED_1",
    #     "logs/rsl_rl/unitree_go2_AMPflat/2025-05-30_18-17-23_fromVision_motions_DepthCam_extended_SEED_2",
    #     "logs/rsl_rl/unitree_go2_AMPflat/2025-05-30_18-17-23_fromVision_motions_DepthCam_extended_SEED_3",
    # ],
    plot_DEFINITIONS.ExperimentNames.video_depth_model: [
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_fromVision_motions_AlignedDepthAnything_SEED_1",
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_fromVision_motions_AlignedDepthAnything_SEED_2",
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_fromVision_motions_AlignedDepthAnything_SEED_3",
    ],
    plot_DEFINITIONS.ExperimentNames.drl_simple_reward: [
        "logs/rsl_rl/unitree_go2_flat/2025-05-16_21-23-07_simpleReward_SEED_1",
        "logs/rsl_rl/unitree_go2_flat/2025-05-16_21-23-07_simpleReward_SEED_2",
        "logs/rsl_rl/unitree_go2_flat/2025-05-16_21-23-07_simpleReward_SEED_3",
    ],
    plot_DEFINITIONS.ExperimentNames.drl_complex_reward: [
        "logs/rsl_rl/unitree_go2_flat/2025-05-16_21-23-07_complexReward_SEED_1",
        "logs/rsl_rl/unitree_go2_flat/2025-05-16_21-23-07_complexReward_SEED_2",
        "logs/rsl_rl/unitree_go2_flat/2025-05-16_21-23-07_complexReward_SEED_3",
    ],
    plot_DEFINITIONS.ExperimentNames.animal_avatar: [
        "logs/rsl_rl/unitree_go2_AMPflat/2025-06-13_18-33-35_from_AnimalAvatar_SEED_1",
        "logs/rsl_rl/unitree_go2_AMPflat/2025-06-13_18-33-35_from_AnimalAvatar_SEED_2",
        "logs/rsl_rl/unitree_go2_AMPflat/2025-06-13_18-33-35_from_AnimalAvatar_SEED_3",
    ],
}


def main():

    save_file_name = "selected_metrics_stairs.pdf"

    runs = runs_stairs

    metrics = collect_metrics_per_run(runs)
    plot_metrics(metrics, runs, save_file_name)


if __name__ == "__main__":
    main()
