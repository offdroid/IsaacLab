import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# To plot joint 5 from three datasets
data_paths = [
    'LatentSpacePD_4_SEED_2.npy',
    'TORQUE_styleRew_24_SEED_42.npy',
    'LatentSpaceTorque_9_SEED_4.npy', 
]

# Custom titles mapping (optional)
title_mapping = {
    'LatentSpacePD_4_SEED_2.npy': 'Position control policy + PD controller',
    'TORQUE_styleRew_24_SEED_42.npy': 'Torque control policy',
    'LatentSpaceTorque_9_SEED_4.npy': 'Torque control policy w/ action rate penalty',
}

joint_idx = 3
output_path = 'torques.pdf'
time_duration = 3 # seconds

# Set font sizes (increased by 50%)
plt.rcParams.update({
    'font.size': 16,          # base font size
    'axes.titlesize': 19,     # subplot title
    'axes.labelsize': 19,     # axis labels
    'xtick.labelsize': 17,    # x tick labels
    'ytick.labelsize': 17,    # y tick labels
})

def load_torque_data(file_path):
    """Load torque data from a numpy file."""
    try:
        data = np.load(file_path)
        print(f"Loaded data from {file_path} with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def create_time_array(data_shape, dt):
    """Create time array based on data shape and time step."""
    n_timesteps = data_shape[0]
    return np.arange(n_timesteps) * dt

def plot_joint_torques(data_paths, joint_idx, title_mapping=None, output_path=None):
    """
    Plot joint torques from multiple datasets.
    
    Args:
        data_paths (list): List of paths to numpy arrays
        joint_idx (int): Joint index to plot (0-11)
        title_mapping (dict, optional): Dictionary mapping file paths to custom titles
        output_path (str, optional): Path to save the plot
    """
    # Validate joint index
    if joint_idx < 0 or joint_idx > 11:
        raise ValueError("Joint index must be between 0 and 11")
    
    # Load data
    datasets = []
    time_steps = []
    valid_paths = []
    
    for path in data_paths:
        data = load_torque_data(path)
        if data is None:
            continue
            
        datasets.append(data)
        valid_paths.append(path)
        
        # Determine time step based on data shape
        # 1000 timesteps = 5ms step, 5000 timesteps = 1ms step
        if data.shape[0] == 1000:
            dt = 0.005  # 5ms
        elif data.shape[0] == 5000:
            dt = 0.001  # 1ms
        else:
            # Calculate dt assuming 5 seconds total time
            dt = 5.0 / data.shape[0]
            print(f"Warning: Unexpected data shape {data.shape}, using dt={dt}")
        
        time_steps.append(dt)
    
    if len(datasets) == 0:
        print("No valid datasets found!")
        return
    
    # Create subplots
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 2*len(datasets)))
    if len(datasets) == 1:
        axes = [axes]
    
    # Plot each dataset
    for i, (data, dt, path) in enumerate(zip(datasets, time_steps, valid_paths)):
        # Create time array
        time_array = create_time_array(data.shape, dt)
        
        # Extract joint torques
        joint_torques = data[:, joint_idx]
        
        # Plot
        axes[i].plot(time_array, joint_torques, color='#0065bd', linewidth=1.5)
        if i == len(datasets) - 1:
            axes[i].set_xlabel('t [s]')
        else:
            axes[i].tick_params(labelbottom=False)
        if i == len(datasets) - 2:
            axes[i].set_ylabel('Commanded torques', weight='bold')
            
        
        # Set title: use custom mapping if provided, otherwise use default
        if title_mapping and path in title_mapping:
            subplot_title = title_mapping[path]
        else:
            # Default title behavior: directory name or filename
            subplot_title = os.path.basename(os.path.dirname(path)) if os.path.dirname(path) else os.path.basename(path)
        
        axes[i].set_title(subplot_title, weight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, time_duration)  # Always show 5 seconds
    
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Validate number of input files
    if len(data_paths) > 3:
        print("Warning: More than 3 data files provided. Only the first 3 will be plotted.")
        data_paths = data_paths[:3]
    
    # Create plot
    plot_joint_torques(data_paths, joint_idx, title_mapping, output_path)

# Usage:
# 1. Modify the parameters at the top of the script
# 2. Customize the title_mapping dictionary to set custom titles for each data file
# 3. Run: python torque_plotter.py