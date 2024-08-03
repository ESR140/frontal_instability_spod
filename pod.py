import netCDF4 as nc
import os
import argparse
import numpy as np
import time
import psutil
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def parse_plane_type(input_file, plane_type):

    try:
        with nc.Dataset(input_file, 'r') as nc_file:
            dimensions = {dim: len(nc_file.dimensions[dim]) for dim in ['record', 'idimension', 'timedimension' , 'jdimension', 'kdimension']}
            zeta = nc_file.variables['zeta']
            
            plane_configs = {
                'XY': ('idimension', 'jdimension', 'x', 'y', lambda z: z[:, :, :, :, 0]),
                'YZ': ('jdimension', 'kdimension', 'y', 'z', lambda z: z[:, 0, :, :, :]),
                'XZ': ('idimension', 'kdimension', 'x', 'z', lambda z: z[:, :, :, 0, :])
            }
            
            if plane_type not in plane_configs:
                raise ValueError(f"Unsupported plane type: {plane_type}")
            
            dim1_name, dim2_name, var1_name, var2_name, zeta_slice = plane_configs[plane_type]
            
            return {
                'dim1': dimensions[dim1_name],
                'dim2': dimensions[dim2_name],
                'var1': nc_file.variables[var1_name][:],
                'var2': nc_file.variables[var2_name][:],
                'zeta': zeta_slice(zeta)[:]
            }
    except OSError as e:
        print(f"OS error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def parse_arguments():
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(description="Calculate vorticity from netCDF file")
    parser.add_argument('input_path', type=str, help='Path of the input netCDF file')
    parser.add_argument('output_path', type=str, nargs='?', default=cwd, help='Path to store the vorticity file (default: current working directory)')
    parser.add_argument('--plane_type', choices=['XY', 'XZ', 'YZ'], required=True, help='Type of plane to calculate the vorticity for')
    #parser.add_argument('--plane_index', type=int, required=True, help='Index of the plane for vorticity calculation')
    return parser.parse_args()

def DataLoader(plane_info):
        
    # Extract necessary data
    zeta = plane_info['zeta'][:]
    var1 = plane_info['var1'][:]
    var2 = plane_info['var2'][:]

    var1_mesh, var2_mesh = np.meshgrid(var1, var2)
    
    zeta_reshaped = zeta.reshape(zeta.shape[0], -1)
    var1_reshaped = var1_mesh.ravel()
    var2_reshaped = var2_mesh.ravel()
    
    data = np.stack([zeta_reshaped, 
                     np.tile(var1_reshaped, (zeta_reshaped.shape[0], 1)),
                     np.tile(var2_reshaped, (zeta_reshaped.shape[0], 1))], 
                    axis=-1)
    
    data = np.swapaxes(data, 0, 1)

    print("Data loaded for computations!")
    print(f"Shape of loaded Data: {data.shape}")
    return data

def pod(data_matrix, savepath, num_modes=None):
    """
    Perform Proper Orthogonal Decomposition (POD) on a given data matrix using Singular Value Decomposition (SVD).
    
    Parameters:
    - data_matrix: 2D numpy array where each row represents a data snapshot.
    - num_modes: Number of POD modes to compute. If None, computes num_modes = minimum dimension of the data_matrix
    - savepath: Path to save the results in an HDF5 file. 
    """

    time_start = time.time()

    print('--------------------------------------')  
    print('POD starts...')
    print('--------------------------------------') 

    data_pod = data_matrix - np.mean(data_matrix, axis=0)

    # Perform SVD on the data matrix
    U, S, Vt = np.linalg.svd(data_pod, full_matrices=False)

    print("U shape: ", U.shape)
    print("S shape: ", S.shape)
    print("Vt shape: ", Vt.shape)
    
    # Compute POD modes and coefficients
    if num_modes is None:
        num_modes = min(data_pod.shape)

    pod_modes = U[:, :num_modes]
    pod_coeffs = np.diag(S)[:num_modes, :num_modes] @ Vt[:num_modes, :]


    print("pod_modes shape: ", pod_modes.shape)
    print("pod_coeffs shape: ", pod_coeffs.shape)

    # Save to HDF5 file
    os.makedirs(savepath, exist_ok=True)
    with h5py.File(os.path.join(savepath, 'pod_results.h5'), 'w') as f:
        f.create_dataset('pod_modes', data=pod_modes)
        f.create_dataset('pod_coeffs', data=pod_coeffs)
        f.create_dataset('singular_values', data=S)


    print("POD results saved to .h5 file")
    print("--------------------------------------")

    process = psutil.Process(os.getpid())
    RAM_usage = process.memory_info().rss / 1024 ** 3  # unit in GBs
        
    time_end = time.time()
    print('--------------------------------------')
    print('POD finished')
    print('Memory usage: %.2f GB' % RAM_usage)
    print('Run time    : %.2f s' % (time_end - time_start))
    print('--------------------------------------')

def plot_pod_energy(savepath):
    """
    Plot the energy of each POD mode and the cumulative energy from the saved HDF5 file.
    
    Parameters:
    - savepath: Path to the directory where POD results are saved.
    """
    # Load data from HDF5 file
    with h5py.File(os.path.join(savepath, 'pod_results.h5'), 'r') as f:
        S = f['singular_values'][:]
    
    # Compute energy of each mode
    energy = S**2
    
    # Compute cumulative energy
    cumulative_energy = np.cumsum(energy)
    total_energy = np.sum(energy)
    cumulative_energy_percentage = cumulative_energy / total_energy * 100

    # Plotting energy of each mode
    plt.figure(figsize=(12, 10))
    
    # Energy plot
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(1, len(S) + 1), energy, marker='o', linestyle='-')
    plt.title('Energy of POD Modes')
    plt.xlabel('Mode Index')
    plt.ylabel('Energy')
    plt.grid(True)
    
    # Cumulative energy plot
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(1, len(S) + 1), cumulative_energy_percentage, marker='o', linestyle='-')
    plt.title('Cumulative Energy of POD Modes')
    plt.xlabel('Mode Index')
    plt.ylabel('Cumulative Energy (%)')
    plt.grid(True)
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'pod_energy_plots.png'))
    plt.close()

    print(f"Energy plots saved to {os.path.join(savepath, 'pod_energy_plots.png')}")

def plot_spatial_modes_separate(plane_info, savepath, modes_to_plot, plane_type):
    with h5py.File(os.path.join(savepath, 'pod_results.h5'), 'r') as f:
        S = f['singular_values'][:]
        pod_modes = f['pod_modes'][:]

    var1 = plane_info['var1'][:] / 1000  # Convert to km
    var2 = plane_info['var2'][:] / 1000  # Convert to km

    # Set labels and grid orientation based on plane type
    if plane_type == 'XY':
        xlabel, ylabel = 'Y [km]', 'X [km]'
        gridx, gridy = np.meshgrid(var2, var1)  # Y on horizontal, X on vertical
    elif plane_type == 'YZ':
        xlabel, ylabel = 'Y [km]', 'Z [km]'
        gridx, gridy = np.meshgrid(var1, var2)  # Y on horizontal, Z on vertical
    elif plane_type == 'XZ':
        xlabel, ylabel = 'X [km]', 'Z [km]'
        gridx, gridy = np.meshgrid(var1, var2)  # X on horizontal, Z on vertical
    else:
        raise ValueError(f"Unknown plane type: {plane_type}")

    # Calculate aspect ratio
    aspect_ratio = (gridx.max() - gridx.min()) / (gridy.max() - gridy.min())

    # Set figure size based on aspect ratio
    base_size = 12  # Base size for the longer dimension
    if aspect_ratio > 1:
        figsize = (base_size, base_size / aspect_ratio)
    else:
        figsize = (base_size * aspect_ratio, base_size)

    FS = 12

    # Determine global min and max values
    vmin, vmax = np.inf, -np.inf
    for i in modes_to_plot:
        plotdata = pod_modes[:, i].reshape(gridy.shape)
        vmin = min(vmin, plotdata.min())
        vmax = max(vmax, plotdata.max())

    # Make vmin and vmax symmetric around zero
    vmax = max(abs(vmin), abs(vmax))
    vmin = -vmax

    for i in modes_to_plot:
        plotdata = pod_modes[:, i].reshape(gridy.shape)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(gridx, gridy, plotdata, cmap=plt.cm.bwr, shading='auto', vmin=vmin, vmax=vmax)
        ax.set_xlim(gridx.min(), gridx.max())
        ax.set_ylim(gridy.min(), gridy.max())
        ax.set_xlabel(xlabel, fontsize=FS)
        ax.set_ylabel(ylabel, fontsize=FS)

        # Add mode number as text box
        text_box_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.8)
        ax.text(0.05, 0.95, f'Mode {i+1}', transform=ax.transAxes, fontsize=FS,
                verticalalignment='top', bbox=text_box_props)

        ax.set_title(f'Spatial Mode {i + 1} - {plane_type} Plane', fontsize=FS)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r'$\zeta/f$', size=FS)

        # Set aspect ratio of the plot
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(os.path.join(savepath, f'pod_modeshape_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Spatial mode plots saved in {savepath}")

def plot_spatial_modes(plane_info, savepath, modes_to_plot, plane_type):
    with h5py.File(os.path.join(savepath, 'pod_results.h5'), 'r') as f:
        S = f['singular_values'][:]
        pod_modes = f['pod_modes'][:]

    var1 = plane_info['var1'][:] / 1000  # Convert to km
    var2 = plane_info['var2'][:] / 1000  # Convert to km

    # Set labels and grid orientation based on plane type
    if plane_type == 'XY':
        xlabel, ylabel = 'Y [km]', 'X [km]'
        gridx, gridy = np.meshgrid(var2, var1)  # Y on horizontal, X on vertical
    elif plane_type == 'YZ':
        xlabel, ylabel = 'Y [km]', 'Z [km]'
        gridx, gridy = np.meshgrid(var1, var2)  # Y on horizontal, Z on vertical
    elif plane_type == 'XZ':
        xlabel, ylabel = 'X [km]', 'Z [km]'
        gridx, gridy = np.meshgrid(var1, var2)  # X on horizontal, Z on vertical
    else:
        raise ValueError(f"Unknown plane type: {plane_type}")

    # Calculate aspect ratio
    aspect_ratio = (gridx.max() - gridx.min()) / (gridy.max() - gridy.min())
    
    # Determine subplot layout
    n_modes = len(modes_to_plot)
    n_cols = min(2, n_modes)  # Max 3 columns
    n_rows = (n_modes + n_cols - 1) // n_cols

    # Set figure size
    base_size = 10  # Base size for each subplot
    figsize = (base_size * n_cols * aspect_ratio, base_size * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(f'Spatial Modes - {plane_type} Plane', fontsize=16)

    FS = 10
    vmin, vmax = np.inf, -np.inf  # Initialize for finding global min and max

    # First pass: determine global min and max
    for mode in modes_to_plot:
        plotdata = pod_modes[:, mode].reshape(gridy.shape)
        vmin = min(vmin, plotdata.min())
        vmax = max(vmax, plotdata.max())

    # Make vmin and vmax symmetric around zero
    vmax = max(abs(vmin), abs(vmax))
    vmin = -vmax

    # Second pass: plot data
    for idx, mode in enumerate(modes_to_plot):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        plotdata = pod_modes[:, mode].reshape(gridy.shape)

        im = ax.pcolormesh(gridx, gridy, plotdata, cmap=plt.cm.bwr, shading='auto', vmin=vmin, vmax=vmax)
        ax.set_xlim(gridx.min(), gridx.max())
        ax.set_ylim(gridy.min(), gridy.max())
        ax.set_xlabel(xlabel, fontsize=FS)
        ax.set_ylabel(ylabel, fontsize=FS)

        ax.set_title(f'Mode {mode + 1}', fontsize=FS)
        ax.set_aspect('equal')

    # Remove any unused subplots
    for idx in range(n_modes, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    # Add a common colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'$\zeta/f$', size=FS)

    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])  # Adjust layout to accommodate suptitle and colorbar
    plt.savefig(os.path.join(savepath, f'pod_modeshapes_subplots.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Spatial modes plot saved as 'pod_modeshapes_subplots.png' in {savepath}")

if __name__ == '__main__':

    args = parse_arguments()
    input_file = args.input_path
    plane_type = args.plane_type
    output_path = args.output_path
    print(f"Input Path: {input_file}")
    parse_plane_type(input_file, plane_type)
    plane_info = parse_plane_type(input_file, plane_type)
    data = DataLoader(plane_info)
    pod(data[:,:,0].filled(), output_path, num_modes=3)
    plot_pod_energy(output_path)
    plot_spatial_modes_separate(plane_info, output_path, [0,1,2], plane_type)
    plot_spatial_modes(plane_info, output_path, [0, 1, 2], plane_type)  
    
