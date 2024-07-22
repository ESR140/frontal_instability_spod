import numpy as np
import h5py
import os
import time
import psutil
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

def pod(data_matrix, savepath='pod_results.h5', num_modes=None):
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
    Plot the energy of each POD mode from the saved HDF5 file.
    
    Parameters:
    - savepath: Path to the directory where POD results are saved.
    """
    # Load data from HDF5 file
    with h5py.File(os.path.join(savepath,'pod_results.h5'), 'r') as f:
        S = f['singular_values'][:]
    
    # Compute energy of each mode
    energy = S**2
    
    # Plotting energy of each mode
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, len(S) + 1), energy, marker='o', linestyle='-')
    plt.title('Energy of POD Modes')
    plt.xlabel('Mode Index')
    plt.ylabel('Energy')
    plt.grid(True)
    plt.savefig(os.path.join(savepath,'pod_energy_plot'))

def plot_spatial_modes(plotdata, gridx, gridy, savepath, modes_to_plot):
    """
    Plot the spatial modes using the data in plotdata.

    Parameters:
    - flowdata: 3D numpy array with dimensions (num_modes, xdim, ydim)
    - gridx: 2D numpy array representing the x-coordinates of the grid
    - gridy: 2D numpy array representing the y-coordinates of the grid
    - savepath: Path to save the plots
    - modes_to_plot: List of mode indices to plot
    """

    FS = 12

    for i in modes_to_plot:

        '''plt.figure()
        plt.pcolormesh(gridx, gridy, plotdata[i,:,:], cmap=plt.cm.bwr)
        plt.title(f'POD Mode {i}')
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.gca().set_aspect(aspect=aspect_ratio) 
        plt.savefig(os.path.join(savepath, f'pod_modeshape_{i}.png'))
        plt.close()'''

        fig = plt.figure(figsize=(17,5))
        ax = fig.add_subplot(111)

        im = ax.pcolormesh(gridx, gridy, plotdata[i,:,:], cmap= plt.cm.bwr)
        ax.set_xlim(gridx.min(), gridx.max())
        ax.set_ylim(gridy.min(), gridy.max())
        ax.set_xlabel('y [km]', fontsize=FS)
        ax.set_ylabel('x [km]', fontsize=FS)

        text_box_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=1)
        ax.text(gridx.min() + 0.6, gridy.min() + 0.6, 'Mode {}'.format(i+1), fontsize=FS,
            color='black', bbox=text_box_props)
        
        ax.set_title('Modeshape Plot - Mode {}'.format(i + 1), fontsize=FS)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r'$\zeta/f$',size=12)

        plt.savefig(os.path.join(savepath, f'pod_modeshape_{i}.png'))
        plt.close()
        
def plot_reconstruct_flow_field(flowdata, gridx, gridy, savepath, modes_to_plot, fps):
    """
    Plot the sreconstructed flow field for each mode in modes_to_plot and save the video in savepath

    Parameters:
    - plotdata: 4D numpy array with dimensions (num_modes, num_snapshots, xdim, ydim)
    - gridx: 2D numpy array representing the x-coordinates of the grid
    - gridy: 2D numpy array representing the y-coordinates of the grid
    - savepath: Path to save the plots
    - modes_to_plot: List of mode indices to plot
    - duration: Duration of the video
    - fps: frames per second of the video
    """

    num_frames = flowdata.shape[1]
    duration = num_frames / fps

    for i in modes_to_plot:
        
        fig = plt.figure(figsize=(17,5))
        ax = fig.add_subplot(111)

        im = ax.pcolormesh(gridx, gridy, flowdata[i, 0], cmap=plt.cm.bwr, vmin=flowdata[i,:,:,:].min(), vmax=flowdata[i,:,:,:].max())
        ax.set_xlim(gridx.min(), gridx.max())
        ax.set_ylim(gridy.min(), gridy.max())
        ax.set_xlabel('y [km]', fontsize=12)
        ax.set_ylabel('x [km]', fontsize=12)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r'$\zeta/f$',size=12)

        def make_frame(t):

            ax.clear()

            frame_index = int(t*fps) % num_frames

            im = ax.pcolormesh(gridx, gridy, flowdata[i,frame_index], cmap= plt.cm.bwr)
            ax.set_xlim(gridx.min(), gridx.max())
            ax.set_ylim(gridy.min(), gridy.max())
            ax.set_xlabel('y [km]', fontsize=12)
            ax.set_ylabel('x [km]', fontsize=12)

            return mplfig_to_npimage(fig)

        animation = VideoClip(make_frame, duration=duration)
        animation.write_gif(os.path.join(savepath, f'rec_flowfield_modeshape_{i}.gif'), fps=fps)


if __name__ == "__main__":
    
    print('--------------------------------------')  
    print('POD Example Case')
    print('--------------------------------------') 

    num_snapshots = 200
    snapshot_dim = 257*1024
    data_matrix = np.random.rand(num_snapshots, snapshot_dim)  
    x = np.linspace(0, 10, 257)
    y = np.linspace(0, 40, 1024)
    xdim = x.shape[0]
    ydim = y.shape[0]
    num_modes = 10

    savepath = 'results/test'

    # Perform POD and save results
    pod(data_matrix, savepath=savepath, num_modes=num_modes)

    with h5py.File(os.path.join(savepath, 'pod_results.h5'), 'r') as f:
        S = f['singular_values'][:]
        pod_modes = f['pod_modes'][:]       # Since it is a snapshot POD, these are temporal modes
        pod_coeffs = f['pod_coeffs'][:]     # Thus, this pod_coeffs will give respective spatial modes

    gridx, gridy = np.meshgrid(y, x)
    plot_modes = [0, 1, 2]

    modeshape_data = np.zeros((num_modes, xdim, ydim))
    for i in range(num_modes):
        modeshape_data[i, :, :] = pod_coeffs[i, :].reshape(xdim, ydim)

    plot_spatial_modes(modeshape_data, gridx, gridy, savepath, plot_modes)

    reconstruct_flow_data = np.zeros((len(plot_modes), num_snapshots, xdim, ydim))
    for i in range(len(plot_modes)):
        temp = pod_modes[:,i][:, np.newaxis] @ pod_coeffs[i,:][np.newaxis, :]    
        reconstruct_flow_data[i, :, :, :] = temp[:, :].reshape(num_snapshots, xdim, ydim)

    plot_reconstruct_flow_field(reconstruct_flow_data, gridx, gridy, savepath, plot_modes, fps=20)


    print("Modeshape Plot Data Shape: ", modeshape_data.shape)
    print("X Shape: ", xdim)
    print("Y Shape: ", ydim)
    print("Reconstruct Flow field Plot Data: ", reconstruct_flow_data.shape)

