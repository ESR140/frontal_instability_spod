import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import argparse
import os
import netCDF4 as nc
from tqdm import tqdm

def parse_arguments():
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(description="Calculate 2D KE turbulence spectra from netCDF file")
    parser.add_argument('input_file', type=str, help='Path of the input netCDF file')
    parser.add_argument('output_path', type=str, nargs='?', default=cwd, help='Path to store the spectrum plot (default: current working directory)')
    parser.add_argument('--plane_index', type=int, help='Index of the XY plane for Spectra Calcualtions')
    return parser.parse_args()

def parse_input(input_file, plane_index):
    
    try:
        with nc.Dataset(input_file, 'r') as nc_file:
            u = nc_file.variables['u'][:,:,0,:,plane_index]
            v = nc_file.variables['v'][:,:,0,:,plane_index]
            xmax = nc_file.variables['x'][0,:].max()
            ymax = nc_file.variables['y'][0,:].max()
            
            print("Data Loaded....")
            print(f"u Shape: {u.shape}")
            print(f"v Shape: {v.shape}")

            return u,v,xmax,ymax
        
    except OSError as e:
        print(f"OS error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def normalize_field(field):
    """Normalize the field by subtracting the mean and dividing by the standard deviation."""
    return (field - np.mean(field)) / np.std(field)

def compute_2d_ke_spectrum(u, v, Lx, Ly):
    # Get dimensions
    Nt, Ny, Nx = u.shape
    
    # Initialize arrays to store results
    k_avg_all = []
    KE_compensated_all = []
    
    # Loop over time snapshots
    for t in tqdm(range(Nt), desc='Processing time snapshots'):
        # Extract single time snapshot
        u_t = u[t]
        v_t = v[t]
        
        # Normalize velocity fields
        u_norm = normalize_field(u_t)
        v_norm = normalize_field(v_t)
        
        dx, dy = Lx / Nx, Ly / Ny
        
        # Compute 2D FFT
        u_fft = np.fft.fft2(u_norm)
        v_fft = np.fft.fft2(v_norm)
        
        # Wavenumbers
        kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
        ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)
        kx_2d, ky_2d = np.meshgrid(kx, ky)
        k_h = np.sqrt(kx_2d**2 + ky_2d**2)
        
        # Compute KE spectral density
        KE_density = (Lx * Ly / (4 * np.pi * Nx**2 * Ny**2)) * (np.abs(u_fft)**2 + np.abs(v_fft)**2)
        
        # Define bins for total wavenumber
        k_max = np.max(k_h)
        dk = np.min([2*np.pi/Lx, 2*np.pi/Ly]) # Use minimum as bin size
        k_bins = np.arange(0, k_max + dk, dk)
        
        # Bin the KE density
        KE_binned = np.zeros_like(k_bins[:-1])
        for i in range(len(k_bins)-1):
            mask = (k_h >= k_bins[i]) & (k_h < k_bins[i+1])
            KE_binned[i] = np.sum(KE_density[mask])
        
        # Compute average k for each bin
        k_avg = 0.5 * (k_bins[1:] + k_bins[:-1])
        
        # Apply noise compensation
        count = np.histogram(k_h, bins=k_bins)[0]
        compensation = (2*np.pi*k_avg*dk) / np.maximum(count, 1) # Avoid division by zero
        KE_compensated = KE_binned * compensation
        
        # Store results for this time snapshot
        k_avg_all.append(k_avg)
        KE_compensated_all.append(KE_compensated)
    
    # Convert lists to numpy arrays
    k_avg_all = np.array(k_avg_all)
    KE_compensated_all = np.array(KE_compensated_all)
    
    return k_avg_all, KE_compensated_all

def plot_ke_spectrum(k, E, savepath):
    """
    Plots the 2D Normalized Kinetic Energy Spectrum for every 10th time step and saves the figure.
    
    Parameters:
    - k: array-like, shape (Nt, num_bins), Wavenumber values for each time step
    - E: array-like, shape (Nt, num_bins), Normalized KE Spectral Density values for each time step
    - savepath: str, Path to save the figure
    """

    print("Plotting Started.....")
    plt.figure(figsize=(10, 6))

    # Plot a line for every 10th time step
    num_timesteps = E.shape[0]
    for i in range(0, num_timesteps, 10):
        plt.loglog(k[i], E[i], linewidth=1, alpha=0.7, label=f'Step {i}')

    plt.xlabel('Wavenumber (k)', fontsize=12)
    plt.ylabel('Normalized KE Spectral Density', fontsize=12)

    # Add reference slopes
    k_ref = np.logspace(np.log10(k[0, 1]), np.log10(k[0, -1]), 100)
    E_ref_53 = k_ref**(-5/3) * E[0, 10] / k[0, 10]**(-5/3)
    E_ref_3 = k_ref**(-3) * E[0, 10] / k[0, 10]**(-3)
    plt.loglog(k_ref, E_ref_53, 'r--', linewidth=2, label='k^(-5/3)')
    plt.loglog(k_ref, E_ref_3, 'g--', linewidth=2, label='k^(-3)')

    plt.legend(fontsize=10, loc='best', ncol=2)  # Adjust legend to fit all lines
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()

    # Save the figure
    plt.savefig(savepath, dpi=300)  # Increased DPI for better quality
    plt.show()
    plt.close()

    print(f"Plot saved to {savepath}")

if __name__ == '__main__':
    print("Starting KE Spectra Calculation!")
    args = parse_arguments()
    input_file = args.input_file
    output_path = args.output_path
    plane_index = args.plane_index
    u, v, xmax, ymax = parse_input(input_file, plane_index)
    k, E = compute_2d_ke_spectrum(u, v, xmax, ymax)
    plot_ke_spectrum(k, E, output_path)
