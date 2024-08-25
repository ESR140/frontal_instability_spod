import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import argparse
import os
import netCDF4 as nc
from tqdm import tqdm
from scipy import signal
from scipy import interpolate


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
            u = []
            v = []
            num_records = len(nc_file.dimensions['record'])
            for t in range(num_records):
                u.append(nc_file.variables['u'][t, :, 0, :, plane_index])
                v.append(nc_file.variables['v'][t, :, 0, :, plane_index])
            u = np.array(u)
            v = np.array(v)
            xmax = nc_file.variables['x'][0, :].max()
            ymax = nc_file.variables['y'][0, :].max()
            print("Data Loaded....")
            print(f"u Shape: {u.shape}")
            print(f"v Shape: {v.shape}")
            return u, v, xmax, ymax
    except OSError as e:
        print(f"OS error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def normalize_timeavg_field(field):

    timeavg = np.mean(field, axis=0)    

    return timeavg

def compute_2d_ke_spectrum(u, v, Lx, Ly):
    # Get dimensions
    Nt, Nx, Ny = u.shape
    
    hann_window = signal.windows.hann(Nx)[:, np.newaxis] * signal.windows.hann(Ny)[np.newaxis, :]

    # Normalize velocity fields
    u_norm = normalize_timeavg_field(u) * hann_window
    v_norm = normalize_timeavg_field(v) * hann_window
    
    dx, dy = Lx / Nx, Ly / Ny
    print(f"dx is {dx}, dy is {dy}")   

    # Compute 2D FFT
    u_fft = np.fft.fft2(u_norm)
    v_fft = np.fft.fft2(v_norm)
    
    # Wavenumbers
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)
    kx_2d, ky_2d = np.meshgrid(kx, ky, indexing='ij')
    dkx = 2 * np.pi / Lx
    dky = 2 * np.pi / Ly

    k_h = np.sqrt(kx_2d**2 + ky_2d**2)
    dk_min = min(dkx,dky)
    dk_max = max(dkx,dky)
    Nmax = np.sqrt(2) * max(Nx/2, Ny/2)

    print(f"Max kh = {np.max(k_h)}")
    print(f"Min kh = {np.min(k_h)}")
    print(f"dkx is {dkx}")
    print(f"dky is {dky}")
    
    # Compute KE spectral density
    KE_density = (dx * dy * dk_min / (8 * np.pi * np.pi * Nx * Ny)) * (np.abs(u_fft)**2 + np.abs(v_fft)**2)   # Equation 24
    #KE_density = (Lx * Ly * dk_min / (8 * np.pi * np.pi * Nx**2 * Ny**2)) * (np.abs(u_fft)**2 + np.abs(v_fft)**2)   #Equation 20

    
    # Define bins for total wavenumber
    k_max = Nmax * dk_max
    dk = dk_max
    k_bins = np.arange(dk/2 , k_max + dk/2, dk)
    
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

    diff = check_validity(u, v, dx, dy, Lx, Ly, KE_compensated, dk)
    print(f"Difference between avg KE and integral of E(kp)*dkh is : {diff}")

    return k_avg, KE_compensated

def plot_ke_spectrum(kh, Eh, savepath):
    """
    Plots the 2D Normalized Kinetic Energy Spectrum and saves the figure.
    
    Parameters:
    - k: array-like, shape (Nt, num_bins), Wavenumber values for each time step
    - E: array-like, shape (Nt, num_bins), Normalized KE Spectral Density values for each time step
    - savepath: str, Path to save the figure
    """
    print("Plotting Started.....")
    
    # Convert wavenumbers to km^-1
    k_km = k * 1000 / (2 * np.pi)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(k_km, E, linewidth=2.5, alpha=0.7, label='KE Spectra')
    
    # Find the index of k second closest to but less than 0.1
    # start_index = np.where(k_km < 0.1)[0][-1]
    # k_start = k_km[start_index]
    # E_start = E[start_index]
    f = interpolate.interp1d(k_km, E, kind='linear')
    k_start = 0.07
    E_start = f(k_start)
    
    # Add reference slopes
    k_ref = np.logspace(np.log10(k_start), np.log10(k_km.max()), 100)
    
    # Factor to translate the lines up (adjust as needed)
    translation_factor = 2
    
    for slope, color, label in [(-2, 'r', 'k^(-2)'), 
                                (-3, 'g', 'k^(-3)')]:
        E_ref = translation_factor * k_ref**slope * (E_start / (k_start**slope))
        plt.loglog(k_ref, E_ref, f'{color}--', linewidth=2, label=label)
    
    plt.xlabel('Wavenumber (k) [km⁻¹]', fontsize=12)
    plt.ylabel('Normalized KE Spectral Density [m³/s²]', fontsize=12)
    plt.legend(fontsize=10, loc='best', ncol=2)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(savepath, dpi=300)
    #plt.show()
    plt.close()
    print(f"Plot saved to {savepath}")

def plot_compensated_spectra(k, E, savepath):
    """
    Plots the compensated spectra and saves the figure.
    
    Parameters:
    - k: array-like, shape (Nt, num_bins), Wavenumber values for each time step
    - E: array-like, shape (Nt, num_bins), Normalized KE Spectral Density values for each time step
    - savepath: str, Path to save the figure
    """
    print("Plotting Compensated Spectra...")
    
    # Convert wavenumbers to km^-1
    k_km = k * 1000 / (2 * np.pi)
    
    plt.figure(figsize=(10, 6))
    
    for n, color, label in [(2, 'r', 'k²E'), (3, 'g', 'k³E'), (5/3, 'b', 'k⁵ᐟ³E')]:
        compensated_E = E * (k_km**n)
        plt.semilogx(k_km, compensated_E, color=color, linewidth=1, alpha=0.7, label=label)
    
    plt.xlabel('Wavenumber (k) [km⁻¹]', fontsize=12)
    plt.ylabel('Compensated Spectral Density', fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(savepath, dpi=300)
    #plt.show()
    plt.close()
    print(f"Compensated Spectra plot saved to {savepath}")

def check_validity(u, v, dx, dy, Lx, Ly, E, dkh):


    # Left side: spatial average of kinetic energy
    left_side = np.sum((u**2 + v**2) / 2) * dx * dy / (Lx * Ly)

    # Right side: sum of energy spectrum
    right_side = np.sum(E * dkh)

    diff = abs(left_side - right_side)

    return diff



if __name__ == '__main__':
    print("Starting KE Spectra Calculation!")
    args = parse_arguments()
    input_file = args.input_file
    output_path = args.output_path
    plane_index = args.plane_index
    u, v, xmax, ymax = parse_input(input_file, plane_index)
    print("Input Parsed")
    k, E = compute_2d_ke_spectrum(u, v, xmax, ymax)
    print(k.shape)
    print(E.shape)
    plot_ke_spectrum(k, E, os.path.join(output_path, f'KESpectraplaneindex{plane_index}.png'))
    plot_compensated_spectra(k, E, os.path.join(output_path, f'CompSpectraplaneindex{plane_index}.png'))