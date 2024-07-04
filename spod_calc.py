import netCDF4 as nc
import numpy as np
import argparse
import spod
import os
import pylab
import h5py
import matplotlib.pyplot as plt

# Creating parser to accept input file path in Command Line 
parser = argparse.ArgumentParser(description="Create a NetCDF file with specified dimensions and data.")
parser.add_argument('input_path', type=str, help='The input file path')
args = parser.parse_args()

# Specifying Input File Path
input_filepath = args.input_path
ncfile_in = nc.Dataset(input_filepath, 'r+')

# Reading necessary dimensions and variables from input file
y = ncfile_in.variables['y'][3, :]   
x = ncfile_in.variables['x'][3, :]   
z = ncfile_in.variables['z'][3, :]
dt = ncfile_in.variables['time'][4, :] - ncfile_in.variables['time'][3, :]

#Recording size of each dimension
num_records = len(ncfile_in.dimensions['record'])
num_idimension = len(ncfile_in.dimensions['idimension'])
num_timedimension = len(ncfile_in.dimensions['timedimension'])
num_jdimension = len(ncfile_in.dimensions['jdimension'])
num_kdimension = len(ncfile_in.dimensions['kdimension'])

ymax = np.max(y)
xmax = np.max(x)


# Creats vorticity variable if it does not exist already
if 'zeta' not in ncfile_in.variables:

    zeta_var = ncfile_in.createVariable('zeta', np.float64, ('record', 'idimension', 'timedimension', 'jdimension', 'kdimension'))
    zeta_var.units = 's^-1'
    zeta_var.long_name = 'vorticity'

    for zplaneindex in range(num_kdimension):
        for recindex in range(num_records):
        
            u = ncfile_in.variables['u'][recindex, :, 0, :, zplaneindex]
            v = ncfile_in.variables['v'][recindex, :, 0, :, zplaneindex]

            dvdx = np.zeros((num_idimension, num_jdimension))
            dudy = np.zeros((num_idimension, num_jdimension))

            for j in range(num_jdimension):
                if j == 0:
                    dvdx[:, j] = (v[:, j+1] - v[:, j]) / xmax  # Forward difference at j=0
                elif j == num_jdimension - 1:
                    dvdx[:, j] = (v[:, j] - v[:, j-1]) / xmax  # Backward difference at j=num_jdimension-1
                else:
                    dvdx[:, j] = (v[:, j+1] - v[:, j-1]) / (2 * xmax)  # Central difference for interior points

            for i in range(num_idimension):
                if i == 0:
                    dudy[i, :] = (u[i+1, :] - u[i, :]) / ymax  # Forward difference at i=0
                elif i == num_idimension - 1:
                    dudy[i, :] = (u[i, :] - u[i-1, :]) / ymax  # Backward difference at i=num_idimension-1
                else:
                    dudy[i, :] = (u[i+1, :] - u[i-1, :]) / (2 * ymax)

            zeta = dvdx - dudy

            # Writing zeta data into the new NetCDF file
            zeta_var[recindex, :, 0, :, zplaneindex] = zeta


# Carrying out SPOD and saving the results

num_cols = num_idimension * num_jdimension
data = np.zeros((num_records, num_cols))
grid_x = np.tile(x, num_jdimension)
grid_y = np.repeat(y, num_idimension)

for zplaneindex in range(num_kdimension):

    savepath = f'results/zplane_{zplaneindex}'
    
    # Check if the directory already exists and contains files
    if os.path.exists(savepath) and os.listdir(savepath):
        print(f"SPOD results for z-plane {z[zplaneindex]} m already exist. Skipping computation.")
        continue
    
    os.makedirs(savepath, exist_ok=True)

    for recindex in range(num_records):
        zeta = ncfile_in.variables['zeta'][recindex, :, 0, :, zplaneindex]
        counter = 0
        for i in range(num_idimension):
            for j in range(num_jdimension):
                data[recindex, counter] = zeta[i, j]
                counter += 1
    
    spod.spod(data, dt, savepath)
    print(f"SPOD for z-plane {z[zplaneindex]} m finished!")


# -------------------------------------------------------------------------
# Plot SPOD result
# Figs: 1. f-mode energy;
#       2. mode shape at given mode number and frequency
#       3. animation of original flow field
#       4. animation of reconstructed flow field
# -------------------------------------------------------------------------

# Parameters for plotting:
plot_modes = [[0,2], [2,2]]  # List the modeshapes to be plotted as [[Mi, fi], ...] where Mi = index of the mode, fi = index of the frequency

# Some Utility functions

def figure_format(xtitle, ytitle, zoom, legend):
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.axis(zoom)
    if legend != 'None':
        plt.legend(loc=legend)

def plot_spectrum(L,P,f,save_path):
    # -------------------------------------------------------------------------
    ### 4.1 Energy spectrum
    fig = spod.plot_spectrum(f,L,hl_idx=5)

    # figure format
    figure_format(xtitle='Frequency (Hz)', ytitle='SPOD mode energy',
                zoom=None, legend='best')

    plt.savefig(os.path.join(save_path,'Spectrum.png'), dpi=300, bbox_inches='tight')
    plt.close()

'''def get_modeshape_data(plot_modes, P, x, y, idim, jdim):  # Purpose: Format data for plotting

    grid = np.zeros((idim, jdim, 3))
    grid[..., 0] = np.tile(x, (idim, 1))  
    grid[..., 1] = np.tile(y[:, np.newaxis], (1, jdim))  
    grid[..., 2] = P[fi, :, Mi]

    return grid'''

def plot_modeshape(plot_modes, P, f, x, y, idim, jdim, save_path):
    for i in range(len(plot_modes)):
        Mi = plot_modes[i][0]
        fi = plot_modes[i][1]

        # Create a grid
        X, Y = np.meshgrid(x, y)
        Z = P[fi,:, Mi].reshape(jdim, idim)  # Reshape P appropriately

        # Plot the contour
        fig = plt.figure(figsize=(6,4))
        plt.contourf(X, Y, Z, cmap='viridis')
        plt.text(5.5, 1.7, 'Mode {}, f = {:.2f} Hz'.format(Mi+1, f[fi]), fontsize=14)
        plt.colorbar()
        
        # Save the plot
        plt.savefig(os.path.join(save_path, 'M{}_f{}_p_mode_shape.png'.format(Mi, fi)), dpi=300, bbox_inches='tight')
        plt.close()

# Plotting and saving the figures

for zplaneindex in range(num_kdimension):

    save_path = f'results/zplane_{zplaneindex}'
    SPOD_LPf  = h5py.File(os.path.join(save_path,'SPOD_LPf.h5'),'r')
    L = SPOD_LPf['L'][:,:]    # modal energy E(f, M)
    P = SPOD_LPf['P'][:,:,:]  # mode shape
    f = SPOD_LPf['f'][:]      # frequency
    SPOD_LPf.close()

    plot_spectrum(L,P,f,save_path)
    print(f"Spectrum plotted for z-plane {z[zplaneindex]} m")
    plot_modeshape(plot_modes, P, f, x, y, num_idimension, num_jdimension, save_path)
    print(f"Modeshapes plotted for z-plane {z[zplaneindex]} m")



# Closing both input NetCDF files
ncfile_in.close()


