import netCDF4 as nc
import numpy as np
import argparse
import spod
import os
import pylab
import h5py
import matplotlib.pyplot as plt
from dst import dst
from tqdm import tqdm
import time 
import shutil
from datetime import datetime
from pod import pod,plot_pod_energy,plot_spatial_modes, plot_reconstruct_flow_field

'''
___________________________________________________________________________

Section 1: Reading the file and saving required variables 

- The input file should be given as an argument in the command line
- The output will be stored in the results folder
___________________________________________________________________________
'''

# Creating parser to accept input file path in Command Line 
parser = argparse.ArgumentParser(description="Create a NetCDF file with specified dimensions and data.")
parser.add_argument('input_path', type=str, help='The input file path')

#Following are for skipping certain parts of the code. If arg is given, will skip that part. If arg is not given, the section will run by default
parser.add_argument('--skip_vort_calc', action= 'store_true', help='Utility to skip vorticity calculation.')
parser.add_argument('--operation', choices=['spod', 'pod', 'skip'], required=True, help='Specify whether to carry out SPOD, POD, or skip both.')
parser.add_argument('--plot', choices=['spod', 'pod', 'skip'], required=True, help='Specify whether to plot out SPOD, POD, or skip both.')

args = parser.parse_args()

# Specifying Input File Path
input_filepath = args.input_path
ncfile_in = nc.Dataset(input_filepath, 'r+')

# Reading necessary dimensions and variables from input file
x = ncfile_in.variables['x'][3, :]   
y = ncfile_in.variables['y'][3, :]   
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

print("--------------------------------------'")
print("Finished reading variables")
print("--------------------------------------'")

# This specifies the z-plane for which calculations are to be done
kdimlist=[97]

'''
___________________________________________________________________________

Section 2: Vorticity Calculations. 

-This section of the code calculates vorticity, and derivatives are calculated 
using discrete fourier transform method (dst funtion)

___________________________________________________________________________
'''

if not args.skip_vort_calc:          # Will run this section by default

    print("--------------------------------------")
    print("Calculating Vorticity......")

    ncfile_out = nc.Dataset('vorticity.nc', 'r+', format='NETCDF4')

    if  not os.path.exists('vorticity.nc'):
        print("vorticity.nc file does not exist. Creating vorticity.nc file")
        record = ncfile_out.createDimension('record', None)  # Unlimited dimension (can be appended)
        idimension = ncfile_out.createDimension('idimension', num_idimension)  # Fixed size dimension
        timedimension = ncfile_out.createDimension('timedimension', num_timedimension)
        jdimension = ncfile_out.createDimension('jdimension', num_jdimension)
        kdimension = ncfile_out.createDimension('kdimension', num_kdimension)

        zeta_var = ncfile_out.createVariable('zeta', np.float64, ('record', 'idimension', 'timedimension', 'jdimension', 'kdimension',))
    
    else: print("vorticity.nc file exists. Modifying this file.")


    for zplaneindex in tqdm(kdimlist, desc='Vorticity Calculation'):

        for recindex in range(num_records):
        
            u = ncfile_in.variables['u'][recindex, :, 0, :, zplaneindex]
            v = ncfile_in.variables['v'][recindex, :, 0, :, zplaneindex]

            dvdx = np.zeros((num_idimension, num_jdimension))
            dudy = np.zeros((num_idimension, num_jdimension))

            for j in range(num_jdimension):

                dvdx[:,j] = dst(v[:,j],1,xmax,-1) 

            for i in range(num_idimension):

                dudy[i,:]=dst(u[i,:],1,ymax,0)

            zeta = dvdx - dudy

            ncfile_out.variables['zeta'][recindex, :, 0, :, zplaneindex] = zeta
    
    print("Vorticity calculated and saved to vorticity.nc")
    print("--------------------------------------")

else: 
    print("Skipping vorticity calculations")
    print("--------------------------------------")


'''
___________________________________________________________________________

Section 3: Decompostition Calculations 

-This section of the code carries out the SPOD or POD Algorithm based on the user input, and saves it to the results folder

___________________________________________________________________________
'''

if args.operation in ['spod', 'pod']:                 # Will run this section by default

    decomp_type = args.operation

    print("--------------------------------------")
    print(f"{decomp_type} starting ............ ")

    ncfile_out = nc.Dataset('vorticity.nc', 'r+', format='NETCDF4')
    num_cols = num_idimension * num_jdimension
    data = np.zeros((num_records, num_cols, 3))

    for zplaneindex in kdimlist:

        savepath = f'results/zplane_{zplaneindex}'
    
        '''# Check if the directory already exists and contains files
        if os.path.exists(savepath) and os.listdir(savepath):

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backuppath = f'archives/{timestamp}'
            os.makedirs(os.path.dirname(backuppath), exist_ok=True)
            shutil.move(savepath, backuppath)'''
        
        os.makedirs(savepath, exist_ok=True)

        for recindex in range(num_records):
            zeta = ncfile_out.variables['zeta'][recindex, :, 0, :, zplaneindex]
            counter = 0
            for i in range(num_idimension):
                for j in range(num_jdimension):
                    data[recindex, counter,0] = zeta[i, j]
                    data[recindex, counter,1] = x[i]
                    data[recindex, counter,2] = y[j]
                    counter += 1
        
        if decomp_type == 'spod':
            spod.spod(data[:,:,0], dt, savepath)
        elif decomp_type == 'pod':
            pod(data[:,:,0], savepath, num_modes= 10)

    print(f"{decomp_type} Completed! Results saved to /results folder")
    print("--------------------------------------")


elif args.operation == 'skip': 
    print("Skipping calculations")
    print("--------------------------------------")


'''
___________________________________________________________________________

Section 4: Plotting the results 

-This section of the code plots the results of the SPOD with the following plots
    4.1 Spectrum plot of Energy v/s Frequency for each mode
    4.2 Mode Shapes for the first two modes and first two frequencies of highest energy
    4.3 Reconstructed flow field for the first three modes

___________________________________________________________________________
'''

if args.plot in ['spod', 'pod']:

    plot_arg = args.plot

    print("--------------------------------------")
    print("Plotting Started")

    for zplaneindex in kdimlist:

        save_path = f'results/zplane_{zplaneindex}'

        if plot_arg == 'spod':
            SPOD_LPf  = h5py.File(os.path.join(save_path,'SPOD_LPf.h5'),'r')
            L = SPOD_LPf['L'][:,:]    # modal energy E(f, M)
            P = SPOD_LPf['P'][:,:,:]  # mode shape (f,point,M)
            f = SPOD_LPf['f'][:]      # frequency
            SPOD_LPf.close()

            #---------------------------------------
            # 4.1 - Energy Spectrum

            fig = spod.plot_spectrum(f,L,hl_idx=5)
            plt.savefig(os.path.join(save_path,'Spectrum.png'), dpi=300, bbox_inches='tight')
            title = f'Spectrum for Z-plane {z[zplaneindex]:.2f} m'
            plt.title(title)
            plt.close()

            #---------------------------------------
            # 4.2 - Mode Shapes

            plot_modes = [[0,0], [0,1], [1,0], [1,1]]  #Specifies which modes to be plotted in the format [Mi, fi], Mi = Mode index, fi = frequency index

            for i in range(len(plot_modes)):

                Mi = plot_modes[i][0]
                fi = plot_modes[i][1]
                plotdata = np.real(P[fi,:,Mi])
                plotdata = plotdata.reshape(num_idimension, num_jdimension)
                gridx, gridy = np.meshgrid(y, x) / (1000*np.ones((num_idimension, num_jdimension)))  # in [km]
                rangex = gridx.max()
                rangey = gridy.max()
                fcor = 7e-5  # Scaling factor = coriolis factor
                FS =12
                plotdata = plotdata / fcor

                fig = plt.figure(figsize=(17,5))
                ax = fig.add_subplot(111)

                im = ax.pcolormesh(gridx, gridy, plotdata, cmap=plt.cm.bwr)
                ax.set_xlim(gridx.min(), gridx.max())
                ax.set_ylim(gridy.min(), gridy.max())
                ax.set_xlabel('y [km]', fontsize=FS)
                ax.set_ylabel('x [km]', fontsize=FS)
                text_box_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=1)
                ax.text(gridx.min() + 0.03, gridy.min() + 0.01, 'Mode {}, f = {:.5f} Hz'.format(Mi + 1, f[fi]), fontsize=FS,
                    color='black', bbox=text_box_props)
                
                ax.set_title('Modeshape Plot - Mode {}, f = {:.5f} Hz'.format(Mi + 1, f[fi]), fontsize=FS)

                cbar = fig.colorbar(im, ax=ax, orientation='vertical')
                cbar.set_label(r'$\zeta/f$', size=FS)


                plt.savefig(os.path.join(save_path, 'M{}_f{}_p_mode_shape.png'.format(Mi, fi)))
                plt.close()

        if plot_arg == 'pod':

            plot_pod_energy(save_path)

            print(f"Mode energies plotted for z-plane {z[zplaneindex]}")

            #Retrieve Mode Data from the file
            with h5py.File(os.path.join(savepath, 'pod_results.h5'), 'r') as f:
                S = f['singular_values'][:]
                pod_modes = f['pod_modes'][:]       # Since it is a snapshot POD, these are temporal modes
                pod_coeffs = f['pod_coeffs'][:]     # Thus, this pod_coeffs will give respective spatial modes

            gridx, gridy = np.meshgrid(y, x) / (1000*np.ones((num_idimension, num_jdimension)))  # [km]
            plot_modes = [0, 1, 2]
            fcor = 7e-5

            plotdata = np.zeros((len(plot_modes), num_idimension, num_jdimension))
            for i in range(len(plot_modes)):
                plotdata[i, :, :] = pod_coeffs[i, :].reshape(num_idimension, num_jdimension)

            plot_spatial_modes(plotdata, gridx, gridy, savepath, plot_modes)

            print(f"Modeshapes plotted for z-plane {z[zplaneindex]} m")

            reconstruct_flow_data = np.zeros((len(plot_modes), num_records, num_idimension, num_jdimension))
            for i in range(len(plot_modes)):
                temp = pod_modes[:,i][:, np.newaxis] @ pod_coeffs[i,:][np.newaxis, :]    
                reconstruct_flow_data[i, :, :, :] = temp[:, :].reshape(num_records, num_idimension, num_jdimension)
            reconstruct_flow_data = reconstruct_flow_data / fcor

            plot_reconstruct_flow_field(reconstruct_flow_data, gridx, gridy, savepath, plot_modes, fps=20)

            print(f"Reconstructed flow field plotted for z-plane {z[zplaneindex]} m")
            

    print("Plotting finished and plots saved in the /results folder")
    print("--------------------------------------")

elif args.plot == 'skip':
    print("--------------------------------------")
    print("Plotting Skipped")
    print("--------------------------------------")

# Closing both input NetCDF files
ncfile_in.close()
