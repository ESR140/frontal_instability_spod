import argparse
import os
import netCDF4 as nc
import numpy as np
from dst import dst
from tqdm import tqdm

def calc_vort_XY(ncfile_in, output_path, plane_type, plane_index):
    savename = f'vorticity_{plane_type}_{plane_index}.nc'
    output_file = os.path.join(output_path, savename)

    # Read dimensions and variables
    dimensions = {dim: len(ncfile_in.dimensions[dim]) for dim in ['record', 'idimension', 'timedimension', 'jdimension']}
    x = ncfile_in.variables['x'][3, :]
    y = ncfile_in.variables['y'][3, :]
    z = ncfile_in.variables['z'][3, :]
    xmax, ymax = np.max(x), np.max(y)

    print("--------------------------------------")
    print("Finished reading variables")
    print("--------------------------------------")
    print(f"Calculating Vorticity for plane {plane_type} {plane_index} (index)")

    # Create output file and dimensions
    with nc.Dataset(output_file, 'w', format='NETCDF4') as vortfile:
        vortfile.createDimension('record', None)  # Unlimited dimension (can be appended)
        vortfile.createDimension('idimension', dimensions['idimension'])
        vortfile.createDimension('timedimension', dimensions['timedimension'])
        vortfile.createDimension('jdimension', dimensions['jdimension'])
        vortfile.createDimension('kdimension', 1)
        
        zeta_var = vortfile.createVariable('zeta', np.float64, ('record', 'idimension', 'timedimension', 'jdimension', 'kdimension'))
        x_var = vortfile.createVariable('x', np.float64, ('idimension',))
        y_var = vortfile.createVariable('y', np.float64, ('jdimension',))
        z_var = vortfile.createVariable('z', np.float64, ('kdimension',))
        x_var[:] = x
        y_var[:] = y
        z_var[:] = z[plane_index]

        # Main calculation loop
        for recindex in tqdm(range(dimensions['record']), desc='Vorticity Calculation -> Iterating through records'):
            u = ncfile_in.variables['u'][recindex, :, 0, :, plane_index]
            v = ncfile_in.variables['v'][recindex, :, 0, :, plane_index]
            
            dvdx = np.zeros((dimensions['idimension'], dimensions['jdimension']))
            dudy = np.zeros((dimensions['idimension'], dimensions['jdimension']))
            
            for j in range(dimensions['jdimension']):
                dvdx[:,j] = dst(v[:,j], 1, xmax, -1)
            
            for i in range(dimensions['idimension']):
                dudy[i,:] = dst(u[i,:], 1, ymax, 0)
            
            zeta = dvdx - dudy
            zeta_var[recindex, :, 0, :, 0] = zeta

    print(f"Vorticity calculated and saved to {savename}")
    print("--------------------------------------")

def calc_vort_XZ(ncfile_in, output_path, plane_type, plane_index):

    savename = f'vorticity_{plane_type}_{plane_index}.nc'
    output_file = os.path.join(output_path, savename)

    # Read dimensions and variables
    dimensions = {dim: len(ncfile_in.dimensions[dim]) for dim in ['record', 'idimension', 'timedimension', 'jdimension', 'kdimension']}
    x = ncfile_in.variables['x'][3, :]
    y = ncfile_in.variables['y'][3, :]
    z = ncfile_in.variables['z'][3, :]
    xmax, ymax = np.max(x), np.max(y)

    print("--------------------------------------")
    print("Finished reading variables")
    print("--------------------------------------")
    print(f"Calculating Vorticity for plane {plane_type} {plane_index} (index)")

    # Create output file and dimensions
    with nc.Dataset(output_file, 'w', format='NETCDF4') as vortfile:
        vortfile.createDimension('record', None)  # Unlimited dimension (can be appended)
        vortfile.createDimension('idimension', dimensions['idimension'])
        vortfile.createDimension('timedimension', dimensions['timedimension'])
        vortfile.createDimension('jdimension', 1)
        vortfile.createDimension('kdimension', dimensions['kdimension'])
        
        zeta_var = vortfile.createVariable('zeta', np.float64, ('record', 'idimension', 'timedimension', 'jdimension', 'kdimension'))
        x_var = vortfile.createVariable('x', np.float64, ('idimension',))
        y_var = vortfile.createVariable('y', np.float64, ('jdimension',))
        z_var = vortfile.createVariable('z', np.float64, ('kdimension',))
        x_var[:] = x
        y_var[:] = y[plane_index]
        z_var[:] = z

        # Main calculation loop
        for recindex in tqdm(range(dimensions['record']), desc='Vorticity Calculation -> Iterating through records'):
            zeta_buffer = np.zeros((dimensions['idimension'], dimensions['timedimension'], 1, dimensions['kdimension']))
            
            for k in tqdm(range(dimensions['kdimension']), desc='                         Iterating through z-values', leave=False):
                u = ncfile_in.variables['u'][recindex, :, 0, :, k]
                v = ncfile_in.variables['v'][recindex, :, 0, :, k]
                
                dudy = np.array([dst(u[i,:], 1, ymax, 0)[plane_index] for i in range(dimensions['idimension'])])
                dvdx = dst(v[:,plane_index], 1, xmax, -1)
                
                zeta = dvdx - dudy
                zeta_buffer[:, :, 0, k] = zeta[:, np.newaxis]
            
            zeta_var[recindex] = zeta_buffer

    print(f"Vorticity calculated and saved to {savename}")
    print("--------------------------------------")

def calc_vort_YZ(ncfile_in, output_path, plane_type, plane_index):

    savename = f'vorticity_{plane_type}_{plane_index}.nc'
    output_file = os.path.join(output_path, savename)

    # Read dimensions and variables
    dimensions = {dim: len(ncfile_in.dimensions[dim]) for dim in ['record', 'idimension', 'timedimension', 'jdimension', 'kdimension']}
    x = ncfile_in.variables['x'][3, :]
    y = ncfile_in.variables['y'][3, :]
    z = ncfile_in.variables['z'][3, :]
    xmax, ymax = np.max(x), np.max(y)

    print("--------------------------------------")
    print("Finished reading variables")
    print("--------------------------------------")
    print(f"Calculating Vorticity for plane {plane_type} {plane_index} (index)")

    # Create output file and dimensions
    with nc.Dataset(output_file, 'w', format='NETCDF4') as vortfile:
        vortfile.createDimension('record', None)  # Unlimited dimension (can be appended)
        vortfile.createDimension('idimension', 1)
        vortfile.createDimension('timedimension', dimensions['timedimension'])
        vortfile.createDimension('jdimension', dimensions['jdimension'])
        vortfile.createDimension('kdimension', dimensions['kdimension'])
        
        zeta_var = vortfile.createVariable('zeta', np.float64, ('record', 'idimension', 'timedimension', 'jdimension', 'kdimension'))
        x_var = vortfile.createVariable('x', np.float64, ('idimension',))
        y_var = vortfile.createVariable('y', np.float64, ('jdimension',))
        z_var = vortfile.createVariable('z', np.float64, ('kdimension',))
        x_var[:] = x[plane_index]
        y_var[:] = y
        z_var[:] = z

        # Main calculation loop
        for recindex in tqdm(range(dimensions['record']), desc='Vorticity Calculation -> Iterating through records'):
            zeta_buffer = np.zeros((1, dimensions['timedimension'], dimensions['jdimension'], dimensions['kdimension']))
            
            for k in tqdm(range(dimensions['kdimension']), desc='                         Iterating through z-values', leave=False):
                u = ncfile_in.variables['u'][recindex, :, 0, :, k]
                v = ncfile_in.variables['v'][recindex, :, 0, :, k]
                
                dvdx = np.array([dst(v[:, j], 1, xmax, -1)[plane_index] for j in range(dimensions['jdimension'])])
                dudy = dst(u[plane_index, :], 1, ymax, 0)
                
                zeta_buffer[0, :, :, k] = dvdx - dudy
            
            zeta_var[recindex, :, 0, :, :] = zeta_buffer

    print(f"Vorticity calculated and saved to {savename}")
    print("--------------------------------------")

def parse_arguments():
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(description="Calculate vorticity from netCDF file")
    parser.add_argument('input_path', type=str, help='Path of the input netCDF file')
    parser.add_argument('output_path', type=str, nargs='?', default=cwd, help='Path to store the vorticity file (default: current working directory)')
    parser.add_argument('--plane_type', choices=['XY', 'XZ', 'YZ'], required=True, help='Type of plane to calculate the vorticity for')
    parser.add_argument('--plane_index', type=int, required=True, help='Index of the plane for vorticity calculation')
    return parser.parse_args()

def calculate_vorticity(ncfile, output_path, plane_type, plane_index):

    vorticity_functions = {
        'XY': calc_vort_XY,
        'XZ': calc_vort_XZ,
        'YZ': calc_vort_YZ
    }
    
    vorticity_function = vorticity_functions.get(plane_type)
    if vorticity_function:
        vorticity_function(ncfile, output_path, plane_type, plane_index)
    else:
        raise ValueError(f"Invalid plane type: {plane_type}")

def check_output_file(output_path, plane_type, plane_index):
    output_file = os.path.join(output_path, f'vorticity_{plane_type}_{plane_index}.nc')
    if os.path.exists(output_file):
        print(f"File vorticity_{plane_type}_{plane_index}.nc already exists.")
        user_input = input("Do you want to overwrite it? (y/n): ").lower().strip()
        if user_input != 'y':
            print("Operation cancelled. Exiting...")
            return False
    print("--------------------------------------")
    print(f"Creating / Overwriting file: vorticity_{plane_type}_{plane_index}.nc")
    return True

def main():

    args = parse_arguments()
    
    # if not check_output_file(args.output_path, args.plane_type, args.plane_index):
    #     return

    with nc.Dataset(args.input_path, 'r') as ncfile_in:
        calculate_vorticity(ncfile_in, args.output_path, args.plane_type, args.plane_index)

if __name__ == "__main__":
    main()