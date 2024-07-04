# frontal_instability_spod

This repo contanis code to calculate vorticity and conduct Spectral orthogonal decomposition (SPOD) from an input netCDF File which should contain the following structure:

### Dimensions -->

- idimension , jdimension , kdimension , record (UNLIMITED) , timedimension

### Variables -->

- double time(record,timedimension) ;
      
- double u(record,idimension,timedimension,jdimension,kdimension) ;
      
- double v(record,idimension,timedimension,jdimension,kdimension) ;
      
- double x(record,idimension) ;

- double y(record,jdimension) ;
      
- double z(record,kdimension) ;


The Script will calculate the vorticity for each z-plane and create a new variable called 'zeta'. After this, it will carry out SPOD and save the results in the 'results' folder.

### Step 1: Download package
#### Download from Git clone in the terminal

git clone [https://github.com/ESR140/frontal_instability_spod.git](https://github.com/ESR140/frontal_instability_spod.git)

### Step 2: Install prerequisites
Launch a terminal (UNIX) window and change directory to your liking. Run the following command line to install/upgrade the prerequisite Python packages. Ideally, also create a virtual environment for better package management.

```
pip install -r requirements.txt
```

## Step 3: Navigating to the required directory

Navigate to the frontal_instability_spod directory from your current directory.

## Step 4: Insert your file 

Insert your netCDF File in the frontal_instaility_spod directory so that the code can access it

### Step 4: Running the code
You can run the spod_calc.py file in the command terminal with the following command

```
python3 spod_calc.py (your .nc file)
```

The results will be stored in the 'results' folder, and it will contain SPOD for each z-plane in a separate folder called 'zplane_(zplane index)'. Inside each foleder there will be two h5py files, a spectrum plot containing the energy for each corresponding frequency and also the modeshape plot for modes 1 and 3 for frequency inex 2.




# References

[1] He, X., Fang, Z., Rigas, G. & Vahdati, M., (2021). Spectral proper orthogonal decomposition of compressor tip leakage flow. Physics of Fluids, 33(10), 105105. [DOI][preprint]
