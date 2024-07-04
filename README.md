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




# References

[1] He, X., Fang, Z., Rigas, G. & Vahdati, M., (2021). Spectral proper orthogonal decomposition of compressor tip leakage flow. Physics of Fluids, 33(10), 105105. [DOI][preprint]
