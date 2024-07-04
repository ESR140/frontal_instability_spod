# frontal_instability_spod

This repo contanis code to calculate vorticity and conduct Spectral orthogonal decomposition (SPOD) from an input netCDF File which should contain the following structure:

Dimensions -->
idimension =  ;
jdimension =  ;
kdimension =  ;
record = UNLIMITED ;
timedimension = 1 ;

Variables -->
double time(record,timedimension) ;
      time:units = "s" ;
double u(record,idimension,timedimension,jdimension,kdimension) ;
      u:units = "m/s" ;
double v(record,idimension,timedimension,jdimension,kdimension) ;
      v:units = "m/s" ;
double x(record,idimension) ;
      x:units = "m" ;
double y(record,jdimension) ;
      y:units = "m" ;
double z(record,kdimension) ;
      z:units = "m" ;


The Script will calculate the vorticity for each z-plane and create a new variable called 'zeta'. After this, it will carry out SPOD and save the results in the 'results' folder.




# References

[1] He, X., Fang, Z., Rigas, G. & Vahdati, M., (2021). Spectral proper orthogonal decomposition of compressor tip leakage flow. Physics of Fluids, 33(10), 105105. [DOI][preprint]
