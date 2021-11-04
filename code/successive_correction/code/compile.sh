#! /bin/sh
 

gfortran su_correction.f90 -fastsse -I${NETCDF}/include -o su_correction.exe -L${NETCDF}/lib -lnetcdf -lnetcdff
