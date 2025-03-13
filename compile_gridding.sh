gcc -c -fPIC c_grid_interface.c
gcc -shared -o libclosest.so closest/src/*.o c_grid_interface.o

#f2py -c -m tb_resample tb_grid.f90 c_grid_interface.o libclosest.so
python -m numpy.f2py -c -L/Users/mgrecu/PMMCCST/  -m tb_resample tb_grid.f90 -lclosest
