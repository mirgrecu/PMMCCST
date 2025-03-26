gfortran -c -fPIC scaler_def.f90
gcc -fPIC -shared -o libonnx_c.so -I onnxruntime-osx-arm64-1.20.1/include/ onnx_gen_interface.c scaler_def.o onnxruntime-osx-arm64-1.20.1/lib/libonnxruntime.dylib

python -m numpy.f2py -c -I/Users/mgrecu/PMMCCST/ -L/Users/mgrecu/PMMCCST/  -m onnx_f90 onnx_interface.f90 read_pickles.f90 write_pickle.f90 -lonnx_c

