gcc -fPIC -shared -o libonnx_c.so -I onnxruntime-osx-arm64-1.20.1/include/ onnx_gen_interface.c onnxruntime-osx-arm64-1.20.1/lib/libonnxruntime.dylib
python -m numpy.f2py -c -L/Users/mgrecu/PMMCCST/  -m onnx_f90 onnx_interface.f90 my_plugin.so -lonnx_c -lmy_plugin

