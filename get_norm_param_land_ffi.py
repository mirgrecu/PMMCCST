

import cffi
ffibuilder = cffi.FFI()

header = """
extern void get_norm_param_(float *tc,float *sfc_type,float *sk_temp,float *oe_wvp,float *near_sfc_precip,float *xenc,float *xenc_prec,float *xenv_enc,char *fname);

"""
module = """
import sys
sys.path.append("/Users/mgrecu/PMMCCST/")
from my_plugin import ffi
import numpy as np


@ffi.def_extern()
def get_norm_param_(ctc,csfc_type,csk_temp,coe_wvp,cnear_sfc_precip,cxenc,cxenc_prec,cxenv_enc,fname):
    py_fname = ffi.string(fname).decode('utf-8')
    npz=np.load(py_fname)
    tc= np.frombuffer(ffi.buffer(ctc,163072*4),np.dtype('f4')).reshape(2,128,49,13)
    tc[:]=npz['tc']
    sfc_type= np.frombuffer(ffi.buffer(csfc_type,12544*4),np.dtype('f4')).reshape(2,128,49)
    sfc_type[:]=npz['sfc_type']
    sk_temp= np.frombuffer(ffi.buffer(csk_temp,12544*4),np.dtype('f4')).reshape(2,128,49)
    sk_temp[:]=npz['sk_temp']
    oe_wvp= np.frombuffer(ffi.buffer(coe_wvp,12544*4),np.dtype('f4')).reshape(2,128,49)
    oe_wvp[:]=npz['oe_wvp']
    near_sfc_precip= np.frombuffer(ffi.buffer(cnear_sfc_precip,12544*4),np.dtype('f4')).reshape(2,128,49)
    near_sfc_precip[:]=npz['near_sfc_precip']
    xenc= np.frombuffer(ffi.buffer(cxenc,50176*4),np.dtype('f4')).reshape(2,128,49,4)
    xenc[:]=npz['xenc']
    xenc_prec= np.frombuffer(ffi.buffer(cxenc_prec,75264*4),np.dtype('f4')).reshape(2,128,49,6)
    xenc_prec[:]=npz['xenc_prec']
    xenv_enc= np.frombuffer(ffi.buffer(cxenv_enc,50176*4),np.dtype('f4')).reshape(2,128,49,4)
    xenv_enc[:]=npz['xenv_enc']


"""
with open("plugin.h", "w") as f:
    f.write(header)

ffibuilder.embedding_api(header)
ffibuilder.set_source("my_plugin", r'''
    #include "plugin.h"
''')

ffibuilder.embedding_init_code(module)
ffibuilder.compile(target="libmy_plugin_2d.dylib", verbose=True)