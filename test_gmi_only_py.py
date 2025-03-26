path="/Users/mgrecu/GPM/ROSES2024/Data/"
import glob
fgmi=glob.glob(path+"1C*GMI*.HDF5")
fcmb=glob.glob(path+"2B.GPM*CORRA*.HDF5")
print(fgmi)
print(fcmb)
import netCDF4 as nc
with nc.Dataset(fgmi[0]) as f:
    lat_gmi_S1=f['S1/Latitude'][:]
    lon_gmi_S1=f['S1/Longitude'][:]
    tc_gmi_S1=f['S1/Tc'][:]
    lat_gmi_S2=f['S2/Latitude'][:]
    lon_gmi_S2=f['S2/Longitude'][:]
    tc_gmi_S2=f['S2/Tc'][:]

with nc.Dataset(fcmb[0]) as f:
    lat=f['KuKaGMI/Latitude'][:]
    lon=f['KuKaGMI/Longitude'][:]
    surf_type=f['KuKaGMI/Input/surfaceType'][:]
    qv=f['KuKaGMI/vaporDensity'][:]
    sk_temp=f['KuKaGMI/skinTemperature'][:]

print(lon.shape)
import numpy as np
norm_param_land=np.load('GMI_ONNX_Models/norm_param_land.npz')

import tb_resample 
tc_s1_resampled = tb_resample.grid_tb(tc_gmi_S1,lon_gmi_S1,lat_gmi_S1,lon,lat)
tc_s2_resampled = tb_resample.grid_tb(tc_gmi_S2,lon_gmi_S2,lat_gmi_S2,lon,lat)

import os
import sys
sys.path.append('/Users/mgrecu/PMMCCST/onnxruntime-osx-arm64-1.20.1/lib/')
os.environ['DYLD_LIBRARY_PATH'] = '/Users/mgrecu/PMMCCST/onnxruntime-osx-arm64-1.20.1/lib/'
os.environ['DYLD_LIBRARY_PATH'] = '/Users/mgrecu/PMMCCST/onnxruntime-osx-arm64-1.20.1/lib/'

print(os.environ['DYLD_LIBRARY_PATH'])
import onnx_f90
print(dir(onnx_f90))
onnx_f90.read_scaler_data()
onnx_f90.init_onnx()

import matplotlib.pyplot as plt
n1=5000
n2=6000
x_qv_enc_2d=np.zeros((n2-n1,49,4))
#vars_string=['tc','sfc_type','sk_temp','oe_wvp','near_sfc_precip','xenc','xenc_prec','xenv_enc']

x_input=np.zeros((150,49,19),dtype=np.float32)
a=np.nonzero(surf_type[n1:n1+150,:]==0)
if len(a[0])/(150*49)>0.5:
    surf_type_b=0
else:
    surf_type_b=1

import onnxruntime as ort
sess_qv_enc = ort.InferenceSession('GMI_ONNX_Models/dense_encoder_land.onnx')
input_name = sess_qv_enc.get_inputs()[0].name
output_name = sess_qv_enc.get_outputs()[0].name
scaler_qv_land=np.load('GMI_ONNX_Models/scaler_land_qv.npz')

land_scaler=np.load('GMI_ONNX_Models/norm_param_150_land.npz')
ocean_scaler=np.load('GMI_ONNX_Models/norm_param_150_ocean.npz')
print(list(land_scaler.keys()))
#onnx_f90.write_pickle_land(land_scaler['tc'],land_scaler['sfc_type'],land_scaler['sk_temp'],land_scaler['oe_wvp'],land_scaler['near_sfc_precip'],land_scaler['xenc'],land_scaler['xenc_prec'],land_scaler['xenv_enc'])
#onnx_f90.write_pickle_ocean(ocean_scaler['tc'],ocean_scaler['sfc_type'],ocean_scaler['sk_temp'],ocean_scaler['oe_wvp'],ocean_scaler['near_sfc_precip'],ocean_scaler['xenc'],ocean_scaler['xenc_prec'],ocean_scaler['xenv_enc'])
#onnx_f90.read_norm_land()

#print(scaler_qv_land['mean'])
#print(scaler_qv_land['std'])
for i in range(n1,n1+150):
    for j in range(49):
        x_qv_enc=onnx_f90.call_dense_qv(sk_temp[i,j],qv[i,j,:],surf_type_b)
        x1=np.array([sk_temp[i,j]]+list(qv[i,j,:]))
        x1=(x1-scaler_qv_land['mean'])/scaler_qv_land['std']
        #print(x1)
        x_qv_enc_py=sess_qv_enc.run([output_name], {input_name: x1[np.newaxis, ...]})[0]
        #print(x_qv_enc)
        #print(x_qv_enc_py)
        #stop
        #break

        x_qv_enc_2d[i-n1,j,:]=x_qv_enc
        x_input[i-n1,j,:9]=tc_s1_resampled[i,j,:]
        x_input[i-n1,j,9:13]=tc_s2_resampled[i,j,:]
        x_input[i-n1,j,13]=surf_type[i,j]
        x_input[i-n1,j,14]=sk_temp[i,j]
        x_input[i-n1,j,15:19]=x_qv_enc

x_output,x_input_sc_f=onnx_f90.call_map_est(x_input,surf_type_b)

x_input_sc=x_input.copy()
x_input_sc[:,:,0:13]=(x_input[:,:,0:13]-land_scaler['tc'][0,:,:,:])/land_scaler['tc'][1,:,:,:]
x_input_sc[:,:,13]=(x_input[:,:,13]-land_scaler['sfc_type'][0,:])/land_scaler['sfc_type'][1,:]
x_input_sc[:,:,14]=(x_input[:,:,14]-land_scaler['sk_temp'][0,:])/land_scaler['sk_temp'][1,:]
x_input_sc[:,:,15:19]=(x_input[:,:,15:19]-land_scaler['xenv_enc'][0,:,:,:])/land_scaler['xenv_enc'][1,:,:,:]
import onnxruntime as ort
sess = ort.InferenceSession('GMI_ONNX_Models/land_model.onnx')
input_name = sess.get_inputs()[0].name
output_name1 = sess.get_outputs()[0].name
output_name2 = sess.get_outputs()[1].name
x_input_T=np.swapaxes(x_input_sc.T,1,2).astype(np.float32)
x_output_onnx = sess.run([output_name1, output_name2], {input_name: x_input_T[np.newaxis, ...]})