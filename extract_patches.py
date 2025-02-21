import glob
import os
import netCDF4 as nc
from global_land_mask import globe

ny=32
import tqdm
import numpy as np
import pyresample
for im in range(4,12):
    fs=glob.glob("/data5/sringeru/1C_GMI/2020/%2.2i/*HDF5"%(im+1))
    fs=sorted(fs)
    tb_patches=[]
    x_lonL=[]
    x_latL=[]
    col_wv_OE_L=[]
    near_sfc_rain_L=[]
    col_cloud_liquid_L=[]
    for f in tqdm.tqdm(fs[:100]):
        s=f
        parts = s.split('-')
        date_part = parts[2]
        orbit_part = parts[-1]
        
        year = int(date_part[2:6])
        month = int(date_part[6:8])
        day = int(date_part[8:10])
        orbit = int(orbit_part.split('.')[1][:])
        fname_corra_wcard='/data5/sringeru/ITE_790/%4.4i/%2.2i/2B*.%4.4i%2.2i%2.2i-*.%6.6i.*.HDF5'%(year,month,year,month,day,orbit)
        fs_corra=glob.glob(fname_corra_wcard)
        #print(fs_corra[0])
        #stop
        if len(fs_corra)!=1:
            continue
        with nc.Dataset(fs_corra[0]) as fh:
            n_max_dpr=fh['KuGMI/Latitude'][:,24].shape[0]
            n_min_dpr=0
            near_sfc_rain_corra=fh['KuKaGMI/nearSurfPrecipTotRate'][n_min_dpr:n_max_dpr,:]
            emiss_corra=fh['KuKaGMI/surfEmissivity'][n_min_dpr:n_max_dpr,:,:]
            column_water_vapor_OE=fh['KuKaGMI/OptEst/OEcolumnWaterVapor'][n_min_dpr:n_max_dpr,:]
            column_cloud_liquid_OE=fh['KuKaGMI/OptEst/OEcolumnCloudLiqWater'][n_min_dpr:n_max_dpr,:]
            lat_dpr=fh['KuKaGMI/Latitude'][n_min_dpr:n_max_dpr,:]
            lon_dpr=fh['KuKaGMI/Longitude'][n_min_dpr:n_max_dpr,:]
            surf_emiss_corra=fh['KuKaGMI/OptEst/OEsurfEmissivity'][n_min_dpr:n_max_dpr,:]
            surface_type_corra=fh['KuKaGMI/Input/surfaceType'][n_min_dpr:n_max_dpr,:]
            precip_type_corra=fh['KuKaGMI/Input/precipitationType'][n_min_dpr:n_max_dpr,:]
            precip_type_corra=(precip_type_corra/1e7).astype(np.int32)
            skin_temperature_corra=fh['KuKaGMI/skinTemperature'][n_min_dpr:n_max_dpr,:]

        with nc.Dataset(f) as ds:
            tc=ds['S1/Tc'][:]
            lon=ds['S1/Longitude'][:]
            lat=ds['S1/Latitude'][:]
            land_mask=globe.is_land(lat[:,:],lon[:,:])
            land_mask_int=land_mask.astype(np.int8)
            nchunks=tc.shape[0]//128
            for i in range(nchunks):
                chunk=tc[i*128:(i+1)*128,111-ny:111+ny,:]
                land_pixels=land_mask_int[i*128:(i+1)*128,111-ny:111+ny].sum()
                if chunk.data.min()<0:
                    continue
                a=np.nonzero(chunk<0)[0]
                if land_pixels/(128*64)<0.001 and len(a)==0:
                    
                    try:    
                        grid_def = pyresample.geometry.GridDefinition(lons=lon.data[i*128:(i+1)*128,111-ny:111+ny], lats=lat.data[i*128:(i+1)*128,111-ny:111+ny])
                        rad_inf=6000.0
                        swath_def = pyresample.geometry.SwathDefinition(lons=lon_dpr.data, lats=lat_dpr.data)
                        col_wv_OE_S1 = pyresample.kd_tree.resample_gauss(swath_def, column_water_vapor_OE[:,:], grid_def, radius_of_influence=rad_inf, fill_value=-999, sigmas=4000)
                        near_sfc_rain_corra_S1 = pyresample.kd_tree.resample_gauss(swath_def, near_sfc_rain_corra[:,:], grid_def, radius_of_influence=rad_inf, fill_value=-999, sigmas=4000)
                        column_cloud_liquid_S1 = pyresample.kd_tree.resample_gauss(swath_def, column_cloud_liquid_OE[:,:], grid_def, radius_of_influence=rad_inf, fill_value=-999, sigmas=4000)
                        
                    except:
                        print( f, fs_corra[0])
                        print(n_max_dpr)
                        continue
                    tb_patches.append(chunk.copy())
                    x_lonL.append(lon[i*128:(i+1)*128,111-ny:111+ny])
                    x_latL.append(lat[i*128:(i+1)*128,111-ny:111+ny])
                    col_wv_OE_L.append(col_wv_OE_S1.copy())
                    near_sfc_rain_L.append(near_sfc_rain_corra_S1.copy())
                    col_cloud_liquid_L.append(column_cloud_liquid_S1.copy())
                #break
        #break
    tb_patches=np.array(tb_patches)
    x_lonL=np.array(x_lonL)
    x_latL=np.array(x_latL)
    col_wv_OE_L=np.array(col_wv_OE_L)
    near_sfc_rain_L=np.array(near_sfc_rain_L)
    col_cloud_liquid_L=np.array(col_cloud_liquid_L)
    print(len(tb_patches))
    np.savez_compressed('tb_patches_2020_%2.2i.npz'%(im+1),tb_patches=tb_patches.astype(dtype=np.float32), x_lonL=x_lonL.astype(dtype=np.float16),x_latL=x_latL.astype(dtype=np.float16), col_wv_OE=col_wv_OE_L.astype(dtype=np.float32), near_sfc_rain=near_sfc_rain_L.astype(dtype=np.float32), col_cloud_liquid=col_cloud_liquid_L.astype(dtype=np.float32))
    
    #break
