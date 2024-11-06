import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

def readCMB(fname): # reads relevant data from the CMB file
    fh_cmb=nc.Dataset(fname)
    qv=fh_cmb["KuKaGMI/vaporDensity"][:,:,:]
    press=fh_cmb["KuKaGMI/airPressure"][:,:,:]
    envNodes=fh_cmb["KuKaGMI/envParamNode"][:,:,:]
    airTemp=fh_cmb["KuKaGMI/airTemperature"][:,:,:]
    skTemp=fh_cmb["KuKaGMI/skinTemperature"][:,:]
    binNodes=fh_cmb["KuKaGMI/phaseBinNodes"][:,:]
    pwc=fh_cmb["KuKaGMI/precipTotWaterCont"][:,:,:]
    sfcEmiss=fh_cmb["KuKaGMI/surfEmissivity"][:,:,:]
    dm=fh_cmb["KuKaGMI/precipTotDm"][:,:,:]
    cldw=fh_cmb["KuKaGMI/cloudLiqWaterCont"][:,:,:]
    sfcBin=fh_cmb["KuKaGMI/Input/surfaceRangeBin"][:,:,:]
    zCorrected=fh_cmb["KuGMI/correctedReflectFactor"][:,:,:]
    pType=fh_cmb["KuKaGMI/Input/precipitationType"][:,:]
    lon=fh_cmb["KuKaGMI/Longitude"][:,:]
    lat=fh_cmb["KuKaGMI/Latitude"][:,:]
    oe_wvp=fh_cmb["KuKaGMI/OptEst/OEcolumnWaterVapor"][:,:]
    oe_lwp=fh_cmb["KuKaGMI/OptEst/OEcolumnCloudLiqWater"][:,:]
    oe_iwp=fh_cmb["KuKaGMI/OptEst/OEcolumnCloudIceWater"][:,:]
    oesfc_precip=fh_cmb["KuKaGMI/OptEst/OEestimSurfPrecipTotRate"][:,:]
    stormTop=fh_cmb["KuKaGMI/Input/stormTopAltitude"][:,:]
    nearSfcPrecip=fh_cmb["KuKaGMI/nearSurfPrecipTotRate"][:,:]
    surfaceType=fh_cmb["KuKaGMI/Input/surfaceType"][:,:]
    zeroDegAltitude=fh_cmb["KuKaGMI/Input/zeroDegAltitude"][:,:]
    return qv,press,envNodes,airTemp,skTemp,binNodes,pwc,sfcEmiss,dm,cldw,sfcBin,zCorrected,pType,lon,lat,oe_wvp,oe_lwp,oe_iwp,oesfc_precip,stormTop,nearSfcPrecip,zeroDegAltitude,surfaceType

def read1CGMI(fname): # reads relevant data from the 1C GMI file
    with nc.Dataset(fname) as f:
        #print(f)
        #print(f['S1'])
        lat=f['S1/Latitude'][:]
        lon=f['S1/Longitude'][:]
        tc=f['S1/Tc'][:]
    return lat,lon,tc