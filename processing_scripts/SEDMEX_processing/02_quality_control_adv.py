# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:36:11 2021

@author: marliesvanderl
"""
import glob
import os
import xarray as xr
import numpy as np
import pandas as pd
import sys
# dirname = os.path.dirname(__file__)
# filename = os.path.join(dirname, '../../modules')
# sys.path.append(filename)
import puv
import pdb

#%%
experimentFolder = r'u:\EURECCA\fieldvisits\20210908_campaign\instruments'
folderinName = 'raw_netcdf'
folderoutName = 'QC_loose'
rho = 1028
g = 9.8


#%% quality check calibration factors
#velocity limits based on settings in the instrument
qc = pd.DataFrame(
    {
     'uLim':[2.1,2.1,2.1,2.1,2.1,1.5,1.5,2.1,2.1], #modified to be 1.5 and not 0.6 for horizontal vectors, not saved to file yet! 10/12
     'vLim':[2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1],
     'wLim':[0.6,0.6,0.6,0.6,0.6,1.5,1.5,0.6,0.6],
     'corTreshold':9*[50],
     'maxFracNans': 9*[0.05],
     'maxGap' : 9*[10]
      },
    index=['L1C1VEC','L2C3VEC','L3C1VEC','L5C1VEC','L6C1VEC','L2C2VEC','L2C4VEC','L2C10VEC','L4C1VEC']
    )

# qc = pd.DataFrame(
#     {
#      'uLim':[2.1,2.1,2.1,2.1,2.1,1.5,1.5,2.1,2.1], #modified to be 1.5 and not 0.6 for horizontal vectors, not saved to file yet! 10/12
#      'vLim':[2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1],
#      'wLim':[0.6,0.6,0.6,0.6,0.6,1.5,1.5,0.6,0.6],
#      'corTreshold':9*[70],
#      'maxFracNans': [0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02],
#      'maxGap' : [4,4,4,4,4,4,4,4,4]
#       },
#     index=['L1C1VEC','L2C3VEC','L3C1VEC','L5C1VEC','L6C1VEC','L2C2VEC','L2C4VEC','L2C10VEC','L4C1VEC']
#     )

def hor2vert_vector_mapping(u,v,w):
    mapMatrix = np.array([[0,0,-1],[0,1,0],[1,0,0]])
    coords = np.vstack((u,v,w))
    u,v,w = mapMatrix @coords
    return u,v,w


#%%

instruments = [
        'L1C1VEC',
        'L2C3VEC',
        'L3C1VEC',
        'L5C1VEC',
        'L6C1VEC',
        'L2C2VEC',
        'L2C4VEC',
        'L2C10VEC',
        'L4C1VEC'
        ]  
      
for instrumentName in instruments:
    fileNames = glob.glob(os.path.join(experimentFolder,instrumentName,folderinName, '*.nc'))
    QC = qc.loc[instrumentName]
    print(instrumentName)
    for file in fileNames:
        print(file)        
        with xr.open_dataset(file) as ds:
            #only work on certain times
            # if np.logical_or(ds.t.min().values<pd.to_datetime('20210917'), ds.t.min().values>=pd.to_datetime('20211001')):
            #     continue
            #if correlation is outside confidence range
            mc1 = ds.cor1 > QC['corTreshold']
            mc2 = ds.cor2 > QC['corTreshold']
            mc3 = ds.cor3 > QC['corTreshold']
            
            #if observation is outside of velocity range
            mu1 = np.abs(ds.u) < QC['uLim']
            mu2 = np.abs(ds.v) < QC['uLim']
            mu3 = np.abs(ds.w) < QC['uLim']
            
            #if du larger than 4*std(u) then we consider it outlier and hence remove:
            md1 = np.abs(ds.u.diff('N'))<3*ds.u.std(dim='N')
            md1 = md1.combine_first(mu1)
            md2 = np.abs(ds.v.diff('N'))<3*ds.v.std(dim='N')
            md2 = md1.combine_first(mu2)
            md3 = np.abs(ds.w.diff('N'))<3*ds.w.std(dim='N')
            md3 = md1.combine_first(mu3)
            
            ds['mc'] = np.logical_and(np.logical_and(mc1,mc2),mc3)
            ds['mu'] = np.logical_and(np.logical_and(mu1,mu2),mu3)
            ds['md'] = np.logical_and(np.logical_and(md1,md2),md3)
            ds['mc'].attrs = {'units':'-','long_name':'mask correlation'}       
            ds['mu'].attrs = {'units':'-','long_name':'mask vel limit'}       
            ds['md'].attrs = {'units':'-','long_name':'mask deviation'}       
            
            mp = np.abs(ds.p.diff('N'))<4*ds.p.std(dim='N')
            mp = xr.concat([mp.isel(N=0), mp], dim="N")
            
            ds.coords['maskp'] = (('t', 'N'), mp.values)
            ds.coords['maskv'] = (('t', 'N'), (np.logical_and(np.logical_and(ds.mc, ds.mu),ds.md)).values)
            
            #correct for the air pressure fluctuations and drift in the instrument
            #by using the pressure of the standalone solo's to match the high water pressures
            # we loose the water level gradients
            # if instrumentName in ['L2C2VEC','L2C4VEC','L2C10VEC','L4C1VEC']:
            dsSolo = xr.open_dataset(experimentFolder + r'\L2C10SOLO\QC\L2C10SOLO.nc')
            dsSolo = dsSolo.p.resample({'t':'1800s'}).mean().interp_like(ds)
            psoloref = dsSolo.max(dim='N')
            ds['pc'] = ds.p - ds.p.mean(dim='N').max() + psoloref
            ds['pc'].attrs = {'units':'Pa + NAP','long_name':'pressure','comments':'referenced to pressure L2C10SOLO'}  
            
            #% compute mean water level, water depth      
            ds['zi'] =  ds['zb'] + ds['h']/100
            ds['zi'].attrs = {'units':'m+NAP','long_name':'position probe'}       
            
            ds['zip'] =  ds['zb'] + ds['hpres']/100
            ds['zip'].attrs = {'units':'m+NAP','long_name':'position pressure sensor'}
            
            ds['eta'] = ds['pc']/rho/g
            ds['eta'].attrs = {'units':'m+NAP','long_name':'hydrostatic water level'}
        
            #remove pressure observations where the estimated water level is 
            #lower than the sensor height with margin of error of 10 cm
            # ds['pc'] = ds['pc'].where(ds['zip']< (ds['eta'] - 0.1) )
            ds.coords['maskd'] = (('t', 'N'), (ds['zi']< (ds['eta'] - 0.1)).values )
                       
            #% rename coordinates if horizontally placed
            if ((instrumentName == 'L2C2VEC') | (instrumentName == 'L2C4VEC')):
                ufunc = lambda u,v,w: hor2vert_vector_mapping(u,v,w)                        
                ds['u'],ds['v'],ds['w'] = xr.apply_ufunc(ufunc,
                                    ds['u'],ds['v'],ds['w'],
                                    input_core_dims=[['N'], ['N'], ['N']],
                                    output_core_dims=[['N'], ['N'], ['N']], 
                                    vectorize=True) 
            #% UU instruments flexheads so downward pos instead of upward pos
            if (instrumentName in ['L1C1VEC','L2C3VEC','L3C1VEC','L5C1VEC','L6C1VEC']):        
                 ds['v'] = -ds['v']
                 ds['w'] = -ds['w']
                 
            #% rotate to ENU coordinates
            ufunc = lambda u,v,thet: puv.rotate_velocities(u,v,thet-90)
            ds['u'],ds['v'] = xr.apply_ufunc(ufunc,
                                ds['u'],ds['v'],ds['io'],
                                input_core_dims=[['N'], ['N'], []],
                                output_core_dims=[['N'],['N']], 
                                vectorize=True) 
            ds['u'].attrs = {'units':'m/s','long_name':'velocity E'}
            ds['v'].attrs = {'units':'m/s','long_name':'velocity N'}
            ds['w'].attrs = {'units':'m/s','long_name':'velocity U'}

            #% saving 
            ds.attrs['version'] = 'v2'
            ds.attrs['comment']='Quality checked data: pressure reference level corrected for airpressure drift, correlation and amplitude checks done and spikes were removed. Velocities rotated to ENU coordinates based on heading and configuration in the field.'

            # save to netCDF
            ds = ds.drop(['a1','a2','a3',
                          'cor1','cor2','cor3',
                          'snr1','snr2','snr3',
                          'heading','pitch','roll',
                          'voltage','pc'])

            if not os.path.isdir(os.path.join(experimentFolder, instrumentName, folderoutName)):
                os.mkdir(os.path.join(experimentFolder, instrumentName, folderoutName))

            ncFilePath = os.path.join(experimentFolder, instrumentName, folderoutName, '{}_{}.nc'.format(
                    ds.name,ds.t.isel(t=0).dt.strftime('%Y%m%d').values
                    ))
            #specify compression for all the variables to reduce file size
            comp = dict(zlib=True, complevel=5)
            ds.encoding = {var: comp for var in ds.data_vars}  
            ds.to_netcdf(ncFilePath, encoding=ds.encoding)
            
    



