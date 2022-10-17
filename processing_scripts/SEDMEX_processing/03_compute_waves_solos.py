# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:28:24 2021

@author: marliesvanderl
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
import puv
import plotfuncs as pf
import xrMethodAccessors
import pdb

experimentFolder = r'o:\phddata\fieldvisits\20210908_campaign\instruments//'
instruments = ['L2C10SOLO','L2C4SOLO','L2C2SOLO','L4C1SOLO']
rho = 1025
g = 9.8
for instr in instruments:
    print(instr)
    
    ds0 = xr.open_dataset(experimentFolder + instr + '//QC//'+ instr + '.nc')
    
    sf = ds0.sf.values
    t = ds0.t.values
    N = ds0.N.values
    
    
    #--------------------------------------------------------------------------
    #make a new dataset that has an extra dimension to accomodate for the frequency axis
    powerStep       = 7
    fresolution = np.round(sf*np.exp(-np.log(2)*powerStep),5)
    
    ds = xr.Dataset(data_vars = {},
                coords = {'t':t, 'N':N, 'f':np.arange(0,sf/2,fresolution)} )
    
    for key in ds0.data_vars:
        ds[key] = ds0[key]
        ds[key].attrs = ds0[key].attrs                            
    
    #--------------------------------------------------------------------------
    #compute wave statistics
    ds['pm'] = ds.p.mean(dim='N')
    ds['zs'] = ds.pm/rho/g
    ds['d'] = ds.zs-ds.zb
    
    ds['p2'] = ds.p.fillna(ds.pm)/rho/g 
    ds['p2'].attrs = {'units':'m+instr','long_name':'eta',
                      'comments':'hydrostatic surface elevation above instrument, nanfilled'}
    
    fx,ds['vy0'] = ds.puv.spectrum_simple('p2',**{'fresolution':fresolution})
    
    ds['elev'] = ds.zi-ds.zb
    ds['Q'] = ds.puv.attenuation_factor(
        'pressure',elev = 'elev',d = 'd')
    ds['vy'] = ds.Q * ds.vy0
    
    kwargs  = {'fmin':0.1,'fmax':1.5}
    ds['Hm0'], ds['Tp'], ds['Tm01'], ds['Tm02'], ds['Tmm10'] = (
        ds.puv.compute_wave_params(var='vy',**kwargs ) )
    
    ds['Hm0'].attrs = {'units':'m','long_name':'Hm0'}
    ds['Tp'].attrs = {'units':'s','long_name':'Tp'}
    ds['Tm01'].attrs = {'units':'s','long_name':'Tm01'}
    ds['Tm02'].attrs = {'units':'s','long_name':'Tm02'}
    ds['Tmm10'].attrs = {'units':'s','long_name':'T_{m-1,0}'}
    
    ds['k'] = xr.apply_ufunc(lambda tm01,h: puv.disper(2*np.pi/tm01,h),ds['Tmm10'],ds['zs']-ds['zb'])
    ds['k'].attrs = {'units':'m-1','long_name':'k'}
                
    ds['Ur'] = xr.apply_ufunc(lambda hm0,k,h: 3/4 * 0.5 * hm0*k/(k*h)**3,ds['Hm0'],ds['k'],ds['zs']-ds['zb'])
    ds['Ur'].attrs = {'units':'-','long_name':'Ursell no.'}
    
    
    sf = ds.sf.values
    ufunc = lambda x, Tp: puv.compute_SkAs(
        sf,
        x,
        fbounds=[0.5/Tp,sf/2]
        )
    
    ds['Sk'],ds['As'], ds['sig'] = xr.apply_ufunc(
        ufunc,
        (ds['p2']).where(ds.Tm01<10).dropna(dim='t'),
        ds['Tmm10'].where(ds.Tm01<10).dropna(dim='t'),
        input_core_dims=[['N'], []],
        output_core_dims=[[], [], []], 
        vectorize=True) 
       
    ds['Sk'].attrs = {'units':'m3/s3','long_name':'skewness'}
    ds['As'].attrs = {'units':'m3/s3','long_name':'asymmetry'}
    ds['sig'].attrs = {'units':'m/s','long_name':'std(p)'}
    
    ds['nAs'] = ds.As/ds.sig**3
    ds['nSk'] = ds.Sk/ds.sig**3
    ds['nAs'].attrs = {'units':'-','long_name':'As'}
    ds['nSk'].attrs = {'units':'-','long_name':'Sk'}
    ds['name'] = instr
    ds.attrs['summary'] = 'SEDMEX field campaign, wave statistics from pressure recordings with solo, calculated in interval 0.1 and 1.5, with maximum attenuationfactor of 5'
    
    #saving to file
    if not os.path.isdir(experimentFolder + '//' +  instr + r'\tailored'):
        os.mkdir(experimentFolder + '//' +  instr + r'\tailored' )
    ncFilePath = experimentFolder + '//' +  instr + r'\tailored\{}.nc'.format(instr)
        
    #if nothing else, at least specify lossless zlib compression
    comp = dict(zlib=True, complevel=5)
    ds.encoding = {var: comp for var in ds.data_vars}                  
    ds.to_netcdf(ncFilePath, encoding = ds.encoding )
    
    
                
    