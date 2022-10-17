# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 16:26:41 2022

@author: marliesvanderl
"""
import xarray as xr
import numpy as np
import glob
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
import puv
import xrMethodAccessors
import pdb

experimentFolder = r'o:/phddata/fieldvisits\20210908_campaign\instruments//'

instruments = ['L2C9OSSI','L2C8OSSI','L2C6OSSI','L1C2OSSI','L4C3OSSI','L5C2OSSI','L6C2OSSI']
rho = 1028
g = 9.8

for instr in instruments[0:3]:
    file = glob.glob(experimentFolder + instr + r'/QC/*.nc')
    print(instr)  
    with xr.open_dataset(file[0]) as ds:
        ds['pm'] = ds.p.mean(dim='N')
        ds['zs'] = ds.pm/1e4
        
        #fix offset with respect to the solo at L2C10:
        #these were identified manually by finding the average discrepenacy over all meas
        if instr=='L2C9OSSI':
            print('added stuff')
            ds['zs'] = ds.zs + 0.58
        elif instr=='L2C8OSSI':
            print('added stuff')
            ds['zs'] = ds.zs + 0.31
        elif instr=='L2C6OSSI':
            print('added stuff')
            ds['zs'] = ds.zs + 0.30
        
        ds['d'] = ds.zs-ds.zb
        
        
        ds['p2'] = ds.p.fillna(ds.pm)/rho/g 
            
        fx,vy0 = ds.puv.spectrum_simple('p2',**{'fresolution':0.02})
        ds2 = xr.Dataset(
               data_vars = dict(                  
                   d=ds['d'],
                   zs=ds['zs'],
                   zb = ds['zb'],
                   sf = ds.sf),                  
               coords = dict(t = ds.t,
                         f =  fx.isel(t=0)
               ))
        ds2['vy0'] = vy0
        
        ds2['elev'] = ds.zi-ds.zb
        ds2['Q'] = ds2.puv.attenuation_factor(
            'pressure',elev = 'elev',d = 'd')
        ds2['vy'] = ds2.Q * ds2.vy0
        
        ds2['Hm0'], ds2['Tp'], ds2['Tm01'], ds2['Tm02'], ds2['Tmm10'] = (
            ds2.puv.compute_wave_params(var='vy') )
        
        ds2['Hm0'].attrs = {'units':'m','long_name':'Hm0'}
        ds2['Tp'].attrs = {'units':'s','long_name':'Tp'}
        ds2['Tm01'].attrs = {'units':'s','long_name':'Tm01'}
        ds2['Tm02'].attrs = {'units':'s','long_name':'Tm02'}
        ds2['Tmm10'].attrs = {'units':'s','long_name':'T_{m-1,0}'}
        
        ds2['k'] = xr.apply_ufunc(lambda tm01,h: puv.disper(2*np.pi/tm01,h),ds2['Tmm10'],ds2['zs']-ds2['zb'])
        ds2['k'].attrs = {'units':'m-1','long_name':'k'}
                    
        ds2['Ur'] = xr.apply_ufunc(lambda hm0,k,h: 3/4 * 0.5 * hm0*k/(k*h)**3,ds2['Hm0'],ds2['k'],ds2['zs']-ds2['zb'])
        ds2['Ur'].attrs = {'units':'-','long_name':'Ursell no.'}
        
        try:
            sf = ds.sf.values
            ufunc = lambda x, Tp: puv.compute_SkAs(
                sf,
                x,
                fbounds=[0.5/Tp,sf/2]
                )
            
            ds2['Sk'],ds2['As'], ds2['sig'] = xr.apply_ufunc(
                ufunc,
                (ds['p2']).where(ds2.Tm01<6).dropna(dim='t'),
                ds2['Tmm10'].where(ds2.Tm01<6).dropna(dim='t'),
                input_core_dims=[['N'], []],
                output_core_dims=[[], [], []], 
                vectorize=True) 

        except:
            ds2['Sk'] = np.nan*ds2.Ur
            ds2['As'] = np.nan*ds2.Ur
            ds2['sig'] = np.nan*ds2.Ur
            
            
        ds2['Sk'].attrs = {'units':'m3/s3','long_name':'skewness'}
        ds2['As'].attrs = {'units':'m3/s3','long_name':'asymmetry'}
        ds2['sig'].attrs = {'units':'m/s','long_name':'std(p)'}
        
        ds2['nAs'] = ds2.As/ds2.sig**3
        ds2['nSk'] = ds2.Sk/ds2.sig**3
        ds2['nAs'].attrs = {'units':'-','long_name':'As'}
        ds2['nSk'].attrs = {'units':'-','long_name':'Sk'}
    
        ds2.attrs = ds.attrs
        ds.attrs['summary'] = 'SEDMEX field campaign, wave statistics from pressure recordings with ossi, calculated in interval 0.1 and 1.5, with maximum attenuationfactor of 5'

        #saving to file
        if not os.path.isdir(experimentFolder + '//' +  instr + r'\tailored'):
            os.mkdir(experimentFolder + '//' +  instr + r'\tailored' )
        ncFilePath = experimentFolder + '//' +  instr + r'\tailored\{}.nc'.format(instr)
            
        #if nothing else, at least specify lossless zlib compression
        comp = dict(zlib=True, complevel=5)
        ds2.encoding = {var: comp for var in ds2.data_vars}                  
        ds2.to_netcdf(ncFilePath, encoding = ds2.encoding )