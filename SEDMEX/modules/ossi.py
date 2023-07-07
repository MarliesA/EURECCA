# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 09:36:34 2022

@author: marliesvanderl
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pdb
    
def ossi_data_reader(path):    
    '''
    author: Paul van Wiechen
    
    Function to read all WLOG_XXX files in a certain subfolder.
    Make sure that only WLOG_XXX files are in this folder and no other files.
    Only WLOG_XXX files with minimally 2 rows are appended to the dataframe.
    A correct WLOG_XXX file should contain a first line with OSSI configuration, and a second line (third row) with starting time
    Timestep and sampling frequency are retrieved from the first row. Starting time from the next row
    Returns a dataframe with a time column and pressure column in dbars
    '''
    
    ossi = pd.DataFrame({
        't': [],
        'p':[]})

    directory = str(path)

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
    
        # checking if it is a file
        if os.path.isfile(f):
            print('Currently concatenating file ' + f)
            ossi_raw = pd.read_csv(f, header=None, nrows=4, sep=',')
            if len(ossi_raw.index) > 2:
                t_0 = datetime(int(str(20) + ossi_raw[0][1][1:]),int(ossi_raw[1][1][1:]),int(ossi_raw[2][1][1:]),int(ossi_raw[3][1][1:]),int(ossi_raw[4][1][1:]),int(ossi_raw[5][1][1:]))
                dt = 1/float(ossi_raw[6][0][1:])
                ossi_tot = pd.read_csv(f, skiprows=3, usecols=[0,1,2,3,4,5,6,7,8,9,10,11], header=None, sep=',', skipinitialspace=True).to_numpy().flatten()
                ossi_temp = pd.DataFrame({
                    't':np.array([t_0 + timedelta(seconds=dt*i) for i in range(len(ossi_tot))]),
                    'p':ossi_tot})
            
                ossi_temp.dropna(inplace=True)
                ossi_temp['p'] = ossi_temp['p'] * 1e5 #Bar to Pa   
                
                ossi = pd.concat([ossi,ossi_temp], ignore_index=True)
                
                
    ossi['p'] = pd.to_numeric(ossi['p'])
    ossi['t'] = pd.to_datetime(ossi['t'])            
    
    return ossi.set_index('t')

def Ossi(instrumentName,experimentFolder,isxy,sf, jasave = False):
           
    path = experimentFolder +  '//' + instrumentName + r'\raw' + '//'       
    dfp = ossi_data_reader(path)
        
    ds = dfp.to_xarray()
    ds['p'] = ds.p.astype('int32')
    #we remove the period before deployment and withrieval
    ds = ds.sel(t=slice('20210910 19:00','20211018 10:00'))
    
    
    ds.p.attrs={'long_name':'pressure','units':'Pa'}
   
    # add global attribute metadata
    ds.attrs = {
       'Conventions':'CF-1.6', 
       'name':'{}'.format(instrumentName),
       'instrument':'{}'.format(instrumentName), 
       'instrument type':'OSSI',
       'instrument serial number': '{}'.format(isxy[instrumentName]['serial number']),
       'epsg':28992,
       'x':isxy[instrumentName]['xRD'],
       'y':isxy[instrumentName]['yRD'],   
       'sf':sf,
       'time zone':'UTC+2',
       'coordinate type':'XYZ',
       'summary': 'SEDMEX field campaign',
       'contact person' : 'Marlies van der Lugt',
       'emailadres':'m.a.vanderlugt@tudelft.nl',
       'construction datetime':datetime.now().strftime("%d-%b-%Y (%H:%M:%S)"),
       'version': 'v1',
       'version comments' : 'constructed with xarray' }    

    # save to netCDF
    if jasave:
        if not os.path.isdir(experimentFolder + '//' +  instrumentName + r'\raw_netcdf'):
            os.mkdir(experimentFolder + '//' +  instrumentName + r'\raw_netcdf' )
        ncFilePath = experimentFolder + '//' +  instrumentName + r'\raw_netcdf\{}.nc'.format(
                ds.name
                )
        
        #if nothing else, at least specify lossless zlib compression
        comp = dict(zlib=True, complevel=5)
        ds.encoding = {var: comp for var in ds.data_vars}                  
        ds.to_netcdf(ncFilePath, encoding = ds.encoding )
        
    return ds 

