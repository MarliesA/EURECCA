# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:52:55 2021

@author: marliesvanderl
"""


import os
import numpy as np
import pandas as pd
from datetime import datetime
import pdb
  
def solo_data_reader(dataFile,sf):
    '''
    Function to read solo datafile.
    Returns a dataframe with a time column and pressure column in Pascal
    '''
    p=[]; datt=[]
    with open(dataFile) as myfile:
        for index,line in enumerate(myfile):
            if index>=1:
              lin = line.split(',')
              datt.append(lin[0])
              p.append(float(lin[1]))
    p = np.array(p) * 1e4 #dBar to Pa      
      
    t = pd.date_range(datt[0],periods=len(datt),freq = '{}S'.format(1/sf))
    
    dfp = pd.DataFrame(data={'p':p},index=t)
    
    dfp.index.name = 't'
    return dfp

    
def Solo(instrumentName,experimentFolder,datafile,isxy,sf,jasave = True):
    '''
    wrapper function that casts solo data into an xarray with appropriate
    metadata and optionally saves it to file
    returns an xarray dataset with metadata
    '''
       
    dataFile = experimentFolder +  '//' + instrumentName + r'\raw' + '//' + datafile      
    dfp = solo_data_reader(dataFile,sf)
   
    ds = dfp.to_xarray()
    ds.p.attrs={'long_name':'pressure','units':'Pa'}
   
    # add global attribute metadata
    ds.attrs = {
       'Conventions':'CF-1.6', 
       'name':'{}'.format(instrumentName),
       'instrument':'{}'.format(instrumentName), 
       'instrument type':'Ruskin RBR Solo',
       'instrument serial number': '{}'.format(isxy[instrumentName]['serial number']),
       'epsg':28992,
       'x':isxy[instrumentName]['xRD'],
       'y':isxy[instrumentName]['yRD'],   
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
        ncFilePath = experimentFolder + '//' +  instrumentName + r'\raw_netcdf\{}_{}.nc'.format(
                ds.name,ds.t.isel(t=0).dt.strftime('%Y%m%d').values
                )
        ds.to_netcdf(ncFilePath )
        
    return ds 