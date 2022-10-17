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
    
def ctdDiver(instrumentName,experimentFolder,datafile,isxy,jasave = True):
           
    dataFile = experimentFolder +  '//' + instrumentName + r'\raw' + '//' + datafile      
    
    header = []
    with open(dataFile) as myfile:
        for index,line in enumerate(myfile):
            if '[Data]' in line:
                startline = index
                break           
            if len(line.strip())!=0 :
                header.append(line.strip())
                            
    t = []; T = []; p = []; c = []
    with open(dataFile) as myfile:
        for index, line in enumerate(myfile):
            if index>startline + 1:
                thisline = (line.strip().split())
                t.append(thisline[0] + ' ' + thisline[1])
                p.append(thisline[2])
                T.append(thisline[3])
                c.append(thisline[4])
     
    t = pd.to_datetime(t)
    p = 1e2*0.980665*np.array([float(ip) for ip in p]) #mH20 to Pa
    T = [float(iT) for iT in T]
    c = [float(ic) for ic in c]
    
    data = pd.DataFrame({'p':p,'T':T,'c':c}, index = t)
    data.index.name = 't'
    ds = data.to_xarray()
    ds.p.attrs={'long_name':'air pressure','units':'Pa'}
    ds.T.attrs={'long_name':'air temperature','units':'deg C'}
    ds.c.attrs = {'long_name':'special conductivity','units':'mS/cm'}
   
    # add global attribute metadata
    ds.attrs = {
       'Conventions':'CF-1.6', 
       'name':'{}'.format(instrumentName),
       'instrument':'{}'.format(instrumentName), 
       'instrument type':'CTD-Diver 17',
       'sample period':'M03',
       'sample method':'TM01',
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