# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:16:29 2021

@author: marliesvanderl
"""
import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

class Echosounder(object):
    
    def __init__(self,name,dataFolder,
                 tstart = None,
                 tstop = None,
                 blockLength=3600,
                 rho=1025,
                 g=9.8):
        
        self.name = name
        self.dataFolder = dataFolder
        
            
        
        self.get_fileNames()  
        self.load_raw_signal_from_file()   
        
        self.cast_to_xarray()
        if not (tstart is None or tstop is None):
            self.tstart = pd.to_datetime(tstart)
            self.tstop = pd.to_datetime(tstop)
            self.crop_data()
            
        

    def get_fileNames(self):
        '''
        Construct required filepaths for later functions

        '''        
        cdir = os.getcwd()
        os.chdir(self.dataFolder)
        self.rawDataFiles =glob.glob(self.dataFolder + '//*')
        
        os.chdir(cdir)
        
        
    
    def load_raw_signal_from_file(self):
        '''
        load raw data from file and casts it in a pandas dataframe
        '''


        xlist = []; self.t = []; self.hpr = []
        for file in self.rawDataFiles:
            metat = []; metao = []; metar = []; dat = []; 
            with open(file) as myfile:
                for index,line in enumerate(myfile):
                    
                    if index < 22:
                        if 'NSamples' in line:
                            self.NSamples = int(line.split(' ')[1][:-1])
                        if 'Resolution' in line:
                            self.resolution = float(line.split('  ')[1][:-1])
                        if 'Tx_Frequency' in line:
                            self.tx_frequency = line
                        elif 'Range' in line:
                            self.range = line
                        elif 'Pulse period' in line:
                            self.pulse_period = line
                        elif 'Pulses in series' in line:
                            self.pulses_in_series = line
                        elif 'Interval' in line:
                            self.interval = line
                        elif 'Threshold' in line:
                            self.treshold = line
                        elif 'Offset' in line:
                            self.offset = line
                        elif 'Deadzone' in line:
                            self.deadzone = line
                        elif 'PulseLength' in line:
                            self.pulselength = line
                        elif 'TVG_Gain' in line:
                            self.TVG_gain = line
                        elif 'TVG_Slope' in line:
                            self.TVG_slope = line
                       
                    else: 
                        
                        if line[0]=='#' or line=='\n':
                            continue
                        elif line[0:6] == '$SDZDA':
                            metat.append(line[7:-2])
                        elif line[0:6] == '$SDXDR':
                            metao.append(line[7:-2])
                        elif line[0:6] == '$SDDBT':
                            metar.append(line[7:-2])
                        else:
                            dat.append(int(line))
        
                nt = len(dat)/self.NSamples
                xlist.append( np.array(dat).reshape(int(nt),self.NSamples) )
                   
                #time in UTC!
                def string_to_datetime(X):
                    x = X[0:-8].split(',')
                    return datetime(int(x[3]),int(x[2]),int(x[1]),int(x[0][0:2]),int(x[0][2:4]),int(x[0][4:6]),int(float(x[0][7:])*1e4))
                self.t+=  [string_to_datetime(it) for it in metat] 
                
                def get_hpr(X):
                    x = X.split(',')
                    return float(x[1]), float(x[5]), int(x[7].split('*')[-1])
                self.hpr +=  [get_hpr(it) for it in metao]
        self.x = np.vstack(xlist)
    
    
    def cast_to_xarray(self): 
        '''
        takes the raw data which are timeseries in pandas DataFrame and casts it
        in blocks (bursts) in an xarray with metadata for easy computations and 
        saving to file (netcdf) later on.
        '''
        h,p,r = list(zip(*self.hpr))
         
        
        #cast all info in dataset
        ds = xr.Dataset(
            data_vars = dict(
                I = (['t','z'],self.x),
                h = (['t'],np.array(h)),
                p = (['t'],np.array(p)),
                r = (['t'],np.array(r)),
                Range = self.range,
                Treshold = self.treshold,  
                Offset = self.offset,
                Gain = self.TVG_gain,
                G_Slope = self.TVG_slope ),
            coords = dict(t = self.t,
                          z = np.arange(self.NSamples)*self.resolution)
                    )
        ds['t'].attrs = {'long_name': 'time'}
        ds['z'].attrs = {'units': 'm','long_name':'distance'} 
        ds['I'].attrs = {'units': 'dB','long_name':'Intensity'} 
        ds['h'].attrs = {'units': 'deg','long_name':'heading'}
        ds['p'].attrs = {'units': 'deg','long_name':'pitch'}        
        ds['r'].attrs = {'units': 'deg','long_name':'roll'}   
        
        self.ds = ds
        
    def crop_data(self):
        self.ds = self.ds.sel(t=slice(self.tstart,self.tstop))
        
                              
    def add_depth_coords(self,zi):
        self.ds['z'] = zi-self.ds.z

         
    def get_averaged_intensity(self,blockLength = 600):
        # f = np.timedelta64(1, 's') / (self.ds.I.t[12]-self.ds.I.t[11]).values
        self.cI = self.ds.resample(t='10min').mean()
        return self.cI
        
        
        
#%%
import matplotlib.pyplot as plt
import sys
sys.path.append(r'c:\Users\marliesvanderl\phd\analysis\scripts\private\modules')
import plotfuncs as pf

#echosounder 1
dataFolder1 = r'c:\Users\marliesvanderl\phddata\fieldvisits\20201130\echosounder\echo1_3011-0112\raw'
zi =  -0.42   
xx1 =  Echosounder('echo1',dataFolder1)
xx1.add_depth_coords(zi)
I1 = xx1.get_averaged_intensity(blockLength = 600)
tstart = '20201130 15:30'
tstop = '20201201 13:00'  
I1 = I1.sel(t=slice(tstart,tstop))

#echosounder 2
dataFolder2 = r'c:\Users\marliesvanderl\phddata\fieldvisits\20201130\echosounder\echo2_0112-0212\raw'
zi =  -0.423   
xx2 =  Echosounder('echo2',dataFolder2)
xx2.add_depth_coords(zi)
I2 = xx2.get_averaged_intensity(blockLength = 600)
tstart = '20201201 14:00'
tstop = '20201202 10:00'  
I2 = I2.sel(t=slice(tstart,tstop))

#echosounder 3
dataFolder3 = r'c:\Users\marliesvanderl\phddata\fieldvisits\20201130\echosounder\echo1_0212-0412\raw'
zi =  -0.258  
xx3 =  Echosounder('echo3',dataFolder3)
xx3.add_depth_coords(zi)
I3 = xx3.get_averaged_intensity(blockLength = 600)


#%%
imax = np.argmax(I3.I.values,axis=-1)

fig, ax = plt.subplots(figsize=[8.3,5.8])
plt.plot(I3.t.values,I3.z.isel(z=imax))
pf.mdates_concise_subplot_axes(ax)
ax.set_ylabel('z [m+NAP]')