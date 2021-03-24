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


class Solo(object):
    
    def __init__(self,name,dataFolder,f,zi,zb,tstart,tstop,
                 blockLength=600,
                 fmaxBelieve = 3, 
                 fmax = 5, 
                 fresolution = 0.01,
                 rho=1000,
                 g=9.8,
                 stationNo = 235,
                 emergedT0 = True):
        
        self.name = name
        self.dataFolder = dataFolder
        self.f = f
        self.zi = zi
        self.zb = zb
        self.fmaxBelieve = fmaxBelieve
        self.fmax = fmax
        self.fresolution = fresolution
        self.rho = rho
        self.g = g
        self.tstart = pd.to_datetime(tstart)
        self.tstop = pd.to_datetime(tstop)
        self.stationNo = stationNo
        self.emergedT0 = emergedT0
        
        self.get_fileNames()        
        self.load_raw_pressure_from_file()
        self.load_air_pressure_from_knmi()
        self.correct_raw_pressure_for_air_pressure(self.dfp,self.pAir)        
        self.cast_to_blocks_in_xarray(blockLength)
        self.compute_block_averages()
        

    def get_fileNames(self):
        '''
        Construct required filepaths for later functions

        '''        
        cdir = os.getcwd()
        os.chdir(self.dataFolder)
        fileName = glob.glob('KNMI*.txt')
        self.knmiFileName =self.dataFolder + '\\' + fileName[0]
        
        fileName = glob.glob('*metadata.txt')
        self.metaDataFile = self.dataFolder + '\\' + fileName[0]
         
        fileName = glob.glob('*data.txt')
        self.dataFile = self.dataFolder + '\\' + fileName[0]
        
        os.chdir(cdir)
        
        
    
    def load_raw_pressure_from_file(self):
        '''
        load raw data from file and casts it in a pandas dataframe
        '''


        p=[]; datetime=[]
        with open(self.dataFile) as myfile:
            for index,line in enumerate(myfile):
                if index>=2:
                  lin = line.split(',')
                  datetime.append(lin[0])
                  p.append(float(lin[1]))
        p = np.array(p) * 1e4 #dBar to Pa      
          
        t = pd.date_range(datetime[0],periods=len(datetime),freq = '{}S'.format(1/self.f))
        
        dfp = pd.DataFrame(data={'p':p},index=t)
        
        #clip to the daterange that we are interested in
        dfp = dfp[dfp.index>=pd.to_datetime(self.tstart)]
        dfp = dfp[dfp.index<pd.to_datetime(self.tstop)]
        
        self.dfp = dfp
        
    
    
    def load_air_pressure_from_knmi(self):
        '''
        casts air pressure data from knmi in their download format to a pandas
        Dataframe
        '''
        date=[]; h=[]; T=[]; pa=[];no = [];
        with open(self.knmiFileName) as fp:
            for line in (fp):
                if line[0]=='#':
                    continue
                else:
                   x =line.split(',')
                   no.append(float(x[0]))
                   date.append(str(x[1]))
                   h.append(float(x[2]))
                   T.append(float(x[3]))
                   pa.append(float(x[4]))
        pa = 10*np.array(pa) # change 0.1hPa to Pascal
        t0 = pd.to_datetime(date[0],format='%Y%m%d')+pd.Timedelta('{}H'.format(h[0]))
        t = pd.date_range(t0.to_datetime64(),periods=len(pa),freq='1H')
        
        if len(t)!=len(pa):
            print('reconstruct time array line by line. This is much more work!')
            t = [pd.to_datetime(ix,format='%Y%m%d')+pd.Timedelta('{}H'.format(ih)) for ix,ih in zip(date,h)]
        
        #prepare the air pressure array to similar frequency and linearly interpolate missing values    
        pAir = pd.DataFrame(data={'stationNo':no,'p':pa,'T':T},index = t)
        
        #drop all other entries apart the station number that we chose
        pAir = pAir[pAir['stationNo']==self.stationNo] #make sure only the Kooij data is in there
        pAir.drop(columns={'stationNo'},inplace=True)  
        
        self.pAir = pAir
           
        
    
    def correct_raw_pressure_for_air_pressure(self,dfp,pAir):
        '''
        Uses air pressure from the pandas dataframe pAir to correct the raw 
        pressure timeseries. In case the instrument was emerged from the water 
        and hence measuring air pressure at the start and at the end of the deployment
        we can also correct for a drift in the reference level of the instrument.
        If the instrument was not emerged at start and end, we can only correct
        for the drift in air pressure, but we can't correct for the drift in 
        reference level of the instrument.
        '''
        
        #resample air pressure df to same frequency as the pressure df
        freq = (dfp.index[1]-dfp.index[0]).total_seconds()
        pAir = pAir.resample('{}S'.format(freq)).asfreq().interpolate('linear')
        
        #join the two df's
        dfp = dfp.join(pAir,rsuffix='Air') 
        
        # find first and last index where there is both solo signal and air pressure 
        i0 = dfp.apply(pd.Series.first_valid_index)['pAir']
        p0 = dfp.loc[i0:i0+pd.Timedelta('0.1H')] #average of first 6 minutes
        i1 = dfp.apply(pd.Series.last_valid_index)['pAir']
        p1 = dfp.loc[i1-pd.Timedelta('0.1H'):i1] #average of last 6 minutes

        #only correct for variations in air pressure while instrument is submerged
        dfp['dpAir'] = dfp['pAir']-dfp['pAir'].loc[i0]
            
        if self.emergedT0 is True:
            #compute drift in instrument measurement of air pressure through checking with knmi station
            dfp['drift']=np.nan
            dfp['drift'].loc[i0] = p0.p.mean()-p0.pAir.mean()
            dfp['drift'].loc[i1] = p1.p.mean()-p1.pAir.mean()
            dfp['drift'] = dfp['drift'].interpolate() - dfp['drift'].loc[i0]
                        
            #correct the pressure signal with dpAir and with drift in instrument pressure
            dfp['pRaw'] = dfp['p']
            dfp['p'] = dfp['pRaw']- dfp['pRaw'].loc[i0]-dfp['dpAir'] - dfp['drift']
        else:
            #correct for air pressure 
            dfp['pRaw'] = dfp['p']
            dfp['p'] = dfp['pRaw'] - dfp['dpAir']  
        self.dfp = dfp  
        
    
    
    def cast_to_blocks_in_xarray(self,blockLength): 
        '''
        takes the raw data which are timeseries in pandas DataFrame and casts it
        in blocks (bursts) in an xarray with metadata for easy computations and 
        saving to file (netcdf) later on.
        '''
        p = self.dfp['p'].values
        t = self.dfp.index
        
        N = len(p)
        
        blockWidth = blockLength * self.f
        NB = int(np.floor(N/blockWidth))
        p = p[0:NB*blockWidth]
        N = len(p)
        blockWidth = blockWidth
        p2 = p.reshape(NB,int(N/NB))
        
        blockStartTimes = t[0::blockWidth]
        if len(blockStartTimes)>int(N/NB):
            blockStartTimes = blockStartTimes.delete(-1)
        
        #cast all info in dataset
        ds = xr.Dataset(data_vars = dict(
                f = self.f,
                zb = self.zb,
                zi = self.zi,  
                rho = self.rho,
                g = self.g,
                p =(['t','N'],p2) ),    
            coords = dict(t = blockStartTimes,
                          N = np.arange(0,blockWidth))
                    )
        ds['t'].attrs = {'long_name': 'burst start times'}
        ds['f'].attrs = {'units': 'Hz','long_name':'sampling frequency'} 
        ds['zb'].attrs = {'units': 'm+NAP','long_name':'bed level, neg down'} 
        ds['zi'].attrs = {'units': 'm+NAP','long_name':'instrument level, neg down'} 
        ds['rho'].attrs = {'units': 'kg m-3','long_name':'water density'}
        ds['g'].attrs = {'units': 'm s-2','long_name':'gravitational acceleration'}
        ds['p'].attrs = {'units': 'Pascal','long_name':'pressure signal corrected for air pressure and  (possibly) instrument drift'} 
                
        self.ds = ds
                                  
    def compute_block_averages(self):   
         '''
         computes some first block averages, specifically average pressure, 
         water level and water depth on the xarray Dataset
         '''
         pm = self.ds['p'].values.mean(axis=1)/self.rho/self.g  #in dBar!    
         zsmean = pm + self.zi
         h = pm + self.zi - self.zb   
         
         ds = self.ds
         
         ds['zsmean'] = (['t'],zsmean)
         ds['h'] = (['t'], h)
         ds['pm'] = (['t'],pm) 
         
         ds['pm'].attrs = {'units': 'dBar','long_name':'burst averaged pressure'} 
         ds['zsmean'].attrs = {'units': 'm+NAP','long_name':'burst averaged surface elevation'} 
         ds['h'].attrs = {'units': 'm','long_name':'burst averaged water depth'} 
         
         self.ds = ds