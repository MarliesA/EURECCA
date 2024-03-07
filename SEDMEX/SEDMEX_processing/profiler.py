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
import pdb

class Profiler(object):
    
    def __init__(self,name,dataFolder,zb,zi,
                 tstart = None,
                 tstop = None):
        
        self.name = name
        self.dataFolder = dataFolder
        self.zb = zb
        self.zi = zi
        
        self.get_fileNames()          
        self.read_hdr()

        # if no tstart and tstop are specified, use the interval from the 
        # header file
        if tstart is None:
            tstart = self.startTime
        else:
            tstart = pd.to_datetime(tstart)
        if tstop is None:
            tstop = self.stopTime
        else:
            tstop = pd.to_datetime(tstop)
        self.tstart = tstart
        self.tstop = tstop           
        

        

            
        

    def get_fileNames(self):
        '''
        Construct required filepaths for later functions

        '''
        
        cdir = os.getcwd()
        os.chdir(self.dataFolder)
        fileName = glob.glob('*.hdr')
        self.hdrFile =self.dataFolder + '\\' + fileName[0]
         
        fileName = glob.glob('*.sen')
        self.senFile = self.dataFolder + '\\' + fileName[0]
         
        fileName = glob.glob('*.v1')
        self.v1File = self.dataFolder + '\\' +fileName[0]
        
        fileName = glob.glob('*.v2')
        self.v2File = self.dataFolder + '\\' +fileName[0]
        
        fileName = glob.glob('*.v3')
        self.v3File = self.dataFolder + '\\' +fileName[0]

        fileName = glob.glob('*.a1')
        self.a1File = self.dataFolder + '\\' +fileName[0]
        
        fileName = glob.glob('*.a2')
        self.a2File = self.dataFolder + '\\' +fileName[0]
        
        fileName = glob.glob('*.a3')
        self.a3File = self.dataFolder + '\\' +fileName[0]        

        fileName = glob.glob('*.c1')
        self.c1File = self.dataFolder + '\\' +fileName[0]
        
        fileName = glob.glob('*.c2')
        self.c2File = self.dataFolder + '\\' +fileName[0]
        
        fileName = glob.glob('*.c3')
        self.c3File = self.dataFolder + '\\' +fileName[0]    
        
        fileName = glob.glob('*.hr2')
        self.hr2File = self.dataFolder + '\\' +fileName[0]  
        
        os.chdir(cdir)
        
        
    def read_hdr(self):
        '''
        Reads the header file and stores all input in a dict so we can access
        those if necessary later on
        '''
         
        columnPosition = 38    
        myvars = {}
        count  = 0
        name   = 'dummy'
        with open(self.hdrFile) as myfile:
            for index,line in enumerate(myfile):
                if index > 133:
                    break           
                NewName = line[0:columnPosition].strip()
                var= line[columnPosition:].strip()
                if len(NewName)==0 & len(var)!=0:
                    myvars[name] = [myvars[name],var]
                else:
                    name = NewName   
                    myvars[name] = var
                count+=1     
             
        self.hdr = myvars
        self.nMeas = float(myvars['Number of measurements'])

        self.startTime = pd.to_datetime(myvars['Time of first measurement'],
                                            format = 'mixed', dayfirst=True)

        self.stopTime = pd.to_datetime(myvars['Time of last measurement'],
                                           format = 'mixed', dayfirst=True)

            
        self.samplingFrequency = float(myvars['Sampling rate'].split()[0])
        self.coordinateSystem = myvars['Coordinate system']
        self.transformationMatrix = myvars['Transformation matrix']
        self.cellSize = float(myvars['Cell size'].split(' ')[0])/1e3 #in meters
        self.nCells = int(myvars['Number of cells'])
        self.nSampBurst = int(myvars['Samples per burst'])
        self.burstInterval = int(myvars['Measurement/Burst interval'][:-3]) # in seconds
        self.blankingDistance = float(myvars['Blanking distance'][:-1]) #in meters
        self.Orientation = myvars['Orientation']
        
        #construct vertical axis
        if self.Orientation == 'DOWNLOOKING':
            self.z = self.zi-self.blankingDistance-np.arange(0,self.nCells)*self.cellSize 
        else:
            self.z = self.zi+self.blankingDistance+np.arange(0,self.nCells)*self.cellSize             
        
        #construct burst axis
        self.N = np.arange(0,self.nSampBurst)/self.samplingFrequency # in seconds
        
    def read_hr_profiler_burstsample_data(self,fileName):
        '''
        reads block of data from burst,sample, cell blocks of data and returns
        the data in an xarray dataArray
        '''
        #we expect to read nSamples from file between tstart and tstop
        #computed from burstInterval and the amount of samples per burst
        nBursts = (self.tstop-self.tstart).total_seconds()/self.burstInterval
        nSamples = nBursts*self.nSampBurst
        
        if self.tstart < self.startTime:
            raise 'must specify a start time after measuring start time: {}'.format(self.startTime)
            
        t = pd.date_range(start =self.tstart, periods = nBursts,
                                freq = '{}S'.format(self.burstInterval))  

        #if we only want to read data from tstart and this is later than the measuring startTime, we need to skip rows while reading  
        skipRowsDat = int((self.tstart - self.startTime).total_seconds()/self.burstInterval*self.nSampBurst   )
     
        #read the file line by line and cast in 2D numpy array
        x=[]
        with open(fileName) as fp:
            for i,line in enumerate(fp):
                # only read the block that we requested
                if ((i>=skipRowsDat) & (i< (skipRowsDat + nSamples))):
                    x.append( [float(ix) for ix in line.split() if len(ix)>0] )                    
                elif i>= (skipRowsDat + nSamples):
                    break

        x = np.vstack(x)

        #drop the bursts where there are samples missing
        (u,c) = np.unique(x[:,0],return_counts=True)
        burst2drop = u[c!=self.nSampBurst]
        if len(burst2drop>0):
            raise 'error: there are incomplete bursts!'
            x = x[x[:,0]!=burst2drop,:]
        
        #reshape into 3D array
        #recompute here in case the self.tstop is later than what is actually recorded
        nBursts = int(x[:,0].max())-int(x[:,0].min())+1
        t = t[0:nBursts]
        # pdb.set_trace()
        x = x.reshape(nBursts,self.nSampBurst,self.nCells+2)
        
        #after checking the reshaping is correct, drop the burst and the sample counters
        x = x[:,:,2:]       
        
        #construct data array
        da = xr.DataArray(
            data = x,
            dims = ['t','N','z'],
            coords = dict(
                t = t,
                N = self.N,
                z = self.z
                )
            )
        da.N.attrs = {'long_name':'time','units':'s'}
        da.z.attrs = {'long_name':'z','units':'m+NAP'}
        
        return da
           
        
    def load_amplitude_data(self):
        '''
        read amplitude components from file and cast in an xarray dataset
        '''        
        
        a1 = self.read_hr_profiler_burstsample_data(self.a1File)
        a1.name = 'a1'
        a1.attrs = {'units':'counts','long_name':'amplitude-1'}
        print('intensity file .a1 was read')
        a2 = self.read_hr_profiler_burstsample_data(self.a2File)
        a2.name = 'a2'
        a2.attrs = {'units':'counts','long_name':'amplitude-2'}
        print('intensity file .a2 was read')
        a3 = self.read_hr_profiler_burstsample_data(self.a3File) 
        a3.name = 'a3'
        a3.attrs = {'units':'counts','long_name':'amplitude-3'}
        print('intensity file .a3 was read')      
        
        self.ds_amp = xr.merge([a1,a2,a3]).astype(int)
        
        
    def load_velocity_data(self):
        '''
        read velocity components file and cast in an xarray dataset
        '''
                      
        v1 = self.read_hr_profiler_burstsample_data(self.v1File)
        v1.name = 'v1'
        v1.attrs = {'units':'m/s','long_name':'velocity-1'}
        print('velocity file .v1 was read')
        v2 = self.read_hr_profiler_burstsample_data(self.v2File)
        v2.name = 'v2'
        v2.attrs = {'units':'m/s','long_name':'velocity-2'}
        print('velocity file .v2 was read')
        v3 = self.read_hr_profiler_burstsample_data(self.v3File) 
        v3.name = 'v3'
        v3.attrs = {'units':'m/s','long_name':'velocity-3'}

        print('velocity file .v3 was read')    
        
        self.ds_vel = xr.merge([v1,v2,v3])

    def load_correlation_data(self):
        '''
        read correlation components file and cast in an xarray dataset
        '''
                      
        c1 = self.read_hr_profiler_burstsample_data(self.c1File)
        c1.name = 'c1'
        c1.attrs = {'units':'%','long_name':'cor-1'}
        print('correlation file .c1 was read')
        c2 = self.read_hr_profiler_burstsample_data(self.c2File)
        c2.name = 'c2'
        c2.attrs = {'units':'%','long_name':'cor-2'}
        print('correlation file .c2 was read')
        c3 = self.read_hr_profiler_burstsample_data(self.c3File) 
        c3.name = 'c3'
        c3.attrs = {'units':'%','long_name':'cor-3'}
        print('correlation file .c3 was read')    
        
        self.ds_cor = xr.merge([c1,c2,c3]).astype(int)        
    
    def load_hr2_correlation(self):
        
         #we expect to read nSamples from file between tstart and tstop
        #computed from burstInterval and the amount of samples per burst
        nBursts = int((self.tstop-self.tstart).total_seconds()/self.burstInterval)
        nSamples = nBursts*self.nSampBurst
        
        if self.tstart < self.startTime:
            raise 'must specify a start time after measuring start time: {}'.format(self.startTime)
            
        t = pd.date_range(start =self.tstart, periods = nBursts,
                                freq = '{}S'.format(self.burstInterval))  

        #if we only want to read data from tstart and this is later than the measuring startTime, we need to skip rows while reading  
        skipRowsDat = int((self.tstart - self.startTime).total_seconds()/self.burstInterval*self.nSampBurst   )
        
        #read the file line by line and cast in 2D numpy array
        x=[]
        with open(self.hr2File) as fp:
            for i,line in enumerate(fp):
                # only read the block that we requested
                if ((i>=skipRowsDat) & (i< (skipRowsDat + nSamples))):
                    splittedLine =  [float(ix) for ix in line.split() if len(ix)>0] 
                    x.append(splittedLine[-3:])                    
                elif i>= (skipRowsDat + nSamples):
                    break
        x = np.vstack(x)
        
        #reshape into 3D array. This goes right only when there is no missing data on the file
        x = x.reshape(nBursts,self.nSampBurst,3)
        c1 = np.squeeze(x[:,:,0]).astype(int)
        c2 = np.squeeze(x[:,:,1]).astype(int)
        c3 = np.squeeze(x[:,:,2]).astype(int)
        
        #construct dataset
        ds = xr.Dataset(
            data_vars = dict(
                hr2c1 = (('t','N'),c1,{'long_name':'hr2c1','unit':'%'}),
                hr2c2 = (('t','N'),c2,{'long_name':'hr2c2','unit':'%'}),
                hr2c3 = (('t','N'),c3,{'long_name':'hr2c3','unit':'%'})),
            coords = dict(
                t = t,
                N = self.N
                )
            )   
        ds.N.attrs = {'long_name':'time','units':'s'}
        
        self.ds_hr2 = ds
        
        print('hr2 file was read')
        
        
    def load_sen_data(self):
        nBursts = int((self.tstop-self.tstart).total_seconds()/self.burstInterval)
        nSamples = nBursts*self.nSampBurst
        
        if self.tstart < self.startTime:
            raise 'must specify a start time after measuring start time: {}'.format(self.startTime)
            
        t = pd.date_range(start =self.tstart, periods = nBursts,
                                freq = '{}S'.format(self.burstInterval))  

        #if we only want to read data from tstart and this is later than the measuring startTime, we need to skip rows while reading  
        skipRowsDat = int((self.tstart - self.startTime).total_seconds()/self.burstInterval*self.nSampBurst   )
        
        #read the file line by line and cast in 2D numpy array
        x=[]
        with open(self.senFile) as fp:
            for i,line in enumerate(fp):
                # only read the block that we requested
                if ((i>=skipRowsDat) & (i< (skipRowsDat + nSamples))):
                    splittedLine =  [float(ix) for ix in line.split() if len(ix)>0] 
                    x.append(splittedLine[12:])
                elif i>= (skipRowsDat + nSamples):
                    break
        x = np.vstack(x)
        
        #reshape into 3D array. This goes right only when there is no missing data on the file
        x = x.reshape(nBursts,self.nSampBurst,7)
        heading = np.squeeze(x[:,:,0]).astype(int) # deg
        pitch = np.squeeze(x[:,:,1]).astype(int) #deg
        roll = np.squeeze(x[:,:,2]).astype(int) #deg       
        pressure= np.squeeze(x[:,:,3])*1e4 #Pa
        temperature = np.squeeze(x[:,:,4]) #deg C
        anl1 = np.squeeze(x[:,:,5]) # counts
        anl2 = np.squeeze(x[:,:,6]) # counts
        #construct dataset
        ds = xr.Dataset(
            data_vars = dict(
                heading = (('t','N'),heading,{'long_name':'heading','unit':'deg'}),
                pitch = (('t','N'),pitch,{'long_name':'pitch','unit':'deg'}),
                roll = (('t','N'),roll,{'long_name':'roll','unit':'deg' }),
                p = (('t','N'),pressure,{'long_name':'pressure','unit':'Pa' }),
                temp = (('t','N'),temperature,{'long_name':'temperature','unit':'deg C' }),
                anl1=(('t', 'N'), anl1, {'long_name': 'analog input 1', 'unit': 'counts'}),
                anl2 = (('t', 'N'), anl2, {'long_name': 'analog input 2', 'unit': 'counts'})),
            coords = dict(
                t = t,
                N = self.N
                )
            )   
        ds.N.attrs = {'long_name':'time','units':'s'}
        
        self.ds_sen = ds
        print('sen file was read')
           

    
    def load_all_data(self):
        '''
        make all data available in one dataset
        '''
        self.load_velocity_data()
        self.load_amplitude_data()
        self.load_correlation_data()
        self.load_hr2_correlation()
        self.load_sen_data()

        ds = self.ds_vel       

        if hasattr(self, 'ds_cor'):
            ds = ds.merge(self.ds_cor)
            
        if hasattr(self, 'ds_amp'):
            ds = ds.merge(self.ds_amp)
            
        if hasattr(self, 'ds_hr2'):
            ds = ds.merge(self.ds_hr2)
        if hasattr(self, 'ds_sen'):
            ds = ds.merge(self.ds_sen)            
        self.ds = ds
        
        ds['name'] = self.name
        return ds    

    def get_dataset(self):
        if hasattr(self, 'ds'):
            ds = self.ds
            ds['name'] = self.name
            ds['zb'] = self.zb
            ds['zi'] = self.zi
            ds.zb.attrs={'long_name':'bed level','units':'m+NAP'}
            ds.zi.attrs={'long_name':'instrument level','units':'m+NAP'}
            
            return ds    
        
        
    def plot_profile(self,time = None, it = None,iN = 0):
        import matplotlib.pyplot as plt
        
        if not time is None:
            dsSlice = self.ds.sel(t=time).sel(N=iN)
        elif not it is None:
            dsSlice = self.ds.isel(t=it).sel(N=iN)
        else:
            raise 'either specify time or timeindex to plot!'
            
        fig, ax = plt.subplots(1,3,figsize=[8.3,5.8])
        ax[0].plot(dsSlice.v1,dsSlice.z,label=dsSlice.v1.attrs['long_name'])
        ax[0].plot(dsSlice.v2,dsSlice.z,label=dsSlice.v1.attrs['long_name'])
        ax[0].plot(dsSlice.v3,dsSlice.z,label=dsSlice.v1.attrs['long_name'])    
        ax[0].set_xlabel('u [m/s]')
        ax[0].set_ylabel('m+NAP')

        ax[1].plot(dsSlice.a1,dsSlice.z,label=dsSlice.v1.attrs['long_name'])
        ax[1].plot(dsSlice.a2,dsSlice.z,label=dsSlice.v1.attrs['long_name'])
        ax[1].plot(dsSlice.a3,dsSlice.z,label=dsSlice.v1.attrs['long_name'])   
        ax[1].set_xlabel('a [count]')
        ax[1].set_title('{}'.format(dsSlice.t.dt.strftime('%Y-%m-%d %H:%M:%S').values))
        
        ax[2].plot(dsSlice.c1,dsSlice.z,label=dsSlice.v1.attrs['long_name'])
        ax[2].plot(dsSlice.c2,dsSlice.z,label=dsSlice.v1.attrs['long_name'])
        ax[2].plot(dsSlice.c3,dsSlice.z,label=dsSlice.v1.attrs['long_name'])               
        ax[2].set_xlabel('cor [%]')  
        ax[2].set_title('hr2 cor: {} {} {}'.format(dsSlice.hr2c1.values,dsSlice.hr2c2.values,dsSlice.hr2c3.values))
        
        fig.tight_layout()
        return fig, ax

    def plot_burst(self,time = None, it = None,beam = 1):
        import matplotlib.pyplot as plt
        
        if not time is None:
            dsSlice = self.ds.sel(t=time)
        elif not it is None:
            dsSlice = self.ds.isel(t=it)
        else:
            raise 'either specify time or timeindex to plot!'
            
        fig, ax = plt.subplots(3,1,figsize=[8.3,5.8])
        dsSlice['v' + str(beam)].plot(ax=ax[0],x='N', y='z')
        dsSlice['a' + str(beam)].plot(ax=ax[1],x='N', y='z',cmap=plt.cm.Reds)
        dsSlice['c' + str(beam)].plot(ax=ax[2],x='N', y='z',cmap=plt.cm.Greens,vmin=0,vmax=100)

        [axi.axhline(self.zb,color='k',linestyle='--') for axi in ax]
        
        ax[1].set_title('')
        ax[2].set_title('')
                
        fig.tight_layout()
        return fig, ax    
    

