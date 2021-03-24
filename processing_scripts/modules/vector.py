# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:15:29 2021

@author: marliesvanderl
"""
import glob
import os
import numpy as np
import pandas as pd
import xarray as xr

class Vector(object):
    
    def __init__(self,name, dataFolder,zb,zi,tstart='',tstop='',
                 blockWidth=600,
                 fmin = 0.01,
                 fmax = 1.5,
                 fresolution = 0.01,
                 fcorrmin = 0.05,
                 fcorrmax = 1.0,
                 maxiter = 100,
                 rho=1000,
                 g= 9.8):
        
        self.name = name
        self.dataFolder = dataFolder
        self.zb = zb
        self.zi = zi
        self.blockWidth = blockWidth
        self.fmin = fmin
        self.fmax = fmax
        self.fresolution = fresolution
        self.fcorrmin = fcorrmin
        self.fcorrmax = fcorrmax
        self.maxiter = maxiter
        self.rho = rho
        self.g = g

        
        self.get_fileNames()
        self.read_hdr()
        
        # if no tstart and tstop are specified, use the interval from the 
        # header file
        if tstart == '':
            tstart = self.startTime
        else:
            tstart = pd.to_datetime(tstart)
        if tstop == '':
            tstop = self.stopTime
        else:
            tstop = pd.to_datetime(tstop)
        self.tstart = tstart
        self.tstop = tstop
        
        self.load_block()
        self.load_air_pressure_from_knmi()
        self.correct_raw_pressure_for_air_pressure(self.dfpuv, self.pAir)
       
    def get_fileNames(self):
        '''
        Construct required filepaths for later functions

        '''
        
        cdir = os.getcwd()
        os.chdir(self.dataFolder)
        fileName = glob.glob('*.hdr')
        self.hdrFile =self.dataFolder + '\\' + fileName[0]
        
        fileName = glob.glob('*.dat')
        self.datFile = self.dataFolder + '\\' + fileName[0]
         
        fileName = glob.glob('*.sen')
        self.senFile = self.dataFolder + '\\' + fileName[0]
         
        fileName = glob.glob('*.pck')
        self.pckFile = self.dataFolder + '\\' +fileName[0]
        
        fileName = glob.glob('KNMI*.txt')
        self.knmiFileName =self.dataFolder + '\\' + fileName[0]
        
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
        try:
            self.startTime = pd.to_datetime(myvars['Time of first measurement'], 
                                            format = '%d-%m-%Y %H:%M:%S') 
        except:
            self.startTime = pd.to_datetime(myvars['Time of first measurement'], 
                                            format = '%d/%m/%Y %H:%M:%S')  
        try:                
            self.stopTime = pd.to_datetime(myvars['Time of last measurement'], 
                                           format = '%d-%m-%Y %H:%M:%S') 
        except:
            self.stopTime = pd.to_datetime(myvars['Time of last measurement'], 
                                           format = '%d/%m/%Y %H:%M:%S') 
            
        self.frequency = float(myvars['Sampling rate'].split()[0])
        self.coordinateSystem = myvars['Coordinate system']
        self.transformationMatrix = myvars['Transformation matrix']   
               
    def load_block(self):
        '''
        read pressure, velocity components and correlation values from .dat
        read heading pitch and roll from .sen. Puts both in the same dataframe 
        and upsamples the .sen data to the same time index as the .dat measurements
        '''
        
        
        # namesDat = ['burst','ensemble','u','v','w','a1','a2','a3','SNR1',
        #             'SNR2','SNR3','cor1','cor2','cor3','pressure','analog1',
        #             'analog2','Checksum']
        # widthsDat = [(0,5),(5,11),(11,20),(20,29),(29,38),
        #                                     (38,44),(44,50),(50,56),(56,63),
        #                                     (63,70),(70,77),(77,83),(83,89),
        #                                     (89,95),(95,103),(103,109),
        #                                     (109,115),(115,119)]
        # namesSen = ['month','day','year','hour','minute','second','error',
        #             'voltage','soundspeed','heading','pitch','roll',
        #             'temperature','analog1','checksum']
        # widthsSen = [(0,2),(2,5),(5,10),(10,13),(13,16),(16,19),(19,28),
        #              (28,37),(37,43),(43,50),(50,56),(56,62),(62,68),(68,75),
        #              (75,81),(81,84)]
        
        nSamples = (self.tstop-self.tstart).total_seconds()*self.frequency

        #----------------------------------------------------------------------
        # on DAT file a new line for every measurement
        #----------------------------------------------------------------------

        timeDat = pd.date_range(start =self.tstart, periods = nSamples,
                                freq = '{}S'.format(1/self.frequency))       
        skipRowsDat = (self.tstart - self.startTime).total_seconds()*self.frequency   
        # print('number of rows to skip = {}'.format(int(skipRowsDat)))         

        u=[];v=[];w=[];cor1=[];cor2=[];cor3=[];p=[];
        with open(self.datFile) as fp:
            for i,line in enumerate(fp):
                # only read the block that we requested
                if ((i>skipRowsDat) & (i<= (skipRowsDat + nSamples))):
                    x = [float(ix) for ix in line.split() if len(ix)>0]
                    u.append(x[2])
                    v.append(x[3])
                    w.append(x[4])
                    cor1.append(x[11])
                    cor2.append(x[12])
                    cor3.append(x[13])
                    p.append(x[14])
                elif i> (skipRowsDat + nSamples):
                    break
        # print('len(x) = {}'.format(len(timeDat)))
        # print('len(u) = {}'.format(len(u)) )
        
        p = [1e4*ip for ip in p] # change dBar to Pascal
        
        if (len(timeDat)!=len(u)):
            timeDat = timeDat[:len(u)]
            
        dat={'t':timeDat,'u':u,'v':v,'w':w,
                  'p':p,
                  'cor1':cor1, 'cor2':cor2, 'cor3':cor3}
        print('.dat file was read')
        
        #----------------------------------------------------------------------   
        # on the SEN file a new line for every second
        #----------------------------------------------------------------------   
        timeSen = pd.date_range(start = self.tstart, periods = nSamples,
                                freq = '1S')
        skipRowsSen = (self.tstart-self.startTime).total_seconds()
        # print('number of rows to skip = {}'.format(int(skipRowsSen)))

        
        heading = []; pitch = []; roll= [];
        with open(self.senFile) as fp:
              for i,line in enumerate(fp):
                  # only read the block that we requested
                  if ((i>skipRowsSen) & (i<= (skipRowsSen + nSamples))):
                      x = [float(ix) for ix in line.split() if len(ix)>0]
                      heading.append(x[9])
                      pitch.append(x[10])
                      roll.append(x[11])
                  elif i> (skipRowsSen + nSamples):
                      break 
                  
        if (len(timeSen)!=len(heading)):
            timeSen = timeSen[:len(heading)]            
        
        sen = {'t':timeSen, 'heading':heading,
                    'pitch':pitch,'roll':roll}
        print('.sen file was read')        
        
        # from here cast everythin in dataframe if checks on length hold
        
        df = pd.DataFrame(dat)
        df = df.set_index(['t'])
        
        df2 = pd.DataFrame(sen)
        df2 = df2.set_index(['t'])
        
        #resample such that it is on same frequency as the .dat df
        df2 = df2.resample('{}S'.format(1/self.frequency)).asfreq()
        df2 = df2.interpolate('linear')
        
        #join into one dataframe
        df3 = df.join(df2)
        #fill the trailing nans in the sen columns where no extrapolation can
        #place to the last true value
        df3 = df3.fillna(method='ffill')
        self.dfpuv = df3
        
        return df  
      

    def load_air_pressure_from_knmi(self,stationNumber = 235):
        '''
        casts air pressure data from knmi in their download format to a pandas
        Dataframe
        '''
        
        df = load_air_pressure_from_knmi(self.knmiFileName,
                                    stationNumber = stationNumber)
        
        self.pAir = df

    def correct_raw_pressure_for_air_pressure(self,dfp,pAir,emergedT0=False):
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
        
        #correct the pressure signal with dpAir and with drift in instrument pressure
        dfp['pRaw'] = dfp['p']
        
        if emergedT0:
            #compute drift in instrument measurement of air pressure through checking with knmi station
            print('warning: pressure correction assumes that the instrument was out of the water for the first 10 min of measurement')
            dfp['drift']=np.nan
            dfp['drift'].loc[i0] = p0.p.mean()-p0.pAir.mean()
            dfp['drift'].loc[i1] = p1.p.mean()-p1.pAir.mean()
            dfp['drift'] = dfp['drift'].interpolate() - dfp['drift'].loc[i0]       
        
            dfp['p'] = dfp['pRaw']- dfp['pRaw'].loc[i0]-dfp['dpAir'] - dfp['drift']
        else:
            dfp['p'] = dfp['pRaw'] - dfp['dpAir']          
        
        self.dfpuv = dfp  

        return dfp



    def cast_to_blocks_in_xarray(self):
        '''
        takes the raw data which are timeseries in pandas DataFrame and casts it
        in blocks (bursts) in an xarray with metadata for easy computations and 
        saving to file (netcdf) later on.
        '''

        t = self.dfpuv.index.values

        p = self.dfpuv['p'].values

        
        N = len(p)
        
        blockLength = int(self.blockWidth * self.frequency)
        NB = int(np.floor(N/blockLength))
        p = p[0:int(NB*blockLength)]
        N = len(p)
        p2 = p.reshape(NB,blockLength)

        blockStartTimes = t[0::blockLength]
        if len(blockStartTimes)>NB:
            blockStartTimes = blockStartTimes[:-1]
            
        #cast all info in dataset
        ds = xr.Dataset(
                data_vars = dict(
                    f = self.frequency,
                    zb = self.zb,
                    zi = self.zi,  
                    rho = self.rho,
                    g = self.g,
                    p =(['t','N'],p2)
                    ),                  
                coords = dict(t = blockStartTimes,
                          N = np.arange(0,blockLength)
                ))
          
        # also cast all other variables in the ds structure 
        for var in ['u','v','w','cor1','cor2','cor3','heading','pitch','roll']:
            tmp = self.dfpuv[var].values
            tmp = tmp[0:int(NB*blockLength)]
            ds[var] = (['t','N'],tmp.reshape(NB,int(N/NB)))
        
            
        ds['t'].attrs = {'long_name': 'burst start times'}
        ds['f'].attrs = {'units': 'Hz','long_name':'sampling frequency'} 
        ds['zb'].attrs = {'units': 'm+NAP','long_name':'bed level, neg down'} 
        ds['zi'].attrs = {'units': 'm+NAP','long_name':'instrument level, neg down'} 
        ds['rho'].attrs = {'units': 'kg m-3','long_name':'water density'}
        ds['g'].attrs = {'units': 'm s-2','long_name':'gravitational acceleration'}
        ds['p'].attrs = {'units': 'Pascal','long_name':'pressure signal corrected for air pressure and  (possibly) instrument drift'} 
        ds['u'].attrs = {'units': 'm/s','long_name':'velocity East component'}             
        ds['v'].attrs = {'units': 'm/s','long_name':'velocity North component'}            
        ds['w'].attrs = {'units': 'm/s','long_name':'velocity Up component'} 
        ds['cor1'].attrs = {'units': '-','long_name':'correlation of beam 1'} 
        ds['cor2'].attrs = {'units': '-','long_name':'correlation of beam 2'} 
        ds['cor3'].attrs = {'units': '-','long_name':'correlation of beam 3'} 
        ds['heading'].attrs = {'units': 'deg','long_name':'instrument heading'} 
        ds['pitch'].attrs = {'units': 'deg','long_name':'instrument pitch'} 
        ds['roll'].attrs = {'units': 'deg','long_name':'instrument roll'}   
                          
        self.ds = ds

    def reference_pressure_to_water_level_observations(self,referenceDataPath):
        '''
        To be executed after pressure correction, before any attenuation 
        functionality is called, in case we can not rely on the reference level of the 
        pressure from the ADV itself. It then takes the reference pressure from the
        alternative instrument provided
        '''
              
        print('using zs measurements from other intrument to correct pressure')

        
        #determine time interval over which the water level observations need to be averaged
        dfp = self.dfpuv
        i0 = dfp.apply(pd.Series.first_valid_index)['p']
        i1 = dfp.apply(pd.Series.last_valid_index)['p']
        
        #compute hydrostatic pressure corresponding to certain water level in Pa
        inst = xr.open_dataset(referenceDataPath)
        phyd = (inst.zsmean.loc[i0:i1]-self.zi)*self.rho * self.g #hydrostatic pressure in Pa
        phyd = phyd.to_series()
        phyd= phyd.resample('{}S'.format(1/self.frequency)).asfreq()
        phyd = phyd.interpolate('linear')
        
        #correct the data in the DataFrame
        dfp['pRaw'] = dfp['p']
        dfp['p'] = dfp['p']-dfp['p'].mean() + phyd
        
        self.dfpuv = dfp
        
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
        

        
            
def load_air_pressure_from_knmi(filePath,stationNumber = 235):
    '''
    reads the text file and casts data in a dataframe in the unit of Pascals
    '''
            
    date=[]; h=[]; T=[]; pa=[];no = [];
    with open(filePath) as fp:
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
    pAir = pAir[pAir['stationNo']==stationNumber] #make sure only the Kooij data is in there
    pAir.drop(columns={'stationNo'},inplace=True)  
    
    return pAir

       