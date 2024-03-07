import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
import KNMI_readers as readers

class Vector(object):
    '''
    Collection of functions to read raw ascii data from Nortek ADV's and cast them in pandas Dataframes and
    optionally into an xarray dataset. For this second step, one will need to specify what duration the user wants the
    blocks to be in which the data will be analysed (i.e. 10 minute blocks, or one hour blocks etc). This restructuring
    is meaningful if in a later stage the user would want to compute block averages, or perform wave analysis.

    The class works both for data collection in burst mode as well as in continuous mode. Also works if each burst is
    written to a separate file. The function infers this from the filenames in the folder and the header file.
    Therefore, make sure that all output files are present in the same dataFolder.
    '''
    
    def __init__(self, name, dataFolder, tstart=None, tstop=None,
                 ):
        
        self.name = name
        self.dataFolder = dataFolder

        self.get_fileNames()
        self.read_hdr()
        
        # if no tstart and tstop are specified, use the interval from the 
        # header file
        if tstart == None:
            self.tstart = self.startTime
        else:
            tstart = pd.to_datetime(tstart)
            if tstart > self.startTime:
                self.tstart = tstart
            else:
                self.tstart = self.startTime
                
        if tstop == None:
            self.tstop = self.stopTime
        else:
            tstop = pd.to_datetime(tstop)
            if tstop < self.stopTime:
                self.tstop = tstop
            else:
                self.tstop = self.stopTime

        
        if self.tstop<self.tstart:
            self.tstop = self.tstart
       
    def get_fileNames(self):
        '''
        Construct required file paths for later functions
        '''
        
        cdir = os.getcwd()
        os.chdir(self.dataFolder)
        fileName = glob.glob('*.hdr')
        self.hdrFile =self.dataFolder + '\\' + fileName[0]
        
        fileName = glob.glob('*.dat')
        if len(fileName)>1:
            self.oneFilePerBurst = 1
            datFiles = [self.dataFolder + '\\' + ifileName for ifileName in fileName]
            
            # make sure the files are in the right order, even if the ordering
            # of the string interpretation is not correct because the number of
            # digits used in the filename is not correct:
            fileOrder = [int(file.split('_')[-1][:-4]) for file in datFiles]
            self.datFiles = [x for _, x in sorted(zip(fileOrder, datFiles))]
        else:
            self.datFile = self.dataFolder + '\\' + fileName[0]
            self.oneFilePerBurst = 0
            
        fileName = glob.glob('*.sen')
        self.senFile = self.dataFolder + '\\' + fileName[0]
         
        fileName = glob.glob('*.pck')
        self.pckFile = self.dataFolder + '\\' +fileName[0]

        fileName = glob.glob('*.vhd')
        self.vhdFile = self.dataFolder + '\\' +fileName[0]
                
        
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
            for index, line in enumerate(myfile):
                if index > 133:
                    break           
                NewName = line[0:columnPosition].strip()
                var = line[columnPosition:].strip()
                if len(NewName) == 0 & len(var) != 0:
                    myvars[name] = [myvars[name], var]
                else:
                    name = NewName   
                    myvars[name] = var
                count += 1
             
        self.hdr = myvars
        self.nMeas = float(myvars['Number of measurements'])
        
        # enforce that hours minutes seconds are part of the datetimestring
        if len(myvars['Time of first measurement']) <= 16:
            myvars['Time of first measurement'] += ' 00:00:00'
            
        try:
            self.startTime = pd.to_datetime(myvars['Time of first measurement'], 
                                            format='%d-%m-%Y %H:%M:%S')
        except:
            self.startTime = pd.to_datetime(myvars['Time of first measurement'], 
                                            format='%d/%m/%Y %H:%M:%S')
        try:                
            self.stopTime = pd.to_datetime(myvars['Time of last measurement'], 
                                           format='%d-%m-%Y %H:%M:%S')
        except:
            self.stopTime = pd.to_datetime(myvars['Time of last measurement'], 
                                           format='%d/%m/%Y %H:%M:%S')
            
        self.frequency = float(myvars['Sampling rate'].split()[0])
        self.coordinateSystem = myvars['Coordinate system']
        self.transformationMatrix = myvars['Transformation matrix'] 
        if myvars['Burst interval'] == 'CONTINUOUS':
            self.burstMode = False
        else:
            self.burstMode = True
            self.burstInterval = int(myvars['Burst interval'][:-3])  # in seconds
            self.samplesPerBurst = int(myvars['Samples per burst'])
          
        
    def read_raw_data(self):
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

        if self.tstop<=self.tstart:
            return
        
        if self.oneFilePerBurst:  # a separate file for every burst

            nBursts = int(np.floor((self.tstop-self.tstart).total_seconds()/self.burstInterval))
            nSamples = nBursts*self.samplesPerBurst
            
            skipBursts = np.floor((self.tstart - self.startTime).total_seconds()/self.burstInterval) 
            
            self.burstStartTimes = pd.date_range(start=self.tstart, periods=nBursts,
                                               freq='{}S'.format(self.burstInterval))
            burstLocalTime = np.cumsum([1/self.frequency]*self.samplesPerBurst) - 1/self.frequency
            
            dt = pd.to_timedelta(burstLocalTime, unit='s')
            
            timeDat = []
            for ib in np.arange(0, nBursts):
                timeDat += list( self.burstStartTimes[ib] + dt)
                
            # on DAT file a new line for every measurement
            burst = []; u=[]; v=[]; w=[]; cor1=[]; cor2=[]; cor3=[]; p=[]; a1=[]; a2=[]; a3=[];
            snr1=[]; snr2=[]; snr3=[]; anl1=[]; anl2=[];
                       
            for ifile,file in enumerate(self.datFiles):
                # if (ifile>skipBursts) & (ifile <=skipBursts + nBursts) : 
                if (ifile>=skipBursts) & (ifile < skipBursts + nBursts) : 
                    with open(file) as fp:
                        for line in fp:
                            x = [float(ix) for ix in line.split() if len(ix)>0]
                            burst.append(x[0])
                            u.append(x[2])
                            v.append(x[3])
                            w.append(x[4])
                            a1.append(x[5])
                            a2.append(x[6])
                            a3.append(x[7])
                            snr1.append(x[8])
                            snr2.append(x[9])
                            snr3.append(x[10])
                            cor1.append(x[11])
                            cor2.append(x[12])
                            cor3.append(x[13])
                            p.append(x[14])
                            anl1.append(x[15])
                            anl2.append(x[16])                
            
        else:  # all data in one long datfile
            if self.burstMode:  # measurements in burstmode
                nBursts = int(np.floor((self.tstop-self.tstart).total_seconds()/self.burstInterval))
                nSamples = nBursts*self.samplesPerBurst
                
                self.burstStartTimes = pd.date_range(start = self.tstart,periods = nBursts,
                                                freq = '{}S'.format(self.burstInterval))
                burstLocalTime = np.cumsum([1/self.frequency]*self.samplesPerBurst) -1/self.frequency
                
                dt = pd.to_timedelta(burstLocalTime, unit='s')
                timeDat = []
                for ib in np.arange(0,nBursts):
                    timeDat += list(self.burstStartTimes[ib] + dt)

                    
                skipRowsDat = np.floor((self.tstart - self.startTime).total_seconds()/self.burstInterval)*self.samplesPerBurst     
            
            else:  # continuous measurements
                nSamples = (self.tstop-self.tstart).total_seconds()*self.frequency
                timeDat = pd.date_range(start =self.tstart, periods = nSamples,
                                        freq = '{}S'.format(1/self.frequency))       
                skipRowsDat = (self.tstart - self.startTime).total_seconds()*self.frequency   
            
            # print('number of rows to skip = {}'.format(int(skipRowsDat)))         
    
            #----------------------------------------------------------------------
            # on DAT file a new line for every measurement
            #----------------------------------------------------------------------
            burst = []; u=[]; v=[]; w=[]; cor1=[]; cor2=[]; cor3=[]; p=[]; a1=[]; a2=[]; a3=[];
            snr1=[]; snr2=[]; snr3=[]; anl1=[]; anl2=[];
            
            
            with open(self.datFile) as fp:
                for i, line in enumerate(fp):
                    # only read the block that we requested
                    if ((i > skipRowsDat) & (i <= (skipRowsDat+nSamples))):
                        x = [float(ix) for ix in line.split() if len(ix) > 0]
                        burst.append(x[0])
                        u.append(x[2])
                        v.append(x[3])
                        w.append(x[4])
                        a1.append(x[5])
                        a2.append(x[6])
                        a3.append(x[7])
                        snr1.append(x[8])
                        snr2.append(x[9])
                        snr3.append(x[10])
                        cor1.append(x[11])
                        cor2.append(x[12])
                        cor3.append(x[13])
                        p.append(x[14])
                        anl1.append(x[15])
                        anl2.append(x[16])
                    elif i > (skipRowsDat + nSamples):
                        break
                
        p = [1e4*ip for ip in p]  # change dBar to Pascal
        
        if (len(timeDat) != len(u)):
            timeDat = timeDat[:len(u)]
            
        dat = {'t': timeDat,'u': u,'v': v,'w': w,
                  'p': p,
                  'anl1': anl1,'anl2': anl2,
                  'a1': a1,'a2': a2,'a3': a3,
                  'snr1': snr1,'snr2': snr2,'snr3': snr3,
                  'cor1': cor1, 'cor2': cor2, 'cor3': cor3}
        if self.burstMode:
            dat['burst'] = burst
        print('.dat file was read')
        
        #----------------------------------------------------------------------   
        # on the SEN file a new line for every second
        #----------------------------------------------------------------------   
        timeSen = pd.date_range(start = self.tstart, periods = nSamples,
                                freq = '1S')
        skipRowsSen = (self.tstart-self.startTime).total_seconds()
        # print('number of rows to skip = {}'.format(int(skipRowsSen)))

        
        heading = []; pitch = []; roll= []; voltage = []; temp = []
        with open(self.senFile) as fp:
              for i,line in enumerate(fp):
                  # only read the block that we requested
                  if ((i>skipRowsSen) & (i<= (skipRowsSen + nSamples))):
                      x = [float(ix) for ix in line.split() if len(ix)>0]
                      voltage.append(x[8])
                      heading.append(x[10])
                      pitch.append(x[11])
                      roll.append(x[12])
                      temp.append(x[13])
                  elif i> (skipRowsSen + nSamples):
                      break 
                  
        if (len(timeSen)!=len(heading)):
            timeSen = timeSen[:len(heading)]            
        
        sen = {'t': timeSen, 'heading': heading,
                    'pitch': pitch,'roll': roll,'voltage': voltage,'temperature': temp}
        print('.sen file was read')        
        
        # from here cast everythin in dataframe if checks on length hold        
        df = pd.DataFrame(dat)
        df = df.set_index(['t'])
        
        df2 = pd.DataFrame(sen)
        df2 = df2.set_index(['t'])
        
        # resample such that it is on same frequency as the .dat df
        df2 = df2.resample('{}S'.format(1/self.frequency)).asfreq()
        df2 = df2.interpolate('linear')
        
        # join into one dataframe
        df3 = df.join(df2)
        # fill the trailing nans in the sen columns where no extrapolation can
        # place to the last true value
        df3 = df3.fillna(method='ffill')
        self.dfpuv = df3
        return df  
      
    def read_vhd(self):
        '''
        RETURNS: dataframe of bottom pings at start of burst (d1) and end of 
        burst (d2) in mm, indexed by the burst start time

         1   Month                            (1-12)
         2   Day                              (1-31)
         3   Year
         4   Hour                             (0-23)
         5   Minute                           (0-59)
         6   Second                           (0-59)
         7   Burst counter
         8   No of velocity samples
         9   Noise amplitude (Beam1)          (counts)
        10   Noise amplitude (Beam2)          (counts)
        11   Noise amplitude (Beam3)          (counts)
        12   Noise correlation (Beam1)        (%)
        13   Noise correlation (Beam2)        (%)
        14   Noise correlation (Beam3)        (%)
        15   Dist from probe - start (Beam1)  (counts)
        16   Dist from probe - start (Beam2)  (counts)
        17   Dist from probe - start (Beam3)  (counts)
        18   Dist from probe - start (Avg)    (mm)
        19   Dist from s.vol - start (Avg)    (mm)
        20   Dist from probe - end (Beam1)    (counts)
        21   Dist from probe - end (Beam2)    (counts)
        22   Dist from probe - end (Beam3)    (counts)
        23   Dist from probe - end (Avg)      (mm)
        24   Dist from s.vol - end (Avg)      (mm)
        '''

        tlist = []; dist1 = []; dist2 = []  ; burst = []
        with open(self.vhdFile) as file: 
            for line in file:
                parts = line.split()
                tlist.append(parts[0:6])
                # burst.append(parts[7])
                # noise = [int(x) for x in parts[8:11]]
                # noiseCor = [int(x) for x in parts[11:14]]
                dist1.append(float(parts[18])) #mm
                dist2.append(float(parts[-1])) #mm
        tList = [pd.to_datetime(''.join([x for _, x in sorted(zip([1,2,0,3,4,5], t))])) for t in tlist]
        df = pd.DataFrame({'t':tList,'d1':dist1,'d2':dist2})
        self.dfvhd = df.set_index('t')

        return self.dfvhd
    
    def cast_to_blocks_in_xarray(self, blockWidth=600):
        '''
        takes the raw data which are timeseries in pandas DataFrame and casts it
        in blocks (bursts) in an xarray with metadata for easy computations and 
        saving to file (netcdf) later on.
        '''
        
        t = self.dfpuv.index.values

        p = self.dfpuv['p'].values

        N = len(p)
        
        if self.burstMode:
            blockLength = int(self.samplesPerBurst)
        else:                
            blockLength = int(blockWidth * self.frequency)
  
        NB = int(np.floor(N/blockLength))
        
        if NB==0:
            print('there is no data between {} and {}'.format(self.tstart,self.tstop))
            return
        
        p = p[0:int(NB*blockLength)]
        N = len(p)
        p2 = p.reshape(NB,blockLength)
        
        if self.burstMode:
            blockStartTimes = self.burstStartTimes
        else:
            blockStartTimes = t[0::blockLength]
            
        if len(blockStartTimes)>NB:
            blockStartTimes = blockStartTimes[:NB]
                
        # cast all info in dataset

        ds = xr.Dataset(
                data_vars=dict(
                    sf=self.frequency,
                    p=(['t', 'N'], p2)
                    ),                  
                coords=dict(t=blockStartTimes,
                          N=np.arange(0, blockLength)/self.frequency
                ))
          
        # also cast all other variables in the ds structure


        vars = ['u','v','w','anl1','anl2','a1','a2','a3',
                    'cor1','cor2','cor3',
                    'snr1','snr2','snr3',
                    'voltage',
                    'heading','pitch','roll']

        for var in vars:
            tmp = self.dfpuv[var].values
            tmp = tmp[0:int(NB*blockLength)]
            ds[var] = (['t', 'N'], tmp.reshape(NB, int(N/NB)))

        ds['t'].attrs = {'long_name': 'burst start time'}
        ds['N'].attrs = {'units': 's', 'long_name': 'burst local time'}
        ds['sf'].attrs = {'units': 'Hz', 'long_name': 'sampling frequency'}
        ds['p'].attrs = {'units': 'Pascal', 'long_name': 'pressure'}
        ds['u'].attrs = {'units': 'm/s', 'long_name': 'velocity 1 component'}
        ds['v'].attrs = {'units': 'm/s', 'long_name': 'velocity 2 component'}
        ds['w'].attrs = {'units': 'm/s', 'long_name': 'velocity 3 component'}
        ds['anl1'].attrs= {'units': '-', 'long_name': 'analog input 1'}
        ds['anl2'].attrs= {'units': '-', 'long_name': 'analog input 2'}
        ds['a1'].attrs = {'units': '-', 'long_name': 'amplitude beam 1'}
        ds['a2'].attrs = {'units': '-', 'long_name': 'amplitude beam 2'}
        ds['a3'].attrs = {'units': '-', 'long_name': 'amplitude beam 3'}
        ds['cor1'].attrs = {'units': '-', 'long_name': 'correlation beam 1'}
        ds['cor2'].attrs = {'units': '-', 'long_name': 'correlation beam 2'}
        ds['cor3'].attrs = {'units': '-', 'long_name': 'correlation beam 3'}
        ds['snr1'].attrs = {'units': 'dB', 'long_name': 'signal to noise beam 1'}
        ds['snr2'].attrs = {'units': 'dB', 'long_name': 'signal to noise beam 2'}
        ds['snr3'].attrs = {'units': 'dB', 'long_name': 'signal to noise beam 3'}
        ds['heading'].attrs = {'units': 'deg', 'long_name': 'instrument heading'}
        ds['pitch'].attrs = {'units': 'deg', 'long_name': 'instrument pitch'}
        ds['roll'].attrs = {'units': 'deg', 'long_name': 'instrument roll'}
        ds['voltage'].attrs = {'units': 'V', 'long_name': 'battery voltage'}

        if self.burstMode:
            tmp = self.dfpuv['burst'].values
            tmp = tmp[0:int(NB * blockLength)]
            ds['burst'] = (['t', 'N'], tmp.reshape(NB, int(N / NB)))
            ds['burst'].attrs = {'units': '-', 'long_name': 'burst number'}


        self.ds = ds       
        
    def compute_block_averages(self):
         '''
         computes some first block averages, specifically average pressure, 
         water level and water depth on the xarray Dataset
         '''
        
         variables = ['p','u','v','w',
                      'anl1','anl2',
                      'cor1','cor2','cor3',
                      'snr1','snr2','snr3',
                      'heading','pitch','roll']
         
         for var in variables:
             varm = self.ds[var].values.mean(axis=1)           
             self.ds[var + 'm'] = (['t'],varm)          
             self.ds[var + 'm'].attrs = self.ds[var].attrs
             
    # def add_positioning_info(self,df):
    #     """
    #     add positioning data to the dataframe, with the same time spacing as 
    #     the block starttimes.

    #     Parameters
    #     ----------
    #     df : TYPE PANDAS DATAFRAME
    #         with columns zi, zb and ori.

    #     Returns
    #     -------
    #     None, but adds the information to the vectors Dataset in matching time
    #     coordinates.

    #     """
    #     #cast in dataset
    #     pos = xr.Dataset.from_dataframe(df)
    #     pos = pos.rename({'index':'t'}) 
        
    #     #slice only the section that we have observations on
    #     pos = pos.resample({'t':'1H'}).interpolate('nearest')               
    #     pos = pos.sel(t=slice(self.ds.t.min(),self.ds.t.max()))

    #     #bring to same time axis as observations       
    #     pos = pos.resample({'t':'1800S'}).interpolate('nearest')
    #     pos = pos.interpolate_na(dim = 't',method='nearest',fill_value="extrapolate")
   
    #     ds = self.ds.merge(pos)  

    #     ds['zb'].attrs = {'units': 'm+NAP','long_name':'bed level, neg down'} 
    #     ds['h'].attrs = {'units': 'cm','long_name':'instrument height above bed, neg down'} 
    #     ds['io'].attrs= {'units':'deg','long_name':'angle x-dir with north clockwise'}
        
    #     self.ds = ds
        
        

        
      
        
        
        


        

        
            

       