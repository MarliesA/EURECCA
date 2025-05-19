import xarray as xr
import puv
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy import signal

@xr.register_dataset_accessor("puv")
class WaveStatMethodAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
    
    def band_pass_filter_ss(self,var,freqband=None, fpminfac=0.5, fpmaxfac=2):
        '''RETURNS part of signal in the Sea Swell Range based on peak period
                   in case freqband is not specified, else uses specified
                   frequencyband 'freqband'.
        '''
        ds = self._obj.dropna(dim='t', subset=[var])
        if len(ds.t) == 0:
            return (('t'), np.nan*np.ones([len(ds.t)]))

        sf = ds.sf.values
        # if by slicing or dropping data the sampling frequency became an array
        if len(np.atleast_1d(sf))>1:
            sf = sf[0][0]
                        
        if freqband==None:
            #velocity components in seaswell range  
            ufunc = lambda x, fp: puv.band_pass_filter2(
                sf,
                x,
                fmin=fpminfac*fp,
                fmax = fpmaxfac*fp,
                retrend=False
                )
            
            return xr.apply_ufunc(ufunc,
                ds[var],ds['fp'],
                input_core_dims=[['N'], []],
                output_core_dims=[['N']], 
                vectorize=True)   
                
        else:
            ufunc = lambda x: puv.band_pass_filter2(
                sf,
                x,
                fmin=freqband[0],
                fmax = freqband[1],
                retrend=False
                )            

            return xr.apply_ufunc(ufunc,
                ds[var],
                input_core_dims=[['N']],
                output_core_dims=[['N']], 
                vectorize=True)   