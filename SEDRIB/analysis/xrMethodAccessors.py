import xarray as xr
import numpy as np
from scipy.fft import fft, ifft
from scipy import signal

def band_pass_filter(sf,x,fmin=0.05,fmax=3,retrend=True):
    '''
    band pass filters a signal to the range fmin and fmax. Gives identical results to band_pass_filter above
        

    Parameters
    ----------
    sf : FLOAT
        SAMPLING FREQUENCY.
    x : NUMPY ARRAY
        SIGNAL.
    fmin : FLOAT, optional
        LOWER BOUND OF BAND PASS FILTER. The default is 0.05.
    fmax : FLOAT, optional
        UPPER BOUND OF BAND PASS FILTER. The default is 3.

    Returns
    -------
    NUMPY ARRAY
        BAND-PASS FILTERED SIGNAL.
    ''' 

    #ML force signal to be of even length 
    if len(x)%2==1:
        x=x[:-1]

    pex = x-signal.detrend(x)
    x = x-pex
           
    #frequency range
    nf = int(np.floor(len(x)/2))
    df = sf/len(x)
    ffa = np.arange(0,nf)
    ffb = -np.arange(nf,0,-1)
    f = df*np.append(ffa,ffb)
    
    Q = fft(x)
        
    Q2 = Q[:]
    Q2[np.logical_or(abs(f)<fmin,abs(f)>=fmax)]=0
    
    if retrend is True:
        return ifft(Q2).real + pex 
    else: 
        return ifft(Q2).real
    
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
            ufunc = lambda x, fp: band_pass_filter(
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
            ufunc = lambda x: band_pass_filter(
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