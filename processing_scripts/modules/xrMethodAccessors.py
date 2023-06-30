import xarray as xr
import puv
import numpy as np
import pandas as pd
from matplotlib import cm

@xr.register_dataset_accessor("puv")
class WaveStatMethodAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
    
    def fill_nans(self,var,maxFracNans=0.04, maxGap=16, **kwargs):
        ds = self._obj

        maxNumNans = int(len(ds.N)*maxFracNans)
        ds[var + 'w'] = ds[var].where(np.isnan(ds[var]).sum(dim='N') < maxNumNans)
        ds[var + 'w'] = ds[var + 'w'].interpolate_na(dim='N', method='cubic', max_gap=maxGap)
        ds[var + 'w'] = ds[var + 'w'].ffill(dim='N').bfill(dim='N')        
        ds[var + 'w'].attrs = {'units': ds[var].attrs['units'], 'long_name': ds[var].attrs['units'], 'comments': 'interpolated per bursts'}
        return ds[var + 'w']
    
    def disper(self, T='Tmm10', d='d'):
        ds = self._obj.dropna(dim='t', subset=[T, d])

        # return nans if there is no data remaining
        if len(ds.t) == 0:
            return (('t'), np.nan*np.ones(len(self._obj.t)))

        k = xr.apply_ufunc(
            lambda tm01, d: puv.disper(2*np.pi/tm01, d),
            ds[T], ds[d],
            input_core_dims=[[], []],
            output_core_dims=[[]],
            vectorize=True
            )
        k.name = 'k'
        k.attrs = {'long_name': 'k', 'units': 'm-1'}
        return k

    def disper_cur(self, T='Tmm10', d='d', u='ud_mean'):
        '''
        current corrected wave number from dispersion relation
        :param T: wave period
        :param d: water depth
        :param u: ambient velocity
        :return: wave number k
        '''
        ds = self._obj.dropna(dim='t', subset=[T, d, u])

        # return nans if there is no data remaining
        if len(ds.t) == 0:
            return (('t'), np.nan*np.ones(len(self._obj.t)))

        ufunc = lambda T, h, u: puv.disper_cur(2*np.pi/T, h, u)
        k = xr.apply_ufunc(
            ufunc,
            ds[T], ds[d], ds[u],
            input_core_dims=[[], [], []],
            output_core_dims=[[]], 
            vectorize=True
            )
        k.name = 'k'
        k.attrs = {'long_name': 'k', 'units': 'm-1', 'comments': 'current corrected'}
        return k
    
    def spectrum_simple(self, var, **kwargs):
        """compute spectrum simple"""
        #only apply on non-nan rows
        ds = self._obj.dropna(dim='t', subset=[var])

        #return nans if there is no data remaining
        if len(ds.t) == 0:
            return (('t', 'f'), np.nan*np.ones([len(ds.t), len(ds.f)]))
        
        ufunc = lambda x: puv.spectrum_simple(ds.sf.values, x, **kwargs )
        return xr.apply_ufunc(ufunc,
            ds[var],
            input_core_dims=[['N']],
            output_core_dims=[['f'],['f']], 
            vectorize=True)

    def attenuate_signal(self, Type, var,zi  = 'zi', zb = 'zb' , d = 'd', **kwargs):
        '''
        reconstructs the depth-attenuation corrected surface elevation from a pressure or velocity signal
        :param Type: the signal type, one of ['pressure','horizontal','vertical']
        :param var: signal's variable name
        :param zi: instrument height
        :param zb: bed level
        :param d: water depth
        :param kwargs:
        :return: reconstructed depth-attenuation corrected surface elevation
        '''
        ds = self._obj

        # only apply on non-nan rows
        ds = self._obj.dropna(dim='t', subset=[var])

        # return nans if there is no data remaining
        if len(ds.t) == 0:
            return (('t', 'N'), np.nan * np.ones([len(ds.t), len(ds.N)]))

        ufunc = lambda x,d,zi,zb: puv.attenuate_signal(
                Type,ds.sf.values,x, d, zi, zb, **kwargs
                )
        return xr.apply_ufunc(
            ufunc,
            ds[var], ds[d], ds[zi], ds[zb],
            input_core_dims=[['N'], [], [], []],
            output_core_dims=[['N']], 
            vectorize=True
            )

    def attenuation_corrected_spectrum(
            self, Type, var,zi = 'zi', zb = 'zb', d = 'd', **kwargs):

        # only apply on non-nan rows
        ds = self._obj.dropna(dim='t', subset=[var])

        # return nans if there is no data remaining
        if len(ds.t) == 0:
            return (('t', 'f'), np.nan * np.ones([len(ds.t), len(ds.f)]))

        ufunc = lambda x,d,zi,zb: puv.attenuation_corrected_wave_spectrum(
                Type,ds.sf.values, x, d, zi, zb, **kwargs
                )
        return xr.apply_ufunc(ufunc,
            ds[var], ds[d], ds[zi], ds[zb],
            input_core_dims=[['N'], [],[], []],
            output_core_dims=[['f'], ['f']],
            vectorize=True
            )
                              
    def attenuation_factor(self, Type, elev = 'elev',d = 'd',**kwargs):
        ds = self._obj

        ufunc = lambda elev, h: puv.attenuation_factor(
            Type, elev, h, ds.f.values,**kwargs)
        return xr.apply_ufunc(ufunc,
            ds[elev],ds[d],
            input_core_dims=[[],[]],
            output_core_dims=[['f']], 
            vectorize=True
            )
    
    def get_peak_frequency(self,var,fpmin=0.01):
        ''' RETURNS peak frequency of spectra in var'''

        # only apply on non-nan rows
        ds = self._obj.dropna(dim='t', subset=[var])

        # return nans if there is no data remaining
        if len(ds.t) == 0:
            return (('t'), np.nan * np.ones([len(ds.t)]))

        fx, vy = ds.puv.spectrum_simple(var)

        ufunc = lambda fx, vy: puv.get_peak_frequency(
            fx = fx,
            vy = vy,
            fpmin=fpmin)  
    
        return xr.apply_ufunc(ufunc,
            fx.dropna(dim='t'),vy.dropna(dim='t'),
            input_core_dims=[['f'],['f']],
            output_core_dims=[[]], 
            vectorize=True)  

    def puv_wavedir(self,p='p',u='u',v='v',**kwargs):

        # only apply on non-nan rows
        ds = self._obj.dropna(dim='t',subset=[p,u,v])

        # return nans if there is no data remaining
        if len(ds.t) == 0:
            return (('t'), np.nan * np.ones([len(ds.t)]))

        sf = ds.sf.values
        ufunc = lambda p,u,v: puv.puv_wavedir(
            sf, _p=p, _u=u, _v=v, **kwargs)
        
        return xr.apply_ufunc(ufunc,
                              ds[p], ds[u], ds[v],
                              input_core_dims=[['N'], ['N'], ['N']],
                              output_core_dims=[[]],
                              vectorize=True)
    
    def band_pass_filter_ss(self,var,freqband=None):
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
                fmin=0.5*fp,
                fmax = 4,
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

    

    def compute_wave_params(self,var='vy',**kwargs):

        # only apply on non-nan rows
        ds = self._obj.dropna(dim='t',subset=[var])

        # return nans if there is no data remaining
        if len(ds.t) == 0:
            return (('t'), np.nan * np.ones([len(ds.t)]),
                    ('t'), np.nan * np.ones([len(ds.t)]),
                    ('t'), np.nan * np.ones([len(ds.t)]),
                    ('t'), np.nan * np.ones([len(ds.t)]),
                    ('t'), np.nan * np.ones([len(ds.t)]),
                    ('t'), np.nan * np.ones([len(ds.t)])
                    )

        ufunc = lambda vy: puv.compute_wave_params(
            ds.f.values, vy, returntype = 'list', **kwargs)
        return xr.apply_ufunc(ufunc,
            ds[var],
            input_core_dims=[['f']],
            output_core_dims=[[], [], [], [], [], []],
            vectorize=True
            )     

    def compute_wave_params_from_S(self,var='S',**kwargs):

        # only apply on non-nan rows
        ds = self._obj.dropna(dim='t',subset=[var])

        # return nans if there is no data remaining
        if len(ds.t) == 0:
            return (('t'), np.nan * np.ones([len(ds.t)]),
                    ('t'), np.nan * np.ones([len(ds.t)]),
                    ('t'), np.nan * np.ones([len(ds.t)]),
                    ('t'), np.nan * np.ones([len(ds.t)]),
                    ('t'), np.nan * np.ones([len(ds.t)]),
                    ('t'), np.nan * np.ones([len(ds.t)]),
                    ('t'), np.nan * np.ones([len(ds.t)]),
                    ('t'), np.nan * np.ones([len(ds.t)])
                    )

        vy = ds[var].integrate(coord='theta')
        ufunc = lambda vy,S: puv.compute_wave_params(
            ds.f.values,vy,S, theta = ds.theta.values,returntype = 'list',**kwargs)
        return xr.apply_ufunc(ufunc,
            vy, ds[var],
            input_core_dims=[['f'],['f','theta']],
            output_core_dims=[[],[],[],[],[],[],[], []],
            vectorize=True
            )     
    
    def compute_SVD_angle(self, u='u_ss', v='v_ss', fp='fp', freqband=None):

        # only apply on non-nan rows
        ds = self._obj.dropna(dim='t', subset=[u, v])

        # return nans if there is no data remaining
        if len(ds.t) == 0:
            return (('t'), np.nan * np.ones([len(ds.t)]))

        if freqband is None:
            ufunc = lambda u, v, fp: ((puv.compute_SVD_angle(
                ds.sf.values, u, v,
                fmin=0.5*fp, fmax=ds.sf.values/2))*180/np.pi)
        else:
            assert len(freqband) == 2, 'prescribe a frequency band through [fmin, fmax]'

            ufunc = lambda u, v, fp: ((puv.compute_SVD_angle(
                ds.sf.values, u, v,
                fmin=freqband[0], fmax=freqband[1]))*180/np.pi)
        ang = xr.apply_ufunc(ufunc,
                            ds[u], ds[v], ds[fp],
                            input_core_dims=[['N'], ['N'], []],
                            vectorize=True) 

        return ang

    def compute_spectral_width(self, var='vy'):

        # only apply on non-nan rows
        ds = self._obj.dropna(dim='t',subset=[var])

        # return nans if there is no data remaining
        if len(ds.t) == 0:
            return (('t'), np.nan * np.ones([len(ds.t)])  )

        fx = ds.f.values
        ufunc = lambda vy: puv.compute_spectral_width(
            fx,vy)
        return xr.apply_ufunc(ufunc,
            ds[var],
            input_core_dims=[['f']],
            output_core_dims=[[]], 
            vectorize=True
            ) 
    
    def rotate_velocities(self, u, v, theta):
        # only apply on non-nan rows
        ds = self._obj.dropna(dim='t',subset=[u, v])

        # return nans if there is no data remaining
        if len(ds.t) == 0:
            return (('t', 'N'), np.nan * np.ones([len(ds.t), len(ds.N)]),
                    ('t', 'N'), np.nan * np.ones([len(ds.t), len(ds.N)]) )
        
        if type(theta) == type('str'):
            ufunc = lambda u, v, thet: puv.rotate_velocities(u, v, thet)
            return xr.apply_ufunc(ufunc,
                        ds[u], ds[v], ds[theta],
                        input_core_dims=[['N'], ['N'], []],
                        output_core_dims=[['N'],['N']], 
                        vectorize=True) 
        else:            
            ufunc = lambda u, v: puv.rotate_velocities(u, v, theta)
            return xr.apply_ufunc(ufunc,
                        ds[u], ds[v],
                        input_core_dims=[['N'], ['N']],
                        output_core_dims=[['N'], ['N']],
                        vectorize=True)     
    
    def wave_MEMpuv(self,eta='etaw',u = 'uw', v='vw', d = 'd', zi = 'zi', zb= 'zb',**kwargs):

        #only apply on non-nan rows
        ds = self._obj.dropna(dim='t',subset=[eta, u, v])

        #return nans if there is no data remaining
        if len(ds.t) == 0:
            return (('t', 'f', 'theta'), np.nan*np.ones([len(ds.t), len(ds.f), len(ds.theta)]))

        fresolution = ds.f.values[1] - ds.f.values[0]
        ufunc = lambda eta,u,v,d,zi,zb: puv.wave_MEMpuv(
                    eta,u,v,d,
                    zi,zb,
                    ds.sf.values,
                    fresolution = fresolution,
                    maxiter = 20,
                    **kwargs)
                   
        fx, vy, theta, S = xr.apply_ufunc(ufunc,
                                 ds[eta],
                                 ds[u],
                                 ds[v],
                                 ds[d], 
                                 ds[zi],
                                 ds[zb],
                                 input_core_dims=[['N'], ['N'], ['N'], [], [], []],
                                 output_core_dims=[['f'], ['f'], ['theta'], ['f', 'theta']],
                                 vectorize=True) 
        
        return S

    def compute_SkAs(self, var, fixedBounds=False, bounds=[0, 8]):

        # only apply on non-nan rows
        ds = self._obj.dropna(dim='t', subset=[var])

        # return nans if there is no data remaining
        if len(ds.t) == 0:
            return (('t'), np.nan * np.ones([len(ds.t)]),
                    ('t'), np.nan * np.ones([len(ds.t)]),
                    ('t'), np.nan * np.ones([len(ds.t)])
                    )

        sf = ds.sf.values

        try:
            if fixedBounds:
                ufunc = lambda x : puv.compute_SkAs(sf,x,fbounds=bounds)
                Sk,As,sig = xr.apply_ufunc(ufunc,
                                    ds[var].dropna(dim='t'),
                                    input_core_dims=[['N']],
                                    output_core_dims=[[], [], []],
                                    vectorize=True)
            else:
                ds['Tp'] = 1 / ds.fp
                ufunc = lambda x, T: puv.compute_SkAs(sf,x,fbounds=[0.5/T, sf/2])
                Sk,As,sig = xr.apply_ufunc(ufunc,
                                    ds[var].dropna(dim='t'),
                                    ds.Tp.dropna(dim='t'),
                                    input_core_dims=[['N'], []],
                                    output_core_dims=[[], [], []],
                                    vectorize=True)
        except:
            Sk = np.nan*ds.Tp
            As = np.nan*ds.Tp
            sig = np.nan*ds.Tp
        return Sk, As, sig   
    
    def Ursell(self,Hm0='Hm0',k='k',d='d'):

        # only apply on non-nan rows
        ds = self._obj.dropna(dim='t', subset=[Hm0, k, d])

        # return nans if there is no data remaining
        if len(ds.t) == 0:
            return (('t'), np.nan * np.ones([len(ds.t)])  )

        Ur =  xr.apply_ufunc(
            lambda hm0, k, d: 3/4 * 0.5 * hm0*k/(k*d)**3,
            ds[Hm0], ds[k], ds[d],
            input_core_dims=[[], [], []],
            output_core_dims=[[]],
            vectorize=True
            )
        Ur.name = 'Ur'
        Ur.attrs = {'long_name': 'Ur', 'units': '-'}
        
        return Ur
    
    
    
@xr.register_dataset_accessor("burst")
class BurstStructureMethodAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def cast_series_to_burst(self,burstDuration):
        '''
        cast_series_to_burst:
            takes an Xarray dataset with a structure of observations and 
            creates bursts of the desired burstDuration. 
        
        Syntax:
        ds = cast_series_to_burst(burstDuration)
        
        Input:
        ds            = xarray Dataset
        burstDuration = desired burst duration in seconds

        Output:
        ds           = xarray Dataset with all original variables
    
        
        Example
        rs = reshape_burst_length(ds, 600, dims=('t','N'))
        
        M. van der Lugt 12 December 2021    
        '''
        ds = self._obj

        ds = ds.rename({'t':'time'})
        sf = ds.sf.values

        #prepare the multiindex:    
        nSamples =len(ds.time)
        burstLength = int(burstDuration*sf)
        nBursts = int(np.floor( nSamples / burstLength ))
        nSamplesNew = nBursts * burstLength
        
        ds = ds.isel(time=slice(None,nSamplesNew))
        
        index = pd.MultiIndex.from_product(
            [ds.time[::burstLength].values,np.arange(burstLength)/sf], 
            names=('t','N')
            )
        
        ds['time'] = index
        reshaped = ds.unstack('time')


        return reshaped.drop('time')          
        
    def reshape_burst_length(self,burstDuration, dims=('t','N')):
        '''
        reshape_burst_length:
            takes an Xarray dataset with a structure of burststarttimes and 
            burstsamples and upsamples or downsamples the bursts to the desired
            burstDuration. Assumes semi-continuous measurements with possibly small
            data gaps between the original bursts. If those moments are present, 
            they are present in the new dataset with nan's on all data_vars.
        
        Syntax:
        ds = reshape_burst_length(ds,burstDuration, [dims=('t','N')])
        
        Input:
        ds            = xarray Dataset
        burstDuration = desired burst duration in seconds
        dims          = name of the dimensions indicating time [t] and burstlength [N]
        
        Output:
        ds           = xarray Dataset with all original variables
    
        
        Example
        rs = reshape_burst_length(ds, 600, dims=('t','N'))
        
        M. van der Lugt 12 December 2021
        M. van der Lugt 08 May 2023: treat burst averaged and parameter variables separately
        '''
        ds = self._obj

        # identify variables that are burst averaged or parameters:
        bavars = [key for key in ds.keys() if (not 'N' in  ds[key].dims and 't' in ds[key].dims)]
        params = [key for key in ds.keys() if (not 'N' in ds[key].dims and (not 't' in ds[key].dims))]

        bavars_ds = ds[bavars]
        params_ds = ds[params]
        ds = ds.drop_vars(bavars)
        ds = ds.drop_vars(params)

        #specify burst time axis in terms of timedeltas
        ds['dt'] = (('N'),pd.to_timedelta(ds.N.values,unit='S') )
        t2, dt2 = xr.broadcast(ds.t,ds.dt)
        #give a time coordinate to every data point
        ds['time'] = t2 + dt2 
           
        #stack the data to a long array for preparation of averaging
        stack0 = ds.stack(z = ('t', 'N') )
        stack0 = stack0.drop_vars({'z', 'N', 't'}, errors='ignore')  # we ignore the error that some variables are not on the dataset
        stack0 = stack0.assign_coords(z=stack0.time)

        #make sure the time axis is complete (and add nans where no data)
        sf = params_ds.sf.values
        stack = stack0.resample(
            z = '{}S'.format(1/sf)).nearest(
                tolerance = '{}S'.format(1/sf)
                )

        # prepare the multiindex:
        nSamples =len(stack.time)
        burstLength = int(burstDuration*sf)
        nBursts = int(np.floor( nSamples / burstLength ))
        nSamplesNew = nBursts * burstLength
        
        stack = stack.isel(z=slice(None,nSamplesNew))
        
        index = pd.MultiIndex.from_product(
            [stack.time[::burstLength].values,np.arange(burstLength)/sf], 
            names=dims
            )
        
        stack['z'] = index
        reshaped = stack.unstack('z')

        # interpolate burst-averaged variables back onto the dataset
        if len(bavars_ds.t) > 1:
            for var in bavars:
                reshaped[var] = bavars_ds[var].interp_like(reshaped)
        else:
            # we can't interpolate if there is just one timestamp on the dataset. In that case just broadcast to all t's
            for var in bavars:
                reshaped[var] = bavars_ds[var]

        # set back parameters onto the dataset
        for var in params:
            reshaped[var] = params_ds[var]

        return reshaped.drop(['dt', 'time'])

    def reduce_burst_length(self, reduce_factor):
        '''
        reshape_burst_length:
            takes an Xarray dataset with a structure of burststarttimes and
            burstsamples and reshapes the bursts to shorter bursts by a factor 'reduce_factor'.
            Simply splits the duration up.

        Syntax:
        ds = reduce_burst_length(ds,reduce_factor)

        Input:
        ds            = xarray Dataset
        reduce_factor = integer larger than 1

        Output:
        ds           = xarray Dataset with all original variables on shorter bursts


        Example
        rs = reshape_burst_length(ds, 3)

        M. van der Lugt 05 June 2023

        '''
        ds = self._obj

        # some parameters from the input dataset
        sf = ds.sf.values
        N0 = len(ds.N)
        nt0 = len(ds.t)
        N1 = int(N0 / reduce_factor)
        # we add timestamps to the time axis based on the amount of blocks we reformat the structure in
        tt = []
        for i in range(reduce_factor):
            tt.append(
                ds.t.values + pd.Timedelta('{}s'.format(int(i * N1 / sf)))
            )

        t1 = np.array(tt).T.reshape(reduce_factor * nt0)
        N1 = int(N0 / 3)

        ds2 = xr.Dataset(
            data_vars={},
            coords={'t': t1,
                    'N': list(range(N1))}
        )

        # differentiate between burst, timeseries and parameter variables
        for var in ds.data_vars:
            if 'N' in ds[var].dims:
                # reshape the raster data
                ds2[var] = (('t', 'N'), ds[var].values.reshape(len(t1), N1))
            elif 't' in ds[var].dims:
                # interpolate onto finer temporal dimension
                if len(ds.t)>1:
                    ds2[var] = ds[var].interp_like(ds2)
                else:
                    ds2[var] = (('t'), list(ds[var].values)*reduce_factor)
            else:
                ds2[var] = ds[var]

        # copy over metadata
        ds2.attrs = ds.attrs

        return ds2


@xr.register_dataset_accessor("plotting") 
class plotMethodAccessor:  
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        
        
        
        
        # TODOOO
        
        
    def plot_velocity_rose(self, dirName, magName, ax=None, wind_dirs=None, spd_bins=None, r_ticks = None, palette=None,legend=True, legkwargs = {}, units = 'm/s'):
    
        if wind_dirs is None:
            wind_dirs = np.arange(0, 360, 15)        
        if spd_bins is None:
            spd_bins = [-1, 2, 5, 10, 15, 20, np.inf]
        if palette is None:
            palette = cm.get_cmap('inferno',len(spd_bins)-1)
            
        bar_dir, bar_width = _convert_dir(wind_dirs)
    
        rosedata = _prepare_rosedata(ds,dirName, magName, spd_bins, units = units)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))       
            # ax.set_theta_direction('clockwise')
            # ax.set_theta_zero_location('N')
    
        for n, (c1, c2) in enumerate(zip(rosedata.columns[:-1], rosedata.columns[1:])):
            if n == 0:
                # first column only
                ax.bar(bar_dir, rosedata[c1].values, 
                       width=bar_width,
                       color=palette(0),
                       edgecolor='none',
                       label=c1,
                       linewidth=0)
                
            ax.bar(bar_dir, rosedata[c2].values, 
                   width=bar_width, 
                   bottom=rosedata.cumsum(axis=1)[c1].values,
                   color=palette((n+1)/rosedata.shape[1]),
                   edgecolor='none',
                   label=c2,
                   linewidth=0)
        
        if legend is True and len(legkwargs)==0:
            leg = ax.legend(loc=(0.75, 0.95), ncol=2,frameon=False)
        elif legend is True:
            leg = ax.legend(**legkwargs)
            
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            xtl = ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SW'])
            
        if not r_ticks is None:
            ax.set_rticks(r_ticks)
        return ax    
    
