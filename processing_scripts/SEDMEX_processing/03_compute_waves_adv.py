# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:28:24 2021

@author: marliesvanderl
"""

import xarray as xr
import numpy as np
import glob
import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../modules'))
import puv
import xrMethodAccessors
import pdb
import pandas as pd
import shutil

#%%
experimentFolder = r'u:\EURECCA\fieldvisits\20210908_campaign\instruments'



instruments = [
        'L1C1VEC',
        'L2C2VEC',
        'L2C4VEC',
        'L2C3VEC',
        'L2C10VEC',
        'L3C1VEC',
        'L5C1VEC',
        'L6C1VEC',
        'L4C1VEC'
]

for instrumentName in instruments:

    #find all files that need to be processed
    fileNames = glob.glob(experimentFolder + r'/' + instrumentName + r'/QC_loose' + r'/*.nc')
    print(instrumentName)

    #prepare the save directory and place a copy of the script in this file
    ncOutDir = experimentFolder + '//' +  instrumentName + r'\tailored_loose'
    if not os.path.isdir(ncOutDir):
        os.mkdir(ncOutDir)

    #make a txtfile copy of the executed script
    # shutil.copy(__file__,ncOutDir + os.sep + os.path.split(__file__)[-1][:-2] + 'txt')


    dsList = []
    for file in fileNames:

        with xr.open_dataset(file) as ds:
            # if np.logical_or(
            #         ds.t.min().values<pd.to_datetime('20210918'),
            #         ds.t.min().values>=pd.to_datetime('20211007')
            #         ):
                # continue
            print(file)
            #we reshape and keep restore the mask coordinates
            ds['mv'] = (('t','N'),ds.coords['maskv'].values)
            ds2 = ds.burst.reshape_burst_length(600)

            ds2.coords['maskv'] = ds2.mv

            #drop bursts with too many nans (>5%) in the velocity data
            ds3 = ds2.where(
                np.isnan(ds2.where(ds2.maskv).u).sum(dim='N')
                < 0.05*len(ds2.N)).dropna(dim='t',how='all')

            # #make sure the vars keep their original dimensions
            ds3 = ds2.sel(t=ds3.t)

            #interpolate nans
            N = len(ds.N)
            for var in ['u','v','eta']:
                #interpolate the bursts where there is less than 5% nans
                data = ds3[var].where(
                    np.isnan(ds3[var]).sum(dim='N') < 0.05*len(ds3.N)
                    ).dropna(dim='t',how='all')
                if len(data.t)!=0:
                    ds3[var] = data.interpolate_na(
                        dim = 'N',
                        method = 'cubic',
                        max_gap = 8 )

                #and fill the gaps more than 8 in length with the burst average
                ds3[var] = ds3[var].fillna(ds3[var].mean(dim='N'))

            ds = ds3

            #if there is completely no data, continue
            if np.sum(~np.isnan(ds.eta))==0:
                print('stuf goes even more wrong')
                continue



            #water level is taken from ossi C9
            eta = xr.open_dataset(experimentFolder + r'\L2C9OSSI\tailored\L2C9OSSI.nc')
            zs = eta.zs.sel(t=slice(ds.t.min(),ds.t.max()))
            if len(ds.t)>1:
                ds['zs'] = zs.interp_like(ds.t)
            else:
                ds['zs'] = zs
            ds['zs'].attrs = {'units':'m+NAP','long_name':'water level'}
            ds = ds.dropna(dim='t',subset=['zs']) #no data when water level is undefined

            ds['d'] = ds.zs-ds.zb
            ds['d'].attrs = {'units':'m ','long_name':'water depth'}
            ds['elev'] = ds.zi-ds.zb
            ds['elev'].attrs = {'units':'m ','long_name':'height probe above bed'}
            ds['elevp'] = ds.zip-ds.zb
            ds['elevp'].attrs = {'units':'m ','long_name':'height ptd above bed'}

            ds['um'] = ds.u.mean(dim='N')
            ds['vm'] = ds.v.mean(dim='N')
            ds['uang'] = np.arctan2(ds.vm,ds.um)*180/np.pi
            ds['umag'] = np.sqrt(ds.um**2+ds.vm**2)
            ds['um'].attrs = {'units':'m/s ','long_name':'u-east'}
            ds['vm'].attrs = {'units':'m/s ','long_name':'v-north'}
            ds['umag'].attrs = {'units':'m/s ','long_name':'flow velocity'}
            ds['uang'].attrs = {'units':'deg ','long_name':'flow direction',
                                'comment':'cartesian convention'}




            #%% extent the dataset with appropriate frequency axis
            print('extending dataset with freq axis')
            sf = ds.sf.values
            powerStep       = 7 #9 for investigating the spectra 9 = 0.03Hz resolution
            fresolution = np.round(sf*np.exp(-np.log(2)*powerStep),5)
            ds2 = xr.Dataset(
                data_vars = {},
                coords = {'t':ds.t,
                         'N':ds.N,
                         'f':np.arange(0,ds.sf.values/2,fresolution)}
                )


            ds2['f'].attrs = {'long_name':'f','units':'Hz'}
            ds2['N'].attrs = {'long_name':'burst time','units':'s'}
            for key in ds.data_vars:
                ds2[key] = ds[key]
            ds2.attrs = ds.attrs
            ds = ds2


            #%% statistics computed from pressure
            print('statistics from pressure')
            _,vy   = ds.puv.spectrum_simple('eta',fresolution = fresolution)

            #compute the attenuation factor
            #attenuation corrected spectra
            Sw = ds.puv.attenuation_factor('pressure',elev='elevp',d='d')
            ds['vyp'] = Sw*vy

            kwargs  = {'fmin':0.1,'fmax':1.5}
            ds['Hm0'],ds['Tp'],ds['Tm01'],_,ds['Tmm10'] = (
                ds.puv.compute_wave_params(var='vyp',**kwargs )
                )


            ds['Hm0'].attrs = {'units':'m','long_name':'Hm0'}
            ds['Tp'].attrs = {'units':'s','long_name':'Tp'}
            ds['Tm01'].attrs = {'units':'s','long_name':'Tm01'}
            ds['Tmm10'].attrs = {'units':'s','long_name':'T_{m-1,0}'}


            #%% rotate velocity in wave direction
            print('near bed orbital velocity computation')

            ds['fp'] = 1/ds.Tp
            ds['u_ss'] = ds.puv.band_pass_filter_ss(var='u')
            ds['v_ss'] = ds.puv.band_pass_filter_ss(var='v')

            #compute angle of main wave propagation through singular value decomposition
            ds['svdtheta'] = ds.puv.compute_SVD_angle()
            ds['svdtheta'].attrs = (
                {'units':'deg','long_name':'angle principle wave axis',
                 'comments':'cartesian convention'}
                )

            ds['puvdir'] = ds.puv.puv_wavedir(p='eta',u='u_ss',v='v_ss',**kwargs)
            ds['puvdir'].attrs = (
                {'units':'deg','long_name':'wave prop dir',
                 'comments':'cartesian convention'}
                )

            #rotate velocities so component in wave prop is in ud, rest in vd
            ds['ud'], ds['vd'] = (
                ds.puv.rotate_velocities(u='u_ss',v = 'v_ss',theta='puvdir')
                )

            #dirspread
            ds['svddspr'] = (
                np.arctan(
                    np.sqrt( (ds['vd']**2).mean(dim='N')
                             /(ds['ud']**2).mean(dim='N')))
                /np.pi*180
                )
            ds['svddspr'].attrs = {'units':'deg','long_name':'angle ud vd',
                                   'comments':'max 45deg'}

            # def custom_func(x,o):
            #     if np.logical_or((x+360-o)%360<=90,(x+360-o)%360>=270) :
            #         return x%360
            #     else:
            #         return (x+180)%360

            # ds['wavedir'] = xr.apply_ufunc(custom_func,
            #                        ds['svdtheta'],135*ds['Hm0']/ds['Hm0'],
            #                        input_core_dims=[[],[]],
            #                        output_core_dims=[[]],
            #                        vectorize=True)


            ds['ang_cw_svd'] = (ds.uang - ds.svdtheta)%360
            ds['ang_cw_puv'] = (ds.uang - ds.puvdir)%360

            ds['ud_mean'] = ds.umag*np.cos(ds.ang_cw_puv*np.pi/180)


            #rms orbital velocity
            ds['u_ssm'] = np.sqrt(ds.ud**2 + ds.vd**2).mean(dim='N')
            ds['u_ssm'].attrs = (
                {'units':'m/s','long_name':'u_o',
                 'comments':'Root mean squared total (ud+vd) orbital velocity'}
                )

            ds['ud_ssm'] = np.sqrt(ds.u_ss**2).mean(dim='N')
            ds['ud_ssm'].attrs = {'units':'m/s','long_name':'u_o',
                                  'comments':'Root mean squared orbital velocity'}


            #%% velocity moments
            print('velocity moments')

            #rotate velocities so component in cross long prop is in ud, rest in vd
            ds['uc'], ds['ul'] = ds.puv.rotate_velocities('u','v',theta=135)
            ds['ucm'] = ds.uc.mean(dim='N')
            ds['ulm'] = ds.ul.mean(dim='N')
            ds['ucm'].attrs = {'units':'m/s','long_name':'u_cross mean',
                               'comments':'burst averaged'}
            ds['ulm'].attrs = {'units':'m/s','long_name':'v_long mean',
                               'comments':'burst averaged'}


            ds['ucm3'] = (ds.uc**3).mean(dim='N')
            ds['ulm3'] = (ds.ul**3).mean(dim='N')
            ds['ucm3'].attrs = {'units':'m3/s3','long_name':'u^3_cross mean',
                                'comments':'burst averaged'}
            ds['ulm3'].attrs = {'units':'m3/s3','long_name':'v^3_long mean',
                                'comments':'burst averaged'}

            ds['ucmud2'] = ((ds.u_ss**2+ds.v_ss**2)*ds.uc).mean(dim='N')
            ds['ulmud2'] = ((ds.u_ss**2+ds.v_ss**2)*ds.ul).mean(dim='N')
            ds['ucmud2'].attrs = {'units':'m3/s3',
                                  'long_name':'u*u_ss^2_cross mean',
                                  'comments':'burst averaged'}
            ds['ulmud2'].attrs = {'units':'m3/s3',
                                  'long_name':'l*u_ss^2_long mean',
                                  'comments':'burst averaged'}


            #%% wave shape in ud dir
            print('wave shape from ov')

            #rotate velocities so component in wave prop is in ud, rest in vd
            u,v = ds.puv.rotate_velocities(u = 'u', v = 'v',theta = 'svdtheta')
            ds['udm'] = u.mean(dim='N')
            ds['vdm'] = v.mean(dim='N')
            ds['udm'].attrs = {'units':'m/s','long_name':'u mean wavdir',
                               'comments':'burst averaged'}
            ds['vdm'].attrs = {'units':'m/s','long_name':'v_mean wavdir ',
                               'comments':'burst averaged'}

            #angle between waves and shore normal:
            ds['obl'] = ds['svdtheta'] - 135
            ds['obl'].attrs = {'units':'deg','long_name':'obliquity waves'}

            #on the original frequency range
            ds['Sk'],ds['As'],ds['sig'] = ds.puv.compute_SkAs('ud')
            ds['Sk'].attrs = {'units':'m3/s3','long_name':'skewness'}
            ds['As'].attrs = {'units':'m3/s3','long_name':'asymmetry'}
            ds['sig'].attrs = {'units':'m/s','long_name':'std(ud)'}

            #in a frequency range adapting to the peak period
            ds['Sk0'],ds['As0'],ds['sig0'] = (
                ds.puv.compute_SkAs('ud',fixedBounds = True, bounds = [0.05,1])
                )
            ds['Sk0'].attrs = {'units':'m3/s3','long_name':'skewness'}
            ds['As0'].attrs = {'units':'m3/s3','long_name':'asymmetry'}
            ds['sig0'].attrs = {'units':'m/s','long_name':'std(ud)'}


            #skewness on total cross shore and longshore signal
            ds['Skc'],ds['Asc'],ds['sigc'] = ds.puv.compute_SkAs('uc')
            ds['Skc'].attrs = {'units':'m3/s3','long_name':'skewness'}
            ds['Asc'].attrs = {'units':'m3/s3','long_name':'asymmetry'}
            ds['sigc'].attrs = {'units':'m/s','long_name':'std(ud)'}


            ds['Skl'],ds['Asl'],ds['sigl'] = ds.puv.compute_SkAs('ul')
            ds['Skl'].attrs = {'units':'m3/s3','long_name':'skewness'}
            ds['Asl'].attrs = {'units':'m3/s3','long_name':'asymmetry'}
            ds['sigl'].attrs = {'units':'m/s','long_name':'std(ud)'}


            #%% shape statistics computed from pressure
            print('wave shape from pressure')

            #in the original freq range
            ds['Skp0'],ds['Asp0'],ds['sigp0'] = (
                ds.puv.compute_SkAs('eta',fixedBounds = True, bounds = [0.05,1])
                )

            ds['Skp0'].attrs = {'units':'m3/s3','long_name':'skewness'}
            ds['Asp0'].attrs = {'units':'m3/s3','long_name':'asymmetry'}
            ds['sigp0'].attrs = {'units':'m/s','long_name':'std(ud)'}

            #in a band scaled with peak period
            ds['Skp'],ds['Asp'],ds['sigp'] = ds.puv.compute_SkAs('eta')

            ds['Skp'].attrs = {'units':'m3/s3','long_name':'skewness'}
            ds['Asp'].attrs = {'units':'m3/s3','long_name':'asymmetry'}
            ds['sigp'].attrs = {'units':'m/s','long_name':'std(ud)'}





            #%%
            dsList.append(ds.drop_dims('N'))
            # dsList.append(ds)
        # except:
        #     print(file + 'not working')
        #     continue

    sds = xr.merge(dsList)
    sds = sds.resample(t='10T').nearest(tolerance='5T')
    sds.attrs = dsList[0].attrs
    sds.attrs = {'comments':'timeseries of velocity characteristics'}


    #%% saving
    #specify compression for all the variables to reduce file size
    comp = dict(zlib=True, complevel=5)
    sds.encoding = {var: comp for var in sds.data_vars}


    # sds.to_netcdf((ncOutDir + '/' + instrumentName + '_all.nc'))
    sds.to_netcdf((ncOutDir + '/' + instrumentName + '.nc'))
