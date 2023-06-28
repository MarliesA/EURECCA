# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:28:24 2021

@author: marliesvanderl
"""

import xarray as xr
import numpy as np
import glob
import os

experimentFolder = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20210908_campaign\instruments'

instruments = [
        'L1C1VEC',
        'L2C2VEC',
        'L2C4VEC',
        'L2C3VEC',
        'L2C10VEC',
        'L3C1VEC',
        'L4C1VEC',
        'L5C1VEC',
        'L6C1VEC',
]

beachOrientation = [165, 135, 135, 135, 135, 122, 122, 135, 142]

for instrumentName, beachOri in zip(instruments, beachOrientation):

    # find all files that need to be processed
    fileNames = glob.glob(os.path.join(experimentFolder, instrumentName, 'QC_loose', '*.nc'))
    print(instrumentName)

    # prepare the save directory and place a copy of the script in this file
    ncOutDir = os.path.join(experimentFolder, instrumentName, 'tailored_loose')
    if not os.path.isdir(ncOutDir):
        os.mkdir(ncOutDir)

    # make a txtfile copy of the executed script
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


            # we reshape the dataset to blocks of 10 minutes. Because extra coordinates are lost in this operation
            # we set the coord maskv to a variable mv and later restore to coord
            ds['mv'] = (('t', 'N'), ds.coords['maskv'].values)
            ds2 = ds.burst.reshape_burst_length(600)
            ds2.coords['maskv'] = ds2.mv

            ######################################################################
            # fill masked observations with interpolation
            ######################################################################

            # drop bursts with too many nans (>5%) in the velocity data
            ds3 = ds2.where(
                np.isnan(ds2.where(ds2.maskv).u).sum(dim='N')
                < 0.05*len(ds2.N)).dropna(dim='t', how='all')

            # make sure the vars keep their original dimensions
            ds3 = ds2.sel(t=ds3.t)

            # interpolate nans
            N = len(ds.N)
            for var in ['u', 'v', 'eta']:
                # interpolate the bursts where there is less than 5% nans
                data = ds3[var].where(
                    np.isnan(ds3[var]).sum(dim='N') < 0.05*len(ds3.N)
                    ).dropna(dim='t', how='all')
                if len(data.t) != 0:
                    ds3[var] = data.interpolate_na(
                        dim='N',
                        method='cubic',
                        max_gap=8)

                # and fill the gaps more than 8 in length with the burst average
                ds3[var] = ds3[var].fillna(ds3[var].mean(dim='N'))

            ds = ds3

            # if there is completely no data, continue
            if np.sum(~np.isnan(ds.eta)) == 0:
                print('stuf goes even more wrong')
                continue

            ##########################################################################
            # water level is taken from ossi C9
            ##########################################################################
            #eta = xr.open_dataset(experimentFolder + r'\L2C9OSSI\tailored\L2C9OSSI.nc')

            #if len(ds.t) > 1:
            #    zs = eta.zs.sel(t=slice(ds.t.min().dt.floor('1H'), ds.t.max().dt.ceil('1H')))
            #    ds['zs'] = zs.interp_like(ds.t)
            #    ds = ds.dropna(dim='t', subset=['zs'])  # no data when water level is undefined
            #elif len(ds.t) == 1:
            #    ds['zs'] = eta.zs.sel(t=ds.t, method='nearest')
            #else:
            #    print('dataset length = 0')
            #    continue
            #ds['zs'].attrs = {'units': 'm+NAP', 'long_name': 'water level'}

            #ds = ds.dropna(dim='t', subset=['zs'])  # no data when water level is undefined

            #ds['d'] = ds.zs-ds.zb
            #ds['d'].attrs = {'units': 'm ', 'long_name': 'water depth'}
            #ds['elev'] = ds.zi-ds.zb
            #ds['elev'].attrs = {'units': 'm ', 'long_name': 'height probe above bed'}
            #ds['elevp'] = ds.zip-ds.zb
            #ds['elevp'].attrs = {'units': 'm ', 'long_name': 'height ptd above bed'}

            ##########################################################################
            # average flow conditions
            ##########################################################################
            ds['um'] = ds.u.mean(dim='N')
            ds['vm'] = ds.v.mean(dim='N')
            ds['uang'] = np.arctan2(ds.vm, ds.um)*180/np.pi
            ds['umag'] = np.sqrt(ds.um**2+ds.vm**2)
            ds['um'].attrs = {'units': 'm/s ', 'long_name': 'u-east'}
            ds['vm'].attrs = {'units': 'm/s ', 'long_name': 'v-north'}
            ds['umag'].attrs = {'units': 'm/s ', 'long_name': 'flow velocity'}
            ds['uang'].attrs = {'units': 'deg ', 'long_name': 'flow direction',
                                'comment': 'cartesian convention'}

            ##########################################################################
            # extent the dataset with appropriate frequency axis
            ##########################################################################
            print('extending dataset with freq axis')
            sf = ds.sf.values
            powerStep = 7  # 9 for investigating the spectra 9 = 0.03Hz resolution
            fresolution = np.round(sf*np.exp(-np.log(2)*powerStep), 5)
            ds2 = xr.Dataset(
                data_vars={},
                coords={'t': ds.t,
                         'N': ds.N,
                         'f': np.arange(0, ds.sf.values/2, fresolution)}
                )


            ds2['f'].attrs = {'long_name': 'f','units': 'Hz'}
            ds2['N'].attrs = {'long_name': 'burst time', 'units': 's'}
            for key in ds.data_vars:
                ds2[key] = ds[key]
            ds2.attrs = ds.attrs
            ds = ds2

            ##########################################################################
            # statistics computed from pressure
            ##########################################################################
            if not 'SONTEK' in instrumentName:

                print('statistics from pressure')
                _,vy   = ds.puv.spectrum_simple('eta', fresolution=fresolution)

                # compute the attenuation factor
                # attenuation corrected spectra
                Sw = ds.puv.attenuation_factor('pressure', elev='elevp', d='d')
                ds['vyp'] = Sw*vy

                kwargs = {'fmin': 0.1, 'fmax': 1.5}
                ds['Hm0'], ds['Tp'], ds['Tm01'], ds['Tm02'], ds['Tmm10'] = (
                    ds.puv.compute_wave_params(var='vyp', **kwargs )
                    )

                ds['Hm0'].attrs = {'units': 'm', 'long_name': 'Hm0'}
                ds['Tp'].attrs = {'units': 's', 'long_name': 'Tp'}
                ds['Tm01'].attrs = {'units': 's', 'long_name': 'Tm01'}
                ds['Tm02'].attrs = {'units': 's', 'long_name': 'Tm02'}
                ds['Tmm10'].attrs = {'units': 's', 'long_name': 'T_{m-1,0}'}

                ds['fp'] = 1 / ds.Tp
                ds['fp'].attrs = {'units': 'Hz', 'long_name': 'f_p}'}

            else:
                ds['fp'] = ds.puv.get_fp('u', fpmin=0.05)
                ds['fp'].attrs = {'units': 'Hz', 'long_name': 'f_p}'}


            ##########################################################################
            # wave direction and directional spread
            ##########################################################################
            print('near bed orbital velocity computation')

            ds['u_ss'] = ds.puv.band_pass_filter_ss(var='u', freqband = [0.05, 1])
            ds['v_ss'] = ds.puv.band_pass_filter_ss(var='v', freqband = [0.05, 1])
            ds['u_ss'].attrs = (
                {'units': 'm/s', 'long_name': 'u_{o,E}',
                 'comments': 'orbital velocity in freqband [0.05, 1] Hz, East-component'}
            )
            ds['v_ss'].attrs = (
                {'units': 'm/s', 'long_name': 'u_{o,N}',
                 'comments': 'orbital velocity in freqband [0.05, 1] Hz, North-component'}
            )

            # compute angle of main wave propagation through singular value decomposition
            ds['svdtheta'] = ds.puv.compute_SVD_angle()
            ds['svdtheta'].attrs = (
                {'units': 'deg', 'long_name': 'angle principle wave axis',
                 'comments': 'cartesian convention'}
                )

            ds['puvdir'] = ds.puv.puv_wavedir(p='eta', u='u_ss', v='v_ss', **kwargs)
            ds['puvdir'].attrs = (
                {'units': 'deg', 'long_name': 'wave direction',
                 'comments': 'cartesian convention, from puv Method of maximum Entropy'}
                )

            Hm0, Tp, Tm01, Tm02, Tmm10, theta0, dspr

            # rotate velocities so component in wave prop is in ud, rest in vd
            ds['ud'], ds['vd'] = (
                ds.puv.rotate_velocities(u='u_ss', v='v_ss', theta='puvdir')
                )
            ds['u_d'].attrs = (
                {'units': 'm/s', 'long_name': 'u_{o,0}',
                 'comments': 'orbital velocity in freqband [0.05, 1] Hz, component in wave direction puvdir'}
            )
            ds['v_d'].attrs = (
                {'units': 'm/s', 'long_name': 'u_{o,1}',
                 'comments': 'orbital velocity in freqband [0.05, 1] Hz, component perpendicular to wave direction puvdir'}
            )

            # rotate velocities so component in wave prop SVD
            udr, vdr = (
                ds.puv.rotate_velocities(u='u_ss', v='v_ss', theta='svdtheta')
                )
            # dirspread
            ds['svddspr'] = (
                np.arctan(
                    np.sqrt( (vdr**2).mean(dim='N')
                             /(udr**2).mean(dim='N')))
                /np.pi*180
                )
            ds['svddspr'].attrs = {'units': 'deg', 'long_name': 'wave spreading',
                                   'comments': 'wave spreading based on Singular Value Decomposition, as in Ruessink et al. 2012, max 45deg'}


            # rms orbital velocity
            ds['u_ssm'] = np.sqrt(ds.u_ss**2 + ds.v_ss**2).mean(dim='N')
            ds['u_ssm'].attrs = (
                {'units': 'm/s', 'long_name': 'u_o',
                 'comments': 'Root mean squared total (u_ss+v_ss) orbital velocity'}
                )

            ds['ud_ssm'] = np.sqrt(ds.ud**2).mean(dim='N')
            ds['ud_ssm'].attrs = {'units': 'm/s', 'long_name': 'u_o',
                                  'comments': 'Root mean squared orbital velocity in wave propagation direction'}

            ##########################################################################
            # velocity mean and moments
            ##########################################################################
            print('velocity moments')

            # rotate velocities so component in cross long prop is in ud, rest in vd
            ds['uc'], ds['ul'] = ds.puv.rotate_velocities('u', 'v', theta=beachOri)
            ds['ucm'] = ds.uc.mean(dim='N')
            ds['ulm'] = ds.ul.mean(dim='N')
            ds['ucm'].attrs = {'units': 'm/s', 'long_name': 'u_cross mean',
                               'comments': 'burst averaged'}
            ds['ulm'].attrs = {'units': 'm/s', 'long_name': 'v_long mean',
                               'comments': 'burst averaged'}


            ds['ucm3'] = (ds.uc**3).mean(dim='N')
            ds['ulm3'] = (ds.ul**3).mean(dim='N')
            ds['ucm3'].attrs = {'units': 'm3/s3', 'long_name': 'u^3_cross mean',
                                'comments': 'burst averaged'}
            ds['ulm3'].attrs = {'units': 'm3/s3', 'long_name': 'v^3_long mean',
                                'comments': 'burst averaged'}

            ds['ucmud2'] = ((ds.u_ss**2+ds.v_ss**2)*ds.uc).mean(dim='N')
            ds['ulmud2'] = ((ds.u_ss**2+ds.v_ss**2)*ds.ul).mean(dim='N')
            ds['ucmud2'].attrs = {'units': 'm3/s3',
                                  'long_name': 'u*u_ss^2_cross mean',
                                  'comments': 'burst averaged'}
            ds['ulmud2'].attrs = {'units': 'm3/s3',
                                  'long_name': 'l*u_ss^2_long mean',
                                  'comments': 'burst averaged'}

            ##########################################################################
            # wave shape velocity based
            ##########################################################################
            print('wave shape from ov')

            # rotate velocities so component in wave prop is in ud, rest in vd
            u, v = ds.puv.rotate_velocities(u='u', v='v', theta='puvdir')
            ds['udm'] = u.mean(dim='N')
            ds['vdm'] = v.mean(dim='N')
            ds['udm'].attrs = {'units': 'm/s', 'long_name': 'u mean wavdir',
                               'comments': 'burst averaged'}
            ds['vdm'].attrs = {'units': 'm/s', 'long_name': 'v_mean wavdir ',
                               'comments': 'burst averaged'}

            # angle between waves and shore normal:
            ds['beachOri'] = beachOri
            ds['beachOri'].attrs = {'units': 'deg', 'long_name': 'orientation of shoreward',
                                    'comment': 'Cartesian convention'}

            # on the frequency range adapting to the peak period
            ds['Sk'], ds['As'], ds['sig'] = ds.puv.compute_SkAs('ud')
            ds['Sk'].attrs = {'units': 'm3/s3', 'long_name': 'skewness', 'comment': 'vel-based between 0.5Tp and 2Tp'}
            ds['As'].attrs = {'units': 'm3/s3', 'long_name': 'asymmetry', 'comment': 'vel-based between 0.5Tp and 2Tp'}
            ds['sig'].attrs = {'units': 'm/s', 'long_name': 'std(ud)', 'comment': 'vel-based between 0.5Tp and 2Tp'}

            # in the traditional freqband 0.05-1 Hz
            ds['Sk0'], ds['As0'], ds['sig0'] = (
                ds.puv.compute_SkAs('ud', fixedBounds=True, bounds=[0.05, 1])
                )
            ds['Sk0'].attrs = {'units': 'm3/s3', 'long_name': 'skewness', 'comment': 'vel-based between 0.05 and 1 Hz'}
            ds['As0'].attrs = {'units': 'm3/s3', 'long_name': 'asymmetry', 'comment': 'vel-based between 0.05 and 1 Hz'}
            ds['sig0'].attrs = {'units': 'm/s', 'long_name': 'std(ud)', 'comment': 'vel-based between 0.05 and 1 Hz'}

            # skewness on total cross shore and longshore signal
            ds['Skc'], ds['Asc'], ds['sigc'] = ds.puv.compute_SkAs('uc')
            ds['Skc'].attrs = {'units': 'm3/s3', 'long_name': 'skewness', 'comment': 'cross shore vel-based between 0.5Tp and 2Tp'}
            ds['Asc'].attrs = {'units': 'm3/s3', 'long_name': 'asymmetry', 'comment': 'cross shore vel-based between 0.5Tp and 2Tp'}
            ds['sigc'].attrs = {'units': 'm/s', 'long_name': 'std(ud)', 'comment': 'cross shore vel-based between 0.5Tp and 2Tp'}

            ds['Skl'], ds['Asl'], ds['sigl'] = ds.puv.compute_SkAs('ul')
            ds['Skl'].attrs = {'units': 'm3/s3', 'long_name': 'skewness', 'comment': 'along shore vel-based between 0.5Tp and 2Tp'}
            ds['Asl'].attrs = {'units': 'm3/s3', 'long_name': 'asymmetry', 'comment': 'along shore vel-based between 0.5Tp and 2Tp'}
            ds['sigl'].attrs = {'units': 'm/s', 'long_name': 'std(ud)', 'comment': 'along shore vel-based between 0.5Tp and 2Tp'}

            ##########################################################################
            # wave shape pressure based
            ##########################################################################
            print('wave shape from pressure')

            # in the original freq range
            ds['Skp0'], ds['Asp0'], ds['sigp0'] = (
                ds.puv.compute_SkAs('eta', fixedBounds=True, bounds=[0.05, 1])
                )

            ds['Skp0'].attrs = {'units': 'm3/s3', 'long_name': 'skewness', 'comment': 'pressure-based between 0.05 and 1 Hz'}
            ds['Asp0'].attrs = {'units': 'm3/s3', 'long_name': 'asymmetry', 'comment': 'pressure-based between 0.05 and 1 Hz'}
            ds['sigp0'].attrs = {'units': 'm/s', 'long_name': 'std(ud)', 'comment': 'pressure-based between 0.05 and 1 Hz'}

            # in a band scaled with peak period
            ds['Skp'], ds['Asp'], ds['sigp'] = ds.puv.compute_SkAs('eta')

            ds['Skp'].attrs = {'units': 'm3/s3', 'long_name': 'skewness', 'comment': 'pressure-based between 0.5Tp and 2Tp'}
            ds['Asp'].attrs = {'units': 'm3/s3', 'long_name': 'asymmetry', 'comment': 'pressure-based between 0.5Tp and 2Tp'}
            ds['sigp'].attrs = {'units': 'm/s', 'long_name': 'std(ud)', 'comment': 'pressure-based between 0.5Tp and 2Tp'}

            dsList.append(ds.drop_dims('N'))
        # except:
        #     print(file + 'not working')
        #     continue

    sds = xr.merge(dsList)
    sds = sds.resample(t='10T').nearest(tolerance='5T')
    sds.attrs = dsList[0].attrs
    sds.attrs = {'comments': '10 minute averaged wave and velocity characteristics'}

    # specify compression for all the variables to reduce file size
    comp = dict(zlib=True, complevel=5)
    sds.encoding = {var: comp for var in sds.data_vars}

    sds.to_netcdf((os.path.join(ncOutDir, instrumentName + '.nc')))
