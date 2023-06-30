# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:28:24 2021

@author: marliesvanderl
"""

import xarray as xr
import numpy as np
import glob
import os
from datetime import datetime
import puv
import xrMethodAccessors

experimentFolder = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20210908_campaign\instruments'

instruments = [
         'L1C1VEC',
         'L2C2VEC',
         'L2C3VEC',
         'L2C4VEC',
         'L2C5SONTEK1',
         'L2C5SONTEK2',
         'L2C5SONTEK3',
         'L2C10VEC',
         'L3C1VEC',
         'L4C1VEC',
         'L5C1VEC',
         'L6C1VEC',
]

beachOrientation = [
     165, 135, 135, 135, 135, 135, 135, 135, 122, 122, 135, 142
]

for instrumentName, beachOri in zip(instruments, beachOrientation):

    # find all files that need to be processed
    fileNames = glob.glob(os.path.join(experimentFolder, instrumentName, 'qc', '*.nc'))
    print(instrumentName)

    # prepare the save directory and place a copy of the script in this file
    ncOutDir = os.path.join(experimentFolder, instrumentName, 'tailored')
    if not os.path.isdir(ncOutDir):
        os.mkdir(ncOutDir)

    dsList = []
    for file in fileNames:
       # try:
        with xr.open_dataset(file) as ds:
            if len(ds.t) == 0:
                print(file + ' is empty!')
                continue

            print(file)

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
            powerStep = 9  # 9 for investigating the spectra 9 = 0.03Hz resolution
            fresolution = np.round(sf*np.exp(-np.log(2)*powerStep), 5)
            ndiscretetheta = 64
            ds2 = xr.Dataset(
                data_vars={},
                coords={'t': ds.t,
                         'N': ds.N,
                         'f': np.arange(0, ds.sf.values/2, fresolution),
                          'theta':np.arange(start=-np.pi,stop=np.pi,step=2*np.pi/ndiscretetheta)}
                )

            ds2['f'].attrs = {'long_name': 'f','units': 'Hz'}
            ds2['N'].attrs = {'long_name': 'burst time', 'units': 's'}
            ds2['theta'].attrs = {'long_name': 'direction', 'units': 'rad'}

            # copy all data over into this new structure
            for key in ds.data_vars:
                ds2[key] = ds[key]
            ds2.attrs = ds.attrs
            ds = ds2

            ##########################################################################
            # statistics computed from pressure
            ##########################################################################
            if not 'SONTEK' in instrumentName:
                print('statistics from pressure')
                _, vy = ds.puv.spectrum_simple('eta', fresolution=fresolution)

                if vy.shape[0] == 0:
                    print('no valid pressure data remains on this day')
                    continue

                # compute the attenuation factor
                # attenuation corrected spectra
                Sw = ds.puv.attenuation_factor('pressure', elev='elevp', d='d')
                ds['vyp'] = Sw*vy

                kwargs = {'fmin': 0.05, 'fmax': 1.5}
                ds['Hm0'], ds['Tp'], ds['Tm01'], ds['Tm02'], ds['Tmm10'], ds['Tps'] = (
                    ds.puv.compute_wave_params(var='vyp', **kwargs )
                    )

                ds['Hm0'].attrs = {'units': 'm', 'long_name': 'Hm0'}

                ds['Tm01'].attrs = {'units': 's', 'long_name': 'Tm01'}
                ds['Tm02'].attrs = {'units': 's', 'long_name': 'Tm02'}
                ds['Tmm10'].attrs = {'units': 's', 'long_name': 'T_{m-1,0}'}
                ds['fp'] = 1 / ds.Tp
            else:
                ds['fp'] = ds.puv.get_peak_frequency('u', fpmin=0.05)
                ds['Tp'] = 1/ds.fp

            ds['fp'].attrs = {'units': 'Hz', 'long_name': 'peak frequency'}
            ds['Tp'].attrs = {'units': 's', 'long_name': 'Tp'}
            ##########################################################################
            # wave direction and directional spread
            ##########################################################################
            print('near bed orbital velocity computation')

            ds['u_ss'] = ds.puv.band_pass_filter_ss(var='u', freqband=[0.05, 8])
            ds['v_ss'] = ds.puv.band_pass_filter_ss(var='v', freqband=[0.05, 8])

            # compute angle of main wave propagation through singular value decomposition
            ds['svdtheta'] = ds.puv.compute_SVD_angle()
            ds['svdtheta'].attrs = (
                {'units': 'deg', 'long_name': 'angle principle wave axis',
                 'comments': 'cartesian convention'}
                )

            # rotate velocities so component in wave prop is in ud, rest in vd
            udr, vdr = (
                ds.puv.rotate_velocities(u='u_ss', v='v_ss', theta='svdtheta')
                )

            # dirspread in Ruessink's approach
            ds['svddspr'] = (
                np.arctan(
                    np.sqrt( (vdr**2).mean(dim='N')
                             /(udr**2).mean(dim='N')))
                /np.pi*180
                )
            ds['svddspr'].attrs = {'units': 'deg', 'long_name': 'angle ud vd',
                                   'comments': 'from principle component analysis as in Ruessink et al. 2012, max 45deg'}

            # MEM method
            ds['S'] = ds.puv.wave_MEMpuv(eta='eta',u='u', v='v', d='d', zi='zi', zb='zb')
            ds['S'].attrs = (
            {'units': 'm2/Hz/rad', 'long_name': '2D energy density',
             'comments': 'cartesian convention, from puv Method of maximum Entropy'}
            )

            _, _, _, _, _,_, ds['puvdir'], ds['dspr'] = ds.puv.compute_wave_params_from_S(var='S')
            ds['puvdir'].attrs = (
                {'units': 'deg', 'long_name': 'wave prop dir',
                 'comments': 'cartesian convention, from puv Method of maximum Entropy'}
            )
            ds['dspr'].attrs = (
                {'units': 'deg', 'long_name': 'directional spread',
                 'comments': 'single-sided directional spread based on kuijk (1988) , spectra from puv Method of maximum Entropy'}
            )

            # rotate velocities so component in wave prop is in ud, rest in vd
            ds['ud'], ds['vd'] = (
                ds.puv.rotate_velocities(u='u_ss', v='v_ss', theta='puvdir')
                )
            ds['ud'].attrs = (
                {'units': 'm/s', 'long_name': 'u_d',
                 'comments': 'orbital velocity in mean wave direction'}
                )
            ds['vd'].attrs = (
                {'units': 'm/s', 'long_name': 'v_d',
                 'comments': 'orbital velocity perpendicular to mean wave direction'}
            )


            ds['ud_ssm'] = np.sqrt(ds.ud ** 2).mean(dim='N')
            ds['ud_ssm'].attrs = {'units': 'm/s', 'long_name': 'u_o',
                                  'comments': 'Root mean squared orbital velocity in wave propagation direction'}

            # rms orbital velocity
            ds['u_ssm'] = np.sqrt(ds.u_ss**2 + ds.v_ss**2).mean(dim='N')
            ds['u_ssm'].attrs = (
                {'units': 'm/s', 'long_name': 'u_o',
                 'comments': 'Root mean squared total (u_ss+v_ss) orbital velocity'}
                )



            ##########################################################################
            # velocity mean and moments
            ##########################################################################
            print('velocity moments')

            # rotate velocities so component in cross long prop is in ud, rest in vd
            ds['uc'], ds['ul'] = ds.puv.rotate_velocities('u', 'v', theta=beachOri)
            ds['ucm'] = ds.uc.mean(dim='N')
            ds['ulm'] = ds.ul.mean(dim='N')
            ds['ucm'].attrs = {'units': 'm/s', 'long_name': 'u_cross mean',
                               'comments': 'burst averaged velocity in cross-shore direction'}
            ds['ulm'].attrs = {'units': 'm/s', 'long_name': 'v_long mean',
                               'comments': 'burst-averaged velocity in alongshore direction'}


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

            # on the frequency range adapting to the peak period
            ds['Sk'], ds['As'], ds['sig'] = ds.puv.compute_SkAs('ud')
            ds['Sk'].attrs = {'units': 'm3/s3', 'long_name': 'skewness',
                              'comment': 'vel-based between 0.5Tp and 2Tp'}
            ds['As'].attrs = {'units': 'm3/s3', 'long_name': 'asymmetry',
                              'comment': 'vel-based between 0.5Tp and 2Tp'}
            ds['sig'].attrs = {'units': 'm/s', 'long_name': 'std(ud)', 'comment': 'vel-based between 0.5Tp and 2Tp'}

            # in the traditional freqband 0.05-1 Hz
            ds['Sk0'], ds['As0'], ds['sig0'] = (
                ds.puv.compute_SkAs('ud', fixedBounds=True, bounds=[0.05, 1])
            )
            ds['Sk0'].attrs = {'units': 'm3/s3', 'long_name': 'skewness',
                               'comment': 'vel-based between 0.05 and 1 Hz'}
            ds['As0'].attrs = {'units': 'm3/s3', 'long_name': 'asymmetry',
                               'comment': 'vel-based between 0.05 and 1 Hz'}
            ds['sig0'].attrs = {'units': 'm/s', 'long_name': 'std(ud)',
                                'comment': 'vel-based between 0.05 and 1 Hz'}

            # angle between waves and shore normal:
            ds['beachOri'] = beachOri
            ds['beachOri'].attrs = {'units': 'deg', 'long_name': 'orientation of shoreward',
                                    'comment': 'Cartesian convention'}

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

            ##########################################################################
            # wave numbers
            ##########################################################################
            ds['zs'] = ds.eta.mean(dim='N')
            ds['zs'].attrs = {'units': 'm+NAP', 'long_name': 'water level',
                               'comment': 'burst averaged'}

            if not 'SONTEK' in instrumentName:
                ds['k'] = xr.apply_ufunc(lambda tm01, h: puv.disper(2 * np.pi / tm01, h), ds['Tmm10'], ds['zs'] - ds['zb'])
                ds['k'].attrs = {'units': 'm-1', 'long_name': 'k'}
                ds['Ur'] = xr.apply_ufunc(lambda hm0, k, h: 3 / 4 * 0.5 * hm0 * k / (k * h) ** 3, ds['Hm0'], ds['k'],
                                          ds['zs'] - ds['zb'])
                ds['Ur'].attrs = {'units': '-', 'long_name': 'Ursell'}

            ds['nAs'] = ds.As / ds.sig ** 3
            ds['nAs'].attrs = {'units': '-', 'long_name': 'near-bed orbital velocity asymmetry'}
            ds['nSk'] = ds.Sk / ds.sig ** 3
            ds['nSk'].attrs = {'units': '-', 'long_name': 'near-bed orbital velocity skewness'}


            # specify compression for all the variables to reduce file size
            ds.attrs['construction datetime']  = datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")
            comp = dict(zlib=True, complevel=5)
            ds.encoding = {var: comp for var in ds.data_vars}

            ds.to_netcdf((os.path.join(ncOutDir, file.split('\\')[-1])))
      #  except:
      #      print('error with: ' + file)
      #      continue


