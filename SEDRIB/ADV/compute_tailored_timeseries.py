import glob
import os
import yaml
from pathlib import Path
import numpy as np
import xarray as xr
from datetime import datetime
import sys
sys.path.append(r'C:\checkouts\python\PhD\modules')
import xrMethodAccessors

config = yaml.safe_load(Path(r'C:\checkouts\python\PhD\field_visit_SEDRIB\sedrib23-processing.yml').read_text())

instrument = config['instruments']['adv']['vector'][0]

# find all files that need to be processed
fileNames = glob.glob(os.path.join(config['experimentFolder'], instrument, 'qc15min', '*.nc'))
print(instrument)

# prepare the save directory and place a copy of the script in this file
ncOutDir = os.path.join(config['experimentFolder'], instrument, 'tailored15min_uc2_v2')
if not os.path.isdir(ncOutDir):
    os.mkdir(ncOutDir)

dsList = []
for file in fileNames:

    with xr.open_dataset(file) as ds:
        # ds = ds.isel(t=range(3))
        
        assert len(ds.t) > 0, 'file is empty!'

        ##########################################################################
        # average flow conditions
        ##########################################################################
        ds['um'] = ds.u.mean(dim='N')
        ds['vm'] = ds.v.mean(dim='N')
        ds['uang'] = np.arctan2(ds.vm, ds.um) * 180 / np.pi
        ds['umag'] = np.sqrt(ds.um ** 2 + ds.vm ** 2)
        ds['um'].attrs = {'units': 'm/s ', 'long_name': 'u-east'}
        ds['vm'].attrs = {'units': 'm/s ', 'long_name': 'v-north'}
        ds['umag'].attrs = {'units': 'm/s ', 'long_name': 'flow velocity'}
        ds['uang'].attrs = {'units': 'deg ', 'long_name': 'flow direction',
                            'comment': 'cartesian convention'}

        ##########################################################################
        # extent the dataset with appropriate frequency axis
        ##########################################################################
        print('extending dataset with freq axis')

        fresolution = config['tailoredWaveSettings']['fresolution']['vector']

        ndiscretetheta = int(360/config['tailoredWaveSettings']['thetaresolution'])
        ds2 = xr.Dataset(
            data_vars={},
            coords={'t': ds.t,
                    'N': ds.N,
                    'f': np.arange(0, ds.sf.values/2+fresolution, fresolution),
                    'theta': np.arange(start=-np.pi, stop=np.pi, step=2 * np.pi / ndiscretetheta)}
        )

        ds2['f'].attrs = {'long_name': 'f', 'units': 'Hz'}
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
        kwargs = {'fmin': config['tailoredWaveSettings']['fmin'],
                    'fmax': config['tailoredWaveSettings']['fmax']}


        print('statistics from pressure')
        ds['p2'] = ds.p / config['physicalConstants']['rho']/config['physicalConstants']['g']
        _,vy   = ds.puv.spectrum_simple('p2', fresolution=fresolution)

        assert vy.shape[0]>0, 'no valid pressure data remains on this day'

        # compute the attenuation factor
        # attenuation corrected spectra
        Sw = ds.puv.attenuation_factor('pressure', elev='zip', d='d', **{'fcorrmax': 1, 'fcorrmaxBelieve': 1})
        ds['vyp'] = Sw * vy

        print('compute wave params from pressure')
        ds['Hm0'], ds['Tp'], ds['Tm01'], ds['Tm02'], ds['Tmm10'], ds['Tps'] = (
            ds.puv.compute_wave_params(var='vyp', **kwargs)
        )

        ds['Hm0'].attrs = {'units': 'm', 'long_name': 'Hm0'}

        ds['Tm01'].attrs = {'units': 's', 'long_name': 'Tm01'}
        ds['Tm02'].attrs = {'units': 's', 'long_name': 'Tm02'}
        ds['Tmm10'].attrs = {'units': 's', 'long_name': 'T_{m-1,0}'}
        ds['fp'] = 1 / ds.Tp

        ds['fp'].attrs = {'units': 'Hz', 'long_name': 'peak frequency'}
        ds['Tp'].attrs = {'units': 's', 'long_name': 'Tp'}

        ##########################################################################
        # wave direction and directional spread
        ##########################################################################
        print('near bed orbital velocity computation')

        # ssBounds = [config['tailoredWaveSettings']['fmin'], config['tailoredWaveSettings']['fmax_ss']]
        ds['u_ss'] = ds.puv.band_pass_filter_ss(var='u')
        ds['v_ss'] = ds.puv.band_pass_filter_ss(var='v')
        ds['u_ss'].attrs = {'units': 'm/s', 'long_name': 'u_ss',
                            'comments': 'seaswell velocity in x-direction between [0.5, 2]fp Hz'}
        ds['v_ss'].attrs = {'units': 'm/s', 'long_name': 'v_ss',
                            'comments': 'seaswell velocity in y-direction between [0.5, 2]fp Hz'}

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
                    np.sqrt((vdr ** 2).mean(dim='N')
                            / (udr ** 2).mean(dim='N')))
                / np.pi * 180
        )
        ds['svddspr'].attrs = {'units': 'deg', 'long_name': 'angle ud vd',
                                'comments': 'from principle component analysis as in Ruessink et al. 2012, max 45deg'}

        # MEM method
        ds['S'] = ds.puv.wave_MEMpuv(eta='p2', u='u', v='v', d='d', zi='zi', zb='zb')
        ds['S'].attrs = (
            {'units': 'm2/Hz/rad', 'long_name': '2D energy density',
                'comments': 'cartesian convention, from puv Method of maximum Entropy'}
        )

        _, _, _, _, _,_, ds['puvdir'], ds['dspr'] = ds.puv.compute_wave_params_from_S(var='S', **kwargs)
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
        ds['u_ssm'] = np.sqrt(ds.u_ss ** 2 + ds.v_ss ** 2).mean(dim='N')
        ds['u_ssm'].attrs = (
            {'units': 'm/s', 'long_name': 'u_o',
                'comments': 'Root mean squared total (u_ss+v_ss) orbital velocity'}
        )

        ##########################################################################
        # velocity mean and moments
        ##########################################################################
        print('velocity moments')

        # rotate velocities so component in cross long prop is in ud, rest in vd
        ds['uc'], ds['ul'] = ds.puv.rotate_velocities('u', 'v', theta=config['beachOrientation'][instrument])
        ds['ucm'] = ds.uc.mean(dim='N')
        ds['ulm'] = ds.ul.mean(dim='N')
        ds['ucm'].attrs = {'units': 'm/s', 'long_name': 'u_cross mean',
                            'comments': 'burst averaged velocity in cross-shore direction'}
        ds['ulm'].attrs = {'units': 'm/s', 'long_name': 'v_long mean',
                            'comments': 'burst-averaged velocity in alongshore direction'}

        ds['uc_ss'] = ds.puv.band_pass_filter_ss(var='uc')
        ds['ul_ss'] = ds.puv.band_pass_filter_ss(var='ul')
        ds['uc_ss'].attrs = {'units': 'm/s', 'long_name': 'uc_ss',
                            'comments': 'seaswell velocity in cross-shore direction between [0.5, 2]fp Hz'}
        ds['ul_ss'].attrs = {'units': 'm/s', 'long_name': 'ul_ss',
                            'comments': 'seaswell velocity in alongshore direction between [0.5, 2]fp Hz'}
        ##########################################################################
        # wave shape velocity based
        ##########################################################################
        print('wave shape from ov')

        # rotate velocities so component in wave prop is in ud, rest in vd
        ds['ud'], ds['vd'] = ds.puv.rotate_velocities(u='u', v='v', theta='puvdir')
        ds['ud'].attrs = {'units': 'm/s', 'long_name': 'u wavdir',
                            'comments': 'burst averaged'}
        ds['vd'].attrs = {'units': 'm/s', 'long_name': 'v wavdir ',
                            'comments': 'burst averaged'}
        
        ds['udm'] = ds.ud.mean(dim='N')
        ds['vdm'] = ds.vd.mean(dim='N')
        ds['udm'].attrs = {'units': 'm/s', 'long_name': 'u mean wavdir',
                            'comments': 'burst averaged'}
        ds['vdm'].attrs = {'units': 'm/s', 'long_name': 'v_mean wavdir ',
                            'comments': 'burst averaged'}

        # on the frequency range adapting to the peak period
        ds['Sk'], ds['As'], ds['sig'] = ds.puv.compute_SkAs('ud', fixedBounds=False)
        ds['Sk'].attrs = {'units': 'm3/s3', 'long_name': 'skewness',
                            'comment': 'vel-based between 0.5Tp and 2Tp'}
        ds['As'].attrs = {'units': 'm3/s3', 'long_name': 'asymmetry',
                            'comment': 'vel-based between 0.5Tp and 2Tp'}
        ds['sig'].attrs = {'units': 'm/s', 'long_name': 'std(ud)', 'comment': 'vel-based between 0.5Tp and 2Tp'}

        # # in the traditional freqband 0.05-1 Hz
        shapeBounds0 = [0.05, 1]
        ds['Sk0'], ds['As0'], ds['sig0'] = (
            ds.puv.compute_SkAs('ud', fixedBounds=True, bounds=shapeBounds0)
        )
        ds['Sk0'].attrs = {'units': 'm3/s3', 'long_name': 'skewness',
                            'comment': 'vel-based between {} and {} Hz'.format(shapeBounds0[0],
                                                                                    shapeBounds0[1])}
        ds['As0'].attrs = {'units': 'm3/s3', 'long_name': 'asymmetry',
                            'comment': 'vel-based between between {} and {} Hz'.format(shapeBounds0[0],
                                                                                    shapeBounds0[1])}
        ds['sig0'].attrs = {'units': 'm/s', 'long_name': 'std(ud)',
                            'comment': 'vel-based between between {} and {} Hz'.format(shapeBounds0[0],
                                                                                    shapeBounds0[1])}


        ##########################################################################
        # wave shape pressure based
        ##########################################################################
        print('wave shape from pressure')
        # in the original freq range
        ds['Skp0'], ds['Asp0'], ds['sigp0'] = (
            ds.puv.compute_SkAs('p', fixedBounds=True, bounds=shapeBounds0)
        )

        ds['Skp0'].attrs = {'units': 'm3', 'long_name': 'skewness',
                            'comment': 'pressure-based between {} and {} Hz'.format(shapeBounds0[0],
                                                                                    shapeBounds0[1])}
        ds['Asp0'].attrs = {'units': 'm3', 'long_name': 'asymmetry',
                            'comment': 'pressure-based between {} and {} Hz'.format(shapeBounds0[0],
                                                                                    shapeBounds0[1])}
        ds['sigp0'].attrs = {'units': 'm', 'long_name': 'std(ud)',
                                'comment': 'pressure-based between {} and {} Hz'.format(shapeBounds0[0],
                                                                                    shapeBounds0[1])}

        # in a band scaled with peak period
        ds['Skp'], ds['Asp'], ds['sigp'] = ds.puv.compute_SkAs('p', fixedBounds=False)

        ds['Skp'].attrs = {'units': 'm3', 'long_name': 'skewness',
                            'comment': 'pressure-based between 0.5Tp and 2Tp'}
        ds['Asp'].attrs = {'units': 'm3', 'long_name': 'asymmetry',
                            'comment': 'pressure-based between 0.5Tp and 2Tp'}
        ds['sigp'].attrs = {'units': 'm', 'long_name': 'std(ud)',
                            'comment': 'pressure-based between 0.5Tp and 2Tp'}

        ##########################################################################
        # wave numbers
        ##########################################################################

        ds['k'] = ds.puv.disper(T='Tmm10', d='d')
        ds['k'].attrs = {'units': 'm-1', 'long_name': 'k'}
        ds['Ur'] = ds.puv.Ursell(Hm0='Hm0', k='k', d='d')
        ds['Ur'].attrs = {'units': '-', 'long_name': 'Ursell'}

        ds['nAs'] = ds.As / ds.sig ** 3
        ds['nAs'].attrs = {'units': '-', 'long_name': 'near-bed orbital velocity asymmetry'}
        ds['nSk'] = ds.Sk / ds.sig ** 3
        ds['nSk'].attrs = {'units': '-', 'long_name': 'near-bed orbital velocity skewness'}

        ##########################################################################
        # TKE spectra
        ##########################################################################
        fx, ds['vyu'] = ds.puv.spectrum_simple('u', fresolution = ds.f.values[1])
        fx, ds['vyv'] = ds.puv.spectrum_simple('v', fresolution = ds.f.values[1])
        fx, ds['vyw'] = ds.puv.spectrum_simple('w', fresolution = ds.f.values[1])
        ds['vyu'].attrs = {'units': 'm2/s2/s', 'long_name': 'vardensity of x-velocity component'}
        ds['vyv'].attrs = {'units': 'm2/s2/s', 'long_name': 'vardensity of y-velocity component'}
        ds['vyw'].attrs = {'units': 'm2/s2/s', 'long_name': 'vardensity of z-velocity component'}

        # coherence
        f, ds['Cup'] = ds.puv.coherence('u','p')
        f, ds['Cvp'] = ds.puv.coherence('v','p')
        f, ds['Cwp'] = ds.puv.coherence('w','p')
        ds['Cup'].attrs = {'units': '-', 'long_name': 'orbital velocity coherence x-velocity component'}
        ds['Cvp'].attrs = {'units': '-', 'long_name': 'orbital velocity c y-velocity component'}
        ds['Cwp'].attrs = {'units': '-', 'long_name': 'orbital velocity c z-velocity component'}
        

        # remove the coherent surface wave variance
        ds['vyut'] = ds.vyu-ds.Cup*ds.vyu
        ds['vyvt'] = ds.vyv-ds.Cvp*ds.vyv
        ds['vywt'] = ds.vyw-ds.Cwp*ds.vyw
        ds['vyut'].attrs = {'units': 'm2/s2/s', 'long_name': 'incoherent vardensity of x-velocity component'}
        ds['vyvt'].attrs = {'units': 'm2/s2/s', 'long_name': 'incoherent velocity c y-velocity component'}
        ds['vywt'].attrs = {'units': 'm2/s2/s', 'long_name': 'incoherent velocity c z-velocity component'}
        
        # estimate rate of energy dissipation by viscosity: C = alpha*epsilon**(2/3)
        Cf = (ds.vywt/ds.f**(-5/3)).sel(f=slice(1,2))
        ds['TKED'] = Cf.mean(dim='f')
        ds['TKED'].attrs = {'units': '-', 'long_name': 'viscuous dissipation rate'}

        ##########################################################################
        # velocity components that are of interest
        ##########################################################################
        ds['uc2'] = ds.puv.band_pass_filter_ss('uc', freqband=[0, 16])
        ds['uc2'].attrs = {'units': 'm/s', 'long_name':'uc2', 'comment': 'bandpassfiltered to [0, 16]Hz'}
        ds['ul2'] = ds.puv.band_pass_filter_ss('ul', freqband=[0, 16])
        ds['ul2'].attrs = {'units': 'm/s', 'long_name':'ul2', 'comment': 'bandpassfiltered to [0, 16]Hz'}

        ds['uc2_ss'] = ds.puv.band_pass_filter_ss('uc',fpminfac=0.5, fpmaxfac=1000)
        ds['uc2_ss'].attrs = {'units': 'm/s', 'long_name':'uc2_ss', 'comment': 'bandpassfiltered to [0.5, 1000]f_p Hz'}
        ds['uc2_ig'] = ds['uc']-ds['uc2_ss']-ds['uc'].mean(dim='N')
        ds['uc2_ig'].attrs = {'units': 'm/s', 'long_name':'uc2_ig', 'comment': 'bandpassfiltered to [0, 0.5] f_p Hz'}
        ds['ul2_ss'] = ds.puv.band_pass_filter_ss('ul',fpminfac=0.5, fpmaxfac=1000)
        ds['ul2_ss'].attrs = {'units': 'm/s', 'long_name':'ucl_ss', 'comment': 'bandpassfiltered to [0.5, 1000]f_p Hz'}
        ds['ul2_ig'] = ds['ul']-ds['ul2_ss']-ds['ul'].mean(dim='N')
        ds['ul2_ig'].attrs = {'units': 'm/s', 'long_name':'ul2_ig', 'comment': 'bandpassfiltered to [0, 0.5]f_p Hz'}

        ds['uc3'] = ds.puv.band_pass_filter_ss('uc', freqband=[0.05, 1.5])
        ds['uc3'].attrs = {'units': 'm/s', 'long_name':'uc3', 'comment': 'bandpassfiltered to [0.5, 1.5] Hz'}
        ds['ul3'] = ds.puv.band_pass_filter_ss('ul', freqband=[0.05, 1.5])
        ds['ul3'].attrs = {'units': 'm/s', 'long_name':'ul3', 'comment': 'bandpassfiltered to [0.5, 1.5] Hz'}
        ds['uc3_ss'] = ds.puv.band_pass_filter_ss('uc3',fpminfac=0.5, fpmaxfac=100)
        ds['uc3_ss'].attrs = {'units': 'm/s', 'long_name':'uc3_ss', 'comment': 'bandpassfiltered to [0.5f_p, 1.5] Hz'}
        ds['uc3_ig'] = ds['uc3']-ds['uc3_ss']
        ds['uc3_ig'].attrs = {'units': 'm/s', 'long_name':'uc3_ig', 'comment': 'uc3-uc3_ss'}
        ds['ul3_ss'] = ds.puv.band_pass_filter_ss('ul3',fpminfac=0.5, fpmaxfac=100)
        ds['ul3_ss'].attrs = {'units': 'm/s', 'long_name':'ul3_ss', 'comment': 'bandpassfiltered to [0.5f_p, 1.5] Hz'}
        ds['ul3_ig'] = ds['ul3']-ds['ul3_ss']
        ds['ul3_ig'].attrs = {'units': 'm/s', 'long_name':'ul3_ig', 'comment': 'ul3-ul3_ss'}



        # specify compression for all the variables to reduce file size
        ds.attrs['construction datetime'] = datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")
        comp = dict(zlib=True, complevel=5)
        ds.encoding = {var: comp for var in ds.data_vars}
        # exclude coordinates from encoding
        for coord in list(ds.coords.keys()):
            ds.encoding[coord] = {'zlib': False, '_FillValue': None}

        ds.to_netcdf((os.path.join(ncOutDir, file.split('\\')[-1])))








