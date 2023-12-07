import glob
import os
import yaml
from pathlib import Path
import numpy as np
import xarray as xr
from datetime import datetime
from sedmex_info_loaders import get_githash
import xrMethodAccessors
from encoding_sedmex import encoding_sedmex

def compute_waves(instrument, config):

    # find all files that need to be processed
    fileNames = glob.glob(os.path.join(config['experimentFolder'], instrument, 'qc', '*.nc'))
    print(instrument)

    # prepare the save directory and place a copy of the script in this file
    ncOutDir = os.path.join(config['experimentFolder'], instrument, 'tailored')
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
            ds['uang'] = np.arctan2(ds.vm, ds.um) * 180 / np.pi
            ds['umag'] = np.sqrt(ds.um ** 2 + ds.vm ** 2)
            ds['um'].attrs = {'units': 'm/s ', 'long_name': 'velocity east-component', 'comment': 'block-averaged'}
            ds['vm'].attrs = {'units': 'm/s ', 'long_name': 'velocity north-component', 'comment': 'block-averaged'}
            ds['umag'].attrs = {'units': 'm/s ', 'long_name': 'flow velocity', 'comment': 'block-averaged'}
            ds['uang'].attrs = {'units': 'deg ', 'long_name': 'flow direction', 'comment': 'block-averaged and in cartesian convention'}

            ##########################################################################
            # extent the dataset with appropriate frequency axis
            ##########################################################################
            print('extending dataset with freq axis')
            if 'SONTEK' in instrument:
                fresolution = config['tailoredWaveSettings']['fresolution']['sontek']
            elif 'VEC' in instrument:
                fresolution = config['tailoredWaveSettings']['fresolution']['vector']

            ndiscretetheta = int(360/config['tailoredWaveSettings']['thetaresolution'])
            ds2 = xr.Dataset(
                data_vars={},
                coords={'t': ds.t,
                        'N': ds.N,
                        'f': np.arange(0, ds.sf.values / 2, fresolution),
                        'theta': np.arange(start=-np.pi, stop=np.pi, step=2 * np.pi / ndiscretetheta)}
            )

            ds2['f'].attrs = {'long_name': 'f', 'units': 'Hz'}
            ds2['N'].attrs = {'long_name': 'block local time', 'units': 's'}
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

            if not 'SONTEK' in instrument:
                print('statistics from pressure')
                _,vy   = ds.puv.spectrum_simple('p', fresolution=fresolution)

                if vy.shape[0] == 0:
                    print('no valid pressure data remains on this day')
                    continue

                # compute the attenuation factor
                # attenuation corrected spectra
                Sw = ds.puv.attenuation_factor('pressure', elev='hpres', d='d')
                ds['vyp'] = Sw * vy

                ds['Hm0'], ds['Tp'], ds['Tm01'], ds['Tm02'], ds['Tmm10'], ds['Tps'] = (
                    ds.puv.compute_wave_params(var='vyp', **kwargs)
                )

                ds['Hm0'].attrs = {'units': 'm', 'long_name': 'Hm0'}

                ds['Tm01'].attrs = {'units': 's', 'long_name': 'Tm01'}
                ds['Tm02'].attrs = {'units': 's', 'long_name': 'Tm02'}
                ds['Tmm10'].attrs = {'units': 's', 'long_name': 'T_{m-1,0}'}
                ds['fp'] = 1 / ds.Tp
            else:
                ds['fp'] = ds.puv.get_peak_frequency('u', fpmin=config['tailoredWaveSettings']['fmin'])
                ds['Tp'] = 1 / ds.fp

            ds['fp'].attrs = {'units': 'Hz', 'long_name': 'peak frequency'}
            ds['Tp'].attrs = {'units': 's', 'long_name': 'Tp'}
            ds['Tps'].attrs = {'units': 's', 'long_name': 'Tps', 'comment': 'smoothed peak wave period'}
            ##########################################################################
            # wave direction and directional spread
            ##########################################################################
            print('near bed orbital velocity computation')

            ssBounds = [config['tailoredWaveSettings']['fmin'], config['tailoredWaveSettings']['fmax_ss']]
            ds['u_ss'] = ds.puv.band_pass_filter_ss(var='u', freqband=ssBounds)
            ds['v_ss'] = ds.puv.band_pass_filter_ss(var='v', freqband=ssBounds)

            # # compute angle of main wave propagation through singular value decomposition
            # ds['svdtheta'] = ds.puv.compute_SVD_angle()
            # ds['svdtheta'].attrs = (
            #     {'units': 'deg', 'long_name': 'angle principle wave axis',
            #      'comments': 'cartesian convention'}
            # )

            # # rotate velocities so component in wave prop is in ud, rest in vd
            # udr, vdr = (
            #     ds.puv.rotate_velocities(u='u_ss', v='v_ss', theta='svdtheta')
            # )

            # # dirspread in Ruessink's approach
            # ds['svddspr'] = (
            #         np.arctan(
            #             np.sqrt((vdr ** 2).mean(dim='N')
            #                     / (udr ** 2).mean(dim='N')))
            #         / np.pi * 180
            # )
            # ds['svddspr'].attrs = {'units': 'deg', 'long_name': 'angle ud vd',
            #                        'comments': 'from principle component analysis as in Ruessink et al. 2012, max 45deg'}

            # MEM method
            ds['S'] = ds.puv.wave_MEMpuv(eta='p', u='u', v='v', d='d', zi='zi', zb='zb')
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
            ds['ucm'].attrs = {'units': 'm/s', 'long_name': 'cross-shore velocity',
                               'comments': 'block-averaged velocity in cross-shore direction'}
            ds['ulm'].attrs = {'units': 'm/s', 'long_name': 'alongshore velocity',
                               'comments': 'block-averaged velocity in alongshore direction'}

            ##########################################################################
            # wave shape velocity based
            ##########################################################################
            print('wave shape from ov')

            # rotate velocities so component in wave prop is in ud, rest in vd
            u, v = ds.puv.rotate_velocities(u='u', v='v', theta='puvdir')
            ds['udm'] = u.mean(dim='N')
            ds['vdm'] = v.mean(dim='N')
            ds['udm'].attrs = {'units': 'm/s', 'long_name': 'velocity in wave direction',
                               'comments': 'block-averaged'}
            ds['vdm'].attrs = {'units': 'm/s', 'long_name': 'velocity perpendicular to wave direction ',
                               'comments': 'block-averaged'}

            # # on the frequency range adapting to the peak period
            # ds['Sk'], ds['As'], ds['sig'] = ds.puv.compute_SkAs('ud', fixedBounds=False)
            # ds['Sk'].attrs = {'units': 'm3/s3', 'long_name': 'velocity skewness',
            #                   'comment': 'vel-based between 0.5Tp and 2Tp'}
            # ds['As'].attrs = {'units': 'm3/s3', 'long_name': 'velocity asymmetry',
            #                   'comment': 'vel-based between 0.5Tp and 2Tp'}
            # ds['sig'].attrs = {'units': 'm/s', 'long_name': 'std(ud)', 'comment': 'vel-based between 0.5Tp and 2Tp'}

            # in the traditional freqband 0.05-1 Hz
            shapeBounds0 = [config['tailoredWaveSettings']['fmin'], config['tailoredWaveSettings']['fmax_skas0']]
            ds['Sk'], ds['As'], ds['sig'] = (
                ds.puv.compute_SkAs('ud', fixedBounds=True, bounds=shapeBounds0)
            )
            ds['Sk'].attrs = {'units': 'm3/s3', 'long_name': 'near-bed velocity skewness',
                               'comment': 'computed on frequency interval [{} {}] Hz'.format(shapeBounds0[0],
                                                                                        shapeBounds0[1])}
            ds['As'].attrs = {'units': 'm3/s3', 'long_name': 'near-bed velocity asymmetry',
                            'comment': 'computed on frequency interval [{} {}] Hz'.format(shapeBounds0[0],
                                                                                        shapeBounds0[1])}
            ds['sig'].attrs = {'units': 'm/s', 'long_name': 'std(ud)',
                                    'comment': 'computed on frequency interval [{} {}] Hz'.format(shapeBounds0[0],
                                                                                        shapeBounds0[1])}

            # angle between waves and shore normal:
            ds['beachOri'] = config['beachOrientation'][instrument]
            ds['beachOri'].attrs = {'units': 'deg', 'long_name': 'orientation of shoreward',
                                    'comment': 'Cartesian convention'}

            # # skewness on total cross shore and longshore signal
            # ds['Skc'], ds['Asc'], ds['sigc'] = ds.puv.compute_SkAs('uc', fixedBounds=False)
            # ds['Skc'].attrs = {'units': 'm3/s3', 'long_name': 'skewness',
            #                    'comment': 'cross shore vel-based between 0.5Tp and 2Tp'}
            # ds['Asc'].attrs = {'units': 'm3/s3', 'long_name': 'asymmetry',
            #                    'comment': 'cross shore vel-based between 0.5Tp and 2Tp'}
            # ds['sigc'].attrs = {'units': 'm/s', 'long_name': 'std(ud)',
            #                     'comment': 'cross shore vel-based between 0.5Tp and 2Tp'}

            # ds['Skl'], ds['Asl'], ds['sigl'] = ds.puv.compute_SkAs('ul', fixedBounds=False)
            # ds['Skl'].attrs = {'units': 'm3/s3', 'long_name': 'skewness',
            #                    'comment': 'along shore vel-based between 0.5Tp and 2Tp'}
            # ds['Asl'].attrs = {'units': 'm3/s3', 'long_name': 'asymmetry',
            #                    'comment': 'along shore vel-based between 0.5Tp and 2Tp'}
            # ds['sigl'].attrs = {'units': 'm/s', 'long_name': 'std(ud)',
            #                     'comment': 'along shore vel-based between 0.5Tp and 2Tp'}

            ##########################################################################
            # wave shape pressure based
            ##########################################################################
            print('wave shape from pressure')

            # in the original freq range
            ds['Skp'], ds['Asp'], ds['sigp'] = (
                ds.puv.compute_SkAs('p', fixedBounds=True, bounds=shapeBounds0)
            )

            ds['Skp'].attrs = {'units': 'm3', 'long_name': 'near-bed pressure skewness',
                               'comment': 'computed on frequency interval [{} {}] Hz'.format(shapeBounds0[0],
                                                                                        shapeBounds0[1])}
            ds['Asp'].attrs = {'units': 'm3', 'long_name': 'near-bed pressure asymmetry',
                               'comment': 'computed on frequency interval [{} {}] Hz'.format(shapeBounds0[0],
                                                                                        shapeBounds0[1])}
            ds['sigp'].attrs = {'units': 'm', 'long_name': 'std(p)',
                               'comment': 'computed on frequency interval [{} {}] Hz'.format(shapeBounds0[0],
                                                                                        shapeBounds0[1])}

            # # in a band scaled with peak period
            # ds['Skp'], ds['Asp'], ds['sigp'] = ds.puv.compute_SkAs('p', fixedBounds=False)

            # ds['Skp'].attrs = {'units': 'm3', 'long_name': 'skewness',
            #                    'comment': 'pressure-based between 0.5Tp and 2Tp'}
            # ds['Asp'].attrs = {'units': 'm3', 'long_name': 'asymmetry',
            #                    'comment': 'pressure-based between 0.5Tp and 2Tp'}
            # ds['sigp'].attrs = {'units': 'm', 'long_name': 'std(ud)',
            #                     'comment': 'pressure-based between 0.5Tp and 2Tp'}

            ##########################################################################
            # wave numbers
            ##########################################################################
            ds['zs'] = ds.p.mean(dim='N')
            ds['zs'].attrs = {'units': 'm+NAP', 'long_name': 'water level',
                              'comment': 'block-averaged'}

            if not 'SONTEK' in instrument:
                ds['k'] = ds.puv.disper(T='Tmm10', d='d')
                ds['k'].attrs = {'units': 'm-1', 'long_name': 'wave number'}
                ds['Ur'] = ds.puv.Ursell(Hm0='Hm0', k='k', d='d')
                ds['Ur'].attrs = {'units': '-', 'long_name': 'Ursell'}

            ds['As'] = ds.As / ds.sig ** 3
            ds['As'].attrs = {'units': '-', 'long_name': 'near-bed orbital velocity asymmetry'}
            ds['Sk'] = ds.Sk / ds.sig ** 3
            ds['Sk'].attrs = {'units': '-', 'long_name': 'near-bed orbital velocity skewness'}
            ds['Asp'] = ds.Asp / ds.sigp ** 3
            ds['Asp'].attrs = {'units': '-', 'long_name': 'near-bed pressure asymmetry'}
            ds['Skp'] = ds.Skp / ds.sigp ** 3
            ds['Skp'].attrs = {'units': '-', 'long_name': 'near-bed pressure skewness'}

            # we no longer need these:
            vars2drop = ['Skc', 'Asc', 'sigc', 'Skl', 'Asl', 'sigl', 'Skp0', 'Asp0', 'sigp0', 'Skp', 'Asp', 'sigp',
                        'Sk0', 'As0', 'sig0', 'svdtheta', 'svddspr', 'fp', 'udm', 'vdm', 'ud_ssm', 'sig']
            ds = ds.drop_vars(vars2drop, errors='ignore')
            
            ds.attrs['construction datetime'] = datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")
            ds.attrs['summary'] = 'SEDMEX field campaign: tailored timeseries of wave and flow statistics from ADV data computed with linear wave theory. '

            # add script version information
            ds.attrs['git repo'] = r'https://github.com/MarliesA/EURECCA/tree/main/sedmex'
            ds.attrs['git hash'] = get_githash()

            encoding = encoding_sedmex(ds)
            ds.to_netcdf((os.path.join(ncOutDir, file.split('\\')[-1])), encoding=encoding)

    return



if __name__ == "__main__":

    config = yaml.safe_load(Path('c:\checkouts\eurecca_rebuttal\SEDMEX\SEDMEX_processing\sedmex-processing.yml').read_text())

    # loop over all sonteks and adv's
    allVectors = []
    if not config['instruments']['adv']['vector'] == None:
        allVectors += config['instruments']['adv']['vector']
    if not config['instruments']['adv']['sontek'] == None:
        allVectors += config['instruments']['adv']['sontek']

    for instrument in allVectors:

        compute_waves(instrument, config)


