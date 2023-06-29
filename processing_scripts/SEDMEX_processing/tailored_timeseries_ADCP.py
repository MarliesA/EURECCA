import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import xrMethodAccessors
import glob
import os


def tailor_this_dataset(instrumentName):
    fileNames = glob.glob(os.path.join(experimentFolder, instrumentName, 'qc', '*.nc'))

    # prepare the save directory and place a copy of the script in this file
    ncOutDir = os.path.join(experimentFolder, instrumentName, 'tailored')
    if not os.path.isdir(ncOutDir):
        os.mkdir(ncOutDir)

    for file in fileNames:
        print(file)
        ds = xr.open_dataset(file)

        ds['zs'] = ds.eta.mean(dim='N')

        ds['uzm'] = ds.u.mean(dim='N')
        ds['vzm'] = ds.v.mean(dim='N')
        ds['uzmag'] = np.sqrt(ds.uzm**2+ds.vzm**2)
        ds['uzang'] = np.arctan2(ds.vzm, ds.uzm)*180/np.pi

        ds.uzm.attrs = {'units': 'm/s', 'long_name':'burst averaged velocity-East'}
        ds.vzm.attrs = {'units': 'm/s', 'long_name':'burst averaged velocity-North'}
        ds.uzmag.attrs = {'units': 'm/s', 'long_name': 'burst averaged velocity magnitude'}
        ds.uzang.attrs = {'units': 'deg', 'long_name': 'burst averaged velocity angle', 'comments': 'cartesian convention'}

        ds['um'] = ds.uzm.mean(dim='z')
        ds['vm'] = ds.vzm.mean(dim='z')
        ds['umag'] = np.sqrt(ds.um**2+ds.vm**2)
        ds['uang'] = np.arctan2(ds.vm, ds.um)*180/np.pi

        ds.um.attrs = {'units': 'm/s', 'long_name': 'burst-depth averaged velocity-East'}
        ds.vm.attrs = {'units': 'm/s', 'long_name': 'burst-depth averaged velocity-North'}
        ds.umag.attrs = {'units': 'm/s', 'long_name': 'burst-depth averaged velocity magnitude'}
        ds.uang.attrs = {'units': 'deg', 'long_name': 'burst-depth averaged velocity angle','comments': 'cartesian convention'}


        ds['u_nb'] = ds.u.sel(z=ds.zb+0.15, method='nearest')
        ds['v_nb'] = ds.v.sel(z=ds.zb+0.15, method='nearest')
        ds['w_nb'] = ds.v.sel(z=ds.zb+0.15, method='nearest')

        ds.u_nb.attrs = {'units': 'm/s', 'long_name': 'near-bed velocity-East'}
        ds.v_nb.attrs = {'units': 'm/s', 'long_name': 'near-bed velocity-North'}

        ds['um_nb'] = ds.u_nb.mean(dim='N')
        ds['vm_nb'] = ds.v_nb.mean(dim='N')
        ds['umag_nb'] = np.sqrt(ds.um_nb**2+ds.vm_nb**2)
        ds['uang_nb'] = np.arctan2(ds.vm_nb, ds.um_nb)*180/np.pi

        ds.um_nb.attrs = {'units': 'm/s', 'long_name': 'burst averaged near-bed velocity-East'}
        ds.vm_nb.attrs = {'units': 'm/s', 'long_name': 'burst averaged near-bed velocity-North'}
        ds.umag_nb.attrs = {'units': 'm/s', 'long_name': 'burst averaged near-bed velocity magnitude'}
        ds.umag_nb.attrs = {'units': 'deg', 'long_name': 'burst averaged near-bed velocity angle','comments': 'cartesian convention'}

        ##########################################################################
        # extent the dataset with appropriate frequency axis
        ##########################################################################
        print('extending dataset with freq axis')
        sf = ds.sf.values
        powerStep = 7  # 9 for investigating the spectra 9 = 0.03Hz resolution
        fresolution = np.round(sf * np.exp(-np.log(2) * powerStep), 5)
        ndiscretetheta = 64
        ds2 = xr.Dataset(
            data_vars={},
            coords={'t': ds.t,
                    'N': ds.N,
                    'z': ds.z,
                    'f': np.arange(0, ds.sf.values / 2, fresolution),
                    'theta': np.arange(start=-np.pi, stop=np.pi, step=2 * np.pi / ndiscretetheta)}
        )

        ds2['f'].attrs = {'long_name': 'f', 'units': 'Hz'}
        ds2['N'].attrs = {'long_name': 'burst time', 'units': 's'}
        ds2['z'].attrs = {'long_name': 'z', 'units': 'm+NAP'}
        ds2['theta'].attrs = {'long_name': 'direction', 'units': 'rad'}

        # copy all data over into this new structure
        for key in ds.data_vars:
            ds2[key] = ds[key]
        ds2.attrs = ds.attrs
        ds = ds2

        ##########################################################################
        # statistics computed from pressure
        ##########################################################################
        print('statistics from pressure')
        _, vy = ds.puv.spectrum_simple('eta', fresolution=fresolution)

        # compute the attenuation factor
        # attenuation corrected spectra
        Sw = ds.puv.attenuation_factor('pressure', elev='elev', d='d')
        ds['vyp'] = Sw * vy

        kwargs = {'fmin': 0.1, 'fmax': 1.5}
        ds['Hm0'], ds['Tp'], ds['Tm01'], ds['Tm02'], ds['Tmm10'], ds['Tps'] = (
            ds.puv.compute_wave_params(var='vyp', **kwargs)
        )

        ds['Hm0'].attrs = {'units': 'm', 'long_name': 'Hm0'}
        ds['Tp'].attrs = {'units': 's', 'long_name': 'Tp'}
        ds['Tps'].attrs = {'units': 's', 'long_name': 'Tps'}
        ds['Tm01'].attrs = {'units': 's', 'long_name': 'Tm01'}
        ds['Tm02'].attrs = {'units': 's', 'long_name': 'Tm02'}
        ds['Tmm10'].attrs = {'units': 's', 'long_name': 'T_{m-1,0}'}

        ##########################################################################
        # wave direction and directional spread
        ##########################################################################
        print('near bed orbital velocity computation')

        ds['fp'] = 1 / ds.Tp
        ds['u_ss'] = ds.puv.band_pass_filter_ss(var='u_nb', freqband=[0.05, 1.5])
        ds['v_ss'] = ds.puv.band_pass_filter_ss(var='v_nb', freqband=[0.05, 1.5])

        # compute angle of main wave propagation through singular value decomposition
        ds['svdtheta'] = ds.puv.compute_SVD_angle(u='u_ss', v='v_ss')
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
                               'comments': 'as in Ruessink et al. 2012, max 45deg'}

        # MEM method: we do this on the cell closest to the bed
        ds['zi_nb'] = ds.z.sel(z=(ds.zb+0.15),method='nearest')
        ds['S'] = ds.puv.wave_MEMpuv(eta='eta', u='u_ss', v='v_ss', d='d', zi='zi_nb', zb='zb')
        ds['S'].attrs = (
            {'units': 'm2/Hz/rad', 'long_name': '2D energy density',
             'comments': 'cartesian convention, from puv Method of maximum Entropy'}
        )

        _, _, _, _, _, _, ds['puvdir'], ds['puv_dspr'] = ds.puv.compute_wave_params_from_S(var='S')
        ds['puvdir'].attrs = (
            {'units': 'deg', 'long_name': 'wave prop dir',
             'comments': 'cartesian convention, from puv Method of maximum Entropy'}
        )

        ds['puvdir'].attrs = (
            {'units': 'deg', 'long_name': 'wave directional spread',
             'comments': 'cartesian convention, from puv Method of maximum Entropy'}
        )

        # rotate velocities so component in wave prop is in ud, rest in vd
        ds['ud'], ds['vd'] = (
            ds.puv.rotate_velocities(u='u_ss', v='v_ss', theta='puvdir')
        )
        ds['ud'].attrs = {'units': 'm/s', 'long_name': 'u_wavdir'}
        ds['vd'].attrs = {'units': 'm/s', 'long_name': 'vv_wavdir '}

        # rms orbital velocity
        ds['u_ssm'] = np.sqrt(ds.u_ss ** 2 + ds.v_ss ** 2).mean(dim='N')
        ds['u_ssm'].attrs = (
            {'units': 'm/s', 'long_name': 'u_o',
             'comments': 'Root mean squared total (u_ss+v_ss) orbital velocity'}
        )

        ds['ud_ssm'] = np.sqrt(ds.ud ** 2).mean(dim='N')
        ds['ud_ssm'].attrs = {'units': 'm/s', 'long_name': 'u_o',
                              'comments': 'Root mean squared orbital velocity in wave propagation direction'}

        # rotate velocities so component in wave prop is in ud, rest in vd
        ds['ud'], ds['v'] = ds.puv.rotate_velocities(u='u_nb', v='v_nb', theta='puvdir')
        ds['ud'].attrs = {'units': 'm/s', 'long_name': 'u mean wavdir',
                           'comments': 'burst averaged'}
        ds['vd'].attrs = {'units': 'm/s', 'long_name': 'v_mean wavdir ',
                           'comments': 'burst averaged'}


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

        ds['k'] = ds.puv.disper()
        ds['k'].attrs = {'units': 'm-1', 'long_name': 'k'}
        ds['Ur'] = ds.puv.Ursell()
        ds['Ur'].attrs = {'units': '-', 'long_name': 'Ursell'}

        ds['nAs'] = ds.As / ds.sig ** 3
        ds['nAs'].attrs = {'units': '-', 'long_name': 'near-bed orbital velocity asymmetry'}
        ds['nSk'] = ds.Sk / ds.sig ** 3
        ds['nSk'].attrs = {'units': '-', 'long_name': 'near-bed orbital velocity skewness'}

        # specify compression for all the variables to reduce file size
        comp = dict(zlib=True, complevel=5)
        ds.encoding = {var: comp for var in ds.data_vars}

        ds.to_netcdf((os.path.join(ncOutDir, file.split('\\')[-1])))


if __name__ == "__main__":

    experimentFolder = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20210908_campaign\instruments'

    for instrumentName in ['L4C1ADCP', 'L2C7ADCP']:
        tailor_this_dataset(instrumentName)
