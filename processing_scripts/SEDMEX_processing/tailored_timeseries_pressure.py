import glob
import os
import yaml
from pathlib import Path
import xarray as xr
import numpy as np
from datetime import datetime
import puv
from sedmex_info_loaders import get_githash
import xrMethodAccessors

def compute_waves(instr, config):
    file = glob.glob(os.path.join(config['experimentFolder'], instr, 'qc', r'*.nc'))
    with xr.open_dataset(file[0]) as ds:
        ds['zs'] = ds.p.mean(dim='N') / config['physicalConstants']['rho'] / config['physicalConstants']['g']
        ds['zs'].attrs = {'units': 'm+NAP', 'long_name': 'water level',
                          'comment': 'burst averaged'}

        ds['d'] = ds.zs - ds.zb
        ds['d'].attrs = {'long_name': 'water depth', 'units': 'm'}

        # read fresolution from config
        if 'OSSI' in instr:
            fresolution = config['tailoredWaveSettings']['fresolution']['ossi']
        elif 'SOLO' in instr:
            fresolution = config['tailoredWaveSettings']['fresolution']['solo']

        ds2 = xr.Dataset(
            data_vars={},
            coords=dict(t=ds.t,
                        N=ds.N,
                        f=np.arange(0, ds.sf.values / 2, fresolution)
                        ))
        ds2['f'].attrs = {'long_name': 'f', 'units': 'Hz'}
        ds2['t'].attrs = {'long_name': 'time', 'units': 's'}
        ds2['N'].attrs = {'long_name': 'burst time', 'units': 's'}
        for key in ds.data_vars:
            ds2[key] = ds[key]
            ds2[key].attrs = ds[key].attrs
        ds2.attrs = ds.attrs
        ds = ds2

        ds['eta'] = ds.p / config['physicalConstants']['rho'] / config['physicalConstants']['g']
        _, vy = ds.puv.spectrum_simple('eta', fresolution=fresolution)

        ds['elev'] = ds.zi - ds.zb
        ds['elev'].attrs = {'long_name': 'instrument height above bed', 'units': 'm'}

        Q = ds.puv.attenuation_factor('pressure', elev='elev', d='d')
        ds['vy'] = Q * vy
        ds['vy'].attrs = {'units': 'm2/hz', 'long_name': 'variance density of surface elevation'}

        kwargs = {'fmin': config['tailoredWaveSettings']['fmin'], 'fmax': config['tailoredWaveSettings']['fmax']}
        ds['Hm0'], ds['Tp'], ds['Tm01'], ds['Tm02'], ds['Tmm10'], ds['Tps'] = (
            ds.puv.compute_wave_params(var='vy', **kwargs)
        )

        ds['Hm0'].attrs = {'units': 'm', 'long_name': 'Hm0'}
        ds['Tp'].attrs = {'units': 's', 'long_name': 'Tp'}
        ds['Tps'].attrs = {'units': 's', 'long_name': 'Tps'}
        ds['Tm01'].attrs = {'units': 's', 'long_name': 'Tm01'}
        ds['Tm02'].attrs = {'units': 's', 'long_name': 'Tm02'}
        ds['Tmm10'].attrs = {'units': 's', 'long_name': 'T_{m-1,0}'}
        ds['fp'] = 1 / ds.Tp
        ds['fp'].attrs = {'units': 'Hz', 'long_name': 'peak frequency'}

        ds['k'] = xr.apply_ufunc(lambda tm01, h: puv.disper(2 * np.pi / tm01, h), ds['Tmm10'], ds['zs'] - ds['zb'])
        ds['k'].attrs = {'units': 'm-1', 'long_name': 'k'}
        ds['Ur'] = xr.apply_ufunc(lambda hm0, k, h: 3 / 4 * 0.5 * hm0 * k / (k * h) ** 3, ds['Hm0'], ds['k'],
                                  ds['zs'] - ds['zb'])
        ds['Ur'].attrs = {'units': '-', 'long_name': 'Ursell'}

        # in the original freq range
        shapeBounds0 = [config['tailoredWaveSettings']['fmin'], config['tailoredWaveSettings']['fmax_skas0']]

        ds['Skp0'], ds['Asp0'], ds['sigp0'] = (
            ds.puv.compute_SkAs('eta', fixedBounds=True, bounds=shapeBounds0)
        )

        ds['Skp0'].attrs = {'units': 'm3', 'long_name': 'skewness',
                            'comment': 'pressure-based between {} and {} Hz'.format(shapeBounds0[0], shapeBounds0[1])}
        ds['Asp0'].attrs = {'units': 'm3', 'long_name': 'asymmetry',
                            'comment': 'pressure-based between {} and {} Hz'.format(shapeBounds0[0], shapeBounds0[1])}
        ds['sigp0'].attrs = {'units': 'm', 'long_name': 'std(ud)',
                             'comment': 'pressure-based between {} and {} Hz'.format(shapeBounds0[0], shapeBounds0[1])}

        # in a band scaled with peak period
        ds['Skp'], ds['Asp'], ds['sigp'] = ds.puv.compute_SkAs('eta', fixedBounds=False)

        ds['Skp'].attrs = {'units': 'm3', 'long_name': 'skewness', 'comment': 'pressure-based between 0.5Tp and 2Tp'}
        ds['Asp'].attrs = {'units': 'm3', 'long_name': 'asymmetry',
                           'comment': 'pressure-based between 0.5Tp and 2Tp'}
        ds['sigp'].attrs = {'units': 'm', 'long_name': 'std(ud)', 'comment': 'pressure-based between 0.5Tp and 2Tp'}

        ds['nAs'] = ds.Asp / ds.sigp ** 3
        ds['nSk'] = ds.Skp / ds.sigp ** 3
        ds['nAs'].attrs = {'units': '-', 'long_name': 'As'}
        ds['nSk'].attrs = {'units': '-', 'long_name': 'Sk'}

        ds = ds.drop_vars(['eta'])

        ds.attrs['summary'] = 'SEDMEX field campaign, wave statistics from pressure recordings'
        ds.attrs['construction datetime'] = datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")

        # saving to file
        fold = os.path.join(config['experimentFolder'], instr, 'tailored')
        if not os.path.isdir(fold):
            os.mkdir(fold)
        ncFilePath = os.path.join(fold, '{}.nc'.format(instr))

        # add script version information
        ds.attrs['git repo'] = r'https://github.com/MarliesA/EURECCA/tree/main/sedmex'
        ds.attrs['git hash'] = get_githash()

        # if nothing else, at least specify lossless zlib compression
        comp = dict(zlib=True, complevel=5)
        ds.encoding = {var: comp for var in ds.data_vars}
        for coord in list(ds.coords.keys()):
            ds.encoding[coord] = {'zlib': False, '_FillValue': None}

        ds.to_netcdf(ncFilePath, encoding=ds.encoding)

        return


if __name__ == "__main__":

    config = yaml.safe_load(Path('sedmex-processing.yml').read_text())

    for instr in config['instruments']['ossi'] + config['instruments']['solo']:
        print(instr)
        compute_waves(instr, config)
