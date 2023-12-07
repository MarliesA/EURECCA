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
from encoding_sedmex import encoding_sedmex

def compute_waves(instr, config):
    file = glob.glob(os.path.join(config['experimentFolder'], instr, 'qc', r'*.nc'))
    with xr.open_dataset(file[0]) as ds:
        ds['zs'] = ds.p.mean(dim='N') 
        ds['zs'].attrs = {'units': 'm+NAP', 'long_name': 'water level',
                          'comment': 'block-averaged'}

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


        _, vy = ds.puv.spectrum_simple('p', fresolution=fresolution)

        ds['h'] = ds.zi - ds.zb
        ds['h'].attrs = {'long_name': 'instrument height above bed', 'units': 'm'}

        Q = ds.puv.attenuation_factor('pressure', elev='h', d='d')
        ds['vy'] = Q * vy
        ds['vy'].attrs = {'units': 'm2/hz', 'long_name': 'variance density of surface elevation'}

        kwargs = {'fmin': config['tailoredWaveSettings']['fmin'], 'fmax': config['tailoredWaveSettings']['fmax']}
        ds['Hm0'], ds['Tp'], ds['Tm01'], ds['Tm02'], ds['Tmm10'], ds['Tps'] = (
            ds.puv.compute_wave_params(var='vy', **kwargs)
        )

        ds['Hm0'].attrs = {'units': 'm', 'long_name': 'Hm0'}
        ds['Tp'].attrs = {'units': 's', 'long_name': 'Tp'}
        ds['Tps'].attrs = {'units': 's', 'long_name': 'Tps', 'comment': 'smoothed peak wave period'}
        ds['Tm01'].attrs = {'units': 's', 'long_name': 'Tm01'}
        ds['Tm02'].attrs = {'units': 's', 'long_name': 'Tm02'}
        ds['Tmm10'].attrs = {'units': 's', 'long_name': 'T_{m-1,0}'}
        ds['fp'] = 1 / ds.Tp
        ds['fp'].attrs = {'units': 'Hz', 'long_name': 'peak frequency'}

        ds['k'] = xr.apply_ufunc(lambda tm01, h: puv.disper(2 * np.pi / tm01, h), ds['Tmm10'], ds['zs'] - ds['zb'])
        ds['k'].attrs = {'units': 'm-1', 'long_name': 'wave number'}
        ds['Ur'] = xr.apply_ufunc(lambda hm0, k, h: 3 / 4 * 0.5 * hm0 * k / (k * h) ** 3, ds['Hm0'], ds['k'],
                                  ds['zs'] - ds['zb'])
        ds['Ur'].attrs = {'units': '-', 'long_name': 'Ursell'}

        # in the original freq range
        shapeBounds0 = [config['tailoredWaveSettings']['fmin'], config['tailoredWaveSettings']['fmax_skas0']]

        ds['Skp'], ds['Asp'], ds['sigp'] = (
            ds.puv.compute_SkAs('p', fixedBounds=True, bounds=shapeBounds0)
        )

        ds['Skp'].attrs = {'units': 'm3', 'long_name': 'pressure skewness',
                               'comment': 'computed on frequency interval [{} {}] Hz'.format(shapeBounds0[0],
                                                                                        shapeBounds0[1])}
        ds['Asp'].attrs = {'units': 'm3', 'long_name': 'pressure asymmetry',
                               'comment': 'computed on frequency interval [{} {}] Hz'.format(shapeBounds0[0],
                                                                                        shapeBounds0[1])}
        ds['sigp'].attrs = {'units': 'm', 'long_name': 'std(p)',
                               'comment': 'computed on frequency interval [{} {}] Hz'.format(shapeBounds0[0],
                                                                                        shapeBounds0[1])}
        ds['Asp'] = ds.Asp / ds.sigp ** 3
        ds['Asp'].attrs = {'units': '-', 'long_name': 'near-bed pressure asymmetry'}
        ds['Skp'] = ds.Skp / ds.sigp ** 3
        ds['Skp'].attrs = {'units': '-', 'long_name': 'near-bed pressure skewness'}

        # we no longer need these:
        vars2drop = ['Skc', 'Asc', 'sigc', 'Skl', 'Asl', 'sigl', 'Skp0', 'Asp0', 'sigp0', 'Skp', 'Asp', 'sigp',
                'Sk0', 'As0', 'sig0', 'svdtheta', 'svddspr', 'fp', 'udm', 'vdm', 'ud_ssm', 'p']
        ds = ds.drop_vars(vars2drop, errors='ignore')
    
        ds['sf'].attrs = {'units': 'Hz', 'long_name': 'sampling frequency'}  
        ds['t'].attrs = {'long_name': 'block start time'} 

        ds.attrs['construction datetime'] = datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")
        ds.attrs['summary'] = 'SEDMEX field campaign: tailored timeseries of wave statistics from pressure data computed with linear wave theory. '

        # saving to file
        fold = os.path.join(config['experimentFolder'], instr, 'tailored')
        if not os.path.isdir(fold):
            os.mkdir(fold)
        ncFilePath = os.path.join(fold, '{}.nc'.format(instr))

        # add script version information
        ds.attrs['git repo'] = r'https://github.com/MarliesA/EURECCA/tree/main/sedmex'
        ds.attrs['git hash'] = get_githash()

        encoding = encoding_sedmex(ds)
        ds.to_netcdf(ncFilePath, encoding=encoding)

        return


if __name__ == "__main__":

    config = yaml.safe_load(Path('c:\checkouts\eurecca_rebuttal\SEDMEX\SEDMEX_processing\sedmex-processing.yml').read_text())

    # loop over all sonteks and adv's
    allInstr = []
    if not config['instruments']['ossi'] == None:
        allInstr += config['instruments']['ossi']
    if not config['instruments']['solo'] == None:
        allInstr += config['instruments']['solo']

    for instr in allInstr:
        print(instr)
        compute_waves(instr, config)
