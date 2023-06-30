import os
import yaml
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cast_to_blocks(ds0, burstDuration):

    # reshape to one row per burst in data array
    pt = ds0.p.values
    nSamples = len(pt)
    dt = ds0.isel(t=1).t - ds0.isel(t=0).t

    burstLength = int(burstDuration / dt)
    nBursts = int(np.floor(nSamples / burstLength))

    pt = pt[:nBursts * burstLength]
    h = ds0.h[::burstLength]
    h = h[:nBursts]
    zi = ds0.zi[::burstLength]
    zi = zi[:nBursts]
    zb = ds0.zb[::burstLength]
    zb = zb[:nBursts]
    t = ds0.t[::burstLength]
    t = t[:nBursts]
    N = (ds0.t.values[:burstLength] - ds0.t.values[0]) / np.timedelta64(1, 's')

    # cast into a 2D array
    ds = xr.Dataset(data_vars={},
                    coords={'t': t, 'N': N})

    # copy all data over into this new structure
    ds['p'] = (('t', 'N'), pt.reshape((nBursts, burstLength)))
    ds['zi'] = zi
    ds['zb'] = zb
    ds['h'] = h
    ds['sf'] = ds0.sf

    return ds

if __name__ == "__main__":

    config = yaml.safe_load(Path('sedmex-processing.yml').read_text())

    dsCtd = xr.open_dataset(os.path.join(config['experimentFolder'], r'L2C1CTD\raw_netcdf\L2C1CTD_20210910.nc'))

    #%% process solo's
    reft = {
    'L2C2SOLO': ['20210910 05:00',
        '20210919 15:00',
        '20210920 14:30',
        '20210926 05:30'],
    'L2C4SOLO':  ['20210910 05:00',
        '20210919 15:00',
        '20210920 14:30',
        '20210926 05:30',
        '20211004 14:00',
        '20211007 16:00',
        '20211011 06:00',
        '20211019 13:30'],
    'L2C10SOLO':  ['20210919 15:00',
        '20211019 13:30'],
    'L4C1SOLO':  ['20210919 15:00',
        '20210920 14:30',
        '20210926 05:30',
        '20211007 16:00',
        '20211011 06:00',
        '20211019 13:30']
    }

    pclist = []
    for instr in config['instruments']['solo']:
        dataFile = os.path.join(config['experimentFolder'], instr, 'raw_netcdf', instr + '.nc')
        ds0 = xr.open_dataset(dataFile)

        # make sure there is bed level information on all moments in time
        ds0['zb'] = ds0.zb.interpolate_na('t', method='nearest', fill_value='extrapolate')
        ds0['h'] = ds0.h.interpolate_na('t', method='nearest', fill_value='extrapolate')/100
        ds0['zi'] = ds0.zb + ds0.h

        # correct for the air pressure fluctuations
        # select those dates on which we know the measured pressure should be equal
        # 0 (i.e. when we are sure they are emerged) and linearly interpolate
        # the drift in reference level in between
        pair = dsCtd.p.interp_like(ds0)
        p2 = ds0.p - pair
        p_dry = p2.resample({'t': '15T'}, loffset='15T').mean().sel({'t': pd.to_datetime(reft[instr])})
        p3 = p2 - p_dry.interp_like(ds0.p,method='linear').bfill(dim='t').ffill(dim='t')
        ds0['p'] = p3 + ds0.zi*config['physicalConstans']['rho']*config['physicalConstans']['g']

        ds = cast_to_blocks(ds0, burstDuration=config['burstDuration']['solo'])

        # remove all bursts where instrument fell dry
        ds['p'] = ds.p.where( ds.p.std(dim='N') > 70)
        ds['p'].attrs = {'units': 'Pa +NAP','long_name': 'pressure','comments': 'corrected for air pressure'}

        ds['zi'].attrs = {'units': 'm+NAP','long_name': 'zi'}
        ds['zb'].attrs = {'units': 'm+NAP','long_name': 'zb'}
        ds['sf'].attrs = {'units': 'Hz','long_name': 'sampling frequency'}
        ds.attrs = ds0.attrs
        ds.attrs['summary'] = 'SEDMEX field campaign, pressure corrected for air pressure and quality checked'
        ds['name'] = instr

        folderOut = os.path.join(config['experimentFolder'], instr, 'qc')
        if not os.path.isdir(folderOut):
            os.mkdir(folderOut)
        ncFilePath = os.path.join(folderOut, '{}.nc'.format(instr))

        comp = dict(zlib=True, complevel=5)
        ds.encoding = {var: comp for var in ds.data_vars}
        for coord in list(ds.coords.keys()):
            ds.encoding[coord] = {'zlib': False, '_FillValue': None}

        ds.to_netcdf(ncFilePath, encoding=ds.encoding )



