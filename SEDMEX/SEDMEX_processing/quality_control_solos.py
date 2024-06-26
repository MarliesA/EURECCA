import os
import yaml
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from sedmex_info_loaders import get_githash
from encoding_sedmex import encoding_sedmex
from datetime import datetime

def cast_to_blocks(ds0, burstDuration):

    # reshape to one row per burst in data array
    pt = ds0.p.values
    nSamples = len(pt)
    dt = pd.to_timedelta((ds0.isel(t=1).t - ds0.isel(t=0).t).to_numpy()).total_seconds()

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
        ds['zb'].attrs = {'units': 'm+NAP','long_name': 'bed level'}

        ds0['h'] = ds0.h.interpolate_na('t', method='nearest', fill_value='extrapolate')/100
        ds['h'].attrs = {'units': 'm','long_name': 'instrument height above bed', 'comment': 'neg down'}

        ds0['zi'] = ds0.zb + ds0.h
        ds['zi'].attrs = {'units': 'm+NAP', 'long_name': 'instrument position'}

        # correct for the air pressure fluctuations
        # select those dates on which we know the measured pressure should be equal
        # 0 (i.e. when we are sure they are emerged) and linearly interpolate
        # the drift in reference level in between
        pair = dsCtd.p.interp_like(ds0)
        p2 = ds0.p - pair
        p_dry = p2.resample({'t': '15T'}, loffset='15T').mean().sel({'t': pd.to_datetime(reft[instr])})
        p3 = p2 - p_dry.interp_like(ds0.p, method='linear').bfill(dim='t').ffill(dim='t')
        ds0['p'] = p3 + ds0.zi*config['physicalConstants']['rho']*config['physicalConstants']['g']

        ds = cast_to_blocks(ds0, burstDuration=config['burstDuration']['solo'])
        ds['sf'] = config['samplingFrequency']['solo']
        ds['sf'].attrs = {'units': 'Hz', 'long_name': 'sampling frequency'}

        # remove all bursts where instrument fell dry
        ds['p'] = ds.p.where( ds.p.std(dim='N') > 70)

        # after rebuttal change unit
        ds['p'] = ds.p/config['physicalConstants']['rho']*config['physicalConstants']['g']
        ds['p'].attrs = {'units': 'm+NAP', 'long_name': 'hydrostatic surface elevation','comments': 'corrected for air pressure and referenced to NAP'}  
        
        ds['N'].attrs = {'units': 's', 'long_name': 'block local time'} 
        ds['t'].attrs = {'long_name': 'block start time'}  

        ds.attrs = ds0.attrs
        ds.attrs['construction datetime'] = datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")
        ds.attrs['summary'] = 'SEDMEX field campaign: quality checked pressure data. Corrected for air pressure and referenced to NAP' 
        # ds['name'] = instr

        folderOut = os.path.join(config['experimentFolder'], instr, 'qc')
        if not os.path.isdir(folderOut):
            os.mkdir(folderOut)
        ncFilePath = os.path.join(folderOut, '{}.nc'.format(instr))

        # add script version information
        ds.attrs['git repo'] = r'https://github.com/MarliesA/EURECCA/tree/main/sedmex'
        ds.attrs['git hash'] = get_githash()

        encoding = encoding_sedmex(ds)
        ds.to_netcdf(ncFilePath, encoding=encoding )



