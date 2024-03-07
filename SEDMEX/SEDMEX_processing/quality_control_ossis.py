import os
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from sedmex_info_loaders import get_githash
from encoding_sedmex import encoding_sedmex
from datetime import datetime

def cast_to_blocks(ds, sf, burstDuration):

    p = ds.p.values
    t = ds.t.values

    blockLength = int(burstDuration * sf)

    # make a row per burst, and only full bursts are included
    NB = int(np.floor(len(p) / blockLength))
    p = p[0:int(NB * blockLength)]
    t = t[0:int(NB * blockLength)]

    p2 = p.reshape(NB, blockLength)

    # check these two are the same
    t2 = t.reshape(NB, blockLength)
    blockStartTimes = t[0::blockLength]

    ds2 = xr.Dataset(
        data_vars=dict(
            sf=sf,
            p=(['t', 'N'], p2)
        ),
        coords=dict(t=blockStartTimes,
                    N=list(np.arange(0, burstDuration * sf) / sf)
                    )
    )

    return ds2

if __name__ == "__main__":

    config = yaml.safe_load(Path('c:\checkouts\eurecca_rebuttal\SEDMEX\SEDMEX_processing\sedmex-processing.yml').read_text())

    # air pressure
    dsCtd = xr.open_dataset(config['experimentFolder'] + r'\L2C1CTD\raw_netcdf\L2C1CTD_20210910.nc')

    pclist = []
    for instr in config['instruments']['ossi']:
        print(instr)

        dataFile = os.path.join(config['experimentFolder'], instr, 'raw_netcdf', instr + '.nc')
        ds = xr.open_dataset(dataFile)

        ds['zb'] = ds.zb.interpolate_na('t', method='nearest', fill_value='extrapolate')
        ds['h'] = ds.h.interpolate_na('t', method='nearest', fill_value='extrapolate')

        # make sure that we have nan's on moments with missing data,
        # because we rely on index number for the timing
        ds2 = ds.resample(t='100ms').nearest(tolerance='100ms')

        # cast to blocks of fixed length to later do wave analysis on
        ds2 = cast_to_blocks(ds2, sf=ds.sf, burstDuration=config['burstDuration']['ossi'])
        
        # add these burst averaged variables back on the dataset
        ds2['h'] = ds.h.interp_like(ds2.t)/100
        ds2['zb'] = ds.zb.interp_like(ds2.t)
        ds2['zi'] = ds2.zb + ds2.h

        # correct for the air pressure fluctuations
        # we assume the minimal pressure on ds is more or less emerged
        pair = dsCtd.p.interp_like(ds2)
        p3 = ds2.p - ds2.p.min() - pair + pair.min()

        # reference pressure to correct height based on instrument height and identified instrument offset
        rhog = config['physicalConstants']['rho']*config['physicalConstants']['g']

        if instr in config['qcOSSISettings']['zsOffset']:
            ds2['p'] = p3 + rhog * (ds2.zi + config['qcOSSISettings']['zsOffset'][instr])
        else:
            ds2['p'] = p3 + rhog * ds2.zi

        # also prescribe offsets at locations where we believe barnacles were covering the membrane creating an offset
        for t in config['qcOSSISettings']['zsOffset_temporal']:
            if instr in config['qcOSSISettings']['zsOffset_temporal'][t]['instr']:
                tstart = pd.to_datetime(str(config['qcOSSISettings']['zsOffset_temporal'][t]['tstart']))
                tstop = pd.to_datetime(str(config['qcOSSISettings']['zsOffset_temporal'][t]['tstop']))
                offset = config['qcOSSISettings']['zsOffset_temporal'][t]['offset']
                ds2['p'].loc[dict(t=ds2.t[(ds2.t > tstart) & (ds2.t < tstop)])] = \
                    ds2['p'].loc[dict(t=ds2.t[(ds2.t > tstart) & (ds2.t < tstop)])] + offset*rhog

        # ML: changed after rebuttal according to Martins
        ds2['p'] = ds2.p/rhog
        ds2['p'].attrs = {'units': 'm+NAP', 'long_name': 'hydrostatic surface elevation', 'comments': 'corrected for drift air pressure and referenced to NAP'}
        ds2['h'].attrs = {'units': 'm', 'long_name': 'instrument height above bed', 'comment': 'neg down'}
        ds2['zb'].attrs = {'units': 'm+NAP', 'long_name': 'bed level'}
        ds2['zi'].attrs = {'units': 'm+NAP', 'long_name': 'instrument position'}

        ds2['sf'].attrs = {'units': 'Hz', 'long_name': 'sampling frequency'}  # to be added
        ds2['N'].attrs = {'units': 's', 'long_name': 'block local time'}  # to be added 
        ds2['t'].attrs = {'long_name': 'block start time'}  # to be added 

        ds2.attrs = ds.attrs
        ds2.attrs['construction datetime'] = datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")
        ds2.attrs['summary'] = 'SEDMEX field campaign: quality checked pressure data. Corrected for air pressure and referenced to NAP' 

        # saving to file
        outFolder = os.path.join(config['experimentFolder'], instr, 'qc')
        if not os.path.isdir(outFolder):
            os.mkdir(outFolder)
        ncFilePath = os.path.join(outFolder, '{}.nc'.format(instr))

        # add script version information
        ds2.attrs['git repo'] = r'https://github.com/MarliesA/EURECCA/tree/main/sedmex'
        ds2.attrs['git hash'] = get_githash()

        encoding = encoding_sedmex(ds2)
        ds2.to_netcdf(ncFilePath, encoding=encoding )

