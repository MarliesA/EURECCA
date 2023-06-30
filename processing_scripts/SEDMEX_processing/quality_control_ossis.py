import os
import yaml
from pathlib import Path
import numpy as np
import xarray as xr
from sedmex_info_loaders import get_githash

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

    config = yaml.safe_load(Path('sedmex-processing.yml').read_text())

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
        ds2['zi'] = ds.zb + ds.h

        # correct for the air pressure fluctuations
        # we assume the minimal pressure on ds is more or less emerged
        pair = dsCtd.p.interp_like(ds2)
        p3 = ds2.p - ds2.p.min() - pair + pair.min()

        #fix offset with respect to the solo at L2C10:
        #these were identified manually by finding the average discrepenacy over all meas
        rhog = config['physicalConstans']['rho']*config['physicalConstans']['g']
        if instr=='L2C9OSSI':
            ds2['p'] = p3 + ds2.zi*rhog + rhog*0.58
        elif instr=='L2C8OSSI':
            ds2['p'] = p3 + ds2.zi*rhog + rhog*0.31
        elif instr=='L2C6OSSI':
            ds2['p'] = p3 + ds2.zi*rhog + rhog*0.30
        else:
            ds2['p'] = p3 + ds2.zi*rhog

        ds2['p'].attrs = {'units': 'Pa +NAP', 'long_name': 'pressure', 'comments': 'corrected for drift air pressure'}
        ds2['h'].attrs = {'units': 'm', 'long_name': 'instrument height above bed', 'comment': 'neg down'}
        ds2['zb'].attrs = {'units': 'm+NAP', 'long_name': 'bed level', 'comment': 'neg down'}
        ds2['zi'].attrs = {'units': 'm+NAP', 'long_name': 'instrument position'}

        ds2.attrs = ds.attrs
        ds2.attrs['summary'] = 'SEDMEX field campaign, pressure corrected for air pressure. There are inconsistencies between de mean pressure measured with ossis and solos.'

        # saving to file
        outFolder = os.path.join(config['experimentFolder'], instr, 'qc')
        if not os.path.isdir(outFolder):
            os.mkdir(outFolder)
        ncFilePath = os.path.join(outFolder, '{}.nc'.format(instr))

        # add script version information
        ds.attrs['git repo'] = r'https://github.com/MarliesA/EURECCA/tree/main/sedmex'
        ds.attrs['git hash'] = get_githash()

        # if nothing else, at least specify lossless zlib compression
        comp = dict(zlib=True, complevel=5)
        ds.encoding = {var: comp for var in ds.data_vars}
        for coord in list(ds.coords.keys()):
            ds.encoding[coord] = {'zlib': False, '_FillValue': None}

        ds2.to_netcdf(ncFilePath, encoding=ds2.encoding )

