# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 08:27:20 2021

@author: marliesvanderl
"""
import os
import numpy as np
import xarray as xr
import pandas as pd

def get_zi(experimentFolder,instr):
    df = pd.read_excel(experimentFolder + r'\instrument_orientation.xlsx',
                       sheet_name = 'solo_zi',
                       header=0)
    df = df.set_index('t')
    return df[instr].to_xarray()

if __name__ == "__main__":
    experimentFolder = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20210908_campaign\instruments'
    rho = 1028
    g = 9.8
    instruments = ['L2C9OSSI', 'L2C8OSSI', 'L2C6OSSI', 'L1C2OSSI', 'L4C3OSSI', 'L5C2OSSI', 'L6C2OSSI']

    # air pressure
    dsCtd = xr.open_dataset(experimentFolder + r'\L2C1CTD\raw_netcdf\L2C1CTD_20210910.nc')

    pclist = []
    for instr in instruments:
        print(instr)
        dataFile = experimentFolder + r'//' + instr + r'\raw_netcdf' + r'//' + instr + '.nc'
        ds = xr.open_dataset(dataFile)

        ds['zb'] = ds.zb.interpolate_na('t', method='nearest', fill_value='extrapolate')
        ds['h'] = ds.h.interpolate_na('t', method='nearest', fill_value='extrapolate')

        # make sure that we have nan's on moments with missing data,
        # because we rely on index number for the timing
        ds2 = ds.resample(t='100ms').nearest(tolerance='100ms')

        p = ds2.p.values
        zb = ds2.zb.values
        h = ds2.h.values
        t = ds2.t.values

        burstDuration = 600  # seconds
        sf = ds.sf
        blockLength = int(burstDuration * sf)

        # make a row per burst, and only full bursts are included
        NB = int(np.floor(len(p)/blockLength))
        p = p[0:int(NB*blockLength)]
        t = t[0:int(NB*blockLength)]

        p2 = p.reshape(NB,blockLength)

        # check these two are the same
        t2 = t.reshape(NB,blockLength)
        blockStartTimes = t[0::blockLength]

        ds2 = xr.Dataset(
                   data_vars=dict(
                       sf=sf,
                       p=(['t', 'N'], p2)
                       ),
                   coords=dict(t=blockStartTimes,
                             N=list(np.arange(0,burstDuration*sf)/sf)
                   )
        )

        # add these burst averaged variables back on the dataset
        ds2['h'] = ds.h.interp_like(ds2.t)
        ds2['zb'] = ds.zb.interp_like(ds2.t)

        # add instrument height data
        zi = get_zi(experimentFolder,instr)
        ds2['zi'] = zi.interp_like(ds2)

        # correct for the air pressure fluctuations
        # we assume the minimal pressure on ds is more or less emerged
        pair = dsCtd.p.interp_like(ds)
        p3 = ds2.p - ds2.p.min() - pair + pair.min()

        ds2['p'] = p3 + ds2.zi*rho*g

        #fix offset with respect to the solo at L2C10:
        #these were identified manually by finding the average discrepenacy over all meas
        if instr=='L2C9OSSI':
            ds2['p'] = ds2.p + rho*g*0.58
        elif instr=='L2C8OSSI':
            ds2['p'] = ds2.p + rho*g*0.31
        elif instr=='L2C6OSSI':
            ds2['p'] = ds2.p + rho*g*0.30


        ds2['p'].attrs = {'units': 'Pa +NAP', 'long_name': 'pressure', 'comments': 'corrected for drift air pressure'}

        ds2.attrs = ds.attrs
        ds2.attrs['summary'] = 'SEDMEX field campaign, pressure corrected for air pressure. There are inconsistencies between de mean pressure measured with ossis and solos.'

        # saving to file
        if not os.path.isdir(experimentFolder + '//' +  instr + r'\qc'):
            os.mkdir(experimentFolder + '//' +  instr + r'\qc' )
        ncFilePath = experimentFolder + '//' +  instr + r'\qc\{}.nc'.format(instr)

        # if nothing else, at least specify lossless zlib compression
        comp = dict(zlib=True, complevel=5)
        ds2.encoding = {var: comp for var in ds2.data_vars}
        ds.encoding['t'] = {'zlib': False, '_FillValue': None}
        ds.encoding['N'] = {'zlib': False, '_FillValue': None}
        ds2.to_netcdf(ncFilePath, encoding=ds2.encoding )

