# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 08:27:20 2021

@author: marliesvanderl
"""
import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def zi_solo(experimentFolder,instr):
    df = pd.read_excel(experimentFolder + r'\instrument_orientation.xlsx',
                       sheet_name='solo_zi',
                       header=0)
    df = df.set_index('t')
    return df[instr].to_xarray()

if __name__ == "__main__":
    experimentFolder = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20210908_campaign\instruments'
    rho = 1028
    g = 9.8

    dsCtd = xr.open_dataset(experimentFolder + r'\L2C1CTD\raw_netcdf\L2C1CTD_20210910.nc')

    #%% process solo's
    reftL2C2 = ['20210910 05:00',
        '20210919 15:00',
        '20210920 14:30',
        '20210926 05:30']
    reftL2C4 = ['20210910 05:00',
        '20210919 15:00',
        '20210920 14:30',
        '20210926 05:30',
        '20211004 14:00',
        '20211007 16:00',
        '20211011 06:00',
        '20211019 13:30']
    reftL2C10 = ['20210919 15:00',
        '20211019 13:30']
    reftL4C1 = ['20210919 15:00',
        '20210920 14:30',
        '20210926 05:30',
        '20211007 16:00',
        '20211011 06:00',
        '20211019 13:30']

    refdates = [pd.to_datetime(instr) for instr in [reftL2C2, reftL2C4, reftL2C10, reftL4C1]]
    instruments = ['L2C2SOLO', 'L2C4SOLO', 'L2C10SOLO', 'L4C1SOLO']
    refdates = dict(zip(instruments, refdates))

    pclist = []
    for instr in instruments[:-1]:
        dataFile = experimentFolder + r'//' + instr + r'\raw_netcdf' + r'//' + instr + '.nc'
        ds0 = xr.open_dataset(dataFile)

        ds0['zb'] = ds0.zb.interpolate_na('t', method='nearest', fill_value='extrapolate')
        ds0['h'] = ds0.h.interpolate_na('t', method='nearest', fill_value='extrapolate')

        zi = zi_solo(experimentFolder,instr)
        ds0['zi'] = zi.interp_like(ds0)

        # correct for the air pressure fluctuations
        pair = dsCtd.p.interp_like(ds0)

        # select those dates on which we know the measured pressure should be equal
        # 0 (i.e. when we are sure they are emerged) and linearly interpolate
        # the drift in reference level in between
        p2 = ds0.p - pair
        p_dry = p2.resample({'t': '15T'}, loffset='15T').mean().sel({'t': refdates[instr]})

        plt.figure()
        p_dry.plot(marker='o')
        plt.title('p-pair when dry ' + instr)

        p3 = p2 - p_dry.interp_like(ds0.p,method='linear').bfill(dim='t').ffill(dim='t')
        ds0['p'] = p3 + ds0.zi*rho*g

        # -----------------------------------------------------------------------------
        # reshape to one row per burst in data array
        pt = ds0.p.values
        nSamples = len(pt)
        dt = ds0.isel(t=1).t-ds0.isel(t=0).t
        sf = np.timedelta64(1, 's')/dt.values

        burstDuration = pd.Timedelta('600S')
        burstLength = int(burstDuration/dt)
        nBursts = int(np.floor( nSamples / burstLength ))

        pt = pt[:nBursts*burstLength]
        h = ds0.h[::burstLength]
        h = h[:nBursts]
        zi = ds0.zi[::burstLength]
        zi = zi[:nBursts]
        zb = ds0.zb[::burstLength]
        zb = zb[:nBursts]
        t = ds0.t[::burstLength]
        t = t[:nBursts]
        N = (ds0.t.values[:burstLength]-ds0.t.values[0] ) / np.timedelta64(1, 's')

        #--------------------------------------------------------------------------
        # cast into a 2D array
        ds = xr.Dataset(data_vars={},
                    coords={'t': t, 'N': N})

        # copy all data over into this new structure
        ds['p'] = (('t', 'N'), pt.reshape((nBursts, burstLength)))

        # remove all bursts where instrument fell dry
        ds['p'] = ds.p.where( ds.p.std(dim='N') > 70)
        ds['zi'] = zi
        ds['zb'] = zb
        ds['sf'] = sf


        #--------------------------------------------------------------------------
        ds['p'].attrs = {'units': 'Pa +NAP','long_name': 'pressure','comments': 'corrected for air pressure'}
        ds['zi'].attrs = {'units': 'm+NAP','long_name': 'zi'}
        ds['zb'].attrs = {'units': 'm+NAP','long_name': 'zb'}
        ds['sf'].attrs = {'units': 'Hz','long_name': 'sampling frequency'}
        ds.attrs = ds0.attrs
        ds.attrs['summary'] = 'SEDMEX field campaign, pressure corrected for air pressure and quality checked'
        ds['name'] = instr

        if not os.path.isdir(experimentFolder + '//' + instr + r'\QC'):
            os.mkdir(experimentFolder + '//' + instr + r'\QC' )
        ncFilePath = experimentFolder + '//' + instr + r'\QC\{}.nc'.format(instr)

        comp = dict(zlib=True, complevel=5)
        ds.encoding = {var: comp for var in ds.data_vars}
        ds.encoding['t'] = {'zlib': False, '_FillValue': None}
        ds.encoding['N'] = {'zlib': False, '_FillValue': None}
        ds.to_netcdf(ncFilePath, encoding=ds.encoding )



