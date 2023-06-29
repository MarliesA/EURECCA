# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 08:37:43 2021

@author: marliesvanderl
"""
import numpy as np
import pandas as pd
import xarray as xr
import os
from datetime import datetime
from vector import Vector
from solo import Solo
from ossi import Ossi
from profiler import Profiler
from sedmex_info_loaders import load_instrument_information, add_positioning_info

def load_solo_data(experimentFolder, isxy):
    allSolos = (
        ('L2C2SOLO', ['20210923//raw//202439_20210919_2012_data.txt',
                      '20211004//raw//202439_20211004_1735_data.txt']),
        ('L2C4SOLO', ['20210923//raw//202440_20210919_1930_data.txt',
                      '20211004//raw//202440_20211004_1634_data.txt',
                      '20211021//raw//202440_20211021_1534_data.txt']),
        ('L2C10SOLO', ['20210919//raw//202441_20210919_1902_data.txt',
                       '20211004//raw//202441_20211004_1803_data.txt',
                       '20211021//raw//202441_20211021_1418_data.txt']),
        ('L4C1SOLO', ['20210923//raw//202438_20210920_1819_data.txt',
                      '20211006//raw//202438_20211006_1828_data.txt',
                      '20211021//raw//202438_20211021_1515_data.txt'])
    )
    for solo in allSolos:
        instrumentName = solo[0]
        dataFiles = solo[1]
        print(instrumentName)

        dsList = []
        for file in dataFiles:
            dsList.append(
                Solo(instrumentName, experimentFolder, file, isxy, sf=8, jasave=False))

        ds = xr.merge(dsList)
        ds.attrs = dsList[0].attrs

        ds = add_positioning_info(ds, instrumentName, experimentFolder)

        ncOutDir = os.path.join(experimentFolder, instrumentName, 'raw_netcdf')
        if not os.path.isdir(ncOutDir):
            os.mkdir(ncOutDir)
        ds.to_netcdf(os.path.join(ncOutDir, instrumentName + '.nc'))

    return

def load_ossi_data(experimentFolder, isxy):
    for instr in ['L2C9OSSI', 'L2C8OSSI', 'L2C6OSSI', 'L1C2OSSI', 'L4C3OSSI', 'L5C2OSSI', 'L6C2OSSI']:
        ds = Ossi(instr,
                  experimentFolder,
                  isxy=isxy)
        ds = add_positioning_info(ds, instr, experimentFolder)

        fold = os.path.join(experimentFolder, instr, 'raw_netcdf')
        if not os.path.isdir(fold):
            os.mkdir(fold)
        ncFilePath = os.path.join(fold, '{}.nc'.format(ds.name))

        # if nothing else, at least specify lossless zlib compression
        comp = dict(zlib=True, complevel=5)
        ds.encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(ncFilePath, encoding=ds.encoding)

    return
def vector_read_write_to_netcdf(instrumentName, experimentFolder, dataPath, isxy, tstart=None, tstop=None):
    '''
    Reads the raw data collected by a Nortek ADV into an xarray dataset. The function reads maximum of 1 day of data
    at a time. If the deployment was longer than one day, it splits the reading operation into several reading
    tasks and merges the result later.
    :param instrumentName:
    :param experimentFolder: root folder of the SEDMEX experiment
    :param dataPath: name of the raw data subfolder
    :param isxy: dataframe including the coordinates of the instrument and its serial number
    :param tstart: starting time
    :param tstop: stopping time
    :return: ds xarray dataset
    '''

    # first check whether there is actually data on the raw data files in the desired time window
    vec = Vector(
        name=instrumentName,
        dataFolder=os.path.join(experimentFolder, instrumentName, dataPath),
        tstart=tstart,
        tstop=tstop)

    if vec.tstop <= vec.tstart:
        print('tstop is smaller than tstart')
        return

    # check whether there is more than 1 day of data on the file. If so, read
    # and write this data to netcdf in blocks of 1 day.
    if (vec.tstop - vec.tstart) <= pd.Timedelta(1, 'D'):
        print('{}'.format(vec.tstart))

        vec.read_raw_data()
        vec.cast_to_blocks_in_xarray()  # recast data from long list to burst blocks casted in an xarray

        # all data is collected in an xarray Dataset ds. We extract this from the
        # class instantiation and we can easily write it to netCDF
        ds = vec.ds

        ds = add_positioning_info(ds, instrumentName, experimentFolder)

        # add global attribute metadata
        ds.attrs = {
            'Conventions': 'CF-1.6',
            'name': '{}'.format(instrumentName),
            'instrument': '{}'.format(instrumentName),
            'instrument serial number': '{}'.format(isxy[instrumentName]['serial number']),
            'epsg': 28992,
            'x': isxy[instrumentName]['xRD'],
            'y': isxy[instrumentName]['yRD'],
            'time zone': 'UTC+2',
            'coordinate type': 'XYZ',
            'summary': 'SEDMEX field campaign',
            'contact person': 'Marlies van der Lugt',
            'emailadres': 'm.a.vanderlugt@tudelft.nl',
            'construction datetime': datetime.now().strftime("%d-%b-%Y (%H:%M:%S)"),
            'version': 'v1',
            'version comments': 'constructed with xarray'}

        # specify compression for all the variables to reduce file size
        comp = dict(zlib=True, complevel=5)
        ds.encoding = {var: comp for var in ds.data_vars}

        # save to netCDF
        fold = os.path.join(experimentFolder, instrumentName, 'raw_netcdf')
        if not os.path.isdir(fold):
            os.mkdir(fold)
        ncFilePath = os.path.join(fold, '{}_{}.nc'.format(
            ds.name,
            ds.t.isel(t=0).dt.strftime('%Y%m%d').values
        ))
        ds.to_netcdf(ncFilePath, encoding=ds.encoding)

    else:

        blockStartTimes = pd.date_range(vec.tstart.floor('D'), vec.tstop.ceil('D'), freq='1D')
        for blockStartTime in blockStartTimes:
            vector_read_write_to_netcdf(
                instrumentName, experimentFolder, dataPath,
                isxy=isxy,
                tstart=blockStartTime,
                tstop=blockStartTime + np.timedelta64(1, 'D'))
    return
def load_vector_data(experimentFolder, isxy):
    allVectors = (
        ('L1C1VEC', [r'raw\20210911-20211011',
                     r'raw\20211011-20211019']),
        ('L2C3VEC', [r'raw\20210911-20211011',
                     r'raw\20211011-20211019']),
        ('L3C1VEC', [r'raw\20210911-20211011',
                     r'raw\20211011-20211019']),
        ('L5C1VEC', [r'raw\20210911-20211011',
                     r'raw\20211011-20211019']),
        ('L6C1VEC', [r'raw\20210911-20211011',
                     r'raw\20211011-20211019']),
        ('L2C2VEC', [r'raw\20210910-20210913\raw',
                     r'raw\20210913-20210919\raw',
                     r'raw\20210920-20211004\raw',
                     r'raw\20211005-20211018\raw']),
        ('L2C4VEC', [r'raw\20210910-20210919\raw',
                     r'raw\20210920-20211004\raw',
                     r'raw\20211005-20211018\raw']),
        ('L2C10VEC', [r'raw\20210910-20210919\raw',
                      r'raw\20210920-20211004\raw',
                      r'raw\20211005-20211019\raw']),
        ('L4C1VEC', [r'raw\20210910-20210920\raw',
                     r'raw\20210921-20211006\raw',
                     r'raw\20211007-20211019\raw'])
    )

    for ivec in allVectors:
        instrumentName = ivec[0]
        dataFolders = ivec[1]
        print('reading in {}'.format(instrumentName))

        for dataFolder in dataFolders:
            print(instrumentName)
            vector_read_write_to_netcdf(instrumentName,
                                        experimentFolder,
                                        dataFolder,
                                        isxy)

    return
def load_ADCP_data(experimentFolder):

    instrument = 'L2C7ADCP'
    parts = {
        '1': {'tstart': '2021-09-11 00:00:00', 'tstop': '2021-09-22 08:00:00'},
        '2': {'tstart': '2021-09-22 08:00:00', 'tstop': '2021-10-03 16:00:00'},
        '3': {'tstart': '2021-10-03 16:00:00', 'tstop': '2021-10-15 00:00:00'},
        '4': {'tstart': '2021-10-15 00:00:00', 'tstop': '2021-10-19 00:00:00'},
    }

    for i in list(parts.keys()):
        dataFolder = os.path.join(experimentFolder, instrument, r'raw\p' + i)
        config = {'name': instrument, 'zb': -1.22, 'zi': -0.27, 'tstart': parts[i]['tstart'],
                  'tstop': parts[i]['tstop']}
        P1 = Profiler(config['name'], dataFolder, zb=config['zb'], zi=config['zi'], tstart=config['tstart'],
                      tstop=config['tstop'])
        P1.load_all_data()
        ds = P1.get_dataset()

        # add global attribute metadata
        ds.attrs = {'Conventions': 'CF-1.6',
                    'title': '{}'.format('HR Profiler part ' + i),
                    'summary': 'SEDMEX field campaign, part ' + i,
                    'contact person': 'Marlies van der Lugt',
                    'emailadres': 'm.a.vanderlugt@tudelft.nl',
                    'version': 'v1',
                    'version comments': 'constructed with xarray'}

        # specify compression for all the variables to reduce file size
        comp = dict(zlib=True, complevel=5)
        ds.encoding = {var: comp for var in ds.data_vars}

        # save to netCDF
        fold = os.path.join(experimentFolder, instrument, 'raw_netcdf')
        if not os.path.isdir(fold):
            os.mkdir(fold)
        ncFilePath = os.path.join(fold, 'part' + i + '.nc')

        ds.to_netcdf(ncFilePath, encoding=ds.encoding)

    # ADCP L4C1
    instrument = 'L4C1ADCP'
    parts = {
        'a1': {'tstart': '2021-09-10 00:00:00', 'tstop': '2021-09-19 18:00:00', 'zi': 0.07, 'zb': -0.70},
        'b1': {'tstart': '2021-09-19 18:00:00', 'tstop': '2021-09-20 18:00:00', 'zi': 0.07, 'zb': -0.65},
        '2': {'tstart': '2021-09-22 17:00:00', 'tstop': '2021-09-24 02:00:00', 'zi': 0.07, 'zb': -0.83},
        'a3': {'tstart': '2021-10-07 14:00:00', 'tstop': '2021-10-14 00:00:00', 'zi': 0.05, 'zb': -0.70},
        'b3': {'tstart': '2021-10-14 00:00:00', 'tstop': '2021-10-17 00:00:00', 'zi': 0.05, 'zb': -0.79},
        'c3': {'tstart': '2021-10-17 00:00:00', 'tstop': '2021-10-19 18:00:00', 'zi': 0.05, 'zb': -0.73},
    }

    for i in list(parts.keys()):
        dataFolder = os.path.join(experimentFolder, instrument, r'raw\p' + i)
        config = {'name': instrument, 'zb': parts[i]['zb'], 'zi': parts[i]['zi'], 'tstart': parts[i]['tstart'],
                  'tstop': parts[i]['tstop']}
        P1 = Profiler(config['name'], dataFolder, zb=config['zb'], zi=config['zi'], tstart=config['tstart'],
                      tstop=config['tstop'])
        P1.load_all_data()
        ds = P1.get_dataset()

        # add global attribute metadata
        ds.attrs = {'Conventions': 'CF-1.6',
                    'title': '{}'.format('HR Profiler part ' + i),
                    'summary': 'SEDMEX field campaign, part ' + i,
                    'contact person': 'Marlies van der Lugt',
                    'emailadres': 'm.a.vanderlugt@tudelft.nl',
                    'version': 'v1',
                    'version comments': 'constructed with xarray'}

        # specify compression for all the variables to reduce file size
        comp = dict(zlib=True, complevel=5)
        ds.encoding = {var: comp for var in ds.data_vars}

        # save to netCDF
        fold = os.path.join(experimentFolder, instrument, 'raw_netcdf')
        if not os.path.isdir(fold):
            os.mkdir(fold)
        ncFilePath = os.path.join(fold, 'part' + i + '.nc')

        ds.to_netcdf(ncFilePath, encoding=ds.encoding)

    return

if __name__ == "__main__":
    experimentFolder = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20210908_campaign\instruments'
    isxy = load_instrument_information(experimentFolder)

    # SOLO data
    load_solo_data(experimentFolder, isxy)

    # ossi data
    load_ossi_data(experimentFolder, isxy)

    # vector data
    load_vector_data(experimentFolder, isxy)

    # adcp data
    load_ADCP_data(experimentFolder)
