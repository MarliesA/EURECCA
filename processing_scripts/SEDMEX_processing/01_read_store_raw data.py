# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 08:37:43 2021

@author: marliesvanderl
"""
import sys
import os
sys.path.append(r'c:\checkouts\python\PHD\modules')
from datetime import datetime
from vector import Vector
from ctd import ctdDiver
from solo import Solo
from ossi import Ossi
import numpy as np
import pandas as pd
import xarray as xr
import subprocess

def get_githash():
    '''
    Get the revision number of the script to be printed in the metadata of the netcdfs
    :return: githash
    '''
    try:
        githash = subprocess.check_output(["git", "describe", "--always"], cwd=os.path.dirname(os.path.abspath(__file__))).strip().decode()
    except:
        print('Mind you: script not run from repository, or GIT not installed')
        githash = '-'
    return githash

def load_instrument_info(experimentFolder, instrumentName):
    '''
    get instrument bedlevel, instrument height above bed and orientation as a timeseries
    from the overview .csv and .xlsx files
    cast data in dataframe for easy merge on the dataset
    :param experimentFolder: Folder where overview .csv and .xlsx are sitting
    :param instrumentName: specific instrument to be read from file
    :return: Pandas DataFrame
    '''

    df = pd.read_excel(experimentFolder + r'\gps_bed_levels.xlsx',
                       sheet_name = 'time_stamps',
                       header=0).T
    df.columns = df.iloc[0]
    df.drop(df.iloc[0].name, inplace=True)
    df.index = pd.to_datetime(df.index)
    df.columns = [s.strip() for s in df.columns]

    zb = pd.read_excel(experimentFolder + r'\gps_bed_levels.xlsx',
                       sheet_name = 'bed_level',
                       header=0,
                       usecols='A:Y').T
    zb.columns = zb.iloc[0]
    zb.drop(zb.iloc[0].name,inplace=True)
    zb.index = pd.to_datetime(zb.index)
    zb.columns = [s.strip() for s in zb.columns]

    zih = pd.read_excel(experimentFolder + r'\gps_bed_levels.xlsx',
                       sheet_name = 'height_above_bed',
                       header=0,
                       usecols='A:Y').T
    zih.columns = zih.iloc[0]
    zih.drop(zih.iloc[0].name,inplace=True)
    zih.index = pd.to_datetime(zb.index)
    zih.columns = [s.strip() for s in zih.columns]

    io = pd.read_excel(experimentFolder + r'\gps_bed_levels.xlsx',
                       sheet_name = 'instrument_orientation',
                       header=0,
                       usecols='A:AO',
                       nrows=14).T
    io.columns = io.iloc[0]
    io.drop(io.iloc[0].name,inplace=True)
    io.index = pd.to_datetime(io.index)
    io.columns = [s.strip() for s in io.columns]

    if 'VEC' in instrumentName:
        presName = instrumentName[:-3]+'DRUK'
        positioning = pd.DataFrame({'zb': zb[instrumentName].dropna().astype(float),
                   'h': zih[instrumentName].dropna().astype(float),
                   'hpres': zih[presName].dropna().astype(float),
                   'io': io[instrumentName].astype(float)})
    else:
        positioning = pd.DataFrame({'zb': zb[instrumentName].dropna().astype(float),
                   'h': zih[instrumentName].dropna().astype(float)})       
        
    return positioning

def add_positioning_info(ds, instrumentName):
    '''
    adds positioning info to an existing instruments' dataset and makes sure there is info on every day
    :param ds: xarray dataset with observations on time axis 't'
    :param instrumentName: the name of the instrument that positioning must be added to
    :return: updated ds
    '''
    
    positioning = load_instrument_info(experimentFolder, instrumentName)

    pos = xr.Dataset.from_dataframe(positioning)
    pos = pos.rename({'index': 't'})
    pos = pos.interpolate_na(dim='t', method='nearest', fill_value="extrapolate")

    # slice only the section that we have observations on
    pos = pos.resample({'t': '1H'}).interpolate('nearest')
    pos = pos.sel(t=slice(ds.t.min(), ds.t.max().dt.ceil(freq='1H')))

    # bring to same time axis as observations
    pos = pos.resample({'t':'1800S'}).interpolate('nearest')
    pos = pos.interpolate_na(dim='t', method='nearest', fill_value="extrapolate")
    # merge and make sure no extra dates are added
    ds = ds.merge(pos.interp_like(ds))

    ds['zb'].attrs = {'units': 'm+NAP','long_name':'bed level, neg down'}
    ds['h'].attrs = {'units': 'cm','long_name':'instrument height above bed, neg down'}
    if 'VEC' in instrumentName:
        ds['hpres'].attrs = {'units': 'cm','long_name':'pressure sensor height above bed, neg down'}
        ds['io'].attrs= {'units':'deg','long_name':'angle x-dir with north clockwise'}

    return ds

def load_instrument_information(experimentFolder):
    # load all instruments' serial numbers and x,y location
    isxy = pd.read_excel(experimentFolder + r'\gps_bed_levels.xlsx',
                         sheet_name='main',
                         header=1,
                         usecols='A:D').T
    isxy.columns = isxy.iloc[0]
    isxy.drop(isxy.iloc[0].name, inplace=True)
    isxy.columns = [s.strip() for s in isxy.columns]
    return isxy

def load_ctd_data(experimentFolder, isxy):
    allCTDs = (
        ('L2C1CTD', 'C1-air-TUD1_211021140929_X1405.MON'),
        ('L1C2CTD', 'L1b-ctd-TUD2_211021140231_X1364.MON'),
        ('L6C2CTD', 'L6b-ctd-TUD3_211021135804_X1384.MON'))

    ds = ctdDiver(allCTDs[0][0], experimentFolder, allCTDs[0][1], isxy)
    ds1 = ctdDiver(allCTDs[1][0], experimentFolder, allCTDs[1][1], isxy)
    ds2 = ctdDiver(allCTDs[2][0], experimentFolder, allCTDs[2][1], isxy)

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

        positioning = load_instrument_info(experimentFolder, instrumentName)

        dsList = []
        for file in dataFiles:
            dsList.append(
                Solo(instrumentName, experimentFolder, file, isxy, sf=8, jasave=False))

        ds = xr.merge(dsList)
        ds.attrs = dsList[0].attrs

        ds = add_positioning_info(ds, instrumentName)

        ncOutDir = os.path.join(experimentFolder, instrumentName, 'raw_netcdf')
        if not os.path.isdir(ncOutDir):
            os.mkdir(ncOutDir)
        ds.to_netcdf(os.path.join(ncOutDir, instrumentName + '.nc'))

def load_ossi_data(experimentFolder, isxy):
    for instr in ['L2C9OSSI', 'L2C8OSSI', 'L2C6OSSI', 'L1C2OSSI', 'L4C3OSSI', 'L5C2OSSI', 'L6C2OSSI']:
        ds = Ossi(instr,
                  experimentFolder,
                  isxy=isxy)
        ds = add_positioning_info(ds, instr)

        fold = os.path.join(experimentFolder, instr, 'raw_netcdf')
        if not os.path.isdir(fold):
            os.mkdir(fold)
        ncFilePath = os.path.join(fold, '{}.nc'.format(ds.name))

        # if nothing else, at least specify lossless zlib compression
        comp = dict(zlib=True, complevel=5)
        ds.encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(ncFilePath, encoding=ds.encoding)

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

        ds = add_positioning_info(ds, instrumentName)

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

        positioning = load_instrument_info(experimentFolder, instrumentName)

        for dataFolder in dataFolders:
            print(instrumentName)
            vector_read_write_to_netcdf(instrumentName,
                                        experimentFolder,
                                        dataFolder,
                                        isxy)


if __name__ == "__main__":
    experimentFolder = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20210908_campaign\instruments'
    isxy = load_instrument_information(experimentFolder)

    # CTD data
    load_ctd_data(experimentFolder, isxy)

    # SOLO data
    load_solo_data(experimentFolder, isxy)

    # ossi data
    load_ossi_data(experimentFolder, isxy)

    # vector data
    load_vector_data(experimentFolder, isxy)

