# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xarray as xr
import os
import glob
from datetime import datetime
import yaml
from pathlib import Path
from vector import Vector
from solo import Solo
from ossi import Ossi
from profiler import Profiler
import sontek
from sedmex_info_loaders import load_instrument_information, add_positioning_info, get_githash

def load_solo_data(config):

    isxy = load_instrument_information(config['experimentFolder'])

    allSoloDataFiles = {
        'L2C2SOLO': ['20210923//raw//202439_20210919_2012_data.txt',
                      '20211004//raw//202439_20211004_1735_data.txt'],
        'L2C4SOLO': ['20210923//raw//202440_20210919_1930_data.txt',
                      '20211004//raw//202440_20211004_1634_data.txt',
                      '20211021//raw//202440_20211021_1534_data.txt'],
        'L2C10SOLO': ['20210919//raw//202441_20210919_1902_data.txt',
                       '20211004//raw//202441_20211004_1803_data.txt',
                       '20211021//raw//202441_20211021_1418_data.txt'],
        'L4C1SOLO': ['20210923//raw//202438_20210920_1819_data.txt',
                      '20211006//raw//202438_20211006_1828_data.txt',
                      '20211021//raw//202438_20211021_1515_data.txt']

    }
    for instrument in config['instruments']['solo']:
        print(instrument)

        dataFiles = allSoloDataFiles[instrument]

        dsList = []
        for file in dataFiles:
            dsList.append(
                Solo(instrument, config['experimentFolder'], file, isxy, sf=config['samplingFrequency']['solo'], jasave=False))

        ds = xr.merge(dsList)
        ds.attrs = dsList[0].attrs

        ds = add_positioning_info(ds, instrument, config['experimentFolder'])

        ncOutDir = os.path.join(config['experimentFolder'], instrument, 'raw_netcdf')
        if not os.path.isdir(ncOutDir):
            os.mkdir(ncOutDir)

        # add script version information
        ds.attrs['git repo'] = r'https://github.com/MarliesA/EURECCA/tree/main/sedmex'
        ds.attrs['git hash'] = get_githash()

        # if nothing else, at least specify lossless zlib compression
        comp = dict(zlib=True, complevel=5)
        ds.encoding = {var: comp for var in ds.data_vars}
        for coord in list(ds.coords.keys()):
            ds.encoding[coord] = {'zlib': False, '_FillValue': None}

        ds.to_netcdf(os.path.join(ncOutDir, instrument + '.nc'), encoding = ds.encoding)

    return

def load_ossi_data(config):

    isxy = load_instrument_information(config['experimentFolder'])

    for instr in config['instruments']['ossi']:
        ds = Ossi(instr,
                  config['experimentFolder'],
                  isxy=isxy,
                  sf=config['samplingFrequency']['ossi'])

        ds = add_positioning_info(ds, instr, config['experimentFolder'])

        fold = os.path.join(config['experimentFolder'], instr, 'raw_netcdf')
        if not os.path.isdir(fold):
            os.mkdir(fold)
        ncFilePath = os.path.join(fold, '{}.nc'.format(ds.name))

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
def vector_read_write_to_netcdf(instrument, experimentFolder, dataPath, isxy, tstart=None, tstop=None):
    '''
    Reads the raw data collected by a Nortek ADV into an xarray dataset. The function reads maximum of 1 day of data
    at a time. If the deployment was longer than one day, it splits the reading operation into several reading
    tasks and merges the result later.
    :param instrument:
    :param experimentFolder: root folder of the SEDMEX experiment
    :param dataPath: name of the raw data subfolder
    :param isxy: dataframe including the coordinates of the instrument and its serial number
    :param tstart: starting time
    :param tstop: stopping time
    :return: ds xarray dataset
    '''

    # first check whether there is actually data on the raw data files in the desired time window
    vec = Vector(
        name=instrument,
        dataFolder=os.path.join(experimentFolder, instrument, dataPath),
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

        ds = add_positioning_info(ds, instrument, experimentFolder)

        # add global attribute metadata
        ds.attrs = {
            'Conventions': 'CF-1.6',
            'name': '{}'.format(instrument),
            'instrument': '{}'.format(instrument),
            'instrument serial number': '{}'.format(isxy[instrument]['serial number']),
            'epsg': 28992,
            'x': isxy[instrument]['xRD'],
            'y': isxy[instrument]['yRD'],
            'time zone': 'UTC+2',
            'coordinate type': 'XYZ',
            'summary': 'SEDMEX field campaign',
            'contact person': 'Marlies van der Lugt',
            'emailadres': 'm.a.vanderlugt@tudelft.nl',
            'construction datetime': datetime.now().strftime("%d-%b-%Y (%H:%M:%S)"),
            'version comments': 'constructed with xarray'}

        # add script version information
        ds.attrs['git repo'] = r'https://github.com/MarliesA/EURECCA/tree/main/sedmex'
        ds.attrs['git hash'] = get_githash()

        # specify compression for all the variables to reduce file size
        # if nothing else, at least specify lossless zlib compression
        comp = dict(zlib=True, complevel=5)
        ds.encoding = {var: comp for var in ds.data_vars}
        for coord in list(ds.coords.keys()):
            ds.encoding[coord] = {'zlib': False, '_FillValue': None}

        # save to netCDF
        fold = os.path.join(experimentFolder, instrument, 'raw_netcdf')
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
                instrument, experimentFolder, dataPath,
                isxy=isxy,
                tstart=blockStartTime,
                tstop=blockStartTime + np.timedelta64(1, 'D'))
    return
def load_vector_data(config):

    isxy = load_instrument_information(config['experimentFolder'])

    allVectors = {
        'L1C1VEC': [r'raw\20210911-20211011',
                     r'raw\20211011-20211019'],
        'L2C3VEC': [r'raw\20210911-20211011',
                     r'raw\20211011-20211019'],
        'L3C1VEC': [r'raw\20210911-20211011',
                     r'raw\20211011-20211019'],
        'L5C1VEC': [r'raw\20210911-20211011',
                     r'raw\20211011-20211019'],
        'L6C1VEC': [r'raw\20210911-20211011',
                     r'raw\20211011-20211019'],
        'L2C2VEC': [r'raw\20210910-20210913\raw',
                     r'raw\20210913-20210919\raw',
                     r'raw\20210920-20211004\raw',
                     r'raw\20211005-20211018\raw'],
        'L2C4VEC': [r'raw\20210910-20210919\raw',
                     r'raw\20210920-20211004\raw',
                     r'raw\20211005-20211018\raw'],
        'L2C10VEC': [r'raw\20210910-20210919\raw',
                      r'raw\20210920-20211004\raw',
                      r'raw\20211005-20211019\raw'],
        'L4C1VEC': [r'raw\20210910-20210920\raw',
                     r'raw\20210921-20211006\raw',
                     r'raw\20211007-20211019\raw']
    }

    for instrument in config['instruments']['adv']['vector']:

        dataFolders = allVectors[instrument]
        print('reading in {}'.format(instrument))

        for dataFolder in dataFolders:
            print(dataFolder)
            vector_read_write_to_netcdf(instrument,
                                        config['experimentFolder'],
                                        dataFolder,
                                        isxy)
    return


def load_sontek_data(config):
    '''
    wrapper function for read_raw_data_file that reads all raw data files in folder
    :param infolder: path to raw data folder
    :param outfolder: path to folder to store a netcdf per raw data file.
    :param sf: float - sampling frequency
    :return: list of pandas DataFrames
    '''

    isxy = load_instrument_information(config['experimentFolder'])

    for instrument in config['instruments']['adv']['sontek']:
        infolder = os.path.join(config['experimentFolder'], instrument, 'raw')
        outfolder = os.path.join(config['experimentFolder'], instrument, 'raw_netcdf')
        # get list of all files in folder
        hd1files = sorted(glob.glob(os.path.join(infolder, '*.hd1')))
        ts1files = sorted(glob.glob(os.path.join(infolder, '*.ts1')))

        assert len(hd1files) == len(ts1files), 'filelist is incomplete: unequal amount of hd1 and ts1 files'

        nfiles = hd1files[-1].split('\\')[-1][-7:-4]

        # for each of these files, read data and save in netcdf format
        for hd1, ts1 in zip(hd1files, ts1files):
            assert hd1.split('\\')[-1][-5] == ts1.split('\\')[-1][-5], 'name of hd1 and ts1 file is not equal'

            rawdatafilenumber = hd1.split('\\')[-1][-7:-4]
            print('reading data ' + rawdatafilenumber + '/' + nfiles)

            df, Fs = sontek.read_raw_data_file(hd1, ts1)

            ds = sontek.cast_to_blocks_in_xarray(df, sf=config['samplingFrequency']['sontek'], blockWidth=1740)

            ds = add_positioning_info(ds, instrument, config['experimentFolder'])

            # add sampling frequency as variable
            ds['sf'] = config['samplingFrequency']['sontek']
            ds['sf'].attrs = {'units': 'Hz', 'long_name': 'sampling frequency'}

            # add global attribute metadata
            ds.attrs = {
                'Conventions': 'CF-1.6',
                'name': '{}'.format(instrument),
                'instrument': '{}'.format(instrument),
                'instrument serial number': '{}'.format(isxy[instrument]['serial number']),
                'epsg': 28992,
                'x': isxy[instrument]['xRD'],
                'y': isxy[instrument]['yRD'],
                'time zone': 'UTC+2',
                'coordinate type': 'XYZ',
                'summary': 'SEDMEX field campaign. Instrument is known to contain some internal clock drift. \
                Timing of the blocks is NOT synchronised to the other instruments. Pressure is present on the dataset,\
                 but only used for computing the directional distribution, \
                not the wave heights itself: pressure is known to drift too much.',
                'contact person': 'Marlies van der Lugt',
                'emailadres': 'm.a.vanderlugt@tudelft.nl',
                'construction datetime': datetime.now().strftime("%d-%b-%Y (%H:%M:%S)"),
                'version comments': 'constructed with xarray. '}

            # add script version information
            ds.attrs['git repo'] = r'https://github.com/MarliesA/EURECCA/tree/main/sedmex'
            ds.attrs['git hash'] = get_githash()

            # specify compression for all the variables to reduce file size
            comp = dict(zlib=True, complevel=5)
            ds.encoding = {var: comp for var in ds.data_vars}
            for coord in list(ds.coords.keys()):
                ds.encoding[coord] = {'zlib': False, '_FillValue': None}

            ds.to_netcdf(os.path.join(outfolder, rawdatafilenumber + '.nc'), encoding=ds.encoding)

    return


def load_ADCP_data(config):

    if 'L2C7ADCP' in config['instruments']['adcp']:
        instrument = 'L2C7ADCP'
        parts = {
            '1': {'tstart': '2021-09-11 00:00:00', 'tstop': '2021-09-22 08:00:00'},
            '2': {'tstart': '2021-09-22 08:00:00', 'tstop': '2021-10-03 16:00:00'},
            '3': {'tstart': '2021-10-03 16:00:00', 'tstop': '2021-10-15 00:00:00'},
            '4': {'tstart': '2021-10-15 00:00:00', 'tstop': '2021-10-19 00:00:00'},
        }

        for i in list(parts.keys()):
            dataFolder = os.path.join(config['experimentFolder'], instrument, r'raw\p' + i)
            conf = {'name': instrument, 'zb': -1.22, 'zi': -0.27, 'tstart': parts[i]['tstart'],
                      'tstop': parts[i]['tstop']}
            P1 = Profiler(conf['name'], dataFolder, zb=conf['zb'], zi=conf['zi'], tstart=conf['tstart'],
                          tstop=conf['tstop'])
            P1.load_all_data()
            ds = P1.get_dataset()
            ds['sf'] = config['samplingFrequency']['adcp']
            ds['sf'].attrs = {'units': 'Hz', 'long_name': 'sampling frequency'}


            # add global attribute metadata
            ds.attrs = {'Conventions': 'CF-1.6',
                        'title': '{}'.format('HR Profiler part ' + i),
                        'summary': 'SEDMEX field campaign, part ' + i,
                        'contact person': 'Marlies van der Lugt',
                        'emailadres': 'm.a.vanderlugt@tudelft.nl',
                        'version comments': 'constructed with xarray'}

            # add script version information
            ds.attrs['git repo'] = r'https://github.com/MarliesA/EURECCA/tree/main/sedmex'
            ds.attrs['git hash'] = get_githash()

            # specify compression for all the variables to reduce file size
            comp = dict(zlib=True, complevel=5)
            ds.encoding = {var: comp for var in ds.data_vars}
            for coord in list(ds.coords.keys()):
                ds.encoding[coord] = {'zlib': False, '_FillValue': None}

            # save to netCDF
            fold = os.path.join(config['experimentFolder'], instrument, 'raw_netcdf')
            if not os.path.isdir(fold):
                os.mkdir(fold)
            ncFilePath = os.path.join(fold, 'part' + i + '.nc')

            ds.to_netcdf(ncFilePath, encoding=ds.encoding)

    if 'L4C1ADCP' in config['instruments']['adcp']:
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
            dataFolder = os.path.join(config['experimentFolder'], instrument, r'raw\p' + i[-1])
            conf = {'name': instrument, 'zb': parts[i]['zb'], 'zi': parts[i]['zi'], 'tstart': parts[i]['tstart'],
                      'tstop': parts[i]['tstop']}
            P1 = Profiler(conf['name'], dataFolder, zb=conf['zb'], zi=conf['zi'], tstart=conf['tstart'],
                          tstop=conf['tstop'])
            P1.load_all_data()
            ds = P1.get_dataset()

            ds['sf'] = config['samplingFrequency']['adcp']
            ds['sf'].attrs = {'units': 'Hz', 'long_name': 'sampling frequency'}

            # add global attribute metadata
            ds.attrs = {'Conventions': 'CF-1.6',
                        'title': '{}'.format('HR Profiler part ' + i),
                        'summary': 'SEDMEX field campaign, part ' + i,
                        'contact person': 'Marlies van der Lugt',
                        'emailadres': 'm.a.vanderlugt@tudelft.nl',
                        'version comments': 'constructed with xarray'}

            # add script version information
            ds.attrs['git repo'] = r'https://github.com/MarliesA/EURECCA/tree/main/sedmex'
            ds.attrs['git hash'] = get_githash()

            # specify compression for all the variables to reduce file size
            comp = dict(zlib=True, complevel=5)
            ds.encoding = {var: comp for var in ds.data_vars}
            for coord in list(ds.coords.keys()):
                ds.encoding[coord] = {'zlib': False, '_FillValue': None}

            # save to netCDF
            fold = os.path.join(config['experimentFolder'], instrument, 'raw_netcdf')
            if not os.path.isdir(fold):
                os.mkdir(fold)
            ncFilePath = os.path.join(fold, 'part' + i + '.nc')

            ds.to_netcdf(ncFilePath, encoding=ds.encoding)

    return

if __name__ == "__main__":

    config = yaml.safe_load(Path('sedmex-processing.yml').read_text())

    # SOLO data
    load_solo_data(config)

    # ossi data
    load_ossi_data(config)

    # vector data
    load_vector_data(config)

    # sontek data
    load_sontek_data(config)

    # adcp data
    load_ADCP_data(config)
