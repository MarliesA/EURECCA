import os
import pandas as pd
import xarray as xr
import subprocess

def load_positioning_info(experimentFolder, instrumentName):
    '''
    get instrument bedlevel, instrument height above bed and orientation as a timeseries
    from the overview and .xlsx files
    cast data in dataframe for easy merge on the dataset
    :param experimentFolder: Folder where overview .csv and .xlsx are sitting
    :param instrumentName: specific instrument to be read from file
    :return: Pandas DataFrame
    '''

    df = pd.read_excel(experimentFolder + r'\gps_bed_levels.xlsx',
                       sheet_name='time_stamps',
                       header=0).T
    df.columns = df.iloc[0]
    df.drop(df.iloc[0].name, inplace=True)
    df.index = pd.to_datetime(df.index)
    df.columns = [s.strip() for s in df.columns]

    zb = pd.read_excel(experimentFolder + r'\gps_bed_levels.xlsx',
                       sheet_name='bed_level',
                       header=0,
                       usecols='A:Y').T
    zb.columns = zb.iloc[0]
    zb.drop(zb.iloc[0].name, inplace=True)
    zb.index = pd.to_datetime(zb.index)
    zb.columns = [s.strip() for s in zb.columns]

    zih = pd.read_excel(experimentFolder + r'\gps_bed_levels.xlsx',
                        sheet_name='height_above_bed',
                        header=0,
                        usecols='A:Y').T
    zih.columns = zih.iloc[0]
    zih.drop(zih.iloc[0].name, inplace=True)
    zih.index = pd.to_datetime(zb.index)
    zih.columns = [s.strip() for s in zih.columns]

    io = pd.read_excel(experimentFolder + r'\gps_bed_levels.xlsx',
                       sheet_name='instrument_orientation',
                       header=0,
                       usecols='A:AO',
                       nrows=14).T
    io.columns = io.iloc[0]
    io.drop(io.iloc[0].name, inplace=True)
    io.index = pd.to_datetime(io.index)
    io.columns = [s.strip() for s in io.columns]

    if 'VEC' in instrumentName:
        presName = instrumentName[:-3] + 'DRUK'
        positioning = pd.DataFrame({'zb': zb[instrumentName].dropna().astype(float),
                                    'h': zih[instrumentName].dropna().astype(float),
                                    'hpres': zih[presName].dropna().astype(float),
                                    'io': io[instrumentName].astype(float)})
    else:
        positioning = pd.DataFrame({'zb': zb[instrumentName].dropna().astype(float),
                                    'h': zih[instrumentName].dropna().astype(float)})

    return positioning


def add_positioning_info(ds, instrumentName, experimentFolder):
    '''
    adds positioning info to an existing instruments' xarray dataset and makes sure
    there is (interpolated) info on every day
    :param ds: xarray dataset with observations on time axis 't'
    :param instrumentName: the name of the instrument that positioning must be added to
    :return: updated ds
    '''

    positioning = load_positioning_info(experimentFolder, instrumentName)

    pos = xr.Dataset.from_dataframe(positioning)
    pos = pos.rename({'index': 't'})
    pos = pos.interpolate_na(dim='t', method='nearest', fill_value="extrapolate")

    # slice only the section that we have observations on
    pos = pos.resample({'t': '1H'}).interpolate('nearest')
    pos = pos.sel(t=slice(ds.t.min(), ds.t.max().dt.ceil(freq='1H')))

    # bring to same time axis as observations
    pos = pos.resample({'t': '1800S'}).interpolate('nearest')
    pos = pos.interpolate_na(dim='t', method='nearest', fill_value="extrapolate")
    # merge and make sure no extra dates are added
    ds = ds.merge(pos.interp_like(ds))

    ds['zb'].attrs = {'units': 'm+NAP', 'long_name': 'bed level, neg down'}
    ds['h'].attrs = {'units': 'cm', 'long_name': 'instrument height above bed, neg down'}
    if 'VEC' in instrumentName:
        ds['hpres'].attrs = {'units': 'cm', 'long_name': 'pressure sensor height above bed, neg down'}
        ds['io'].attrs = {'units': 'deg', 'long_name': 'angle x-dir with north clockwise'}

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

