#%%
import xarray as xr
import pandas as pd
import numpy as np
import pickle
import matplotlib 
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm

dataPath = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\ublab-3c\dep1'
filez = glob.glob(os.path.join(dataPath, 'raw', 'a_1_*.pickle'))
nparts = len(filez)
print(nparts)
#%%
dsList = []
for ipart in tqdm(range(nparts)):
    with open(os.path.join(dataPath, 'raw', r"time_1_{}.pickle".format(ipart)), "rb") as input_file:
        time = pd.to_datetime( pickle.load(input_file) )

    ds = xr.Dataset(data_vars={},
                    coords={'z':-np.arange(100)/5,
                            'time':time})

    with open(os.path.join(dataPath, 'raw', r"vel_1_{}.pickle".format(ipart)), "rb") as input_file:
        ds['v1'] = (('z', 'time'), np.array(pickle.load(input_file)).T)
    with open(os.path.join(dataPath, 'raw', r"vel_2_{}.pickle".format(ipart)), "rb") as input_file:
        ds['v2'] = (('z', 'time'), np.array(pickle.load(input_file)).T)
    with open(os.path.join(dataPath, 'raw', r"vel_3_{}.pickle".format(ipart)), "rb") as input_file:
        ds['v3'] = (('z', 'time'), np.array(pickle.load(input_file)).T)
    with open(os.path.join(dataPath, 'raw', r"vel_4_{}.pickle".format(ipart)), "rb") as input_file:
        ds['v4'] = (('z', 'time'), np.array(pickle.load(input_file)).T)

    with open(os.path.join(dataPath, 'raw', r"a_1_{}.pickle".format(ipart)), "rb") as input_file:
        ds['a1'] = (('z', 'time'), np.array(pickle.load(input_file)).T)
    with open(os.path.join(dataPath, 'raw', r"a_2_{}.pickle".format(ipart)), "rb") as input_file:
        ds['a2'] = (('z', 'time'), np.array(pickle.load(input_file)).T)
    with open(os.path.join(dataPath, 'raw', r"a_3_{}.pickle".format(ipart)), "rb") as input_file:
        ds['a3'] = (('z', 'time'), np.array(pickle.load(input_file)).T)
    with open(os.path.join(dataPath, 'raw', r"a_4_{}.pickle".format(ipart)), "rb") as input_file:
        ds['a4'] = (('z', 'time'), np.array(pickle.load(input_file)).T)

    with open(os.path.join(dataPath, 'raw', r"snr_1_{}.pickle".format(ipart)), "rb") as input_file:
        ds['snr1'] = (('z', 'time'), np.array(pickle.load(input_file)).T)
    with open(os.path.join(dataPath, 'raw', r"snr_2_{}.pickle".format(ipart)), "rb") as input_file:
        ds['snr2'] = (('z', 'time'), np.array(pickle.load(input_file)).T)
    with open(os.path.join(dataPath, 'raw', r"snr_3_{}.pickle".format(ipart)), "rb") as input_file:
        ds['snr3'] = (('z', 'time'), np.array(pickle.load(input_file)).T)
    with open(os.path.join(dataPath, 'raw', r"snr_4_{}.pickle".format(ipart)), "rb") as input_file:
        ds['snr4'] = (('z', 'time'), np.array(pickle.load(input_file)).T)

    ds['v1'].attrs = {'units':'m/s', 'long_name':'u-1 (cross)'}
    ds['v2'].attrs = {'units':'m/s', 'long_name':'u-2 (cross)'}
    ds['v3'].attrs = {'units':'m/s', 'long_name':'u-3 (along)'}
    ds['v4'].attrs = {'units':'m/s', 'long_name':'u-4 (along)'}
    ds['a1'].attrs = {'units': 'counts', 'long_name': 'intensity 1'}
    ds['a2'].attrs = {'units': 'counts', 'long_name': 'intensity 2'}
    ds['a3'].attrs = {'units': 'counts', 'long_name': 'intensity 3'}
    ds['a4'].attrs = {'units': 'counts', 'long_name': 'intensity 4'}
    ds['snr1'].attrs = {'units': 'db', 'long_name': 'snr 1'}
    ds['snr2'].attrs = {'units': 'db', 'long_name': 'snr 2'}
    ds['snr3'].attrs = {'units': 'db', 'long_name': 'snr 3'}
    ds['snr4'].attrs = {'units': 'db', 'long_name': 'snr 4'}
    ds['time'].attrs = {'long_name':'time'}
    ds['z'].attrs = {'units':'cm', 'long_name':'depth'}

    # specify compression for all the variables to reduce file size
    comp = dict(zlib=True, complevel=5)
    ds.encoding = {var: comp for var in ds.data_vars}
    for coord in list(ds.coords.keys()):
        ds.encoding[coord] = {'zlib': False, '_FillValue': None}

    ds.to_netcdf(os.path.join(dataPath, 'raw_netcdf','dep1_{}.nc'.format(ipart)))

# on dep1, a4 is one too short in comparison with the time array of vel1 on last part