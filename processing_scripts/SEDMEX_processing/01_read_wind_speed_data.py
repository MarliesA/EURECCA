# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import xarray as xr

dataPath = r'c:\Users\marliesvanderl\phddata\weerstationHogeBerg//'

file = dataPath + r'sept-21.csv'
dat = pd.read_csv(file)
dat.columns = [key.strip() for key in list(dat.keys())]
dat['t'] = pd.to_datetime( dat[['year', 'month', 'day', 'hour','minute']])
dat = dat.set_index('t')
dat['u10'] = 2*dat['windspeed'] #I don't know why, but there is a factor 2 missing!
dat['u10dir'] = dat['direction']
dat['baro'] = dat['barometer']
df = dat[['u10','u10dir','temperature','baro']]
ds = df.to_xarray()
ds = ds.resample(t='10T').mean()
ds1 = ds.sel(t=slice('20210908','20210926'))

file = dataPath + r'hoge_berg.csv'
dat = pd.read_csv(file,sep=';')
dat.columns = [key.strip() for key in list(dat.keys())]
dat['t'] = pd.to_datetime(dat.datetime)
dat = dat.set_index('t')
dat['u10'] = dat['Windsnelheid [km/u]']
dat['u10dir'] = dat['windrichting [uit richting]']
dat['temperature']= dat['Temperatuur [deg C]']
dat['baro'] = dat['Barometer [hPa]']*100
dat = dat.drop_duplicates(subset=['datetime'])
df = dat[['u10','u10dir','temperature','baro']]
ds = df.to_xarray()
ds = ds.resample(t='10T').mean()
ds2 = ds.sel(t=slice('20210927','20211020'))

ds = xr.merge([ds1,ds2])
ds['u10dir'] = ds.u10dir.where(ds.u10dir>-500)
ds['u10'] = ds.u10/3.6
ds['u10'].attrs = {'units':'m/s','long_name':'wind speed'}
ds['u10dir'].attrs = {'units':'deg','long_name':'wind dir','comments':'coming from, clockwise North'}
ds['temperature'].attrs = {'units':'deg C','long_name':'temperature'}
ds['baro'].attrs = {'units':'Pa','long_name':'air pressure'}

ds.to_netcdf(dataPath + 'tailored/wind.nc')










