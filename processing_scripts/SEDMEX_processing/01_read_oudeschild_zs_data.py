# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 14:33:55 2022

@author: marliesvanderl
"""
import pandas as pd

dat = pd.read_csv(r'c:\Users\marliesvanderl\phddata\waterinfo\20220106_012_oudeschild.csv',
                  sep=';',
                  usecols=['WAARNEMINGDATUM',
                                   'WAARNEMINGTIJD (MET/CET)',
                                   'NUMERIEKEWAARDE'])
dat['t'] = pd.to_datetime(dat['WAARNEMINGDATUM'] + ' ' + dat['WAARNEMINGTIJD (MET/CET)'], format = '%d-%m-%Y %H:%M:%S')
dat['t'] = dat['t'] + pd.Timedelta('1H')
dat = dat.set_index('t')
dat['zs'] = dat['NUMERIEKEWAARDE']/100

ds = dat[['zs']].to_xarray()
ds.zs.plot()
ds['zs'].attrs = {'units':'m+NAP','long_name':'water level'}
ds.attrs = {'source':'waterinfo',
               'station':'Oudeschild',
               'time zone':'UTC+2, Dutch summer time'}
ds.to_netcdf(r'c:\Users\marliesvanderl\phddata\waterinfo\SEDMEX_zs_oudeschild.nc')