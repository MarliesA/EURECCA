# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 14:33:55 2022

@author: marliesvanderl
"""
import pandas as pd

def read_oudeschild_zs_data(filein, fileout):
    """
    :param filein: csv file with RWS waterinfo water level observations
    :param fileout: nc file with just the water level info at oudeschild
    :return:
    """
    dat = pd.read_csv(filein,
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
    ds.to_netcdf(fileout)

if __name__ == '__main__':
    # executed as script
    filein = r'c:\Users\marliesvanderl\phddata\waterinfo\20220106_012_oudeschild.csv'
    fileout = r'c:\Users\marliesvanderl\phddata\waterinfo\SEDMEX_zs_oudeschild.nc'
    read_oudeschild_zs_data(filein, fileout)

