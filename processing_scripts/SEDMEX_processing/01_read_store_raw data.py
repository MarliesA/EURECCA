# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 08:37:43 2021

@author: marliesvanderl
"""
import sys
import os
# dirname = os.path.dirname(__file__)
# filename = os.path.join(dirname, '../../modules')
# sys.path.append(filename)
sys.path.append(r'C:\Users\marliesvanderl\phd\analysis\scripts\private\modules')
from datetime import datetime
from vector import Vector
from ctd import ctdDiver
from solo import Solo
from ossi import Ossi
import numpy as np
import pandas as pd
import xarray as xr
import pdb

# experimentFolder = r'u:\EURECCA\fieldvisits\20210908_campaign\instruments'
experimentFolder = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20210908_campaign\instruments'

#%% load_instrument_info(experimentFolder)       
def load_instrument_info(experimentFolder, instrumentName):
    # read orientation information from file
    orien = pd.read_csv(experimentFolder + r'\instrument_orientation.csv')
    orien.set_index('date',inplace=True)
    
    #instrument position and orientation
    #------------------------------------
    #load time stamps of observations
    df = pd.read_excel(experimentFolder + r'\gps_bed_levels.xlsx',
                       sheet_name = 'time_stamps',
                       header=0).T
    df.columns = df.iloc[0]
    df.drop(df.iloc[0].name,inplace=True)
    df.index = pd.to_datetime(df.index)
    df.columns = [s.strip() for s in df.columns]
    
    #load bed levels
    zb = pd.read_excel(experimentFolder + r'\gps_bed_levels.xlsx',
                       sheet_name = 'bed_level',
                       header=0,
                       usecols='A:Y').T
    zb.columns = zb.iloc[0]
    zb.drop(zb.iloc[0].name,inplace=True)
    zb.index = pd.to_datetime(zb.index)
    zb.columns = [s.strip() for s in zb.columns]
    
    #load instrument height above bed
    zih = pd.read_excel(experimentFolder + r'\gps_bed_levels.xlsx',
                       sheet_name = 'height_above_bed',
                       header=0,
                       usecols='A:Y').T
    zih.columns = zih.iloc[0]
    zih.drop(zih.iloc[0].name,inplace=True)
    zih.index = pd.to_datetime(zb.index)
    zih.columns = [s.strip() for s in zih.columns]
    
    #load instrument orientation
    io = pd.read_excel(experimentFolder + r'\gps_bed_levels.xlsx',
                       sheet_name = 'instrument_orientation',
                       header=0,
                       usecols = 'A:AO',
                       nrows=14).T
    io.columns = io.iloc[0]
    io.drop(io.iloc[0].name,inplace=True)
    io.index = pd.to_datetime(io.index)
    io.columns = [s.strip() for s in io.columns]
    
    # #instrument height NAP
    # zi = (zb+zih/100).astype(float).round(decimals=2)

    #cast positioning data in dataframe for easy merge on the dataset
    #for now, just use the approximate time stamps that are already in the index
    # pdb.set_trace()
    if 'VEC' in instrumentName:
        presName = instrumentName[:-3]+'DRUK'
        positioning = pd.DataFrame({'zb': zb[instrumentName].dropna().astype(float),
                   'h': zih[instrumentName].dropna().astype(float),
                   'hpres':zih[presName].dropna().astype(float),
                   'io': io[instrumentName].astype(float)})
    else:
        positioning = pd.DataFrame({'zb': zb[instrumentName].dropna().astype(float),
                   'h': zih[instrumentName].dropna().astype(float)})       
        
    return positioning

def add_positioning_info(ds,instrumentName):
    
        positioning = load_instrument_info(experimentFolder, instrumentName)
        #add positioning infoand make sure there is info on every day 
        pos = xr.Dataset.from_dataframe(positioning)
        pos = pos.rename({'index':'t'}) 
        pos = pos.interpolate_na(dim = 't',method='nearest',fill_value="extrapolate")
        
        #slice only the section that we have observations on
        pos = pos.resample({'t':'1H'}).interpolate('nearest')               
        pos = pos.sel(t=slice(ds.t.min(),ds.t.max()))

        #bring to same time axis as observations 
        pos = pos.resample({'t':'1800S'}).interpolate('nearest')
        pos = pos.interpolate_na(dim = 't',method='nearest',fill_value="extrapolate")
   
        ds = ds.merge(pos)  

        ds['zb'].attrs = {'units': 'm+NAP','long_name':'bed level, neg down'} 
        ds['h'].attrs = {'units': 'cm','long_name':'instrument height above bed, neg down'} 
        if 'VEC' in instrumentName:
            ds['hpres'].attrs = {'units': 'cm','long_name':'pressure sensor height above bed, neg down'} 
            ds['io'].attrs= {'units':'deg','long_name':'angle x-dir with north clockwise'}
        
        return ds

#%% instrument positioning
#load instrument serial numbers and x,y location

isxy = pd.read_excel(experimentFolder + r'\gps_bed_levels.xlsx',
                   sheet_name = 'main',
                   header=1,
                   usecols = 'A:D').T
isxy.columns = isxy.iloc[0]
isxy.drop(isxy.iloc[0].name,inplace=True)
isxy.columns = [s.strip() for s in isxy.columns]
  
#%%  
# #%% LOAD CTD data
# allCTDs = (
#     ('L2C1CTD','C1-air-TUD1_211021140929_X1405.MON'),
#     ('L1C2CTD','L1b-ctd-TUD2_211021140231_X1364.MON'),
#     ('L6C2CTD','L6b-ctd-TUD3_211021135804_X1384.MON'))


# ds = ctdDiver(allCTDs[0][0], experimentFolder,allCTDs[0][1], isxy)
# ds1 = ctdDiver(allCTDs[1][0], experimentFolder,allCTDs[1][1], isxy)
# ds2 = ctdDiver(allCTDs[2][0], experimentFolder,allCTDs[2][1], isxy)

#%% load SOLO data
# sf = 8
# allSolos = (
#     ('L2C2SOLO',[r'\20210923\raw\202439_20210919_2012_data.txt',
#                   r'\20211004\raw\202439_20211004_1735_data.txt']),
#     ('L2C4SOLO',[r'\20210923\raw\202440_20210919_1930_data.txt',
#                   r'\20211004\raw\202440_20211004_1634_data.txt',
#                   r'\20211021\raw\202440_20211021_1534_data.txt']),
#     ('L2C10SOLO',[r'\20210919\raw\202441_20210919_1902_data.txt',
#                   r'\20211004\raw\202441_20211004_1803_data.txt',
#                   r'\20211021\raw\202441_20211021_1418_data.txt']),
#     ('L4C1SOLO',[r'\20210923\raw\202438_20210920_1819_data.txt',
#                   r'\20211006\raw\202438_20211006_1828_data.txt',
#                   r'\20211021\raw\202438_20211021_1515_data.txt'])
#     )
# for solo in allSolos:
#     instrumentName = solo[0]
#     dataFiles = solo[1]
#     positioning = load_instrument_info(experimentFolder, instrumentName)
#     print(instrumentName)
#     dsList=[]
#     for file in dataFiles:
#         dsList.append(
#             Solo(instrumentName,experimentFolder,file,isxy,sf,jasave = False))
#     ds = xr.merge(dsList)
#     ds = add_positioning_info(ds,instrumentName)
#     ds.attrs = dsList[0].attrs

#     ncOutDir = experimentFolder + '//' +  instrumentName + '/raw_netcdf'
#     if not os.path.isdir(ncOutDir):
#         os.mkdir(ncOutDir )
#     ds.to_netcdf(ncOutDir + r'//' + instrumentName + '.nc')

#%% load ossie data
# for instr in ['L2C9OSSI','L2C8OSSI','L2C6OSSI','L1C2OSSI','L4C3OSSI','L5C2OSSI','L6C2OSSI']:
#     ds = Ossi(instr,
#         experimentFolder,
#         isxy = isxy)
#     ds = add_positioning_info(ds,instr)
    
#     if not os.path.isdir(experimentFolder + '//' +  instr + r'\raw_netcdf'):
#         os.mkdir(experimentFolder + '//' +  instr + r'\raw_netcdf' )
#     ncFilePath = experimentFolder + '//' +  instr + r'\raw_netcdf\{}.nc'.format(ds.name)
    
#     #if nothing else, at least specify lossless zlib compression
#     comp = dict(zlib=True, complevel=5)
#     ds.encoding = {var: comp for var in ds.data_vars}                  
#     ds.to_netcdf(ncFilePath, encoding = ds.encoding )

#%% vector data!
def vector_read_write_to_netcdf(instrumentName, 
                                experimentFolder, 
                                dataPath, 
                                isxy, #dataframe with serial number, x and y location                               
                                tstart=None, 
                                tstop=None):
    
    #instantiate the vector class and initialize variables from the .hdr file
    vec = Vector(
        name = instrumentName, 
        dataFolder = experimentFolder + '//' + instrumentName + dataPath,
        tstart = tstart,
        tstop = tstop ) 
    
    if vec.tstop<=vec.tstart:
        print('tstop is smaller than tstart')
        return
    
    #check whether there is more than 1 day of data on the file. If so, read 
    # and write this data to netcdf in blocks of 1 day.    
    if (vec.tstop-vec.tstart)<=pd.Timedelta(1,'D'):
        print('{}'.format(vec.tstart))
        
        vec.read_raw_data()
        vec.cast_to_blocks_in_xarray() #recast data from long list to burst blocks casted in an xarray
        
        # all data is collected in an xarray Dataset ds. We extract this from the 
        #class instantiation and we can easily write it to netCDF    
        ds = vec.ds
        
        ds = add_positioning_info(ds,instrumentName)
                
        
        # add global attribute metadata
        ds.attrs = {
            'Conventions':'CF-1.6', 
            'name':'{}'.format(instrumentName),
            'instrument':'{}'.format(instrumentName), 
            'instrument serial number': '{}'.format(isxy[instrumentName]['serial number']),
            'epsg':28992,
            'x':isxy[instrumentName]['xRD'],
            'y':isxy[instrumentName]['yRD'],   
            'time zone':'UTC+2',
            'coordinate type':'XYZ',
            'summary': 'SEDMEX field campaign',
            'contact person' : 'Marlies van der Lugt',
            'emailadres':'m.a.vanderlugt@tudelft.nl',
            'construction datetime':datetime.now().strftime("%d-%b-%Y (%H:%M:%S)"),
            'version': 'v1',
            'version comments' : 'constructed with xarray' }
        
        #specify compression for all the variables to reduce file size
        comp = dict(zlib=True, complevel=5)
        ds.encoding = {var: comp for var in ds.data_vars}   
        
        # save to netCDF
        if not os.path.isdir(experimentFolder + '//' +  instrumentName + r'\raw_netcdf'):
            os.mkdir(experimentFolder + '//' +  instrumentName + r'\raw_netcdf' )
        ncFilePath = experimentFolder + '//' +  instrumentName + r'\raw_netcdf\{}_{}.nc'.format(
                ds.name,ds.t.isel(t=0).dt.strftime('%Y%m%d').values
                )
        ds.to_netcdf(ncFilePath,encoding = ds.encoding )
            
    else:
        
        blockStartTimes = pd.date_range(vec.tstart.floor('D'), vec.tstop.ceil('D'), freq='1D')
        for blockStartTime in blockStartTimes:
            vector_read_write_to_netcdf(
                instrumentName, experimentFolder, dataPath, 
                isxy = isxy,
                tstart = blockStartTime, 
                tstop = blockStartTime + np.timedelta64(1,'D') ) 

#%% vector analysis
          
#L1,L2 al gedaan, 
#L2C4, L3, L5 en L6 moeten nog. De SEN bestanden moeten van het web gedownload!

                  
allVectors = (
    # ('L1C1VEC',[r'\raw\20210911-20211011',
    #             r'\raw\20211011-20211019']),
    # ('L2C3VEC',[r'\raw\20210911-20211011',
    #             r'\raw\20211011-20211019']),
    ('L3C1VEC',[r'\raw\20210911-20211011',
                r'\raw\20211011-20211019']),
    # ('L5C1VEC',[r'\raw\20210911-20211011',
    #             r'\raw\20211011-20211019']),
    # ('L6C1VEC',[r'\raw\20210911-20211011',
    #             r'\raw\20211011-20211019']),    
    # ('L2C2VEC',[r'\raw\20210910-20210913\raw',
    #             r'\raw\20210913-20210919\raw',
    #             r'\raw\20210920-20211004\raw',
    #             r'\raw\20211005-20211018\raw']),
    # ('L2C4VEC',[r'\raw\20210910-20210919\raw',
    #             r'\raw\20210920-20211004\raw',
    #             r'\raw\20211005-20211018\raw']),
    # ('L2C10VEC',[r'\raw\20210910-20210919\raw',
    #             r'\raw\20210920-20211004\raw',
    #             r'\raw\20211005-20211019\raw']),
    # ('L4C1VEC',[r'\raw\20210910-20210920\raw',
    #            r'\raw\20210921-20211006\raw',
    #            r'\raw\20211007-20211019\raw'])
    )

                  
for ivec in allVectors[2:]:
    instrumentName = ivec[0]
    dataFolders= ivec[1]
    
    positioning = load_instrument_info(experimentFolder, instrumentName)

    for dataFolder in dataFolders:
        print(instrumentName)
        vector_read_write_to_netcdf(instrumentName, 
                                    experimentFolder, 
                                    dataFolder, 
                                    isxy)

