# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 13:20:17 2021

@author: marliesvanderl
"""
import os
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from osgeo import gdal, ogr
import fiona
import pandas as pd
import geopandas as gpd
import math
from cycler import cycler

def load_field_visit_20201016(dataPath = r'C:/Users/marliesvanderl/phd/data',
                              Transects = None):
    '''
    Loads the data from raw files and saves the tracks in a DataFrame named
    Transects. If such a DataFrame is already specified, it concatenates this 
    data to that DataFrame
    '''
    
    # load from file
    data = pd.read_csv(dataPath + r'\fieldvisits\gps\20201016\PHZD01.txt', 
                       sep="\t", header=1)
    data.columns = ['ID', 'date','time','x', 'y', 'z','he','ve']
    
    data['time2'] = data['date'] + ' ' + data['time']
    
    #Time is documented in GMT+1 (Dutch winter time), we have to subtract one 
    #hour to be in UTC : - pd.Timedelta(hours=1)
    data['DateTime'] = pd.to_datetime(data['time2'],
                            format='%d/%m/%Y %H:%M:%S')- pd.Timedelta(hours=1) 
    
    #make sure height is referenced to NAP
    poleLength=1.5
    data['z'] = data['z']-poleLength
    
    data2 = data[data['ID'].str.contains('WL')]
    tref = data2['DateTime'].iloc[1250]
    data3 = data2[data2['DateTime']<tref]
    
    tref2 = data2['DateTime'].iloc[1395]
    data4 = data2[data2['DateTime']>tref2]
    
    WL = pd.concat([data3,data4])
    
    data5 = pd.concat([data, WL, WL]).drop_duplicates(keep=False)
    
    #identfy individual transects:
    dx = np.insert(np.diff(data5['x']), 0, 0., axis=0)
    dy = np.insert(np.diff(data5['y']), 0, 0., axis=0)
    dn = np.sqrt(dx**2+dy**2)
    trackStarts = dn>100
    trackStarts[0] = True
    # data6 = data5.iloc[trackStarts] 
        
    trackStartIndices = [i for i, x in enumerate(trackStarts) if x]
    # manually modify trackStartIndex for first track to true start
    trackStartIndices[0] += 45
    trackStopIndices = trackStartIndices[1:]
    trackStopIndices.append(len(trackStarts))
    
    #extract list of untied tracks
    tracks=[]
    for itrack in np.arange(len(trackStartIndices)):    
        tracks.append(data5.iloc[trackStartIndices[itrack]:trackStopIndices[itrack]].copy())
    
    T = pd.DataFrame(data={'ray':np.arange(0,len(tracks)),
                           'x':[track['x'].values for track in tracks],
                           'y':[track['y'].values for track in tracks],
                           'z':[track['z'].values for track in tracks],
                           'ID':[track['time2'].values.min() for track in tracks],
                           'source':['visit_20201016'] * len(tracks)})                           
    
    # if a dataframe was added to the function, concatenate the new data, otherwise instantiate the dataframe
    if Transects is None:
        Transects = T
    else:
        Transects = pd.concat([Transects,T])
        
    return Transects




def load_field_visit_20201130(dataPath = r'C:\Users\marliesvander\phd\data',
                              Transects = None):  
    '''
    Loads the data from raw files and saves the tracks in a DataFrame named
    Transects. If such a DataFrame is already specified, it concatenates this 
    data to that DataFrame   
    '''
    
    data1 = pd.read_csv(r'c:\Users\marliesvanderl\phd\fieldVisits\20201130_fieldweek\gps\PHZD_PROF_2dec.txt',
                        sep="\t", header=1)
    data1.columns = ['ID', 'x', 'y', 'z','u1','u2','u3','u4']

    data2 = pd.read_csv(r'c:\Users\marliesvanderl\phd\fieldVisits\20201130_fieldweek\gps\PHZDDOL_profielenPart2.txt',
                        sep="\t", header=1)
    data2.columns = ['ID', 'x', 'y', 'z','u1','u2','u3','u4']

    data =pd.concat([data1,data2],ignore_index=False)

    # Manually determined the starting points and stopping points and the date 
    # it was measured from the log book
    trackStartIndices = [395,989, 1333, 2350, 3088, 3431, 4055, 4810, 5550]
    trackStopIndices = [610, 1105, 1690, 2515, 3195, 3541, 4150, 4970, 5795]
    trackID = [0,1,1,2,3,3,4,5,6]
    date = ['02/12/2020','02/12/2020','03/12/2020','02/12/2020','02/12/2020',
            '03/12/2020','03/12/2020','03/12/2020','03/12/2020']
    tracks = []
    for itrack in np.arange(len(trackStartIndices)):   
        tracks.append(data.iloc[trackStartIndices[itrack]:trackStopIndices[itrack]].copy())

    # chuck it all into a DataFrame
    T = pd.DataFrame(data={'ray':trackID,
                           'x':[track['x'].values for track in tracks],
                           'y':[track['y'].values for track in tracks],
                           'z':[track['z'].values for track in tracks],
                           'ID':date,
                           'source':['visit_20201130'] * len(tracks)})     
    
    # if a dataframe was added to the function, concatenate the new data, otherwise instantiate the dataframe
    if Transects is None:
        Transects = T
    else:
        Transects = pd.concat([Transects,T])
        
    return Transects 




#%% usage:
T1 = load_field_visit_20201016()
T2 = load_field_visit_20201130()