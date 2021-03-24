# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:54:36 2021

@author: marliesvanderl
"""
import sys
sys.path.append(r'c:\Users\marliesvanderl\phd\analysis\scripts\private\modules')
from vector import Vector
from plotfuncs import plot_wave_stats
   
  

#%%    
rho = 1030
g = 9.8

rawDataPath = r'c:\Users\marliesvanderl\phd\data\fieldvisits\20201130\vector\raw\vec2'
plotFilePath = r'c:\Users\marliesvanderl\phd\analysis\fig\fieldworkweek20201130'
KNMI_filePath = r'c:\Users\marliesvanderl\phd\data\fieldvisits\20201130\KNMI\KNMI_20201208_hourly.txt'

zb = -0.48
zi = -0.36

#load from raw data and set important parameters
vec2 = Vector('vec2',rawDataPath,zb, zi,tstart = '2020-11-30 15:20:00',tstop = '2020-11-30 17:10:00') 
print(vec2.dfpuv)
plot_wave_stats(vec2.waveStat, plotFilePath)