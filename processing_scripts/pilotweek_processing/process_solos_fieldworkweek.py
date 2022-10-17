# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 08:37:43 2021

@author: marliesvanderl
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
sys.path.append(r'c:\Users\marliesvanderl\phd\analysis\scripts\private\modules')
from solo import Solo
from plotfuncs import plot_wave_stats
#%%

def plot_compare_wave_stats(solo1,solo2,solo3,plotFilePath):
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=[8.3,5.8])
    ax1.plot(solo1.waveStat['Hm0'],label='df1')
    ax1.plot(solo2.waveStat['Hm0'],label='df2')
    ax1.plot(solo3.waveStat['Hm0'],label='df3')
    ax1.set_ylabel('Hm0 [m]')
    ax1.legend()
    
    
    ax2.plot(solo1.waveStat['Tm01'])
    ax2.plot(solo2.waveStat['Tm01'])
    ax2.plot(solo3.waveStat['Tm01'])           
    ax2.set_ylabel('Tm01 [s]')
    ax2.set_ylim([0,5])
    
    ax3.plot(solo1.bursts['zsmean'])
    ax3.plot(solo2.bursts['zsmean'])
    ax3.plot(solo3.bursts['zsmean'])    
    ax3.set_ylabel('eta [m+MSL]')
    
    major_locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    minor_locator = mdates.AutoDateLocator(minticks=10, maxticks=52)
    formatter = mdates.ConciseDateFormatter(major_locator)
    [ax.spines['top'].set_visible(False) for ax in (ax1,ax2,ax3)]
    [ax.spines['right'].set_visible(False) for ax in (ax1,ax2,ax3)]
    [ax.xaxis.set_major_locator(major_locator) for ax in (ax1,ax2,ax3)]
    [ax.xaxis.set_minor_locator(minor_locator) for ax in (ax1,ax2,ax3)]
    [ax.xaxis.set_major_formatter(formatter) for ax in (ax1,ax2,ax3)]  
    # [ax.autoscale(enable=True, axis='x', tight=True) 
    fig.tight_layout() 
        
    fig.savefig(plotFilePath + r'_wavestats_compared.png',        
            dpi=200,
            bbox_inches = 'tight',
            pad_inches = 0.1)    
#%%
rho = 1025 # kg/m3
g = 9.81 # m/s2
blockLength = int(600) #seconds 
tstart = pd.to_datetime('2020-11-30 14:00')
tstop = pd.to_datetime('2020-12-4 16:00')

plotFilePath = r'c:\Users\marliesvanderl\phd\analysis\fig\fieldworkweek20201130\Solo1'
dataFolder = r'c:\Users\marliesvanderl\phd\data\fieldvisits\20201130\solo\202438_20201207_1119_solo1'
f  = 8 # Hz
zi = -0.985 #m NAP
zb = -1.31 #m NAP
solo1 = Solo('solo1',dataFolder,f,zi,zb,rho,g,tstart,tstop)

solo1.recast_signal_to_blocks(blockLength,zi,zb,rho=rho,g=g)
solo1.compute_wave_stats_from_pressure_bursts(jaFig=False,plotFilePath = '')
plot_wave_stats(solo1.waveStat,plotFilePath)

plotFilePath = r'c:\Users\marliesvanderl\phd\analysis\fig\fieldworkweek20201130\Solo2'
dataFolder = r'c:\Users\marliesvanderl\phd\data\fieldvisits\20201130\solo\202441_20201207_1104_solo2'
zi = -0.560 #m NAP
zb = -1.31 #m NAP
solo2 = Solo('solo2',dataFolder,f,zi,zb,tstart,tstop,rho,g)

solo2.recast_signal_to_blocks(blockLength,zi,zb,rho=rho,g=g)
solo2.compute_wave_stats_from_pressure_bursts(jafig=False,plotFilePath = '')
plot_wave_stats(solo2.waveStat,plotFilePath)

plotFilePath = r'c:\Users\marliesvanderl\phd\analysis\fig\fieldworkweek20201130\Solo4'
dataFolder = r'c:\Users\marliesvanderl\phd\data\fieldvisits\20201130\solo\202440_20201207_1309_solo4'
zi = -0.635 #m NAP
zb = -1.15 #m NAP
solo4 = Solo('solo4',dataFolder,f,zi,zb,tstart,tstop,rho,g)

solo4.recast_signal_to_blocks(blockLength,zi,zb,rho=rho,g=g)
solo4.compute_wave_stats_from_pressure_bursts(jafig=False,plotFilePath = '')
plot_wave_stats(solo4.waveStat,plotFilePath)

plotFilePath =  r'c:\Users\marliesvanderl\phd\analysis\fig\fieldworkweek20201130\Solo'
plot_compare_wave_stats(solo1,solo2,solo4,plotFilePath)



 
    
  