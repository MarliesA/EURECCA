# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 08:37:43 2021

@author: marliesvanderl
"""
# import pdb; pdb.set_trace()

import sys
sys.path.append(r'c:\Users\marliesvanderl\phd\analysis\scripts\private\modules')
from vector import Vector
from solo import Solo
import numpy as np
from profiler import Profiler
    
#%% general settings

rho = 1025 # kg/m3
g = 9.81 # m/s2
blockLength = int(600) #seconds 

ncOutDir = r'c:\Users\marliesvanderl\phd\analysis\processed_data'



#%% solo data to netcdf

# tstart = '2020-11-30 14:00'
# tstop = '2020-12-4 16:00'   
# f  = 8 # Hz

# names = ['solo1','solo2','solo4']
# dataFolders = [r'c:\Users\marliesvanderl\phd\data\fieldvisits\20201130\solo\202438_20201207_1119_solo1',
#               r'c:\Users\marliesvanderl\phd\data\fieldvisits\20201130\solo\202441_20201207_1104_solo2',
#               r'c:\Users\marliesvanderl\phd\data\fieldvisits\20201130\solo\202440_20201207_1309_solo4']
# zis = [-0.98,-0.560,-0.635]#m NAP instrument position
# zbs = [-1.31,-1.31,-1.15]#m NAP bed level position

# for name,dataFolder,zi,zb in zip(names,dataFolders,zis,zbs):
     
#     #use the class Solo to read raw data, correct for air pressure and cast in bursts
#     solo = Solo(name,dataFolder,f,zi,zb,tstart,tstop,
#                   rho = rho, 
#                   g = g,
#                   emergedT0 = True)
    
#     #all data is collected in an xarray Dataset ds which we can easily write to netCDF
#     ds = solo.ds
#     # add global attribute metadata
#     ds.attrs={'Conventions':'CF-1.6', 
#                 'title':'{}'.format(solo.name), 
#                 'summary': 'December pilot field campaign',
#                 'contact person' : 'Marlies van der Lugt',
#                 'emailadres':'m.a.vanderlugt@tudelft.nl',
#                 'version': 'v1',
#                 'version comments' : 'constructed with xarray'}
    
#     #specify compression for all the variables to reduce file size
#     comp = dict(zlib=True, complevel=5)
#     ds.encoding = {var: comp for var in ds.data_vars}
    
#     # save to netCDF
#     ds.to_netcdf(ncOutDir + r'\{}_pilot.nc'.format(solo.name))

#%% ADV's to netcdf 
referenceDataPath = r'c:\Users\marliesvanderl\phd\analysis\processed_data\solo1_pilot.nc'

names = ['vec1','vec2']
dataFolders = [r'c:\Users\marliesvanderl\phd\data\fieldvisits\20201130\vector\raw\vec1',
              r'c:\Users\marliesvanderl\phd\data\fieldvisits\20201130\vector\raw\vec2']
tstarts = ['2020-11-30 16:00','2020-11-30 16:00:00']
tstops = ['2020-12-04 12:00','2020-12-03 03:00:00'] #vec2 stopped recording earlier
zis = [-0.36,-0.56]#m NAP
zbs = [-0.48,-1.13]#m NAP

for name,dataFolder,zi,zb,tstart,tstop,jaReference in zip(names,dataFolders,zis,zbs,tstarts,tstops,[True,False]): 
    vec = Vector(name,dataFolder,zb, zi,tstart = tstart,tstop = tstop) 
    
    if jaReference:
        vec.reference_pressure_to_water_level_observations(referenceDataPath)
    vec.cast_to_blocks_in_xarray()
    vec.compute_block_averages()
    #all data is collected in an xarray Dataset ds which we can easily write to netCDF    
    ds = vec.ds    
    
    if jaReference:
        #in this case the corrected pressure was only used to compute block averages
        #with the aid of a reference instrument. Therefore this on the Netcdf
        #should be set to zero
        ds['p'] = (('t','N'),np.nan*np.zeros(ds.p.shape))
        ds['pm'] = (('t'),np.nan*np.zeros(ds.pm.shape))
        
    # add global attribute metadata
    ds.attrs={'Conventions':'CF-1.6', 
                'title':'{}'.format(vec.name), 
                'summary': 'December pilot field campaign',
                'contact person' : 'Marlies van der Lugt',
                'emailadres':'m.a.vanderlugt@tudelft.nl',
                'version': 'v1',
                'version comments' : 'constructed with xarray'}
    
    #specify compression for all the variables to reduce file size
    comp = dict(zlib=True, complevel=5)
    ds.encoding = {var: comp for var in ds.data_vars}
    
    # save to netCDF
    ds.to_netcdf(ncOutDir + r'\{}_pilot.nc'.format(vec.name))



#%% HR profiler
dataFolder = r'c:\Users\marliesvanderl\phddata\fieldvisits\20201130\profiler\raw\ascii'
ncOutDir = r'c:\Users\marliesvanderl\phd\analysis\processed_data'

#######part 1, before we lowered the instrument and levelled it
config = {'name':'hrprofiler_part1','zb':-1.46,'zi':-0.55,'tstart':'2020-11-30 16:00:00','tstop':'2020-12-02 14:30:00'}
P1 = Profiler(config['name'], dataFolder,zb = config['zb'],zi = config['zi'],tstart = config['tstart'],tstop = config['tstop'])
P1.load_all_data()
ds = P1.get_dataset()

# add global attribute metadata
ds.attrs={'Conventions':'CF-1.6', 
            'title':'{}'.format('HR Profiler part 1'), 
            'summary': 'December pilot field campaign, before levelling of the instrument on Wednesday',
            'contact person' : 'Marlies van der Lugt',
            'emailadres':'m.a.vanderlugt@tudelft.nl',
            'version': 'v1',
            'version comments' : 'constructed with xarray'}
#specify compression for all the variables to reduce file size
comp = dict(zlib=True, complevel=5)
ds.encoding = {var: comp for var in ds.data_vars}

# save to netCDF
ds.to_netcdf(ncOutDir + r'\{}_pilot.nc'.format(ds.name.values)) 

#######part 1, after levelling and lowering
config = {'name':'hrprofiler_part2','zb':-1.46,'zi':-0.31,'tstart':'2020-12-02 15:30:00','tstop':'2020-12-04 13:00:00'}
P2 = Profiler(config['name'], dataFolder,zb = config['zb'],zi = config['zi'],tstart = config['tstart'],tstop = config['tstop'])
P2.load_all_data()
ds = P2.get_dataset()

# add global attribute metadata
ds.attrs={'Conventions':'CF-1.6', 
            'title':'{}'.format('HR Profiler part 2'), 
            'summary': 'December pilot field campaign, before levelling of the instrument on Wednesday',
            'contact person' : 'Marlies van der Lugt',
            'emailadres':'m.a.vanderlugt@tudelft.nl',
            'version': 'v1',
            'version comments' : 'constructed with xarray'}
#specify compression for all the variables to reduce file size
comp = dict(zlib=True, complevel=5)
ds.encoding = {var: comp for var in ds.data_vars}

# save to netCDF
ds.to_netcdf(ncOutDir + r'\{}_pilot.nc'.format(ds.name.values)) 






