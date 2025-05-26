#%%

import os
import sys
import sys
# sys.path.append(r'c:\checkouts\python\PHD\modules')
sys.path.append(r'C:\checkouts\PhD\modules')
from vector import Vector
from datetime import datetime



# location of raw data
dataFolder = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\vec008\raw\all'
# name of the instantiated vector class
name = 'vec008'

ncOutDir = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\vec008\raw_netcdf\all'

# raw data to netcdf
vec = Vector(name, dataFolder)

# reads the raw data from tstart to tstop and casts all data in a pandas DataFrame that is stored under vec.dfpuv.
# in case there is no data between tstart and tstop the DataFrame is not instantiated
vec.read_raw_data()

# break up the data into burst blocks
vec.cast_to_blocks_in_xarray()

# compute burst averages (make sure to read vector.py what is happening exactly!)
vec.compute_block_averages()

# all data is collected in an xarray Dataset ds. We extract this from the class instantiation and
# we can easily write it to netCDF
ds = vec.ds

# add global attribute metadata
ds.attrs = {'instrument': '{}'.format(vec.name),
            'instrument serial number': '{}'.format(16725),
            'epsg': 28992,
            'x': 116086.4, 
            'y': 558942.0,
            'time zone': 'UTC+1',
            'coordinate type': 'XYZ',
            'summary': '2023 PHZD field campaign. Vec008 was installed downward looking with control volume approximately 30 cm above the bed',
            'contact person': 'Marlies van der Lugt',
            'emailadres': 'm.a.vanderlugt@tudelft.nl',
            'construction datetime': datetime.now().strftime("%d-%b-%Y (%H:%M:%S)"),
            'version': 'v1',
            'version comments': 'constructed with xarray'}

#specify compression for all the variables to reduce file size
comp = dict(zlib=True, complevel=5)
ds.encoding = {var: comp for var in ds.data_vars}

# save to netCDF
if not os.path.exists(ncOutDir):
    os.mkdir(ncOutDir)
ds.to_netcdf(ncOutDir + r'\{}.nc'.format(vec.name))
#








