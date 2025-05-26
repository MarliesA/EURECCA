import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import scipy as sc
import glob
import sys
sys.path.append(r'C:\checkouts\\python\PhD\modules')
from local_functions import estimate_slope_footprint

# Script to inspect the slope directly underneath the ripple scanner (in the manuscript only mentioned within the text)

Clist = []
tlist = []

# my own processing
scanz = glob.glob(r'\\tudelft.net\staff-umbrella\EURECCA\Floris\vanMarlies\reconstruct\movmean_footprint2\data\*')

for file in scanz:
    time = pd.to_datetime(file.split('\\')[-1][:-4], format='%H%M%d%m%Y') 
    print(time)

    dat = sc.io.loadmat(file)
    x = dat['data05'][0]['x'][0]
    y = dat['data05'][0]['y'][0]
    z = dat['data05'][0]['z'][0]
    
    # remove timestamps that are too empty 
    if np.sum(np.sum(~np.isnan(z)))<50:
        print('footprint too empty at {}'.format(time))
        continue
    
    C = estimate_slope_footprint(x,y,z)
    Clist.append(C)
    tlist.append(time)

c0 = np.array([c[0] for c in Clist])
c1 = np.array([c[1] for c in Clist])
c2 = np.array([c[2] for c in Clist])

# plot to see the slope of the bed below the SRPS over time
fig, ax = plt.subplots()
ax.plot(tlist, np.sqrt(c0**2+c1**2))
ax.set_ylabel('slope [-]')