import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import scipy as sc
import glob
import sys
sys.path.append(r'C:\checkouts\\python\PhD\modules')
from local_functions import estimate_slope_footprint


######################################################################
# code to reconstruct these myself
######################################################################

plt.ioff()
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

def plot_migrating_moments(ax):
    
    if len(np.atleast_1d(ax))==1:
        ax=[ax]

    for axi in ax:
        axi.grid(color='grey', linewidth=0.5, linestyle=':')
        # migrating ripples
        axi.axvspan(pd.to_datetime('20231102 00:00'), pd.to_datetime('20231102 02:30'), alpha=0.2, color='green') #2D only
        axi.axvspan(pd.to_datetime('20231102 02:30'), pd.to_datetime('20231102 07:30'), alpha=0.2, color='grey') # missing data
        axi.axvspan(pd.to_datetime('20231102 07:30'), pd.to_datetime('20231102 14:00'), alpha=0.2, color='red') # 2D only
        axi.axvspan(pd.to_datetime('20231102 23:00'), pd.to_datetime('20231103 00:30'), alpha=0.2, color='red')
        axi.axvspan(pd.to_datetime('20231103 14:30'), pd.to_datetime('20231103 16:45'), alpha=0.2, color='green')
        axi.axvspan(pd.to_datetime('20231104 01:30'), pd.to_datetime('20231104 02:30'), alpha=0.2, color='red')
        axi.axvspan(pd.to_datetime('20231104 03:30'), pd.to_datetime('20231104 08:00'), alpha=0.2, color='green')
        axi.axvspan(pd.to_datetime('20231104 09:45'), pd.to_datetime('20231104 13:30'), alpha=0.2, color='red')
        axi.axvspan(pd.to_datetime('20231104 14:00'), pd.to_datetime('20231104 15:30'), alpha=0.2, color='green')
        axi.axvspan(pd.to_datetime('20231104 15:30'), pd.to_datetime('20231104 18:30'), alpha=0.2, color='grey') # missing data
        axi.axvspan(pd.to_datetime('20231104 18:30'), pd.to_datetime('20231104 20:00'), alpha=0.2, color='red')
        axi.axvspan(pd.to_datetime('20231105 01:00'), pd.to_datetime('20231105 07:30'), alpha=0.2, color='red')
        axi.axvspan(pd.to_datetime('20231105 19:30'), pd.to_datetime('20231105 20:30'), alpha=0.2, color='green') # 1D only
        axi.axvspan(pd.to_datetime('20231106 05:30'), pd.to_datetime('20231106 09:00'), alpha=0.2, color='green') # 1D only
        axi.axvspan(pd.to_datetime('20231106 17:00'), pd.to_datetime('20231106 20:00'), alpha=0.2, color='green') # 1D only
        axi.axvspan(pd.to_datetime('20231107 07:00'), pd.to_datetime('20231107 08:00'), alpha=0.2, color='green') # 1D only
        axi.axvspan(pd.to_datetime('20231107 19:15'), pd.to_datetime('20231107 20:00'), alpha=0.2, color='green') # 1D only
        axi.axvspan(pd.to_datetime('20231108 07:30'), pd.to_datetime('20231108 10:30'), alpha=0.2, color='green') 
        axi.axvspan(pd.to_datetime('20231108 15:45'), pd.to_datetime('20231108 18:00'), alpha=0.2, color='red')

fig, ax = plt.subplots()
ax.plot(tlist, np.sqrt(c0**2+c1**2))
ax.set_ylabel('slope [-]')
plot_migrating_moments(ax)

a=1
