import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import scipy as sc
import glob

plt.ioff()
Clist = []
tlist = []

# my own processing
scanz = glob.glob(r'\\tudelft.net\staff-umbrella\EURECCA\Floris\vanMarlies\reconstruct\movmean_footprint2\data\*')
ds = xr.open_dataset(r'\\tudelft.net\staff-umbrella\EURECCA\Floris\vanMarlies\reconstruct\processed\movmean_footprint2_stats2D_jawindow0_lf1_excludesmallk_width.nc')

sprlist = []
tlist = []
for file in scanz:
    time = pd.to_datetime(file.split('\\')[-1][:-4], format='%H%M%d%m%Y') 
    print(time)

    phi = ds.sel(time=time).phi.values
    if np.isnan(phi):
        continue

    dat = sc.io.loadmat(file)
    x = dat['data05'][0]['x'][0]
    y = dat['data05'][0]['y'][0]
    z = dat['data05'][0]['z'][0]
    
    # remove timestamps that are too empty 
    if np.sum(np.sum(~np.isnan(z)))<50:
        print('footprint too empty at {}'.format(time))
        continue
    
    # crop to footprint
    ix0 = np.argwhere(x[0, :]>=-0.5)[0][0]
    ix1 = np.argwhere(x[0, :]<=0.5)[-1][0]
    iy0 = np.argwhere(y[:, 0]>=-0.5)[0][0]
    iy1 = np.argwhere(y[:, 0]<=0.5)[-1][0]
    xc = x[iy0:iy1, ix0:ix1]
    yc = y[iy0:iy1, ix0:ix1]
    zc = z[iy0:iy1, ix0:ix1]

    zc[np.isnan(zc)] = np.nanmean(zc)

    if np.sum(np.isnan(zc.flatten()))>0:
        A = np.c_[xc.flatten(), yc.flatten(), np.ones(xc.shape)]
        C,_,_,_ = sc.linalg.lstsq(A, zc.flatten())    # coefficients

        # evaluate it on grid
        plane = C[0]*xc + C[1]*yc + C[2]
        zc = zc-plane
    
    vary = np.mean(np.var(zc, axis=0))
    varx = np.mean(np.var(zc, axis=1))

    radphi = phi/180*np.pi+np.pi/2
    varr = np.cos(radphi)*varx+np.sin(radphi)*vary
    varc = np.abs(-np.sin(radphi)*varx+np.cos(radphi)*vary)
    spreading = np.arctan2(varc, varr)*180/np.pi
    print('spreading = {:.1f} deg'.format(spreading))
    sprlist.append(spreading)
    tlist.append(time)

spreading = xr.DataArray(data=sprlist, coords={'time': tlist}, dims=['time'], name='spreading')
spreading.to_netcdf(r'\\tudelft.net\staff-umbrella\EURECCA\Floris\vanMarlies\reconstruct\processed\directional_spreading.nc')

a=1

