import pandas as pd
import numpy as np
import xarray as xr
import scipy as sc
import glob
import os
from scipy import signal

jacorrectplane = True
jawindow = True
philist = []
plist = []
dmiglist = []
time1list = []
time2list = []
percvalid1 = []
percvalid2 = []

# my own processing
fold = r'\\tudelft.net\staff-umbrella\EURECCA\DataCiaran\data'
scanz = glob.glob(os.path.join(fold, 'SRPS', 'qc_2D', '*.mat'))

for file1, file2 in zip(scanz[:-1], scanz[1:]):
    time1 = pd.to_datetime(file1.split('\\')[-1][:-4], format='%H%M%d%m%Y') 
    time2 = pd.to_datetime(file2.split('\\')[-1][:-4], format='%H%M%d%m%Y') 
    time1list.append(time1)
    time2list.append(time2)
    print(time2)
    
    dat = sc.io.loadmat(file1)
    x = dat['data05'][0]['x'][0]
    y = dat['data05'][0]['y'][0]
    z1 = dat['data05'][0]['z'][0]
    # remove timestamps that are too empty 
    percvalid1.append(np.sum(np.sum(~np.isnan(z1)))/len(z1.flatten()))
    if np.sum(np.sum(~np.isnan(z1)))<0.1*len(z1.flatten()):
        print('footprint 1 too empty at {}'.format(time1))
        philist.append(np.nan)
        plist.append(np.nan)
        dmiglist.append(np.nan)
        percvalid2.append(np.nan)
        continue

    dat = sc.io.loadmat(file2)
    x = dat['data05'][0]['x'][0]
    y = dat['data05'][0]['y'][0]
    z2 = dat['data05'][0]['z'][0]

    # remove timestamps that are too empty 
    percvalid2.append(np.sum(np.sum(~np.isnan(z1)))/len(z1.flatten()))
    if np.sum(np.sum(~np.isnan(z2)))<0.1*len(z2.flatten()):
        print('footprint 2 too empty at {}'.format(time2))
        philist.append(np.nan)
        plist.append(np.nan)
        dmiglist.append(np.nan)
        continue

    # set to nan if either has a nan
    z1[np.isnan(z2)] = np.nan
    z2[np.isnan(z1)] = np.nan

    # fill with mean
    z1[np.isnan(z1)] = np.nanmean(z1)
    z2[np.isnan(z2)] = np.nanmean(z2)

    # remove mean
    z1 -= np.nanmean(z1)
    z2 -= np.nanmean(z2)

    # planar detrend
    if jacorrectplane:

        z0 = z1.copy()
        ny, nx = x.shape
        if np.sum(np.isnan(z1.flatten()))>0:
            a=1
        A = np.c_[x.flatten(), y.flatten(), np.ones(ny*nx)]
        C,_,_,_ = sc.linalg.lstsq(A, z1.flatten())    # coefficients

        # evaluate it on grid
        plane = C[0]*x + C[1]*y + C[2]
        z1 = z1-plane
        z1 = np.where(~np.isnan(z0), z1, z0)

        z0 = z2.copy()
        if np.sum(np.isnan(z2.flatten()))>0:
            a=1
        A = np.c_[x.flatten(), y.flatten(), np.ones(ny*nx)]
        C,_,_,_ = sc.linalg.lstsq(A, z2.flatten())    # coefficients

        # evaluate it on grid
        plane = C[0]*x + C[1]*y + C[2]
        z2 = z2-plane
        z2 = np.where(~np.isnan(z0), z2, z0)

    # apply window
    if jawindow:
        ny, nx = x.shape
        wx= sc.signal.windows.hann(nx)
        wy= sc.signal.windows.hann(ny)
        WX, WY = np.meshgrid(wx, wy)
        z1 = WX*WY*z1
        z2 = WX*WY*z2

    # attempt to normalize this. Unsure why this doesn't happen under the hood?
    cor = signal.correlate2d(z1, z2, mode='same') / np.std(z1)/np.std(z2) /z1.shape[0]/z1.shape[1]

    ipx, ipy = np.unravel_index(np.argmax(cor), [ny, nx])
    phi = 180/np.pi*np.arctan2(y[ipx, ipy], x[ipx, ipy])-90
    dmig = np.sqrt(y[ipx, ipy]**2+x[ipx, ipy]**2)*100    

    philist.append(phi)
    plist.append(np.max(cor))
    dmiglist.append(dmig)

isort = np.argsort(np.array(time1list))
ds = xr.Dataset(data_vars={},
                coords={'time': np.array([time2list[i] for i in isort])})
ds['p'] = (('time'), np.array([plist[i] for i in isort]))
ds['phi'] = (('time'), np.array([philist[i] for i in isort]))
ds['dmig'] = (('time'), np.array([dmiglist[i] for i in isort]))
ds['tprev'] = (('time'), np.array([time1list[i] for i in isort]))
ds['percvalid1'] = (('time'), np.array([percvalid1[i] for i in isort]))
ds['percvalid2'] = (('time'), np.array([percvalid2[i] for i in isort]))

ds.to_netcdf(os.path.join(fold, 'SRPS', 'tailored', 'migrationrates2D.nc'))


