import pandas as pd
import numpy as np
import xarray as xr
import scipy as sc
import glob
import os
from scipy.interpolate import interp1d
from local_functions import spectrum_simple_1D
import warnings
from scipy import signal

jawindow = False
lf = 4

# #######################################################################
# # find ripple geometry from individual swaths
# #######################################################################

timel = []
Hsl = []
Lpl = []
phil = []

fold = r'\\tudelft.net\staff-umbrella\EURECCA\DataCiaran\data'
scanz = glob.glob(os.path.join(fold, 'SRPS', 'qc_1D', '*.mat'))

theta = -np.linspace(-45,45,51)
Lp = np.zeros([len(scanz), 51])
Hs = np.zeros([len(scanz), 51])
time = []

for ifile, file in enumerate(scanz):
    t = pd.to_datetime(file.split('\\')[-1][:-4], format='%H%M%d%m%Y') 
    print(t)
    time.append(t)

    dat = sc.io.loadmat(file)
    xBed = dat['data05'][0]['xBed'][0]
    zBed = dat['data05'][0]['zBed'][0]

    xb = np.arange(-0.9, 0.9, 0.01)

    # find ripple wave length and height for a sector of 50 degrees on each side of shore normal
    # implementatation now is: central bin is 24, therefore first is 51st index 
    for ithet, itheta in enumerate(range(1, 51)):

        # 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xbedithet = np.nanmean(xBed[:, itheta-2:itheta+3], axis=-1)
            zbedithet = np.nanmean(zBed[:, itheta-2:itheta+3], axis=-1)

        #exclude profilees that have too little datapoints
        if np.sum(~np.isnan(xbedithet))<0.8*len(xbedithet):  #50:
            Hs[ifile, ithet] = np.nan
            Lp[ifile, ithet] = np.nan
            continue

        # inteprolate bed onto regular grid xb
        zb_lin = np.interp(xb, xbedithet, zbedithet)
        ix = ~np.logical_or(np.isnan(xbedithet), np.isnan(zbedithet)) 
        func = interp1d(xbedithet[ix], zbedithet[ix], bounds_error=False, fill_value=np.nan, kind='cubic')
        zb_cubic = func(xb)

        # only apply cubic interpolator on missing data
        zb = np.where(~np.isnan(zb_lin), zb_lin, zb_cubic)
        # remove unintended extrapolation by np.interp
        valid_range = np.where(~np.isnan(zb_cubic))[0]
        zb[:valid_range[0]] = np.nan
        zb[valid_range[-1]:] = np.nan


        # to fourier space
        xbb = xb[~np.isnan(zb)]
        zbb = zb[~np.isnan(zb)]

        zbb = signal.detrend(zbb)
        varpre = np.var(zbb)

        if jawindow:
            wx = signal.windows.hann(len(zbb))
            zbb = wx*zbb

        # to wave spectra
        kx, vy = spectrum_simple_1D(xbb, zbb, lf=lf, jacorrectplane=False, jawindow=False)

        # peak ripple period smaller than 50 cm    
        vy_tmp = np.where(kx>=2, vy, 0)
        imx = np.argmax(vy_tmp)
        kp = kx[imx]
        if kp>=2:
            Lp[ifile, ithet] = 1/kp
        else: 
            Lp[ifile, ithet] = np.nan

        # significant wave height
        Hs[ifile, ithet] = 4*np.sqrt(varpre)

isort = np.argsort(time)
ds = xr.Dataset(data_vars={},
                coords={'time': np.array([time[i] for i in isort]),
                        'theta': theta})
ds['eta'] = (('time', 'theta'), np.array([Hs[i, :] for i in isort]))
ds['eta'].attrs = {'long_name':'eta', 'units': 'm'}
ds['labda'] = (('time', 'theta'), np.array([Lp[i, :] for i in isort]))
ds['labda'].attrs = {'long_name':'lambda', 'units': 'm'}

ds2 = ds.where(ds.theta>-45).dropna(dim='time', how='all', subset=['labda'])

# the average of all the angles with minimum wave length
labmin = ds2.labda.min(dim='theta').expand_dims(dim={"theta": len(ds.theta)})
thetmin = ds2['theta'].expand_dims(dim={"time": len(ds2.time)}).where(ds2.labda==labmin).mean(dim='theta')
argmin = (np.abs(ds2.theta-thetmin)).argmin(dim='theta')


etamin = []
labdamin = []
for it in range(len(ds2.time)):
    etamin.append(ds2.isel(time=it).eta[argmin[it]])
    labdamin.append(ds2.isel(time=it).labda[argmin[it]])

ds2['labda'] = ('time', np.array(labdamin))
ds2['eta'] = ('time', np.array(etamin))
ds2['thetmin'] = thetmin

# place them back on the total dataset
ds['thetmin'] = ds2['thetmin']
ds['labdamin'] = ds2['labda']
ds['etamin'] = ds2['eta']

ds.to_netcdf(os.path.join(fold, 'SRPS', 'tailored', 'geometrystats1D.nc'))



