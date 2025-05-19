import pandas as pd
import numpy as np
import xarray as xr
import scipy as sc
import glob
from scipy import signal

rib1d=xr.open_dataset(r'\\tudelft.net\staff-umbrella\EURECCA\Floris\vanMarlies\reconstruct\processed\stats1D_window0_lf4.nc')
rib1dsel = rib1d.where(rib1d['theta']>-45).dropna(dim='time', subset=['labda'], how='all')
argmin = rib1dsel.labda.argmin(dim='theta')
rib1d['argmin'] = argmin
thetmin = rib1dsel.theta[argmin]

ridsel = rib1d.dropna(dim='time', subset=['labda'], how='all')
ripple_dir = np.floor((ridsel.labda.argmin(dim='theta')).rolling(time=8, min_periods=6, center=True).median())

etamin = []
labdamin = []
for it in range(len(rib1dsel.time)):
    arg = ripple_dir[it]
    if np.isnan(arg):
        etamin.append(np.nan)
        labdamin.append(np.nan)
        continue
    else:
        arg = int(arg)
        etamin.append(rib1dsel.isel(time=it).eta[arg])
        labdamin.append(rib1dsel.isel(time=it).labda[arg])

rib1dsel['labda'] = ('time', np.array(labdamin))
rib1dsel['eta'] = ('time', np.array(etamin))
rib1dsel['thetmin'] = thetmin # ('time', thetmin)
rib1d['thetmin'] = rib1dsel['thetmin']
rib1d['labdamin'] = rib1dsel['labda']
rib1d['etamin'] = rib1dsel['eta']

# remove unrealistic estimates of eta
rib1d = rib1d.where(rib1d.etamin<0.25)

jacorrectplane = True
jawindow = True
philist = []
plist = []
dmiglist = []
time1list = []
time2list = []

scanz = glob.glob(r'\\tudelft.net\staff-umbrella\EURECCA\Floris\vanMarlies\reconstruct\data\*')
for file1, file2 in zip(scanz[:-1], scanz[1:]):
    time1 = pd.to_datetime(file1.split('\\')[-1][:-4], format='%H%M%d%m%Y') 
    time2 = pd.to_datetime(file2.split('\\')[-1][:-4], format='%H%M%d%m%Y') 
    if time1<pd.to_datetime('20231102 21:00'):
        continue

    time1list.append(time1)
    time2list.append(time2)
    print(time2)


    dat = sc.io.loadmat(file1)
    x1 = dat['data05'][0]['xBed'][0]
    z1 = dat['data05'][0]['zBed'][0]

    # remove timestamps where no wave direction is computed
    if np.isnan(rib1d.sel(time=time1)['argmin'].values):
        print('footprint 1 has no direction {}'.format(time1))
        plist.append(np.nan)
        dmiglist.append(np.nan)
        philist.append(np.nan)
        continue
    else:
        philist.append(rib1d.sel(time=time1)['argmin'].values)

    ct = int(ripple_dir.sel(time=time1).values)    
    if np.isnan(ct):
        plist.append(np.nan)
        dmiglist.append(np.nan)
        print('ct is nan')
        continue
    
    X1 = np.nanmean(x1[:, ct-1:ct+2], axis=1)
    Z1 = np.nanmean(z1[:, ct-1:ct+2], axis=1)
    Zlength = len(Z1.flatten())
    itrue = np.logical_and(~np.isnan(X1), ~np.isnan(Z1))
    X1 = X1[itrue]
    Z1 = Z1[itrue]

    # remove timestamps that are too empty 
    if np.sum(itrue)<0.8*Zlength:
        print('footprint 1 too empty at {}'.format(time1))
        plist.append(np.nan)
        dmiglist.append(np.nan)
        continue    
        
    dat = sc.io.loadmat(file2)
    x2 = dat['data05'][0]['xBed'][0]
    z2 = dat['data05'][0]['zBed'][0]
    
    X2 = np.nanmean(x2[:, ct-1:ct+2], axis=1)
    Z2 = np.nanmean(z2[:, ct-1:ct+2], axis=1)
    itrue = np.logical_and(~np.isnan(X2), ~np.isnan(Z2))
    X2 = X2[itrue]
    Z2 = Z2[itrue]

    # remove timestamps that are too empty 
    if np.sum(itrue)<0.8*Zlength:
        print('footprint 2 too empty at {}'.format(time2))
        plist.append(np.nan)
        dmiglist.append(np.nan)
        continue

    Xt = np.linspace(-0.75, 0.75, 601)    
    Z1t = np.interp(Xt, X1, Z1)
    Z2t = np.interp(Xt, X2, Z2)

    Z1t = signal.detrend(Z1t)
    Z2t = signal.detrend(Z2t)

    # attempt to normalize this. Unsure why this doesn't happen under the hood?
    a = (Z1t - np.mean(Z1t)) / (np.std(Z1t) * len(Z1t))
    b = (Z2t - np.mean(Z2t)) / (np.std(Z2t))
    cor = np.correlate(a, b, 'same')

    ipx = np.argmax(cor)
    dmig = Xt[ipx]*100    

    plist.append(np.max(cor))
    dmiglist.append(dmig)

isort = np.argsort(np.array(time1list))
ds = xr.Dataset(data_vars={},
                coords={'time': np.array([time2list[i] for i in isort])})
ds['p'] = (('time'), np.array([plist[i] for i in isort]))
ds['phi'] = (('time'), np.array([philist[i] for i in isort]))
ds['dmig'] = (('time'), np.array([dmiglist[i] for i in isort]))
ds['tprev'] = (('time'), np.array([time1list[i] for i in isort]))

ds.to_netcdf(r'\\tudelft.net\staff-umbrella\EURECCA\Floris\vanMarlies\reconstruct\processed\stats1D_migrationrates_250114_refined5_avdir.nc')

a=1


