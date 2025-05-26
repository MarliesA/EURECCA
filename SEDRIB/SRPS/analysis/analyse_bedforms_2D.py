import pandas as pd
import numpy as np
import xarray as xr
import scipy as sc
import glob
import os
from local_functions import spec2d2ripplegeom, spectrum_simple_2D

timel = []
Hsl = []
Lpl = []
phil = []
phiminl = []
phimaxl=[]
Lm01l = []
phi_meanl = []
nu_xl = []
nu_yl = []
nu_cl = []
nu_rl = []

fold = r'\\tudelft.net\staff-umbrella\EURECCA\DataCiaran\data'
scanz = glob.glob(os.path.join(fold, 'SRPS', 'qc_2D', '*.mat'))

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

    KKX, KKY, V = spectrum_simple_2D(x,y,z, jacorrectplane=True, jacorrectvar=True, jafig=False, jawindow=True, lf=4)

    Hs, Lp, phi, Lm01, phi_mean, nu, phi05min, phi05max = spec2d2ripplegeom(KKX, KKY, V) 
    
    timel.append(time)
    Hsl.append(Hs)
    if Lp<=0.45:
        Lpl.append(Lp)
        phil.append(phi)
        phiminl.append(phi05min)
        phimaxl.append(phi05max)
    else:
        Lpl.append(np.nan)
        phil.append(np.nan)
        phiminl.append(np.nan)
        phimaxl.append(np.nan)
    Lm01l.append(Lm01)
    phi_meanl.append(phi_mean)
    nu_xl.append(nu[0])
    nu_yl.append(nu[1])
    nu_rl.append(nu[2])
    nu_cl.append(nu[3])    

isort = np.argsort(timel)
ds = xr.Dataset(data_vars={},
                coords={'time': np.array([timel[i] for i in isort])})
ds['eta'] = (('time'), np.array([Hsl[i] for i in isort]))
ds['phi'] = (('time'), np.array([phil[i] for i in isort]))
ds['phimin'] = (('time'), np.array([phiminl[i] for i in isort]))
ds['phimax'] = (('time'), np.array([phimaxl[i] for i in isort]))
ds['labda'] = (('time'), np.array([Lpl[i] for i in isort]))
ds['Lm01'] = (('time'), np.array([Lm01l[i] for i in isort]))
ds['phi_mean'] = (('time'), np.array([phi_meanl[i] for i in isort]))
ds['nu_xl'] = (('time'), np.array([nu_xl[i] for i in isort]))
ds['nu_yl'] = (('time'), np.array([nu_yl[i] for i in isort]))
ds['nu_rl'] = (('time'), np.array([nu_rl[i] for i in isort]))
ds['nu_cl'] = (('time'), np.array([nu_cl[i] for i in isort]))

ds = ds.resample(time='15T', origin=timel[0]).nearest(tolerance='15T')
ds['phi'] = (('time'), np.where(ds.labda<0.45, ds.phi, np.nan))
ds['labda'] = (('time'), np.where(ds.labda<0.45, ds.labda, np.nan))

ds.to_netcdf(os.path.join(fold, 'SRPS', 'tailored', 'geometrystats2D.nc'))
