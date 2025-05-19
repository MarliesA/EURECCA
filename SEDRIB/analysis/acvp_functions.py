import pandas as pd
import xarray as xr
from scipy import signal
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\checkouts\python\PHD\modules')
import puv
import glob
import os
import matplotlib.dates as mdates

def xr_bursts_to_series(ds):
    #prepare the multiindex:    
    #specify burst time axis in terms of timedeltas
    ds['dt'] = (('N'),pd.to_timedelta(ds.N.values,unit='S') )
    t2, dt2 = xr.broadcast(ds.t, ds.dt)

    #give a time coordinate to every data point
    time = (t2+dt2).values.flatten()    

    # new dataset 
    ds2 = xr.Dataset(
        data_vars = {},
        coords = {'time':time} 
    )
    print(len(ds2.time))
    for var in ds.data_vars:
        if ('N' in ds[var].coords) and ('t' in ds[var].coords):
            print(var)
            ds2[var] = (('time'), ds[var].values.flatten())
            ds2[var].attrs = ds[var].attrs
            
    return ds2

def fix_time_coord_UBlab(ds):
    """
    sometimes the time dimension is not uniformly increasing or it makes jumps. 
    In that case, throw away the suspicious coords and replace by the most expected time increment
    """
    time = ds.time.values
    a = (time-time[0])/np.timedelta64(1, 'ns')
    idx = np.logical_or(np.logical_or(np.abs(np.diff(a))>2.5e7, np.diff(a)<0), np.abs(np.diff(a))<1.5e7)
    for ix in np.where(idx):
        a[ix+1] = a[ix-1]+ 2*np.median(np.diff(a))
        a[ix+2] = a[ix-1]+ 3*np.median(np.diff(a))
        
    t2 = a*np.timedelta64(1, 'ns')+time[0]
    return ds.assign_coords({'time': t2})

def load_adcv_corrected(file, thet1, thet2, zi=None, coarsen=False, ncoarsenz = 1, remove_T1=False):
    theta1 = thet1/180*np.pi
    theta2 = thet2/180*np.pi

    dist = sc.io.loadmat(r'c:\Users\marliesvanderl\phd_archive\communication\vanNoemie\dist15.mat')
    dist = dist['dist'][:,0]

    ds = xr.open_dataset(file)
    ds = fix_time_coord_UBlab(ds)

    if not zi is None:
        ds = ds.assign_coords({'z': zi-dist*np.cos(thet1/180*np.pi)})  # correct for the angle of the transducers with the vertical
        ds['z'].attrs = {'units': 'm+NAP', 'long_name': 'z'}        
    else:
        ds = ds.assign_coords({'z': -dist})
        ds['z'].attrs = {'units': 'm', 'long_name': 'dist from transducer'}   

    d0 = 0.1
    gamma = 120/180*np.pi
    ds['alpha'] = ('z', np.pi/2-np.arctan2((dist-d0*np.cos(gamma)), (d0*np.sin(gamma)))) # positive alpha values
    # ds['alpha'] = ('z', -np.pi/2+np.arctan2((dist-d0*np.cos(gamma)), (d0*np.sin(gamma))))  # negative alpha values 
    
    if coarsen is True:
        ds = ds.coarsen(z=ncoarsenz).mean()

    u = []; v = []; w1 = []; w2 = []
    for i in range(len(ds.z)):
        dsi = ds.isel(z=i)

        if remove_T1:
            u.append((2*dsi.v2-dsi.v3-dsi.v4)/np.sin(dsi['alpha']))
            w1.append((dsi.v4+dsi.v3)/(1+np.cos(dsi['alpha'])))
            v.append((dsi.v4-dsi.v3)/np.sin(dsi['alpha']))
            w2.append((dsi.v4+dsi.v3)/(1+np.cos(dsi['alpha'])))
        else:
            u.append((dsi.v2-dsi.v1)/np.sin(dsi['alpha']))
            w1.append((dsi.v2+dsi.v1)/(1+np.cos(dsi['alpha'])))
            v.append((dsi.v4-dsi.v3)/np.sin(dsi['alpha']))
            w2.append((dsi.v4+dsi.v3)/(1+np.cos(dsi['alpha'])))

    ds['u'] = (('z','time'), np.array(u))
    ds['w1'] = (('z','time'), np.array(w1))
    ds['v'] = (('z','time'), np.array(v))
    ds['w2'] = (('z','time'), np.array(w2))
    ds['u'].attrs = {'units':'m/s', 'long_name': 'u-cross'}
    ds['v'].attrs = {'units':'m/s', 'long_name': 'u-along'}
    ds['w1'].attrs = {'units':'m/s', 'long_name': 'u-up1'}
    ds['w2'].attrs = {'units':'m/s', 'long_name': 'u-up2'}

    urlist = []; w1rlist = []; vrlist = []; w2rlist = [];
    for i in range(len(ds.z)):
        dsi = ds.isel(z=i)

        # rotation in the cross-shore direction over angle thet1, rotation in the alongshore direction over angle thet2. 
        # Always this order: first rotate over thet1, then thet2
        # R_2 * R_1 * [u,v,w].T = 
        matrix2rots = [[np.cos(theta1),                     0,                 np.sin(theta1)                  ], 
                       [-np.sin(theta1)*np.sin(theta2),     np.cos(theta2),    np.sin(theta1)*np.sin(theta2)   ], 
                       [-np.sin(theta1)*np.cos(theta2),     -np.sin(theta2),   np.cos(theta1)*np.cos(theta2)   ]]
        
        # estimate ur with the first vertical measurement
        ur, _, w1r = list(np.matmul(matrix2rots, np.array([dsi.u.values, dsi.v.values, dsi.w1.values])))

        # and vr with the second vertical measurement 
        _, vr, w2r = list(np.matmul(matrix2rots, np.array([dsi.u.values, dsi.v.values, dsi.w2.values])))
        
        urlist.append(ur)
        w1rlist.append(w1r)
        vrlist.append(vr)
        w2rlist.append(w2r)

    ds['ur'] = (('z','time'), np.array(urlist))
    ds['w1r'] = (('z','time'), np.array(w1rlist))
    ds['vr'] = (('z','time'), np.array(vrlist))
    ds['w2r'] = (('z','time'), np.array(w2rlist))
    ds['ur'].attrs = {'units':'m/s', 'long_name': 'u-cross', 'comment': 'rotation corrected'}
    ds['w1r'].attrs = {'units':'m/s', 'long_name': 'u-up1', 'comment': 'rotation corrected'}
    ds['vr'].attrs = {'units':'m/s', 'long_name': 'u-along', 'comment': 'rotation corrected'}
    ds['w2r'].attrs = {'units':'m/s', 'long_name': 'u-up2', 'comment': 'rotation corrected'}
    return ds

def load_urms_ublab(snrmin=2, thet1=-30, thet2=0, recompute=False):

    if recompute is False:
        try:
            ublab_rms = xr.open_dataset(os.path.join(
                r'\\tudelft.net\staff-umbrella\EURECCA\01AnalyseRemote\data\ublab',
                'ublab_urms_{:.0f}deg_{:.0f}deg_5min.nc'.format(np.abs(thet1), np.abs(thet2)))
            )
            
        except:
            print('data not yet saved to file, recompute it')
            ublab_rms = load_urms_ublab(snrmin=snrmin, thet1=thet1, thet2=thet2, recompute=True)

    else:

        folder = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\ublab-3c\dep1\raw_netcdf'
        filez = glob.glob(os.path.join(folder, '*.nc'))

        tlist = []
        urmslist = []
        vrmslist = []
        wrmslist = []
        wrms2list = []
        for file in filez[14:]:
            print(file)
            ds = load_adcv_corrected(file, thet1=thet1, thet2=thet2)
            tmin = ds.time.min().values
            tmax = ds.time.max().values
            tstarts = pd.date_range(tmin, tmax, freq='490s')

            for tstart in tstarts:
                dst = ds.sel(time=slice(tstart, tstart+pd.to_timedelta('490s')))
                
                if not len(dst.time)>0:
                    continue

                tlist.append(tstart)

                urms = np.zeros([82])
                vrms = np.zeros([82])
                wrms = np.zeros([82])
                wrms2 = np.zeros([82])
                for iz in range(82):
                    dstz = dst.isel(z=iz)

                    u_c_bpf = puv.band_pass_filter2(51, dstz.ur.values, fmin=0.1,fmax=2, retrend=False)
                    v_c_bpf = puv.band_pass_filter2(51, dstz.vr.values, fmin=0.1,fmax=2, retrend=False)
                    w1_c_bpf = puv.band_pass_filter2(51, dstz.w1r.values, fmin=0.1,fmax=2, retrend=False)
                    w2_c_bpf = puv.band_pass_filter2(51, dstz.w2r.values, fmin=0.1,fmax=2, retrend=False)  

                    urms[iz] = np.sqrt(np.mean(u_c_bpf**2))
                    vrms[iz] = np.sqrt(np.mean(v_c_bpf**2))
                    wrms[iz] = np.sqrt(np.mean(w1_c_bpf**2))
                    wrms2[iz] = np.sqrt(np.mean(w2_c_bpf**2))


                urmslist.append(urms)
                vrmslist.append(vrms)
                wrmslist.append(wrms)
                wrms2list.append(wrms2)
                    
        #% cast to xarray structure
        wrms_ub = np.vstack(wrmslist).T
        urms_ub = np.vstack(urmslist).T
        vrms_ub = np.vstack(vrmslist).T
        wrms2_ub = np.vstack(wrms2list).T

        ublab_rms = xr.Dataset({}, coords = {'t': tlist, 'z': ds.z.values[:82]})
        ublab_rms['w'] = (('z', 't'), wrms_ub)
        ublab_rms['w2'] = (('z', 't'), wrms2_ub)
        ublab_rms['u'] = (('z', 't'), urms_ub)
        ublab_rms['v'] = (('z', 't'), vrms_ub)
        ublab_rms['z'].attrs = {'units': 'm', 'long_name': 'z'}
        ublab_rms['u'].attrs = {'units': 'm/s', 'long_name': 'rms orbital u', 'comments': 'bpf [0.05-2] Hz'}
        ublab_rms['v'].attrs = {'units': 'm/s', 'long_name': 'rms orbital v', 'comments': 'bpf [0.05-2] Hz'}   
        ublab_rms['w'].attrs = {'units': 'm/s', 'long_name': 'rms orbital w', 'comments': 'bpf [0.05-2] Hz'}            
        ublab_rms['w2'].attrs = {'units': 'm/s', 'long_name': 'rms orbital w2', 'comments': 'bpf [0.05-2] Hz'}    

        ublab_rms.to_netcdf(os.path.join(
                r'\\tudelft.net\staff-umbrella\EURECCA\01AnalyseRemote\data\ublab',
                'ublab_urms_{:.0f}deg_{:.0f}deg_5min.nc'.format(np.abs(thet1), np.abs(thet2)))
            )

    return ublab_rms

def load_skas_ublab(snrmin=2, thet1=-30, thet2=0, recompute=False):

    if recompute is False:
        try:
            ublab_skas = xr.open_dataset(os.path.join(
                r'\\tudelft.net\staff-umbrella\EURECCA\01AnalyseRemote\data\ublab',
                'ublab_skas_{:.0f}deg_{:.0f}deg_5min.nc'.format(np.abs(thet1), np.abs(thet2)))
            )
        except:
            print('data not yet saved to file, recompute it')
            ublab_skas = load_skas_ublab(snrmin=snrmin, thet1=thet1, thet2=thet2, recompute=True)

    else:

        folder = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\ublab-3c\dep1\raw_netcdf'
        filez = glob.glob(os.path.join(folder, '*.nc'))

        tlist = []
        siglist = []
        usklist = []
        uaslist = []
        # for file in filez[14:]:
        for file in filez[14:19]:
            print(file)
            ds = load_adcv_corrected(file, thet1=thet1, thet2=thet2)
            tmin = ds.time.min().values
            tmax = ds.time.max().values
            tstarts = pd.date_range(tmin, tmax, freq='490s')
            # tstarts = pd.date_range(tmin, tmax, freq='900s')
            for tstart in tstarts:
                dst = ds.sel(time=slice(tstart, tstart+pd.to_timedelta('490s')))
                # dst = ds.sel(time=slice(tstart, tstart+pd.to_timedelta('900s')))
                if not len(dst.time)>0:
                    continue

                tlist.append(tstart)

                sig = np.zeros([82])
                usk = np.zeros([82])
                uas = np.zeros([82])

                for iz in range(82):
                    dstz = dst.isel(z=iz)

                    ur = dstz.ur.where(dstz.snr1>snrmin).where(dstz.snr2>snrmin).coarsen(time=8, boundary='trim').mean()
                    sf_coarse = np.timedelta64(1,'s')/np.diff(ur.time).mean()
                    
                    ur_filled = ur.interpolate_na(dim='time').dropna(dim='time')

                    if len(ur_filled)>60:
                        # u_c_bpf = puv.band_pass_filter2(sf_coarse, ur_filled.values, fmin=0.1,fmax=2, retrend=False)

                        # usk[iz], uas[iz], sig[iz] = puv.compute_SkAs(51, u_c_bpf, fpfac=[0.5, 2])
                        usk[iz], uas[iz], sig[iz] = puv.compute_SkAs(sf_coarse, ur_filled.values, fpfac=[0.5, 2])
                    else:
                        usk[iz] = uas[iz] = sig[iz] = np.nan


                siglist.append(sig)
                usklist.append(usk)
                uaslist.append(uas)
                    
        #% cast to xarray structure
        sig_ub = np.vstack(siglist).T
        usk_ub = np.vstack(usklist).T
        uas_ub = np.vstack(uaslist).T

        ublab_skas = xr.Dataset({}, coords = {'t': tlist, 'z': ds.z.values[:82]})
        ublab_skas['sig'] = (('z', 't'), sig_ub)
        ublab_skas['Sk'] = (('z', 't'), usk_ub)
        ublab_skas['As'] = (('z', 't'), uas_ub)
        ublab_skas['z'].attrs = {'units': 'm', 'long_name': 'z'}
        ublab_skas['sig'].attrs = {'units': 'm/s', 'long_name': 'sigma(u)', 'comments': '0.5fp-2fp Hz'}
        ublab_skas['Sk'].attrs = {'units': 'm^3', 'long_name': 'Su', 'comments': '0.5fp-2fp Hz'}   
        ublab_skas['As'].attrs = {'units': 'm^3/s^3', 'long_name': 'Au', 'comments': '0.5fp-2fp Hz'}            

        ublab_skas.to_netcdf(os.path.join(
                r'\\tudelft.net\staff-umbrella\EURECCA\01AnalyseRemote\data\ublab',
                'ublab_skas_{:.0f}deg_{:.0f}deg_5min.nc'.format(np.abs(thet1), np.abs(thet2)))
            )

    return ublab_skas

def load_adv_urms_5min():
    adv0 = xr.open_dataset(r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\vec008\qc4\vec008.nc')
    adv = xr_bursts_to_series(adv0)

    beachOri = 122
    uc, ul = puv.rotate_velocities(adv.u.values, adv.v.values, beachOri)
    adv['uc'] = (('time'), uc) 
    adv['ul'] = (('time'), ul)
    adv['uc'].attrs = {'units': 'm/s', 'long_name': 'u_cross',
                            'comments': 'velocity in cross-shore direction'}
    adv['ul'].attrs = {'units': 'm/s', 'long_name': 'v_long',
                            'comments': 'velocity in alongshore direction'}


    tmin = adv.time.min().values
    tmax = '20231102 05:00'

    tlist = []
    urmslist = []
    vrmslist = []
    wrmslist = []
    tstarts = pd.date_range(tmin, tmax, freq='490s')
    for tstart in tstarts:
        dst = adv.sel(time=slice(tstart, tstart+pd.to_timedelta('490s')))
        
        if not len(dst.time)>0:
            continue

        if np.sum(np.isnan(dst.uc))>0:
            tlist.append(tstart)
            urmslist.append(np.nan)
            vrmslist.append(np.nan)
            wrmslist.append(np.nan)
            continue

        tlist.append(tstart)

        u_c_bpf = puv.band_pass_filter2(16, dst.uc.values, fmin=0.1, fmax=2, retrend=False)
        v_c_bpf = puv.band_pass_filter2(16, dst.ul.values, fmin=0.1, fmax=2, retrend=False)
        w_c_bpf = puv.band_pass_filter2(16, dst.w.values, fmin=0.1, fmax=2, retrend=False)

        urms = np.sqrt(np.mean(u_c_bpf**2))
        vrms = np.sqrt(np.mean(v_c_bpf**2))
        wrms = np.sqrt(np.mean(w_c_bpf**2))

        urmslist.append(urms)
        vrmslist.append(vrms)
        wrmslist.append(wrms)

    # cast to xarray structure
    wrms_adv = np.array(wrmslist)
    urms_adv = np.array(urmslist)
    vrms_adv = np.array(vrmslist)
    adv_rms = xr.Dataset({}, coords = {'t': tlist})
    adv_rms['w'] = (('t'), wrms_adv)
    adv_rms['u'] = (('t'), urms_adv)
    adv_rms['v'] = (('t'), vrms_adv)
    adv_rms['u'].attrs = {'units': 'm/s', 'long_name': 'rms orbital u', 'comments': 'bpf [0.1-2] Hz'}
    adv_rms['v'].attrs = {'units': 'm/s', 'long_name': 'rms orbital v', 'comments': 'bpf [0.1-2] Hz'}
    adv_rms['w'].attrs = {'units': 'm/s', 'long_name': 'rms orbital w', 'comments': 'bpf [0.1-2] Hz'}

    return adv_rms

def ublab_profiles3_u(ublab, z_zb2, var='ur', timestamps=None, coarsenfac=5, factor=1, **kwargs):
    """
    plots the tz diagram of ublab[var] and on top plots profiles on timestamps 
    that are coarsened by a factor "coarsenfac" in time
    """

    t0 = pd.to_datetime('20231102 00:15')

    # the profiles plotted are based on a smoothed moving mean of factor 5
    umm = ublab[var].coarsen(time=coarsenfac, boundary='trim').mean()
    u_x = (pd.to_datetime(ublab.time.coarsen(time=coarsenfac, boundary='trim').mean().values)-t0)/pd.to_timedelta('1s')

    fig, ax = plt.subplots(figsize=[8, 3])

    #plot bed level
    zb_x = (pd.to_datetime(z_zb2.t.values)-t0)/pd.to_timedelta('1s')
    zb_y =  z_zb2.values
    ax.plot(zb_x, zb_y, color='k', linewidth=0.5)
    ax.set_ylabel('z [m]')
    # plot velocity

    # x-axis uses seconds instead of datetime to enable later plotting of profiles on top
    # u_x = (pd.to_datetime(ublab.time.values)-t0)/pd.to_timedelta('1s')
    u_y = ublab.z.values
    u_z = umm.values  # ublab[var].values
    # im = ax.pcolor(u_x, u_y, u_z, shading='nearest', **kwargs)
    # plt.colorbar(im, label='<{}> [m/s]'.format(var))

    ax.set_xlim(0, u_x.max())
    ax.set_ylim(bottom=-0.30)
    ax.set_xlabel('time [s]')

    tslist = []

    # none specified, take the standard timestamps for plotting of profiles
    if timestamps is None:
        # timestamps = ['00:30','00:45','01:00','01:15', '01:30', '01:45', '02:00', '02:15']
        timestamps = ['00:25','00:40','00:55','01:10', '01:25', '01:40', '01:55', '02:10']

    for ix, strtime in enumerate(timestamps):
        time = pd.to_datetime('20231102 {}'.format(strtime))

        # x-origin of the profile
        ts = (time-t0)/pd.to_timedelta('1s')
        tslist.append(ts)
        ummp = umm.sel(time=time, method='nearest')
        zbix = z_zb2.sel(t=ummp.time.values, method='nearest')
        ummp = ummp.where(ummp.z>zbix)

        # 0.1 m/s is set to 15 minutes  
        x = ts+(ummp.values*15*60*10)/factor
        y = ummp.z.values.copy()
        # remove below bed
        zb_iy = z_zb2.sel(t=time, method='nearest').values
        y[u_y<zb_iy] = np.nan

        plt.scatter(x,y,marker='o', edgecolor='k', facecolor='None',s=2)
        plt.axvline(ts, color='k', linewidth=0.5, zorder=1)

    ax.set_xticks(tslist)
    ax.set_xticklabels(timestamps)
    ax.fill_between(zb_x, zb_y, -0.4, color='white')

    tline0 = (pd.to_datetime('20231102 01:40')-t0)/pd.to_timedelta('1s')
    tline1 = (pd.to_datetime('20231102 01:45')-t0)/pd.to_timedelta('1s')
    tline = [tline0, tline1]
    zline = [-0.275, -0.275]
    ax.plot(tline, zline, color='k') #tline[0], zline[0], tline[1]-tline[0], zline[1]-zline[0], color='k', head_width=0.01, width=0.001, overhang=10)
    ax.text(tline[0], -0.285, '{:.2f} m/s'.format(0.3/factor), ha='left', va='top')
    return fig, ax