# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:36:11 2021

@author: marliesvanderl
"""
import glob
import os
import xarray as xr
import numpy as np
import pandas as pd
import puv


# quality check calibration factors
# velocity limits based on settings in the instrument
qc = pd.DataFrame(
    {
     'uLim': [2.1, 2.1, 2.1, 2.1, 2.1, 1.5, 1.5, 2.1, 2.1, 2.1, 2.1, 2.1],  # modified to be 1.5 and not 0.6 for horizontal vectors, not saved to file yet! 10/12
     'vLim': [2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1],
     'wLim': [0.6, 0.6, 0.6, 0.6, 0.6, 1.5, 1.5, 0.6, 0.6, 0.6, 0.6, 0.6],
     'corTreshold': 12*[50],
     'ampTreshold': 5*[80] + 2*[40] + 2*[80] + 3*[100],  # horizontal vectors had lower returns than downwardlooking ones
     'maxFracNans': 12*[0.05],
     'maxGap': 12*[10],
     'waterHeadMargin': 12*[0.05]
      },
    index=['L1C1VEC', 'L2C3VEC', 'L3C1VEC', 'L5C1VEC', 'L6C1VEC', 'L2C2VEC', 'L2C4VEC', 'L2C10VEC', 'L4C1VEC', 'L2C5SONTEK1', 'L2C5SONTEK2', 'L2C5SONTEK3']
    )

def hor2vert_vector_mapping(u, v, w):
    '''
    function to permute the ADV velocities from a horizontally deployed ADV to the standard uv horizontal and w pos
    upward velocities
    :param u:
    :param v:
    :param w:
    :return: u, v, w
    '''
    mapMatrix = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    coords = np.vstack((u, v, w))
    u, v, w = mapMatrix @coords
    return u, v, w

def resample_quality_check_replace_dataset(ds):
    import xrMethodAccessors

    # we reshape the dataset to blocks of blockLength seconds.
    if 'SONTEK' in ds.name:
        # we distribute the missing 60 seconds every half hour evenly over the three 10min blocks
        ds = ds.burst.reduce_burst_length(3)
    else:
        # only the last of the 3 blocks contains 10 secs of missing data
        ds = ds.burst.reshape_burst_length(600)

    # correct for the air pressure fluctuations and drift in the instrument
    # by using the pressure of the standalone solo's to match the high water pressures
    # we loose the water level gradients
    pres = xr.open_dataset(os.path.join(experimentFolder, 'waterlevel.nc'))

    # half hour average pressure, on similar time-axis as ADV data
    presref = pres.p.interp_like(ds)

    # correct the raw pressure signal by
    ds['pc'] = ds.p - ds.p.mean(dim='N') + presref
    ds['pc'].attrs = {'units': 'Pa', 'long_name': 'pressure',
                      'comments': 'referenced to pressure L2C9OSSI'}

    # compute mean water level, water depth
    ds['zi'] = ds['zb'] + ds['h'] / 100
    ds['zi'].attrs = {'units': 'm+NAP', 'long_name': 'position ADV control volume'}

    ds['zip'] = ds['zb'] + ds['hpres'] / 100
    ds['zip'].attrs = {'units': 'm+NAP', 'long_name': 'position pressure sensor'}

    ds['eta'] = ds['pc'] / rho / g
    ds['eta'].attrs = {'units': 'm+NAP', 'long_name': 'hydrostatic water level'}

    ds['d'] = ds.eta.mean(dim='N') - ds.zb
    ds['d'].attrs = {'units': 'm ', 'long_name': 'water depth'}
    ds['elev'] = ds.zi - ds.zb
    ds['elev'].attrs = {'units': 'm ', 'long_name': 'height probe control volume above bed'}
    ds['elevp'] = ds.zip - ds.zb
    ds['elevp'].attrs = {'units': 'm ', 'long_name': 'height ptd above bed'}

    # if amplitude is too low, the probe was emerged
    ma1 = ds.a1 > QC['ampTreshold']
    ma2 = ds.a2 > QC['ampTreshold']
    ma3 = ds.a3 > QC['ampTreshold']

    # if correlation is outside confidence range
    if QC['corTreshold'] == 'elgar':
        # estimate correlation threshold, Elgar et al., 2005; pp. 1891
        sf = ds.sf.values
        if len(sf) > 0:
            sf = sf[0]
            criticalCorrelation = (0.3 + 0.4 * np.sqrt(sf / 25)) * 100
    else:
        criticalCorrelation = QC['corTreshold']

    mc1 = ds.cor1 > criticalCorrelation
    mc2 = ds.cor2 > criticalCorrelation
    mc3 = ds.cor3 > criticalCorrelation

    # if observation is outside of velocity range
    mu1 = np.abs(ds.u) < QC['uLim']
    mu2 = np.abs(ds.v) < QC['uLim']
    mu3 = np.abs(ds.w) < QC['uLim']

    # if du larger than 4*std(u) then we consider it outlier and hence remove:
    md1 = np.abs(ds.u.diff('N')) < 3 * ds.u.std(dim='N')
    md2 = np.abs(ds.v.diff('N')) < 3 * ds.v.std(dim='N')
    md3 = np.abs(ds.w.diff('N')) < 3 * ds.w.std(dim='N')

    ds['mc'] = np.logical_and(np.logical_and(mc1, mc2), mc3)
    ds['mu'] = np.logical_and(np.logical_and(mu1, mu2), mu3)
    ds['md'] = np.logical_and(np.logical_and(md1, md2), md3)
    ds['ma'] = np.logical_and(np.logical_and(ma1, ma2), ma3)
    ds['mc'].attrs = {'units': '-', 'long_name': 'mask correlation'}
    ds['mu'].attrs = {'units': '-', 'long_name': 'mask velocity limit'}
    ds['md'].attrs = {'units': '-', 'long_name': 'mask unexpected jumps in signal'}
    ds['ma'].attrs = {'units': '-', 'long_name': 'mask amplitude'}

    mp = np.abs(ds.p.diff('N')) < 4 * ds.p.std(dim='N')
    mp = xr.concat([mp.isel(N=0), mp], dim="N")

    ds.coords['maskp'] = (('t', 'N'), mp.values)
    ds.coords['maskd'] = (('t', 'N'), ds.ma.values)
    ds.coords['maskv'] = (('t', 'N'), (np.logical_and(np.logical_and(ds.mc, ds.mu), ds.md)).values)

    ######################################################################
    # fill masked observations with interpolation
    ######################################################################

    # set outliers to nan
    ds['u'] = ds.u.where(ds.maskv).where(ds.maskd)
    ds['v'] = ds.v.where(ds.maskv).where(ds.maskd)
    ds['eta'] = ds.eta.where(ds.maskp).where(ds.maskd)

    # drop bursts with too many nans (>maxFracNans) in the velocity data
    ds2 = ds.where( np.isnan(ds.u).sum(dim='N') < QC['maxFracNans'] * len(ds.N)).dropna(dim='t', how='all')
    ds2['sf'] = ds['sf']

    if len(ds2.t) == 0:
        print('no valid data remains on this day')
        return ds2

    # interpolate nans
    for var in ['u', 'v', 'w', 'eta']:
        ds[var] = ds2[var].interpolate_na(
            dim='N',
            method='cubic',
            max_gap=QC['maxGap'])

        # and fill the gaps more than maxGap in length with the burst average
        ds2[var] = ds2[var].fillna(ds2[var].mean(dim='N'))

    ds = ds2


    # rename coordinates if horizontally placed
    if ((instrumentName == 'L2C2VEC') | (instrumentName == 'L2C4VEC')):
        ufunc = lambda u, v, w: hor2vert_vector_mapping(u, v, w)
        ds['u'], ds['v'], ds['w'] = xr.apply_ufunc(ufunc,
                                                   ds['u'], ds['v'], ds['w'],
                                                   input_core_dims=[['N'], ['N'], ['N']],
                                                   output_core_dims=[['N'], ['N'], ['N']],
                                                   vectorize=True)

    # UU instruments flexheads so downward pos instead of upward pos
    if (instrumentName in ['L1C1VEC', 'L2C3VEC', 'L3C1VEC', 'L5C1VEC', 'L6C1VEC']):
        ds['v'] = -ds['v']
        ds['w'] = -ds['w']

    # rotate to ENU coordinates
    ufunc = lambda u, v, thet: puv.rotate_velocities(u, v, thet - 90)
    ds['u'], ds['v'] = xr.apply_ufunc(ufunc,
                                      ds['u'], ds['v'], ds['io'],
                                      input_core_dims=[['N'], ['N'], []],
                                      output_core_dims=[['N'], ['N']],
                                      vectorize=True)
    ds['u'].attrs = {'units': 'm/s', 'long_name': 'velocity E'}
    ds['v'].attrs = {'units': 'm/s', 'long_name': 'velocity N'}
    ds['w'].attrs = {'units': 'm/s', 'long_name': 'velocity U'}

    # saving
    ds.attrs['version'] = 'v3'
    ds.attrs['comment'] = 'Quality checked data: pressure reference level corrected for airpressure drift,' \
                          ' correlation checks done and spikes were removed. ' \
                          'Velocities rotated to ENU coordinates based on heading and configuration in the field.' \
                          'data that was marked unfit has been removed or replaced by interpolation.'

    # save to netCDF
    # all variables that are only used for the QC are block averaged to reduce amount of info on QC files
    ds['a1'] = ds.a1.mean(dim='N')
    ds['a1'].attrs = {'units': '-', 'long_name': 'block averaged amplitude beam 1'}
    ds['a2'] = ds.a2.mean(dim='N')
    ds['a2'].attrs = {'units': '-', 'long_name': 'block averaged amplitude beam 2'}
    ds['a3'] = ds.a3.mean(dim='N')
    ds['a3'].attrs = {'units': '-', 'long_name': 'block averaged amplitude beam 3'}
    ds['cor1'] = ds.cor1.mean(dim='N')
    ds['cor1'].attrs = {'units': '-', 'long_name': 'block averaged correlation beam 1'}
    ds['cor2'] = ds.cor2.mean(dim='N')
    ds['cor2'].attrs = {'units': '-', 'long_name': 'block averaged correlation beam 2'}
    ds['cor3'] = ds.cor3.mean(dim='N')
    ds['cor3'].attrs = {'units': '-', 'long_name': 'block averaged correlation beam 3'}

    try:
        ds['snr1'] = ds.snr1.mean(dim='N')
        ds['snr1'].attrs = {'units': '-', 'long_name': 'block averaged SNR beam 1'}
        ds['snr2'] = ds.snr2.mean(dim='N')
        ds['snr2'].attrs = {'units': '-', 'long_name': 'block averaged SNR beam 2'}
        ds['snr3'] = ds.snr3.mean(dim='N')
        ds['snr3'].attrs = {'units': '-', 'long_name': 'block averaged SNR beam 3'}
    except:
        print('no SNR data on this instrument')

    # we no longer need these:
    ds = ds.drop_vars(['heading', 'pitch', 'roll',
                  'voltage', 'pc'], errors='ignore')

    return ds
if __name__ == "__main__":
    experimentFolder = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20210908_campaign\instruments'
    # experimentFolder = r'u:\staff-umbrella\EURECCA\fieldvisits\20210908_campaign\instruments'
    folderinName = 'raw_netcdf'
    folderoutName = 'qc'
    rho = 1028
    g = 9.8

    instruments = [
             'L1C1VEC',
             'L2C3VEC',
             'L3C1VEC',
             'L5C1VEC',
             'L6C1VEC',
             'L2C2VEC',
             'L2C4VEC',
             'L2C10VEC',
             'L4C1VEC',
             'L2C5SONTEK1',
             'L2C5SONTEK2',
             'L2C5SONTEK3'
            ]

    for instrumentName in instruments:
        print(instrumentName)

        # extract the instrument specific quality criteria
        QC = qc.loc[instrumentName]


        # find all raw data on file for this instrument
        fileNames = glob.glob(os.path.join(experimentFolder, instrumentName, folderinName, '*.nc'))

        # ensure the output folder exists
        fold = os.path.join(experimentFolder, instrumentName, folderoutName)
        if not os.path.isdir(fold):
            os.mkdir(fold)

        # loop over all the raw data files
        for file in fileNames:
            print(file)

            with xr.open_dataset(file) as ds:

                # do the actual work
                ds = resample_quality_check_replace_dataset(ds)

                # save to file
                if len(ds.t) > 0:

                    # what is the minimum and maximum day represented in the data?
                    t0_string = ds.t.isel(t=0).dt.strftime('%Y%m%d').values.flatten()[0]
                    tend_string = ds.t.isel(t=-1).dt.strftime('%Y%m%d').values.flatten()[0]

                    # one netcdf per day, therefore make a list of dates with pandas
                    dates = pd.date_range(t0_string, tend_string, freq='1D')
                    dates = [str(date)[:10].replace('-', '') for date in dates]

                    # we split data to one netcdf per day. For SONTEK instruments, this is not the default structure
                    # therefore,  split into several netcdf's when necessary
                    for date in dates:
                        ds_sel = ds.sel(t=date)

                        if len(ds.t) > 0:
                            ncFilePath = os.path.join(fold, '{}_{}.nc'.format(ds.name, date))

                            # if there is already a file on the drive, merge the results
                            if os.path.exists(ncFilePath):
                                nc_onfile = xr.load_dataset(ncFilePath)
                                ds_merge = xr.merge([nc_onfile, ds_sel])
                            else:
                                ds_merge = ds_sel

                            # write to file
                            # specify compression for all the variables to reduce file size
                            comp = dict(zlib=True, complevel=5)
                            ds_merge.encoding = {var: comp for var in ds_merge.data_vars}
                            ds_merge.encoding['t'] = {'zlib': False, '_FillValue': None}
                            ds_merge.encoding['N'] = {'zlib': False, '_FillValue': None}

                            ds_merge.to_netcdf(ncFilePath, encoding=ds_merge.encoding)





