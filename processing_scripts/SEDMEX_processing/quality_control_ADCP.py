import os
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import puv
from sedmex_info_loaders import get_githash

def qc_this_rawdatafile(instrument, heading, part, config):
    ds = xr.open_dataset(os.path.join(config['experimentFolder'], instrument, r'raw_netcdf\part' + part + '.nc'))

    # what range of dates sits in this part of the datase?
    ds_tmin = ds.t.min().dt.floor('1D').values
    ds_tmax = ds.t.max().dt.floor('1D').values

    # we do the work 1 day per time
    for tstart in pd.date_range(ds_tmin, ds_tmax):

        tstart_string = str(tstart)[:10].replace('-', '')
        print(tstart_string)

        tstop = tstart + pd.to_timedelta('1D')

        ds = xr.open_dataset(os.path.join(config['experimentFolder'], instrument, r'raw_netcdf\part' + part + '.nc'))
        ds = ds.sel(t=slice(tstart, tstop))

        # check burstDuration
        if config['burstDuration'] != 1800:
            print('ADCP bursts other than 1800 are not implemented')

        # compute quality criteria
        crit_amp = config['qcADCPSettings']['ampTreshold']
        ma = np.logical_and(np.logical_and(ds.a1 > crit_amp, ds.a2 > crit_amp), ds.a3 > crit_amp)
        crit_cor = config['qcADCPSettings']['corTreshold']
        mc = np.logical_and(np.logical_and(ds.c1 > crit_cor, ds.c2 > crit_cor), ds.c3 > crit_cor)
        umag = np.sqrt(ds.v1 ** 2 + ds.v2 ** 2 + ds.v3 ** 2)
        crit_v = config['qcADCPSettings']['uLim']
        mv = umag < crit_v

        # if du larger than xx*std(u) then we consider it outlier and hence remove:
        crit_outlier = config['qcADCPSettings']['outlierCrit']
        md1 = np.abs(ds.v1.diff('N')) < crit_outlier * ds.v1.std(dim='N')
        md2 = np.abs(ds.v2.diff('N')) < crit_outlier * ds.v2.std(dim='N')
        md3 = np.abs(ds.v3.diff('N')) < crit_outlier * ds.v3.std(dim='N')
        md = np.logical_and(np.logical_and(md1, md2), md3)

        qc_maxfracnans = config['qcADCPSettings']['maxFracNans']
        for var in ['v1', 'v2', 'v3']:
            # mask bad samples, ignoring the correlation
            ds[var] = ds[var].where(ma).where(mv).where(md)  # .where(mc)
            # drop bursts with too many nans (>maxFracNans) in the velocity data
            ds[var] = ds[var].where(np.isnan(ds[var]).sum(dim='N') < qc_maxfracnans * len(ds.N))
            # delete all cells below the bed level
            ds[var] = ds[var].where(ds.z>ds.zb)

        # drop pressure data on bursts that where disqualified based on velocity in top cell
        ds['p'] = ds['p'].where(np.isnan(ds['v1'].isel(z=0)).sum(dim='N') < qc_maxfracnans * len(ds.N))

        # interpolate nans in velocity data
        N = len(ds.N)
        qc_maxgap = config['qcADCPSettings']['maxGap']
        for var in ['v1', 'v2', 'v3']:
            if len(ds.t) != 0:
                ds[var] = ds[var].interpolate_na(
                    dim='N',
                    method='cubic',
                    max_gap=qc_maxgap)

            # and fill the gaps more than maxGap in length with the burst average
            ds[var] = ds[var].fillna(ds[var].mean(dim='N'))

        # ADCP downward positioned instead of upward pos so, therefore flip coordinate system
        ds['v2'] = -ds['v2']
        ds['v3'] = -ds['v3']

        # rotate to ENU coordinates
        ufunc = lambda u, v: puv.rotate_velocities(u, v, heading - 90)
        ds['u'], ds['v'] = xr.apply_ufunc(ufunc,
                                          ds['v1'], ds['v2'],
                                          input_core_dims=[['N'], ['N']],
                                          output_core_dims=[['N'], ['N']],
                                          vectorize=True)
        ds['w'] = ds['v3']
        ds['u'].attrs = {'units': 'm/s', 'long_name': 'velocity E'}
        ds['v'].attrs = {'units': 'm/s', 'long_name': 'velocity N'}
        ds['w'].attrs = {'units': 'm/s', 'long_name': 'velocity U'}

        # correct for the air pressure fluctuations and drift in the instrument
        # by using the pressure of the standalone solo's to match the high water pressures
        # we loose the water level gradients
        pres = xr.open_dataset(os.path.join(config['experimentFolder'], 'waterlevel.nc'))

        # half hour average pressure, on similar time-axis as ADV data
        presref = pres.p.interp_like(ds)

        # correct the raw pressure signal by
        ds['pc'] = ds.p - ds.p.mean(dim='N') + presref
        ds['pc'].attrs = {'units': 'Pa', 'long_name': 'pressure',
                          'comments': 'referenced to pressure L2C10SOLO/L2C9OSSI'}

        ds['eta'] = ds['pc'] / config['physicalConstants']['rho'] / config['physicalConstants']['g']
        ds['eta'].attrs = {'units': 'm+NAP', 'long_name': 'hydrostatic water level'}

        ds['d'] = ds.eta.mean(dim='N') - ds.zb
        ds['d'].attrs = {'units': 'm ', 'long_name': 'water depth'}


        # make zb a function of time
        ds['zb'] = (('t'), ds.zb.values * np.ones(len(ds.t.values)))
        ds['zb'].attrs = {'units': 'm+NAP ', 'long_name': 'bed level'}
        ds['zi'] = (('t'), ds.zi.values * np.ones(len(ds.t.values)))
        ds['zi'].attrs = {'units': 'm+NAP ', 'long_name': 'instrument height'}

        ds['elev'] = ds.zi - ds.zb
        ds['elev'].attrs = {'units': 'm ', 'long_name': 'height probe control volume above bed'}

        # save to netCDF
        # all variables that are only used for the QC are block averaged to reduce amount of info on QC files
        ds['temp'] = ds.temp.mean(dim='N')
        ds['temp'].attrs = {'units': 'deg C', 'long_name': 'temperature'}
        ds['a1'] = ds.a1.mean(dim='N')
        ds['a1'].attrs = {'units': '-', 'long_name': 'block averaged amplitude beam 1'}
        ds['a2'] = ds.a2.mean(dim='N')
        ds['a2'].attrs = {'units': '-', 'long_name': 'block averaged amplitude beam 2'}
        ds['a3'] = ds.a3.mean(dim='N')
        ds['a3'].attrs = {'units': '-', 'long_name': 'block averaged amplitude beam 3'}
        ds['c1'] = ds.c1.mean(dim='N')
        ds['c1'].attrs = {'units': '-', 'long_name': 'block averaged correlation beam 1'}
        ds['c2'] = ds.c2.mean(dim='N')
        ds['c2'].attrs = {'units': '-', 'long_name': 'block averaged correlation beam 2'}
        ds['c3'] = ds.c3.mean(dim='N')
        ds['c3'].attrs = {'units': '-', 'long_name': 'block averaged correlation beam 3'}
        ds['hr2c1'] = ds.hr2c1.mean(dim='N')
        ds['hr2c1'].attrs = {'units': '-', 'long_name': 'block averaged hr2 correlation beam 1'}
        ds['hr2c2'] = ds.hr2c2.mean(dim='N')
        ds['hr2c2'].attrs = {'units': '-', 'long_name': 'block averaged hr2 correlation beam 2'}
        ds['hr2c3'] = ds.hr2c3.mean(dim='N')
        ds['hr2c3'].attrs = {'units': '-', 'long_name': 'block averaged hr2 correlation beam 3'}



        # save if there is valid data on the day
        ds.attrs['version'] = 'v1'
        ds.attrs['comment'] = 'Quality checked data:' \
                              ' correlation and amplitude checks done and spikes were removed. ' \
                              'Velocities rotated to ENU coordinates based on heading and configuration in the field.' \
                              'data that was marked unfit has been removed and replaced by interpolation.'
        try:
            ds.attrs['instrument'] = ds.name.values[0]
        except:
            ds.attrs['instrument'] = ds.name.values

        # we no longer need these:
        ds = ds.drop(['heading', 'pitch', 'roll',
                       'pc', 'p', 'v1', 'v2', 'v3', 'name'], errors='ignore')

        # generate file name
        ncFilePath = os.path.join(config['experimentFolder'], instrument, r'qc\{}.nc'.format(tstart_string))

        if len(ds.dropna(dim='t').t) > 0:
            continue

        if os.path.exists(ncFilePath):
            with xr.open_dataset(ncFilePath) as ds0:
                ds = xr.merge([ds0, ds])

        # add script version information
        ds.attrs['git repo'] = r'https://github.com/MarliesA/EURECCA/tree/main/sedmex'
        ds.attrs['git hash'] = get_githash()

        # write to file
        # specify compression for all the variables to reduce file size
        comp = dict(zlib=True, complevel=5)
        ds.encoding = {var: comp for var in ds.data_vars}
        for coord in list(ds.coords.keys()):
            ds.encoding[coord] = {'zlib': False, '_FillValue': None}

    return

if __name__ == "__main__":

    config = yaml.safe_load(Path('sedmex-processing.yml').read_text())

    if 'L2C7ADCP' in config['instruments']['adcp']:
        instrument = 'L2C7ADCP'
        heading = 78  # because we don't trust the internal compass, we measured this in-situ using gps
        for part in ['1', '2', '3', '4']:
            qc_this_rawdatafile(instrument, heading, part, config)

    if 'L4C1ADCP' in config['instruments']['adcp']:
        instrument = 'L4C1ADCP'
        heading = 199  # because we don't trust the internal compass, we measured this in-situ using gps
        for part in ['a1', 'b1', 'a3', 'b3','c3']:
            qc_this_rawdatafile(instrument, heading, part, config)
