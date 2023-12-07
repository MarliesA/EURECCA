import glob
import os
import yaml
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import puv
from sedmex_info_loaders import get_githash
from encoding_sedmex import encoding_sedmex
from datetime import datetime


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

def resample_quality_check_replace_dataset(ds, ds_presref, config):
    import xrMethodAccessors

    # we reshape the dataset to blocks of blockLength seconds.
    if 'SONTEK' in ds.instrument:
        # the sontek measured in blocks of half hour. if we want analysis in another duration, we distribute the
        # burst into blocks of data such that the missing 60 seconds every half hour get evenly distributed
        # over the three 10min blocks
        ndivide = int(1800 / config['burstDuration']['sontek'])
        ds = ds.burst.reduce_burst_length(ndivide)
    else:
        # only the last of the 3 blocks contains 10 secs of missing data
        ds = ds.burst.reshape_burst_length(config['burstDuration']['vector'])

    # correct for the air pressure fluctuations and drift in the instrument
    # by using the pressure of the standalone solo's to match the high water pressures
    # Mind you: we loose the water level gradients over the cross-shore
    # half hour average pressure, on similar time-axis as ADV data
    presref = ds_presref.p.interp_like(ds)

    # correct the raw pressure signal by
    ds['pc'] = ds.p - ds.p.mean(dim='N') + presref
    ds['pc'].attrs = {'units': 'Pa', 'long_name': 'pressure',
                      'comments': 'referenced to pressure L2C10SOLO/L2C9OSSI'}

    # compute mean water level, water depth
    ds['zi'] = ds['zb'] + ds['h'] / 100
    ds['zi'].attrs = {'units': 'm+NAP', 'long_name': 'position ADV control volume'}

    ds['p'] = ds['pc'] / config['physicalConstants']['rho'] / config['physicalConstants']['g']
    ds['p'].attrs = {'units': 'm+NAP', 'long_name': 'hydrostatic surface elevation', 'comments':'corrected for air pressure fluctuations and referenced to NAP'}

    ds['d'] = ds.p.mean(dim='N') - ds.zb
    ds['d'].attrs = {'units': 'm ', 'long_name': 'water depth'}
    ds['h'] = ds.zi - ds.zb
    ds['h'].attrs = {'units': 'm ', 'long_name': 'height probe control volume above bed'}

    if not 'SONTEK' in ds.instrument:
        ds['zip'] = ds['zb'] + ds['hpres'] / 100
        ds['zip'].attrs = {'units': 'm+NAP', 'long_name': 'position pressure sensor'}

        ds['hpres'] = ds.zip - ds.zb
        ds['hpres'].attrs = {'units': 'm ', 'long_name': 'height pressure sensor above bed'}

    # if amplitude is too low, the probe was emerged
    ma1 = ds.a1 > config['qcADVSettings']['ampTreshold'][ds.instrument]
    ma2 = ds.a2 > config['qcADVSettings']['ampTreshold'][ds.instrument]
    ma3 = ds.a3 > config['qcADVSettings']['ampTreshold'][ds.instrument]

    # if correlation is outside confidence range
    if config['qcADVSettings']['corTreshold'] == 'elgar':
        # estimate correlation threshold, Elgar et al., 2005; pp. 1891
        sf = ds.sf.values
        if len(sf) > 0:
            sf = sf[0]
            criticalCorrelation = (0.3 + 0.4 * np.sqrt(sf / 25)) * 100
    else:
        criticalCorrelation = config['qcADVSettings']['corTreshold']

    mc1 = ds.cor1 > criticalCorrelation
    mc2 = ds.cor2 > criticalCorrelation
    mc3 = ds.cor3 > criticalCorrelation

    # if observation is outside of velocity range
    mu1 = np.abs(ds.u) < config['qcADVSettings']['uLim'][ds.instrument]
    mu2 = np.abs(ds.v) < config['qcADVSettings']['uLim'][ds.instrument]
    mu3 = np.abs(ds.w) < config['qcADVSettings']['uLim'][ds.instrument]

    # if du larger than outlierCrit*std(u) then we consider it outlier and hence remove:
    md1 = np.abs(ds.u.diff('N')) < config['qcADVSettings']['outlierCrit'] * ds.u.std(dim='N')
    md2 = np.abs(ds.v.diff('N')) < config['qcADVSettings']['outlierCrit'] * ds.v.std(dim='N')
    md3 = np.abs(ds.w.diff('N')) < config['qcADVSettings']['outlierCrit'] * ds.w.std(dim='N')

    ds['mc'] = np.logical_and(np.logical_and(mc1, mc2), mc3)
    ds['mu'] = np.logical_and(np.logical_and(mu1, mu2), mu3)
    ds['md'] = np.logical_and(np.logical_and(md1, md2), md3)
    ds['ma'] = np.logical_and(np.logical_and(ma1, ma2), ma3)
    ds['mc'].attrs = {'units': '-', 'long_name': 'mask correlation'}
    ds['mu'].attrs = {'units': '-', 'long_name': 'mask velocity limit'}
    ds['md'].attrs = {'units': '-', 'long_name': 'mask unexpected jumps in signal'}
    ds['ma'].attrs = {'units': '-', 'long_name': 'mask amplitude'}

    mp = np.abs(ds.p.diff('N')) < config['qcADVSettings']['outlierCrit'] * ds.p.std(dim='N')
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
    ds['p'] = ds.p.where(ds.maskp).where(ds.maskd)

    # drop bursts with too many nans (>maxFracNans) in the velocity data
    ds2 = ds.where( np.isnan(ds.u).sum(dim='N') < config['qcADVSettings']['maxFracNans'] * len(ds.N)).dropna(dim='t', how='all')
    ds2['sf'] = ds['sf']

    if len(ds2.t) == 0:
        print('no valid data remains on this day')
        return ds2

    # interpolate nans
    for var in ['u', 'v', 'w', 'p']:
        ds[var] = ds2[var].interpolate_na(
            dim='N',
            method='cubic',
            max_gap=config['qcADVSettings']['maxGap'])

        # and fill the gaps more than maxGap in length with the burst average
        ds2[var] = ds2[var].fillna(ds2[var].mean(dim='N'))

    ds = ds2


    # rename coordinates if horizontally placed
    if ((instrument == 'L2C2VEC') | (instrument == 'L2C4VEC')):
        ufunc = lambda u, v, w: hor2vert_vector_mapping(u, v, w)
        ds['u'], ds['v'], ds['w'] = xr.apply_ufunc(ufunc,
                                                   ds['u'], ds['v'], ds['w'],
                                                   input_core_dims=[['N'], ['N'], ['N']],
                                                   output_core_dims=[['N'], ['N'], ['N']],
                                                   vectorize=True)

    # UU instruments flexheads so downward pos instead of upward pos
    if (instrument in ['L1C1VEC', 'L2C3VEC', 'L3C1VEC', 'L5C1VEC', 'L6C1VEC']):
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
    if 'SONTEK' in instrument:
        ds.attrs['summary'] = 'SEDMEX field campaign: quality checked ADV data. Correlation checks done and spikes were removed.' \
                'Velocities rotated to ENU coordinates based on heading and configuration in the field.' \
                                       'data that was marked unfit has been removed or replaced by interpolation.'\
                                       'Pressure was referenced to air pressure and NAP.' \
                                       'Variance of the pressure signal only to be used for directional wave distribution,' \
                                       ' not for wave heights (scale is off and the Sontek pressure sensor is uncalibrated).'
    else:
        ds.attrs['summary'] = 'SEDMEX field campaign: quality checked ADV data. Correlation checks done and spikes were removed,' \
                              'Velocities rotated to ENU coordinates based on heading and configuration in the field.' \
                              'data that was marked unfit has been removed or replaced by interpolation.'\
                              'Pressure was referenced to air pressure and NAP.'


    ds['N'].attrs = {'units': 's', 'long_name': 'block local time'} 
    ds['t'].attrs = {'long_name': 'block start time'}  

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
                  'voltage', 'pc', 'maskp', 'maskv', 'maskd', 'ma', 'md', 'mu', 'mc'], errors='ignore')

    return ds


if __name__ == "__main__":

    config = yaml.safe_load(Path('c:\checkouts\eurecca_rebuttal\SEDMEX\SEDMEX_processing\sedmex-processing.yml').read_text())

    # preload the reference waterlevel to which the pressure recordings are calibrated
    fileZsRef = os.path.join(config['experimentFolder'], 'waterlevel.nc')
    ds_presref = xr.open_dataset(fileZsRef)

    # loop over all sonteks and adv's
    allVectors = []
    if not config['instruments']['adv']['vector'] == None:
        allVectors += config['instruments']['adv']['vector']
    if not config['instruments']['adv']['sontek'] == None:
        allVectors += config['instruments']['adv']['sontek']

    for instrument in allVectors:
        print(instrument)

        # find all raw data on file for this instrument
        fileNames = glob.glob(os.path.join(config['experimentFolder'], instrument, 'raw_netcdf', '*.nc'))

        # ensure the output folder exists
        folderOut = os.path.join(config['experimentFolder'], instrument, 'qc')
        if not os.path.isdir(folderOut):
            os.mkdir(folderOut)

        # loop over all the raw data files
        for file in fileNames:
            print(file)

            with xr.open_dataset(file) as ds:

                # do the actual work
                ds = resample_quality_check_replace_dataset(ds, ds_presref, config)

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
                            ncFilePath = os.path.join(folderOut, '{}_{}.nc'.format(ds.instrument, date))

                            # if there is already a file on the drive, merge the results
                            if os.path.exists(ncFilePath):
                                nc_onfile = xr.load_dataset(ncFilePath)
                                ds_merge = xr.merge([nc_onfile, ds_sel])
                            else:
                                ds_merge = ds_sel

                            # add script version information
                            ds_merge.attrs['construction datetime'] = datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")
                            ds_merge.attrs['git repo'] = r'https://github.com/MarliesA/EURECCA/tree/main/sedmex'
                            ds_merge.attrs['git hash'] = get_githash()

                            # write to file
                            encoding = encoding_sedmex(ds_merge)
                            ds_merge.to_netcdf(ncFilePath, encoding=encoding)





