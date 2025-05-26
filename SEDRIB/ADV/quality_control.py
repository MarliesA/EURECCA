
import glob
import os
import sys
sys.path.append(r'C:\checkouts\python\PhD\modules')
import yaml
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import puv
import xrMethodAccessors

def read_zs_data(filein, stationid):
    """
    :param filein: csv file with RWS waterinfo water level observations
    :param stationid: the stationid of the one station to retrieve from potentially a larger csv file
    :return: xarray dataset
    """
    dat = pd.read_csv(filein,
                      sep=';',
                      usecols=['MEETPUNT_IDENTIFICATIE', 
                                        'WAARNEMINGDATUM',
                                       'WAARNEMINGTIJD (MET/CET)',
                                       'NUMERIEKEWAARDE'])

    dat = dat[dat['MEETPUNT_IDENTIFICATIE']==stationid]
    
    dat['t'] = pd.to_datetime(dat['WAARNEMINGDATUM'] + ' ' + dat['WAARNEMINGTIJD (MET/CET)'], format = '%d-%m-%Y %H:%M:%S')
    dat['t'] = dat['t']
    dat = dat.drop_duplicates('t').set_index('t')
    dat['zs'] = dat['NUMERIEKEWAARDE']/100

    ds = dat[['zs']].to_xarray().sortby('t')
    
    ds.zs.plot()
    ds['zs'].attrs = {'units':'m+NAP','long_name':'water level'}
    ds.attrs = {'source':'waterinfo',
                   'station':'Oudeschild',
                   'time zone':'UTC+1, Dutch winter time'}
    return ds

config = yaml.safe_load(Path(r'C:\checkouts\python\PhD\field_visit_SEDRIB\sedrib23-processing.yml').read_text())


# instrument name from config file
instrument = 'vec008'

# ensure the output folder exists
folderOut = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\vec008\qc15min'
if not os.path.isdir(folderOut):
    os.mkdir(folderOut)

# loop over all the raw data files
for file in [r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\vec008\raw_netcdf\vec008.nc']:

    ds = xr.open_dataset(file).rename({'u':'u0', 'v':'v0', 'w':'w0', 'p':'p0'})
    ds = ds.burst.reshape_burst_length(900)

    # set instrument deployment variables
    zb0 = xr.open_dataarray(r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\vec008\zb_adv.nc')
    ds['zb'] =zb0.interp_like(ds.t)
    ds['zb'].attrs = {'units': 'm+NAP', 'long_name': 'position bed'}
    ds['zip'] = -0.723 
    ds['zip'].attrs = {'units': 'm+NAP', 'long_name': 'position pressure sensor'}
    ds['zi'] = -0.67
    ds['zi'].attrs = {'units': 'm+NAP', 'long_name': 'position control volume'}
    ds['io'] = 234+11 # x pointing towards, pos from north
    ds['io'].attrs = {'units': 'deg', 'long_name': 'direction of positive x-direction, positive from north clockwise'}

    ds['h'] = ds.zi - ds.zb
    ds['h'].attrs = {'units': 'm ', 'long_name': 'height control volume above bed'}

    ds['hpres'] = ds['zip'] - ds['zb'] 
    ds['hpres'].attrs = {'units': 'm', 'long_name': 'height pressure sensor above bed'}

    rwsfile = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\rws\20231122_026.csv'
    ods = read_zs_data(filein=rwsfile, stationid='Oudeschild')
    dhd = read_zs_data(filein=rwsfile, stationid='Den Helder')
    zs_rws = 0.5*(ods.zs+dhd.zs)

    ds['zs'] = zs_rws.interp_like(ds.t)
    ds['zs'].attrs = {'units': 'm+NAP ', 'long_name': 'water level'}

    ds['d'] = ds.zs - ds.zb  
    ds['d'].attrs = {'units': 'm ', 'long_name': 'water depth'}   

    ######################################################################
    # compute qc masks
    ######################################################################

    # if amplitude is too low, the probe was emerged
    ma1 = ds.a1 > 40
    ma2 = ds.a2 > 40
    ma3 = ds.a3 > 40
    ma = np.logical_and(np.logical_and(ma1, ma2), ma3)
    ds['maska'] = ma

    # if snr is too low, the probe was emerged
    msnr1 = ds.snr1 > config['qcADVSettings']['snrLim']
    msnr2 = ds.snr2 > config['qcADVSettings']['snrLim']
    msnr3 = ds.snr3 > config['qcADVSettings']['snrLim']
    msnr = np.logical_and(np.logical_and(msnr1, msnr2), msnr3)
    ds['masksnr'] = msnr

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
    mc = np.logical_and(np.logical_and(mc1, mc2), mc3)
    ds['maskc'] = mc

    # if observation is outside of velocity range
    mu1 = np.abs(ds.u0) < config['qcADVSettings']['uLim']
    mu2 = np.abs(ds.v0) < config['qcADVSettings']['vLim']
    mu3 = np.abs(ds.w0) < config['qcADVSettings']['wLim']
    ds['masku'] = np.logical_and(np.logical_and(mu1, mu2), mu3)
    ds['masku'].attrs = {'units': '-', 'long_name': 'mask velocity value out of range'}

    # if du larger than outlierCrit*std(u) then we consider it outlier and hence remove:
    md1 = np.abs(ds.u0.diff('N')) < config['qcADVSettings']['outlierCrit'] * ds.u0.std(dim='N')
    md2 = np.abs(ds.v0.diff('N')) < config['qcADVSettings']['outlierCrit'] * ds.v0.std(dim='N')
    md3 = np.abs(ds.w0.diff('N')) < config['qcADVSettings']['outlierCrit'] * ds.w0.std(dim='N')
    ds['maskd'] = np.logical_and(np.logical_and(md1, md2), md3)
    ds['maskd'].attrs = {'units': '-', 'long_name': 'mask unexpected jumps in velocity signal'}

    ds['maskp'] = np.abs(ds.p0.diff('N')) < config['qcADVSettings']['outlierCrit'] * ds.p0.std(dim='N')
    ds['maskp'].attrs = {'units': '-', 'long_name': 'mask unexpected jumps in pressure signal'}

    # check water depth
    p_zs = (ds.p0-ds.p0.mean(dim='N'))/config['physicalConstants']['rho']/config['physicalConstants']['g']+zs_rws.interp_like(ds.t)
    ds['mask_depth_p'] = p_zs > ds.zip 
    ds['mask_depth_p'].attrs = {'units': '-', 'long_name': 'mask pressure sensor emerged'}
    ds['mask_depth_v'] = p_zs > ds.zi+0.0077+0.1
    ds['mask_depth_v'].attrs = {'units': '-', 'long_name': 'mask velocity control volume emerged'}

    # combine coords for correlation, spikes and jumps
    ds.coords['maskv'] = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(ds.maska, ds.masksnr), ds.maskc), ds.masku), ds.maskd), ds.mask_depth_v)
    ds.coords['maskp'] = np.logical_and(ds.maskp, ds.mask_depth_p)
    ds['maskv'].attrs = {'units': '-', 'long_name': 'mask velocity', 'comments': 'all causes combined: amplitude, snr, correlation, velocity outliers, jumps, water depth'}
    ds['maskp'].attrs = {'units': '-', 'long_name': 'mask pressure', 'comments': 'all causes combined: jumps, water depth'}

    # set individual outliers to nan
    ds['u'] = ds.u0.where(ds.maskv)
    ds['v'] = ds.v0.where(ds.maskv)
    ds['w'] = ds.w0.where(ds.maskv)
    ds['p'] = ds.p0.where(ds.maskp)

    # create (t) mask of bursts with more than maxFracNans of Nans in the velocity data
    ds.coords['maskallv'] = (ds.maskv.sum(dim='N')/len(ds.N)) >= (1-config['qcADVSettings']['maxFracNans'])
    ds.coords['maskallp'] = (ds.maskp.sum(dim='N')/len(ds.N)) >= (1-config['qcADVSettings']['maxFracNans'])

    # drop bursts with too many nans (>maxFracNans) in the velocity data
    ds['u'] = ds.u.where(ds.maskallv)
    ds['v'] = ds.v.where(ds.maskallv)
    ds['w'] = ds.w.where(ds.maskallv)
    ds['p'] = ds.p.where(ds.maskallp)

    ######################################################################
    # fill masked observations with interpolation
    ######################################################################

    for var in ['u', 'v', 'w', 'p']:
        ds[var + '1'] = ds[var].interpolate_na(
            dim='N',
            method='cubic',
            max_gap=0.5)
                
        # and fill the gaps more than maxGap in length with the burst average
        ds[var] = ds[var].fillna(ds[var].mean(dim='N'))

    # # instrument flexhead so downward pos instead of upward pos
    #ds['v'] = -ds['v']
    #ds['w'] = -ds['w']

    # # rotate to ENU coordinates
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
    ds.attrs['summary'] = 'Quality checked data: correlation checks done and spikes were removed,' \
                           'data that was marked unfit has been removed or replaced by interpolation.'\
                           'Pressure was referenced to air pressure.'

    # save to netCDF
    # all variables that are only used for the QC are block averaged to reduce amount of info on QC files

    assert len(ds.t) > 0, 'no valid data remaining!'

    # ensure the output folder exists
    ncFilePath = os.path.join(folderOut, '{}.nc'.format(instrument))

    # write to file
    # specify compression for all the variables to reduce file size
    comp = dict(zlib=True, complevel=5)
    ds.encoding = {var: comp for var in ds.data_vars}
    for coord in list(ds.coords.keys()):
        ds.encoding[coord] = {'zlib': False, '_FillValue': None}

    ds.to_netcdf(ncFilePath, encoding=ds.encoding)


