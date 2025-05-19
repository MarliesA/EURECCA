import glob
import os
import sys
sys.path.append(r'c:\checkouts\python\PHD\modules')
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

config = yaml.safe_load(Path(r'c:\checkouts\python\PHD\field_visit_SEDRIB\sedrib23-processing.yml').read_text())

# instrument name from config file
instrument = config['instruments']['adv']['vector'][0]

# find all raw data on file for this instrument
fileNames = glob.glob(os.path.join(config['experimentFolder'], instrument, 'raw_netcdf', '*.nc'))

# ensure the output folder exists
folderOut = os.path.join(config['experimentFolder'], instrument, 'qc')
if not os.path.isdir(folderOut):
    os.mkdir(folderOut)

# loop over all the raw data files
for file in [r'c:\Temp\vec1_raw_reshaped.nc']:#fileNames:

    ds = xr.open_dataset(file)

    # set instrument deployment variables
    ds.attrs['xRD'] = 116086.4
    ds.attrs['yRD'] = 558941.9
    ds['zb'] = -0.97  # THIS NEEDS TO BE UPDATED FROM THE BOTTOM PING STILL
    ds['zb'].attrs = {'units': 'm+NAP', 'long_name': 'position bed'}
    ds['zip'] = -0.67  # THIS IS ONLY BALLPARK RIGHT
    ds['zip'].attrs = {'units': 'm+NAP', 'long_name': 'position pressure sensor'}
    ds['zi'] = -0.67
    ds['zi'].attrs = {'units': 'm+NAP', 'long_name': 'position control volume'}
    ds['io'] = 234 # x pointing towards, pos from north
    ds['io'].attrs = {'units': 'deg', 'long_name': 'direction of positive x-direction, positive from north clockwise'}
    ds_presref = []

    ds['h'] = ds.zi - ds.zb
    ds['h'].attrs = {'units': 'm ', 'long_name': 'height control volume above bed'}

    ds['hpres'] = ds['zip'] - ds['zb'] 
    ds['hpres'].attrs = {'units': 'm', 'long_name': 'height pressure sensor above bed'}

    rwsfile = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231001_ripples_frame\rws\20231122_026.csv'
    ods = read_zs_data(filein=rwsfile, stationid='Oudeschild')
    dhd = read_zs_data(filein=rwsfile, stationid='Den Helder')
    zs_rws = 0.5*(ods.zs+dhd.zs)

    ds['p'] = (ds.p-ds.p.mean(dim='N'))/config['physicalConstants']['rho']/config['physicalConstants']['g']+zs_rws.interp_like(ds.t)
    ds['p'].attrs = {'units': 'm+NAP', 'long_name': 'pressure',
                      'comments': 'pressure referenced to RWS gauge stations den helder and oudeschild'}

    ds['d'] = ds.p.mean(dim='N') - ds.zb
    ds['d'].attrs = {'units': 'm ', 'long_name': 'water depth'}

    ds['zs'] = ds.p.mean(dim='N')
    ds['zs'].attrs = {'units': 'm+NAP ', 'long_name': 'water level'}

    
    ######################################################################
    # compute qc masks
    ######################################################################

    # if amplitude is too low, the probe was emerged
    ma1 = ds.a1 > config['qcADVSettings']['ampTreshold']
    ma2 = ds.a2 > config['qcADVSettings']['ampTreshold']
    ma3 = ds.a3 > config['qcADVSettings']['ampTreshold']
    ds['ma'] = np.logical_and(np.logical_and(ma1, ma2), ma3)
    ds['ma'].attrs = {'units': '-', 'long_name': 'mask amplitude'}

    # if correlation is outside confidence range
    if config['qcADVSettings']['corTreshold'] == 'elgar':
        # estimate correlation threshold, Elgar et al., 2005; pp. 1891
        sf = ds.sf.values
        if len(sf) > 0:
            sf = sf[0]
            criticalCorrelation = (0.3 + 0.4 * np.sqrt(sf / 25)) * 100
    else:
        criticalCorrelation = config['qcADVSettings']['corTreshold']

    # check the ping to ping correlation 
    mc1 = ds.cor1 > criticalCorrelation
    mc2 = ds.cor2 > criticalCorrelation
    mc3 = ds.cor3 > criticalCorrelation
    ds['mc'] = np.logical_and(np.logical_and(mc1, mc2), mc3)
    ds['mc'].attrs = {'units': '-', 'long_name': 'mask correlation'}

    # if observation is outside of velocity range
    mu1 = np.abs(ds.u) < config['qcADVSettings']['uLim']
    mu2 = np.abs(ds.v) < config['qcADVSettings']['vLim']
    mu3 = np.abs(ds.w) < config['qcADVSettings']['wLim']
    ds['mu'] = np.logical_and(np.logical_and(mu1, mu2), mu3)
    ds['mu'].attrs = {'units': '-', 'long_name': 'mask velocity limit'}

    # if du larger than outlierCrit*std(u) then we consider it outlier and hence remove:
    md1 = np.abs(ds.u.diff('N')) < config['qcADVSettings']['outlierCrit'] * ds.u.std(dim='N')
    md2 = np.abs(ds.v.diff('N')) < config['qcADVSettings']['outlierCrit'] * ds.v.std(dim='N')
    md3 = np.abs(ds.w.diff('N')) < config['qcADVSettings']['outlierCrit'] * ds.w.std(dim='N')
    ds['md'] = np.logical_and(np.logical_and(md1, md2), md3)
    ds['md'].attrs = {'units': '-', 'long_name': 'mask unexpected jumps in velocity signal'}

    ds['mp'] = np.abs(ds.p.diff('N')) < config['qcADVSettings']['outlierCrit'] * ds.p.std(dim='N')
    ds['mp'].attrs = {'units': '-', 'long_name': 'mask unexpected jumps in pressure signal'}

    # create (t, N) mask for individual outliers
    ds.coords['maskp'] = (('t', 'N'), ds.mp.values)
    # ds.coords['maskd'] = (('t', 'N'), (ds.zs-0.2) < ds.zi)  
    ds.coords['maskd'] = (('t', 'N'), (ds.ma.values))  
    ds.coords['maskv'] = (('t', 'N'), (np.logical_and(np.logical_and(ds.mc, ds.mu), ds.md)).values)

    # create (t) mask of bursts with more than maxFracNans of Nans in the velocity data
    ds.coords['maskallv'] = ((ds.maskv*ds.maskd).sum(dim='N')/len(ds.N)) >= (1-config['qcADVSettings']['maxFracNans'])
    ds.coords['maskallp'] = ((ds.maskp*ds.maskd).sum(dim='N')/len(ds.N)) >= (1-config['qcADVSettings']['maxFracNans'])

    # set individual outliers to nan
    ds['u'] = ds.u.where(ds.maskv*ds.maskd)
    ds['v'] = ds.v.where(ds.maskv*ds.maskd)
    ds['w'] = ds.w.where(ds.maskv*ds.maskd)
    ds['p'] = ds.p.where(ds.maskp*ds.maskd)

    # drop bursts with too many nans (>maxFracNans) in the velocity data
    ds['u'] = ds.u.where(ds.maskallv)
    ds['v'] = ds.v.where(ds.maskallv)
    ds['w'] = ds.w.where(ds.maskallv)
    ds['p'] = ds.p.where(ds.maskallp)


    ######################################################################
    # fill masked observations with interpolation
    ######################################################################
    # interpolate nans
    for var in ['u', 'v', 'w', 'p']:
        ds[var] = ds[var].interpolate_na(
            dim='N',
            method='cubic',
            max_gap=config['qcADVSettings']['maxGap'])

    # # instrument flexhead so downward pos instead of upward pos
    ds['v'] = -ds['v']
    ds['w'] = -ds['w']

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
    ds['snr1'] = ds.snr1.mean(dim='N')
    ds['snr1'].attrs = {'units': '-', 'long_name': 'block averaged SNR beam 1'}
    ds['snr2'] = ds.snr2.mean(dim='N')
    ds['snr2'].attrs = {'units': '-', 'long_name': 'block averaged SNR beam 2'}
    ds['snr3'] = ds.snr3.mean(dim='N')
    ds['snr3'].attrs = {'units': '-', 'long_name': 'block averaged SNR beam 3'}

    # we no longer need these:
    ds = ds.drop_vars(['heading', 'pitch', 'roll',
                  'voltage'], errors='ignore')

    assert len(ds.t) > 0, 'no valid data remaining!'

    # ensure the output folder exists
    folderOut = os.path.join(config['experimentFolder'], instrument, 'qc')
    if not os.path.isdir(folderOut):
        os.mkdir(folderOut)

    ncFilePath = os.path.join(folderOut, '{}.nc'.format(instrument))

    # write to file
    # specify compression for all the variables to reduce file size
    comp = dict(zlib=True, complevel=5)
    ds.encoding = {var: comp for var in ds.data_vars}
    for coord in list(ds.coords.keys()):
        ds.encoding[coord] = {'zlib': False, '_FillValue': None}

    ds.to_netcdf(ncFilePath, encoding=ds.encoding)


