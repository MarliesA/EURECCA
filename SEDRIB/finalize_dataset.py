# %%
import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# %%
fold = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\4TU\data'
fold_out = r'\\tudelft.net\staff-umbrella\EURECCA\DataCiaran\data'

# %%
def get_githash():
    '''
    Get the revision number of the script to be printed in the metadata of the netcdfs
    :return: githash
    '''
    import subprocess
    try:
        githash = subprocess.check_output(["git", "describe", "--always"], cwd=os.path.dirname(os.path.abspath(__file__))).strip().decode()
    except:
        print('Mind you: script not run from repository, or GIT not installed')
        githash = '-'
    return githash

# %%
def encoding_sedrib(ds):
    """
    returns a dictionary with netcdf encoding attributes for all variables used in SEDRIB processing with appropriate precision and specification of possible compression.
    """
    encoding = {}
    for var in ['t', 'time']:
        encoding[var] = dict(zlib=False, _FillValue=None, units='seconds since 2023-11-01 00:00:00', calendar='proleptic_gregorian')
    for var in ['sf', 'N', 'z', 'f','theta', 'io', 'zip', 'zi']:
        encoding[var] = dict(zlib=False, _FillValue=None, )
    for var in ['a1', 'a2', 'a3', 'a4', 'cor1', 'cor2', 'cor3', 'burst', 'mc', 'mu', 'md', 'ma', 'beachOri']:
        encoding[var] = dict(zlib=True, complevel=5, dtype='int16', scale_factor=1, _FillValue=-9999)
    for var in ['snr1', 'snr2', 'snr3', 'snr4', 'heading', 'pitch', 'roll']:
        encoding[var] = dict(zlib=True, complevel=5, dtype='int16', scale_factor=0.1, _FillValue=-9999)
    for var in ['p', 'anl1', 'anl2']:
        encoding[var] = dict(zlib=True, complevel=5, dtype='int32', scale_factor=0.1, _FillValue=-9999)
    for var in ['io', 'zi', 'zip', 'elevp', 'elev', 'd']:
        encoding[var] = dict(zlib=True, complevel=5, dtype='int16', scale_factor=0.01, _FillValue=-9999)
    for var in ['hpres', 'h', 'eta']:
        encoding[var] = dict(zlib=True, complevel=5, dtype='int16', scale_factor=0.001, _FillValue=-9999)
    for var in ['udm', 'vdm', 'Sk0', 'As0', 'sig0', 'Skp0', 'Asp0', 'sigp0', 'Skp', 'Asp', 'sigp', 'ud_ssm', 'S', 'fp', 'vyp', 
                'v1', 'v2', 'v3', 'v4', 'u', 'v', 'w', 'umag', 'Hm0', 'u_ssm', 'ucm', 'ulm', 'Sk', 'As', 'sig', 'zs', 'k', 'Ur', 
                'nAs', 'nSk', 'zb']:
        encoding[var] = dict(zlib=True, complevel=5, dtype='int16', scale_factor=0.0001, _FillValue=-9999)
    for var in ['uang','svdtheta', 'svddspr', 'dspr', 'puvdir']:
        encoding[var] = dict(zlib=True, complevel=5, dtype='int16', scale_factor=0.01, _FillValue=-9999)
    for var in ['Tp', 'Tm01', 'Tm02', 'Tmm10', 'Tps']:
        encoding[var] = dict(zlib=True, complevel=5, dtype='int16', scale_factor=0.001, _FillValue=-9999)
        
    # return only the subset on variables that are actually on the dataset
    encoding_sedrib = {var: encoding[var] for var in encoding if var in ds.data_vars}
    encoding_sedrib.update({var: encoding[var] for var in encoding if var in list(ds.coords.keys())})
    
    return encoding_sedrib

# %%
def fix_metadata_acvp(ds): 

    for var in ['snr1', 'snr2', 'snr3', 'snr4']:
        ds[var].attrs['long_name'] = 'signal-to-noise-ratio beam ' + var[-1]
        ds[var].attrs['units'] ='dB'
    for var in ['a1', 'a2', 'a3', 'a4']:
        ds[var].attrs['long_name'] = 'signal intensity beam ' + var[-1]
        ds[var].attrs['units'] ='Counts'
    for var in ['v1', 'v2', 'v3', 'v4']:
        ds[var].attrs['long_name'] = 'velocity in direction beam' + var[-1]
        ds[var].attrs['units'] ='m/s'
    ds['z'].attrs['long_name'] = 'gate height wrt position of central transducer'

    ds.attrs = {
    'conventions': 'CF-1.6',
    'dataset': 'The SEDRIB campaign aims to gain new insights into the driving mechanisms of bed load transport over a rippled bed on a low-energy field site. Field measurements from were conducted in October 2023 at the Prins Hendrik Zanddijk: a man-made beach on the leeside of the barrier island Texel, bordering the Marsdiep basin that is part of the Dutch Wadden Sea. This data set consists of ADV, ACVP and SRPS data.',
    'summary': 'SEDRIB field campaign ACVP data: raw data part 14 of the October 1st deployment. ',
    'instrument': 'Ubertone UBFlow-3C',
    'instrument serial number': '220001',
    'product_id': 'apf06proto1',
    'apf_handler': 'apf06_handler',
    'epsg': 28992,
    'x': 116088.3,
    'y': 558943.4,
    'time zone': 'UTC+1',
    'instrument configs': 'f0=1000000.0, n_echo=128, n_profile=1, n_vol=200, phase_coding=True, prf=600.6006006006006, r_dvol=0.002245091561425781, r_em=0.0029934554152343746, r_vol1=0.005238546976660156, static_echo_filter=False, sound_speed=1496.7277076171874',
    'coordinate type': 'beam-coordinates',
    'contact person': 'Marlies van der Lugt',
    'emailadres': 'm.a.vanderlugt@tudelft.nl',
    'construction datetime': '23-05-2025',
    'version comments': 'constructed with xarray',
    'git repo': 'https://github.com/MarliesA/EURECCA/tree/main/sedrib',
    'git hash': '{}'.format(get_githash())}

    return ds

# ACVP data process file by file
filez = glob.glob(os.path.join(fold, 'ACVP', 'raw_netcdf', '*.nc'))
folds_out = os.path.join(fold_out, 'ACVP', 'raw_netcdf')
if not os.path.exists(folds_out):
    os.makedirs(folds_out)

for file in filez:
    print(file)
    ds = xr.open_dataset(file)
    encoding = encoding_sedrib(ds)
    ds.to_netcdf((os.path.join(folds_out, file.split('\\')[-1])), encoding=encoding)

# %%
# compress and finalize the raw netcdf ADV data
folds_out = os.path.join(fold_out, 'ADV', 'raw_netcdf')
if not os.path.exists(folds_out):
    os.makedirs(folds_out)
file = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\4TU\data\ADV\raw_netcdf\vec008.nc'

ds = xr.open_dataset(file)

# fix the attributes
attrs = ds.attrs
addition = {'conventions': 'CF-1.6',
    'dataset': 'The SEDRIB campaign aims to gain new insights into the driving mechanisms of bed load transport over a rippled bed on a low-energy field site. Field measurements from were conducted in October 2023 at the Prins Hendrik Zanddijk: a man-made beach on the leeside of the barrier island Texel, bordering the Marsdiep basin that is part of the Dutch Wadden Sea. This data set consists of ADV, ACVP and SRPS data.',
}
ds.attrs = {**addition,          
            **attrs,
            'instrument': 'Nortek Vector',
            'summary': 'ADV raw data: Vec008 was installed downward looking with control volume approximately 30 cm above the bed',
            'git repo': 'https://github.com/MarliesA/EURECCA/tree/main/sedrib',
            'git hash': '{}'.format(get_githash())
 }

encoding = encoding_sedrib(ds)
ds.to_netcdf((os.path.join(folds_out, file.split('\\')[-1])), encoding=encoding)

# %%
# compress and finalize the quality controled ADV data
folds_out = os.path.join(fold_out, 'ADV', 'qc')
if not os.path.exists(folds_out):
    os.makedirs(folds_out)
file = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\4TU\data\ADV\qc\vec008.nc'

ds = xr.open_dataset(file)

# fix the attributes
attrs = ds.attrs
addition = {'conventions': 'CF-1.6',
    'dataset': 'The SEDRIB campaign aims to gain new insights into the driving mechanisms of bed load transport over a rippled bed on a low-energy field site. Field measurements from were conducted in October 2023 at the Prins Hendrik Zanddijk: a man-made beach on the leeside of the barrier island Texel, bordering the Marsdiep basin that is part of the Dutch Wadden Sea. This data set consists of ADV, ACVP and SRPS data.',
}
ds.attrs = {**addition,          
            **attrs,
            'instrument': 'Nortek Vector',
            'git repo': 'https://github.com/MarliesA/EURECCA/tree/main/sedrib',
            'git hash': '{}'.format(get_githash())
 }

encoding = encoding_sedrib(ds)
ds.to_netcdf((os.path.join(folds_out, file.split('\\')[-1])), encoding=encoding)

# %%
# compress and finalize the tailored ADV data

folds_out = os.path.join(fold_out, 'ADV', 'tailored')
if not os.path.exists(folds_out):
    os.makedirs(folds_out)
file = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\4TU\data\ADV\tailored\vec008.nc'

ds = xr.open_dataset(file)
ds = ds.drop_dims(['f','theta'])
ds = ds.drop_vars(['pm', 'um', 'vm', 'wm', 'anl1m', 'anl2m', 'cor1m', 'cor2m', 'cor3m', 'snr1m', 'snr2m', 'snr3m', 'headingm', 
                   'pitchm', 'rollm','TKED',  'Skp0', 'Asp0', 'sigp0', 'Skp', 'Asp', 'sigp'])

# fix 180 rotation over undefined positive axis:
ds['svdtheta2'] = np.where(ds.svdtheta>0, ds.svdtheta, ds.svdtheta+180)

attrs = ds.attrs
addition = {'conventions': 'CF-1.6',
    'dataset': 'The SEDRIB campaign aims to gain new insights into the driving mechanisms of bed load transport over a rippled bed on a low-energy field site. Field measurements from were conducted in October 2023 at the Prins Hendrik Zanddijk: a man-made beach on the leeside of the barrier island Texel, bordering the Marsdiep basin that is part of the Dutch Wadden Sea. This data set consists of ADV, ACVP and SRPS data.',
    'summary': 'ADV tailored timeseries: quality controlled data was converted to timeseries wave and flow characteristics at 15 minute resolution.',
}
ds.attrs = {**addition,  
            
            **attrs,
            'summary': 'ADV tailored timeseries: quality controlled data was converted to timeseries wave and flow characteristics at 15 minute resolution.',
            'instrument': 'Nortek Vector',
            'git repo': 'https://github.com/MarliesA/EURECCA/tree/main/sedrib',
            'git hash': '{}'.format(get_githash())
 }

encoding = encoding_sedrib(ds)
ds.to_netcdf((os.path.join(folds_out, file.split('\\')[-1])), encoding=encoding)

# %%
# compress and finalize the extracted bed level position below the ADV
file = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\vec008\zb_adv.nc'
ds = xr.open_dataarray(file)
# add meta data
ds.attrs = {
    'conventions': 'CF-1.6',
    'dataset': 'The SEDRIB campaign aims to gain new insights into the driving mechanisms of bed load transport over a rippled bed on a low-energy field site. Field measurements from were conducted in October 2023 at the Prins Hendrik Zanddijk: a man-made beach on the leeside of the barrier island Texel, bordering the Marsdiep basin that is part of the Dutch Wadden Sea. This data set consists of ADV, ACVP and SRPS data.',
    'summary': '2023 PHZD field campaign. Vec008 was installed downward looking with control volume approximately 30 cm above the bed',
    'instrument': 'Nortek Vector',
    'instrument serial number': '{}'.format(16725),
    'epsg': 28992,
    'x': 116086.4, 
    'y': 558942.0,
    'units': 'm+NAP', 
    'long_name': 'bed position', 'comments': 'extracted from the the bed level ping at every 0.5 hr interval',
    'time zone': 'UTC+1',
    'coordinate type': 'XYZ',
    'contact person': 'Marlies van der Lugt',
    'emailadres': 'm.a.vanderlugt@tudelft.nl',
    'construction datetime': '23-05-2025',
    'version comments': 'constructed with xarray',
    'git repo': 'https://github.com/MarliesA/EURECCA/tree/main/sedrib',
    'git hash': '{}'.format(get_githash())}

folds_out = os.path.join(fold_out, 'ADV', 'raw_netcdf')
ds.to_netcdf((os.path.join(folds_out, file.split('\\')[-1])))

# %%
# auxiliary data to reference the water level of the ADV
file = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\rws\oudeschild.nc'
ds = xr.open_dataset(file)
ds.attrs['source_url'] = r'https://waterinfo.rws.nl'
ds.attrs['source_access_date'] = '2023-11-23'
ds.attrs['git repo'] = r'https://github.com/MarliesA/EURECCA/tree/main/sedrib'
ds.attrs['git hash'] = get_githash()
ds.attrs['summary'] = 'Water level gauge Oudeschild, referenced to NAP'
ds.attrs['source'] = 'Rijkswaterstaat'
folds_out = os.path.join(fold_out, 'ADV', 'raw_netcdf')
ds.to_netcdf((os.path.join(folds_out, file.split('\\')[-1])))

# auxiliary data to reference the water level
file = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\rws\denhelder.nc'
ds = xr.open_dataset(file)
ds.attrs['source_url'] = r'https://waterinfo.rws.nl'
ds.attrs['source_access_date'] = '2023-11-23'
ds.attrs['git repo'] = r'https://github.com/MarliesA/EURECCA/tree/main/sedrib'
ds.attrs['git hash'] = get_githash()
ds.attrs['summary'] = 'Water level gauge Den Helder, referenced to NAP'
ds.attrs['source'] = 'Rijkswaterstaat'
folds_out = os.path.join(fold_out, 'ADV', 'raw_netcdf')
ds.to_netcdf((os.path.join(folds_out, file.split('\\')[-1])))

# %%
## add metadata to 1D estimated geometry
ds = xr.open_dataset(r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\4TU\data\SRPS\tailored\geometrystats1D.nc')

ds['theta'].attrs['long_name'] = 'angle with respect to shore-normal'
ds['theta'].attrs['units'] ='degrees'

ds['eta'].attrs['long_name'] = 'swath-dependent ripple wave height'
ds['eta'].attrs['units'] ='m'

ds['labda'].attrs['long_name'] = 'swath-dependent ripple wave length'
ds['labda'].attrs['units'] ='m'

ds['thetmin'].attrs['long_name'] = 'angle of swath minimizing ripple wave length'
ds['thetmin'].attrs['units'] ='degrees with respect to shore-normal'

ds['labdamin'].attrs['long_name'] = 'ripple wave length'
ds['labdamin'].attrs['units'] ='m'

ds['etamin'].attrs['long_name'] = 'ripple wave height'
ds['etamin'].attrs['units'] ='m'

encoding = {}
encoding['time'] = dict(zlib=False, _FillValue=None, units='seconds since 2023-11-01 00:00:00', calendar='proleptic_gregorian')

# add meta data
ds.attrs = {
    'conventions': 'CF-1.6',
    'dataset': 'The SEDRIB campaign aims to gain new insights into the driving mechanisms of bed load transport over a rippled bed on a low-energy field site. Field measurements from were conducted in October 2023 at the Prins Hendrik Zanddijk: a man-made beach on the leeside of the barrier island Texel, bordering the Marsdiep basin that is part of the Dutch Wadden Sea. This data set consists of ADV, ACVP and SRPS data.',
    'summary': '2023 PHZD field campaign. Ripple geometry estimated from 1D methodology',
    'instrument': 'Marine Electronics Sand Ripple Profile Scanner',
    'epsg': 28992,
    'x': 116086, 
    'y': 558942,
    'time zone': 'UTC+1',
    'contact person': 'Marlies van der Lugt',
    'emailadres': 'm.a.vanderlugt@tudelft.nl',
    'construction datetime': '26-05-2025',
    'version comments': 'constructed with xarray and the script analyse_bedforms_1D.py',
    'git repo': 'https://github.com/MarliesA/EURECCA/tree/main/sedrib',
    'git hash': '{}'.format(get_githash())}

ds.to_netcdf(r'\\tudelft.net\staff-umbrella\EURECCA\DataCiaran\data\SRPS\tailored\geometrystats1D.nc', encoding=encoding)


# %%
## add metadata to 1D estimated geometry
ds = xr.open_dataset(r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\4TU\data\SRPS\tailored\geometrystats2D.nc')

ds['eta'].attrs['long_name'] = 'ripple wave height'
ds['eta'].attrs['units'] ='m'

ds['labda'].attrs['long_name'] = 'ripple wave length'
ds['labda'].attrs['units'] ='m'

ds['phi'].attrs['long_name'] = 'ripple orientation'
ds['phi'].attrs['units'] ='deg with respect to shore-normal'

ds = ds.drop_vars(['phimin', 'phimax', 'Lm01', 'phi_mean', 'nu_xl', 'nu_yl', 'nu_rl', 'nu_cl'])

encoding = {}
encoding['time'] = dict(zlib=False, _FillValue=None, units='seconds since 2023-11-01 00:00:00', calendar='proleptic_gregorian')

# add meta data
ds.attrs = {
    'conventions': 'CF-1.6',
    'dataset': 'The SEDRIB campaign aims to gain new insights into the driving mechanisms of bed load transport over a rippled bed on a low-energy field site. Field measurements from were conducted in October 2023 at the Prins Hendrik Zanddijk: a man-made beach on the leeside of the barrier island Texel, bordering the Marsdiep basin that is part of the Dutch Wadden Sea. This data set consists of ADV, ACVP and SRPS data.',
    'summary': '2023 PHZD field campaign. Ripple geometry estimated from 2D methodology',
    'instrument': 'Marine Electronics Sand Ripple Profile Scanner',
    'epsg': 28992,
    'x': 116086, 
    'y': 558942,
    'time zone': 'UTC+1',
    'contact person': 'Marlies van der Lugt',
    'emailadres': 'm.a.vanderlugt@tudelft.nl',
    'construction datetime': '26-05-2025',
    'version comments': 'constructed with xarray and the script analyse_bedforms_2D.py',
    'git repo': 'https://github.com/MarliesA/EURECCA/tree/main/sedrib',
    'git hash': '{}'.format(get_githash())}

ds.to_netcdf(r'\\tudelft.net\staff-umbrella\EURECCA\DataCiaran\data\SRPS\tailored\geometrystats_2D.nc', encoding=encoding)

# %%
## add metadata to 1D estimated migration rates
ds = xr.open_dataset(r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\4TU\data\SRPS\tailored\migrationrates1D.nc')

ds['p'].attrs['long_name'] = 'p-value of cross-correlation between consecutive swaths'
ds['p'].attrs['units'] ='-'

ds['phi'].attrs['long_name'] = 'ripple migration direction'
ds['phi'].attrs['units'] ='deg with respect to shore-normal'

ds['dmig'].attrs['long_name'] = 'displacement between consecutive swaths'
ds['dmig'].attrs['units'] ='m'

ds = ds.drop_vars(['tprev'])

encoding = {}
encoding['time'] = dict(zlib=False, _FillValue=None, units='seconds since 2023-11-01 00:00:00', calendar='proleptic_gregorian')

# add meta data
ds.attrs = {
    'conventions': 'CF-1.6',
    'dataset': 'The SEDRIB campaign aims to gain new insights into the driving mechanisms of bed load transport over a rippled bed on a low-energy field site. Field measurements from were conducted in October 2023 at the Prins Hendrik Zanddijk: a man-made beach on the leeside of the barrier island Texel, bordering the Marsdiep basin that is part of the Dutch Wadden Sea. This data set consists of ADV, ACVP and SRPS data.',
    'summary': '2023 PHZD field campaign. Ripple migration estimated from 1D methodology',
    'instrument': 'Marine Electronics Sand Ripple Profile Scanner',
    'epsg': 28992,
    'x': 116086, 
    'y': 558942,
    'time zone': 'UTC+1',
    'contact person': 'Marlies van der Lugt',
    'emailadres': 'm.a.vanderlugt@tudelft.nl',
    'construction datetime': '26-05-2025',
    'version comments': 'constructed with xarray and the script analyse_migration_1D.py',
    'git repo': 'https://github.com/MarliesA/EURECCA/tree/main/sedrib',
    'git hash': '{}'.format(get_githash())}

ds.to_netcdf(r'\\tudelft.net\staff-umbrella\EURECCA\DataCiaran\data\SRPS\tailored\migrationrates_1D.nc', encoding=encoding)

# %%
## add metadata to 2D estimated migration rates
ds = xr.open_dataset(r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\4TU\data\SRPS\tailored\migrationrates2D.nc')

ds['p'].attrs['long_name'] = 'p-value of cross-correlation between consecutive swaths'
ds['p'].attrs['units'] ='-'

ds['phi'].attrs['long_name'] = 'ripple migration direction'
ds['phi'].attrs['units'] ='deg with respect to shore-normal'

ds['dmig'].attrs['long_name'] = 'displacement between consecutive footprints'
ds['dmig'].attrs['units'] ='m'

ds = ds.drop_vars(['tprev'])

ds['percvalid1'].attrs['long_name'] = 'percentage of non-NaN bed level estimates within footprint 1'
ds['percvalid1'].attrs['units'] ='%'

ds['percvalid2'].attrs['long_name'] = 'percentage of non-NaN bed level estimates within footprint 2'
ds['percvalid2'].attrs['units'] ='%'


encoding = {}
encoding['time'] = dict(zlib=False, _FillValue=None, units='seconds since 2023-11-01 00:00:00', calendar='proleptic_gregorian')

# add meta data
ds.attrs = {
    'conventions': 'CF-1.6',
    'dataset': 'The SEDRIB campaign aims to gain new insights into the driving mechanisms of bed load transport over a rippled bed on a low-energy field site. Field measurements from were conducted in October 2023 at the Prins Hendrik Zanddijk: a man-made beach on the leeside of the barrier island Texel, bordering the Marsdiep basin that is part of the Dutch Wadden Sea. This data set consists of ADV, ACVP and SRPS data.',
    'summary': '2023 PHZD field campaign. Ripple migration estimated from 2D methodology',
    'instrument': 'Marine Electronics Sand Ripple Profile Scanner',
    'epsg': 28992,
    'x': 116086, 
    'y': 558942,
    'time zone': 'UTC+1',
    'contact person': 'Marlies van der Lugt',
    'emailadres': 'm.a.vanderlugt@tudelft.nl',
    'construction datetime': '26-05-2025',
    'version comments': 'constructed with xarray and the script analyse_migration_2D.py',
    'git repo': 'https://github.com/MarliesA/EURECCA/tree/main/sedrib',
    'git hash': '{}'.format(get_githash())}

ds.to_netcdf(r'\\tudelft.net\staff-umbrella\EURECCA\DataCiaran\data\SRPS\tailored\migrationrates2D.nc', encoding=encoding)

# %%
ds = xr.open_dataset(r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\4TU\data\SRPS\tailored\directional_spreading.nc')


# %%
ds.spreading

# %%
## add metadata to 2D estimated migration rates
ds = xr.open_dataset(r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\4TU\data\SRPS\tailored\directional_spreading.nc')

ds['spreading'].attrs['long_name'] = 'directional spreading in the 2D wave number power spectrum of the bed footprint'
ds['spreading'].attrs['units'] ='degrees'

encoding = {}
encoding['time'] = dict(zlib=False, _FillValue=None, units='seconds since 2023-11-01 00:00:00', calendar='proleptic_gregorian')

# add meta data
ds.attrs = {
    'conventions': 'CF-1.6',
    'dataset': 'The SEDRIB campaign aims to gain new insights into the driving mechanisms of bed load transport over a rippled bed on a low-energy field site. Field measurements from were conducted in October 2023 at the Prins Hendrik Zanddijk: a man-made beach on the leeside of the barrier island Texel, bordering the Marsdiep basin that is part of the Dutch Wadden Sea. This data set consists of ADV, ACVP and SRPS data.',
    'summary': '2023 PHZD field campaign. Ripple directional spread estimated from 2D methodology',
    'instrument': 'Marine Electronics Sand Ripple Profile Scanner',
    'epsg': 28992,
    'x': 116086, 
    'y': 558942,
    'time zone': 'UTC+1',
    'contact person': 'Marlies van der Lugt',
    'emailadres': 'm.a.vanderlugt@tudelft.nl',
    'construction datetime': '26-05-2025',
    'version comments': 'constructed with xarray and the script analyse_2D3Dimensionality.py',
    'git repo': 'https://github.com/MarliesA/EURECCA/tree/main/sedrib',
    'git hash': '{}'.format(get_githash())}

ds.to_netcdf(r'\\tudelft.net\staff-umbrella\EURECCA\DataCiaran\data\SRPS\tailored\directionalspreading.nc', encoding=encoding)
