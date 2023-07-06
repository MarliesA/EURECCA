import os
import glob
import yaml
from pathlib import Path
from datetime import datetime
import xarray as xr
from sedmex_info_loaders import get_githash


config = yaml.safe_load(Path('sedmex-processing.yml').read_text())
def combine_ncs(instrument, config):
    instrumentFolder = os.path.join(config['experimentFolder'], instrument)

    fold = os.path.join(instrumentFolder, 'tailored', r'L*_*.nc')
    filez = glob.glob(fold)

    dslist = []
    for file in filez:
        ds0 = xr.open_dataset(file)
        dslist.append(ds0.drop_dims(['f', 'theta', 'N'], errors='ignore'))

    return xr.merge(dslist)

if __name__ == "__main__":

    allInstruments = []
    instrumentType = []
    if not config['instruments']['adv']['vector'] == None:
        allInstruments += config['instruments']['adv']['vector']
        instrumentType += ['vector']
    if not config['instruments']['adv']['sontek'] == None:
        allInstruments += config['instruments']['adv']['sontek']
        instrumentType += ['sontek']
    if not config['instruments']['adcp'] == None:
        allInstruments += config['instruments']['adcp']
        instrumentType += ['adcp']
    if not config['instruments']['ossi'] == None:
        allInstruments += config['instruments']['ossi']
        instrumentType += ['pres']
    if not config['instruments']['solo'] == None:
        allInstruments += config['instruments']['solo']
        instrumentType += ['pres']

    for instrument, Type in zip(allInstruments, instrumentType):
        print(instrument)

        if Type == 'pres':
            ds = xr.open_dataset(os.path.join(config['experimentFolder'], instrument, 'tailored', 'instrument.nc'))
            ds = ds.drop_dims(['f', 'N'], errors='ignore')
            # update the summary
            ds.attrs['summary'] = 'tailored timeseries of wave statistics'
        else:
            # only keep coord t and merge all days
            ds = combine_ncs(instrument, config)
            # update the summary
            ds.attrs['summary'] = 'tailored timeseries of wave and current statistics'

        # on the repository we only publish the most relevant parameters, so drop the others
        vars2drop = ['Skc', 'Asc', 'sigc', 'Skl', 'Asl', 'sigl', 'Skp0', 'Asp0', 'sigp0', 'Skp', 'Asp', 'sigp',
                     'Sk0', 'As0', 'sig0', 'svdtheta', 'svddspr', 'fp', 'udm', 'vdm', 'ud_ssm']

        ds = ds.drop_vars(vars2drop, errors='ignore')

        # resample to timeseries at burstduration interval, such that missing data get awarded a nan.
        t_resolution = '{}s'.format(config['burstDuration'][Type])
        ds = ds.resample(t=t_resolution).nearest(tolerance=t_resolution)


        # update the construction date
        ds.attrs['construction datetime'] = datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")



        # update script version information
        ds.attrs['git hash'] = get_githash()

        ds.to_netcdf(os.path.join(config['experimentFolder'], instrument, 'tailored', 'tailored_' + instrument + '.nc'),
                     encoding=ds.encoding)
