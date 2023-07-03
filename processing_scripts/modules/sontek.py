import xarray as xr
import os
import numpy as np
import pandas as pd
import glob
import puv

def read_raw_data_file(hd1file, ts1file):
    '''
    read_raw_data(hd1file, ts1file)

    reads SONTEK data from .hd1 and .ts1 file into a pandas dataframe

    :param hd1file: path to .hd1 header file
    :param ts1file: path to .ts1 data file
    :return: pandas dataframe with data on this raw file
    '''

    # start with reading hd1
    burst = []
    time = []
    bed1 = []
    bed2 = []
    with open(hd1file) as fp:
        for line in fp:
            x = [ix for ix in line.split() if len(ix) > 0]
            if len(x) > 0:
                burst.append(float(x[0]))
                time.append(
                    '{:02d}'.format(int(x[1]))+'-' +
                    '{:02d}'.format(int(x[2]))+'-' +
                    '{:02d}'.format(int(x[3]))+' ' +
                    '{:02d}'.format(int(x[4]))+':' +
                    '{:02d}'.format(int(x[5]))+':' +
                    '{:02d}'.format(int(x[6]))
                )
                bed1.append(float(x[11])/100 if float(x[11]) > 0 else np.nan)  # to meter
                bed2.append(float(x[12])/100 if float(x[12]) > 0 else np.nan)  # to meter
        Fs = float(x[7])
        time = pd.to_datetime(time)

    # cast in dataframe
    dftime = pd.DataFrame({'bt': time, 'bed1': bed1, 'bed2': bed2, 'burst': burst})

    # continue with reading ts1
    # on DAT file a new line for every measurement
    burst = []
    sample = []
    u = []
    v = []
    w = []
    cor1 = []
    cor2 = []
    cor3 = []
    a1 = []
    a2 = []
    a3 = []
    heading = []
    pitch = []
    roll = []
    T = []
    p = []
    with open(ts1file) as fp:
        for line in fp:
            x = [ix for ix in line.split() if len(ix) > 0]
            if len(x) > 0:
                burst.append(float(x[0]))
                sample.append(float(x[1]))
                u.append(float(x[2]))
                v.append(float(x[3]))
                w.append(float(x[4]))
                a1.append(float(x[5]))
                a2.append(float(x[6]))
                a3.append(float(x[7]))
                cor1.append(float(x[8]))
                cor2.append(float(x[9]))
                cor3.append(float(x[10]))
                heading.append(float(x[11]))
                pitch.append(float(x[12]))
                roll.append(float(x[13]))
                T.append(float(x[14][:4]))
                try:
                    p.append(float(x[15]))
                except:
                    p.append(float(x[14][4:]))

    dat = {'u': np.array(u)/100, 'v': np.array(v)/100, 'w': np.array(w)/100,
           'a1': a1, 'a2': a2, 'a3': a3,
           'cor1': cor1, 'cor2': cor2, 'cor3': cor3,
           'T': T, 'p': p,
           'heading': heading, 'pitch': pitch, 'roll': roll, 'burst': burst, 'localtime': (np.array(sample)-1)/Fs}

    dfdat = pd.DataFrame(dat)

    # join time and data into one dataframe
    df = pd.merge(dfdat, dftime, on='burst')
    df['t'] = pd.to_timedelta(df['localtime'].values, unit='S') + df['bt']
    df = df.set_index('t')

    return df, Fs

def cast_to_blocks_in_xarray(df, sf=10, blockWidth=1740):
    '''
    takes the raw data which are timeseries in pandas DataFrame and casts it
    in blocks (bursts) in an xarray with metadata for easy computations and
    saving to file (netcdf) later on.
    '''

    blockLength = int(sf*blockWidth)
    N = len(df.u.values)
    NB = int(np.floor(N / blockLength))
    t = df.index.values
    blockStartTimes = t[0::blockLength]

    if len(blockStartTimes) > NB:
        blockStartTimes = blockStartTimes[:NB]

    # cast all info in dataset

    ds = xr.Dataset(
        data_vars=dict(
            sf=sf
        ),
        coords=dict(
            t=blockStartTimes,
            N=np.arange(0, blockLength)/sf
        )
    )

    # cast all variables in the ds structure
    for var in ['u', 'v', 'w', 'a1', 'a2', 'a3',
                'cor1', 'cor2', 'cor3',
                'T', 'p',
                'heading', 'pitch', 'roll', 'burst', 'bed1', 'bed2']:
        tmp = df[var].values
        tmp = tmp[0:int(NB * blockLength)]
        N = len(tmp)
        ds[var] = (['t', 'N'], tmp.reshape(NB, int(N / NB)))
    ds['bed1'] = ds.bed1.mean(dim='N')
    ds['bed2'] = ds.bed2.mean(dim='N')

    ds['t'].attrs = {'long_name': 'burst start times'}
    ds['N'].attrs = {'units': 's', 'long_name': 'time'}
    ds['sf'].attrs = {'units': 'Hz', 'long_name': 'sampling frequency'}
    ds['u'].attrs = {'units': 'm/s', 'long_name': 'velocity 1 component'}
    ds['v'].attrs = {'units': 'm/s', 'long_name': 'velocity 2 component'}
    ds['w'].attrs = {'units': 'm/s', 'long_name': 'velocity 3 component'}
    ds['a1'].attrs = {'units': '-', 'long_name': 'amplitude beam 1'}
    ds['a2'].attrs = {'units': '-', 'long_name': 'amplitude beam 2'}
    ds['a3'].attrs = {'units': '-', 'long_name': 'amplitude beam 3'}
    ds['cor1'].attrs = {'units': '-', 'long_name': 'correlation beam 1'}
    ds['cor2'].attrs = {'units': '-', 'long_name': 'correlation beam 2'}
    ds['cor3'].attrs = {'units': '-', 'long_name': 'correlation beam 3'}
    ds['T'].attrs = {'units': 'deg C', 'long_name': 'temperature'}
    ds['p'].attrs = {'units': 'dB', 'long_name': 'pressure'}
    ds['heading'].attrs = {'units': 'deg', 'long_name': 'instrument heading'}
    ds['pitch'].attrs = {'units': 'deg', 'long_name': 'instrument pitch'}
    ds['roll'].attrs = {'units': 'deg', 'long_name': 'instrument roll'}
    ds['burst'].attrs = {'units': '-', 'long_name': 'burst number'}
    ds['bed1'].attrs = {'units': '-', 'long_name': 'bed level begin of burst'}
    ds['bed2'].attrs = {'units': '-', 'long_name': 'bed level end of burst'}

    return ds

def quality_control(infolder, outfolder, ampTreshold=100, criticalCorType='elgar', mincor=70, max_gap=8):
    '''
    quality checking raw SonTek data using a minimal signal amplitude and a minimal beam correlation. If there are
    sufficient (>5%) valid samples, the masked samples are replaced by cubic interpolation, but only if there are
    no trains of masked samples of more than max_gap gaps.
    :param infolder: folder with preprocessed raw data files
    :param outfolder: folder to save the qc data to
    :param ampTreshold: minimum amplitude for valid data sample
    :param criticalCorType: in case is left to 'elgar', the critical correlation type is a function of sampling frequency
    :param mincor: if criticalCorType is set to something else than elgar, this fixed minimum value for the beam correlation is used
    :param max_gap: maximum length of consecutive masked samples in a valid burst
    :return: netcdf files with output saved to outfolder
    '''

    # quality control
    rawfiles = glob.glob(os.path.join(infolder, '*.nc'))

    for file in rawfiles:
        filestr = file.split('\\')[-1]
        print('quality checking ' + filestr[:-2])

        ds = xr.open_dataset(file)

        # we reshape the dataset to blocks of 10 minutes.
        ds = ds.burst.reshape_burst_length(600)

        # max 5% of samples can have amplitude lower than treshold, otherwise probe emerged from water
        N = len(ds.N.values)
        ma = (np.logical_or(
            np.logical_or(
                ds['a1'] < ampTreshold,
                ds['a2'] < ampTreshold),
            ds['a3'] < ampTreshold)
        ).sum(dim='N') < 0.05*N

        # mask all those where probe was emerged
        ds = ds.where(ma).dropna(dim='t')
        if len(ds.t) == 0:
            continue

        # correlation not too low
        if criticalCorType == 'elgar':
            # estimate correlation threshold, Elgar et al., 2005; pp. 1891
            sf = ds.sf.values
            if len(sf) > 0:
                sf = sf[0]
            criticalCorrelation = (0.3 + 0.4 * np.sqrt(sf / 25)) * 100
        else:
            criticalCorrelation = mincor

        ds['mc'] = np.logical_or(
            np.logical_or(
                ds['cor1'] > criticalCorrelation,
                ds['cor2'] > criticalCorrelation),
            ds['cor3'] > criticalCorrelation)

        # check if the correlation dropped no more than 5% of the signal, otherwise drop
        mc = ds.mc.sum(dim='N') > 0.95*N
        ds = ds.where(mc).dropna(dim='t')
        if len(ds.t) == 0: continue

        # interpolate cubic spline when the correlation dropped too low
        ds['u'] = ds.u.where(ds.mc).interpolate_na(dim='N', method='cubic', max_gap=max_gap)
        ds['v'] = ds.v.where(ds.mc).interpolate_na(dim='N', method='cubic', max_gap=max_gap)
        ds['w'] = ds.w.where(ds.mc).interpolate_na(dim='N', method='cubic', max_gap=max_gap)

        # check if all gaps are filled, otherwise drop
        mg = np.isnan(ds.u).sum(dim='N') == 0
        ds = ds.where(mg).dropna(dim='t')
        if len(ds.t) == 0: continue

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

        # mean velocities
        ds['um'] = ds.u.mean(dim='N')
        ds['vm'] = ds.v.mean(dim='N')
        ds['wm'] = ds.w.mean(dim='N')
        ds['um'].attrs = {'units': 'm/s', 'long_name': 'mean velocity E'}
        ds['vm'].attrs = {'units': 'm/s', 'long_name': 'mean velocity N'}
        ds['wm'].attrs = {'units': 'm/s', 'long_name': 'mean velocity U'}



        ds.to_netcdf(os.path.join(outfolder, filestr))

