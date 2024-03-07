def encoding_sedmex(ds):
    """
    returns a dictionary with netcdf encoding attributes for all variables used in SEDMEX processing with appropriate precision and specification of possible compression.
    """
    encoding = {}
    for var in ['t']:
        encoding[var] = dict(zlib=False, _FillValue=None, units='seconds since 2021-09-01 00:00:00', calendar='proleptic_gregorian')
    for var in ['sf', 'N']:
        encoding[var] = dict(zlib=False, _FillValue=None, )
    for var in ['a1', 'a2', 'a3', 'cor1', 'cor2', 'cor3', 'anl1', 'anl2', 'burst', 'mc', 'mu', 'md', 'ma', 'beachOri']:
        encoding[var] = dict(zlib=True, complevel=5, dtype='int16', scale_factor=1, _FillValue=-9999)
    for var in ['snr1', 'snr2', 'snr3', 'heading', 'pitch', 'roll']:
        encoding[var] = dict(zlib=True, complevel=5, dtype='int16', scale_factor=0.1, _FillValue=-9999)
<<<<<<< HEAD
    for var in ['sf', 'h', 'hpres', 'io', 'zi', 'zip', 'elevp', 'elev', 'd']:
        encoding[var] = dict(zlib=True, complevel=5, dtype='int16', scale_factor=0.01, _FillValue=-9999)
    for var in ['p', 'eta']:
        encoding[var] = dict(zlib=True, complevel=5, dtype='int16', scale_factor=0.001, _FillValue=-9999)
    for var in ['u', 'v', 'w', 'uang', 'umag', 'Hm0', 'Tp', 'Tm01', 'Tm02', 'Tmm10', 'Tps', 'puvdir', 'dspr', 'u_ssm', 'ucm', 'ulm', 'Sk', 'As', 'sig', 'zs', 'k', 'Ur', 'nAs', 'nSk' ]:
        encoding[var] = dict(zlib=True, complevel=5, dtype='int16', scale_factor=0.0001, _FillValue=-9999)
=======
    for var in ['sf', 'h', 'hpres', 'io', 'zi', 'zip', 'elevp', 'elev', 'd', 'zb']:
        encoding[var] = dict(zlib=True, complevel=5, dtype='int16', scale_factor=0.01, _FillValue=-9999)
    for var in ['p', 'eta']:
        encoding[var] = dict(zlib=True, complevel=5, dtype='int32', scale_factor=0.001, _FillValue=-9999)
    for var in ['u', 'v', 'w', 'uang', 'umag', 'Hm0', 'Tp', 'Tm01', 'Tm02', 'Tmm10', 'Tps','puvdir', 'dspr',  'u_ssm', 'ucm', 'ulm', 'Sk', 'As', 'sig', 'Skp', 'Asp', 'sigp', 'zs', 'k', 'Ur']:
        encoding[var] = dict(zlib=True, complevel=5, dtype='int32', scale_factor=0.0001, _FillValue=-9999)

>>>>>>> 9fd7e8a96eb2d4c102a99236399514a9c1c04e49

    # return only the subset on variables that are actually on the dataset
    encoding_sedmex = {var: encoding[var] for var in encoding if var in ds.data_vars}
    encoding_sedmex.update({var: encoding[var] for var in encoding if var in list(ds.coords.keys())})
    
    return encoding_sedmex