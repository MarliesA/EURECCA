# %%
from datetime import datetime  # pour time count
from raw_extract import raw_extract
import os
import pickle
import numpy as np

# %%
path = r"c:\Temp\dep1\raw\raw_20231101_160545.udt"
pathout = r"c:\Temp\dep1"

# %% [markdown]
# Do the actual debinarization

# %%
extract_start = datetime.now()
(
    device_name,
    time_begin,
    time_end,
    param_us_dicts,
    data_us_dicts,
    data_dicts,
    settings_dict
) = raw_extract(path)

print(
        "=============\nextract duration:%s\n==========="
        % (datetime.now() - extract_start)
    )
print(device_name)
print(time_begin, time_end)


# %% [markdown]
# Extracted data are arranged in dictionnaries:

# %%
print("================\nUltrasound (US) measurement parameter\n==================")
print("Configuration numbers available: ", param_us_dicts.keys()) # gives the configuration numbers available in this set of data
first_configuration = list(param_us_dicts.keys())[0] # gives the configuration number of the first available configuration
print("Receiving channels available for one of the configurations: ", param_us_dicts[first_configuration].keys()) # gives the list of receiving channels (transducers) for this first configuration
first_channel = list(param_us_dicts[first_configuration].keys())[0] # gives the channel number of the first available channel for this first configuration
print("US Parameters for one receiving channel of one of the configurations: ", param_us_dicts[first_configuration][first_channel]) # gives the set of parameters used for the measurement of the data associated to this first configuration anad first receiving channel (emission frequency, PRF etc.)

# %% [markdown]
# data_dicts: measured data not measured by ultrasound or not related directly those who ultrasound

# %%
print("\n================\nNon Ultrasound (US) measured/recorded data\n==================")
# thus, they are not related to a number of configuration nor a receiving channel
# could be empty if no such data is available or has been recorded
if data_dicts:
    print("Available non US datatypes: ", data_dicts.keys()) # gives the datatypes available for this recording
    first_datatype = list(data_dicts.keys())[0] # one of those datatypes
    print("timestamp list for the first datatype %s: "%first_datatype, data_dicts[first_datatype]["time"]) # gives the list of timestamps associated to the list of values for this datatype
    print("and corresponding data: ", data_dicts[first_datatype]["data"]) # gives the list of values associated to the list of timestamps for this datatype. The values cas be arrays.


# %%

print("\n================\nUltrasound (US) measured/recorded data\n==================")
# those are also first arranged by configuration number and receiving channel number
print(data_us_dicts[first_configuration][first_channel].keys()) # gives the US datatypes available for one configuration and one channel
first_us_datatype = list(data_us_dicts[first_configuration][first_channel].keys())[0]
timestamp_first_US_datatype = data_us_dicts[first_configuration][first_channel][first_us_datatype]["time"] # gives the list of timestamps associated to the list of values for this datatype
print("earliest available data: %s\nlastest available data: %s"%(min(timestamp_first_US_datatype),max(timestamp_first_US_datatype)))
corresponding_US_data = data_us_dicts[first_configuration][first_channel][first_us_datatype]["data"] # gives the list of values associated to the list of timestamps for this datatype. The values cas be arrays.
print(len(corresponding_US_data))

# %% [markdown]
# Saving to file in parts

# %%
print("\n================\nSaving the data in parts to file\n==================")
nt = len(data_us_dicts[1][1]["velocity_profile"]['time'])
nparts = int(np.ceil(nt/1e5))
bl = int(1e5)
# save in blocks of 1e5 moments in time
for ipart in range(nparts-1):
    with open(os.path.join(pathout, 'time_1_{}.pickle'.format(ipart)), 'wb') as handle:
        pickle.dump(data_us_dicts[1][1]["velocity_profile"]['time'][ipart*bl:(ipart+1)*bl], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(pathout, 'time_2_{}.pickle'.format(ipart)), 'wb') as handle:
        pickle.dump(data_us_dicts[1][1]["velocity_profile"]['time'][ipart*bl:(ipart+1)*bl], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(pathout, 'time_3_{}.pickle'.format(ipart)), 'wb') as handle:
        pickle.dump(data_us_dicts[1][1]["velocity_profile"]['time'][ipart*bl:(ipart+1)*bl], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(pathout, 'time_4_{}.pickle'.format(ipart)), 'wb') as handle:
        pickle.dump(data_us_dicts[1][1]["velocity_profile"]['time'][ipart*bl:(ipart+1)*bl], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(pathout, 'vel_1_{}.pickle'.format(ipart)), 'wb') as handle:
        pickle.dump(data_us_dicts[1][1]["velocity_profile"]['data'][ipart*bl:(ipart+1)*bl], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(pathout, 'vel_2_{}.pickle'.format(ipart)), 'wb') as handle:
        pickle.dump(data_us_dicts[1][2]["velocity_profile"]['data'][ipart*bl:(ipart+1)*bl], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(pathout, 'vel_3_{}.pickle'.format(ipart)), 'wb') as handle:
        pickle.dump(data_us_dicts[1][3]["velocity_profile"]['data'][ipart*bl:(ipart+1)*bl], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(pathout, 'vel_4_{}.pickle'.format(ipart)), 'wb') as handle:
        pickle.dump(data_us_dicts[1][4]["velocity_profile"]['data'][ipart*bl:(ipart+1)*bl], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(pathout, 'a_1_{}.pickle'.format(ipart)), 'wb') as handle:
        pickle.dump(data_us_dicts[1][1]["echo_profile"]['data'][ipart*bl:(ipart+1)*bl], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(pathout, 'a_2_{}.pickle'.format(ipart)), 'wb') as handle:
        pickle.dump(data_us_dicts[1][2]["echo_profile"]['data'][ipart*bl:(ipart+1)*bl], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(pathout, 'a_3_{}.pickle'.format(ipart)), 'wb') as handle:
        pickle.dump(data_us_dicts[1][3]["echo_profile"]['data'][ipart*bl:(ipart+1)*bl], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(pathout, 'a_4_{}.pickle'.format(ipart)), 'wb') as handle:
        pickle.dump(data_us_dicts[1][4]["echo_profile"]['data'][ipart*bl:(ipart+1)*bl], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(pathout, 'snr_1_{}.pickle'.format(ipart)), 'wb') as handle:
        pickle.dump(data_us_dicts[1][1]["snr_doppler_profile"]['data'][ipart*bl:(ipart+1)*bl], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(pathout, 'snr_2_{}.pickle'.format(ipart)), 'wb') as handle:
        pickle.dump(data_us_dicts[1][2]["snr_doppler_profile"]['data'][ipart*bl:(ipart+1)*bl], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(pathout, 'snr_3_{}.pickle'.format(ipart)), 'wb') as handle:
        pickle.dump(data_us_dicts[1][3]["snr_doppler_profile"]['data'][ipart*bl:(ipart+1)*bl], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(pathout, 'snr_4_{}.pickle'.format(ipart)), 'wb') as handle:
        pickle.dump(data_us_dicts[1][4]["snr_doppler_profile"]['data'][ipart*bl:(ipart+1)*bl], handle, protocol=pickle.HIGHEST_PROTOCOL)   

#save the last part
with open(os.path.join(pathout, 'time_1_{}.pickle'.format(nparts-1)), 'wb') as handle:
    pickle.dump(data_us_dicts[1][1]["velocity_profile"]['time'][(nparts-1)*bl:], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(pathout, 'time_2_{}.pickle'.format(nparts-1)), 'wb') as handle:
    pickle.dump(data_us_dicts[1][1]["velocity_profile"]['time'][(nparts-1)*bl:], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(pathout, 'time_3_{}.pickle'.format(nparts-1)), 'wb') as handle:
    pickle.dump(data_us_dicts[1][1]["velocity_profile"]['time'][(nparts-1)*bl:], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(pathout, 'time_4_{}.pickle'.format(nparts-1)), 'wb') as handle:
    pickle.dump(data_us_dicts[1][1]["velocity_profile"]['time'][(nparts-1)*bl:], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(pathout, 'vel_1_{}.pickle'.format(nparts-1)), 'wb') as handle:
    pickle.dump(data_us_dicts[1][1]["velocity_profile"]['data'][(nparts-1)*bl:], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(pathout, 'vel_2_{}.pickle'.format(nparts-1)), 'wb') as handle:
    pickle.dump(data_us_dicts[1][2]["velocity_profile"]['data'][(nparts-1)*bl:], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(pathout, 'vel_3_{}.pickle'.format(nparts-1)), 'wb') as handle:
    pickle.dump(data_us_dicts[1][3]["velocity_profile"]['data'][(nparts-1)*bl:], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(pathout, 'vel_4_{}.pickle'.format(nparts-1)), 'wb') as handle:
    pickle.dump(data_us_dicts[1][4]["velocity_profile"]['data'][(nparts-1)*bl:], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(pathout, 'a_1_{}.pickle'.format(nparts-1)), 'wb') as handle:
    pickle.dump(data_us_dicts[1][1]["echo_profile"]['data'][(nparts-1)*bl:], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(pathout, 'a_2_{}.pickle'.format(nparts-1)), 'wb') as handle:
    pickle.dump(data_us_dicts[1][2]["echo_profile"]['data'][(nparts-1)*bl:], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(pathout, 'a_3_{}.pickle'.format(nparts-1)), 'wb') as handle:
    pickle.dump(data_us_dicts[1][3]["echo_profile"]['data'][(nparts-1)*bl:], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(pathout, 'a_4_{}.pickle'.format(nparts-1)), 'wb') as handle:
    pickle.dump(data_us_dicts[1][4]["echo_profile"]['data'][(nparts-1)*bl:-1], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(pathout, 'snr_1_{}.pickle'.format(nparts-1)), 'wb') as handle:
    pickle.dump(data_us_dicts[1][1]["snr_doppler_profile"]['data'][(nparts-1)*bl:-1], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(pathout, 'snr_2_{}.pickle'.format(nparts-1)), 'wb') as handle:
    pickle.dump(data_us_dicts[1][1]["snr_doppler_profile"]['data'][(nparts-1)*bl:-1], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(pathout, 'snr_3_{}.pickle'.format(nparts-1)), 'wb') as handle:
    pickle.dump(data_us_dicts[1][1]["snr_doppler_profile"]['data'][(nparts-1)*bl:-1], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(pathout, 'snr_4_{}.pickle'.format(nparts-1)), 'wb') as handle:
    pickle.dump(data_us_dicts[1][1]["snr_doppler_profile"]['data'][(nparts-1)*bl:-1], handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%
print("\n================\nSaving the data in parts to file\n==================")
with open(os.path.join(pathout, 'time_orientation.pickle'.format(ipart)), 'wb') as handle:
    pickle.dump(data_dicts["pitch"]['time'], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(pathout, 'pitch.pickle'.format(ipart)), 'wb') as handle:
    pickle.dump(data_dicts["pitch"]['data'], handle, protocol=pickle.HIGHEST_PROTOCOL)    
with open(os.path.join(pathout, 'roll.pickle'.format(ipart)), 'wb') as handle:
    pickle.dump(data_dicts["roll"]['data'], handle, protocol=pickle.HIGHEST_PROTOCOL)


