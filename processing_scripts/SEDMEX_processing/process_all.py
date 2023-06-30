# -*- coding: utf-8 -*-
"""
Created on Fri Mar  23 10:02:14 2023

@author: marliesvanderl

- This script contains the workflow to do all processing on hydrodynamics of the PHZD.
- It reads and write to the TU Delft drive \\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20210908_campaign\instruments
- It reads all data and saves to netcdf, it publishes quality checked sets as well as tailored time series
- Check the respective scripts for further description of the processing steps performed
"""
from datetime import datetime

if __name__ == "__main__":

    script_list = [
        "read_store_raw_data.py",
        "quality_control_adv.py",
        "quality_control_solos.py",
        "quality_control_ossis.py",
        "quality_control_ADCP.py",
        "tailored_timeseries_pressure.py"
        "tailored_timeseries_adv.py",
        "tailored_timeseries_ADCP.py"
    ]

    logFile = r'\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20210908_campaign\instruments\proclog.txt'

    # run through the scripts one by one
    for script in script_list:
        print(script)
        try:

            with open(script) as f:
                exec(f.read())

        except Exception as e:

            now = str(datetime.now())[:-7]
            errstring = "{}: {} not completed because of {}\n".format(now, script, e)

            # print to console
            print(errstring)

            # print to log
            with open(logFile, "a") as file_object:
                file_object.write(errstring)

