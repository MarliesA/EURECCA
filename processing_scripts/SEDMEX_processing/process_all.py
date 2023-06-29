# -*- coding: utf-8 -*-
"""
Created on Fri Mar  23 10:02:14 2023

@author: marliesvanderl

- This script contains the workflow to do all processing on hydrodynamics of the PHZD.
- It reads and write to the TU Delft drive \\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20210908_campaign\instruments
- It reads all data and saves to netcdf, it publishes quality checked sets as well as tailored time series
- Check the respective scripts for further description of the processing steps performed
"""

if __name__ == "__main__":

    script_list = [
        "01_read_store_raw_data.py",
        "02_quality_control_adv.py",
        "02_quality_control_solos.py",
        "02_quality_control_ossis.py",
        "02_quality_control_ADCP.py",
        "03_compute_waves_pressure_loggers",
        "03_compute_waves_adv.py",
        "03_tailored_dataset_ADCP.py"
    ]

    # loop over scripts
    for script in script_list:
        with open(script) as f:
            exec(f.read())

