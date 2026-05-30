"""
camanchaca.loader
------------------

ERA5 data loading and preprocessing.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

VARIABLE_META = {
    "tp":{
        "label": "Hourly Precipitation",
        "units": "mm/hr",
        "cmap": "Blues"
        },
    "msl":{
        "label": "Mean Sea Level Pressure",
        "units": "hPa",
        "cmap": "RdBu_r"
    },
    "t2m":{
        "label": "2m-Temperature",
        "units": "°C",
        "cmap": "RdBu_r"
    },
    "u10":{
        "label": "10-m u wind component",
        "units": "m/s",
        "cmap": "RdBu"
    },
    "v10":{
        "label": "10-m v wind component",
        "units": "m/s",
        "cmap": "RdBu"
    },
}

def load_era5(path):
    """
    Load and preprocess an ERA5 .nc file.

    Parameters:
    ----------

    path : str or Path

        Path to the ERA5 .nc file.
    Returns
    -------
    xr.Dataset

        Analysis-ready dataset with converted units
    """
    path = Path(path) # Convert path to a Path object

    if not path.exists():
        raise FileNotFoundError(f"ERA5 file not found: {path}")
    
    ds = xr.open_dataset(path, engine="netcdf4")

    # ERA5 files that span ERA5 + ERA5T (near-real-time) carry an 'expver'
    # coordinate with conflicting values per variable; drop it so the dataset
    # can be reconstructed cleanly.
    if "expver" in ds.coords:
        ds = ds.drop_vars("expver")

    # If your time coord is valid_time, rename it (harmless if already 'time')
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    
    processed = {}
    for var in ds.data_vars:
        da = ds[var]
        if var == "tp":
            da = da.diff("time")*1000.0
            da = da.assign_coords(time=ds.time.isel(time=slice(1, None))) # Convert m to mm/h
            da = da.clip(min=0)     # clipped negative values
        elif var == "t2m":
            da = da - 273.15 # K to C
        elif var == "msl":
            da = da / 100 # Pa to hPa

        processed[var] = da #add to processed dict
        
    return xr.Dataset(processed)   # return cleaned dataset

def get_time_labels(ds):
    """Return list of readable time strings for the dashboard slider."""
    return [pd.Timestamp(t).strftime("%b %d %H:%M UTC") for t in ds.time.values]

def get_variable_array(ds, var, time_idx):
    """Return 2D numpy array for a variable at a given time index.

    Parameters
    ----------
    ds : xr.Dataset
        Cleaned ERA5 dataset from load_era5().
    var : str
        Variable name e.g. 'tp', 't2m', 'msl'.
    time_idx : int
        Integer index along the time dimension

    Returns
    -------
    np.ndarray
        2D array of shape (latitude, longitude)
    """
    da = ds[var] 
    da_sub = da.isel(time=time_idx)
    return da_sub.values