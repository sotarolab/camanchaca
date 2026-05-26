import sys
sys.path.insert(0, 'src')
from weathercase.loader import load_era5

ds = load_era5("data/snowzilla_2016_2016-01-21_2016-01-25_merged.nc")
print(ds)