# weathercase

**Understand extreme weather events in context.**

`weathercase` animates historical weather events from ERA5 reanalysis data 
and contextualizes them against climatology.

## Quickstart

```bash
git clone https://github.com/sotarolab/weathercase.git
cd weathercase
conda env create -f environment.yml
conda activate weathercase
python app.py
# Open http://localhost:8050
```

## Event Catalog

Named events defined in `catalog/events.yaml`. Add any event in 5 lines of YAML.

## Built with

`xarray` · `ERA5` · `Plotly Dash` · `NumPy` · `pandas`