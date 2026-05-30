# camanchaca

**Understand extreme weather events in context.**

`camanchaca` animates historical weather events from ERA5 reanalysis data
and benchmarks them against climatology — answering the question:
*how rare was this event?*

Named after the coastal fog phenomenon of northern Chile, shaped by the
Humboldt Current and large-scale atmospheric dynamics off the South American coast.

---

## What it does (v1)

- Select an event from a catalog of historic extremes
- Animate ERA5 fields (precipitation, temperature, pressure, wind) across the event window
- Show how rare the event was — percentile rank against the ERA5 climatological baseline

## Event catalog (v1)

| Event | Type | Region | Year |
|---|---|---|---|
| Snowzilla | Blizzard | US East Coast | 2016 |
| Russian River AR | Atmospheric River | California | 2023 |
| Hurricane Harvey | Tropical / Extreme Precip | Gulf Coast | 2017 |

## Quickstart

```bash
git clone https://github.com/sotarolab/camanchaca.git
cd camanchaca
conda env create -f environment.yml
conda activate camanchaca
python app.py
# Open http://localhost:8050
```

## Project layout

```
camanchaca/
├── app.py                        Dash application
├── catalog/
│   └── events.yaml               Event definitions — add any event in ~10 lines
├── data/                         Local ERA5 NetCDF files (gitignored)
└── src/camanchaca/
    ├── loader.py                 ERA5 loader and unit conversions
    ├── catalog.py                YAML catalog interface
    └── benchmark/
        ├── base.py               BaseBenchmark — shared interface
        └── percentile.py         PercentileRankBenchmark (v1)
```

## Roadmap

- **v1** — event toggling, ERA5 animation, percentile rank benchmarking
- **v2** — return period analysis, extreme value distributions
- **v3** — NWP forecast verification, ML-based error characterization

## Built with

`xarray` · `ERA5 / CDS` · `Plotly Dash` · `NumPy` · `pandas`

---

*Part of [sotarolab](https://github.com/sotarolab)*
