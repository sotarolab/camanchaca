# camanchaca

**Understand extreme weather events in context.**

`camanchaca` visualizes historical weather events from ERA5 reanalysis data and
benchmarks them against climatology — answering the question: *how rare was this event?*

It also retrieves operational NWP forecasts for each event and trains a neural
network to predict forecast error across the lead-time horizon, enabling dynamic
bias correction for extreme weather.

Named after the coastal fog phenomenon of northern Chile, shaped by the
Humboldt Current and large-scale atmospheric dynamics off the South American coast.

---

## Current status

**Implemented:**
- ERA5 loading and preprocessing (`loader.py`) — unit conversion, precipitation
  accumulation handling, time coordinate cleanup
- Event catalog system (`catalog.py`, `catalog/events.yaml`) — events defined
  declaratively (date range, bounding box, variables, description), with a
  `get_data_file()` helper that derives the expected ERA5 filename per event
- Interactive Dash app — animated map of ERA5 fields (precipitation, temperature,
  pressure, wind) with play/pause and time slider

**In progress / planned — see Roadmap below.**

---

## Event catalog

| Event | Type | Region | Year | Status |
|---|---|---|---|---|
| Snowzilla | Blizzard | US East Coast | 2016 | Working in app |
| Russian River AR | Atmospheric River | California | 2023 | Catalog only |
| Hurricane Harvey | Tropical / Extreme Precip | Gulf Coast | 2017 | Catalog only |
| Winter Storm Uri | Arctic Outbreak / Winter | Texas / South-Central US | 2021 | Planned |
| Hurricane Ida | Tropical / Inland Flooding | Gulf Coast → Northeast | 2021 | Planned |
| Pacific NW Heat Dome | Heatwave | Pacific Northwest | 2021 | Planned |
| European Windstorm Klaus | Extratropical Cyclone | Bay of Biscay / SW Europe | 2009 | Planned |
| Hurricane Sandy | Hybrid / Coastal Surge | US East Coast | 2012 | Planned |

Events span five distinct peril types (winter storm, atmospheric river, tropical,
heatwave, extratropical wind) and two continents. Each entry is ~10 lines of YAML
in `catalog/events.yaml`.

---

## Architecture

```
camanchaca/
│
├── app.py                        Dash application (event selector, animated map)     ✅
├── requirements.txt              Python dependencies                                  ✅
│
├── catalog/
│   └── events.yaml               Declarative event definitions — add any event        ✅
│                                 in ~10 lines: dates, bbox, variables, tags
│
├── data/                         Local ERA5 / GFS files (gitignored)                 ✅
├── outputs/                      Generated figures, animations (gitignored)           ✅
│
├── docs/
│   ├── ml_error_model.md         Design doc: ML forecast bias correction model        ✅
│   └── architecture/             Diagrams, demo media                                 🔲 Phase 4
│
├── tests/
│   ├── test_catalog.py           Catalog loading and event lookup                     🔲 Phase 1
│   └── test_loader.py            ERA5 loading and unit conversions                    🔲 Phase 1
│
└── src/camanchaca/
    ├── __init__.py                                                                     ✅
    ├── loader.py                 ERA5 loader and unit conversions                     ✅
    ├── catalog.py                YAML catalog interface + get_data_file()             ✅
    │
    ├── fetch.py                  Automated ERA5 download via the Copernicus           🔲 Phase 1
    │                             CDS API, driven by catalog event definitions.
    │                             Replaces manual .nc placement in data/.
    │
    ├── benchmark/                Climatological benchmarking against ERA5             🔲 Phase 1
    │   ├── __init__.py
    │   ├── base.py               BaseBenchmark — shared interface for all
    │   │                         benchmark types (fit on climatology, score
    │   │                         an event field, return a labeled result)
    │   ├── percentile.py         PercentileRankBenchmark — where does an event
    │   │                         value sit in the historical distribution?
    │   └── return_period.py      ReturnPeriodBenchmark — extreme value analysis
    │                             (GEV / GPD fitting) answering "how rare in years?"
    │
    ├── forecast/                 Operational NWP forecast retrieval and               🔲 Phase 2
    │   ├── __init__.py           verification against ERA5 truth
    │   ├── fetch_gfs.py          Pull GFS archive for each catalog event via
    │   │                         Herbie; stores fields matching catalog variables
    │   │                         and spatial domain
    │   └── verify.py             Forecast skill metrics by lead time: bias,
    │                             RMSE, skill score vs. climatology. Produces
    │                             the baseline that Phase 3 must beat.
    │
    └── ml/                       Neural network forecast error correction             🔲 Phase 3
        ├── __init__.py
        ├── dataset.py            ForecastErrorDataset — constructs (X, y) samples
        │                         from catalog events: X = GFS forecast trajectory
        │                         (L, F), y = error vs. ERA5 at each lead time
        ├── models/
        │   ├── mlp.py            MLPErrorModel — per-lead-time feedforward baseline,
        │   │                     no sequence structure; establishes whether any
        │   │                     learned model adds skill over climatology
        │   ├── lstm.py           LSTMErrorModel — seq2seq LSTM that reads the full
        │   │                     forecast trajectory and predicts error at every
        │   │                     lead time simultaneously
        │   └── transformer.py    TransformerErrorModel — attention-based alternative
        │                         for longer horizons (L > 72 hr); stretch goal
        ├── loss.py               Weighted MSE — optionally up-weights long-lead
        │                         errors where NWP drift is largest
        ├── train.py              Training loop with AdamW + cosine LR schedule,
        │                         catalog-stratified train/val split to prevent
        │                         data leakage across a storm's forecast window
        ├── correction.py         BiasCorrector — loads a trained checkpoint and
        │                         applies f_corrected = f − ε_pred at inference;
        │                         drop-in replacement for raw forecast in the app
        └── evaluate.py           MAE by lead time, skill score vs. climatological
                                  baseline, bias diagnostics, sharpness metrics
```

✅ implemented · 🔲 planned

---

## Roadmap

### Phase 1 — Reproducible v1

- Automated ERA5 fetching via the CDS API (`fetch.py`), driven by catalog entries
- Wire the event selector in the Dash app (all catalog events, not just Snowzilla)
- Expand catalog to 8 events covering 5 peril types and 2 continents (see table above)
- Climatological benchmarking (`benchmark/`) — percentile rank and return period
  analysis, answering *how rare was this?* with real extreme value statistics

### Phase 2 — Forecast verification

- Pull GFS archive forecasts for each cataloged event (`forecast/fetch_gfs.py`)
- Compute bias and RMSE by lead time against ERA5 truth (`forecast/verify.py`)
- Establish the climatological bias correction baseline that Phase 3 must beat

### Phase 3 — ML-based bias correction

- Train the model sequence described in [`docs/ml_error_model.md`](docs/ml_error_model.md):
  MLP baseline → LSTM → Transformer (stretch)
- Primary result: skill score vs. climatological baseline across held-out events
  and lead times — does the model beat the naive correction?
- Toggle in the Dash app: raw forecast / corrected forecast / error field

### Phase 4 — Polish

- Architecture diagrams, demo walkthrough, example outputs

---

## Quickstart

```bash
git clone https://github.com/sotarolab/camanchaca.git
cd camanchaca
pip install -r requirements.txt
```

You'll need an ERA5 `.nc` file for the Snowzilla event placed in `data/`
(download via the [Copernicus CDS](https://cds.climate.copernicus.eu/)).
Automated fetching is the first item in Phase 1.

```bash
python app.py
# Open http://localhost:8050
```

---

## Built with

`xarray` · `ERA5 / CDS` · `Plotly Dash` · `NumPy` · `pandas` ·
`Herbie` (planned, Phase 2) · `PyTorch` (planned, Phase 3)

---

*Part of [sotarolab](https://github.com/sotarolab)*