# weathercase — Framework Overview

## What It Is

`weathercase` is a Python framework for visualizing, contextualizing, and analyzing
extreme weather events. It combines ERA5 reanalysis data, climatological benchmarking,
and (planned) ML-based forecast error characterization into a single interactive tool.

It is not a forecast model. It is the **infrastructure** for loading, analyzing, and
rendering weather events: a common event lifecycle, a swappable analysis registry,
and an interactive Dash application. The primary application is **extreme event
contextualization** — answering the question *how unusual was this event, and how
well did forecasts capture it?*

---

## Architecture

```
weathercase/
├── src/
│   └── weathercase/
│       ├── core/
│       │   ├── interfaces.py         IWeatherCase — event lifecycle base class
│       │   ├── enums.py              EventState enum (lifecycle states)
│       │   └── registry.py           EventRegistry — discover analyzers by name
│       │
│       ├── events/                   Concrete event analyzers
│       │   ├── era5_event.py         ERA5Event — load and animate ERA5 fields
│       │   ├── climatology.py        ClimatologyBenchmark — ERA5 percentile context
│       │   └── forecast_verif.py     [planned] ForecastVerification — NWP vs ERA5
│       │
│       ├── ml/                       [planned] ML-based forecast error module
│       │   ├── error_model.py        ErrorPredictor — train/predict NWP bias
│       │   └── correction.py         BiasCorrector — apply learned correction
│       │
│       ├── loader.py                 ERA5 NetCDF loader, VARIABLE_META registry
│       ├── catalog.py                YAML event catalog interface
│       └── visualization.py          [planned] Top-level plot re-exports
│
├── catalog/
│   └── events.yaml                   Named event definitions (5 lines per event)
│
├── data/                             Local ERA5 NetCDF files (gitignored)
├── app.py                            Dash application entry point
└── docs/
    ├── FRAMEWORK.md                  This document
    ├── climatology.md                [planned] Benchmarking method design
    └── ml_error_model.md             [planned] Forecast error ML design
```

---

## Core Concepts

### 1. Event Lifecycle

Every weather case follows a strict state progression enforced by `EventState`:

```
CREATED → LOADED → BENCHMARKED → VERIFIED → RENDERED
```

| Method | Responsibility |
|---|---|
| `load()` | Fetch or read ERA5 fields for the event window |
| `benchmark()` | Compute climatological percentiles from ERA5 baseline |
| `verify()` | Compare NWP forecast against ERA5 truth (optional) |
| `render()` | Return Plotly figures for the Dash layout |

This mirrors the lifecycle contract in `sparsehydro` (`IModel`) and provides the same
benefits: analysis tools know exactly what state an event is in, making the
benchmarking and ML layers composable and testable.

### 2. Event Registry

Concrete event analyzers self-register using `@registry.register`. Any registered
analyzer can be instantiated by name — useful for the catalog loader and batch
comparisons across events.

```python
@registry.register
class ERA5Event(IWeatherCase):
    event_type = "era5"
    ...

event = registry.create("era5", name="snowzilla_2016")
```

### 3. Event Catalog

Events are defined in `catalog/events.yaml`. Each entry specifies the event name,
type, date range, spatial domain, and variables of interest. The catalog is the
only place where event-specific configuration lives — no hardcoded paths in app code.

```yaml
events:
  - name: snowzilla_2016
    type: era5
    label: "Snowzilla — January 2016"
    start: "2016-01-21"
    end: "2016-01-25"
    domain: { lon: [-85, -70], lat: [35, 45] }
    variables: [snow_depth, total_precipitation, u10, v10]
    tags: [blizzard, northeast, historic]

  - name: pineapple_express_2023
    type: era5
    label: "Pineapple Express — January 2023"
    start: "2023-01-09"
    end: "2023-01-13"
    domain: { lon: [-130, -110], lat: [32, 50] }
    variables: [total_precipitation, ivt, u850, v850]
    tags: [atmospheric-river, california, flooding]
```

Adding a new event requires only a YAML entry — no code changes.

---

## Current State (v0.1)

What is implemented and working:

- ERA5 NetCDF loader with `VARIABLE_META` registry (label, units, colormap per variable)
- Plotly Dash app with animated time slider and variable selector
- Mapbox tile background with DC reference marker
- Play/pause animation via `dcc.Interval`
- Single hardcoded event (Snowzilla 2016) — catalog selector UI is next

---

## Planned Modules

### Climatological Benchmarking

For each event, compute where the observed field sits relative to the ERA5 historical
distribution (1979–present) at each grid point. Output: percentile maps and
domain-averaged exceedance statistics.

```python
benchmark = ClimatologyBenchmark(
    variable="total_precipitation",
    baseline_period=(1979, 2020),
    aggregation="event_total",
)
percentile_map = benchmark.compute(event_ds)
# → xr.DataArray of percentile ranks, same grid as event
```

Key design decision: benchmarking is a separate registered component, not embedded
in the event loader. This allows swapping aggregation methods (event total,
peak intensity, duration above threshold) without changing the visualization layer.

### Forecast Verification and ML Error Characterization

The planned ML module addresses a specific question: **how does NWP forecast error
evolve during extreme events, and can it be predicted from forecast fields?**

Phase 1 — Retrospective analysis (no live infrastructure required):
- Archive HRRR/GFS forecasts for catalog events from NOAA NOMADS/AWS
- Pair with ERA5 truth at matching grid points
- Compute error fields: `ε(x, y, t) = F(x, y, t) - O(x, y, t)`

Phase 2 — ML error model:
- Features: NWP forecast fields at lead time T
- Target: forecast error at verification time T+Δt
- Model: gradient-boosted or shallow CNN operating on spatial error fields
- Output: corrected forecast field + uncertainty estimate

Phase 3 — Operational application (future):
- Same correction model applied to live HRRR output
- ASOS observations as real-time error signal for online updating

This architecture keeps Phases 1 and 2 entirely offline and reproducible, with
Phase 3 as an optional operational layer added later.

```python
# Phase 2 usage (planned)
from weathercase.ml import ErrorPredictor

model = ErrorPredictor(variable="total_precipitation", lead_hours=24)
model.fit(training_events)                        # list of catalog event names
correction = model.predict(forecast_ds)           # xr.DataArray correction field
corrected = forecast_ds["tp"] + correction
```

---

## Dependency Map

```
Core (always available)
  xarray, numpy, pandas, plotly, dash

weathercase[era5]
  + cdsapi          (ERA5 download from Copernicus CDS)

weathercase[ml]
  + scikit-learn    (baseline error models)
  + torch           (CNN spatial error model, optional)
  + lightgbm        (gradient-boosted error model, optional)
```

---

## Key Design Decisions

- **Lifecycle enforcement** keeps the analysis pipeline predictable — the Dash app
  knows a rendered event has already been loaded and benchmarked.
- **Event registry** decouples the catalog loader from concrete implementations —
  new event types (NWP ensembles, satellite retrievals) plug in without touching
  the app layer.
- **Catalog-first configuration** — all event-specific parameters live in YAML,
  making the codebase event-agnostic and the catalog the single source of truth.
- **Offline-first ML design** — Phase 1 and 2 are fully reproducible from archived
  data, avoiding live infrastructure complexity until the science is validated.
- **Optional dependencies are guarded** — importing `weathercase` always succeeds
  even if cdsapi, torch, or lightgbm are not installed.
