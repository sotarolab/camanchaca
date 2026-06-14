# camanchaca — ML Forecast Error Model Design

## Motivation

NWP models produce systematic, lead-time-dependent errors that are partially
predictable from the forecast fields themselves. Rather than applying a static
climatological bias correction, this module trains a neural network to predict
the full error profile across the forecast horizon — producing a dynamic,
event-aware correction that improves with lead time as the forecast converges
toward observed conditions.

The central question: **given a single NWP model run evolving forward in time,
can we predict the error at every lead time simultaneously?**

This is the Phase 3 capstone of camanchaca, building on:
- Phase 1's event catalog and climatological benchmarking (return periods, percentile rank)
- Phase 2's forecast retrieval and verification (`src/camanchaca/forecast/`) — bias/RMSE
  by lead time, which establishes the naive baseline this model must beat

---

## Problem Formulation

### Setup

For a single forecast point P and a single NWP model run initialized at time T₀:

- The model produces forecasts at lead times `τ ∈ {1, 2, ..., L}` hours
- Each lead time has a forecast value `f(τ)` for variable X
- The verification observation is `o(τ)` — ERA5 reanalysis or ASOS station obs
- The forecast error is `ε(τ) = f(τ) - o(τ)`

### Task

Learn a mapping:

```
g : X_input → ε_pred

X_input  shape (L, F)   — forecast time series + features at each lead time
ε_pred   shape (L,)     — predicted error at every lead time
```

At inference time, the corrected forecast is:

```
f_corrected(τ) = f(τ) - ε_pred(τ)
```

### Why sequence-to-sequence

Predicting the full error profile `(L,)` rather than a scalar at one lead time:

- Captures how errors grow and decay across the forecast horizon
- Allows the correction to taper gracefully at short lead times (where NWP
  is already accurate) and amplify at long lead times (where drift accumulates)
- Enables downstream uncertainty quantification by examining the error profile shape

---

## Data Pipeline

### Sources

| Source | Role | Access |
|---|---|---|
| GFS archive | Primary NWP forecast | NOAA NOMADS / AWS, via `Herbie` (`src/camanchaca/forecast/fetch_gfs.py`) |
| ERA5 (ECMWF) | Verification truth (gridded) | Copernicus CDS via `cdsapi` (`src/camanchaca/fetch.py`) |
| ASOS (FAA/NWS) | Verification truth (point obs) | Iowa State ASOS API |

GFS is the Phase 2 default for forecast retrieval. HRRR (higher resolution,
shorter horizon) is a possible alternative source for CONUS events if GFS
lead-time coverage proves insufficient — same interface, different fetch module.

### Sample Construction

One training sample = one forecast run at one spatial point:

```
sample = {
    "X":     ndarray (L, F),   # input features at each lead time
    "y":     ndarray (L,),     # error = forecast - observation at each lead time
    "meta":  {
        "point":    (lat, lon),
        "init_time": T0,
        "variable": "t2m",
        "event":    "snowzilla_2016",
    }
}
```

Samples are drawn from the camanchaca event catalog (`catalog/events.yaml`) —
each catalog entry defines the event window, spatial domain, and variables of
interest. This keeps the training dataset reproducible and event-stratified.

Initial catalog events available for sampling: `snowzilla_2016`,
`russian_river_ar_2023`, `hurricane_harvey_2017`. More events can be added to
the catalog (~10 lines of YAML) to grow the training set.

### Feature Set (F channels)

**Baseline (F = 1):**

| Feature | Description |
|---|---|
| `f(τ)` | NWP forecast value of variable X at lead time τ |

**Extended (F = 4, planned):**

| Feature | Description |
|---|---|
| `f(τ)` | NWP forecast value |
| `f(τ) - f(τ-Δ)` | Forecast tendency (first difference) |
| `hour_of_day(τ)` | Encoded as sin/cos pair — captures diurnal error patterns |
| `obs_nearest(τ)` | Nearest ASOS observation (when available) — real-time signal |

Feature engineering is additive — the model architecture accepts arbitrary F,
so extending the feature set requires no architectural changes.

---

## Baselines (established in Phase 1-2, required before this model is meaningful)

Before training any model, two baselines must be computed for comparison:

1. **Naive / persistence**: `ε_pred(τ) = 0` (no correction) — the raw forecast
2. **Climatological bias correction**: `ε_pred(τ) = mean(ε(τ))` across all
   historical samples at that lead time — a constant-per-lead-time correction,
   computed via `src/camanchaca/forecast/verify.py`

Any learned model (MLP, LSTM, Transformer) must demonstrably beat #2 to be
worth the added complexity. This comparison is the headline result of Phase 3.

---

## Model Architecture

### Step 0: MLP baseline (simplest learned model)

Before any sequence model, a per-lead-time feedforward network that ignores
sequence structure — treats each `(τ, f(τ))` pair independently. This isolates
"can a model learn anything beyond the climatological mean" from "does sequence
structure help."

```python
import torch.nn as nn

class MLPErrorModel(nn.Module):
    """Per-lead-time MLP baseline — no sequence structure."""

    def __init__(self, n_features: int = 1, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # x: (batch, L, F) -> treat each (L) step independently
        return self.net(x).squeeze(-1)  # (batch, L)
```

### Step 1: LSTM Encoder (sequence baseline)

A standard sequence-to-sequence LSTM that reads the full forecast trajectory
and decodes an error estimate at every step.

```
Input (L, F)
    │
    ▼
LSTM Encoder
  hidden_size = 128
  num_layers  = 2
  dropout     = 0.2
    │
    ▼  hidden states (L, hidden_size)
    │
Linear Head
  hidden_size → 1
    │
    ▼
Output (L,)   ← predicted error at each lead time
```

```python
import torch
import torch.nn as nn

class LSTMErrorModel(nn.Module):
    """
    Sequence-to-sequence LSTM for NWP forecast error prediction.

    Args:
        n_features:   Number of input channels F (default 1 = forecast value only)
        hidden_size:  LSTM hidden dimension
        num_layers:   Number of stacked LSTM layers
        dropout:      Dropout between LSTM layers
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, L, F) — forecast time series
        Returns:
            error: (batch, L) — predicted error at each lead time
        """
        out, _ = self.lstm(x)          # (batch, L, hidden_size)
        return self.head(out).squeeze(-1)  # (batch, L)
```

### Step 2 (stretch): Transformer Encoder

For longer forecast horizons (L > 72hr) or richer feature sets, a Transformer
encoder with positional encoding outperforms LSTM by capturing non-local
dependencies across lead times — e.g. a forecast anomaly at T+6 informing
the correction at T+48.

```
Input (L, F)
    │
Linear Projection → d_model
    │
Positional Encoding (lead-time aware)
    │
Transformer Encoder
  d_model  = 128
  nhead    = 8
  n_layers = 3
  dim_ff   = 256
    │
Linear Head → 1
    │
Output (L,)
```

```python
class TransformerErrorModel(nn.Module):
    """
    Transformer encoder for NWP forecast error prediction.
    Lead-time positional encoding replaces standard sequence position encoding.
    """

    def __init__(
        self,
        n_features: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_lead_hours: int = 48,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = nn.Embedding(max_lead_hours, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, L, F)
        Returns:
            error: (batch, L)
        """
        L = x.shape[1]
        positions = torch.arange(L, device=x.device)          # (L,)
        z = self.input_proj(x) + self.pos_enc(positions)      # (batch, L, d_model)
        z = self.encoder(z)                                    # (batch, L, d_model)
        return self.head(z).squeeze(-1)                        # (batch, L)
```

---

## Training

### Loss Function

Mean squared error over all lead times, with optional lead-time weighting to
emphasize errors at longer lead times where NWP drift is largest:

```python
def weighted_mse_loss(
    pred: torch.Tensor,     # (batch, L)
    target: torch.Tensor,   # (batch, L)
    weights: torch.Tensor | None = None,  # (L,) optional lead-time weights
) -> torch.Tensor:
    loss = (pred - target) ** 2           # (batch, L)
    if weights is not None:
        loss = loss * weights.unsqueeze(0)
    return loss.mean()
```

Lead-time weights can be set to linearly increase with `τ` — up-weighting long
lead errors without ignoring short-lead performance.

### Training Loop Sketch

```python
from camanchaca.ml import LSTMErrorModel, ForecastErrorDataset
from torch.utils.data import DataLoader

dataset = ForecastErrorDataset(
    catalog_events=["snowzilla_2016", "russian_river_ar_2023", "hurricane_harvey_2017"],
    variable="t2m",
    lead_hours=48,
    n_features=1,
)
train_ds, val_ds = dataset.split(val_fraction=0.2, stratify_by="event")

model = LSTMErrorModel(n_features=1, hidden_size=128, num_layers=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(100):
    for X, y in DataLoader(train_ds, batch_size=64, shuffle=True):
        pred = model(X)
        loss = weighted_mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
```

### Evaluation Metrics

| Metric | Description |
|---|---|
| MAE by lead time | Primary metric — error magnitude at each τ |
| Skill score vs. climatological baseline | `1 - MAE_model / MAE_climatology` — does the model beat baseline #2? |
| Bias by lead time | `mean(ε_pred - ε_true)` — is the model systematically off? |
| Sharpness | Std of predicted error across events — is the model collapsing to mean? |

---

## Module Layout

Matches the planned structure in the project [README](README.md):

```
src/camanchaca/ml/
├── __init__.py               Guards torch import (optional dependency)
├── dataset.py                ForecastErrorDataset — sample construction from catalog
├── models/
│   ├── mlp.py                 MLPErrorModel (baseline)
│   ├── lstm.py                LSTMErrorModel
│   └── transformer.py         TransformerErrorModel (stretch)
├── loss.py                    weighted_mse_loss
├── train.py                   Training loop, checkpointing, validation
├── correction.py               BiasCorrector — apply trained model at inference
└── evaluate.py                MAE by lead time, skill score, bias diagnostics
```

Depends on `src/camanchaca/forecast/` (Phase 2) for forecast/truth pairs and
the climatological baseline, and `catalog/events.yaml` for event definitions.

---

## Integration with camanchaca App

At inference time, the corrected forecast is a drop-in replacement for the raw
NWP field in the Dash visualization:

```python
from camanchaca.ml import BiasCorrector

corrector = BiasCorrector.from_checkpoint("checkpoints/lstm_t2m.pt")
corrected_ds = corrector.apply(forecast_ds, points=asos_stations)

# Dash app receives corrected_ds in place of raw forecast_ds
# Visualization layer is unchanged
```

The Dash UI gains a toggle: **Raw Forecast / Corrected Forecast / Error Field** —
allowing the user to see the correction magnitude spatially across the event.

---

## Development Phases (within camanchaca Phase 3)

| Step | Scope | Infrastructure |
|---|---|---|
| 3a — Baselines | Compute naive + climatological bias correction from Phase 2 forecast/truth pairs | Fully offline, reproducible |
| 3b — MLP | Train MLP baseline, compare skill vs. 3a | Offline |
| 3c — LSTM | Train LSTM, compare skill vs. 3a/3b | Offline |
| 3d — Validation | Skill score vs. baselines across held-out events, lead-time diagnostics | Offline |
| 3e — Transformer (stretch) | If LSTM shows skill and L > 72hr matters | Offline |
| 3f — ASOS features (stretch) | Add nearest-station obs as input feature, retrain | Offline (ASOS API batch) |

Operational/live ingestion is explicitly out of scope — this is a retrospective,
reproducible research tool, not an operational forecasting system.

---

## Key Design Decisions

- **Variable-agnostic architecture** — `n_features` is a constructor argument;
  the same model class handles temperature, precipitation, and wind without
  subclassing or conditional logic.
- **Sequence-to-sequence output** — predicting `(L,)` rather than a scalar
  produces a full corrected forecast trajectory, not just a point estimate.
- **Catalog-stratified splits** — train/val splits respect event boundaries,
  preventing data leakage across a storm's forecast window.
- **Baseline-first** — a model is only worth building if it beats the
  climatological bias correction from Phase 2. MLP before LSTM before
  Transformer, each step justified by the previous one's results.
- **Correction is additive** — `f_corrected = f - ε_pred` keeps the physical
  interpretation clean and makes the model's contribution auditable.
