# weathercase — ML Forecast Error Model Design

## Motivation

NWP models produce systematic, lead-time-dependent errors that are partially
predictable from the forecast fields themselves. Rather than applying a static
climatological bias correction, this module trains a neural network to predict
the full error profile across the forecast horizon — producing a dynamic,
event-aware correction that improves with lead time as the forecast converges
toward observed conditions.

The central question: **given a single NWP model run evolving forward in time,
can we predict the error at every lead time simultaneously?**

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
| HRRR (NOAA) | Primary NWP forecast | NOAA NOMADS / AWS S3 `noaa-hrrr-bdp-pds` |
| ERA5 (ECMWF) | Verification truth (gridded) | Copernicus CDS via `cdsapi` |
| ASOS (FAA/NWS) | Verification truth (point obs) | Iowa State ASOS API |

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

Samples are drawn from the weathercase event catalog — each catalog entry defines
the event window, spatial domain, and variables of interest. This keeps the
training dataset reproducible and event-stratified.

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

## Model Architecture

### Baseline: LSTM Encoder

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

### Extended: Transformer Encoder

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
from weathercase.ml import LSTMErrorModel, ForecastErrorDataset
from torch.utils.data import DataLoader

dataset = ForecastErrorDataset(
    catalog_events=["snowzilla_2016", "pineapple_express_2023"],
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
| Skill score vs. raw forecast | `1 - MAE_corrected / MAE_raw` — does correction help? |
| Bias by lead time | `mean(ε_pred - ε_true)` — is the model systematically off? |
| Sharpness | Std of predicted error across events — is the model collapsing to mean? |

---

## Module Layout

```
weathercase/ml/
├── __init__.py               Guards torch import
├── dataset.py                ForecastErrorDataset — sample construction from catalog
├── models/
│   ├── lstm.py               LSTMErrorModel
│   └── transformer.py        TransformerErrorModel
├── loss.py                   weighted_mse_loss
├── train.py                  Training loop, checkpointing, validation
├── correction.py             BiasCorrector — apply trained model at inference
└── evaluate.py               MAE by lead time, skill score, bias diagnostics
```

---

## Integration with weathercase App

At inference time, the corrected forecast is a drop-in replacement for the raw
NWP field in the Dash visualization:

```python
from weathercase.ml import BiasCorrector

corrector = BiasCorrector.from_checkpoint("checkpoints/lstm_t2m.pt")
corrected_ds = corrector.apply(forecast_ds, points=asos_stations)

# Dash app receives corrected_ds in place of raw forecast_ds
# Visualization layer is unchanged
```

The Dash UI gains a toggle: **Raw Forecast / Corrected Forecast / Error Field** —
allowing the user to see the correction magnitude spatially across the event.

---

## Development Phases

| Phase | Scope | Infrastructure |
|---|---|---|
| 1 — Retrospective | Archive HRRR + ERA5 for catalog events, build dataset, train LSTM | Fully offline, reproducible |
| 2 — Validation | Skill score vs. raw NWP across held-out events, lead-time diagnostics | Offline |
| 3 — ASOS features | Add nearest-station obs as input feature, retrain | Offline (ASOS API batch) |
| 4 — Operational | Live HRRR ingestion, real-time correction, ASOS streaming | Live infrastructure |

Phase 4 is explicitly deferred until Phase 1–3 demonstrate skill. This avoids
building operational infrastructure before the science is validated.

---

## Key Design Decisions

- **Variable-agnostic architecture** — `n_features` is a constructor argument;
  the same model class handles temperature, precipitation, and wind without
  subclassing or conditional logic.
- **Sequence-to-sequence output** — predicting `(L,)` rather than a scalar
  produces a full corrected forecast trajectory, not just a point estimate.
- **Catalog-stratified splits** — train/val splits respect event boundaries,
  preventing data leakage across a storm's forecast window.
- **LSTM first, Transformer second** — LSTM is the baseline because it is
  simpler to debug and trains faster on short sequences (L ≤ 48). Transformer
  is the upgrade path for longer horizons or richer feature sets.
- **Correction is additive** — `f_corrected = f - ε_pred` keeps the physical
  interpretation clean and makes the model's contribution auditable.
