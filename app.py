import sys
from pathlib import Path
import numpy as np
import pandas as pd 
import plotly.graph_objects as go 

from dash import Dash, Input, Output, State, callback, dcc, html

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from camanchaca.loader import VARIABLE_META, load_era5, get_time_labels, get_variable_array
from camanchaca.catalog import get_event, get_data_file, list_events 

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
event_keys = list_events()
default_event = event_keys[0]

filename = get_data_file(default_event)
DATA_FILE = ROOT / "data" / filename
ds = load_era5(DATA_FILE)

TIMES = ds.time.values
TIME_LABELS = get_time_labels(ds)
AVAILABLE_VARS = [v for v in VARIABLE_META if v in ds.data_vars]

# DC marker coordinates
DC_LON, DC_LAT = -77.0369, 38.9072

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Dash(__name__, title="camanchaca")

app.layout = html.Div(
    style={
        "fontFamily": "Inter, sans-serif",
        "backgroundColor": "#0f1117",
        "color": "#e0e0e0",
        "minHeight": "100vh",
        "padding": "24px",
    },
    children=[

        # Header
        html.H1("camanchaca",
                style={"color": "#60a5fa", "margin": "0", "fontSize": "1.8rem"}),
        html.P("Snowzilla 2016 — ERA5 Event Explorer",
               style={"color": "#9ca3af", "margin": "4px 0 16px 0"}),
        # Event dropdown
        dcc.Dropdown(
            id="event-name",
            options=[{"label": get_event(k).name, "value": k}
                     for k in event_keys],
            value=default_event,
            clearable=False,
            style={"width": "300px", "marginBottom": "16px"},
        ),

        # Variable dropdown
        dcc.Dropdown(
            id="var-select",
            options=[{"label": VARIABLE_META[v]["label"], "value": v}
                     for v in AVAILABLE_VARS],
            value=AVAILABLE_VARS[0],
            clearable=False,
            style={"width": "300px", "marginBottom": "16px"},
        ),

        # Time slider
        dcc.Slider(
            id="time-slider",
            min=0,
            max=len(TIMES) - 1,
            step=1,
            value=0,
            marks={i: TIME_LABELS[i] for i in range(0, len(TIMES), len(TIMES) // 6)},
        ),

        # Map
        dcc.Graph(id="map", style={"height": "500px"}),

        # Animation interval and state
        dcc.Interval(id="interval", interval=150, n_intervals=0, disabled=True),
        dcc.Store(id="playing", data=False),

        # Play button
        html.Button("▶ Play", id="play-btn", n_clicks=0,
                    style={"marginTop": "16px", "padding": "8px 20px",
                           "backgroundColor": "#2563eb", "color": "white",
                           "border": "none", "borderRadius": "6px",
                           "cursor": "pointer"}),
    ]
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("map", "figure"),
    Input("time-slider", "value"),
    Input("var-select", "value"),
)
def update_map(time_idx, var):
    z = get_variable_array(ds, var, time_idx)
    meta = VARIABLE_META[var]
    ts = pd.Timestamp(TIMES[time_idx]).strftime("%B %d, %Y %H:%M UTC")

    lats = ds.latitude.values
    lons = ds.longitude.values

    fig = go.Figure()

    fig.add_trace(go.Densitymapbox(
        lat=np.repeat(lats, len(lons)),
        lon=np.tile(lons, len(lats)),
        z=z.flatten(),
        radius=15,
        colorscale=meta["cmap"],
        zmin=float(ds[var].quantile(0.02)),
        zmax=float(ds[var].quantile(0.98)),
        colorbar=dict(title=meta["units"]),
        hovertemplate=f"Lon: %{{lon:.2f}}<br>Lat: %{{lat:.2f}}<br>{meta['label']}: %{{z:.2f}} {meta['units']}<extra></extra>",
    ))
    # fig.add_trace(go.Heatmap(
    #     z=z,
    #     x=lons,
    #     y=lats,
    #     colorscale=meta["cmap"],
    #     zmin=float(ds[var].quantile(0.02)),
    #     zmax=float(ds[var].quantile(0.98)),
    #     colorbar=dict(title=meta["units"]),
    #     hovertemplate=f"Lon: %{{x:.2f}}<br>Lat: %{{y:.2f}}<br>{meta['label']}: %{{z:.2f}} {meta['units']}<extra></extra>",
    #     zsmooth="best",
    # ))

    # fig.add_trace(go.Scatter(
    #     x=[DC_LON], y=[DC_LAT],
    #     mode="markers+text",
    #     marker=dict(size=10, color="#f59e0b"),
    #     text=["DC"],
    #     textposition="top right",
    #     showlegend=False,
    # ))
    #
    fig.add_trace(go.Scattermapbox(
    lat=[DC_LAT],
    lon=[DC_LON],
    mode="markers+text",
    marker=dict(size=10, color="#f59e0b"),
    text=["DC"],
    textposition="top right",
    showlegend=False,
    ))

    # fig.update_layout(
    #     title=f"Snowzilla 2016 — {meta['label']} | {ts}",
    #     paper_bgcolor="#1e2130",
    #     plot_bgcolor="#1e2130",
    #     font=dict(color="#e0e0e0"),
    #     margin=dict(l=60, r=20, t=50, b=50),
    #     xaxis=dict(title="Longitude", gridcolor="#374151"),
    #     yaxis=dict(title="Latitude", gridcolor="#374151",
    #                scaleanchor="x", scaleratio=1),
    # )
    fig.update_layout(
    title=f"Snowzilla 2016 — {meta['label']} | {ts}",
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=39.0, lon=-76.0),
        zoom=5,
    ),
    margin=dict(l=0, r=0, t=50, b=0),
    paper_bgcolor="#1e2130",
    font=dict(color="#e0e0e0"),
    )


    return fig
@callback(
    Output("interval", "disabled"),
    Output("playing", "data"),
    Output("play-btn", "children"),
    Input("play-btn", "n_clicks"),
    State("playing", "data"),
    prevent_initial_call=True,
)
def toggle_play(n_clicks, playing):
    new_state = not playing
    label = "⏸ Pause" if new_state else "▶ Play"
    return not new_state, new_state, label


@callback(
    Output("time-slider", "value"),
    Input("interval", "n_intervals"),
    State("time-slider", "value"),
    prevent_initial_call=True,
)
def advance_frame(n_intervals, current):
    return (current + 1) % len(TIMES)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("camanchaca running at http://localhost:8050")
    app.run(debug=True)