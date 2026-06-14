"""
camanchaca.catalog
-------------------
Loads named weather events from the YAML catalog.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

CATALOG_PATH = Path(__file__).parent.parent.parent / "catalog" / "events.yaml"

class EventMeta:
    def __init__(self, key, name, start, end, bbox, variables,
                 description="", tags=None, marker=None):
        self.key = key
        self.name = name
        self.start = start
        self.end = end 
        self.bbox = bbox
        self.variables = variables
        self.description = description
        self.tags = tags if tags is not None else []
        self.marker = marker
    def __repr__(self):
        return (
            f"EventMeta(key='{self.key}', "
            f"name='{self.name}', "
            f"start='{self.start}', "
            f"end='{self.end}')"
        )
    
def load_catalog(path = CATALOG_PATH):
    # Open file
    with open(path) as f:
        raw = yaml.safe_load(f)
    catalog = {} # container to collect results
    for key, meta in raw["events"].items():
        # key  = "snowzilla_2016"
        # meta = nested dict with name, start, end
        catalog[key] = EventMeta(
            key = key,
            name = meta["name"],
            start = meta["start"],
            end = meta["end"],
            bbox = meta["bbox"],
            variables = meta["variables"],
            description = meta.get("description", ""), #dictionary.get(key, default) # meta["description"] returns error if description not in YAML
            tags = meta.get("tags", []),
            marker = meta.get("marker")
        )
    return catalog

# Non-Pythonic format 
def get_event(key):
    # key = "Snowzilla_2016"
    catalog = load_catalog()
    
    included = False
    for event_key,_ in catalog.items():
        if key == event_key:
            included = True
    if not included:
        raise KeyError(f"Event '{key}' not found. Available: {list(catalog.keys())}")
    else:
        return catalog[key]
# Return list of events in catalog, e.g. 
#['snowzilla_2016', 'russian_river_ar_2017', 'hurricane_harvey_2017']
def list_events():
    catalog = load_catalog()
    return list(catalog.keys())

# Professional Python
# def get_event(key):
#     # key = "Snowzilla_2016"
#     catalog = load_catalog()
#     if key not in catalog:
#         raise KeyError(f"Event '{key}' not found. Available: {list(catalog.keys())}")
#     return catalog[key]


# Return something like: snowzilla_2016_2016-01-21_2016-01-25_merged.nc"
def get_data_file(event_key):
    event = get_event(event_key)
    return f"{event_key}_{event.start}_{event.end}_merged.nc"