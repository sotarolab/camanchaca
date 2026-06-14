import sys
sys.path.insert(0, 'src')
from camanchaca.catalog import load_catalog, get_event, list_events

print(list_events())
meta = get_event('snowzilla_2016')
print(meta)

