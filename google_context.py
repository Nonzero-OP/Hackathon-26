"""
google_context.py — Optional Google API enrichment for location context.

If no API key is configured this module degrades gracefully and returns
a placeholder context string so the rest of the pipeline is unaffected.

Requires:  pip3 install googlemaps
"""

from config import GOOGLE_API_KEY

_gmaps = None


def _client():
    global _gmaps
    if _gmaps is not None:
        return _gmaps
    if not GOOGLE_API_KEY:
        return None
    try:
        import googlemaps
        _gmaps = googlemaps.Client(key="AIzaSyCxWPHkYXopmWERS__oOR-rkKvw5mOqQ9w")
        print("[google_context] Google Maps client initialised.")
    except ImportError:
        print("[google_context] 'googlemaps' package not found. Run: pip3 install googlemaps")
    except Exception as exc:
        print(f"[google_context] Could not initialise client: {exc}")
    return _gmaps


def get_location_context(lat=None, lng=None):
    client = _client()
    if client is None or lat is None or lng is None:
        return "Location context unavailable (no API key or coordinates)."
    try:
        results = client.reverse_geocode((lat, lng))
        if results:
            address = results[0].get("formatted_address", "Unknown location")
            return f"Near: {address}"
    except Exception as exc:
        return f"Geocoding error: {exc}"
    return "No geocoding result."


def get_speed_limit(lat, lng):
    client = _client()
    if client is None:
        return "Speed limit: N/A"
    try:
        result = client.nearest_roads([(lat, lng)])
        if result and "speedLimits" in result:
            limit = result["speedLimits"][0].get("speedLimit", "?")
            units = result["speedLimits"][0].get("units", "MPH")
            return f"Speed limit: {limit} {units}"
    except Exception:
        pass
    return "Speed limit: N/A"


def build_context_line(lat=None, lng=None):
    loc = get_location_context(lat, lng)
    if lat is not None and lng is not None and GOOGLE_API_KEY:
        spd = get_speed_limit(lat, lng)
        return f"{loc} | {spd}"
    return loc