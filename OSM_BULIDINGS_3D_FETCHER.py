import requests
import struct
import math

class RealTimeCityFetcher:
    def __init__(self):
        self.api_url = "https://overpass-api.de/api/interpreter"
        self.mat_map = {'concrete': 0, 'brick': 1, 'glass': 2, 'wood': 3}
        self.roof_map = {'flat': 0, 'gabled': 1, 'pyramidal': 2, 'dome': 3}

    def _haversine_offset(self, lon, lat, center_lon, center_lat):
        r_earth = 6378137.0
        d_lon = math.radians(lon - center_lon)
        d_lat = math.radians(lat - center_lat)
        x = r_earth * d_lon * math.cos(math.radians(center_lat))
        y = r_earth * d_lat
        return x, y

    def pull_tile(self, fetch_state: dict) -> dict | None:
        """
        Rewritten to match SentinelFetcher signature.
        Returns metadata + binary geometry.
        """
        bbox = fetch_state["bbox"]
        min_lon, min_lat, max_lon, max_lat = bbox
        center_lon, center_lat = (min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0

        query = f"""
        [out:json][timeout:25];
        (way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
         relation["building"]({min_lat},{min_lon},{max_lat},{max_lon}););
        out body; >; out skel qt;
        """
        
        try:
            response = requests.post(self.api_url, data={'data': query}, timeout=30)
            if response.status_code != 200: return None
            data = response.json()
            
            nodes = {n['id']: (n['lon'], n['lat']) for n in data.get('elements', []) if n['type'] == 'node'}
            binary_payload = bytearray()

            for el in data.get('elements', []):
                if el['type'] == 'way' and 'tags' in el and 'nodes' in el:
                    node_refs = el['nodes']
                    if len(node_refs) < 3: continue

                    # Attributes & Packing
                    h = float(el['tags'].get('height', float(el['tags'].get('building:levels', 1)) * 3.5))
                    mat = self.mat_map.get(el['tags'].get('building:material', 'concrete'), 0)
                    roof = self.roof_map.get(el['tags'].get('roof:shape', 'flat'), 0)
                    
                    # Pack Header & Vertices
                    binary_payload.extend(struct.pack('<H f B B', len(node_refs), h, mat, roof))
                    for n_id in node_refs:
                        if n_id in nodes:
                            x, y = self._haversine_offset(nodes[n_id][0], nodes[n_id][1], center_lon, center_lat)
                            binary_payload.extend(struct.pack('<f f', x, y))

            return {
                "pixels": bytes(binary_payload), # This is your geometry stream
                "bbox": bbox,
                "key": fetch_state.get("key"),
                "slot": fetch_state.get("slot"),
                "type": "geometry"
            }
        except Exception:
            return None