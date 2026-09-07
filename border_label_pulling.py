import geopandas as gpd
import requests
import logging
import warnings
import concurrent.futures
import io
import math
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from shapely.geometry import box

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class BorderAndLabel:
    def __init__(self):
        self.timeout = 30
        
        # Core Sources
        self.NE_BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
        self.GB_API_BASE = "https://www.geoboundaries.org/api/current/gbOpen/"
        self.OVERPASS_URL = "http://overpass-api.de/api/interpreter"
        self.WIKIDATA_URL = "https://query.wikidata.org/sparql" # New Source 4
        
        self.LAT_OFFSET = 2.0 

        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self._cache = {}

        # New Source 1: Global Land Mask (No Oceans)
        self.land_mask = self._safe_fetch(f"{self.NE_BASE}ne_10m_land.geojson")

    def _get_zoom_depth(self, bbox):
        area = abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])
        if area > 50: return 0  # Global (ADM0)
        if area > 10: return 1  # Regional (ADM1)
        if area > 2:  return 2  # Local (ADM2)
        return 3                # Township/Village (ADM3/ADM4)

    def _safe_fetch(self, url, is_geojson=True):
        if url in self._cache: return self._cache[url]
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            if is_geojson:
                gdf = gpd.read_file(io.BytesIO(response.content), engine="pyogrio")
                if not gdf.empty:
                    gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
                self._cache[url] = gdf
                return gdf
            return response.json()
        except Exception as e:
            logging.debug(f"Fetch skipped/failed: {url} | {e}")
            return gpd.GeoDataFrame() if is_geojson else {}

    # --- SHADER READY MATH (Cube Sphere Conversion) ---
    def _lonlat_to_cube(self, lon, lat):
        """Converts raw geo coordinates to 6-Face Cube Sphere UVs"""
        lat_adj = lat + self.LAT_OFFSET
        phi = lat_adj * (math.pi / 180.0)
        theta = lon * (math.pi / 180.0)

        # Spherical to Cartesian
        x = math.cos(phi) * math.cos(theta)
        y = math.cos(phi) * math.sin(theta)
        z = math.sin(phi)

        absX, absY, absZ = abs(x), abs(y), abs(z)
        maxAxis = max(absX, absY, absZ)
        
        if maxAxis == absX:
            face = 0 if x > 0 else 1
            u, v = (y/absX, z/absX) if x > 0 else (-y/absX, z/absX)
        elif maxAxis == absY:
            face = 2 if y > 0 else 3
            u, v = (-x/absY, z/absY) if y > 0 else (x/absY, z/absY)
        else:
            face = 4 if z > 0 else 5
            u, v = (y/absZ, -x/absZ) if z > 0 else (-y/absZ, -x/absZ)
            
        return face, round(u, 6), round(v, 6)

    def _process_geom_to_shader_arrays(self, geom, shader_data):
        """Extracts vertices, clips oceans, and pushes straight to face buckets"""
        if not geom or geom.is_empty: return
        
        def process_coords(coords):
            for lon, lat in coords:
                face, u, v = self._lonlat_to_cube(lon, lat)
                shader_data[f"face_{face}"].extend([u, v])

        if geom.geom_type in ['Polygon', 'LineString']:
            process_coords(geom.exterior.coords if hasattr(geom, 'exterior') else geom.coords)
        elif geom.geom_type in ['MultiPolygon', 'MultiLineString']:
            for part in geom.geoms:
                process_coords(part.exterior.coords if hasattr(part, 'exterior') else part.coords)

    def _fuse_and_flatten(self, gdf, bbox, hints, admin_level, shader_data):
        if gdf.empty: return []
        
        view_limit = box(*bbox)
        clipped = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].copy()
        if clipped.empty: return []
        
        cols = {c.lower(): c for c in clipped.columns}
        name_col = next((cols[h.lower()] for h in hints if h.lower() in cols), None)
        
        labels = []
        for _, row in clipped.iterrows():
            orig_geom = row.geometry
            visible_geom = orig_geom.intersection(view_limit)
            
            # NO OCEANS: Intersect with land mask
            if not self.land_mask.empty:
                visible_geom = visible_geom.intersection(self.land_mask.unary_union)
                
            if visible_geom.is_empty: continue

            # Send lines directly to shader buckets
            self._process_geom_to_shader_arrays(visible_geom, shader_data)
            
            # Extract Label Data
            pt = visible_geom.representative_point() 
            face, u, v = self._lonlat_to_cube(pt.x, pt.y)
            labels.append({
                "n": str(row[name_col]) if name_col and str(row[name_col]) != 'nan' else "Unknown",
                "face": face,
                "u": u, "v": v,
                "lvl": admin_level
            })
        return labels

    def _fetch_rich_labels(self, bbox):
        """Expanded label query combining OSM and NE Populated Places"""
        labels = []
        
        # Source 2: Natural Earth Major Cities (Fallback/Rich base)
        ne_cities = self._safe_fetch(f"{self.NE_BASE}ne_10m_populated_places.geojson")
        if not ne_cities.empty:
            clipped_cities = ne_cities.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            for _, row in clipped_cities.iterrows():
                face, u, v = self._lonlat_to_cube(row.geometry.x, row.geometry.y)
                labels.append({"n": row['NAME'], "face": face, "u": u, "v": v, "lvl": 2})

        # Source 3: Upgraded OSM Query (More detail)
        overpass_query = f"""
        [out:json][timeout:25];
        (
          node["place"~"city|town|village|suburb|hamlet|isolated_dwelling"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
          node["amenity"~"university|hospital|marketplace"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
          node["historic"~"ruins|monument|archaeological_site"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        );
        out body;
        """
        try:
            resp = self.session.post(self.OVERPASS_URL, data={'data': overpass_query}, timeout=self.timeout)
            for element in resp.json().get('elements', []):
                tags = element.get('tags', {})
                if 'name' in tags:
                    face, u, v = self._lonlat_to_cube(element['lon'], element['lat'])
                    labels.append({"n": tags['name'], "face": face, "u": u, "v": v, "lvl": 4})
        except Exception as e:
            logging.error(f"OSM fetch failed: {e}")
            
        return labels

    def get_tactical_payload(self, bbox):
        depth = self._get_zoom_depth(bbox)
        
        # Initialize Shader-Ready Buckets
        shader_data = {f"face_{i}": [] for i in range(6)}
        ui_labels = []

        # 1. Base Countries
        nat_gdf = self._safe_fetch(f"{self.NE_BASE}ne_10m_admin_0_countries.geojson")
        ui_labels.extend(self._fuse_and_flatten(nat_gdf, bbox, ['NAME_EN', 'NAME'], 0, shader_data))

        if depth == 0: 
            return {"shader_borders": shader_data, "labels": ui_labels, "meta": {"depth": depth}}

        isos = nat_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]['ADM0_A3'].dropna().unique().tolist()
        isos = [i for i in isos if len(str(i)) == 3 and i != '-99']

        # Parallel Fetching for Deep Admin Levels
        def process_country_hierarchy(iso):
            local_labels = []
            levels = ["ADM1", "ADM2"]
            if depth >= 3: levels.extend(["ADM3", "ADM4"]) # Source 4: Townships and Wards
            
            for lvl_idx, adm_level in enumerate(levels):
                if depth >= lvl_idx + 1:
                    api = self._safe_fetch(f"{self.GB_API_BASE}{iso}/{adm_level}/", is_geojson=False)
                    if api and api.get("gjDownloadURL"):
                        gdf = self._safe_fetch(api.get("gjDownloadURL"))
                        local_labels.extend(self._fuse_and_flatten(gdf, bbox, ['shapeName', 'name'], lvl_idx + 1, shader_data))
            return local_labels

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_country_hierarchy, iso) for iso in isos]
            for future in concurrent.futures.as_completed(futures):
                ui_labels.extend(future.result())

        # Inject high-density OSM/Natural Earth labels if zoomed in
        if depth >= 2:
            ui_labels.extend(self._fetch_rich_labels(bbox))

        return {
            "shader_borders": shader_data,
            "labels": ui_labels,
            "meta": {
                "depth": depth,
                "bbox": bbox,
                "format": "cube_sphere_uv",
                "point_count": sum(len(shader_data[k]) for k in shader_data) // 2
            }
        }