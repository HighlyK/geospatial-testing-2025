#i am gonna upgrade this to biome classify logic and return biome mask
import os
import numpy as np
import cv2
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point, box
from shapely.ops import unary_union
from shapely.prepared import prep
import geopandas as gpd
import rasterio # You'll need this for the NASA Mosaic

class WorldNavigator:
    # ==========================================================
    # SET YOUR PATHS HERE ONCE
    # ==========================================================
    SHAPEFILE_PATH = "C:\\Users\\ASUS\\Documents\\VisionSphere\\Ecoregions2017.zip"
    NASA_MOSAIC_PATH = "C:\\Users\\ASUS\Documents\\VisionSphere\\earth pyopengl texture\earth nightlights.png"
    # ==========================================================

    def __init__(self, resolution='10m'):
        self.deg_per_tile = 18.0

        # 1. CORE LAND/OCEAN
        land_file = shpreader.natural_earth(resolution=resolution, category='physical', name='land')
        self.land_polygons = unary_union(list(shpreader.Reader(land_file).geometries()))
        self.prepared_land = prep(self.land_polygons)

        # 2. POPULATED PLACES
        pop_file = shpreader.natural_earth(resolution=resolution, category='cultural', name='populated_places')
        pop_geoms = [geom.buffer(0.1) for geom in shpreader.Reader(pop_file).geometries()]
        self.pop_polygons = unary_union(pop_geoms)
        self.prepared_pop = prep(self.pop_polygons)

        # 3. BIOME DATA (Directly from ZIP)
        # Note: We use the zip:// prefix for geopandas to look inside the archive
        try:
            gdf = gpd.read_file(f"zip://{self.SHAPEFILE_PATH}")
            self.biome_preps = {
                2: prep(unary_union(gdf[gdf['BIOME_NAME'].str.contains('Tundra|Boreal', case=False)].geometry)),
                3: prep(unary_union(gdf[gdf['BIOME_NAME'].str.contains('Desert|Xeric', case=False)].geometry)),
                4: prep(unary_union(gdf[gdf['BIOME_NAME'].str.contains('Tropical.*Moist', case=False)].geometry))
            }
        except Exception as e:
            print(f"⚠️ Biome Load Failed (Check if zip is corrupted): {e}")
            self.biome_preps = {}

    def sample_nasa_mosaic(self, r, c, target_size=512):
        """Standalone class method to sample the 4096x4096 texture"""
        if not hasattr(self, 'nasa_full_img'):
            self.nasa_full_img = cv2.imread(self.NASA_MOSAIC_PATH, cv2.IMREAD_GRAYSCALE)
            if self.nasa_full_img is None:
                return np.zeros((target_size, target_size), dtype=np.uint8)

        h, w = self.nasa_full_img.shape 
        row_step, col_step = h / 10.0, w / 20.0

        y1, y2 = int(r * row_step), int((r + 1) * row_step)
        x1, x2 = int(c * col_step), int((c + 1) * col_step)

        crop = self.nasa_full_img[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((target_size, target_size), dtype=np.uint8)
            
        return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    def get_tile_logic(self, r, c, tile_size=512):
        lat_top = 90.0 - (r * self.deg_per_tile)
        lat_bot = 90.0 - ((r + 1) * self.deg_per_tile)
        lon_left = -180.0 + (c * self.deg_per_tile)
        lon_right = -180.0 + ((c + 1) * self.deg_per_tile)

        pixel_step = self.deg_per_tile / tile_size
        lats = lat_top - (np.arange(tile_size) + 0.5) * pixel_step
        lons = lon_left + (np.arange(tile_size) + 0.5) * pixel_step
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Initialize Masks
        land_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
        pop_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
        biome_mask = np.zeros((tile_size, tile_size), dtype=np.uint8) # 0:Ocean, 1:Land, 2:Tundra, 3:Desert, 4:Tropical
        nasa_ref = np.zeros((tile_size, tile_size), dtype=np.float32)

        tile_box = box(lon_left, lat_bot, lon_right, lat_top)

        # --- PIXEL SAMPLING (Vector Data) ---
        if self.prepared_land.intersects(tile_box):
            pts = [Point(x, y) for x, y in zip(lon_grid.ravel(), lat_grid.ravel())]
            
            # Map Land & Set Base Biome to 1
            land_hits = np.array([self.prepared_land.contains(pt) for pt in pts]).reshape(tile_size, tile_size)
            land_mask[land_hits] = 1
            biome_mask[land_hits] = 1

            # Map Populated Places
            if self.prepared_pop.intersects(tile_box):
                pop_hits = np.array([self.prepared_pop.contains(pt) for pt in pts]).reshape(tile_size, tile_size)
                pop_mask[pop_hits] = 1

            # Map Biomes (Overwrites Land ID 1 with 2, 3, or 4)
            for b_id, b_poly in self.biome_preps.items():
                if b_poly.intersects(tile_box):
                    hits = np.array([b_poly.contains(pt) for pt in pts]).reshape(tile_size, tile_size)
                    biome_mask[hits] = b_id

        nasa_ref = self.sample_nasa_mosaic(r, c, target_size=tile_size)
        # Final Cleanup
        ocean_mask = (land_mask == 0).astype(np.uint8)

        return {
            "land": land_mask,
            "ocean": ocean_mask,
            "pop": pop_mask,
            "biomes": biome_mask,
            "nasa_reference": nasa_ref,
            "coords": (lat_top, lat_bot, lon_left, lon_right)
        }