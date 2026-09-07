import os
from pathlib import Path
import concurrent.futures

# Your Pipeline Imports
from earth_map_3Months import earth_mosaic
from nightlights_nrt import night_mosaic
from cloud_real_time_extract import main as cloud_main

from HIGH_QUALITY_PULL_SENTINEL import SentinelFetcher
from OSM_BULIDINGS_3D_FETCHER import RealTimeCityFetcher

# --- ADD THIS IMPORT ---
from border_label_pulling import BorderAndLabel

class SatelliteEngine:
    def __init__(self):
        self.data_root = Path("gpu_compressed_nrt_data_vs")
        
        # 1. Tile & Vector Fetchers
        self.sentinel = SentinelFetcher() 
        self.city_fetcher = RealTimeCityFetcher()
        
        # --- INITIALIZE THE VECTOR ENGINE HERE ---
        # This solves the 'AttributeError: object has no attribute vectors'
        self.vectors = BorderAndLabel()

    def run_full_update(self):
        """
        Triggers all 6-face KTX2 generations.
        Returns the local paths for the frontend to load into the GPU.
        """
        print("[ENGINE] Forging Global Cubemaps (6 Faces Each)...")
        
        return {
            "earth": earth_mosaic(),
            "night": night_mosaic(),
            "clouds": cloud_main(),
            "height": {
                face: str(self.data_root / "height_mosaic" / f"height_face_{face}.ktx2")
                for face in ['px', 'nx', 'py', 'ny', 'pz', 'nz']
            }
        }

    def fetch_tile_stack(self, bbox, key, slot):
        """
        The Real-Time Detail Stream.
        Now includes a vectorized hierarchy for borders and labels.
        """
        state = {"bbox": bbox, "key": key, "slot": slot}
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f_img = executor.submit(self.sentinel.pull_tile, state)
            f_bld = executor.submit(self.city_fetcher.pull_tile, state)
            # You can also pre-fetch vectors here if you want them in the same payload
            f_vec = executor.submit(self.vectors.get_tactical_payload, bbox)

            return {
                "imagery": f_img.result(),
                "buildings": f_bld.result(),
                "vectors": f_vec.result()
            }