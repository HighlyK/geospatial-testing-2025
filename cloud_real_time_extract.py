import os
import sys
import time
import datetime
import shutil
import bz2
import boto3
import numpy as np
import earthaccess
import eumdac
from pathlib import Path
from PIL import Image, ImageFilter
from botocore import UNSIGNED
from botocore.config import Config
from dotenv import load_dotenv
from scipy.ndimage import distance_transform_edt, grey_closing, gaussian_filter, sobel
import warnings
import hashlib
import requests
from tqdm import tqdm
import zipfile
from scipy.ndimage import gaussian_filter, grey_closing
from scipy.ndimage import distance_transform_edt
import scipy.ndimage as ndimage
import gc
from scipy.ndimage import map_coordinates, binary_erosion
import pathlib
# Silence Dask/Numpy warnings about curvature math
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- SATPY IMPORTS ---
from satpy import Scene
from pyresample import create_area_def

# Load Secrets
load_dotenv()

# --- [CONFIGURATION] ---
OUTPUT_SIZE = (4096, 8192)
FINAL_EXPORT_SIZE = (8192, 8192)
DIRS = {
    "cache": Path("data_cache"),
    "data_cache_vs": Path("data_cache_vs"),
    "output": Path("gpu_compressed_nrt_data_vs\cloud_mosaic")
}

def get_cube_areas(face_size=2048):
    """
    Defines 6 Gnomonic projections to cover the sphere perfectly as a cube.
    This replaces the Equirectangular global grid.
    """
    from pyresample import create_area_def
    
    # Face definitions: (Name, Center Lat, Center Lon, Rotation)
    # Order follows standard WebGL Cubemap: +X, -X, +Y, -Y, +Z, -Z
    configs = [
        ('px', 0, 90, 0),    ('nx', 0, -90, 0),  # Right, Left
        ('py', 90, 0, 0),    ('ny', -90, 0, 0),  # Top, Bottom
        ('pz', 0, 0, 0),     ('nz', 0, 180, 0)   # Front, Back
    ]
    
    areas = []
    for name, lat, lon, rot in configs:
        proj_dict = {
            'proj': 'gnom', 'lat_0': lat, 'lon_0': lon, 
            'datum': 'WGS84', 'units': 'm'
        }
        # Earth Radius ~6378137m. Extent for a 90-degree FOV cube face:
        extent = [-6378137, -6378137, 6378137, 6378137]
        areas.append(create_area_def(
            name, proj_dict, area_extent=extent, shape=(face_size, face_size)
        ))
    return areas

# Initialize AWS S3 (Public Access)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

SOURCES = [
    {"name": "GOES-19 (East)", "bucket": "noaa-goes19", "prod": "ABI-L2-MCMIPF", "type": "satpy_goes"},
    {"name": "GOES-18 (West)", "bucket": "noaa-goes18", "prod": "ABI-L2-MCMIPF", "type": "satpy_goes"},
    {"name": "Himawari-9 (East)", "bucket": "noaa-himawari9", "prod": "AHI-L1b-FLDK", "type": "satpy_himawari"},
    
    # --- METEOSAT RAW (DIRECT) ---
    {"name": "Meteosat-11 (Prime)", "coll": "EO:EUM:DAT:MSG:HRSEVIRI", "type": "meteosat_raw"},
    {"name": "Meteosat-9 (IODC)", "coll": "EO:EUM:DAT:MSG:HRSEVIRI-IODC", "type": "meteosat_raw"},
    
    # --- POLAR (NOAA + NPP) ---
    {"name": "NOAA-20 (Pure M15)", "prod": "VJ102MOD", "geo": "VJ103MOD", "type": "satpy_viirs"},
    {"name": "NOAA-21 (Pure M15)", "prod": "VJ202MOD", "geo": "VJ203MOD", "type": "satpy_viirs"},
    #{"name": "Suomi-NPP (Pure M15)", "prod": "VNP02MOD", "geo": "VNP03MOD", "type": "satpy_viirs"} # ADDED NPP
]

# --- [CORE UTILITIES] ---
    
def get_file_hash(files):
    """Creates a unique ID based on filenames AND their last-modified timestamps."""
    tag_parts = []
    for f in sorted([Path(f) for f in files]):
        # Get the filename + the last modified timestamp
        mtime = f.stat().st_mtime if f.exists() else 0
        tag_parts.append(f"{f.name}_{mtime}")
    
    tag = "".join(tag_parts)
    return hashlib.md5(tag.encode()).hexdigest()

def save_to_cache(name, data, weight, file_hash):
    cache_path = DIRS["cache"] / f"proc_{name}_{file_hash}.npz"
    np.savez_compressed(cache_path, data=data, weight=weight)
    print(f"      >> Cached processed layer: {cache_path.name}")

def cleanup_old_cache(max_age_hours=999999999999999):
    """Scans the cache directory and removes any stale .npz files older than 24h."""
    now = time.time()
    max_age_seconds = max_age_hours * 99999999999999999
    count = 0
    bytes_cleared = 0
    
    for f in DIRS["cache"].glob("proc_*.npz"):
        f_stat = f.stat()
        if now - f_stat.st_mtime > max_age_seconds:
            try:
                bytes_cleared += f_stat.st_size
                f.unlink()
                count += 1
            except:
                pass 
    if count > 0:
        mb = bytes_cleared / (1024 * 1024)
        print(f"      > Janitor: Purged {count} stale cache files ({mb:.1f} MB cleared).")

def cleanup_all_raw_data():
    """
    Nukes all raw satellite files and temp extracts.
    Ensures that new runs don't mix old orbital data with fresh data.
    """
    print("    🧹 [JANITOR] Stale data detected. Performing full raw data purge...")
    extensions = ['*.nc', '*.nc.bz2', '*.DAT', '*.DAT.bz2', '*.zip', '*.nat']
    
    # Clean both cache directories
    for directory in [DIRS["cache"], DIRS["data_cache_vs"]]:
        for ext in extensions:
            for f in directory.glob(ext):
                try:
                    f.unlink()
                except:
                    pass
    
    # Remove temp extract folder
    temp_dir = DIRS["data_cache_vs"] / "temp_extract"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    print("    🧹 [JANITOR] Purge complete. System is ready for fresh download.")

def load_from_cache(name, file_hash, max_age_hours=99999999999):
    """
    Checks if a processed layer exists. If it's older than 24h, 
    it triggers a global purge of all raw files to ensure a fresh start.
    """
    cache_path = DIRS["cache"] / f"proc_{name}_{file_hash}.npz"
    now = time.time()
    max_age_sec = max_age_hours * 3600
    
    if cache_path.exists():
        file_age = now - cache_path.stat().st_mtime
        
        if file_age < max_age_sec:
            try:
                with np.load(cache_path) as loader:
                    return loader['data'], loader['weight']
            except:
                cache_path.unlink()
        else:
            # DATA IS STALE: Perform the "Clean Slate" purge
            print(f"    ⚠️  Cache for {name} is stale ({file_age/3600:.1f} hours old).")
            cache_path.unlink()
            cleanup_all_raw_data()
            
    return None, None

def get_feather_mask(data_mask, fade_distance=150, power=2.0):
    """
    Creates a soft alpha mask. 
    """
    # Distance from the edge of the data
    dist = distance_transform_edt(data_mask)
    
    # Normalize
    feather = np.clip(dist / fade_distance, 0, 1)
    
    # The "Rapid Scale" trick:
    feather = np.power(feather, power)
    
    return feather
# --- [DOWNLOADERS] ---

def download_s3_files(bucket, keys):
    local_files = []
    # Ensure the directory is definitely there before boto3 touches it
    DIRS["cache"].mkdir(parents=True, exist_ok=True)
    
    for key in keys:
        fname = Path(key).name
        local_path = DIRS["cache"] / fname
        
        if local_path.exists() and (time.time() - local_path.stat().st_mtime < 84600):
            print(f"   >> [S3 CACHE HIT] {fname}")
        else:
            print(f"   > Downloading {fname} from S3...")
            try:
                # FIX: Force str() for Windows path compatibility
                s3.download_file(bucket, key, str(local_path))
            except Exception as e:
                print(f"   ❌ S3 Download Failed: {e}")
                continue
        
        # Handle BZ2 decompression with cache check
        if fname.endswith('.bz2'):
            dat_path = DIRS["cache"] / fname.replace('.bz2', '')
            if not dat_path.exists():
                print(f"   > Decompressing {fname}...")
                with bz2.open(local_path, 'rb') as s, open(dat_path, 'wb') as d:
                    shutil.copyfileobj(s, d)
            local_files.append(dat_path)
        else:
            local_files.append(local_path)
    return local_files

def find_latest_himawari_set(bucket, prod, hours_back=6):
    now = datetime.datetime.utcnow()
    minute_slots = ["00", "10", "20", "30", "40", "50"]
    for h in range(hours_back + 1):
        t = now - datetime.timedelta(hours=h)
        # Himawari structure: YYYY/MM/DD/HHmm
        prefix_base = f"{prod}/{t.year}/{t.strftime('%m')}/{t.strftime('%d')}/{t.strftime('%H')}"
        for M in reversed(minute_slots):
            prefix = f"{prefix_base}{M}/"
            try:
                resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
                if 'Contents' not in resp: continue
                # Band 13 is Clean IR
                files = [x['Key'] for x in resp['Contents'] if '_B13_' in x['Key'] and x['Key'].endswith('.DAT.bz2')]
                if len(files) == 10:
                    files.sort(key=lambda k: int(k.split('_')[-1].replace('.DAT.bz2', '')[2:4]))
                    return files
            except: continue
    return None

def find_latest_nc_file(bucket, prod):
    now = datetime.datetime.utcnow()
    for h in range(12):
        t = now - datetime.timedelta(hours=h)
        doy = t.strftime('%j')
        prefix = f"{prod}/{t.year}/{doy}/{t.strftime('%H')}/"
        try:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=200)
            if 'Contents' in resp:
                candidates = [obj for obj in resp['Contents'] if obj['Key'].endswith('.nc')]
                if candidates:
                    return [max(candidates, key=lambda x: x['LastModified'])['Key']]
        except: continue
    return None

def fetch_pure_noaa_multi(prod, geo_prod, count=20, force_nasa=False):
    # 1. SCAN LOCAL DISK
    all_nc = list(DIRS["cache"].glob(f"*{prod[:3]}*.nc")) 
    time_groups = {}
    
    for f in all_nc:
        parts = f.name.split(".")
        # Fix: Support any year from 2020-2029
        ts_part = next((p for p in parts if p.startswith('A20') and len(p) >= 8), None)
        time_part = next((p for p in parts if len(p)==4 and p.isdigit()), "")
        
        if ts_part:
            key = ts_part + time_part
            if key not in time_groups: time_groups[key] = []
            if f not in time_groups[key]:
                time_groups[key].append(f)

    valid_pairs = []
    seen_files = set()

    for key in sorted(time_groups.keys(), reverse=True):
        group = time_groups[key]
        d_file = next((f for f in group if prod in f.name), None)
        g_file = next((f for f in group if geo_prod in f.name), None)
        
        if d_file and g_file:
            if d_file not in seen_files:
                valid_pairs.append([d_file, g_file])
                seen_files.add(d_file)
                seen_files.add(g_file)

    # 2. OFFLINE CHECK
    if len(valid_pairs) >= count and not force_nasa:
        print(f"   >> [OFFLINE] Found {len(valid_pairs)} local pairs for {prod}. Skipping NASA.")
        return valid_pairs[:count]

    # 3. NASA SEARCH (DYNAMIC YEAR FIX)
    now = datetime.datetime.utcnow()
    print(f"   > Syncing {prod} with NASA (Current Year {now.year})...")
    
    try:
        earthaccess.login(strategy="environment")
        # Search window: Last 48 hours to be safe during year-over transitions
        start = (now - datetime.timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        rad_res = earthaccess.search_data(short_name=prod, temporal=(start, None), count=count)
        geo_res = earthaccess.search_data(short_name=geo_prod, temporal=(start, None), count=count)
        
        matched_nasa = []
        # Use a more flexible matching key (the timestamp part of the filename)
        rad_dict = {}
        for r in rad_res:
            try:
                # Map by the actual filename timestamp if metadata is weird
                url = r['umm']['RelatedUrls'][0]['URL']
                ts = url.split('/')[-1].split('.')[1] # e.g., A2026013
                tm = url.split('/')[-1].split('.')[2] # e.g., 1010
                rad_dict[ts+tm] = r
            except: continue
        
        for g in geo_res:
            try:
                url = g['umm']['RelatedUrls'][0]['URL']
                ts = url.split('/')[-1].split('.')[1]
                tm = url.split('/')[-1].split('.')[2]
                if ts+tm in rad_dict:
                    matched_nasa.append((rad_dict[ts+tm], g))
            except: continue
        
        for r_meta, g_meta in matched_nasa:
            r_url = r_meta['umm']['RelatedUrls'][0]['URL'].split('/')[-1]
            if any(r_url in str(p) for p in seen_files): continue

            print(f"   > Downloading {r_url}...")
            paths = earthaccess.download([r_meta, g_meta], str(DIRS["cache"]))
            if paths and len(paths) == 2:
                valid_pairs.append([Path(p) for p in paths])
                
    except Exception as e:
        print(f"   ! NASA Sync Error for {prod}: {e}")

    return valid_pairs[:count]

class TailorRecipe:
    def __init__(self, config):
        self.config = config
    def asdict(self):
        return self.config

def fetch_meteosat_raw(sat_name, coll_id, key, secret):
    try:
        # ENSURE DIRECTORY EXISTS
        DIRS["data_cache_vs"].mkdir(parents=True, exist_ok=True)
        
        token = eumdac.AccessToken((key, secret))
        datastore = eumdac.DataStore(token)
        coll = datastore.get_collection(coll_id)
        
        product = coll.search().first()
        if not product: return None

        print(f" > Found Raw {sat_name}: {product.sensing_start} UTC")
        local_p = DIRS["data_cache_vs"] / f"meteosat_{sat_name}_raw.zip"
        
        # FIX: Ensure we use str(local_p) to avoid Windows Path errors
        with product.open() as fsrc:
            total_size = int(fsrc.headers.get('Content-Length', 0))
            
            with open(str(local_p), "wb") as fdst, tqdm(
                desc=f" > Downloading {sat_name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                while True:
                    chunk = fsrc.read(1024 * 1024) 
                    if not chunk: break
                    fdst.write(chunk)
                    bar.update(len(chunk))
            
        return local_p
    except Exception as e:
        print(f" ! Raw Fetch Error: {e}")
        return None
        
def find_local_viirs_strips(prod_prefix, geo_prefix, cache_dir):
    """Finds matching Data and Geo files already on disk."""
    data_files = sorted(list(cache_dir.glob(f"{prod_prefix}*.nc")))
    strips = []
    for d in data_files:
        # Match by the timestamp in the filename (e.g., A2024123.1234)
        timestamp = d.name.split('.')[1] + '.' + d.name.split('.')[2]
        geo_matches = list(cache_dir.glob(f"{geo_prefix}.{timestamp}*.nc"))
        if geo_matches:
            strips.append([d, geo_matches[0]])
    return strips

def process_raw_meteosat(zip_path, cube_grids, cube_areas, sat_label=None):
    """Decodes raw EUMETSAT binary using SatPy directly to 6 Cube Faces."""
    try:
        f_hash = hashlib.md5(str(zip_path.name).encode()).hexdigest()
        label = sat_label if sat_label else "METEOSAT"
        gc.collect() 

        # --- CACHE CHECK (Check Face 0 to see if we processed this zip) ---
        c_data, _ = load_from_cache(f"{label}_{cube_areas[0].area_id}", f_hash)
        if c_data is not None:
            # If hit, load all 6 faces from cache
            for i, area in enumerate(cube_areas):
                face_label = f"{label}_{area.area_id}"
                c_data, _ = load_from_cache(face_label, f_hash)
                if c_data is not None:
                    mask = c_data > 100 
                    cube_grids[i][mask] = c_data[mask]
            print(f"      >> [CACHE HIT] {label} 6-Faces restored from .npz")
            return True

        # --- EXTRACTION ---
        extract_dir = DIRS["data_cache_vs"] / "temp_extract"
        extract_dir.mkdir(exist_ok=True)
        nat_file = None
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith('.nat'):
                    zip_ref.extract(file, extract_dir)
                    nat_file = extract_dir / file
                    break
        
        if not nat_file: return False

        # --- SATPY LOAD ---
        scn = Scene(reader='seviri_l1b_native', filenames=[str(nat_file)])
        channel = 'IR_108' 
        scn.load([channel], calibration='brightness_temperature')
        
        hit_any_data = False

        # --- 6-FACE LOOP ---
        for i, area in enumerate(cube_areas):
            face_label = f"{label}_{area.area_id}"
            
            # 1. Resample specifically to this face
            resampled_scn = scn.resample(area, resampler='nearest')
            
            # 2. Trigger computation safely
            dask_array = resampled_scn[channel].astype(np.float32)
            data = dask_array.values 

            # Clear heavy SatPy objects per loop
            del resampled_scn, dask_array
            gc.collect()

            # 3. Define Valid Data
            valid_mask = np.isfinite(data) & (data > 160) & (data < 330)
            valid_mask = binary_erosion(valid_mask, iterations=15)
            
            if np.any(valid_mask):
                hit_any_data = True
                alpha = get_feather_mask(valid_mask, fade_distance=60)
                
                print(f"    >> [DEBUG] {label} ({area.area_id}): Blending...")
                save_to_cache(face_label, data, np.ones_like(data), f_hash)
                
                # Smart Blend specifically on this face's grid
                current_pixels = cube_grids[i][valid_mask]
                new_pixels = data[valid_mask]
                blend_weights = alpha[valid_mask]
                
                bg_exists = current_pixels > 100
                blended = new_pixels * blend_weights + current_pixels * (1 - blend_weights)
                final_pixels = np.where(bg_exists, blended, new_pixels)
                
                cube_grids[i][valid_mask] = final_pixels
                
                del data, valid_mask, alpha, current_pixels, new_pixels, blended, final_pixels
                gc.collect()

        # Final cleanup
        del scn
        gc.collect()
        if nat_file.exists():
            try: nat_file.unlink()
            except: pass
            
        return hit_any_data
        
    except Exception as e:
        print(f"      ! Meteosat Error: {e}")
        gc.collect()
        return False

def process_with_satpy(files, reader, cube_grids, cube_areas, sat_label=None):
    try:
        f_hash = get_file_hash(files)
        label = sat_label if sat_label else reader.upper()
        
        # --- CACHE CHECK ---
        c_data, _ = load_from_cache(f"{label}_{cube_areas[0].area_id}", f_hash)
        if c_data is not None:
            for i, area in enumerate(cube_areas):
                face_label = f"{label}_{area.area_id}"
                c_data, _ = load_from_cache(face_label, f_hash)
                if c_data is not None:
                    mask = c_data > 160
                    cube_grids[i][mask] = c_data[mask]
            print(f"      >> [CACHE HIT] {label} 6-Faces restored from .npz")
            return True

        # --- SATPY LOAD ---
        scn = Scene(reader=reader, filenames=[str(f) for f in files])
        channel = {
            'abi_l2_nc': 'C13', 'ahi_hsd': 'B13', 
            'viirs_l1b': 'M15', 'seviri_l1b_native': 'IR_108'
        }.get(reader)
        
        scn.load([channel], calibration='brightness_temperature')
        
        hit_any_data = False

        # --- 6-FACE LOOP ---
        for i, area in enumerate(cube_areas):
            face_label = f"{label}_{area.area_id}"
            
            # Resample directly to the face
            resampled_scn = scn.resample(area, resampler='nearest')
            data = resampled_scn[channel].values.astype(np.float32)
            if hasattr(data, 'compute'): data = data.compute()
            
            # Define Valid Data
            valid_mask = np.isfinite(data) & (data > 160) & (data < 330)
            valid_mask = binary_erosion(valid_mask, iterations=15)
            
            if np.any(valid_mask):
                hit_any_data = True
                alpha_map = get_feather_mask(valid_mask, fade_distance=250)
                alpha_map = np.power(alpha_map, 1.2)
                
                # Cache the raw data for this specific face
                save_to_cache(face_label, data, np.ones_like(data), f_hash)
                
                current_pixels = cube_grids[i][valid_mask]
                new_pixels = data[valid_mask]
                blend_weights = alpha_map[valid_mask]
                
                # SMART BLEND
                bg_exists = current_pixels > 1.0 
                blended = (new_pixels * blend_weights) + (current_pixels * (1.0 - blend_weights))
                final_pixels = np.where(bg_exists, blended, new_pixels)
                
                cube_grids[i][valid_mask] = final_pixels

        return True if hit_any_data else False

    except Exception as e:
        print(f"    ❌ SATPY ERROR: {e}")
        return "RETRY_ME"

def smooth_hard_edges(grid):
    """Seamless horizontal smoothing using periodic padding."""
    print("    >> Detecting and smoothing harsh edges (Seamless)...")
    
    # 1. Pad horizontally by 100 pixels on each side to create a 'virtual loop'
    pad = 100
    padded = np.hstack([grid[:, -pad:], grid, grid[:, :pad]])
    
    # 2. Find edges on the padded version
    dx = sobel(padded, axis=0)
    dy = sobel(padded, axis=1)
    edge_map = np.hypot(dx, dy)
    edge_map = edge_map / (np.max(edge_map) + 1e-5)
    
    # 3. Blur the padded version
    blurred_padded = ndimage.gaussian_filter(padded, sigma=3)
    
    # 4. Blend
    threshold = 0.2 
    smooth_mask = np.clip(edge_map / threshold, 0, 1)
    final_padded = (padded * (1 - smooth_mask)) + (blurred_padded * smooth_mask)
    
    # 5. Crop back to original size
    return final_padded[:, pad:-pad]

def fill_orbital_gaps(grid, mask_threshold=10.0):
    """Gap fill with seamless horizontal polar blurring."""
    print(f"    >> FORCING gap fill (threshold < {mask_threshold}K)...")
    
    filled_grid = grid.copy()
    H, W = filled_grid.shape

    # (Keep your existing np.interp loop here as it already uses period=W)
    for i in range(H):
        row = filled_grid[i, :]
        invalid_mask = row < mask_threshold
        if not np.any(invalid_mask) or np.all(invalid_mask): continue
        valid_idx = np.where(~invalid_mask)[0]
        invalid_idx = np.where(invalid_mask)[0]
        filled_grid[i, invalid_idx] = np.interp(invalid_idx, valid_idx, row[valid_idx], period=W)

    # SEAMLESS POLAR BLUR FIX:
    # We use horizontal padding here too so the polar blur doesn't create a vertical line
    edge = int(H * 0.15)
    pad = 100
    for slice_idx in [slice(0, edge), slice(-edge, None)]:
        region = filled_grid[slice_idx, :]
        padded_region = np.hstack([region[:, -pad:], region, region[:, :pad]])
        blurred = ndimage.gaussian_filter(padded_region, sigma=(2, 12))
        filled_grid[slice_idx, :] = blurred[:, pad:-pad]

    return filled_grid

def forge_ktx2(png_path):
    """
    [SOVEREIGN UPGRADE] The KTX2 Forge. 
    Uses the VisionSphere local binary to compress to GPU-Native UASTC.
    """
    import subprocess
    import os

    # 1. PATH CONFIGURATION
    # Make sure this points to your actual toktx.exe
    TOKTX_PATH = r"KTX-Software\bin\toktx.exe" 
    
    # 2. FILE NAMING
    # Ensure we define out_ktx2 before using it
    out_ktx2 = png_path.with_suffix('.ktx2')

    # 3. COMMAND CONSTRUCTION
    cmd = [
        TOKTX_PATH,
        "--t2",
        "--2d",
        "--genmipmap",
        "--encode", "uastc",
        "--uastc_quality", "2",
        "--",               # The "Ambiguity Separator" (Crucial!)
        str(out_ktx2),      # Output file first
        str(png_path)       # Input file last
    ]

    # 4. EXECUTION
    try:
        # We use capture_output=True to grab errors if toktx fails
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"      >> Forged KTX2: {out_ktx2.name}")
        
        # Cleanup: Remove temporary PNG to save space
        if png_path.exists():
            png_path.unlink()

    except subprocess.CalledProcessError as e:
        print(f"      ❌ toktx failed on {png_path.name}")
        print(f"      [DETAILS] {e.stderr}") # This tells us EXACTLY why toktx is mad
    except Exception as e:
        print(f"      ❌ General Forge Error: {e}")

def main():
    face_size = 2048
    cloud_out_dir = Path(r"gpu_compressed_nrt_data_vs\cloud_mosaic")
    cloud_out_dir.mkdir(parents=True, exist_ok=True)

    face_names = ['px', 'nx', 'py', 'ny', 'pz', 'nz']
    
    # Pre-define paths (We now check the first face for freshness)
    check_file = cloud_out_dir / f"cloud_face_px.png"
    max_age_seconds = 999999999999999999
    full_rebuild_required = False
    
    if not check_file.exists():
        full_rebuild_required = True
    else:
        file_age = time.time() - check_file.stat().st_mtime
        if file_age > max_age_seconds:
            full_rebuild_required = True
            cleanup_all_raw_data() 

    # --- 2. THE SKIP LOGIC ---
    if not full_rebuild_required:
        print(f" >> [SKIP] Map is fresh ({file_age/3600:.1f}h old). No new processing needed.")
        return {"cubemap": {f: str(cloud_out_dir / f"cloud_face_{f}.png") for f in face_names}}
    
    DIRS["output"].mkdir(parents=True, exist_ok=True)

    # --- SOVEREIGN INIT: 6 CUBE FACES INSTEAD OF 1 FLAT GRID ---
    cube_areas = get_cube_areas(face_size=face_size)
    cube_grids = [np.zeros((face_size, face_size), dtype=np.float32) for _ in range(6)]

    # --- PHASE 1: GEOSTATIONARY ---
    print(f"\n[PHASE 1] Processing Geostationary Base (Direct to 6-Face)...")

    for src in SOURCES:
        if src['type'] == 'satpy_viirs': continue 
        print(f"\n >> Source: {src['name']}")

        try:
            if src['type'] == 'satpy_goes':
                sat_id = "G19" if "19" in src['name'] else "G18"
                files = sorted(list(DIRS["cache"].glob(f"OR_{src['prod']}*_{sat_id}_*.nc")))
                if not files:
                    keys = find_latest_nc_file(src['bucket'], src['prod'])
                    if keys: files = download_s3_files(src['bucket'], keys)

                for f in files:
                    # Pass cube_grids and cube_areas instead of global_grid
                    res = process_with_satpy([f], 'abi_l2_nc', cube_grids, cube_areas, sat_label=sat_id)
                    if res == "RETRY_ME" and f.exists(): f.unlink()

            elif src['type'] == 'satpy_himawari':
                all_h = sorted(list(DIRS["cache"].glob("HS_H09_*_B13_FLDK_*.DAT")))
                if len(all_h) < 10:
                    h_keys = find_latest_himawari_set(src['bucket'], src['prod'])
                    if h_keys: all_h = download_s3_files(src['bucket'], h_keys)

                if len(all_h) >= 10:
                    all_h.sort() 
                    for i in range(0, (len(all_h) // 10) * 10, 10):
                        chunk = all_h[i:i+10]
                        res = process_with_satpy(chunk, 'ahi_hsd', cube_grids, cube_areas, sat_label="HIMAWARI9")
                        if res == "RETRY_ME":
                            for f in chunk:
                                if f.exists(): f.unlink()

            elif src['type'] == 'meteosat_raw':
                sat_slug = "IODC" if "IODC" in src['name'] else "PRIME"
                current_label = f"METEOSAT_{sat_slug}" 
                zips = list(DIRS["data_cache_vs"].glob(f"meteosat_{sat_slug.capitalize()}_raw*.zip"))
                if not zips:
                    new_zip = fetch_meteosat_raw(sat_slug, src['coll'], os.getenv("EUMETSAT_KEY"), os.getenv("EUMETSAT_SECRET"))
                    if new_zip: zips = [new_zip]

                for z in zips:
                    process_raw_meteosat(z, cube_grids, cube_areas, sat_label=current_label)

        except Exception as e:
            print(f"   ❌ Error processing {src['name']}: {e}")

    print("\n" + "-"*60)
    print("[PHASE 2] Accumulating Polar Swaths (VIIRS)")
    print("-"*60)

    for src in [s for s in SOURCES if s['type'] == 'satpy_viirs']:
        sat_name = src['name'].split(" ")[0].replace("-", "").upper()
        sat_label = f"{sat_name}_VIIRS"

        strips = find_local_viirs_strips(src['prod'], src['geo'], DIRS["cache"])
        if not strips:
            strips = fetch_pure_noaa_multi(src['prod'], src['geo'], count=25)

        print(f"\n >> {sat_label}: Processing {len(strips)} unique strips.")
        processed_count = 0
        for strip in strips:
            res = process_with_satpy(strip, 'viirs_l1b', cube_grids, cube_areas, sat_label=sat_label)
            if res is True: processed_count += 1 

        print(f"    ✔ {processed_count} unique {sat_label} strips merged.")

    # --- PHASE 2.9: HEALING (Applied per-face) ---
    print("   >> Healing seams on cube faces...")
    for i in range(6):
        # We apply the closing filter to each face to kill minor noise
        cube_grids[i] = grey_closing(cube_grids[i], size=(5, 5))

    # --- PHASE 3: RENDER DIRECTLY TO KTX2 (No stretch math required!) ---
    print("\n[PHASE 3] Finalizing Reconstruction & Forging WebGL Assets...")

    cubemap_paths = {}
    total_coverage = 0

    for i, name in enumerate(face_names):
        grid = cube_grids[i]
        data_mask = grid > 100 
        
        # Calculate coverage for debug
        total_coverage += np.count_nonzero(data_mask)

        # 1. Temperature to Alpha Conversion
        final_clouds = np.zeros_like(grid)
        upper_bound, lower_bound = 276.0, 185.0
        cloud_val = np.clip((upper_bound - grid[data_mask]) / (upper_bound - lower_bound), 0, 1)
        final_clouds[data_mask] = np.power(cloud_val, 0.7)

        # 2. Add Dither to prevent banding
        noise = (np.random.uniform(-0.5, 0.5, grid.shape) + np.random.uniform(-0.5, 0.5, grid.shape)) / 255.0
        final_clouds = np.clip(final_clouds + noise, 0, 1)

        # 3. Create RGBA Buffer
        rgba_u8 = np.zeros((face_size, face_size, 4), dtype=np.uint8)
        rgba_u8[..., 0:3] = 255 
        rgba_u8[..., 3] = (final_clouds * 255).astype(np.uint8)

        # 4. Save PNG
        png_path = cloud_out_dir / f"cloud_face_{name}.png"
        Image.fromarray(rgba_u8).save(png_path)
        print(f"    >> Saved: {png_path.name}")
        
        # 5. FORGE TO KTX2 (The Monopoly Move)
        forge_ktx2(png_path)
        
        cubemap_paths[name] = str(png_path.absolute()).replace("\\", "\\\\")

    coverage_pct = total_coverage / (6 * face_size * face_size)
    print(f"    >> Global Satellite Coverage: {coverage_pct:.1%}")

    if coverage_pct < 0.005:
        print(f"❌ FATAL: No satellite data in grid. Check Cache or fetchers!")
        return

    print(f"🚀 SUCCESS: 6-Face Gnomonic Architecture Ready.")
    
    return {
        "cubemap": cubemap_paths
    }


if __name__ == "__main__":
    main()