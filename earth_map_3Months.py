import datetime, requests, io, sys, os, shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import os
import shutil
import time
from datetime import timedelta
import gc
import cv2
import subprocess

def get_or_refresh_mosaic_path(tiles_dir, final_dir, ktx_dir, max_age_days=90):
    """
    Checks for fresh Mosaic and existing KTX2 files to provide paths to the backend.
    """
    target_file = "Global_CloudFree_Mosaic.jpg"
    final_image_path = os.path.join(final_dir, target_file)
    
    # Define the 6 expected KTX2 faces
    expected_faces = ['px.ktx2', 'nx.ktx2', 'py.ktx2', 'ny.ktx2', 'pz.ktx2', 'nz.ktx2']
    ktx_paths = []

    # Ensure directories exist
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(ktx_dir, exist_ok=True)

    # 1. Gather existing KTX2 paths if they exist
    existing_ktx = [f for f in os.listdir(ktx_dir) if f.endswith('.ktx2')]
    if all(face in existing_ktx for face in expected_faces):
        ktx_paths = [os.path.abspath(os.path.join(ktx_dir, f)) for f in expected_faces]

    # 2. Check if the high-res 4k Mosaic already exists and is fresh
    mosaic_ready = False
    if os.path.exists(final_image_path):
        file_age = (time.time() - os.path.getmtime(final_image_path)) / (24 * 3600)
        if file_age < max_age_days:
            print(f"[*] Found fresh 4K Mosaic ({int(file_age)} days old).")
            mosaic_ready = True
        else:
            print(f"[!] Mosaic expired ({int(file_age)} days).")

    # 3. Handle Tile expiration
    if os.path.exists(tiles_dir):
        folder_age = (time.time() - os.path.getmtime(tiles_dir)) / (24 * 3600)
        if folder_age > max_age_days:
            print(f"[!] Tiles are stale. Wiping...")
            shutil.rmtree(tiles_dir)
            os.makedirs(tiles_dir)

    # Return both. If ktx_paths is empty, the script knows it needs to forge them.
    return os.path.abspath(final_image_path), ktx_paths

# --- CONFIG ---
# 1. We start 7 days ago because "Today" often has no data yet on GIBS
START_DATE_OFFSET = 120 # 120 later
STACK_DEPTH = 60 # crank to 120 later
MAX_THREADS = 4

LAYERS = [
    #"MODIS_Terra_CorrectedReflectance_TrueColor" # This is the most reliable layer
    "VIIRS_NOAA20_CorrectedReflectance_TrueColor"
]
ZOOM = 4
TILE_SIZE = 512 # change to 512 if error happen
# A smaller test area (Mainland US/Atlantic) to see results fast
# Once this works, change it back to your full range
TILE_COORDS = [(r, c) for r in range(0, 10) for c in range(0, 20)]

BASE_DIR    = os.path.abspath(".")
TILES_DIR   = os.path.join(BASE_DIR, "tiles")
FINAL_DIR   = os.path.join(BASE_DIR, "gpu_compressed_nrt_data_vs", "earth_land_mosaic")
# This must match what the functions below use
KTX_OUT_DIR = os.path.join(FINAL_DIR, "ktx2_faces") 
TOKTX_PATH  = os.path.join(BASE_DIR, "KTX-Software", "bin", "toktx.exe")

# --- NETWORK ---
def get_robust_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    return session

def split_to_cubemap(image_path, out_dir):
    print("📦 Reprojecting Equirectangular to Spherified Cube faces...")
    img = cv2.imread(image_path)
    if img is None: raise Exception("Could not open source image.")
    
    # 1. FORCE EXACT 2:1 RATIO
    h, w = img.shape[:2]
    if w != 2 * h:
        img = cv2.resize(img, (4096, 2048), interpolation=cv2.INTER_CUBIC)
        h, w = 2048, 4096

    face_size = 1024
    names = ['px', 'nx', 'py', 'ny', 'pz', 'nz']
    face_paths = []

    # 2. CREATE BASE LINEAR GRID WITH HALF-PIXEL OFFSET
    # Using +0.5 ensures we sample the exact center of the pixel.
    # This prevents the edge pixels from sampling "outside" the bounds, fixing the 1-pixel seam gap.
    y, x = np.indices((face_size, face_size), dtype=np.float64)
    u_raw = 2.0 * (x + 0.5) / face_size - 1.0
    v_raw = 2.0 * (y + 0.5) / face_size - 1.0

    # 3. THE TANGENT WARP (Real World Scaling Fix)
    # This compresses the center and expands the edges of the texture in 2D.
    # When Three.js wraps this around the 3D sphere, the stretch is perfectly reversed, resulting in true scale.
    warp_factor = np.pi / 4.0
    u = np.tan(u_raw * warp_factor)
    v = np.tan(v_raw * warp_factor)

    for name in names:
        # 4. GET BASE CUBE VECTORS
        if name == 'px':   X, Y, Z = np.ones_like(u), -v, -u
        elif name == 'nx': X, Y, Z = -np.ones_like(u), -v, u
        elif name == 'py': X, Y, Z = u, np.ones_like(u), v
        elif name == 'ny': X, Y, Z = u, -np.ones_like(u), -v
        elif name == 'pz': X, Y, Z = u, -v, np.ones_like(u)
        elif name == 'nz': X, Y, Z = -u, -v, -np.ones_like(u)

        # 5. APPLY THREE.JS SPHERIFIED MATH
        x2, y2, z2 = X**2, Y**2, Z**2
        sx = X * np.sqrt(1.0 - (y2 / 2.0) - (z2 / 2.0) + (y2 * z2 / 3.0))
        sy = Y * np.sqrt(1.0 - (z2 / 2.0) - (x2 / 2.0) + (z2 * x2 / 3.0))
        sz = Z * np.sqrt(1.0 - (x2 / 2.0) - (y2 / 2.0) + (x2 * y2 / 3.0))

        vectors = np.stack([sx, sy, sz], axis=-1)

        # 6. CONVERT TO SPHERICAL (Lat / Lon)
        # Add a tiny epsilon (1e-10) to r to prevent division by zero exactly at the poles
        r = np.linalg.norm(vectors, axis=-1)
        r = np.maximum(r, 1e-10)
        
        lon = np.arctan2(vectors[..., 0], vectors[..., 2])
        lat = np.arcsin(np.clip(vectors[..., 1] / r, -1.0, 1.0))

        # 7. MAP BACK TO EQUIRECTANGULAR PIXELS
        pixel_x = (lon / np.pi + 1.0) * 0.5 * w
        pixel_y = (0.5 - lat / np.pi) * h

        # --- SEAM FIX: MANUAL WRAP & CLAMP ---
        # OpenCV's BORDER_WRAP applies to both X and Y. Wrapping Y causes the North Pole
        # to sample the South Pole, creating massive black lines/glitches.
        # Instead, we manually wrap longitude (X) around the Earth:
        pixel_x = np.mod(pixel_x, w)
        # And safely clamp latitude (Y) to the top/bottom pixels:
        pixel_y = np.clip(pixel_y, 0, h - 1)

        # 8. REMAP WITH BORDER_REPLICATE
        # Replicate ensures that if precision math pushes a value to 1024.0001,
        # it duplicates the very last valid edge pixel, creating a seamless weld.
        face_img = cv2.remap(
            img, 
            pixel_x.astype(np.float32), 
            pixel_y.astype(np.float32), 
            cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.png")
        cv2.imwrite(path, face_img)
        face_paths.append(path)

    return face_paths

def forge_ktx2(face_paths):
    """Compresses the PNG faces into KTX2 using your specific toktx flags."""
    print(f"✨ Starting Forge: Compressing to KTX2...")
    ktx2_output_paths = []
    
    for img_path in face_paths:
        face_name = os.path.basename(img_path).split('.')[0]
        out_path = os.path.join(KTX_OUT_DIR, f"{face_name}.ktx2")
        
        # MATCHING YOUR TOOL'S SPEC:
        # 1. Use --t2 for KTX2 output
        # 2. Use --2d as a standalone flag
        # 3. Use --genmipmap to create the chain
        # 4. Use --encode uastc for the compression format
        cmd = [
            TOKTX_PATH,
            "--t2",
            "--2d",
            "--genmipmap",
            "--encode", "uastc",
            "--uastc_quality", "2",
            "--",              # Mandatory separator before file paths
            out_path,          # <outfile>
            img_path           # <infile>
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True)
            print(f"  ✅ Forged: {face_name}.ktx2")
            ktx2_output_paths.append(os.path.abspath(out_path))
            
            if os.path.exists(img_path):
                os.remove(img_path) 
        except subprocess.CalledProcessError as e:
            # This will now print the EXACT reason if it fails again
            print(f"  ❌ Failed to forge {face_name}!")
            print(f"     Error: {e.stderr.decode().strip()}")

    return ktx2_output_paths

# --- NO VALIDATION (SAVE EVERYTHING) ---
def fetch_tile_stack(r, c, session, dates):
    stack = []
    used_paths = []

    for date_str in dates:
        if len(stack) >= STACK_DEPTH: break
        
        for layer in LAYERS:
            # URL for EPSG:4326 (Geographic)
            img_url = f"https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/{layer}/default/{date_str}/250m/{ZOOM}/{r}/{c}.jpg"
            try:
                res = session.get(img_url, timeout=5)
                if res.status_code == 200:
                    # 🛑 VALIDATION REMOVED 🛑
                    # We accept EVERYTHING unless it is literally 0 bytes
                    if len(res.content) > 100:
                        try:
                            img = Image.open(io.BytesIO(res.content)).convert("RGB")
                            data = np.array(img)
                            stack.append(data)
                            used_paths.append(img_url)
                            break # Found a frame, next date
                        except:
                            pass
                elif res.status_code == 404:
                    # This happens if the tile is "off the map" or date is wrong
                    continue
            except: continue
    if not stack:
        print(f"[DEBUG] Stack empty at tile ({r},{c})")
    if stack:
        # --- PHASE 0: PRE-PROCESSING & CLEANING ---
        stack_np = np.stack(stack, axis=0).astype(np.float32)  # (Frames, H, W, RGB)

        # Mark missing data (0,0,0) as NaN
        mask_missing = np.all(stack_np == 0, axis=-1)
        stack_np[mask_missing] = np.nan

        # Replace NaNs with per-pixel median (prevents black specks)
        pixel_median = np.nanmedian(stack_np, axis=0)
        stack_np = np.nan_to_num(stack_np, nan=pixel_median)

        # --- PHASE 1: SMART CLASSIFIER BOUNCER ---
        f_mean = np.mean(stack_np, axis=(1, 2, 3))
        f_var  = np.var(stack_np, axis=(1, 2, 3))

        frame_means = np.mean(stack_np, axis=(1, 2))  # (Frames, RGB)
        R, G, B = frame_means[:,0], frame_means[:,1], frame_means[:,2]

        # Cloud detection: bright, low-variance, low-saturation frames
        is_cloud = (
            (f_mean > 190) &
            (f_var < 20) &
            (np.abs(R-G) < 8) &
            (np.abs(R-B) < 8)
        )

        # Thin cloud / mist detection
        sat = np.ptp(frame_means, axis=1)
        is_mist = (f_mean > 160) & (sat < 15)

        def allow_ice(r, c):
            # Polar rows always allowed (Arctic/Antarctic bands)
            if r in [0, 1, 2, 14, 15]:  # expand polar rows
                return True

            HIGH_ALTITUDE_TILES = {
                # Himalayas / Tibet
                (4, 10), (4, 11), (4, 12), (5, 13), (5, 14),
                # Andes
                (6, 7), (6, 8), (7, 7), (7, 8),
                # Rockies
                (8, 2), (8, 3),
                # Alps / Caucasus
                (9, 11), (9, 12),
                # Atlas
                (9, 9),
                # Ethiopian Highlands
                (9, 15),
                # Papua Highlands
                (7, 16), (7, 17),
                # Southern Alps (NZ)
                (12, 19),
                # Scandinavian Mountains
                (10, 6),
                # Altai / Tien Shan / Pamirs
                (6, 12), (6, 13), (7, 13),
                # NEW: Northern Canada / Greenland tiles
                (2, 3), (2, 4), (2, 5), (3, 3), (3, 4),
                # NEW: Siberia / Russia Arctic tiles
                (1, 10), (1, 11), (2, 10), (2, 11), (3, 11),
            }

            return (r, c) in HIGH_ALTITUDE_TILES

        if allow_ice(r, c):
            # Temporal stability: low std across frames
            persistence = np.nanstd(stack_np, axis=(1,2,3)) < 2
            # Spectral ratio: blue dominance
            blue_dom = (B > R + 20) & (B > G + 15)
            # Texture: edge detection on median frame
            median_frame = np.nanmedian(stack_np, axis=0).astype(np.uint8)
            edges = cv2.Canny(median_frame, 50, 150)
            edge_strength = np.mean(edges) > 10

            is_ice = persistence & blue_dom & edge_strength
            # Clouds = bright but not ice
            is_cloud = is_cloud & ~is_ice
        else:
            # In tropics, persistent bright = cloud, not ice
            is_ice = np.zeros_like(is_cloud, dtype=bool)

        # Keep only frames not flagged as cloud/mist
        valid_indices = np.where(~(is_cloud | is_mist))[0]
        if len(valid_indices) < int(len(stack_np) * 0.85):
            valid_indices = np.arange(len(stack_np))

        active_stack = stack_np[valid_indices]

        # --- PHASE 2: PIXEL SCORING ---
        p_br = np.mean(active_stack, axis=3)              # Brightness
        p_sat = np.ptp(active_stack, axis=3)              # Saturation
        p_blue_dom = active_stack[..., 2] - active_stack[..., 0]  # Blue dominance

        # Score formula: lower is better
        p_score = p_br - (p_sat * 2.0) - (p_blue_dom * 1.5)

        # --- PHASE 3: SELECTION ---
        num_frames = active_stack.shape[0]
        K = min(max(8, int(num_frames * 0.25)), 20)

        best_indices = np.argpartition(p_score, K, axis=0)[:K]
        best_pixels = np.take_along_axis(
            active_stack,
            best_indices[..., None],
            axis=0
        )

        # --- PHASE 4: FINAL COMPOSITE ---
        # Use 10th percentile to suppress residual thin clouds
        composite = np.percentile(best_pixels, 10, axis=0)

        # Clip and cast
        composite = np.clip(composite, 0, 255).astype(np.uint8)
        return (r, c, composite, used_paths)

def earth_mosaic():
    """Handles downloading, assembly, and returns both JPG and KTX2 paths."""
    # 1. Setup paths using the updated Smart Cache Control
    # This now returns a tuple: (jpg_path, list_of_ktx_paths)
    jpg_path, ktx_paths = get_or_refresh_mosaic_path(TILES_DIR, FINAL_DIR, KTX_OUT_DIR, max_age_days=90)
    
    # 2. THE CACHE CHECK
    # If the JPG is fresh AND we already have the KTX2 files, we can exit immediately
    if os.path.exists(jpg_path):
        file_age = (time.time() - os.path.getmtime(jpg_path)) / (24 * 3600)
        if file_age < 90 and len(ktx_paths) == 6: 
            print(f"⏩ Assets are fresh ({int(file_age)} days old). Skipping all work.")
            return jpg_path, ktx_paths
        elif file_age < 90 and len(ktx_paths) < 6:
            print(f"📦 JPG is fresh, but KTX2 faces are missing. Proceeding to Split/Forge...")
            # We don't return here; we let the function continue to assembly/splitting
        else:
            print(f"[!] Assets expired ({int(file_age)} days). Re-generating...")

    # 3. THE TILES RESUME LOGIC (Only runs if JPG/KTX are missing or old)
    total_needed = len(TILE_COORDS)
    existing_tiles = [f for f in os.listdir(TILES_DIR) if f.endswith('.png')]

    if len(existing_tiles) < total_needed:
        print(f"🧩 Missing tiles ({len(existing_tiles)}/{total_needed}). Starting download...")
        session = get_robust_session()
        dates = [(datetime.datetime.now() - datetime.timedelta(days=d)).strftime("%Y-%m-%d")
                 for d in range(START_DATE_OFFSET, START_DATE_OFFSET + STACK_DEPTH)]

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            for r, c, data, paths in tqdm(
                executor.map(lambda coords: fetch_tile_stack(*coords, session, dates), TILE_COORDS),
                total=total_needed, unit="tile"
            ):
                if data is not None:
                    out_path = os.path.join(TILES_DIR, f"tile_{r}_{c}.png")
                    Image.fromarray(data).save(out_path)

    # 4. ASSEMBLY
    print("🎨 Assembling Tiles into Mosaic...")
    full_w = 20 * TILE_SIZE
    full_h = 10 * TILE_SIZE
    u8 = np.zeros((full_h, full_w, 3), dtype=np.uint8)
    
    for r, c in TILE_COORDS:
        tile_path = os.path.join(TILES_DIR, f"tile_{r}_{c}.png")
        if os.path.exists(tile_path):
            tile_img = Image.open(tile_path).convert("RGB")
            y, x = r * TILE_SIZE, c * TILE_SIZE
            u8[y:y+TILE_SIZE, x:x+TILE_SIZE] = np.array(tile_img)

    # 5. POST-PROCESSING (Scaling to exactly 4096)
    print("🪄 Finalizing 4K Scale and Encoding...")
    rgb_4k = cv2.resize(u8, (4096, 2048), interpolation=cv2.INTER_CUBIC)
    
    del u8
    gc.collect()

    bgr_4k = cv2.cvtColor(rgb_4k, cv2.COLOR_RGB2BGR)
    is_success, buffer = cv2.imencode(".jpg", bgr_4k, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    if is_success:
        with open(jpg_path, 'wb') as f:
            f.write(buffer)
        print(f"🌍 WEBGL-READY JPEG SAVED: {jpg_path}")
        
        # After saving the JPG, we MUST generate the KTX2s to get the paths
        temp_png_faces = split_to_cubemap(jpg_path)
        ktx_paths = forge_ktx2(temp_png_faces)
        
        return jpg_path, ktx_paths
    else:
        raise Exception("Failed to encode JPEG buffer")

#earth_mosaic()