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
import scipy.stats as sp_stats
import io
from NIGHTLIGHTS_AI_TRAINED import WorldNavigator
import gc
navigator = WorldNavigator()
import subprocess
from scipy.ndimage import gaussian_filter

def get_or_refresh_mosaic_path(tiles_dir, final_dir, ktx_dir, max_age_days=90):
    """
    Checks for fresh Mosaic and existing KTX2 files to provide paths to the backend.
    """
    target_file = "NIGHTLIGHTS_MOSAIC.jpg"
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

def split_to_cubemap(image_path, out_dir="KTX_OUT_DIR"):
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

# --- CONFIG ---
# 1. We start 7 days ago because "Today" often has no data yet on GIBS
START_DATE_OFFSET = 1 # 120 later
STACK_DEPTH = 30 # crank to 120 later
MAX_THREADS = 4

dates = [(datetime.datetime.now() - datetime.timedelta(days=d)).strftime("%Y-%m-%d")
             for d in range(START_DATE_OFFSET, START_DATE_OFFSET + STACK_DEPTH)]
LAYERS = ["VIIRS_SNPP_DayNightBand_At_Sensor_Radiance"] # night version
ZOOM = 4
TILE_SIZE = 512 # change to 512 if error happen
# A smaller test area (Mainland US/Atlantic) to see results fast
# Once this works, change it back to your full range
TILE_COORDS = [(r, c) for r in range(0, 9) for c in range(0, 20)]

# --- PATHS ---
BASE_DIR    = os.path.abspath(".")
TILES_DIR   = os.path.join(BASE_DIR, "night_tiles")
FINAL_DIR   = os.path.join(BASE_DIR, "gpu_compressed_nrt_data_vs", "night_mosaic")
# This must match what the functions below use
KTX_OUT_DIR = os.path.join(FINAL_DIR, "ktx2_night_faces") 
TOKTX_PATH  = os.path.join(BASE_DIR, "KTX-Software", "bin", "toktx.exe")


# --- NETWORK ---
def get_robust_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    return session

def fetch_tile_stack(r, c, session, dates):
    """
    NEBULA-CLASS SOVEREIGN ENGINE: 
    Multi-spectral temporal processing with Biome-Specific Execution.
    """
    stack = []
    used_paths = []
    
    # ====================================================================
    # --- LAYER 1: DATA ACQUISITION & FRAME REJECTION ---
    # ====================================================================
    for date_str in dates:
        if len(stack) >= STACK_DEPTH: break
        
        img_url = f"https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/{LAYERS[0]}/default/{date_str}/500m/{ZOOM}/{r}/{c}.png"
        try:
            res = session.get(img_url, timeout=5)
            if res.status_code == 200 and len(res.content) > 500:
                img = Image.open(io.BytesIO(res.content)).convert("RGBA")
                data = np.array(img)
                
                # Nuke NASA blue "No Data" placeholders
                if np.mean(data[...,2]) > 130 and np.mean(data[...,0]) < 45: continue
                
                radiance = data[..., 0].astype(np.float32)
                h, w = radiance.shape
                
                edge_energy = cv2.Laplacian(radiance, cv2.CV_32F).var()

                # B. FFT Energy Ratio (Strict Glow Detector)
                f = np.fft.fft2(radiance)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
                cy, cx = h // 2, w // 2
                
                # We shrink the low-freq box to be more sensitive to blur
                low_freq_sum = np.sum(magnitude_spectrum[cy-8:cy+8, cx-8:cx+8])
                total_sum = np.sum(magnitude_spectrum)
                hf_ratio = (total_sum - low_freq_sum) / (total_sum + 1e-6)

                # --- NEW STRICT GATES ---
                # 1. Reject if too blurry (hf_ratio < 0.15 instead of 0.12)
                # 2. Reject if high radiance but low edge energy (Classic Aurora/Cloud wash)
                if hf_ratio < 0.15 or (np.max(radiance) > 14.0 and edge_energy < 150):
                    print(f"[DEBUG] L0 Gate | Rejected blurry/glow tile. HF: {hf_ratio:.4f}")
                    continue

                stack.append(radiance * 1e9)
        except: continue

    # [Assume 'stack' and 'used_paths' are generated here from valid, sharp images]
    if not stack:
        #print(f"[DEBUG] Stack empty at tile ({r},{c})")
        return (r, c, None, ["Error: Stack Depleted"])

    h, w = stack[0].shape[:2]
    #print(f"[DEBUG] Tile ({r},{c}) size: {h}x{w}, stack length: {len(stack)}")

    # --- LAYER 1: GEOGRAPHIC CONTEXT ---
    geo_logic = navigator.get_tile_logic(r, c, tile_size=512)
    #print(f"[DEBUG] Geo logic keys: {list(geo_logic.keys())}")

    land_mask = (geo_logic['land'].astype(np.float32) > 0.5).astype(np.uint8)
    ocean_mask = geo_logic['ocean'].astype(bool)
    pop_raw = geo_logic['pop'].astype(np.float32)
    biomes = geo_logic['biomes']
    nasa_ref = geo_logic['nasa_reference']

    #print(f"[DEBUG] Land mask sum: {np.sum(land_mask)}, Ocean mask sum: {np.sum(ocean_mask)}")

    nasa_norm = np.clip(nasa_ref.astype(np.float32) / 255.0, 0, 1)
    pop_norm = np.clip(pop_raw / (np.max(pop_raw) + 1e-6), 0, 1)

    truth_mask = (0.85 * (nasa_norm > 0.15)) + (0.15 * (pop_norm > 0.35))
    halo_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    truth_shield = cv2.dilate(truth_mask.astype(np.uint8), halo_kernel, iterations=1).astype(bool)
    #print(f"[DEBUG] Truth shield coverage: {np.sum(truth_shield)} pixels")

    # --- LAYER 2: LOG TRANSFORM ---
    raw_cube = np.stack(stack, axis=0).astype(np.float32)
    log_cube = np.log1p(np.maximum(raw_cube, 0.0))
    #print(f"[DEBUG] Log cube shape: {log_cube.shape}, min: {log_cube.min()}, max: {log_cube.max()}")

    persistence_map = np.sum(log_cube > 0.15, axis=0) / len(stack)
    persistence_mask = (persistence_map > 0.45).astype(np.float32)
    #print(f"[DEBUG] Persistence mask coverage: {np.sum(persistence_mask)}")

    base_composite = np.nanmedian(log_cube, axis=0)
    base_composite = np.nan_to_num(base_composite, nan=0.0)
    #print(f"[DEBUG] Base composite stats: min={base_composite.min()}, max={base_composite.max()}")

    # --- LAYER 3: MACRO WIPE ---
    base_composite[(biomes == 3) & ~truth_shield] = 0
    black_list = [(0,13), (0,14), (0,15), (0,16), (0,17), (1,2), (1,4), (1,5), (1,17), (1,18), (2,2), (1,3), (0,6), (0,7), (1, 7), (0,8), (0,9), (6, 12)]
    base_composite[ocean_mask] = 0
    if (r, c) in black_list:
        #print(f"[DEBUG] Tile ({r},{c}) blacklisted")
        base_composite.fill(0)
    # ====================================================================
    # --- LAYER 3.5: THE TUNDRA PHYSICS SNIPER (Biome 2 Special) ---
    # ====================================================================
    # Tundra (Biome 2) often has 'flat' albedo noise. 
    # Real lights have high local entropy; noise has low local entropy.
    
    if (biomes == 2).any():
        tundra_mask = (biomes == 2)
        tundra_pixels = base_composite[tundra_mask & (base_composite > 0)]
        
        if tundra_pixels.size > 500: # Ensure we have enough data to be statistically valid
            # 1. Tundra-Specific Histogram
            t_hist, t_edges = np.histogram(tundra_pixels, bins=128, range=(20.0, 30.0))
            
            # Find the Snow/Albedo Peak
            t_peak_idx = np.argmax(t_hist)
            t_noise_peak = t_edges[t_peak_idx]
            
            # Calculate Tundra-specific variance (The 'thickness' of the snow haze)
            t_mean = np.average(t_edges[:-1], weights=t_hist)
            t_variance = np.average((t_edges[:-1] - t_mean)**2, weights=t_hist)
            t_width = np.sqrt(t_variance)

            # 2. THE STRICTNESS MULTIPLIER
            # Globally we used 0.5 * width. For Tundra, we use 1.5 * width + 0.2 offset.
            # This makes it "strict like a lot."
            tundra_surgical_floor = t_noise_peak + (1.5 * t_width) + 0.2
            
            # 3. Execution (Pixel Wipe)
            # We preserve the Truth Shield even in the Tundra
            tundra_wipe_mask = (tundra_mask) & (base_composite < tundra_surgical_floor) & (~truth_shield)
            pixels_killed = np.sum(tundra_wipe_mask)
            base_composite[tundra_wipe_mask] = 0
            
            #print(f"[DEBUG] L3.5 TUNDRA | Peak: {t_noise_peak:.2f} | Width: {t_width:.4f}")
            #print(f"[DEBUG] L3.5 TUNDRA | STRICT Floor: {tundra_surgical_floor:.2f} | Killed: {pixels_killed} px")
        else:
            #print(f"[DEBUG] L3.5 TUNDRA | Not enough tundra pixels for histogram.")
            pass
    # ====================================================================
    # --- LAYER 4: HISTOGRAM PEAK-ANALYSIS (No Purge, Pixel Wipe) ---
    # ====================================================================
    
    bins = np.linspace(20, 30, 101) 
    hist, bin_edges = np.histogram(base_composite, bins=bins)
    
    # --- SAFETY CHECK: If the histogram is empty, we skip the peak math ---
    if np.sum(hist) > 0:
        # Find the peak of the noise
        peak_idx = np.argmax(hist)
        noise_peak_value = bin_edges[peak_idx]
        
        # Calculate the "width" safely
        mean_val = np.average(bin_edges[:-1], weights=hist)
        variance = np.average((bin_edges[:-1] - mean_val)**2, weights=hist)
        noise_width = np.sqrt(variance)
        
        # Set the surgical floor
        surgical_floor = noise_peak_value + (0.5 * noise_width)
        #print(f"[DEBUG] L4 Hist | Peak: {noise_peak_value:.2f} | Width: {noise_width:.4f}")
    else:
        # FALLBACK: If no noise hump is found, use a safe default or 0
        surgical_floor = 0.0
        #print(f"[DEBUG] L4 Hist | No noise hump detected. Floor set to 0.0")
    
    # Ensure we don't accidentally wipe out bright cities by capping the floor
    #surgical_floor = min(surgical_floor, 24.5) 
    
    floor_mask = (base_composite < surgical_floor) & (base_composite > 0)
    base_composite[floor_mask] = 0
    #print(f"[DEBUG] L4 Baseline | Peak-Hunter Floor ({surgical_floor:.2f}) wiped {np.sum(floor_mask)} px.")

    # 3. High-Frequency Edge Analysis (Same robust math, improved sensitivity)
    # 3. Compute High-Dimensional Arrays
    laplacian = cv2.Laplacian(base_composite, cv2.CV_32F)
    sobel_x = cv2.Sobel(base_composite, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(base_composite, cv2.CV_32F, 0, 1, ksize=3)
    
    abs_lap = np.abs(laplacian)
    abs_sobel_x = np.abs(sobel_x)
    abs_sobel_y = np.abs(sobel_y)

    # ====================================================================
    # --- BIOME-AWARE STRICTNESS CONFIGURATION ---
    # ====================================================================
    is_tundra = (biomes == 2).any()
    
    if is_tundra:
        # 🥶 TUNDRA MODE: Extreme strictness for snow albedo/arctic haze
        grain_limit = 2.8    # Much lower = more aggressive grain killing
        stripe_sens = 2.2    # Lower ratio = kills even subtle vertical stripes
        fog_limit   = 0.15   # Higher = wipes more "flat" glowing snow blobs
        p_threshold = 99.5   # Protect only the top 0.5% (Tundra is mostly fake light)
        #print("[DEBUG] L4 | BIOME 2 DETECTED: Engaging High-Strictness Mode.")
    else:
        # 🌍 GLOBAL MODE: Standard settings
        grain_limit = 4.5 
        stripe_sens = 3.0
        fog_limit   = 0.05
        p_threshold = 99.0
        #print("[DEBUG] L4 | Standard Biome logic applied.")

    # 4. Safe Zone Protection (Biome-Adjusted)
    p_safe = np.percentile(base_composite, p_threshold)
    safe_mask = truth_shield.astype(bool) | (base_composite > p_safe)

    # ====================================================================
    # --- THE BIOME-AWARE EXECUTIONERS ---
    # ====================================================================
    
    # A: The Stripe Sniper (Tile 4,11 specialized)
    stripe_ratio = abs_sobel_x / (abs_sobel_y + 1e-6)
    stripe_mask = (stripe_ratio > stripe_sens) & (abs_sobel_x > 0.2) & (~safe_mask)
    
    # B: The Grain Sniper (Tile 0,6 specialized)
    # Kills sharp 'salt and pepper' noise
    grain_mask = (abs_lap > grain_limit) & (base_composite < p_safe) & (~safe_mask)

    # C: The Fog Sniper (Tile 0,8 specialized)
    # Kills the "Gray Soup" where edges are too soft to be real cities
    fog_mask = (abs_lap < fog_limit) & (base_composite > 0) & (~safe_mask)

    # --- FINAL BIOME-SPECIFIC HISTOGRAM WIPE ---
    # In Tundra, we run an extra histogram wipe just for the Biome 2 pixels
    if is_tundra:
        t_pixels = base_composite[biomes == 2]
        if t_pixels.size > 0:
            t_hist, t_edges = np.histogram(t_pixels, bins=50, range=(20, 30))
            t_peak = t_edges[np.argmax(t_hist)]
            # Wipe everything within 0.2 of the tundra-specific noise peak
            t_hump_mask = (biomes == 2) & (base_composite < (t_peak + 0.25)) & (~safe_mask)
            base_composite[t_hump_mask] = 0
            #print(f"[DEBUG] L4 | Tundra-Specific Hump Wipe: {np.sum(t_hump_mask)} pixels.")

    # Execute Global Masks
    total_wipe_mask = stripe_mask | grain_mask | fog_mask
    base_composite[total_wipe_mask] = 0

    # 6. Final Telemetry (Don't remove debug!)
    #print(f"[DEBUG] L4 Purge | Stripe-Pixels: {np.sum(stripe_mask)}")
    #print(f"[DEBUG] L4 Purge | Grain-Pixels: {np.sum(grain_mask)}")
    #print(f"[DEBUG] L4 Purge | Fog-Pixels: {np.sum(fog_mask)}")

    #SAVED
    # ====================================================================
    # --- LAYER 5: RANGE RESTORATION & DYNAMIC STRETCH ---
    # ====================================================================
    c_min, c_max = base_composite.min(), base_composite.max()
    #print(f"[DEBUG] Raw Composite Range: {c_min:.4f} to {c_max:.4f}")

    if c_max - c_min < 0.0001:
        #print(f"[DEBUG] Tile ({r},{c}) nuked: Zero variance / Dead signal.")
        return (r, c, np.zeros((h, w, 4), dtype=np.uint8), ["Purged: Zero Variance"])

    # 1. Stretch the log-cliff (e.g., 23.9-26.2) to (0.0-1.0)
    # This formula is the "Anti-Flattening" fix:
    norm = (base_composite - c_min) / (c_max - c_min + 1e-6)
    
    # 2. Aggressive Sigmoid to isolate high-intensity centers
    contrast = 1.0 / (1.0 + np.exp(-25.0 * (norm - 0.40)))
    final_u8 = (contrast * 255).astype(np.uint8)

    # 3. CLAHE to sharpen local road networks
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10,10))
    final_pop = clahe.apply(final_u8)
    #print(f"[DEBUG] Post-Stretch Stats: Min={final_pop.min()}, Max={final_pop.max()}")

    #SAVED DONT DELETE
    # ====================================================================
    # --- LAYER 5: LINEAR STRETCH & GAMMA CORRECTION ---
    # ====================================================================
    # Use percentiles to find the 'real' data floor, ignoring the mask's zeros
    data_pixels = base_composite[base_composite > 0]
    if data_pixels.size == 0:
        return (r, c, np.zeros((h, w, 4), dtype=np.uint8), ["Purged: Empty"])
    
    p5 = np.percentile(data_pixels, 5)
    p99 = np.percentile(data_pixels, 99)
    #print(f"[DEBUG] Stretch Range (P5-P99): {p5:.2f} to {p99:.2f}")

    # 1. Linear Stretch to 0-1 range
    norm = np.clip((base_composite - p5) / (p99 - p5 + 1e-6), 0, 1)
    
    # 2. Gamma Correction (Power Law) 
    # Gamma > 1.0 makes the darks darker and prevents 'The White Out'
    gamma = 2.2 
    gamma_corrected = np.power(norm, gamma)
    final_u8 = (gamma_corrected * 255).astype(np.uint8)

    # 3. Gentle CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    final_pop = clahe.apply(final_u8)
    #print(f"[DEBUG] Final Pop - Max: {final_pop.max()}, Mean: {final_pop.mean():.2f}")

    # ====================================================================
    # --- LAYER 6: THE TEXTURE-AWARE EXECUTIONER ---
    # ====================================================================
    mask_final = (final_pop > 20).astype(np.uint8)
    num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(mask_final)
    
    for i in range(1, num_labels):
        x, y, w_i, h_i, area = stats_cc[i]
        cluster_mask = (labels[y:y+h_i, x:x+w_i] == i)
        cluster_pixels = final_pop[y:y+h_i, x:x+w_i][cluster_mask]
        
        # 🛡️ SOVEREIGN SHIELD
        cluster_truth = truth_shield[y:y+h_i, x:x+w_i]
        truth_score = np.mean(cluster_truth[cluster_mask])

        # --- PHYSICS METRICS ---
        c_std = np.std(cluster_pixels)
        c_max = np.max(cluster_pixels)
        peakiness = c_max / (np.mean(cluster_pixels) + 1e-6)
        
        # Entropy (Information Density)
        hist = np.histogram(cluster_pixels, bins=10, range=(0, 255))[0]
        prob = hist / (np.sum(hist) + 1e-6)
        entropy = -np.sum(prob * np.log2(prob + 1e-6))

        kill = False
        reason = ""

        # --- THE EXECUTION RULES ---
        if truth_score > 0.10:
            pass # Trust the ground truth map
        else:
            # RULE 1: THE FLATNESS KILLER
            # If a huge area has 0 variance, it's a sensor error or albedo wall.
            if c_std < 1.5 and area > 100:
                kill, reason = True, "Solid Flat Wall"
            
            # RULE 2: BIOME SPECIFIC
            local_biomes = biomes[y:y+h_i, x:x+w_i][cluster_mask]
            if (local_biomes == 2).any(): # TUNDRA
                if entropy < 0.8 or peakiness < 1.4:
                    kill, reason = True, "Arctic Albedo"
            elif (local_biomes == 4).any(): # TROPICAL
                if peakiness < 1.3:
                    kill, reason = True, "Tropical Bloom"

        # --- DEBUG LOGGING ---
        status = "KILLED" if kill else "LIVE"
        if area > 50: # Only log significant clusters to keep console clean
            #print(f"  [C-{i}] Area: {area} | Peak: {peakiness:.2f} | Std: {c_std:.2f} | Ent: {entropy:.2f} | {status} {reason}")
            pass
        if kill:
            final_pop[y:y+h_i, x:x+w_i][cluster_mask] = 0

    #print(f"[DEBUG] Final Pixel Max: {np.max(final_pop)}")
    final_pop[final_pop < 50] = 0 # Higher snap threshold
    # Final redundant mask to ensure no background noise slipped through
    rgba_out = np.dstack([final_pop, final_pop, final_pop, np.full_like(final_pop, 255)])
    #print(f"[DEBUG] Tile ({r},{c}) completed | Output shape={rgba_out.shape}")
    return (r, c, rgba_out, used_paths)

def night_mosaic():
    """Handles downloading, assembly, and returns both JPG and KTX2 paths for Night."""
    # 1. Setup paths using the updated Smart Cache Control
    # This now returns a tuple: (jpg_path, list_of_ktx_paths)
    jpg_path, ktx_paths = get_or_refresh_mosaic_path(TILES_DIR, FINAL_DIR, KTX_OUT_DIR, max_age_days=90)
    
    # 2. THE CACHE CHECK
    # If the JPG is fresh AND we already have the KTX2 files, we can exit immediately
    if os.path.exists(jpg_path):
        file_age = (time.time() - os.path.getmtime(jpg_path)) / (24 * 3600)
        if file_age < 90 and len(ktx_paths) == 6: 
            print(f"⏩ Night Assets are fresh ({int(file_age)} days old). Skipping all work.")
            return jpg_path, ktx_paths
        elif file_age < 90 and len(ktx_paths) < 6:
            print(f"📦 Night JPG is fresh, but KTX2 faces are missing. Proceeding to Split/Forge...")
            # We don't return here; we let the function continue to assembly/splitting
        else:
            print(f"[!] Night Assets expired ({int(file_age)} days). Re-generating...")

    # 3. THE TILES RESUME LOGIC (Only runs if JPG/KTX are missing or old)
    total_needed = len(TILE_COORDS)
    existing_tiles = [f for f in os.listdir(TILES_DIR) if f.endswith('.png')]

    if len(existing_tiles) < total_needed:
        print(f"🧩 Missing night tiles ({len(existing_tiles)}/{total_needed}). Starting download...")
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
    print("🎨 Assembling Night Tiles into Mosaic...")
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
    print("🪄 Finalizing 4K Scale and Encoding for Night...")
    # NOTE: I kept this at 4096x4096 to match your earth_mosaic exactly. 
    # Change the second value to 3072 if your split_to_cubemap specifically needs the 4:3 ratio.
    rgb_4k = cv2.resize(u8, (4096, 2048), interpolation=cv2.INTER_CUBIC)
    
    del u8
    gc.collect()

    bgr_4k = cv2.cvtColor(rgb_4k, cv2.COLOR_RGB2BGR)
    is_success, buffer = cv2.imencode(".jpg", bgr_4k, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    if is_success:
        with open(jpg_path, 'wb') as f:
            f.write(buffer)
        print(f"🌙 WEBGL-READY NIGHT JPEG SAVED: {jpg_path}")
        
        # After saving the JPG, we MUST generate the KTX2s to get the paths
        temp_png_faces = split_to_cubemap(jpg_path)
        ktx_paths = forge_ktx2(temp_png_faces)
        
        return jpg_path, ktx_paths
    else:
        raise Exception("Failed to encode Night JPEG buffer")

#night_mosaic()