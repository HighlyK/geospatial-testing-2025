import numpy as np
import tifffile
from PIL import Image
from pathlib import Path
import subprocess
import scipy.ndimage as ndimage
import time

# --- CONFIG ---
MASTER_TIF = r"C:\Users\ASUS\Downloads\gebco_08_rev_elev_21600x10800.tif"
FACE_SIZE = 2048 # Matches your JS segments/texture resolution
TOKTX_PATH = r"KTX-Software\bin\toktx.exe"

DIRS = {"output": Path(r"gpu_compressed_nrt_data_vs\height_mosaic")}
DIRS["output"].mkdir(parents=True, exist_ok=True)

def get_face_vectors(face_name, size):
    """
    Generates 3D unit vectors matching Three.js SubjectFactory Spherification.
    Upgraded with Half-Pixel Offset and Tangent Warp for true equal-area scale.
    """
    grid_y, grid_x = np.indices((size, size), dtype=np.float64)
    
    # 1. HALF-PIXEL OFFSET
    # Samples the exact mathematical center of the pixel, preventing edge-bleed.
    u_raw = 2.0 * (grid_x + 0.5) / size - 1.0
    v_raw = 2.0 * (grid_y + 0.5) / size - 1.0 

    # 2. TANGENT WARP (Real World Scaling)
    warp_factor = np.pi / 4.0
    u = np.tan(u_raw * warp_factor)
    v = np.tan(v_raw * warp_factor)

    # 3. Base Cube Face Vectors
    if face_name == 'px':   X, Y, Z = np.ones_like(u), -v, -u
    elif face_name == 'nx': X, Y, Z = -np.ones_like(u), -v, u
    elif face_name == 'py': X, Y, Z = u, np.ones_like(u), v
    elif face_name == 'ny': X, Y, Z = u, -np.ones_like(u), -v
    elif face_name == 'pz': X, Y, Z = u, -v, np.ones_like(u)
    elif face_name == 'nz': X, Y, Z = -u, -v, -np.ones_like(u)

    # 4. APPLY THE BULGE (Spherification)
    x2, y2, z2 = X**2, Y**2, Z**2
    sx = X * np.sqrt(1.0 - (y2 / 2.0) - (z2 / 2.0) + (y2 * z2 / 3.0))
    sy = Y * np.sqrt(1.0 - (z2 / 2.0) - (x2 / 2.0) + (z2 * x2 / 3.0))
    sz = Z * np.sqrt(1.0 - (x2 / 2.0) - (y2 / 2.0) + (x2 * y2 / 3.0))

    return np.stack([sx, sy, sz], axis=-1)
    
def sample_height_face(master_array, face_name, global_min, global_range):
    h, w = master_array.shape
    vectors = get_face_vectors(face_name, FACE_SIZE)
    
    # 1. Convert vectors to Lat/Lon (Spherical)
    r = np.linalg.norm(vectors, axis=-1)
    r = np.maximum(r, 1e-10) # Prevent divide-by-zero at exact poles
    
    lon = np.arctan2(vectors[..., 0], vectors[..., 2])
    lat = np.arcsin(np.clip(vectors[..., 1] / r, -1.0, 1.0))
    
    # 2. Map to Pixel Coordinates
    px = (lon / np.pi + 1.0) * 0.5 * w
    py = (0.5 - lat / np.pi) * h
    
    # 3. SEAM FIX: MANUAL WRAP AND CLAMP
    # SciPy's mode='wrap' wraps BOTH axes. We only want longitude (X) to wrap.
    # Latitude (Y) MUST be clamped to prevent the North Pole sampling the South Pole.
    px = np.mod(px, w)
    py = np.clip(py, 0, h - 1)
    
    print(f"  [MATH] Interpolating {face_name}...")
    
    # 4. SAMPLE
    # mode='nearest' acts like BORDER_REPLICATE, ensuring our clamped edge pixels bleed perfectly
    sampled = ndimage.map_coordinates(master_array, [py, px], order=1, mode='nearest')

    # 5. GLOBAL NORMALIZATION (Prevents seam tears in 3D)
    normalized = (sampled - global_min) / global_range * 65535
    return np.clip(normalized, 0, 65535).astype(np.uint16)

def forge_ktx2(png_path):
    out_ktx2 = png_path.with_suffix(".ktx2")
    cmd = [
        TOKTX_PATH,
        "--t2", "--2d", "--genmipmap",
        "--encode", "uastc",
        "--uastc_quality", "2",
        "--assign_oetf", "linear", 
        "--", str(out_ktx2), str(png_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def main():
    start_time = time.time()
    
    print(f"[SYSTEM] Loading Master Elevation TIFF...")
    master_array = tifffile.imread(MASTER_TIF)
    
    print("[SYSTEM] Calculating Global Elevation Range...")
    global_min = np.min(master_array)
    global_max = np.max(master_array)
    global_range = global_max - global_min
    print(f"  [INFO] World Range: {global_min} to {global_max}")

    face_names = ['px', 'nx', 'py', 'ny', 'pz', 'nz']
    
    for name in face_names:
        print(f"\n>> Processing: {name}")
        
        face_data = sample_height_face(master_array, name, global_min, global_range)
        
        png_path = DIRS["output"] / f"height_face_{name}.png"
        Image.fromarray(face_data, mode='I;16').save(png_path)
        
        print(f"  [FORGE] Compressing to KTX2...")
        forge_ktx2(png_path)
        
        png_path.unlink()

    print(f"\n[SUCCESS] 6 faces forged in {round(time.time() - start_time, 1)}s")

if __name__ == "__main__":
    main()