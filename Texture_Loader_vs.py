from OpenGL.GL import *
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import math
import numpy as np
import struct

def load_texture(path, texture):
    glBindTexture(GL_TEXTURE_2D, texture)
    
    # 1. STOP USING RGBA. JPEG has no Alpha. Use RGB.
    # 2. INTERNAL FORMAT: GL_RGB10_A2 (0x8059) or GL_R11F_G11F_B10F (0x8C3A)
    # These formats provide way more "room" for the gradients to stay smooth.
    GL_RGB10_A2 = 0x8059 
    
    image = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM).convert("RGB")
    img_data = image.tobytes()

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB10_A2, image.width, image.height, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, img_data)
    
    glGenerateMipmap(GL_TEXTURE_2D)
    
    # Ensure high quality filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return texture

def load_raw_cubemap(file_path, face_size=2048):
    """
    Loads a vertical strip RAW (6 faces) into an OpenGL Cubemap Texture.
    Expects RGBA8 (4 bytes per pixel).
    """
    # 1. Generate and Bind
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex_id)

    # 2. Read the RAW binary data directly into a NumPy buffer
    try:
        # We use fromfile for speed; it maps the binary straight to a 1D array
        raw_data = np.fromfile(file_path, dtype=np.uint8)
    except Exception as e:
        print(f"❌ Could not read RAW file: {e}")
        return None

    # Calculate expected size: (FaceSize * FaceSize) * 6 faces * 4 channels (RGBA)
    expected_size = face_size * face_size * 6 * 4
    if raw_data.size != expected_size:
        print(f"⚠️ RAW Size Mismatch! Expected {expected_size}, got {raw_data.size}")
        # If it's just slightly off, we trim or pad, but usually this means a config error
        raw_data = np.resize(raw_data, expected_size)

    # 3. Upload each face
    # Order in your Python mosaic: PX, NX, PY, NY, PZ, NZ
    face_stride = face_size * face_size * 4
    
    for i in range(6):
        start = i * face_stride
        end = start + face_stride
        face_pixels = raw_data[start:end]

        # Target: GL_TEXTURE_CUBE_MAP_POSITIVE_X (0x8515) 
        # Adding 'i' works because the constants are sequential in OpenGL
        glTexImage2D(
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
            0, GL_RGBA8, face_size, face_size, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, face_pixels
        )

    # 4. Parameters to kill the "Rings" and "Seams"
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    # CLAMP_TO_EDGE is mandatory to prevent lines at the cube corners
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    # 5. Generate Mipmaps (Essential for smoothing distant views)
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP)

    print(f"🌍 Cubemap Loaded: {file_path} ({face_size}px faces)")
    return tex_id

def load_bin_texture(path, res=256):
    """
    Minimalist RAW loader for LUTs.
    - path: path to the .bin or .raw file
    - res: The resolution (usually 256 for high-end LUTs or 64 for basic ones)
    """
    if not os.path.exists(path):
        print(f"❌ LUT missing: {path}")
        return None

    # 1. Read raw bytes
    with open(path, "rb") as f:
        raw_data = f.read()

    # 2. Convert to NumPy for GPU upload
    # We don't flipud() here because LUTs are usually addressed by 
    # raw coordinates, but if your colors look 'inverted', add np.flipud back.
    level_data = np.frombuffer(raw_data, dtype=np.uint8)
    
    # Ensure it's reshaped to the expected dimensions (H, W, RGBA)
    try:
        level_data = level_data.reshape((res, res, 4))
    except ValueError:
        print(f"❌ Data size mismatch for {res}x{res} RGBA LUT.")
        return None

    # 3. Standard OpenGL Texture Setup
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)

    # LUTs MUST use GL_NEAREST or GL_LINEAR with no mipmaps
    # GL_NEAREST ensures no "color bleeding" between LUT entries
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    # 4. Upload
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA8,
        res, res, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, level_data
    )

    print(f"🎨 LUT Loaded: {res}x{res}")
    return tex_id

def load_ktx_tex(path):
    """
    High-Precision VRAM dump for 4096x4096x4 RGBA16 KTX.
    Matches 16-bit PNG smoothing pipeline to eliminate rings.
    """
    if not os.path.exists(path):
        print(f"❌ ERROR: {path} not found.")
        return None

    # 1. Open and Parse KTX 1.1 Header
    with open(path, 'rb') as f:
        # Skip the 12-byte identifier + 48 bytes of header constants = 60 bytes
        f.seek(60) 
        
        # The next 4 bytes tell us the size of the 'Key/Value' metadata area
        key_value_size_raw = f.read(4)
        key_value_size = struct.unpack('I', key_value_size_raw)[0]
        
        # The next 4 bytes are the 'Image Data Size' (skip them)
        # The pixel data starts immediately after the metadata
        data_start_offset = 64 + key_value_size + 4
        f.seek(data_start_offset)
        
        pixel_data = f.read()

    # 2. Setup OpenGL Texture
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    
    # 0x805B = GL_RGBA16 (Internal storage in VRAM)
    # 0x1403 = GL_UNSIGNED_SHORT (16-bit data coming from Python)
    GL_RGBA16 = 0x805B
    GL_UNSIGNED_SHORT = 0x1403
    
    # RTX 2050 optimization: 16-bit RGBA is 8 bytes per pixel (always 4-byte aligned)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4)

    print(f"🚀 Uploading 16-bit texture to VRAM ({len(pixel_data)} bytes)...")
    
    glTexImage2D(
        GL_TEXTURE_2D, 
        0, 
        GL_RGBA16,         # Internal format: 16-bit
        4096, 4096, 0, 
        GL_RGBA, 
        GL_UNSIGNED_SHORT, # Data type: 16-bit Short
        pixel_data
    )

    # 3. High-Quality Filtering
    # Since we have 65535 levels of color, GL_LINEAR will now be perfectly smooth.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    print(f"🌍 16-bit KTX Loaded Successfully: {path}")
    
    del pixel_data 
    return tex_id