import numpy as np
import random
import struct
import itertools
from OpenGL.GL import *
import matplotlib.pyplot as plt
import math

def compute_buffer(data_length, stride, dtype=np.float32):
    # stride is bytes per element (Unity passes this), dtype controls storage type
    # Here we allocate an array shaped (data_length, stride_in_floats)
    floats_per_elem = stride // np.dtype(dtype).itemsize
    return np.zeros((data_length, floats_per_elem), dtype=dtype)

def set_data(buffer, data):
    buffer[:] = data

def noise_compute_set_buffer(kernel, buffer_name, buffer):
    # Return a binding record (pretend "SetBuffer")
    return {"kernel": kernel, "buffer_name": buffer_name, "buffer": buffer}

def CreateBuffer(data, stride_bytes, buffer_name, kernel, buffers_to_release):
    # Allocate
    buffer = compute_buffer(len(data), stride_bytes)
    # Track for release (Unity: buffersToRelease.Add(buffer))
    buffers_to_release.append(buffer)
    # Set data
    set_data(buffer, data)
    # Bind
    binding = noise_compute_set_buffer(kernel, buffer_name, buffer)
    return binding


def CreateWorleyPointsBuffer(numCellsPerAxis, bufferName, kernel=None, buffers_to_release=None):
    points = np.zeros((numCellsPerAxis ** 3, 3), dtype=np.float32)
    cellSize = 1.0 / numCellsPerAxis
    nx = ny = nz = numCellsPerAxis

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                randomoffset = np.random.rand(3)
                position = (np.array([x, y, z]) + randomoffset) * cellSize
                index = x + nx * (y + ny * z)
                points[index] = position

    # Optional: upload to a compute-like buffer
    if kernel is not None and buffers_to_release is not None:
        CreateBuffer(points, struct.calcsize("f") * 3, bufferName, kernel, buffers_to_release)

    return points


def min_component(vec):
    return min(vec)

def max_component(vec):
    return max(vec)

def worley(points, num_cells, sample_pos):
    # Normalize sample_pos to [0,1]^3 so we share the same domain as points
    # If sample_pos is already normalized, this reduces to identity
    boundsMin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    boundsMax = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    # If you want world-space bounds, pass them in and normalize here:
    # sample_pos = (sample_pos - worldMin) / (worldMax - worldMin)
    sp = np.array(sample_pos, dtype=np.float32)

    nx, ny, nz = map(int, num_cells)
    cell_ID = np.floor(sp * num_cells).astype(int)

    minsqrDst = 1e9
    offsets = list(itertools.product([-1, 0, 1], repeat=3))
    cell_size = 1.0 / num_cells  # per-axis scale for neighbor tiles
    cell_bounds = np.array([nx, ny, nz], dtype=np.int32)

    for off in offsets:
        adjID = cell_ID + np.array(off, dtype=int)

        # Border check per axis
        out_of_bounds = (
            adjID[0] < 0 or adjID[0] >= nx or
            adjID[1] < 0 or adjID[1] >= ny or
            adjID[2] < 0 or adjID[2] >= nz
        )

        wrappedID = (adjID + cell_bounds) % cell_bounds
        adjCellIndex = int(wrappedID[0] + nx * (wrappedID[1] + ny * wrappedID[2]))
        basePoint = points[adjCellIndex]

        if out_of_bounds:
            # Shift the wrapped point by the neighbor tile offset scaled by cell size
            tile_shift = np.array(off, dtype=np.float32) * cell_size
            candidatePoint = basePoint + tile_shift
        else:
            candidatePoint = basePoint

        # Distance to candidate point
        d = sp - candidatePoint
        minsqrDst = min(minsqrDst, np.dot(d, d))

    return np.sqrt(minsqrDst)

def generate_cloud_cubes(num_cubes, cloud_radius=5.05, center=np.array([0.0, 0.0, 0.0]), offset=0.0):
    cubes = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~2.399963...

    for i in range(num_cubes):
        # Fibonacci sphere distribution
        y = 1.0 - (i / float(num_cubes - 1)) * 2.0   # from +1 to -1
        radius = np.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i

        dir = np.array([
            np.cos(theta) * radius,
            y,
            np.sin(theta) * radius
        ])

        # Apply radial jitter offset (distance variation)
        radial_variation = 1.0
        if offset > 0.0:
            radial_variation += offset * np.random.uniform(-1.0, 1.0)

        pos = center + dir * (cloud_radius * radial_variation)

        # Orientation basis: align cube "up" with sphere normal
        up = dir
        right = np.cross(up, np.array([0, 1, 0]))
        if np.linalg.norm(right) < 1e-6:  # handle degeneracy
            right = np.cross(up, np.array([1, 0, 0]))
        right /= np.linalg.norm(right)
        forward = np.cross(right, up)

        rotation_matrix = np.column_stack((right, up, forward))

        cubes.append({
            "position": pos,
            "rotation": rotation_matrix,
            "scale": 0.25
        })

    return cubes

def generate_worley_volume(points, num_cells, resolution, normalize_sample=True):
    nx, ny, nz = resolution
    volume = np.zeros((nx, ny, nz), dtype=np.float32)

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                # Directly map to [0,1]^3 domain
                sp_norm = np.array([x / nx, y / ny, z / nz], dtype=np.float32)

                # If normalize_sample is False, you could skip this step
                sample_pos = sp_norm if normalize_sample else np.array([x, y, z], dtype=np.float32)

                volume[x, y, z] = worley(points, num_cells, sample_pos)

    return volume.flatten(order='C').astype(np.float32)


def create_worley_3d_texture(numCellsPerAxis=8, resolution=(64,64,64)):
    # 1) Generate Worley points
    points = CreateWorleyPointsBuffer(numCellsPerAxis, bufferName="WorleyPoints")

    # 2) Generate Worley volume
    num_cells = np.array([numCellsPerAxis]*3, dtype=np.int32)
    volume = generate_worley_volume(points, num_cells, resolution, normalize_sample=True)
    volume = 1 - volume
    # 3) Upload to OpenGL 3D texture (no inversion, original wrap/filter)
    nx, ny, nz = resolution
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_3D, tex_id)

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT)

    # Swizzle so red channel is used for all components
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_SWIZZLE_R, GL_RED)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_SWIZZLE_G, GL_RED)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_SWIZZLE_B, GL_RED)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_SWIZZLE_A, GL_ONE)

    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, nx, ny, nz, 0, GL_RED, GL_FLOAT, volume)

    # 4) Generate mipmaps
    glGenerateMipmap(GL_TEXTURE_3D)

    return tex_id


