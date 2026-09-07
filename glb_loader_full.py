import numpy as np
from pygltflib import GLTF2
import DracoPy

# ----------------------------
# Accessor-based extraction
# ----------------------------
def extract_accessor_data(gltf, accessor_idx):
    acc = gltf.accessors[accessor_idx]
    if acc.bufferView is None:
        raise ValueError(f"Accessor {accessor_idx} has no bufferView")
    if acc.sparse is not None:
        raise NotImplementedError("Sparse accessors not supported yet")

    bv = gltf.bufferViews[acc.bufferView]
    blob = gltf.binary_blob()

    component_dtype = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5124: np.int32,
        5125: np.uint32,
        5126: np.float32
    }[acc.componentType]

    comps_per_elem = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16
    }[acc.type]

    itemsize = np.dtype(component_dtype).itemsize
    base_offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    length = acc.count * comps_per_elem * itemsize
    raw = blob[base_offset:base_offset + length]

    if bv.byteStride and bv.byteStride != itemsize * comps_per_elem:
        stride = bv.byteStride
        data = np.zeros((acc.count, comps_per_elem), dtype=component_dtype)
        for i in range(acc.count):
            offset = i * stride
            chunk = raw[offset: offset + itemsize * comps_per_elem]
            data[i] = np.frombuffer(chunk, dtype=component_dtype, count=comps_per_elem)
    else:
        total = acc.count * comps_per_elem
        data = np.frombuffer(raw, dtype=component_dtype, count=total)
        data = data.reshape((acc.count, comps_per_elem))

    return data

def get_primitive_attribute(prim, name):
    idx = getattr(prim.attributes, name, None)
    if idx is None:
        raise KeyError(f"Primitive has no attribute '{name}'")
    return idx

# ----------------------------
# Draco decoding path
# ----------------------------
def decode_draco_from_primitive(gltf, prim):
    draco_ext = prim.extensions.get("KHR_draco_mesh_compression")
    if not draco_ext:
        raise ValueError("Primitive does not have Draco compression")

    bv_index = draco_ext["bufferView"]  # dict access
    bv = gltf.bufferViews[bv_index]
    blob = gltf.binary_blob()
    base_offset = bv.byteOffset or 0
    length = bv.byteLength
    drc_bytes = blob[base_offset:base_offset + length]

    decoded = DracoPy.decode(drc_bytes)  # DracoMesh object

    vertices = decoded.points.astype(np.float32)

    normals = (
        decoded.normals.astype(np.float32)
        if hasattr(decoded, "normals") and decoded.normals is not None
        else np.zeros_like(vertices)
    )

    uvs = (
        decoded.tex_coord.astype(np.float32)
        if hasattr(decoded, "tex_coord") and decoded.tex_coord is not None
        else np.zeros((vertices.shape[0], 2), dtype=np.float32)
    )

    indices = decoded.faces.flatten().astype(np.uint32)

    tangents = np.zeros((vertices.shape[0], 4), dtype=np.float32)

    return vertices, normals, uvs, tangents, indices

# ----------------------------
# Unified attribute loader
# ----------------------------
def get_vertices_normals_uvs_tangents(gltf, prim):
    pos_acc_idx = get_primitive_attribute(prim, "POSITION")
    norm_acc_idx = get_primitive_attribute(prim, "NORMAL")
    uv_acc_idx = get_primitive_attribute(prim, "TEXCOORD_0")

    vertices = extract_accessor_data(gltf, pos_acc_idx).astype(np.float32)
    normals = extract_accessor_data(gltf, norm_acc_idx).astype(np.float32)
    uvs = extract_accessor_data(gltf, uv_acc_idx).astype(np.float32)

    tangents_acc_idx = getattr(prim.attributes, "TANGENT", None)
    if tangents_acc_idx is not None:
        tangents = extract_accessor_data(gltf, tangents_acc_idx).astype(np.float32)
    else:
        tangents = np.zeros((vertices.shape[0], 4), dtype=np.float32)

    indices = extract_accessor_data(gltf, prim.indices).flatten().astype(np.uint32)

    return vertices, normals, uvs, tangents, indices

# ----------------------------
# Public API
# ----------------------------
def load_model(path):
    gltf = GLTF2().load(path)
    mesh = gltf.meshes[0]
    prim = mesh.primitives[0]

    if prim.extensions and "KHR_draco_mesh_compression" in prim.extensions:
        return decode_draco_from_primitive(gltf, prim)
    else:
        return get_vertices_normals_uvs_tangents(gltf, prim)

def expand_vertices(vertices, uvs, normals, tangents, indices):
    M = indices.shape[0]
    expanded = np.empty((M, 12), dtype=np.float32)
    for i, idx in enumerate(indices):
        expanded[i, 0:3] = vertices[idx]
        expanded[i, 3:5] = uvs[idx][:2]
        expanded[i, 5:8] = normals[idx] if normals.shape[1] >= 3 else [0, 0, 1]
        expanded[i, 8:12] = tangents[idx][:4]
    return np.ascontiguousarray(expanded.flatten(), dtype=np.float32)

def load_glb_as_expanded_buffer(path):
    vertices, normals, uvs, tangents, indices = load_model(path)
    return expand_vertices(vertices, uvs, normals, tangents, indices)

def load_drc_glb_as_expanded_buffer(path):
    """
    Loads a Draco-compressed GLB and returns a flat interleaved buffer
    ready for VAO: [3, 2, 3, 4]
    """
    gltf = GLTF2().load(path)
    mesh = gltf.meshes[0]
    prim = mesh.primitives[0]

    if not prim.extensions or "KHR_draco_mesh_compression" not in prim.extensions:
        raise ValueError("GLB is not Draco-compressed")

    vertices, normals, uvs, tangents, indices = decode_draco_from_primitive(gltf, prim)
    return expand_vertices(vertices, uvs, normals, tangents, indices)
