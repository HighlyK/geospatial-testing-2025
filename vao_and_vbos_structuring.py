from OpenGL.GL import *
import ctypes

def setup_expanded_vao(buffer, attribute_sizes: list[int]):
    """
    Create a VAO/VBO for a non-indexed (expanded) buffer.
    - buffer: 1D float32 array of length M * sum(attribute_sizes)
    - attribute_sizes: list of sizes per attribute (e.g. [3, 2, 3])
    Returns: vao, vbo, vertex_count
    """
    # 1. Generate and bind VAO
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # 2. Generate and bind VBO
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, buffer.nbytes, buffer, GL_STATIC_DRAW)

    # 3. Configure interleaved attributes
    float_size = ctypes.sizeof(ctypes.c_float)
    stride = sum(attribute_sizes) * float_size
    offset = 0

    for loc, size in enumerate(attribute_sizes):
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(
            loc,                 # attribute location
            size,                # number of components
            GL_FLOAT,            # type
            GL_FALSE,            # normalized?
            stride,              # byte stride
            ctypes.c_void_p(offset)  # offset in bytes
        )
        offset += size * float_size

    # 4. Unbind VAO and VBO
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    # 5. Calculate vertex count
    floats_per_vertex = sum(attribute_sizes)
    vertex_count = len(buffer) // floats_per_vertex

    return vao, vbo, vertex_count


def setup_indexed_vao(buffer, attribute_sizes, indices):
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # VBO
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, buffer.nbytes, buffer, GL_STATIC_DRAW)

    # EBO
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Attributes
    float_size = ctypes.sizeof(ctypes.c_float)
    stride = sum(attribute_sizes) * float_size
    offset = 0
    for loc, size in enumerate(attribute_sizes):
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        offset += size * float_size

    glBindVertexArray(0)
    return vao, vbo, ebo, len(indices)