# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ctypes
import unittest
import warnings

import numpy as np
import warp as wp

from newton._src.viewer.gl.opengl import MeshGL, MeshInstancerGL, RendererGL


class _FakeGL:
    """Records the OpenGL calls made by the mesh classes without a real context."""

    GL_ARRAY_BUFFER = 0x8892
    GL_ELEMENT_ARRAY_BUFFER = 0x8893
    GL_STATIC_DRAW = 0x88E4
    GL_DYNAMIC_DRAW = 0x88E8
    GL_FLOAT = 0x1406
    GL_FALSE = 0
    GL_TRIANGLES = 0x0004
    GL_UNSIGNED_INT = 0x1405
    GL_CULL_FACE = 0x0B44
    GL_TEXTURE_2D = 0x0DE1
    GL_TEXTURE1 = 0x84C1
    GL_RGBA = 0x1908
    GL_UNSIGNED_BYTE = 0x1401
    GL_TEXTURE_MIN_FILTER = 0x2801
    GL_TEXTURE_MAG_FILTER = 0x2800
    GL_NEAREST = 0x2600

    GLuint = ctypes.c_uint
    GLubyte = ctypes.c_ubyte

    def __init__(self):
        self._next_id = 1
        self._bound_vao = 0
        self._bound_buffers = {}
        self.buffer_data = {}
        # {vao id: {attribute index: dict or None when disabled}}
        self.vao_attribs = {}
        # generic (non-array) vertex attribute values, keyed by attribute index
        self.generic_attribs = {}

    def _allocate(self, out):
        out.value = self._next_id
        self._next_id += 1

    def _attribs(self):
        return self.vao_attribs.setdefault(self._bound_vao, {})

    # --- object creation ---

    def glGenVertexArrays(self, n, arrays):
        self._allocate(arrays)

    def glGenBuffers(self, n, buffers):
        self._allocate(buffers)

    def glGenTextures(self, n, textures):
        self._allocate(textures)

    def glDeleteVertexArrays(self, n, arrays):
        pass

    def glDeleteBuffers(self, n, buffers):
        self.buffer_data.pop(int(buffers.value), None)

    def glDeleteTextures(self, n, textures):
        pass

    # --- binding and uploads ---

    def glBindVertexArray(self, vao):
        self._bound_vao = int(getattr(vao, "value", vao))

    def glBindBuffer(self, target, buffer):
        self._bound_buffers[target] = int(getattr(buffer, "value", buffer))

    def glBufferData(self, target, size, data, usage):
        buffer = self._bound_buffers.get(target, 0)
        self.buffer_data[buffer] = ctypes.string_at(data, size) if data else None

    # --- vertex attributes ---

    def glVertexAttribPointer(self, index, size, type_, normalized, stride, offset):
        self._attribs()[index] = {
            "buffer": self._bound_buffers.get(self.GL_ARRAY_BUFFER, 0),
            "size": size,
            "stride": stride,
            "enabled": False,
            "divisor": 0,
        }

    def glEnableVertexAttribArray(self, index):
        self._attribs().setdefault(index, {})["enabled"] = True

    def glDisableVertexAttribArray(self, index):
        self._attribs()[index] = None

    def glVertexAttribDivisor(self, index, divisor):
        self._attribs().setdefault(index, {})["divisor"] = divisor

    def glVertexAttrib3f(self, index, x, y, z):
        self.generic_attribs[index] = (x, y, z)

    def glVertexAttrib4f(self, index, x, y, z, w):
        self.generic_attribs[index] = (x, y, z, w)

    # --- draw state (no-ops) ---

    def glEnable(self, cap):
        pass

    def glDisable(self, cap):
        pass

    def glActiveTexture(self, texture):
        pass

    def glBindTexture(self, target, texture):
        pass

    def glTexImage2D(self, *args):
        pass

    def glTexParameteri(self, *args):
        pass

    def glDrawElements(self, *args):
        pass

    def glDrawElementsInstanced(self, *args):
        pass


class TestViewerGLVertexColors(unittest.TestCase):
    """Per-vertex color plumbing in the OpenGL mesh and instancer buffers."""

    VERTEX_COLOR_LOCATION = 9

    def setUp(self):
        self._saved_gl = RendererGL.gl
        self._saved_fallback = RendererGL._fallback_texture
        self.gl = _FakeGL()
        RendererGL.gl = self.gl
        RendererGL._fallback_texture = None

        self.device = wp.get_device("cpu")
        self.points = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3, device=self.device)
        self.indices = wp.array([0, 1, 2], dtype=wp.int32, device=self.device)

    def tearDown(self):
        RendererGL.gl = self._saved_gl
        RendererGL._fallback_texture = self._saved_fallback

    def _make_mesh(self):
        return MeshGL(len(self.points), len(self.indices), self.device)

    def _mesh_attrib(self, mesh):
        return self.gl.vao_attribs[int(mesh.vao.value)].get(self.VERTEX_COLOR_LOCATION)

    def test_mesh_without_vertex_colors_zeroes_blend_weight(self):
        """Leave the color attribute array disabled and neutralize its generic value."""
        mesh = self._make_mesh()
        mesh.update(self.points, self.indices, None, None)

        self.assertIsNone(mesh.color_vbo)
        self.assertIsNone(self._mesh_attrib(mesh))

        mesh.render()

        # the generic attribute default is (0,0,0,1), which would otherwise select black
        self.assertEqual(self.gl.generic_attribs[self.VERTEX_COLOR_LOCATION], (0.0, 0.0, 0.0, 0.0))

    def test_mesh_uploads_vertex_colors_with_unit_blend_weight(self):
        """Upload per-vertex colors as vec4 with alpha 1 and enable the color attribute."""
        colors_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        colors = wp.array(colors_np, dtype=wp.vec3, device=self.device)

        mesh = self._make_mesh()
        mesh.update(self.points, self.indices, None, None, vertex_colors=colors)

        self.assertIsNotNone(mesh.color_vbo)
        attrib = self._mesh_attrib(mesh)
        self.assertEqual(attrib["buffer"], int(mesh.color_vbo.value))
        self.assertEqual(attrib["size"], 4)
        self.assertEqual(attrib["stride"], MeshGL.vertex_color_byte_size)
        self.assertTrue(attrib["enabled"])

        uploaded = np.frombuffer(self.gl.buffer_data[int(mesh.color_vbo.value)], dtype=np.float32).reshape(-1, 4)
        np.testing.assert_allclose(uploaded[:, :3], colors_np)
        np.testing.assert_allclose(uploaded[:, 3], np.ones(len(colors_np)))

        self.gl.generic_attribs.clear()
        mesh.render()
        self.assertNotIn(self.VERTEX_COLOR_LOCATION, self.gl.generic_attribs)

    def test_mesh_drops_vertex_colors_when_none(self):
        """Release the color buffer and disable the attribute when colors are logged as None."""
        colors = wp.array([[1.0, 0.0, 0.0]] * 3, dtype=wp.vec3, device=self.device)

        mesh = self._make_mesh()
        mesh.update(self.points, self.indices, None, None, vertex_colors=colors)
        color_vbo = int(mesh.color_vbo.value)

        mesh.update(self.points, self.indices, None, None)

        self.assertIsNone(mesh.color_vbo)
        self.assertIsNone(self._mesh_attrib(mesh))
        self.assertNotIn(color_vbo, self.gl.buffer_data)

    def test_mesh_rejects_mismatched_vertex_color_count(self):
        """Raise when the number of vertex colors does not match the number of points."""
        colors = wp.array([[1.0, 0.0, 0.0]] * 2, dtype=wp.vec3, device=self.device)

        mesh = self._make_mesh()
        with self.assertRaises(RuntimeError):
            mesh.update(self.points, self.indices, None, None, vertex_colors=colors)

    def test_instancer_binds_vertex_colors_added_after_allocation(self):
        """Mirror the source mesh color buffer into the instancer VAO as a per-vertex attribute.

        The instancer can be allocated before the mesh has any vertex colors, so the
        binding is refreshed at draw time whenever the mesh color buffer changes.
        """
        mesh = self._make_mesh()
        mesh.update(self.points, self.indices, None, None)

        instancer = MeshInstancerGL(1, mesh)
        instancer_vao = int(instancer.vao.value)
        self.assertIsNone(self.gl.vao_attribs[instancer_vao].get(self.VERTEX_COLOR_LOCATION))

        colors = wp.array([[1.0, 0.0, 0.0]] * 3, dtype=wp.vec3, device=self.device)
        mesh.update(self.points, self.indices, None, None, vertex_colors=colors)
        instancer.render()

        attrib = self.gl.vao_attribs[instancer_vao][self.VERTEX_COLOR_LOCATION]
        self.assertEqual(attrib["buffer"], int(mesh.color_vbo.value))
        self.assertTrue(attrib["enabled"])
        self.assertEqual(attrib["divisor"], 0)

        # dropping the colors must disable the attribute again
        mesh.update(self.points, self.indices, None, None)
        instancer.render()
        self.assertIsNone(self.gl.vao_attribs[instancer_vao].get(self.VERTEX_COLOR_LOCATION))
        self.assertEqual(self.gl.generic_attribs[self.VERTEX_COLOR_LOCATION], (0.0, 0.0, 0.0, 0.0))


class TestViewerRerunVertexColors(unittest.TestCase):
    """Per-vertex color forwarding to rerun's Mesh3D archetype."""

    def _create_viewer(self):
        """Create a ViewerRerun with a mocked rerun backend."""
        from unittest.mock import Mock, patch  # noqa: PLC0415

        self.mesh3d_calls = []

        # a real function so the viewer's signature introspection sees rerun's parameters
        def mesh3d(
            vertex_positions=None,
            triangle_indices=None,
            vertex_normals=None,
            vertex_colors=None,
            vertex_texcoords=None,
            albedo_texture=None,
            albedo_texture_buffer=None,
            albedo_texture_format=None,
        ):
            self.mesh3d_calls.append(dict(locals()))
            return Mock()

        self.mock_rr = Mock()
        self.mock_rr.Mesh3D = mesh3d
        mock_rrb = Mock()

        with (
            patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr),
            patch("newton._src.viewer.viewer_rerun.rrb", mock_rrb),
            patch("newton._src.viewer.viewer_rerun.is_jupyter_notebook", return_value=False),
            warnings.catch_warnings(),
        ):
            from newton._src.viewer.viewer_rerun import ViewerRerun  # noqa: PLC0415

            warnings.simplefilter("ignore")
            return ViewerRerun(serve_web_viewer=False)

    def _mesh3d_kwargs(self):
        return self.mesh3d_calls[-1]

    def test_log_mesh_forwards_vertex_colors_as_u8(self):
        """Pass per-vertex colors to rerun's Mesh3D as uint8 RGB triplets."""
        from unittest.mock import patch  # noqa: PLC0415

        viewer = self._create_viewer()
        points = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3)
        indices = wp.array([0, 1, 2], dtype=wp.int32)
        colors = wp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=wp.vec3)

        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            viewer.log_mesh("/tri", points, indices, vertex_colors=colors)

        np.testing.assert_array_equal(
            self._mesh3d_kwargs()["vertex_colors"],
            np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
        )

    def test_log_instances_prefers_vertex_colors_over_instance_color(self):
        """Keep a mesh's per-vertex colors instead of tiling the first instance color."""
        from unittest.mock import patch  # noqa: PLC0415

        viewer = self._create_viewer()
        points = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3)
        indices = wp.array([0, 1, 2], dtype=wp.int32)
        colors = wp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=wp.vec3)

        with patch("newton._src.viewer.viewer_rerun.rr", self.mock_rr):
            viewer.log_mesh("/tri", points, indices, vertex_colors=colors)
            viewer.log_instances(
                "/inst",
                "/tri",
                wp.array([wp.transform_identity()], dtype=wp.transform),
                None,
                wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3),
                None,
            )

        np.testing.assert_array_equal(
            self._mesh3d_kwargs()["vertex_colors"],
            np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
        )


if __name__ == "__main__":
    unittest.main()
