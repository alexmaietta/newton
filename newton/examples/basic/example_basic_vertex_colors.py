# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Vertex Colors
#
# Shows how to log a mesh with animated per-vertex colors via
# Viewer.log_mesh(vertex_colors=...). A rippling grid is false-colored by
# height with a colormap. No physics model is involved.
#
# Command: python -m newton.examples basic_vertex_colors
###########################################################################

import numpy as np
import warp as wp

import newton.examples
from newton.viewer import Colormap, sample_colormap

GRID_RES = 64
GRID_SPACING = 0.1


@wp.kernel
def update_wave_mesh(
    time: float,
    res: int,
    spacing: float,
    positions: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
):
    tid = wp.tid()
    i = tid % res
    j = tid // res
    x = (float(i) - float(res - 1) * 0.5) * spacing
    y = (float(j) - float(res - 1) * 0.5) * spacing
    r = wp.sqrt(x * x + y * y)
    z = 0.5 * wp.sin(3.0 * r - 2.0 * time) * wp.exp(-0.3 * r)
    positions[tid] = wp.vec3(x, y, z)
    colors[tid] = sample_colormap(int(Colormap.TURBO), 0.5 + z)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0

        i, j = np.meshgrid(np.arange(GRID_RES - 1), np.arange(GRID_RES - 1))
        v0 = (j * GRID_RES + i).ravel()
        v1, v2, v3 = v0 + 1, v0 + GRID_RES, v0 + GRID_RES + 1
        indices = np.stack([v0, v2, v1, v1, v2, v3], axis=1).ravel().astype(np.int32)
        self.indices = wp.array(indices, dtype=wp.int32)

        vertex_count = GRID_RES * GRID_RES
        self.positions = wp.zeros(vertex_count, dtype=wp.vec3)
        self.colors = wp.zeros(vertex_count, dtype=wp.vec3)

        self.viewer.set_camera(pos=wp.vec3(4.0, -4.0, 3.0), pitch=-25.0, yaw=135.0)

    def step(self):
        self.sim_time += self.frame_dt

    def render(self):
        wp.launch(
            update_wave_mesh,
            dim=len(self.positions),
            inputs=[self.sim_time, GRID_RES, GRID_SPACING, self.positions, self.colors],
        )

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_mesh("/wave", self.positions, self.indices, vertex_colors=self.colors)
        self.viewer.end_frame()

    def test_final(self):
        """Verify the grid renders with visible per-vertex color variation."""
        colors = self.colors.numpy()
        assert np.ptp(colors, axis=0).max() > 0.1, "expected visible color variation across the mesh"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
