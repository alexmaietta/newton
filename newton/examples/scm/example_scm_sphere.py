# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example SCM Sphere
#
# Drops a sphere onto SCM deformable terrain. The sphere sinks into the soil
# and leaves a permanent crater; the terrain mesh is false-colored by sinkage.
#
# The terrain has no collision geometry: the sphere is supported entirely by
# the soil reaction wrench. Drag the sphere with the mouse to plough the soil.
#
# Soil parameters are taken from the Chrono Viper rover SCM demo.
#
# Command: uv run -m newton.examples scm_sphere
###########################################################################

import warp as wp

import newton
import newton.examples
from newton.terramechanics import SCMPlotField, SCMTerrain, scm_soil_parameters


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        # Soil parameters from the Chrono Viper rover demo,
        # src/demos/robot/viper/demo_ROBOT_Viper_SCM.cpp:201
        soil = scm_soil_parameters(
            bekker_kphi=2.0e6,
            bekker_kc=0.0,
            bekker_n=1.1,
            mohr_cohesion=0.0,
            mohr_friction=30.0,
            janosi_shear=0.01,
            elastic_k=2.0e8,
            damping_r=3.0e4,
        )

        builder = newton.ModelBuilder()

        # No ground shape: the SCM terrain is a pure force field with no
        # collision geometry and holds the sphere up on its own.
        self.sphere = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.2)), label="sphere")
        builder.add_shape_sphere(self.sphere, radius=0.25)

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=4)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.terrain = SCMTerrain(
            self.model,
            size=(3.0, 3.0),
            spacing=0.03,
            soil=soil,
            bulldozing=True,
            erosion_angle=55.0,
            flow_factor=1.0,
            erosion_iterations=5,
            erosion_propagations=6,
            plot_field=SCMPlotField.SINKAGE,
            auto_range=True,
        )
        self.terrain.add_body(self.sphere)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.4, -2.4, 1.6), pitch=-25.0, yaw=135.0)

        self.capture()

    def capture(self):
        # The terrain step is a pure sequence of device kernel launches, so it
        # captures into the same graph as the solver.
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Mouse-drag forces from the viewer.
            self.viewer.apply_forces(self.state_0)

            # Advance the soil one substep, then re-apply its wrench: the soil
            # model is history dependent, and clear_forces() above has already
            # wiped the previous substep's contribution.
            self.terrain.step(self.state_0, self.sim_dt)
            self.terrain.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.terrain.update_visualization(self.frame_dt)
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/terrain",
            self.terrain.vertices,
            self.terrain.indices,
            normals=self.terrain.normals,
            vertex_colors=self.terrain.colors,
        )
        self.viewer.end_frame()

    def gui(self, ui):
        """Plot field, colormap and range controls for the terrain."""
        self.terrain.draw_ui(ui)

    def test_final(self):
        """Verify the sphere came to rest in a crater it dug itself."""
        import numpy as np  # noqa: PLC0415

        levels = self.terrain.nodes.numpy()["level"]
        rut_depth = -float(levels.min())
        assert rut_depth > 5.0e-3, f"expected a visible crater, got {rut_depth:.5f} m"

        q = self.state_0.body_q.numpy()[0]
        qd = self.state_0.body_qd.numpy()[0]
        assert q[2] < 0.25, f"sphere should have settled into the soil, z={q[2]:.4f}"
        assert np.linalg.norm(qd[:3]) < 0.5, f"sphere should be nearly at rest, |v|={np.linalg.norm(qd[:3]):.4f}"

        # Bulldozing conserves material: what leaves the crater piles up beside it.
        assert float(levels.max()) > 1.0e-4, "expected bulldozed material around the crater"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
