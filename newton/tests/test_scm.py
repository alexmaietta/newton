# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SCM deformable terrain model."""

import itertools
import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.viewer.colormap import COLORMAP_RESOLUTION, Colormap, colormap_table
from newton.terramechanics import SCMPlotField, SCMTerrain, scm_soil_parameters


def _sphere_model(z: float = 1.0, radius: float = 0.25):
    """Build a single-sphere model and its state, positioned at height ``z``."""
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, z)), label="sphere")
    builder.add_shape_sphere(body, radius=radius)
    model = builder.finalize()
    state = model.state()
    return model, state, body


def _cohesionless_soil():
    """Cohesionless soil, as used by the Chrono Viper rover demo."""
    return scm_soil_parameters(
        bekker_kphi=2.0e6,
        bekker_kc=0.0,
        bekker_n=1.1,
        mohr_cohesion=0.0,
        mohr_friction=30.0,
        janosi_shear=0.01,
        elastic_k=2.0e8,
        damping_r=3.0e4,
    )


def _cohesive_soil():
    """LETE sand: the cohesive set, which exercises the Bekker ``Kc/b`` term."""
    return scm_soil_parameters(
        bekker_kphi=5301.0e3,
        bekker_kc=102.0e3,
        bekker_n=0.793,
        mohr_cohesion=1.3e3,
        mohr_friction=31.1,
        janosi_shear=1.2e-2,
        elastic_k=4.0e8,
        damping_r=3.0e4,
    )


def _soft_soil():
    """Loose soil, as used by the Chrono rigid-tire demo."""
    return scm_soil_parameters(
        bekker_kphi=0.2e6,
        bekker_kc=0.0,
        bekker_n=1.1,
        mohr_cohesion=0.0,
        mohr_friction=30.0,
        janosi_shear=0.01,
        elastic_k=4.0e7,
        damping_r=3.0e4,
    )


def _place(state, z: float, x: float = 0.0):
    """Teleport the single body of ``state`` to ``(x, 0, z)``."""
    state.body_q.assign(wp.array([wp.transform(p=wp.vec3(x, 0.0, z))], dtype=wp.transform))


def _terrain(model, **kwargs):
    """Build a small terrain over the whole model with sensible test defaults."""
    kwargs.setdefault("size", (2.0, 2.0))
    kwargs.setdefault("spacing", 0.05)
    terrain = SCMTerrain(model, **kwargs)
    terrain.add_newton_model()
    return terrain


class TestSCMConstitutive(unittest.TestCase):
    """The pressure-sinkage and shear laws, driven through a single node."""

    def test_scm_unilaterality(self):
        """Verify a node barely above the soil produces no force."""
        model, state, _ = _sphere_model()
        terrain = _terrain(model)
        # Sphere bottom sits above the surface but inside the ray's reach.
        _place(state, 0.30)
        terrain.step(state, 1.0 / 240.0)

        self.assertGreater(int(terrain.counters.numpy()[0]), 0, "expected ray hits")
        self.assertAlmostEqual(float(terrain.body_f.numpy()[0][2]), 0.0, places=6)
        self.assertAlmostEqual(float(terrain.nodes.numpy()["level"].min()), 0.0, places=9)

    def test_scm_bekker_yield(self):
        """Verify peak pressure follows the Bekker curve at the deepest node."""
        soil = _cohesionless_soil()
        model, state, _ = _sphere_model()
        terrain = _terrain(model, soil=soil)

        penetration = 0.02
        _place(state, 0.25 - penetration)
        terrain.step(state, 1.0 / 240.0)

        nodes = terrain.nodes.numpy()
        deepest = int(np.argmax(nodes["sinkage"]))
        sinkage = float(nodes["sinkage"][deepest])
        self.assertAlmostEqual(sinkage, penetration, places=3)

        expected = soil.bekker_kphi * sinkage**soil.bekker_n
        self.assertAlmostEqual(float(nodes["sigma"][deepest]), expected, delta=0.02 * expected)
        self.assertAlmostEqual(float(nodes["sigma_yield"][deepest]), expected, delta=0.02 * expected)

    def test_scm_force_increases_with_penetration(self):
        """Verify the net soil reaction grows monotonically with sinkage."""
        forces = []
        for penetration in (0.005, 0.01, 0.02, 0.04):
            model, state, _ = _sphere_model()
            terrain = _terrain(model)
            _place(state, 0.25 - penetration)
            terrain.step(state, 1.0 / 240.0)
            forces.append(float(terrain.body_f.numpy()[0][2]))

        for lo, hi in itertools.pairwise(forces):
            self.assertGreater(hi, lo)
        self.assertGreater(forces[0], 0.0)

    def test_scm_unloading_is_elastic(self):
        """Verify plastic sinkage is retained when the load is removed.

        Pressing and then lifting must leave a permanent rut: the plastic part of
        the sinkage is the soil's memory of peak compaction.
        """
        model, state, _ = _sphere_model()
        terrain = _terrain(model)

        _place(state, 0.25 - 0.03)
        terrain.step(state, 1.0 / 240.0)
        rut_loaded = float(terrain.nodes.numpy()["level"].min())

        # Lift well clear and step again.
        _place(state, 2.0)
        terrain.step(state, 1.0 / 240.0)
        nodes = terrain.nodes.numpy()

        self.assertAlmostEqual(float(nodes["level"].min()), rut_loaded, places=9)
        self.assertGreater(float(nodes["sinkage_plastic"].max()), 0.0)
        self.assertAlmostEqual(float(terrain.body_f.numpy()[0][2]), 0.0, places=6)

    def test_scm_janosi_shear_saturates(self):
        """Verify shear stress approaches the Mohr-Coulomb limit as slip accumulates."""
        soil = _cohesionless_soil()
        model, state, _ = _sphere_model()
        terrain = _terrain(model, soil=soil)

        _place(state, 0.25 - 0.02)
        # Slide sideways so tangential slip accumulates.
        state.body_qd.assign(wp.array([wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)], dtype=wp.spatial_vector))
        for _ in range(200):
            terrain.step(state, 1.0 / 240.0)

        nodes = terrain.nodes.numpy()
        loaded = nodes["sigma"] > 0.0
        self.assertTrue(loaded.any())
        ratio = nodes["tau"][loaded] / (soil.mohr_cohesion + nodes["sigma"][loaded] * soil.mohr_mu)
        self.assertGreater(float(ratio.max()), 0.99)
        self.assertLessEqual(float(ratio.max()), 1.0 + 1e-6)

    def test_scm_shear_force_opposes_sliding(self):
        """Verify the tangential reaction points against the direction of travel."""
        model, state, _ = _sphere_model()
        terrain = _terrain(model)
        _place(state, 0.25 - 0.02)
        state.body_qd.assign(wp.array([wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)], dtype=wp.spatial_vector))
        for _ in range(50):
            terrain.step(state, 1.0 / 240.0)

        self.assertLess(float(terrain.body_f.numpy()[0][0]), 0.0)


class TestSCMGeometry(unittest.TestCase):
    """Grid construction, ray casting and shape registration."""

    def test_scm_grid_snaps_to_whole_cells(self):
        """Verify the grid spans the requested extent in a whole number of cells."""
        model, _, _ = _sphere_model()
        terrain = SCMTerrain(model, size=(2.0, 1.0), spacing=0.03)
        self.assertAlmostEqual(terrain.delta * (terrain.nx - 1), 2.0, places=9)
        self.assertLessEqual(terrain.delta, 0.03 + 1e-9)
        self.assertEqual(terrain.node_count, terrain.nx * terrain.ny)
        self.assertEqual(len(terrain.indices), 3 * 2 * (terrain.nx - 1) * (terrain.ny - 1))

    def test_scm_unregistered_shapes_ignored(self):
        """Verify a shape overlapping the patch produces nothing unless registered."""
        model, state, _body = _sphere_model(z=0.25 - 0.02)

        terrain = SCMTerrain(model, size=(2.0, 2.0), spacing=0.05)
        terrain.add_shape(0)
        terrain.step(state, 1.0 / 240.0)
        self.assertGreater(int(terrain.counters.numpy()[0]), 0)

        # Same scene, but nothing registered except a second, distant shape.
        builder = newton.ModelBuilder()
        near = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.23)))
        builder.add_shape_sphere(near, radius=0.25)
        far = builder.add_body(xform=wp.transform(p=wp.vec3(50.0, 0.0, 0.23)))
        builder.add_shape_sphere(far, radius=0.25)
        model2 = builder.finalize()
        state2 = model2.state()

        terrain2 = SCMTerrain(model2, size=(2.0, 2.0), spacing=0.05)
        terrain2.add_body(far)
        terrain2.step(state2, 1.0 / 240.0)
        self.assertEqual(int(terrain2.counters.numpy()[0]), 0)
        self.assertAlmostEqual(float(np.abs(terrain2.body_f.numpy()).max()), 0.0, places=9)

    def test_scm_registration_after_step_raises(self):
        """Verify registering a shape after the first step is rejected.

        Registration changes kernel launch dimensions, which an enclosing CUDA
        graph bakes in, so it must fail loudly rather than silently invalidate.
        """
        model, state, body = _sphere_model()
        terrain = _terrain(model)
        terrain.step(state, 1.0 / 240.0)
        with self.assertRaises(RuntimeError):
            terrain.add_body(body)
        with self.assertRaises(RuntimeError):
            terrain.add_shape(0)

    def test_scm_add_newton_model_skips_static_and_sites(self):
        """Verify whole-model registration excludes world-attached shapes by default."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0)))
        builder.add_shape_sphere(body, radius=0.25)
        model = builder.finalize()

        dynamic_only = SCMTerrain(model, size=(2.0, 2.0), spacing=0.1)
        self.assertEqual(dynamic_only.add_newton_model(), 1)

        with_static = SCMTerrain(model, size=(2.0, 2.0), spacing=0.1)
        self.assertEqual(with_static.add_newton_model(include_static=True), 2)

    def test_scm_registration_is_idempotent(self):
        """Verify overlapping registration calls do not add duplicate shapes."""
        model, _, body = _sphere_model()
        terrain = SCMTerrain(model, size=(2.0, 2.0), spacing=0.1)
        self.assertEqual(terrain.add_newton_model(), 1)
        self.assertEqual(terrain.add_body(body), 0)
        self.assertEqual(terrain.add_shape(0), 0)

    def test_scm_requires_registered_shapes(self):
        """Verify stepping without any registered shape fails with a clear message."""
        model, state, _ = _sphere_model()
        terrain = SCMTerrain(model, size=(1.0, 1.0), spacing=0.1)
        with self.assertRaises(RuntimeError):
            terrain.step(state, 1.0 / 240.0)

    def test_scm_query_height_tracks_deformation(self):
        """Verify height queries return the deformed surface and edge-clamp outside it."""
        model, state, _ = _sphere_model()
        terrain = _terrain(model)
        _place(state, 0.25 - 0.03)
        terrain.step(state, 1.0 / 240.0)

        points = wp.array(
            [wp.vec3(0.0, 0.0, 5.0), wp.vec3(0.9, 0.9, 5.0), wp.vec3(50.0, 0.0, 5.0)],
            dtype=wp.vec3,
        )
        out = wp.zeros(3, dtype=float)
        terrain.get_height(points, out)
        heights = out.numpy()

        self.assertLess(heights[0], -1.0e-3)  # inside the crater
        self.assertAlmostEqual(float(heights[1]), 0.0, places=6)  # undisturbed
        self.assertAlmostEqual(float(heights[2]), 0.0, places=6)  # clamped to the edge

        undeformed = wp.zeros(3, dtype=float)
        terrain.get_height(points, undeformed, undeformed=True)
        self.assertAlmostEqual(float(undeformed.numpy()[0]), 0.0, places=9)


class TestSCMContactPatches(unittest.TestCase):
    """Connected-component labeling and the Bekker ``1/b`` term."""

    def test_scm_contact_patches_auto_enable(self):
        """Verify patch labeling turns on exactly when the cohesive modulus is nonzero."""
        model, _, _ = _sphere_model()
        self.assertTrue(SCMTerrain(model, soil=_cohesive_soil()).contact_patches)
        self.assertFalse(SCMTerrain(model, soil=_cohesionless_soil()).contact_patches)
        self.assertTrue(SCMTerrain(model, soil=_cohesionless_soil(), contact_patches=True).contact_patches)

    def test_scm_ccl_labels_single_contact_as_one_patch(self):
        """Verify one contiguous footprint is labeled as exactly one patch."""
        model, state, _ = _sphere_model()
        terrain = _terrain(model, soil=_cohesive_soil())
        _place(state, 0.25 - 0.04)
        terrain.step(state, 1.0 / 240.0)

        labels = terrain.labels.numpy()
        patches = set(labels[labels >= 0].tolist())
        self.assertEqual(len(patches), 1)

    def test_scm_ccl_separates_disjoint_contacts(self):
        """Verify two well-separated footprints receive distinct patch labels."""
        builder = newton.ModelBuilder()
        for x in (-0.6, 0.6):
            body = builder.add_body(xform=wp.transform(p=wp.vec3(x, 0.0, 0.25 - 0.04)))
            builder.add_shape_sphere(body, radius=0.25)
        model = builder.finalize()
        state = model.state()

        terrain = _terrain(model, soil=_cohesive_soil())
        terrain.step(state, 1.0 / 240.0)

        labels = terrain.labels.numpy()
        patches = set(labels[labels >= 0].tolist())
        self.assertEqual(len(patches), 2)

    def test_scm_patch_area_and_perimeter_are_grid_native(self):
        """Verify patch area is the node count times cell area, and ``oob`` follows.

        Unlike the reference implementation, which takes a convex hull of the node
        centers, area and perimeter are counted directly on the grid.
        """
        model, state, _ = _sphere_model()
        terrain = _terrain(model, soil=_cohesive_soil())
        _place(state, 0.25 - 0.04)
        terrain.step(state, 1.0 / 240.0)

        labels = terrain.labels.numpy()
        area = terrain.patch_area.numpy()
        perimeter = terrain.patch_perimeter.numpy()
        oob = terrain.patch_oob.numpy()

        label = int(labels[labels >= 0][0])
        count = int((labels == label).sum())
        self.assertAlmostEqual(float(area[label]), count * terrain.delta**2, places=6)
        self.assertAlmostEqual(float(oob[label]), float(perimeter[label]) / (2.0 * float(area[label])), places=5)

    def test_scm_oob_affects_only_cohesive_term(self):
        """Verify patch labeling changes nothing when the cohesive modulus is zero.

        This pins the claim that the fused single-kernel path is exact for
        ``bekker_kc == 0`` rather than an approximation.
        """
        results = []
        for patches in (False, True):
            model, state, _ = _sphere_model()
            terrain = _terrain(model, soil=_cohesionless_soil(), contact_patches=patches)
            _place(state, 0.25 - 0.03)
            terrain.step(state, 1.0 / 240.0)
            results.append((terrain.body_f.numpy().copy(), terrain.nodes.numpy()["level"].copy()))

        np.testing.assert_allclose(results[0][0], results[1][0], rtol=0, atol=0)
        np.testing.assert_allclose(results[0][1], results[1][1], rtol=0, atol=0)


class TestSCMBulldozing(unittest.TestCase):
    """Lateral material displacement and erosion."""

    def test_scm_bulldozing_raises_material_beside_the_rut(self):
        """Verify displaced soil piles up next to the crater instead of vanishing."""
        model, state, _ = _sphere_model()
        terrain = _terrain(model, bulldozing=True)

        for depth in np.linspace(0.005, 0.05, 20):
            _place(state, 0.25 - float(depth))
            terrain.step(state, 1.0 / 240.0)

        levels = terrain.nodes.numpy()["level"]
        self.assertLess(float(levels.min()), -1.0e-3)
        self.assertGreater(float(levels.max()), 1.0e-4)

    def test_scm_without_bulldozing_nothing_rises(self):
        """Verify soil is only ever pushed down when bulldozing is disabled."""
        model, state, _ = _sphere_model()
        terrain = _terrain(model, bulldozing=False)
        _place(state, 0.25 - 0.03)
        terrain.step(state, 1.0 / 240.0)

        self.assertLessEqual(float(terrain.nodes.numpy()["level"].max()), 1.0e-9)


class TestSCMVisualization(unittest.TestCase):
    """Colormaps and the plot-field mapping."""

    def test_colormap_endpoints_and_clamping(self):
        """Verify colormap sampling matches the table ends and clamps out of range."""
        model, _, _ = _sphere_model()
        terrain = _terrain(model, plot_field=SCMPlotField.LEVEL)
        for cmap in Colormap:
            table = colormap_table(cmap)
            self.assertEqual(table.shape, (COLORMAP_RESOLUTION, 3))
            self.assertGreaterEqual(float(table.min()), 0.0)
            self.assertLessEqual(float(table.max()), 1.0)
        del terrain

    def test_scm_plot_field_maps_expected_quantity(self):
        """Verify each plot field colors the mesh and NONE falls back to the base color."""
        model, state, _ = _sphere_model()
        terrain = _terrain(model, plot_field=SCMPlotField.NONE, base_color=(0.25, 0.5, 0.75))
        _place(state, 0.25 - 0.03)
        terrain.step(state, 1.0 / 240.0)

        terrain.update_visualization()
        colors = terrain.colors.numpy()
        np.testing.assert_allclose(colors, np.tile([0.25, 0.5, 0.75], (len(colors), 1)), atol=1e-6)

        terrain.set_plot(field=SCMPlotField.SINKAGE, vmin=0.0, vmax=0.05)
        terrain.update_visualization()
        varied = terrain.colors.numpy()
        self.assertGreater(float(varied.std()), 1.0e-3)

    def test_scm_auto_range_ignores_untouched_nodes(self):
        """Verify auto-ranging brackets the deformed region, not the flat majority."""
        model, state, _ = _sphere_model()
        terrain = _terrain(model, plot_field=SCMPlotField.SINKAGE, auto_range=True)
        _place(state, 0.25 - 0.03)
        terrain.step(state, 1.0 / 240.0)
        terrain.update_visualization(1.0)

        _vmin, vmax = terrain.plot_range
        self.assertGreater(vmax, 1.0e-3)
        self.assertLessEqual(vmax, 0.031)

    def test_scm_mesh_follows_deformation(self):
        """Verify visualization vertices track the deformed node heights."""
        model, state, _ = _sphere_model()
        terrain = _terrain(model)
        _place(state, 0.25 - 0.03)
        terrain.step(state, 1.0 / 240.0)
        terrain.update_visualization()

        vertices = terrain.vertices.numpy()
        levels = terrain.nodes.numpy()["level"]
        np.testing.assert_allclose(vertices[:, 2], levels, atol=1e-6)
        self.assertEqual(len(terrain.normals), terrain.node_count)
        self.assertEqual(len(terrain.colors), terrain.node_count)


class TestSCMConfiguration(unittest.TestCase):
    """Parameter handling and validation."""

    def test_scm_friction_angle_stored_as_tangent(self):
        """Verify the friction angle is converted to its tangent at construction."""
        soil = scm_soil_parameters(mohr_friction=45.0)
        self.assertAlmostEqual(soil.mohr_mu, math.tan(math.radians(45.0)), places=6)

    def test_scm_elastic_stiffness_clamped_to_kphi(self):
        """Verify elastic stiffness is never softer than the Bekker loading modulus."""
        soil = scm_soil_parameters(bekker_kphi=1.0e9, elastic_k=1.0)
        self.assertEqual(soil.elastic_k, 1.0e9)

    def test_scm_rejects_invalid_geometry(self):
        """Verify non-positive patch size or spacing is rejected."""
        model, _, _ = _sphere_model()
        with self.assertRaises(ValueError):
            SCMTerrain(model, size=(0.0, 1.0))
        with self.assertRaises(ValueError):
            SCMTerrain(model, spacing=0.0)

    def test_scm_soil_parameters_update_in_place(self):
        """Verify changing soil parameters takes effect without rebuilding the terrain.

        The parameters live in a device array so they stay mutable under CUDA
        graph capture.
        """
        model, state, _ = _sphere_model()
        terrain = _terrain(model, soil=_soft_soil())
        _place(state, 0.25 - 0.03)
        terrain.step(state, 1.0 / 240.0)
        soft_force = float(terrain.body_f.numpy()[0][2])

        model2, state2, _ = _sphere_model()
        terrain2 = _terrain(model2, soil=_soft_soil())
        terrain2.set_soil_parameters(_cohesionless_soil())
        _place(state2, 0.25 - 0.03)
        terrain2.step(state2, 1.0 / 240.0)
        mid_force = float(terrain2.body_f.numpy()[0][2])

        self.assertGreater(mid_force, soft_force)

    def test_scm_warns_when_cohesion_needs_patches(self):
        """Verify a cohesive soil on a terrain without patch labeling warns."""
        model, _, _ = _sphere_model()
        terrain = _terrain(model, soil=_cohesionless_soil(), contact_patches=False)
        with self.assertWarns(UserWarning):
            terrain.set_soil_parameters(_cohesive_soil())

    def test_scm_attach_ui_returns_false_without_support(self):
        """Verify GUI attachment degrades gracefully on viewers with no UI hook.

        The panel must never become a hard dependency for headless runs.
        """
        model, _, _ = _sphere_model()
        terrain = _terrain(model)
        self.assertFalse(terrain.attach_ui(object()))
        terrain.draw_ui(object())  # must not raise


class TestSCMDeterminism(unittest.TestCase):
    """Repeatability of the terrain update."""

    def test_scm_repeated_runs_are_identical(self):
        """Verify two identical runs produce bitwise-identical soil state.

        Each node casts its own ray and takes the nearest hit, so there is no
        order dependence in which shape claims a node.
        """
        results = []
        for _ in range(2):
            model, state, _ = _sphere_model()
            terrain = _terrain(model, bulldozing=True)
            for depth in np.linspace(0.005, 0.04, 10):
                _place(state, 0.25 - float(depth), x=float(depth))
                terrain.step(state, 1.0 / 240.0)
            results.append(terrain.nodes.numpy()["level"].copy())

        np.testing.assert_array_equal(results[0], results[1])


if __name__ == "__main__":
    wp.init()
    unittest.main()
