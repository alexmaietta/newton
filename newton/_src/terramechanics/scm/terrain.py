# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ...geometry.flags import ShapeFlags
from ...viewer.colormap import Colormap
from . import kernels as K
from .types import SCMNodeState, SCMPlotField, SCMSoilParameters, scm_soil_parameters

if TYPE_CHECKING:
    from ...geometry.types import Heightfield
    from ...sim import Model, State

__all__ = ["SCMTerrain"]


class SCMTerrain:
    """Deformable terrain based on the Soil Contact Model (SCM).

    Implements a Bekker-Janosi-Hanamoto semi-empirical terramechanics model on a
    regular grid. Deformation is resolved by casting one vertical ray per grid
    node against the shapes registered with :meth:`add_newton_model`,
    :meth:`add_body` or :meth:`add_shape`; the resulting sinkage drives an
    elastoplastic pressure-sinkage law and a Janosi-Hanamoto shear law, and the
    tractions are accumulated into :attr:`newton.State.body_f`.

    The terrain contributes **no collision geometry**. Bodies are supported
    entirely by the soil reaction wrench.

    Call :meth:`step` **once per substep**, with the substep timestep, exactly
    where a collision pipeline would run. The soil model carries history, so it
    must be advanced once per integration step and never re-evaluated within
    one. Stepping it once per frame and holding the wrench across substeps is
    unstable: the viscous term is velocity-dependent, so a stale force keeps
    pushing after the velocity it opposed has already reversed.

    Every stage is a device kernel with no host-side work, so :meth:`step` can be
    captured in a CUDA graph together with the solver.

    Example:
        .. code-block:: python

            terrain = SCMTerrain(model, size=(8.0, 3.0), spacing=0.04)
            terrain.add_body(wheel)

            for _ in range(substeps):
                state.clear_forces()
                terrain.step(state, sim_dt)
                terrain.apply_forces(state)
                solver.step(state, state_next, control, None, sim_dt)
                state, state_next = state_next, state
    """

    def __init__(
        self,
        model: Model,
        *,
        size: tuple[float, float] = (10.0, 10.0),
        spacing: float = 0.05,
        xform: wp.transform | None = None,
        heightfield: Heightfield | None = None,
        soil: SCMSoilParameters | None = None,
        test_offset_up: float = 0.1,
        test_offset_down: float = 0.5,
        contact_patches: bool | None = None,
        ccl_iterations: int = 8,
        bulldozing: bool = False,
        erosion_angle: float = 40.0,
        flow_factor: float = 1.2,
        erosion_iterations: int = 3,
        erosion_propagations: int = 10,
        damping_unilateral: bool = False,
        plot_field: SCMPlotField = SCMPlotField.NONE,
        plot_range: tuple[float, float] = (0.0, 0.2),
        auto_range: bool = False,
        range_decay: float = 2.0,
        colormap: Colormap = Colormap.VIRIDIS,
        base_color: tuple[float, float, float] = (0.62, 0.52, 0.40),
    ):
        """Create an SCM terrain patch.

        Args:
            model: Model supplying shape geometry and body poses. Which shapes
                interact is set by :meth:`add_newton_model`, :meth:`add_body` or
                :meth:`add_shape`; no model-wide BVH is required.
            size: Patch extent along the terrain frame's X and Y [m].
            spacing: Requested grid spacing [m]. Adjusted so the patch spans a
                whole number of cells.
            xform: Terrain frame. Deformation occurs along its +Z axis. Defaults
                to identity.
            heightfield: Optional undeformed profile. ``None`` gives a flat patch.
            soil: Soil parameters, built with
                :func:`~newton.terramechanics.scm_soil_parameters`. Defaults to
                that function's own defaults.
            test_offset_up: How far above a node its ray ends [m].
            test_offset_down: How far below a node its ray starts [m].
            contact_patches: Compute connected contact patches to evaluate the
                Bekker cohesive term ``Kc/b``. ``None`` enables it when
                ``soil.bekker_kc > 0``. Construction-time only.
            ccl_iterations: Label-propagation passes for patch labeling; must
                cover ``log2`` of the widest expected footprint in cells.
                Construction-time only.
            bulldozing: Enable lateral material displacement and erosion.
                Construction-time only: it changes the kernel launch sequence
                and would invalidate an enclosing CUDA graph.
            erosion_angle: Angle of repose for displaced material [deg].
            flow_factor: Lateral volume raised per unit volume displaced.
            erosion_iterations: Red-black relaxation sweeps per step.
                Construction-time only.
            erosion_propagations: Dilation rings of the erosion domain.
                Construction-time only.
            damping_unilateral: Suppress viscous damping while a node is
                separating. The reference implementation damps in both
                directions, which lets the soil pull on a departing body.
            plot_field: Scalar field encoded in :attr:`colors`.
            plot_range: ``(vmin, vmax)`` for color mapping. Ignored when
                ``auto_range`` is set.
            auto_range: Rescale ``plot_range`` each frame from the touched nodes.
            range_decay: Rate [1/s] at which an auto range shrinks back after a
                peak passes. Larger is more responsive and more flickery.
            colormap: Colormap applied to ``plot_field``.
            base_color: RGB used when ``plot_field`` is
                :attr:`SCMPlotField.NONE`.

        Raises:
            ValueError: If ``size`` or ``spacing`` is non-positive, or the
                heightfield resolution does not match the grid.
        """
        if size[0] <= 0.0 or size[1] <= 0.0:
            raise ValueError(f"size must be positive, got {size}")
        if spacing <= 0.0:
            raise ValueError(f"spacing must be positive, got {spacing}")

        self.model = model
        self.device = model.device

        # Snap the grid to a whole number of cells; the requested spacing is a
        # target, and the X extent is authoritative.
        self.nx = max(int(math.ceil(size[0] / spacing)), 1) + 1
        self.ny = max(int(math.ceil(size[1] / spacing)), 1) + 1
        self.delta = size[0] / (self.nx - 1)
        self.cell_area = self.delta * self.delta
        self.size = (self.delta * (self.nx - 1), self.delta * (self.ny - 1))
        self.node_count = self.nx * self.ny

        self.xform = wp.transform_identity() if xform is None else wp.transform(*xform)
        self._xform_inv = wp.transform_inverse(self.xform)

        self.test_offset_up = float(test_offset_up)
        self.test_offset_down = float(test_offset_down)
        self.damping_unilateral = int(bool(damping_unilateral))

        soil = scm_soil_parameters() if soil is None else soil
        self._soil = wp.array([soil], dtype=SCMSoilParameters, device=self.device)
        self._soil_stride = 0

        self.contact_patches = bool(soil.bekker_kc > 0.0) if contact_patches is None else bool(contact_patches)
        self.ccl_iterations = int(ccl_iterations)

        self.bulldozing = bool(bulldozing)
        self.erosion_iterations = int(erosion_iterations)
        self.erosion_propagations = int(erosion_propagations)
        self._bulldoze_params = wp.array(
            [math.tan(math.radians(erosion_angle)), float(flow_factor)],
            dtype=float,
            device=self.device,
        )

        self.plot_field = SCMPlotField(plot_field)
        self.colormap = Colormap(colormap)
        self.auto_range = bool(auto_range)
        self.range_decay = float(range_decay)
        self.base_color = wp.vec3(*base_color)
        # [display_min, display_max, measured_min, measured_max]
        self._plot_range = wp.array(
            [plot_range[0], plot_range[1], plot_range[0], plot_range[1]],
            dtype=float,
            device=self.device,
        )

        # --- node state -----------------------------------------------------
        self.nodes = wp.zeros(self.node_count, dtype=SCMNodeState, device=self.device)
        heights, use_heights = self._resolve_heights(heightfield)
        wp.launch(
            K.scm_init_nodes,
            dim=self.node_count,
            inputs=[self.nx, self.ny, self.delta, heights, use_heights],
            outputs=[self.nodes],
            device=self.device,
        )
        if use_heights:
            wp.launch(
                K.scm_init_normals,
                dim=self.node_count,
                inputs=[self.nx, self.ny, self.delta],
                outputs=[self.nodes],
                device=self.device,
            )

        # --- visualization buffers ------------------------------------------
        self.vertices = wp.zeros(self.node_count, dtype=wp.vec3, device=self.device)
        self.normals = wp.zeros(self.node_count, dtype=wp.vec3, device=self.device)
        self.colors = wp.zeros(self.node_count, dtype=wp.vec3, device=self.device)
        tri_count = 2 * (self.nx - 1) * (self.ny - 1)
        self.indices = wp.zeros(3 * tri_count, dtype=wp.int32, device=self.device)
        wp.launch(
            K.scm_build_mesh_topology,
            dim=(self.nx - 1) * (self.ny - 1),
            inputs=[self.nx, self.ny],
            outputs=[self.indices],
            device=self.device,
        )

        # --- diagnostics ----------------------------------------------------
        self.counters = wp.zeros(2, dtype=wp.int32, device=self.device)
        self._bounds = wp.zeros(4, dtype=float, device=self.device)

        # The soil wrench is computed once per frame but the solver clears
        # State.body_f every substep, so the terrain owns its own accumulator
        # and callers re-apply it with apply_forces().
        self.body_f = wp.zeros(max(model.body_count, 1), dtype=wp.spatial_vector, device=self.device)

        # --- contact patches --------------------------------------------------
        if self.contact_patches:
            self.labels = wp.zeros(self.node_count, dtype=wp.int32, device=self.device)
            self._labels_alt = wp.zeros(self.node_count, dtype=wp.int32, device=self.device)
            self.patch_area = wp.zeros(self.node_count, dtype=float, device=self.device)
            self.patch_perimeter = wp.zeros(self.node_count, dtype=float, device=self.device)
            self.patch_flow = wp.zeros(self.node_count, dtype=float, device=self.device)
            self.patch_boundary = wp.zeros(self.node_count, dtype=wp.int32, device=self.device)
            self.patch_oob = wp.zeros(self.node_count, dtype=float, device=self.device)
        else:
            self.labels = None
            self.patch_oob = None

        if self.bulldozing:
            self._erosion = wp.zeros(self.node_count, dtype=wp.int32, device=self.device)
            self._erosion_alt = wp.zeros(self.node_count, dtype=wp.int32, device=self.device)
            if not self.contact_patches:
                # Bulldozing needs per-patch flow sums, so it needs labels even
                # when the cohesive term is inactive.
                self.labels = wp.zeros(self.node_count, dtype=wp.int32, device=self.device)
                self._labels_alt = wp.zeros(self.node_count, dtype=wp.int32, device=self.device)
                self.patch_area = wp.zeros(self.node_count, dtype=float, device=self.device)
                self.patch_perimeter = wp.zeros(self.node_count, dtype=float, device=self.device)
                self.patch_flow = wp.zeros(self.node_count, dtype=float, device=self.device)
                self.patch_boundary = wp.zeros(self.node_count, dtype=wp.int32, device=self.device)
                self.patch_oob = wp.zeros(self.node_count, dtype=float, device=self.device)

        # --- shape registration ----------------------------------------------
        self._registered: list[int] = []
        self._finalized = False
        self._bvh = None
        self._bvh_root = 0

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _resolve_heights(self, heightfield: Heightfield | None) -> tuple[wp.array, int]:
        """Resample an optional heightfield onto the grid.

        Returns a ``[node_count]`` height array and a flag indicating whether it
        should be used. A dummy single-element array is returned for flat
        patches so the kernel signature stays uniform.
        """
        if heightfield is None:
            return wp.zeros(1, dtype=float, device=self.device), 0

        data = np.asarray(heightfield._data, dtype=np.float32)
        z = heightfield.min_z + data * (heightfield.max_z - heightfield.min_z)

        # Heightfield rows follow ij-indexing over [-hx, hx] x [-hy, hy]; sample
        # it at the grid nodes with bilinear interpolation.
        gx = np.linspace(0.0, heightfield.nrow - 1, self.nx)
        gy = np.linspace(0.0, heightfield.ncol - 1, self.ny)
        i0 = np.clip(np.floor(gx).astype(int), 0, heightfield.nrow - 1)
        i1 = np.clip(i0 + 1, 0, heightfield.nrow - 1)
        j0 = np.clip(np.floor(gy).astype(int), 0, heightfield.ncol - 1)
        j1 = np.clip(j0 + 1, 0, heightfield.ncol - 1)
        fx = (gx - i0)[:, None]
        fy = (gy - j0)[None, :]

        z00 = z[np.ix_(i0, j0)]
        z10 = z[np.ix_(i1, j0)]
        z01 = z[np.ix_(i0, j1)]
        z11 = z[np.ix_(i1, j1)]
        sampled = (1 - fx) * (1 - fy) * z00 + fx * (1 - fy) * z10 + (1 - fx) * fy * z01 + fx * fy * z11

        # Node k = i + nx * j, so transpose from (nx, ny) to row-major (ny, nx).
        return wp.array(sampled.T.reshape(-1).astype(np.float32), dtype=float, device=self.device), 1

    def add_shape(self, shape: int) -> int:
        """Register one shape as interacting with the terrain.

        Args:
            shape: Shape index into the model.

        Returns:
            ``1`` if newly registered, ``0`` if already present.

        Raises:
            RuntimeError: If called after the first :meth:`step`.
            IndexError: If ``shape`` is out of range.
        """
        self._require_mutable()
        if not 0 <= shape < self.model.shape_count:
            raise IndexError(f"shape index {shape} out of range for {self.model.shape_count} shapes")
        if shape in self._registered:
            return 0
        self._registered.append(shape)
        return 1

    def add_body(self, body: int) -> int:
        """Register every shape attached to ``body``.

        Args:
            body: Body index into the model.

        Returns:
            Number of shapes newly registered.

        Raises:
            RuntimeError: If called after the first :meth:`step`.
        """
        self._require_mutable()
        shape_body = self.model.shape_body.numpy()
        added = 0
        for shape in np.nonzero(shape_body == body)[0]:
            added += self.add_shape(int(shape))
        return added

    def add_newton_model(self, model: Model | None = None, *, include_static: bool = False) -> int:
        """Register every eligible shape in a Newton model.

        The convenience path when the whole model should interact with the soil.
        Prefer :meth:`add_body` when the model has parts that never touch soil:
        a smaller shape set means a shallower BVH and fewer primitive tests per
        ray.

        Shapes flagged :attr:`~newton.ShapeFlags.SITE` are always skipped.

        Args:
            model: Model to register. Defaults to the model this terrain was
                constructed with.
            include_static: Also register world-attached shapes. Off by default
                because a ground plane would be hit by every ray in the patch.
                Static shapes deform the soil but receive no reaction, since
                forces are applied through :attr:`newton.State.body_f`.

        Returns:
            Number of shapes newly registered.

        Raises:
            RuntimeError: If called after the first :meth:`step`.
            ValueError: If ``model`` is not the model this terrain was built on.
        """
        self._require_mutable()
        if model is not None and model is not self.model:
            raise ValueError("SCMTerrain can only register shapes from the model it was constructed with")

        shape_body = self.model.shape_body.numpy()
        shape_flags = self.model.shape_flags.numpy()
        added = 0
        for shape in range(self.model.shape_count):
            if shape_flags[shape] & int(ShapeFlags.SITE):
                continue
            if not include_static and shape_body[shape] < 0:
                continue
            added += self.add_shape(shape)
        return added

    def finalize(self, state: State) -> None:
        """Build the terrain's shape BVH. Called lazily by the first :meth:`step`.

        Args:
            state: State supplying the body poses to build the initial BVH from.

        Raises:
            RuntimeError: If no shapes have been registered.
        """
        if self._finalized:
            return
        if not self._registered:
            raise RuntimeError(
                "SCMTerrain has no registered shapes. Call add_newton_model(), add_body() or "
                "add_shape() before stepping."
            )

        from ...geometry.bvh import compute_bvh_group_roots, compute_shape_local_bounds  # noqa: PLC0415

        model = self.model
        n = len(self._registered)

        self._shape_index = wp.array(np.asarray(self._registered, dtype=np.uint32), dtype=wp.uint32, device=self.device)
        self._shape_index_i32 = wp.array(
            np.asarray(self._registered, dtype=np.int32), dtype=wp.int32, device=self.device
        )

        self._shape_local_bounds = wp.empty((model.shape_count, 2), dtype=wp.vec3f, ndim=2, device=self.device)
        wp.launch(
            compute_shape_local_bounds,
            dim=model.shape_count,
            inputs=[model.shape_type, model.shape_source_ptr, model.gaussians_data, self._shape_local_bounds],
            device=self.device,
        )

        self._shape_world_transform = wp.empty(model.shape_count, dtype=wp.transformf, device=self.device)
        # Far-away degenerate boxes, so any slot the bounds kernel skips (a
        # shape outside the world range) can never be hit by a ray.
        far = np.full((n, 3), 1.0e20, dtype=np.float32)
        self._bvh_lowers = wp.array(far, dtype=wp.vec3f, device=self.device)
        self._bvh_uppers = wp.array(far, dtype=wp.vec3f, device=self.device)
        self._bvh_groups = wp.zeros(n, dtype=wp.int32, device=self.device)

        self._refit_bounds(state)

        self._bvh = wp.Bvh(self._bvh_lowers, self._bvh_uppers)
        roots = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(compute_bvh_group_roots, dim=1, inputs=[self._bvh.id, roots], device=self.device)
        self._bvh_root = int(roots.numpy()[0])

        self._finalized = True

    def _require_mutable(self) -> None:
        if self._finalized:
            raise RuntimeError(
                "SCMTerrain shape registration is construction-time only: it changes kernel launch "
                "dimensions, which an enclosing CUDA graph bakes in. Register shapes before the first step()."
            )

    def _refit_bounds(self, state: State) -> None:
        """Recompute world transforms and BVH bounds for the registered shapes."""
        from ...geometry.bvh import compute_shape_bvh_bounds, compute_shape_world_transforms  # noqa: PLC0415

        model = self.model
        wp.launch(
            compute_shape_world_transforms,
            dim=model.shape_count,
            inputs=[state.body_q, model.shape_body, model.shape_transform, self._shape_world_transform],
            device=self.device,
        )
        wp.launch(
            compute_shape_bvh_bounds,
            dim=len(self._registered),
            inputs=[
                len(self._registered),
                model.world_count + 1,
                model.shape_world,
                self._shape_index,
                model.shape_type,
                model.shape_scale,
                self._shape_world_transform,
                self._shape_local_bounds,
                self._bvh_lowers,
                self._bvh_uppers,
                self._bvh_groups,
            ],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------

    def step(self, state: State, dt: float) -> None:
        """Advance the soil state by ``dt`` and compute the soil wrench.

        Refits the terrain's own shape BVH, casts one ray per grid node, updates
        per-node soil state, and writes the resulting wrench into
        :attr:`body_f`.

        Call **once per substep**, passing the substep timestep. The soil model
        carries history, so it is advanced exactly once per integration step.

        The wrench is not applied here; follow with :meth:`apply_forces` so it
        survives :meth:`newton.State.clear_forces`.

        Args:
            state: State providing body poses. Not modified.
            dt: Substep timestep [s].
        """
        self.finalize(state)

        self._refit_bounds(state)
        self._bvh.refit()

        self.counters.zero_()
        self.body_f.zero_()

        wp.launch(K.scm_reset_bounds, dim=1, inputs=[self._bounds], device=self.device)
        wp.launch(
            K.scm_accumulate_bounds,
            dim=len(self._registered),
            inputs=[self._bvh_lowers, self._bvh_uppers, self._xform_inv, self._bounds],
            device=self.device,
        )

        ray_args = [
            self.nx,
            self.ny,
            self.delta,
            self.xform,
            self._bounds,
            self.test_offset_up,
            self.test_offset_down,
            self._bvh.id,
            self._bvh_root,
            self._shape_index_i32,
            self.model.shape_body,
            self.model.shape_type,
            self.model.shape_scale,
            self.model.shape_source_ptr,
            self._shape_world_transform,
        ]

        if self.labels is None:
            wp.launch(
                K.scm_raycast_and_update,
                dim=self.node_count,
                inputs=[
                    self.nx,
                    self.ny,
                    self.delta,
                    self.cell_area,
                    dt,
                    self.xform,
                    self._bounds,
                    self.test_offset_up,
                    self.test_offset_down,
                    self._bvh.id,
                    self._bvh_root,
                    self._shape_index_i32,
                    self.model.shape_body,
                    self.model.shape_type,
                    self.model.shape_scale,
                    self.model.shape_source_ptr,
                    self._shape_world_transform,
                    state.body_q,
                    state.body_qd,
                    self.model.body_com,
                    self._soil,
                    self._soil_stride,
                    self.damping_unilateral,
                    self.nodes,
                    self.body_f,
                    self.counters,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                K.scm_raycast,
                dim=self.node_count,
                inputs=[*ray_args, self.nodes, self.counters],
                device=self.device,
            )
            self._label_patches()
            wp.launch(
                K.scm_constitutive,
                dim=self.node_count,
                inputs=[
                    self.nx,
                    self.ny,
                    self.delta,
                    self.cell_area,
                    dt,
                    self.xform,
                    self.labels,
                    self.patch_oob,
                    state.body_q,
                    state.body_qd,
                    self.model.body_com,
                    self._soil,
                    self._soil_stride,
                    self.damping_unilateral,
                    self.nodes,
                    self.body_f,
                ],
                device=self.device,
            )

        if self.bulldozing:
            self._apply_bulldozing(dt)

    def apply_forces(self, state: State) -> None:
        """Add the soil wrench computed by :meth:`step` into ``state.body_f``.

        :meth:`newton.State.clear_forces` runs every substep while the soil
        model runs once per frame, so the wrench must be re-applied after each
        clear. This is a single kernel launch and is safe inside a captured
        CUDA graph.

        Args:
            state: State to accumulate into.
        """
        wp.launch(
            _accumulate_body_f,
            dim=len(self.body_f),
            inputs=[self.body_f],
            outputs=[state.body_f],
            device=self.device,
        )

    def _label_patches(self) -> None:
        """Label connected contact patches and reduce their area and perimeter."""
        wp.launch(K.scm_ccl_init, dim=self.node_count, inputs=[self.nodes, self.labels], device=self.device)

        # Alternating neighbor propagation with pointer jumping (parallel
        # union-find path compression) collapses a component of diameter d in
        # O(log d) rounds instead of O(d).
        src, dst = self.labels, self._labels_alt
        for _ in range(self.ccl_iterations):
            wp.launch(
                K.scm_ccl_propagate,
                dim=self.node_count,
                inputs=[self.nx, self.ny, src, dst],
                device=self.device,
            )
            src, dst = dst, src
            wp.launch(K.scm_ccl_jump, dim=self.node_count, inputs=[src, dst], device=self.device)
            src, dst = dst, src
        if src is not self.labels:
            wp.copy(self.labels, src)

        wp.launch(
            K.scm_patch_clear,
            dim=self.node_count,
            inputs=[self.patch_area, self.patch_perimeter, self.patch_flow, self.patch_boundary, self.patch_oob],
            device=self.device,
        )
        wp.launch(
            K.scm_patch_reduce,
            dim=self.node_count,
            inputs=[self.nx, self.ny, self.delta, self.labels, self.patch_area, self.patch_perimeter],
            device=self.device,
        )
        wp.launch(
            K.scm_patch_finalize,
            dim=self.node_count,
            inputs=[self.patch_area, self.patch_perimeter, self.patch_oob],
            device=self.device,
        )

    def _apply_bulldozing(self, dt: float) -> None:
        """Displace material to the rut sides and relax it to the angle of repose."""
        wp.launch(
            K.scm_bulldoze_accumulate,
            dim=self.node_count,
            inputs=[self.nx, self.ny, self.labels, self.nodes, self.patch_flow, self.patch_boundary],
            device=self.device,
        )
        wp.launch(
            K.scm_bulldoze_raise,
            dim=self.node_count,
            inputs=[
                self.nx,
                self.ny,
                dt,
                self._bulldoze_params,
                self.labels,
                self.patch_flow,
                self.patch_boundary,
                self.nodes,
            ],
            device=self.device,
        )

        self._seed_erosion()
        src, dst = self._erosion, self._erosion_alt
        for _ in range(self.erosion_propagations):
            wp.launch(
                K.scm_erosion_dilate,
                dim=self.node_count,
                inputs=[self.nx, self.ny, self.nodes, src, dst],
                device=self.device,
            )
            src, dst = dst, src
        if src is not self._erosion:
            wp.copy(self._erosion, src)

        for _ in range(self.erosion_iterations):
            for parity in (0, 1):
                wp.launch(
                    K.scm_erosion_relax,
                    dim=self.node_count,
                    inputs=[
                        self.nx,
                        self.ny,
                        self.delta,
                        self._bulldoze_params,
                        parity,
                        self._erosion,
                        self.nodes,
                    ],
                    device=self.device,
                )

    def _seed_erosion(self) -> None:
        """Copy the per-node erosion flags set by the boundary raise into the mask."""
        wp.launch(
            _seed_erosion_kernel,
            dim=self.node_count,
            inputs=[self.nodes, self._erosion],
            device=self.device,
        )

    def update_visualization(self, dt: float = 1.0 / 60.0) -> None:
        """Refresh :attr:`vertices`, :attr:`normals` and :attr:`colors`.

        Kept separate from :meth:`step` so headless and training runs can skip
        it, and so the physics graph can be captured once while visuals refresh
        at a lower rate.

        Args:
            dt: Time since the previous refresh [s]. Only affects how fast an
                auto range decays.
        """
        if self.auto_range and self.plot_field != SCMPlotField.NONE:
            wp.launch(K.scm_reset_plot_range, dim=1, inputs=[self._plot_range], device=self.device)
            wp.launch(
                K.scm_reduce_plot_range,
                dim=self.node_count,
                inputs=[int(self.plot_field), self.nodes, self._plot_range],
                device=self.device,
            )
            wp.launch(
                K.scm_smooth_plot_range,
                dim=1,
                inputs=[self.range_decay, dt, self._plot_range],
                device=self.device,
            )

        wp.launch(
            K.scm_update_mesh,
            dim=self.node_count,
            inputs=[
                self.nx,
                self.ny,
                self.delta,
                self.xform,
                int(self.plot_field),
                int(self.colormap),
                self._plot_range,
                self.base_color,
            ],
            outputs=[self.nodes, self.vertices, self.normals, self.colors],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------

    def set_soil_parameters(self, soil: SCMSoilParameters) -> None:
        """Replace the uniform soil parameters.

        Written in place into a device array, so this takes effect without
        re-capturing an enclosing CUDA graph.

        Args:
            soil: New parameters.
        """
        if self._soil_stride != 0:
            raise RuntimeError("terrain uses per-node soil parameters; use set_soil_parameters_field()")
        if soil.bekker_kc > 0.0 and not self.contact_patches:
            import warnings  # noqa: PLC0415

            warnings.warn(
                "bekker_kc > 0 but contact_patches is disabled, so the Kc/b term will be ignored. "
                "Construct SCMTerrain with contact_patches=True.",
                stacklevel=2,
            )
        self._soil.assign(wp.array([soil], dtype=SCMSoilParameters, device=self.device))

    def set_soil_parameters_field(self, soil: wp.array[Any]) -> None:
        """Set spatially varying soil parameters.

        Args:
            soil: Array of :class:`SCMSoilParameters`, shape ``[node_count]``,
                indexed as ``i + nx * j``.

        Raises:
            ValueError: If the array length does not match the grid.
        """
        if len(soil) != self.node_count:
            raise ValueError(f"expected {self.node_count} soil entries, got {len(soil)}")
        self._soil = soil
        self._soil_stride = 1

    def set_bulldozing_parameters(self, erosion_angle: float | None = None, flow_factor: float | None = None) -> None:
        """Set the continuous bulldozing coefficients.

        Only coefficients that do not affect the launch sequence are settable;
        ``bulldozing``, ``erosion_iterations`` and ``erosion_propagations`` are
        construction-time only.

        Args:
            erosion_angle: Angle of repose for displaced material [deg].
            flow_factor: Lateral volume raised per unit volume displaced.
        """
        host = self._bulldoze_params.numpy()
        if erosion_angle is not None:
            host[0] = math.tan(math.radians(erosion_angle))
        if flow_factor is not None:
            host[1] = float(flow_factor)
        self._bulldoze_params.assign(host)

    def set_plot(
        self,
        field: SCMPlotField | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        colormap: Colormap | None = None,
    ) -> None:
        """Update the visualization mapping.

        Args:
            field: Scalar field to encode in :attr:`colors`.
            vmin: Lower end of the color range.
            vmax: Upper end of the color range.
            colormap: Colormap to apply.
        """
        if field is not None:
            self.plot_field = SCMPlotField(field)
        if colormap is not None:
            self.colormap = Colormap(colormap)
        if vmin is not None or vmax is not None:
            host = self._plot_range.numpy()
            if vmin is not None:
                host[0] = float(vmin)
            if vmax is not None:
                host[1] = float(vmax)
            self._plot_range.assign(host)

    def set_auto_range(self, enabled: bool, decay: float | None = None) -> None:
        """Enable or disable per-frame rescaling of the color range.

        Args:
            enabled: Whether to rescale from the touched nodes each frame.
            decay: Rate [1/s] at which the range shrinks back after a peak.
        """
        self.auto_range = bool(enabled)
        if decay is not None:
            self.range_decay = float(decay)

    @property
    def plot_range(self) -> tuple[float, float]:
        """Current color range as ``(vmin, vmax)``.

        Reading this synchronizes with the device; do not call it inside a
        captured region.
        """
        host = self._plot_range.numpy()
        return float(host[0]), float(host[1])

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def get_height(self, points: wp.array[wp.vec3], out: wp.array[float], *, undeformed: bool = False) -> None:
        """Write the terrain height below each query point into ``out``.

        Uses nearest-node lookup, so the result is piecewise constant over cells.

        Args:
            points: Query positions in world space [m].
            out: Output heights [m], same length as ``points``.
            undeformed: Sample the initial profile instead of the deformed one.
        """
        wp.launch(
            K.scm_query_height,
            dim=len(points),
            inputs=[
                self.nx,
                self.ny,
                self.delta,
                self._xform_inv,
                self.xform,
                int(undeformed),
                self.nodes,
                points,
            ],
            outputs=[out],
            device=self.device,
        )

    def get_normal(self, points: wp.array[wp.vec3], out: wp.array[wp.vec3]) -> None:
        """Write the deformed terrain normal below each query point into ``out``.

        Args:
            points: Query positions in world space [m].
            out: Output unit normals in world space, same length as ``points``.
        """
        wp.launch(
            K.scm_query_normal,
            dim=len(points),
            inputs=[self.nx, self.ny, self.delta, self._xform_inv, self.xform, self.nodes, points],
            outputs=[out],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # gui
    # ------------------------------------------------------------------

    def draw_ui(self, ui: Any) -> None:
        """Draw the SCM controls into an ImGui panel.

        Intended to be called from an example's ``gui()`` method, which the
        example runner routes into the viewer's side panel.

        Args:
            ui: Active ImGui context, as passed to ``gui(ui)``.
        """
        from .gui import draw_scm_ui  # noqa: PLC0415

        draw_scm_ui(self, ui)

    def attach_ui(self, viewer: Any, position: str = "side") -> bool:
        """Register :meth:`draw_ui` with a viewer directly.

        For scripts that do not go through the example runner.

        Args:
            viewer: Viewer to register with.
            position: Callback slot; see ``ViewerGL.register_ui_callback``.

        Returns:
            ``False`` if the viewer has no ``register_ui_callback``, so callers
            need no backend check of their own.
        """
        register = getattr(viewer, "register_ui_callback", None)
        if register is None:
            return False
        register(self.draw_ui, position=position)
        return True


@wp.kernel(enable_backward=False)
def _seed_erosion_kernel(nodes: wp.array[SCMNodeState], erosion: wp.array[wp.int32]):
    """Copy per-node erosion flags into the standalone dilation mask."""
    k = wp.tid()
    erosion[k] = nodes[k].erosion


@wp.kernel(enable_backward=False)
def _accumulate_body_f(src: wp.array[wp.spatial_vector], dst: wp.array[wp.spatial_vector]):
    """Add the terrain's per-body wrench into a state force array."""
    b = wp.tid()
    dst[b] = dst[b] + src[b]
