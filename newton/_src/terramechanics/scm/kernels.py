# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp

from ...geometry.raycast import map_ray_to_local, ray_intersect_mesh_no_normal, ray_intersect_shape_no_normal
from ...geometry.types import GeoType
from ...viewer.colormap import sample_colormap
from .types import SCMNodeState, SCMPlotField, SCMSoilParameters

# Sentinel stored in ``hit_level`` when a node's ray missed everything.
NO_HIT = 1.0e9

# Minimum tangential speed [m/s] below which the shear direction is ill-defined.
_MIN_TANGENT_SPEED = 1.0e-8

# Contact patches smaller than this area [m^2] get ``oob = 0``, matching Chrono.
_MIN_PATCH_AREA = 1.0e-6


# ---------------------------------------------------------------------------
# grid helpers
# ---------------------------------------------------------------------------


@wp.func
def node_local_xy(i: int, j: int, nx: int, ny: int, delta: float) -> wp.vec2:
    """Return the terrain-frame (x, y) of grid node ``(i, j)``.

    The grid is centered on the terrain frame origin, matching the symmetric
    index range used by the reference implementation.
    """
    x = (float(i) - 0.5 * float(nx - 1)) * delta
    y = (float(j) - 0.5 * float(ny - 1)) * delta
    return wp.vec2(x, y)


@wp.func
def clamp_index(i: int, n: int) -> int:
    """Clamp a grid index into ``[0, n)``.

    Edge clamping is what makes the patch behave as if it extended to infinity:
    a query outside the initialized area returns the closest initialized node.
    """
    return wp.clamp(i, 0, n - 1)


@wp.func
def node_index(i: int, j: int, nx: int, ny: int) -> int:
    """Return the linear index of grid node ``(i, j)``, with edge clamping."""
    return clamp_index(i, nx) + nx * clamp_index(j, ny)


# ---------------------------------------------------------------------------
# initialization
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def scm_init_nodes(
    nx: int,
    ny: int,
    delta: float,
    heights: wp.array[float],
    use_heights: int,
    nodes: wp.array[SCMNodeState],
):
    """Seed node state from the undeformed profile.

    Args:
        nx: Grid node count along terrain X.
        ny: Grid node count along terrain Y.
        delta: Grid spacing [m].
        heights: Initial heights [m], shape ``[nx * ny]``. Ignored when
            ``use_heights`` is zero.
        use_heights: Nonzero to read ``heights``; zero for a flat patch.
        nodes: Node state to initialize, shape ``[nx * ny]``.
    """
    k = wp.tid()

    h = float(0.0)
    if use_heights != 0:
        h = heights[k]

    nodes[k].level = h
    nodes[k].level_initial = h
    nodes[k].level_undeformed = h
    nodes[k].normal_z = 1.0
    nodes[k].sinkage = 0.0
    nodes[k].sinkage_plastic = 0.0
    nodes[k].sinkage_elastic = 0.0
    nodes[k].sigma = 0.0
    nodes[k].sigma_yield = 0.0
    nodes[k].kshear = 0.0
    nodes[k].tau = 0.0
    nodes[k].hit_level = NO_HIT
    nodes[k].step_plastic_flow = 0.0
    nodes[k].massremainder = 0.0
    nodes[k].hit_shape = -1
    nodes[k].hit_body = -1
    nodes[k].erosion = 0


@wp.kernel(enable_backward=False)
def scm_init_normals(
    nx: int,
    ny: int,
    delta: float,
    nodes: wp.array[SCMNodeState],
):
    """Compute per-node surface normals from the undeformed profile.

    Stores only the Z component, which is the cosine between the local normal
    and the terrain frame's +Z axis, since that is all the constitutive update
    needs. Runs after :func:`scm_init_nodes` so heights are populated.
    """
    k = wp.tid()
    i = k % nx
    j = k / nx

    h_e = nodes[node_index(i + 1, j, nx, ny)].level_undeformed
    h_w = nodes[node_index(i - 1, j, nx, ny)].level_undeformed
    h_n = nodes[node_index(i, j + 1, nx, ny)].level_undeformed
    h_s = nodes[node_index(i, j - 1, nx, ny)].level_undeformed

    n = wp.normalize(wp.vec3(h_w - h_e, h_s - h_n, 2.0 * delta))
    nodes[k].normal_z = n[2]


@wp.kernel(enable_backward=False)
def scm_build_mesh_topology(
    nx: int,
    ny: int,
    indices: wp.array[wp.int32],
):
    """Fill the static triangle index buffer for the visualization mesh.

    Two counter-clockwise triangles per grid cell, ``2 * (nx-1) * (ny-1)``
    triangles in total.
    """
    c = wp.tid()
    cx = c % (nx - 1)
    cy = c / (nx - 1)

    v0 = cx + nx * cy
    v1 = v0 + 1
    v2 = v0 + nx + 1
    v3 = v0 + nx

    o = 6 * c
    indices[o + 0] = v0
    indices[o + 1] = v1
    indices[o + 2] = v2
    indices[o + 3] = v0
    indices[o + 4] = v2
    indices[o + 5] = v3


# ---------------------------------------------------------------------------
# active region
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def scm_reset_bounds(bounds: wp.array[float]):
    """Reset the active-region AABB accumulator to an inverted box."""
    bounds[0] = 1.0e30
    bounds[1] = 1.0e30
    bounds[2] = -1.0e30
    bounds[3] = -1.0e30


@wp.kernel(enable_backward=False)
def scm_accumulate_bounds(
    lowers: wp.array[wp.vec3],
    uppers: wp.array[wp.vec3],
    xform_inv: wp.transform,
    bounds: wp.array[float],
):
    """Reduce registered-shape world AABBs into a terrain-frame XY box.

    Rays are only generated for grid nodes inside this box, which keeps the cost
    proportional to the contact region rather than the patch size.

    Args:
        lowers: Per-shape world AABB minima, shape ``[shape_count]``.
        uppers: Per-shape world AABB maxima, shape ``[shape_count]``.
        xform_inv: World-to-terrain transform.
        bounds: Output ``[min_x, min_y, max_x, max_y]`` in terrain frame.
    """
    s = wp.tid()
    lo = lowers[s]
    hi = uppers[s]

    # Transform all eight corners: the terrain frame may be rotated relative to
    # world, so the AABB is not axis-aligned after mapping.
    for c in range(8):
        corner = wp.vec3(
            wp.where(c & 1, hi[0], lo[0]),
            wp.where(c & 2, hi[1], lo[1]),
            wp.where(c & 4, hi[2], lo[2]),
        )
        p = wp.transform_point(xform_inv, corner)
        wp.atomic_min(bounds, 0, p[0])
        wp.atomic_min(bounds, 1, p[1])
        wp.atomic_max(bounds, 2, p[0])
        wp.atomic_max(bounds, 3, p[1])


# ---------------------------------------------------------------------------
# ray casting
# ---------------------------------------------------------------------------


@wp.func
def scm_cast_node_ray(
    bvh_id: wp.uint64,
    bvh_root: int,
    shape_index: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_world_transform: wp.array[wp.transform],
    origin: wp.vec3,
    direction: wp.vec3,
    max_t: float,
):
    """Cast one upward ray against the registered shapes.

    Returns the nearest hit distance and shape index, or ``(max_t, -1)`` on a
    miss. Casting upward from below the surface means the hit is the lowest
    point of the penetrating object, which is what defines the rut.
    """
    best_t = max_t
    best_shape = wp.int32(-1)

    query = wp.bvh_query_ray(bvh_id, origin, direction, bvh_root)
    slot = wp.int32(0)
    while wp.bvh_query_next(query, slot, best_t):
        s = shape_index[slot]
        geom_type = shape_type[s]

        hit_t = float(-1.0)
        if geom_type == GeoType.MESH or geom_type == GeoType.CONVEX_MESH or geom_type == GeoType.HFIELD:
            xf = shape_world_transform[s]
            o_local, d_local = map_ray_to_local(xf, origin, direction, shape_scale[s])
            hit_t, _n, _u, _v, _f = ray_intersect_mesh_no_normal(
                o_local, d_local, shape_scale[s], shape_source_ptr[s], False, best_t
            )
        else:
            hit_t, _sn = ray_intersect_shape_no_normal(
                shape_world_transform[s], shape_scale[s], geom_type, origin, direction, False
            )

        if hit_t >= 0.0 and hit_t < best_t:
            best_t = hit_t
            best_shape = s

    return best_t, best_shape


@wp.func
def scm_node_ray_step(
    nodes: wp.array[SCMNodeState],
    k: int,
    i: int,
    j: int,
    nx: int,
    ny: int,
    delta: float,
    xform: wp.transform,
    bounds: wp.array[float],
    offset_up: float,
    offset_down: float,
    bvh_id: wp.uint64,
    bvh_root: int,
    shape_index: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_world_transform: wp.array[wp.transform],
    counters: wp.array[wp.int32],
) -> int:
    """Reset a node and cast its ray. Returns 1 on a hit, 0 otherwise."""
    # Per-step reset. Quantities not cleared here (sigma_yield, kshear,
    # sinkage_plastic, massremainder, level) are the soil's persistent memory.
    nodes[k].sigma = 0.0
    nodes[k].sinkage_elastic = 0.0
    nodes[k].step_plastic_flow = 0.0
    nodes[k].erosion = 0
    nodes[k].hit_level = NO_HIT
    nodes[k].hit_shape = -1
    nodes[k].hit_body = -1

    xy = node_local_xy(i, j, nx, ny, delta)
    if xy[0] < bounds[0] or xy[0] > bounds[2] or xy[1] < bounds[1] or xy[1] > bounds[3]:
        return 0

    wp.atomic_add(counters, 1, 1)

    level = nodes[k].level
    p_local = wp.vec3(xy[0], xy[1], level - offset_down)
    origin = wp.transform_point(xform, p_local)
    direction = wp.transform_vector(xform, wp.vec3(0.0, 0.0, 1.0))
    max_t = offset_down + offset_up

    hit_t, hit_shape = scm_cast_node_ray(
        bvh_id,
        bvh_root,
        shape_index,
        shape_type,
        shape_scale,
        shape_source_ptr,
        shape_world_transform,
        origin,
        direction,
        max_t,
    )

    if hit_shape < 0:
        return 0

    # The ray travels along the terrain +Z, so the hit height follows directly
    # from the start height plus the hit distance.
    nodes[k].hit_level = level - offset_down + hit_t
    nodes[k].hit_shape = hit_shape
    nodes[k].hit_body = shape_body[hit_shape]
    wp.atomic_add(counters, 0, 1)
    return 1


# ---------------------------------------------------------------------------
# constitutive model
# ---------------------------------------------------------------------------


@wp.func
def scm_apply_constitutive(
    nodes: wp.array[SCMNodeState],
    k: int,
    soil: wp.array[SCMSoilParameters],
    soil_index: int,
    oob: float,
    cell_area: float,
    dt: float,
    xform: wp.transform,
    xy: wp.vec2,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    damping_unilateral: int,
):
    """Advance one node's soil state and accumulate its wrench on the hit body.

    Implements the Bekker pressure-sinkage law with an elastic predictor and
    plastic corrector, viscous damping, and Mohr-Coulomb capacity mobilized by
    the Janosi-Hanamoto shear law.

    Args:
        nodes: Node state array.
        k: Index of the node to update.
        soil: Soil parameters, indexed by ``soil_index``.
        soil_index: Index into ``soil``; ``0`` for uniform soil.
        oob: Reciprocal footprint width ``1/b`` [1/m] for this node's contact
            patch. Zero disables the Bekker cohesive term.
        cell_area: Area represented by one grid node [m^2].
        dt: Timestep [s].
        xform: Terrain-to-world transform.
        xy: Terrain-frame (x, y) of this node [m].
        body_q: Body transforms.
        body_qd: Body spatial velocities.
        body_com: Body centers of mass, body frame.
        body_f: Body wrench accumulator, world frame about the COM.
        damping_unilateral: Nonzero to suppress damping while separating, which
            avoids the adhesive pull the reference implementation exhibits.
    """
    ca = nodes[k].normal_z
    if ca <= 0.0:
        return

    p = soil[soil_index]

    hit_level = nodes[k].hit_level
    level_initial = nodes[k].level_initial

    # Sinkage measured along the local surface normal.
    p_hit_offset = ca * (level_initial - hit_level)

    # Elastic predictor.
    sigma = p.elastic_k * (p_hit_offset - nodes[k].sinkage_plastic)
    if sigma < 0.0:
        # Unilateral contact: soil cannot pull.
        nodes[k].sigma = 0.0
        return

    body = nodes[k].hit_body

    # Velocity of the contacting body at this node, evaluated at the node's
    # pre-update level so the shear increment uses the incoming configuration.
    p_local = wp.vec3(xy[0], xy[1], nodes[k].level)
    p_world = wp.transform_point(xform, p_local)

    vel = wp.vec3(0.0)
    com_world = p_world
    if body >= 0:
        qd = body_qd[body]
        v_com = wp.spatial_top(qd)
        omega = wp.spatial_bottom(qd)
        com_world = wp.transform_point(body_q[body], body_com[body])
        vel = v_com + wp.cross(omega, p_world - com_world)

    normal_world = wp.transform_vector(xform, wp.vec3(0.0, 0.0, 1.0))
    v_n = wp.dot(vel, normal_world)
    v_t = vel - v_n * normal_world
    speed_t = wp.length(v_t)

    tangent = wp.vec3(0.0)
    if speed_t > _MIN_TANGENT_SPEED:
        tangent = -v_t / speed_t

    nodes[k].sinkage = p_hit_offset
    nodes[k].level = hit_level

    # Janosi-Hanamoto shear displacement is a monotone accumulator.
    nodes[k].kshear = nodes[k].kshear + speed_t * dt

    # Plastic corrector: the Bekker curve is the yield surface and the yield
    # pressure ratchets upward, so the soil remembers its peak compaction.
    if sigma > nodes[k].sigma_yield:
        sinkage = wp.max(p_hit_offset, 0.0)
        sigma = (oob * p.bekker_kc + p.bekker_kphi) * wp.pow(sinkage, p.bekker_n)
        nodes[k].sigma_yield = sigma
        plastic_old = nodes[k].sinkage_plastic
        nodes[k].sinkage_plastic = p_hit_offset - sigma / p.elastic_k
        nodes[k].step_plastic_flow = (nodes[k].sinkage_plastic - plastic_old) / dt

    nodes[k].sinkage_elastic = p_hit_offset - nodes[k].sinkage_plastic

    # Viscous damping, applied outside the yield clamp.
    if damping_unilateral == 0 or v_n < 0.0:
        sigma = sigma - v_n * p.damping_r

    if sigma < 0.0:
        sigma = 0.0

    nodes[k].sigma = sigma

    # Mohr-Coulomb capacity, mobilized by accumulated shear displacement.
    tau_max = p.mohr_cohesion + sigma * p.mohr_mu
    tau = tau_max * (1.0 - wp.exp(-nodes[k].kshear / p.janosi_shear))
    nodes[k].tau = tau

    force = normal_world * (cell_area * sigma) + tangent * (cell_area * tau)

    if body >= 0:
        moment = wp.cross(p_world - com_world, force)
        wp.atomic_add(body_f, body, wp.spatial_vector(force, moment))


# ---------------------------------------------------------------------------
# fused path (no contact patches)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def scm_raycast_and_update(
    nx: int,
    ny: int,
    delta: float,
    cell_area: float,
    dt: float,
    xform: wp.transform,
    bounds: wp.array[float],
    offset_up: float,
    offset_down: float,
    bvh_id: wp.uint64,
    bvh_root: int,
    shape_index: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_world_transform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    soil: wp.array[SCMSoilParameters],
    soil_stride: int,
    damping_unilateral: int,
    nodes: wp.array[SCMNodeState],
    body_f: wp.array[wp.spatial_vector],
    counters: wp.array[wp.int32],
):
    """Reset, ray cast and apply the soil model in one pass.

    Used when the Bekker cohesive modulus is zero, where the ``Kc/b`` term
    vanishes identically and no contact-patch information is needed.
    """
    k = wp.tid()
    i = k % nx
    j = k / nx

    hit = scm_node_ray_step(
        nodes,
        k,
        i,
        j,
        nx,
        ny,
        delta,
        xform,
        bounds,
        offset_up,
        offset_down,
        bvh_id,
        bvh_root,
        shape_index,
        shape_body,
        shape_type,
        shape_scale,
        shape_source_ptr,
        shape_world_transform,
        counters,
    )
    if hit == 0:
        return

    scm_apply_constitutive(
        nodes,
        k,
        soil,
        k * soil_stride,
        0.0,
        cell_area,
        dt,
        xform,
        node_local_xy(i, j, nx, ny, delta),
        body_q,
        body_qd,
        body_com,
        body_f,
        damping_unilateral,
    )


# ---------------------------------------------------------------------------
# split path (with contact patches)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def scm_raycast(
    nx: int,
    ny: int,
    delta: float,
    xform: wp.transform,
    bounds: wp.array[float],
    offset_up: float,
    offset_down: float,
    bvh_id: wp.uint64,
    bvh_root: int,
    shape_index: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_world_transform: wp.array[wp.transform],
    nodes: wp.array[SCMNodeState],
    counters: wp.array[wp.int32],
):
    """Reset and ray cast only, leaving the constitutive update to a later pass."""
    k = wp.tid()
    scm_node_ray_step(
        nodes,
        k,
        k % nx,
        k / nx,
        nx,
        ny,
        delta,
        xform,
        bounds,
        offset_up,
        offset_down,
        bvh_id,
        bvh_root,
        shape_index,
        shape_body,
        shape_type,
        shape_scale,
        shape_source_ptr,
        shape_world_transform,
        counters,
    )


@wp.kernel(enable_backward=False)
def scm_ccl_init(
    nodes: wp.array[SCMNodeState],
    labels: wp.array[wp.int32],
):
    """Seed connected-component labels: a hit node labels itself, misses get -1."""
    k = wp.tid()
    if nodes[k].hit_shape >= 0:
        labels[k] = k
    else:
        labels[k] = -1


@wp.kernel(enable_backward=False)
def scm_ccl_propagate(
    nx: int,
    ny: int,
    labels_in: wp.array[wp.int32],
    labels_out: wp.array[wp.int32],
):
    """One min-label propagation pass over 4-connected neighbors.

    Alone this converges in ``O(component diameter)`` passes; paired with
    :func:`scm_ccl_jump` it converges in ``O(log diameter)`` rounds, which is
    what makes a fixed iteration count practical. Testing for convergence would
    need a host readback.
    """
    k = wp.tid()
    label = labels_in[k]
    if label < 0:
        labels_out[k] = -1
        return

    i = k % nx
    j = k / nx

    best = label
    if i > 0:
        n = labels_in[(i - 1) + nx * j]
        if n >= 0:
            best = wp.min(best, n)
    if i < nx - 1:
        n = labels_in[(i + 1) + nx * j]
        if n >= 0:
            best = wp.min(best, n)
    if j > 0:
        n = labels_in[i + nx * (j - 1)]
        if n >= 0:
            best = wp.min(best, n)
    if j < ny - 1:
        n = labels_in[i + nx * (j + 1)]
        if n >= 0:
            best = wp.min(best, n)

    labels_out[k] = best


@wp.kernel(enable_backward=False)
def scm_ccl_jump(
    labels_in: wp.array[wp.int32],
    labels_out: wp.array[wp.int32],
):
    """One pointer-jumping pass: follow each node's label to its label's label.

    This is union-find path compression run in parallel. It halves the length of
    every label chain, so alternating it with :func:`scm_ccl_propagate` collapses
    a component of diameter ``d`` in ``O(log d)`` rounds rather than ``O(d)``.
    """
    k = wp.tid()
    label = labels_in[k]
    if label < 0:
        labels_out[k] = -1
        return
    labels_out[k] = labels_in[label]


@wp.kernel(enable_backward=False)
def scm_patch_clear(
    patch_area: wp.array[float],
    patch_perimeter: wp.array[float],
    patch_flow: wp.array[float],
    patch_boundary: wp.array[wp.int32],
    patch_oob: wp.array[float],
):
    """Zero the per-patch accumulators."""
    p = wp.tid()
    patch_area[p] = 0.0
    patch_perimeter[p] = 0.0
    patch_flow[p] = 0.0
    patch_boundary[p] = 0
    patch_oob[p] = 0.0


@wp.kernel(enable_backward=False)
def scm_patch_reduce(
    nx: int,
    ny: int,
    delta: float,
    labels: wp.array[wp.int32],
    patch_area: wp.array[float],
    patch_perimeter: wp.array[float],
):
    """Accumulate grid-native area and perimeter for each contact patch.

    On a regular grid both follow directly from node and boundary-edge counts,
    so no convex hull is needed.
    """
    k = wp.tid()
    label = labels[k]
    if label < 0:
        return

    wp.atomic_add(patch_area, label, delta * delta)

    i = k % nx
    j = k / nx
    edges = int(0)
    if i == 0 or labels[(i - 1) + nx * j] != label:
        edges += 1
    if i == nx - 1 or labels[(i + 1) + nx * j] != label:
        edges += 1
    if j == 0 or labels[i + nx * (j - 1)] != label:
        edges += 1
    if j == ny - 1 or labels[i + nx * (j + 1)] != label:
        edges += 1

    if edges > 0:
        wp.atomic_add(patch_perimeter, label, delta * float(edges))


@wp.kernel(enable_backward=False)
def scm_patch_finalize(
    patch_area: wp.array[float],
    patch_perimeter: wp.array[float],
    patch_oob: wp.array[float],
):
    """Compute ``1/b = perimeter / (2 * area)`` for each contact patch."""
    p = wp.tid()
    area = patch_area[p]
    if area < _MIN_PATCH_AREA:
        patch_oob[p] = 0.0
    else:
        patch_oob[p] = patch_perimeter[p] / (2.0 * area)


@wp.kernel(enable_backward=False)
def scm_constitutive(
    nx: int,
    ny: int,
    delta: float,
    cell_area: float,
    dt: float,
    xform: wp.transform,
    labels: wp.array[wp.int32],
    patch_oob: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    soil: wp.array[SCMSoilParameters],
    soil_stride: int,
    damping_unilateral: int,
    nodes: wp.array[SCMNodeState],
    body_f: wp.array[wp.spatial_vector],
):
    """Apply the soil model to hit nodes, using their patch's ``1/b``."""
    k = wp.tid()
    if nodes[k].hit_shape < 0:
        return

    label = labels[k]
    oob = float(0.0)
    if label >= 0:
        oob = patch_oob[label]

    i = k % nx
    j = k / nx
    scm_apply_constitutive(
        nodes,
        k,
        soil,
        k * soil_stride,
        oob,
        cell_area,
        dt,
        xform,
        node_local_xy(i, j, nx, ny, delta),
        body_q,
        body_qd,
        body_com,
        body_f,
        damping_unilateral,
    )


# ---------------------------------------------------------------------------
# bulldozing
# ---------------------------------------------------------------------------


@wp.func
def scm_add_material(nodes: wp.array[SCMNodeState], k: int, amount: float):
    """Raise a node, banking anything blocked by an object above it.

    Material that cannot be placed is held in ``massremainder`` and flows to
    neighbors during erosion rather than being lost.
    """
    headroom = nodes[k].hit_level - nodes[k].level
    add = amount
    if add > headroom:
        nodes[k].massremainder = nodes[k].massremainder + (add - headroom)
        add = headroom
    if add > 0.0:
        nodes[k].level = nodes[k].level + add
        nodes[k].level_initial = nodes[k].level_initial + add


@wp.func
def scm_remove_material(nodes: wp.array[SCMNodeState], k: int, amount: float):
    """Lower a node, drawing from banked material first.

    Unlike the reference implementation, banked material is consumed *instead*
    of lowering the surface rather than in addition to it, so volume is
    conserved.
    """
    remaining = amount
    banked = nodes[k].massremainder
    if banked > 0.0:
        used = wp.min(banked, remaining)
        nodes[k].massremainder = banked - used
        remaining = remaining - used
    if remaining > 0.0:
        nodes[k].level = nodes[k].level - remaining
        nodes[k].level_initial = nodes[k].level_initial - remaining


@wp.kernel(enable_backward=False)
def scm_bulldoze_accumulate(
    nx: int,
    ny: int,
    labels: wp.array[wp.int32],
    nodes: wp.array[SCMNodeState],
    patch_flow: wp.array[float],
    patch_boundary: wp.array[wp.int32],
):
    """Sum displaced material per patch and count each patch's boundary nodes.

    A boundary node is an untouched 4-neighbor of a loaded node: that is where
    displaced soil piles up to form the rut walls.
    """
    k = wp.tid()
    label = labels[k]
    if label >= 0 and nodes[k].sigma > 0.0:
        wp.atomic_add(patch_flow, label, nodes[k].step_plastic_flow)

    # A node is a boundary node if it is itself untouched but adjacent to a
    # loaded node; attribute it to that neighbor's patch.
    if nodes[k].sigma > 0.0:
        return

    i = k % nx
    j = k / nx
    owner = wp.int32(-1)
    if i > 0 and nodes[(i - 1) + nx * j].sigma > 0.0:
        owner = labels[(i - 1) + nx * j]
    elif i < nx - 1 and nodes[(i + 1) + nx * j].sigma > 0.0:
        owner = labels[(i + 1) + nx * j]
    elif j > 0 and nodes[i + nx * (j - 1)].sigma > 0.0:
        owner = labels[i + nx * (j - 1)]
    elif j < ny - 1 and nodes[i + nx * (j + 1)].sigma > 0.0:
        owner = labels[i + nx * (j + 1)]

    if owner >= 0:
        wp.atomic_add(patch_boundary, owner, 1)


@wp.kernel(enable_backward=False)
def scm_bulldoze_raise(
    nx: int,
    ny: int,
    dt: float,
    params: wp.array[float],
    labels: wp.array[wp.int32],
    patch_flow: wp.array[float],
    patch_boundary: wp.array[wp.int32],
    nodes: wp.array[SCMNodeState],
):
    """Raise each patch's boundary nodes by its share of displaced material.

    Args:
        nx: Grid node count along terrain X.
        ny: Grid node count along terrain Y.
        dt: Timestep [s].
        params: Bulldozing coefficients ``[tan(erosion_angle), flow_factor]``,
            read device-side so they stay mutable under CUDA graph capture.
        labels: Contact patch labels.
        patch_flow: Per-patch displaced material rate [m/s].
        patch_boundary: Per-patch boundary node count.
        nodes: Node state array.
    """
    k = wp.tid()
    if nodes[k].sigma > 0.0:
        return

    i = k % nx
    j = k / nx
    owner = wp.int32(-1)
    if i > 0 and nodes[(i - 1) + nx * j].sigma > 0.0:
        owner = labels[(i - 1) + nx * j]
    elif i < nx - 1 and nodes[(i + 1) + nx * j].sigma > 0.0:
        owner = labels[(i + 1) + nx * j]
    elif j > 0 and nodes[i + nx * (j - 1)].sigma > 0.0:
        owner = labels[i + nx * (j - 1)]
    elif j < ny - 1 and nodes[i + nx * (j + 1)].sigma > 0.0:
        owner = labels[i + nx * (j + 1)]

    if owner < 0:
        return

    # Mark the erosion domain even when nothing flowed this step, so material
    # deposited earlier keeps relaxing after the contact has moved on.
    nodes[k].erosion = 1

    count = patch_boundary[owner]
    if count <= 0:
        return

    total = params[1] * patch_flow[owner] * dt
    if total <= 0.0:
        return

    scm_add_material(nodes, k, total / float(count))


@wp.kernel(enable_backward=False)
def scm_erosion_dilate(
    nx: int,
    ny: int,
    nodes: wp.array[SCMNodeState],
    erosion_in: wp.array[wp.int32],
    erosion_out: wp.array[wp.int32],
):
    """Grow the erosion domain by one ring of untouched 4-neighbors."""
    k = wp.tid()
    if nodes[k].sigma > 0.0:
        erosion_out[k] = 0
        return
    if erosion_in[k] != 0:
        erosion_out[k] = 1
        return

    i = k % nx
    j = k / nx
    flag = wp.int32(0)
    if i > 0 and erosion_in[(i - 1) + nx * j] != 0:
        flag = 1
    if i < nx - 1 and erosion_in[(i + 1) + nx * j] != 0:
        flag = 1
    if j > 0 and erosion_in[i + nx * (j - 1)] != 0:
        flag = 1
    if j < ny - 1 and erosion_in[i + nx * (j + 1)] != 0:
        flag = 1
    erosion_out[k] = flag


@wp.kernel(enable_backward=False)
def scm_erosion_relax(
    nx: int,
    ny: int,
    delta: float,
    params: wp.array[float],
    parity: int,
    erosion: wp.array[wp.int32],
    nodes: wp.array[SCMNodeState],
):
    """One red-black slope-limiting sweep over the erosion domain.

    Checkerboard ordering makes the sweep parallel and deterministic, unlike the
    sequential in-place relaxation of the reference implementation. Material
    moves downhill only where the slope exceeds the soil's angle of repose.

    Args:
        nx: Grid node count along terrain X.
        ny: Grid node count along terrain Y.
        delta: Grid spacing [m].
        params: Bulldozing coefficients ``[tan(erosion_angle), flow_factor]``.
        parity: ``0`` updates red cells, ``1`` black cells.
        erosion: Erosion-domain mask, shape ``[nx * ny]``.
        nodes: Node state array.
    """
    k = wp.tid()
    i = k % nx
    j = k / nx
    if (i + j) % 2 != parity:
        return
    if erosion[k] == 0:
        return

    dy_lim = delta * params[0]

    height = nodes[k].level + nodes[k].massremainder

    for d in range(4):
        ni = i
        nj = j
        if d == 0:
            ni = i - 1
        elif d == 1:
            ni = i + 1
        elif d == 2:
            nj = j - 1
        else:
            nj = j + 1

        if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
            continue

        n = ni + nx * nj
        if nodes[n].sigma > 0.0:
            continue

        dy = height - (nodes[n].level + nodes[n].massremainder)
        excess = wp.abs(dy) - dy_lim
        if excess <= 0.0:
            continue

        move = 0.25 * 0.5 * excess
        if dy > 0.0:
            scm_remove_material(nodes, k, move)
            scm_add_material(nodes, n, move)
        else:
            scm_remove_material(nodes, n, move)
            scm_add_material(nodes, k, move)

        height = nodes[k].level + nodes[k].massremainder


# ---------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------


@wp.func
def scm_plot_value(nodes: wp.array[SCMNodeState], k: int, field: int) -> float:
    """Return the node scalar selected by ``field``, a :class:`SCMPlotField`."""
    if field == int(SCMPlotField.LEVEL):
        return nodes[k].level
    if field == int(SCMPlotField.SINKAGE):
        return nodes[k].sinkage
    if field == int(SCMPlotField.SINKAGE_ELASTIC):
        return nodes[k].sinkage_elastic
    if field == int(SCMPlotField.SINKAGE_PLASTIC):
        return nodes[k].sinkage_plastic
    if field == int(SCMPlotField.PRESSURE):
        return nodes[k].sigma
    if field == int(SCMPlotField.PRESSURE_YIELD):
        return nodes[k].sigma_yield
    if field == int(SCMPlotField.SHEAR):
        return nodes[k].tau
    if field == int(SCMPlotField.SHEAR_DISPLACEMENT):
        return nodes[k].kshear
    if field == int(SCMPlotField.PLASTIC_FLOW):
        return nodes[k].step_plastic_flow
    return 0.0


@wp.kernel(enable_backward=False)
def scm_reset_plot_range(plot_range: wp.array[float]):
    """Reset the auto-range accumulator slots to an inverted interval."""
    plot_range[2] = 1.0e30
    plot_range[3] = -1.0e30


@wp.kernel(enable_backward=False)
def scm_reduce_plot_range(
    field: int,
    nodes: wp.array[SCMNodeState],
    plot_range: wp.array[float],
):
    """Reduce the plot field over touched nodes only.

    Including untouched nodes would pin the range to the undisturbed value and
    collapse all the interesting variation into a single color.
    """
    k = wp.tid()
    if nodes[k].sigma <= 0.0 and nodes[k].sinkage <= 0.0:
        return
    v = scm_plot_value(nodes, k, field)
    wp.atomic_min(plot_range, 2, v)
    wp.atomic_max(plot_range, 3, v)


@wp.kernel(enable_backward=False)
def scm_smooth_plot_range(decay: float, dt: float, plot_range: wp.array[float]):
    """Blend the measured range into the displayed one.

    Expands immediately so a new peak is never clipped, then decays slowly, which
    stops the colors flickering as the contact patch moves.
    """
    lo_m = plot_range[2]
    hi_m = plot_range[3]
    if lo_m > hi_m:
        return

    lo = plot_range[0]
    hi = plot_range[1]
    a = wp.clamp(decay * dt, 0.0, 1.0)

    plot_range[0] = wp.min(lo_m, lo + (lo_m - lo) * a)
    plot_range[1] = wp.max(hi_m, hi + (hi_m - hi) * a)


@wp.kernel(enable_backward=False)
def scm_update_mesh(
    nx: int,
    ny: int,
    delta: float,
    xform: wp.transform,
    field: int,
    colormap: int,
    plot_range: wp.array[float],
    base_color: wp.vec3,
    nodes: wp.array[SCMNodeState],
    vertices: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
):
    """Refresh the visualization mesh from current node state."""
    k = wp.tid()
    i = k % nx
    j = k / nx

    xy = node_local_xy(i, j, nx, ny, delta)
    vertices[k] = wp.transform_point(xform, wp.vec3(xy[0], xy[1], nodes[k].level))

    h_e = nodes[node_index(i + 1, j, nx, ny)].level
    h_w = nodes[node_index(i - 1, j, nx, ny)].level
    h_n = nodes[node_index(i, j + 1, nx, ny)].level
    h_s = nodes[node_index(i, j - 1, nx, ny)].level
    n_local = wp.normalize(wp.vec3(h_w - h_e, h_s - h_n, 2.0 * delta))
    normals[k] = wp.transform_vector(xform, n_local)

    if field == int(SCMPlotField.NONE):
        colors[k] = base_color
        return

    lo = plot_range[0]
    hi = plot_range[1]
    span = hi - lo
    t = float(0.0)
    if span > 1.0e-12:
        t = (scm_plot_value(nodes, k, field) - lo) / span
    colors[k] = sample_colormap(colormap, t)


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def scm_query_height(
    nx: int,
    ny: int,
    delta: float,
    xform_inv: wp.transform,
    xform: wp.transform,
    undeformed: int,
    nodes: wp.array[SCMNodeState],
    points: wp.array[wp.vec3],
    out_height: wp.array[float],
):
    """Sample terrain height below each query point, in world space.

    Uses nearest-node lookup with edge clamping, so queries outside the patch
    return the height of the closest initialized node.
    """
    q = wp.tid()
    p_local = wp.transform_point(xform_inv, points[q])

    i = int(wp.round(p_local[0] / delta + 0.5 * float(nx - 1)))
    j = int(wp.round(p_local[1] / delta + 0.5 * float(ny - 1)))
    k = node_index(i, j, nx, ny)

    h = nodes[k].level
    if undeformed != 0:
        h = nodes[k].level_undeformed

    xy = node_local_xy(clamp_index(i, nx), clamp_index(j, ny), nx, ny, delta)
    out_height[q] = wp.transform_point(xform, wp.vec3(xy[0], xy[1], h))[2]


@wp.kernel(enable_backward=False)
def scm_query_normal(
    nx: int,
    ny: int,
    delta: float,
    xform_inv: wp.transform,
    xform: wp.transform,
    nodes: wp.array[SCMNodeState],
    points: wp.array[wp.vec3],
    out_normal: wp.array[wp.vec3],
):
    """Sample the deformed terrain normal below each query point, in world space."""
    q = wp.tid()
    p_local = wp.transform_point(xform_inv, points[q])

    i = int(wp.round(p_local[0] / delta + 0.5 * float(nx - 1)))
    j = int(wp.round(p_local[1] / delta + 0.5 * float(ny - 1)))

    h_e = nodes[node_index(i + 1, j, nx, ny)].level
    h_w = nodes[node_index(i - 1, j, nx, ny)].level
    h_n = nodes[node_index(i, j + 1, nx, ny)].level
    h_s = nodes[node_index(i, j - 1, nx, ny)].level
    n_local = wp.normalize(wp.vec3(h_w - h_e, h_s - h_n, 2.0 * delta))
    out_normal[q] = wp.transform_vector(xform, n_local)
