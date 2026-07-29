# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from enum import IntEnum

import warp as wp

__all__ = ["SCMNodeState", "SCMPlotField", "SCMSoilParameters", "scm_soil_parameters"]


class SCMPlotField(IntEnum):
    """Per-node scalar field encoded in :attr:`~newton.terramechanics.SCMTerrain.colors`."""

    NONE = 0
    """Uniform base color; no field is mapped."""

    LEVEL = 1
    """Node height along the terrain frame's +Z [m]."""

    SINKAGE = 2
    """Total sinkage along the local surface normal [m]."""

    SINKAGE_ELASTIC = 3
    """Recoverable part of the sinkage [m]."""

    SINKAGE_PLASTIC = 4
    """Irrecoverable part of the sinkage [m]."""

    PRESSURE = 5
    """Normal pressure on the soil [Pa]."""

    PRESSURE_YIELD = 6
    """Bekker yield pressure, a monotone record of peak compaction [Pa]."""

    SHEAR = 7
    """Janosi-Hanamoto shear stress [Pa]."""

    SHEAR_DISPLACEMENT = 8
    """Accumulated tangential shear displacement [m]."""

    PLASTIC_FLOW = 9
    """Rate of plastic sinkage growth [m/s]."""


@wp.struct
class SCMSoilParameters:
    """Bekker-Janosi-Hanamoto soil parameters.

    Stored device-side as a ``wp.array`` so values can be updated in place
    without re-capturing an enclosing CUDA graph.

    No presets are provided: soil parameters are strongly site- and
    material-specific, and a plausible-looking default invites using numbers that
    were never measured for the soil being modeled. Build a set explicitly with
    :func:`scm_soil_parameters`, ideally from a published identification for
    your soil.
    """

    bekker_kphi: float
    """Bekker frictional modulus [Pa/m^n]."""

    bekker_kc: float
    """Bekker cohesive modulus [Pa/m^(n-1)]. Zero disables the ``Kc/b`` term."""

    bekker_n: float
    """Bekker sinkage exponent, typically 0.6 to 1.8 [dimensionless]."""

    mohr_cohesion: float
    """Mohr-Coulomb cohesion [Pa]."""

    mohr_mu: float
    """Tangent of the Mohr-Coulomb internal friction angle [dimensionless]."""

    janosi_shear: float
    """Janosi-Hanamoto shear deformation modulus [m]."""

    elastic_k: float
    """Elastic stiffness per unit area [Pa/m]. Never below ``bekker_kphi``."""

    damping_r: float
    """Viscous damping per unit area [Pa*s/m]."""


def scm_soil_parameters(
    bekker_kphi: float = 2.0e6,
    bekker_kc: float = 0.0,
    bekker_n: float = 1.1,
    mohr_cohesion: float = 0.0,
    mohr_friction: float = 30.0,
    janosi_shear: float = 0.01,
    elastic_k: float = 2.0e8,
    damping_r: float = 3.0e4,
) -> SCMSoilParameters:
    """Build an :class:`SCMSoilParameters` set from human-facing units.

    Converts the Mohr-Coulomb friction angle from degrees to the tangent the
    kernels consume, and enforces ``elastic_k >= bekker_kphi``.

    This is a module-level function rather than a method on
    :class:`SCMSoilParameters` because ``wp.struct`` produces a ``Struct``
    instance rather than a class, so methods declared in the struct body are not
    reached through the descriptor protocol.

    Args:
        bekker_kphi: Bekker frictional modulus [Pa/m^n].
        bekker_kc: Bekker cohesive modulus [Pa/m^(n-1)]. Zero disables the
            ``Kc/b`` term and the contact-patch pass along with it.
        bekker_n: Bekker sinkage exponent, typically 0.6 to 1.8 [dimensionless].
        mohr_cohesion: Mohr-Coulomb cohesion [Pa].
        mohr_friction: Mohr-Coulomb internal friction angle [deg].
        janosi_shear: Janosi-Hanamoto shear deformation modulus [m].
        elastic_k: Elastic stiffness per unit area [Pa/m]. Raised to
            ``bekker_kphi`` if smaller, since a soil cannot unload more softly
            than its own loading curve.
        damping_r: Viscous damping per unit area [Pa*s/m].

    Returns:
        The populated parameter struct.
    """
    params = SCMSoilParameters()
    params.bekker_kphi = float(bekker_kphi)
    params.bekker_kc = float(bekker_kc)
    params.bekker_n = float(bekker_n)
    params.mohr_cohesion = float(mohr_cohesion)
    params.mohr_mu = float(math.tan(math.radians(mohr_friction)))
    params.janosi_shear = float(janosi_shear)
    params.elastic_k = float(max(elastic_k, bekker_kphi))
    params.damping_r = float(damping_r)
    return params


@wp.struct
class SCMNodeState:
    """Soil state at one grid node.

    Stored as an array-of-struct: the constitutive kernel touches most fields of
    a single node and no fields of any other, so a contiguous per-node record
    beats separate per-field arrays.
    """

    level: float
    """Current node height along the terrain frame's +Z [m]."""

    level_initial: float
    """Datum sinkage is measured from [m]. Bulldozing moves this with deposited
    material, so it is not the undeformed profile; see :attr:`level_undeformed`."""

    level_undeformed: float
    """Height of the initial, undisturbed profile [m]. Never modified."""

    normal_z: float
    """Cosine between the undeformed surface normal and the terrain +Z axis."""

    sinkage: float
    """Total sinkage along the local normal [m]."""

    sinkage_plastic: float
    """Irrecoverable sinkage along the local normal [m]."""

    sinkage_elastic: float
    """Recoverable sinkage along the local normal [m]."""

    sigma: float
    """Normal pressure [Pa]."""

    sigma_yield: float
    """Bekker yield pressure [Pa]. Monotone: soil remembers peak compaction."""

    kshear: float
    """Accumulated Janosi-Hanamoto shear displacement [m]."""

    tau: float
    """Shear stress along the local tangent [Pa]."""

    hit_level: float
    """Height of the last ray hit along +Z [m]; ``+inf`` when there was no hit."""

    step_plastic_flow: float
    """Rate of plastic sinkage growth over the last step [m/s]."""

    massremainder: float
    """Bulldozed material that could not be deposited at this node [m]."""

    hit_shape: wp.int32
    """Index of the shape hit by this node's ray, or ``-1``."""

    hit_body: wp.int32
    """Index of the body owning :attr:`hit_shape`, or ``-1``."""

    erosion: wp.int32
    """Nonzero when the node is inside the bulldozing erosion domain."""
