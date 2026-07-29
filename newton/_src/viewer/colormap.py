# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Colormaps for false-coloring scalar fields.

Provides a small set of perceptually uniform colormaps as Warp-resident lookup
tables, together with :func:`sample_colormap` for use inside kernels.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import warp as wp

__all__ = ["COLORMAP_RESOLUTION", "Colormap", "colormap_table", "sample_colormap"]

COLORMAP_RESOLUTION = 16
"""Number of control points per colormap lookup table."""


class Colormap(IntEnum):
    """Colormap used to false-color a scalar field."""

    VIRIDIS = 0
    """Perceptually uniform blue-green-yellow. The default."""

    PLASMA = 1
    """Perceptually uniform blue-magenta-yellow."""

    INFERNO = 2
    """Perceptually uniform black-red-yellow."""

    TURBO = 3
    """High-contrast rainbow. Higher detail, less perceptually uniform."""

    COOLWARM = 4
    """Diverging blue-white-red. Use for signed quantities."""

    GRAY = 5
    """Linear grayscale."""


# Control points sampled at 16 evenly spaced positions in [0, 1].
# VIRIDIS, PLASMA and INFERNO are from matplotlib (CC0); TURBO is Google's
# Improved Rainbow (Apache-2.0); COOLWARM follows Moreland's diverging map.
_COLORMAP_DATA: dict[Colormap, list[tuple[float, float, float]]] = {
    Colormap.VIRIDIS: [
        (0.267, 0.005, 0.329),
        (0.283, 0.100, 0.422),
        (0.278, 0.181, 0.487),
        (0.254, 0.265, 0.530),
        (0.222, 0.339, 0.549),
        (0.191, 0.407, 0.556),
        (0.165, 0.470, 0.557),
        (0.144, 0.532, 0.556),
        (0.128, 0.594, 0.547),
        (0.134, 0.659, 0.518),
        (0.185, 0.720, 0.470),
        (0.290, 0.771, 0.399),
        (0.426, 0.812, 0.307),
        (0.584, 0.843, 0.204),
        (0.751, 0.867, 0.115),
        (0.993, 0.906, 0.144),
    ],
    Colormap.PLASMA: [
        (0.051, 0.030, 0.528),
        (0.196, 0.018, 0.591),
        (0.302, 0.009, 0.632),
        (0.400, 0.001, 0.657),
        (0.494, 0.012, 0.658),
        (0.580, 0.077, 0.634),
        (0.655, 0.139, 0.589),
        (0.723, 0.197, 0.538),
        (0.784, 0.254, 0.484),
        (0.839, 0.313, 0.432),
        (0.887, 0.375, 0.380),
        (0.928, 0.442, 0.328),
        (0.960, 0.516, 0.276),
        (0.983, 0.598, 0.224),
        (0.993, 0.690, 0.176),
        (0.940, 0.975, 0.131),
    ],
    Colormap.INFERNO: [
        (0.001, 0.000, 0.014),
        (0.042, 0.028, 0.140),
        (0.113, 0.045, 0.264),
        (0.199, 0.038, 0.354),
        (0.281, 0.056, 0.402),
        (0.360, 0.079, 0.432),
        (0.437, 0.104, 0.450),
        (0.515, 0.128, 0.454),
        (0.593, 0.155, 0.442),
        (0.669, 0.187, 0.415),
        (0.741, 0.229, 0.373),
        (0.806, 0.283, 0.318),
        (0.862, 0.353, 0.252),
        (0.906, 0.437, 0.176),
        (0.938, 0.532, 0.091),
        (0.988, 0.998, 0.645),
    ],
    Colormap.TURBO: [
        (0.190, 0.072, 0.232),
        (0.231, 0.318, 0.752),
        (0.253, 0.531, 0.970),
        (0.223, 0.720, 0.876),
        (0.164, 0.855, 0.706),
        (0.140, 0.940, 0.520),
        (0.245, 0.988, 0.348),
        (0.442, 0.999, 0.220),
        (0.635, 0.971, 0.169),
        (0.783, 0.900, 0.196),
        (0.891, 0.799, 0.216),
        (0.963, 0.671, 0.196),
        (0.995, 0.520, 0.144),
        (0.981, 0.359, 0.084),
        (0.916, 0.212, 0.032),
        (0.480, 0.016, 0.011),
    ],
    Colormap.COOLWARM: [
        (0.230, 0.299, 0.754),
        (0.303, 0.388, 0.826),
        (0.383, 0.474, 0.886),
        (0.466, 0.556, 0.932),
        (0.551, 0.631, 0.962),
        (0.636, 0.698, 0.976),
        (0.718, 0.754, 0.973),
        (0.794, 0.798, 0.954),
        (0.858, 0.797, 0.910),
        (0.907, 0.760, 0.849),
        (0.941, 0.708, 0.779),
        (0.958, 0.643, 0.701),
        (0.958, 0.566, 0.617),
        (0.941, 0.478, 0.529),
        (0.907, 0.381, 0.438),
        (0.706, 0.016, 0.150),
    ],
    Colormap.GRAY: [(t / 15.0, t / 15.0, t / 15.0) for t in range(16)],
}

_COLORMAP_COUNT = len(Colormap)

_table_np = (
    np.zeros((_COLORMAP_COUNT, COLORMAP_RESOLUTION), dtype=np.float32)
    .repeat(3)
    .reshape(_COLORMAP_COUNT, COLORMAP_RESOLUTION, 3)
)
for _cmap, _entries in _COLORMAP_DATA.items():
    _table_np[int(_cmap)] = np.asarray(_entries, dtype=np.float32)

_COLORMAP_TABLE = wp.constant(
    wp.types.matrix(shape=(_COLORMAP_COUNT * COLORMAP_RESOLUTION, 3), dtype=wp.float32)(_table_np.reshape(-1, 3))
)


@wp.func
def sample_colormap(cmap: int, t: float) -> wp.vec3:
    """Sample a colormap by linear interpolation in its lookup table.

    Args:
        cmap: Colormap index, a :class:`Colormap` value.
        t: Position in the colormap. Clamped to ``[0, 1]``.

    Returns:
        Linear RGB color with components in ``[0, 1]``.
    """
    u = wp.clamp(t, 0.0, 1.0) * float(COLORMAP_RESOLUTION - 1)
    i0 = int(wp.floor(u))
    i1 = wp.min(i0 + 1, COLORMAP_RESOLUTION - 1)
    f = u - float(i0)

    base = cmap * COLORMAP_RESOLUTION
    r = wp.lerp(_COLORMAP_TABLE[base + i0, 0], _COLORMAP_TABLE[base + i1, 0], f)
    g = wp.lerp(_COLORMAP_TABLE[base + i0, 1], _COLORMAP_TABLE[base + i1, 1], f)
    b = wp.lerp(_COLORMAP_TABLE[base + i0, 2], _COLORMAP_TABLE[base + i1, 2], f)
    return wp.vec3(r, g, b)


def colormap_table(cmap: Colormap) -> np.ndarray:
    """Return the raw lookup table for a colormap.

    Args:
        cmap: Colormap to look up.

    Returns:
        Array of shape ``[COLORMAP_RESOLUTION, 3]`` with linear RGB in ``[0, 1]``.
    """
    return _table_np[int(cmap)].copy()
