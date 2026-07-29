# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...viewer.colormap import Colormap
from .types import SCMPlotField

if TYPE_CHECKING:
    from .terrain import SCMTerrain

__all__ = ["draw_scm_ui"]

_FIELD_LABELS = {
    SCMPlotField.NONE: "None",
    SCMPlotField.LEVEL: "Level (m)",
    SCMPlotField.SINKAGE: "Sinkage (m)",
    SCMPlotField.SINKAGE_ELASTIC: "Elastic sinkage (m)",
    SCMPlotField.SINKAGE_PLASTIC: "Plastic sinkage (m)",
    SCMPlotField.PRESSURE: "Pressure (Pa)",
    SCMPlotField.PRESSURE_YIELD: "Yield pressure (Pa)",
    SCMPlotField.SHEAR: "Shear stress (Pa)",
    SCMPlotField.SHEAR_DISPLACEMENT: "Shear displacement (m)",
    SCMPlotField.PLASTIC_FLOW: "Plastic flow (m/s)",
}

_FIELDS = list(_FIELD_LABELS)
_FIELD_NAMES = [_FIELD_LABELS[f] for f in _FIELDS]

_COLORMAPS = list(Colormap)
_COLORMAP_NAMES = [c.name.title() for c in _COLORMAPS]


def draw_scm_ui(terrain: SCMTerrain, ui: Any) -> None:
    """Draw the SCM terrain controls.

    Args:
        terrain: Terrain to inspect and control.
        ui: Active ImGui context, as passed to an example's ``gui(ui)``.
    """
    if not hasattr(ui, "collapsing_header"):
        return
    if not ui.collapsing_header("SCM Terrain"):
        return

    ui.text(f"Grid       {terrain.nx} x {terrain.ny}  (delta {terrain.delta:.3f} m)")

    # One small readback per frame, outside any captured region.
    hits, active = (int(v) for v in terrain.counters.numpy())
    ui.text(f"Ray hits   {hits:,} / {active:,} active")

    patches = "on" if terrain.contact_patches else "off"
    bulldoze = "on" if terrain.bulldozing else "off"
    ui.text(f"Patches {patches}   Bulldozing {bulldoze}")

    ui.separator()

    changed, index = ui.combo("Plot field", _FIELDS.index(terrain.plot_field), _FIELD_NAMES)
    if changed:
        terrain.set_plot(field=_FIELDS[index])

    changed, index = ui.combo("Colormap", _COLORMAPS.index(terrain.colormap), _COLORMAP_NAMES)
    if changed:
        terrain.set_plot(colormap=_COLORMAPS[index])

    changed, enabled = ui.checkbox("Auto range", terrain.auto_range)
    if changed:
        terrain.set_auto_range(enabled)

    vmin, vmax = terrain.plot_range
    if terrain.auto_range:
        ui.text(f"   min {vmin:.4g}   max {vmax:.4g}")
        changed, decay = ui.slider_float("Range decay", terrain.range_decay, 0.1, 20.0)
        if changed:
            terrain.set_auto_range(True, decay)
    else:
        changed_min, new_min = ui.slider_float("Range min", vmin, 0.0, max(vmax, 1e-6))
        changed_max, new_max = ui.slider_float("Range max", vmax, 0.0, max(vmax * 4.0, 1e-6))
        if changed_min or changed_max:
            terrain.set_plot(vmin=new_min, vmax=max(new_max, new_min + 1e-9))
