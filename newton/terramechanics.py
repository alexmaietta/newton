# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Terramechanics models for interaction with deformable ground.

Currently provides the Soil Contact Model (SCM), a semi-empirical
Bekker-Janosi-Hanamoto terrain model suitable for wheeled and tracked vehicles
and legged robots on soft soil.
"""

from ._src.terramechanics import SCMPlotField, SCMSoilParameters, SCMTerrain, scm_soil_parameters

__all__ = ["SCMPlotField", "SCMSoilParameters", "SCMTerrain", "scm_soil_parameters"]
