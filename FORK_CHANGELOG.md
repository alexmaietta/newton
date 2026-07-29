# Fork Changelog

Changes made in this fork (`alexmaietta/newton`) that are not part of upstream `newton-physics/newton`.

## [Unreleased]

### Added

- Add `newton.terramechanics.SCMTerrain`, deformable terrain based on the Soil Contact Model (SCM). Register interacting shapes with `add_newton_model()`, `add_body()` or `add_shape()`; the terrain owns a BVH over just those shapes, applies Bekker-Janosi-Hanamoto soil reactions through `State.body_f`, and supports bulldozing, contact patches, spatially varying soil, and false-colored visualization. See `example_scm_sphere.py`
- Add `newton.viewer.Colormap` and `sample_colormap()` for false-coloring scalar fields on a mesh, with six perceptually uniform colormaps.
- Add `vertex_colors` to `log_mesh()` for per-vertex mesh colors, which take the place of `color` and of the per-instance colors passed to `log_instances()`. Supported by `ViewerGL` (vertex color attribute), `ViewerUSD` and `ViewerRTX` (`displayColor` primvar with `vertex` interpolation), and `ViewerRerun` (`Mesh3D` vertex colors); other viewers accept and ignore it.
- Add `example_basic_vertex_colors`, a minimal viewer-only example animating a rippling grid mesh false-colored per-vertex by height.

### Changed

### Deprecated

### Removed

### Fixed

- Fix `ViewerGL` failing to start without a usable CUDA driver. `_build_packed_vbo_arrays()` requested pinned host memory unconditionally; pinned allocation goes through the CUDA driver even for host arrays, so it raised `Failed to allocate ... bytes on device 'cpu'` on CPU-only machines. Pinned memory is now requested only when the viewer device is CUDA.
