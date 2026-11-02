# AeroSplat v7 — Dynamic Results Fix

This version fixes initialization order so the existing COLMAP model and 3DGS model are analyzed at startup. It populates real reconstruction telemetry and image-quality screening values immediately instead of leaving Frames Screened at 0.

The 3D view remains a local, color-preserving PLY point preview. For photorealistic Gaussian splat rendering, replace it with a bundled GaussianSplats3D viewer or other local WebGL renderer.
