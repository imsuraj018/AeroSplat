# AeroSplat v9 — Proper Gaussian 3D Viewer

This version replaces the browser point-cloud approximation with the actual Three.js Gaussian-splat renderer used to display INRIA/GraphDeco `.ply` scenes.

## 1. Prerequisites

Use the same Conda environment that successfully ran GraphDeco 3DGS. Also install Node.js LTS so `npm` is available.

## 2. Install the local browser renderer

From this folder run:

```cmd
setup_viewer.cmd
```

It installs:

- `@mkkellogg/gaussian-splats-3d@0.4.7`
- `three@0.160.0`

and copies the browser bundles into `aerosplat_system\static\vendor\`.

The GaussianSplats3D project documents support for INRIA/GraphDeco `.ply` files, spherical harmonics, and the `addSplatScene()` API used here. It also documents orbit/pan controls and optional optimization settings. https://github.com/mkkellogg/GaussianSplats3D

## 3. Start AeroSplat

```cmd
conda activate myenv
cd aerosplat_system
python server.py
```

Open:

`http://127.0.0.1:8000`

## 4. Existing model

The dashboard reads the latest trained model from the active workspace, including:

`C:\aerosplat_workspace\gaussian_output\point_cloud\iteration_7000\point_cloud.ply`

No retraining is needed just to open the dashboard.

## 5. What the viewer now does

The central panel is a real Gaussian-splat renderer rather than a conventional XYZ point plot. The renderer consumes Gaussian position, opacity, scale/rotation and spherical-harmonic appearance data from the PLY and renders them with Three.js/WebGL.

Controls:

- Left drag: orbit
- Right drag: pan
- Wheel: zoom
- `C`: mesh cursor
- `I`: diagnostics
- `P`: point-cloud mode
- `=` / `-`: splat scale
- `O`: orthographic mode

These controls are from the GaussianSplats3D viewer documentation.

## 6. New-image behavior

Adding images as **Append to current scene** marks the current twin stale. The existing twin remains visible. `Rebuild digital twin` then runs the real COLMAP -> undistortion -> GraphDeco 3DGS pipeline, and the dashboard switches to the new model only after the new PLY exists.

Evidence-only images are stored separately and do not trigger reconstruction.
