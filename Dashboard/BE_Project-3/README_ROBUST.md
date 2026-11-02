# AeroSplat — Robust Dashboard + 3DGS Integration

This is a hardened version of the uploaded AeroSplat dashboard. It preserves the existing API/UI approach while improving:

- Local ES-module GaussianSplats3D loading (no unpkg CDN dependency).
- Correct GraphDeco PLY model discovery and serving.
- Dynamic scene bounds and camera framing via `/api/model/bounds`.
- COLMAP camera pose lookup via `/api/camera/{filename}`.
- Correct COLMAP `points3D.bin` track stride (16 bytes per track entry).
- Startup analysis always refreshes from the active imagery instead of silently staying at zero.
- GraphDeco training launched with the active environment's `sys.executable`.
- Existing 3D twin remains available while a new reconstruction is running.
- Append/evidence workflow and stale-model tracking.
- Atomic JSON state writes and bounded logs.
- Explicit error messages for missing COLMAP/GraphDeco/model files.

## Required local viewer files

Inside `aerosplat_system/static/vendor/` place:

- `three.module.js`
- `gaussian-splats-3d.module.js`

From a project directory containing `node_modules`, you can run `setup_viewer.cmd`.

## Run

```cmd
C:\Users\Admin\anaconda3\Scripts\activate.bat C:\Users\Admin\anaconda3\envs\myenv
cd <this-folder>\aerosplat_system
pip install -r requirements.txt
python server.py
```

Open `http://127.0.0.1:8002` unless `AEROSPLAT_PORT` is set.

## Existing legacy scene

The default `legacy` scene reads the existing workspace:

`C:\aerosplat_workspace\images`

and the existing GraphDeco model under:

`C:\aerosplat_workspace\gaussian_output\point_cloud\iteration_<N>\point_cloud.ply`

Do not retrain simply to open the dashboard.

## When adding images

- **Append:** adds them to the reconstruction dataset and marks the current model stale.
- **Evidence:** stores them for inspection without changing the reconstruction dataset.
- **New incident:** create a separate scene first using `+ New incident`.

## Safety/analysis note

The current crack-like and geometry indicators are screening baselines. They are not structural certification and should not be used as a substitute for qualified engineering inspection.

## Responder Route Planning

The command center now includes a 2D, image-plane responder route planner.

Workflow:
1. Create a new disaster site with disaster type and site/location reference.
2. Upload reconstruction imagery. The current screening analysis runs automatically after upload.
3. Select a drone frame in `Drone Image Inspection`.
4. In `Responder Route Planning`, click `Set START`, then `Set INSPECTION TARGET`.
5. Select `Add HAZARD` and mark visible hazards such as unstable walls, voids, fire, gas, utilities, debris, water, or restricted zones.
6. Use `Approach / inspection corridor` for an approach plan.
7. Use `Controlled-entry planning` only when qualified site authority has explicitly authorized entry.
8. Generate the route. The backend uses weighted A* over the selected 2D frame, treating hard hazards as blocked and soft hazards as exposure costs.

The route is a planning overlay and does not provide structural clearance, measured physical distance, or permission to enter a damaged structure. Controlled-entry route generation is blocked without explicit authorization, and a RED preliminary screening state blocks automated entry-route generation.

### Route API
- `POST /api/route/plan`
- `GET /api/route`
- `DELETE /api/route`
