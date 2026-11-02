# AeroSplat – Integrated Dynamic Command Center

This build connects the browser dashboard to the actual AeroSplat workspace, COLMAP pipeline, and GraphDeco 3D Gaussian Splatting output.

## Expected local setup

- Workspace: `C:\aerosplat_workspace`
- COLMAP: `C:\bin\colmap.exe`
- GraphDeco repository: `C:\Users\Admin\Downloads\AeroSplat_Upgraded\external\gaussian-splatting`
- Python environment: your existing `myenv`
- CUDA toolkit: 12.4 for the training subprocess

Override locations with environment variables if needed:

- `AEROSPLAT_WORKSPACE`
- `COLMAP_EXE`
- `GAUSSIAN_SPLATTING_DIR`
- `AEROSPLAT_CUDA_HOME`
- `AEROSPLAT_ITERATIONS`
- `AEROSPLAT_RESOLUTION`
- `AEROSPLAT_MAX_JOBS`

## Run

Use the x64 Native Tools Command Prompt for VS 2022 that you used for 3DGS.

```cmd
C:\Users\Admin\anaconda3\Scripts\activate.bat C:\Users\Admin\anaconda3\envs\myenv
cd C:\path\to\AeroSplat\BE_Project\aerosplat_system
pip install -r requirements.txt
python server.py
```

Open `http://127.0.0.1:8000`.

## Existing model

The dashboard automatically detects an existing model such as:

`C:\aerosplat_workspace\gaussian_output\point_cloud\iteration_7000\point_cloud.ply`

It also loads all current workspace images from `C:\aerosplat_workspace\images`.

## Dynamic workflow

1. Add new drone images from the dashboard.
2. The dashboard marks the currently displayed model as stale.
3. Click **Run Reconstruction**.
4. A background job copies the dataset into an isolated job folder.
5. COLMAP feature extraction, matching, mapping, model validation, and undistortion run.
6. GraphDeco 3DGS trains on the undistorted COLMAP scene.
7. The new `point_cloud.ply` becomes the current model.
8. The dashboard updates telemetry and the 3D viewer.

## Safety / scientific scope

The dashboard intentionally does not invent crack, structural, rubble, triage, or route values. Those panels are ready for validated models to be connected later.
