# AeroSplat – Dynamic 3DGS Dashboard

## What this version does
- Starts by loading an existing trained GraphDeco Gaussian model from `C:\aerosplat_workspace\gaussian_output` (or `gaussian_test`) without retraining.
- Shows the current reconstruction images dynamically.
- Image upload asks whether files are appended to the current reconstruction or stored as evidence.
- New reconstruction images mark the active model as `STALE`; the old model stays visible.
- `REBUILD DIGITAL TWIN` snapshots the active images into an isolated job directory, then runs COLMAP -> undistortion -> GraphDeco 3DGS.
- When the new model completes, it becomes the active model and the browser viewer reloads it.
- New incidents are isolated under `C:\aerosplat_workspace\scenes\<incident_id>`.

## Verified project prerequisites on the development machine
- Python 3.12.x
- PyTorch 2.5.1+cu121
- CUDA Toolkit 12.4 (used for native extension compilation)
- Visual Studio 2022 x64 Build Tools
- COLMAP 4.x
- GraphDeco Gaussian Splatting with `diff-gaussian-rasterization` and `simple-knn` compiled

## Run
Use the x64 Native Tools Command Prompt for VS 2022:

```cmd
C:\Users\Admin\anaconda3\Scripts\activate.bat C:\Users\Admin\anaconda3\envs\myenv
cd C:\path\to\BE_Project\aerosplat_system
pip install -r requirements.txt
python server.py
```

Open `http://127.0.0.1:8000`.

## Existing model
The server detects the newest `iteration_*\point_cloud.ply` under:
- `C:\aerosplat_workspace\gaussian_output`
- `C:\aerosplat_workspace\gaussian_test`
- completed reconstruction job directories

No 3DGS training occurs merely by opening the dashboard.

## Rebuild behavior
1. Add reconstruction images.
2. The current model becomes `STALE`.
3. Click `REBUILD DIGITAL TWIN` and select the desired iteration count.
4. The current model remains visible while the job is running.
5. A stable snapshot of all active reconstruction images is used for COLMAP/3DGS.
6. The new model is copied to the active scene's `gaussian_output` only after successful completion.

## Important
The damage, triage, and route values remain pending until validated analysis modules are connected. The 3DGS reconstruction is spatial evidence, not an engineering certification.
