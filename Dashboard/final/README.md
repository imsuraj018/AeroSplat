# AeroSplat Formal Dynamic Dashboard

## What is included
- Dynamic scene/incident management.
- Add images as: append to current scene, new incident, or evidence-only.
- Existing 3DGS model remains visible while a new reconstruction is running.
- Automatic discovery of the latest GraphDeco Gaussian model for the active scene.
- Interactive browser Gaussian viewer.
- COLMAP -> undistortion -> GraphDeco 3DGS reconstruction pipeline.
- Reconstruction telemetry and formal command-center UI.

## Expected local tools
- COLMAP: `C:\bin\colmap.exe`
- GraphDeco repository: `C:\Users\Admin\Downloads\AeroSplat_Upgraded\external\gaussian-splatting`
- Python environment with your validated PyTorch / CUDA stack.

## Run
```cmd
C:\Users\Admin\anaconda3\Scripts\activate.bat C:\Users\Admin\anaconda3\envs\myenv
cd <this-folder>
pip install -r requirements.txt
python server.py
```
Open http://127.0.0.1:8000

## Initial existing model
The legacy scene reads from `C:\aerosplat_workspace`, including the existing 7,000-iteration model under `gaussian_output`. New incidents are stored beneath `C:\aerosplat_workspace\scenes`.

## Operational note
Damage, triage and route values remain pending until validated analysis modules are connected. Do not interpret placeholders as engineering certification.

## Response Intelligence
See `README_RESPONSE_INTELLIGENCE.md` for automatic hazard detection, 3D hazard mapping, spatial route projection, and approach recommendation.
