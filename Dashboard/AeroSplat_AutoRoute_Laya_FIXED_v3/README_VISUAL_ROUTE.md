# AeroSplat - Visual Route Recommendation v2

This version makes the recommended route visibly render on the selected drone frame.

## Visual route behavior

- Route coordinates are normalized to the source image.
- The canvas uses the actual displayed image rectangle, including letterboxing.
- Selected route is drawn prominently over the image.
- Alternative candidates are drawn in muted lines.
- Direction arrows show travel direction.
- START and INSPECTION labels are drawn on the image.
- Automatic response analysis immediately invokes route recommendation.
- Manual START/TARGET remain optional overrides.

## Run

### Laya environment
```powershell
conda activate laya_env
python -m pip install -r requirements_laya.txt
```
Start Laya using the included `run_laya.cmd`.

### AeroSplat environment
```powershell
conda activate myenv
cd aerosplat_system
python server.py
```
Open http://127.0.0.1:8002

## Typical workflow

1. Select a disaster site.
2. Select a drone frame.
3. Click `Analyze selected frame`.
4. The system automatically runs route recommendation.
5. The selected route appears on the planning image.
6. Green line = recommended candidate; grey lines = alternatives.
7. Arrows indicate the travel direction.

## Safety
The route is a decision-support corridor based on available visual evidence. It is not a structural safety certification, physical clearance measurement, or authorization to enter a damaged structure.
