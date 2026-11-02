# AeroSplat + Laya Route Recommendation

This version adds a decision-support route recommendation layer:

`drone frame -> hazards -> candidate routes -> hard-safety filter -> Laya evaluation -> route recommendation -> guidance`

## Responsibilities

- **AeroSplat / weighted A*** generates image-plane route candidates.
- **Hard constraints** reject routes that intersect explicit no-go hazards.
- **Laya** evaluates only eligible candidates using structured route evidence.
- **AeroSplat guidance rules** produce operational guidance from the observed hazards and screening state.
- **3D mapping** projects the selected 2D route to nearest COLMAP points when a valid COLMAP reconstruction is available.

Laya is not used to create path geometry and cannot override hard constraints.

## Environments

Keep your existing `myenv` unchanged for COLMAP/3DGS.
Create a separate environment for Laya:

```powershell
conda create -n laya_env python=3.12 -y
conda activate laya_env
python -m pip install "laya[serve]==0.3.20"
```

The current Laya project requires Python 3.10+ and provides the `laya-serve` HTTP server. Its current HTTP API is `POST /v1/systemone` with a `GET /health` probe. See the official project documentation for current package/runtime details.

## Start Laya

From the `BE_Project-2` folder:

```cmd
run_laya.cmd
```

Or manually in PowerShell:

```powershell
conda activate laya_env
$env:LAYA_HOST="127.0.0.1"
$env:LAYA_PORT="8100"
$env:LAYA_DEVICE="cpu"
$env:LAYA_PRELOAD="0"
laya-serve
```

Test:

```text
http://127.0.0.1:8100/health
```

You should see JSON containing `status: ok`.

## Start AeroSplat

In a second terminal:

```powershell
conda activate myenv
cd "C:\path\to\BE_Project-2\aerosplat_system"
python server.py
```

AeroSplat will use:

```text
http://127.0.0.1:8100/v1/systemone
```

Override with:

```powershell
$env:LAYA_URL="http://127.0.0.1:8100/v1/systemone"
```

Optional bearer token:

```powershell
$env:LAYA_API_KEY="your-token"
```

## UI flow

1. Select or create a disaster site.
2. Select a reconstruction frame.
3. Set `START` and `INSPECTION TARGET`.
4. Run `Analyze selected frame` to populate automatic hazard indicators.
5. Review or add manual hazards.
6. Click `Generate routes + Laya recommendation`.
7. AeroSplat creates several candidate corridors.
8. Hard hazards are filtered out.
9. Laya evaluates the remaining candidates.
10. The dashboard highlights the selected candidate, displays alternatives, route metrics, review priority and reassessment probability.

If Laya is unavailable, AeroSplat falls back to its deterministic route score so the planner remains usable. The UI makes that fallback explicit.

## Safety boundary

These outputs are decision-support overlays. They are not measured physical distances, structural certification, walkability guarantees, rescue authorization, or permission to enter a damaged structure. A qualified human responder/site authority must verify conditions on scene.
