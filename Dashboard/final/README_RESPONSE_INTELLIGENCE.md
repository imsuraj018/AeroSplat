# AeroSplat Response Intelligence

This version adds four response-intelligence layers:

1. Automatic visual hazard indicators
2. 3D spatial hazard mapping using COLMAP camera poses + sparse 3D points
3. A risk-aware responder route projected into 3D coordinates
4. Automatic visible-frame approach candidate recommendation

## Important scope

The automatic detector is a baseline visual-indicator engine. It uses image-processing heuristics by default and can optionally use a custom YOLO model when configured through `AEROSPLAT_HAZARD_MODEL` and the `ultralytics` package. It does not certify buildings, structural stability, or responder safety.

The current 3D route is a spatially anchored representation of the image-plane route. It is not a verified walkable surface. Building a true traversability graph requires a validated floor/ground extraction stage.

## Run

```cmd
C:\Users\Admin\anaconda3\Scripts\activate.bat C:\Users\Admin\anaconda3\envs\myenv
cd C:\path\to\BE_Project-2\aerosplat_system
pip install -r requirements.txt
python server.py
```

Open:

`http://127.0.0.1:8002`

## Response analysis workflow

1. Select a disaster site.
2. Select a reconstruction frame.
3. Optionally mark START and INSPECTION TARGET in the existing route planner.
4. Click **Analyze selected frame** in the new Response Intelligence panel.
5. The backend runs automatic visual hazard detection.
6. Detected hazard boxes and confidence are shown on the selected frame.
7. When COLMAP geometry is available, each hazard is projected to a supporting 3D position.
8. A frame-boundary approach candidate is computed from heuristic exposure.
9. If START + TARGET exist, the existing weighted-A* planner uses the automatically detected hazards and the resulting path is spatially anchored to COLMAP points.
10. The Response Intelligence panel shows the hazard map, risk summary, graph stats, and 3D route coordinates.

## Site-wide analysis

The API also supports scanning every reconstruction image in the current site:

```http
POST /api/response/analyze
Content-Type: application/json

{
  "frame_name": "frame_001.jpg",
  "scope": "all"
}
```

The selected frame remains the one used for 3D spatial projection.

## API

- `GET /api/response` — saved response analysis for the active site
- `POST /api/response/analyze` — run automatic hazards + risk + 3D mapping + optional route
- `DELETE /api/response` — clear response analysis

## Optional custom detector

Set:

```cmd
set AEROSPLAT_HAZARD_MODEL=C:\path\to\your_hazard_model.pt
pip install ultralytics
```

The code recognizes common class-name fragments such as crack, rubble, debris, fire, smoke, water, and collapse.

## Tests

```cmd
python test_route_planner.py
python test_response_engine.py
```
