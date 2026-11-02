# AeroSplat Responder Recommendation & Route Planning

## What this adds

- Multi-site incident workspaces with disaster type and location/reference metadata.
- Automatic screening refresh after new reconstruction imagery is uploaded.
- A human-in-the-loop responder route planner using a selected drone frame.
- Click-to-place START, INSPECTION TARGET, and hazard zones.
- Hazard-aware weighted A* route generation.
- Hard no-go hazards and soft exposure-cost hazards.
- Controlled-entry authorization gate.
- Safety instructions generated from screening indicators and marked hazards.
- Per-site route persistence and reporting.

## Operational workflow

`New Disaster Site → Upload Drone Images → Screening → Rebuild Digital Twin → Select Frame → Mark START/TARGET/Hazards → Generate Lowest-Exposure Approach Route`

For controlled entry, the workflow additionally requires a qualified site authority to explicitly authorize entry. The route planner remains a decision-support system; it does not certify structural safety or grant permission to enter.

## Route model

The current route engine plans on the selected 2D image plane because the project does not yet have a validated, metrically scaled walkable 3D occupancy/hazard map. The engine uses normalized image coordinates and weighted A*.

- Hard hazards: unstable wall, falling-debris zone, fire, gas, exposed utility, void/drop, restricted area.
- Soft hazards: debris, water, contamination.
- Relative distance is an image-plane metric and should not be presented as meters until metric calibration is connected.

## Next engineering stage

The next stage is to replace manual hazard points with spatial hazard segmentation and project those hazards into the Gaussian/photogrammetric coordinate frame. After metric calibration, route planning can move from 2D frame-space to a 3D traversability graph.
