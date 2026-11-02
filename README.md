# AeroSplat — Upgraded Post-Disaster Digital Twin & Route Intelligence

**AI-powered disaster assessment, 3D reconstruction, damage analysis, and risk-aware responder route planning.**

AeroSplat is an AI-assisted post-disaster analysis platform designed to transform drone imagery and video into an interactive 3D digital representation of affected areas. It combines photogrammetry, 3D Gaussian Splatting (3DGS), computer vision, geometric damage assessment, and graph-based pathfinding to support preliminary disaster assessment and emergency response planning.

The system analyzes reconstructed environments, identifies potential damage and hazards, generates preliminary risk classifications, and recommends candidate ground-responder routes while considering detected obstacles and hazardous areas.

> **Project status:** Research and decision-support prototype. Its outputs are preliminary and require validation by qualified professionals before operational use.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Project Workflow](#project-workflow)
6. [3D Gaussian Splatting Integration](#3d-gaussian-splatting-integration)
7. [AI-Based Hazard Detection](#ai-based-hazard-detection)
8. [Geometry-Based Damage Assessment](#geometry-based-damage-assessment)
9. [Preliminary Damage Triage](#preliminary-damage-triage)
10. [Risk-Aware Route Planning](#risk-aware-route-planning)
11. [Dynamic Route Replanning](#dynamic-route-replanning)
12. [Synthetic Demo Mode](#synthetic-demo-mode)
13. [Project Structure](#project-structure)
14. [Installation and Setup](#installation-and-setup)
15. [Running AeroSplat](#running-aerosplat)
16. [API Documentation](#api-documentation)
17. [Configuration](#configuration)
18. [Expected Outputs](#expected-outputs)
19. [Limitations and Safety Considerations](#limitations-and-safety-considerations)
20. [Future Enhancements](#future-enhancements)

---

## Project Overview

During natural disasters, building collapses, fires, earthquakes, and other emergency situations, responders need timely information about the affected environment.

Traditional assessment methods may require significant time and physical access to potentially hazardous locations.

AeroSplat addresses this challenge by providing a digital analysis workflow that uses aerial imagery to generate a reconstructed environment and support preliminary damage and route assessment.

The platform aims to:

- Reconstruct affected environments from drone images and video.
- Generate an interactive 3D digital twin using photogrammetry and 3D Gaussian Splatting.
- Detect potential damage indicators and environmental hazards using computer vision.
- Estimate geometric damage measurements with associated confidence values.
- Generate preliminary GREEN, YELLOW, and RED triage categories informed by ATC-20 terminology.
- Create risk-aware ground-responder routes using the A* pathfinding algorithm.
- Provide structured incident information through a FastAPI backend and dashboard.
- Support synthetic demonstrations when real drone imagery is unavailable.

### Primary Objective

To develop an integrated AI-assisted disaster intelligence platform that connects 3D scene reconstruction, computer vision, geometric assessment, risk analysis, and route planning in a single workflow.

---

## Key Features

| Module | Description |
|---|---|
| Drone Data Processing | Processes uploaded drone images and videos. |
| COLMAP Integration | Performs Structure-from-Motion reconstruction. |
| 3D Gaussian Splatting | Integrates the official GraphDeco-Inria implementation. |
| Binary/Text COLMAP Loader | Supports robust loading of COLMAP reconstruction formats. |
| AI Hazard Segmentation | Optional YOLO-based segmentation of disaster-related objects and hazards. |
| Geometric Damage Assessment | Estimates wall tilt, crack dimensions, and debris measurements. |
| Confidence Estimation | Associates confidence values with detected and measured damage indicators. |
| Preliminary Triage | Generates ATC-20-informed preliminary GREEN/YELLOW/RED categories. |
| Risk Mapping | Represents assessed hazards and risk regions for route planning. |
| A* Route Planning | Calculates candidate responder paths using risk-aware graph search. |
| Route Replanning Design | Provides a D* Lite-ready API design for future dynamic replanning. |
| FastAPI Backend | Exposes analysis, report, visualization, and routing endpoints. |
| Synthetic Demo | Enables demonstrations without real drone data. |

---

## System Architecture

AeroSplat follows a modular architecture in which each component performs a specific stage of the disaster analysis pipeline.

```text
                  DRONE DATA
              Images / Video
                      |
                      v
             DATA PREPROCESSING
          Frame Extraction / Filtering
                      |
                      v
               COLMAP / SfM
          Camera Pose Estimation
           Sparse Reconstruction
                      |
                      v
                3D SCENE
          Point Cloud / Geometry
                      |
                      v
          3D GAUSSIAN SPLATTING
           Interactive 3D Scene
                      |
                      v
         AI & GEOMETRIC ANALYSIS
         /                    \
        v                      v
  YOLO Segmentation       Geometry Analysis
  Hazard Detection        Tilt / Crack / Debris
        \                      /
         \                    /
          v                  v
             DAMAGE ASSESSMENT
                      |
                      v
             PRELIMINARY TRIAGE
              GREEN / YELLOW / RED
                      |
                      v
                RISK MAPPING
                      |
                      v
              A* ROUTE PLANNING
                      |
                      v
             RESPONDER ROUTE
                      |
                      v
             FASTAPI DASHBOARD
          Reports / 3D View / APIs
```

### Architecture Components

**1. Data acquisition**

Accepts drone images and video as input.

**2. Preprocessing**

Prepares image data for reconstruction and analysis through image and video preprocessing operations.

**3. Structure-from-Motion**

COLMAP estimates camera positions and reconstructs the scene using feature extraction, feature matching, and geometric reconstruction.

**4. 3D reconstruction**

The reconstructed scene is used to generate a 3D representation, with optional integration of the official GraphDeco-Inria 3D Gaussian Splatting pipeline.

**5. AI and geometric analysis**

Computer vision and geometric methods analyze potential hazards and physical damage indicators.

**6. Risk assessment**

Combines available assessment outputs into preliminary risk information for downstream analysis.

**7. Route planning**

Uses risk-aware A* search to generate candidate responder routes.

**8. Dashboard and API**

Makes the results accessible through a FastAPI-based application.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| API Server | Uvicorn |
| Computer Vision | OpenCV |
| Object Detection and Segmentation | YOLO-compatible segmentation models |
| 3D Reconstruction | COLMAP |
| 3D Representation | 3D Gaussian Splatting |
| 3DGS Implementation | GraphDeco-Inria |
| Deep Learning | PyTorch |
| Numerical Computing | NumPy |
| Route Planning | A*, graph-based pathfinding |
| Dynamic Replanning Design | D* Lite-ready API |
| Data Exchange | JSON |
| Visualization | Browser-based dashboard and 3D visualization |

---

## Project Workflow

### Step 1: Drone Data Collection

The system accepts aerial images or video captured using drones.

Input data may include:

- Building exteriors.
- Collapsed structures.
- Road networks.
- Rubble and debris.
- Fire-affected areas.
- Surrounding infrastructure.

Image quality, camera overlap, lighting, and flight coverage influence reconstruction quality.

### Step 2: Image and Video Preprocessing

The input data is prepared for the reconstruction pipeline.

Possible operations include:

- Video frame extraction.
- Image validation.
- Image filtering.
- Preparation of reconstruction datasets.

### Step 3: COLMAP Reconstruction

COLMAP performs Structure-from-Motion (SfM) to estimate camera poses and reconstruct scene geometry.

The reconstruction process includes:

1. Feature extraction.
2. Feature matching.
3. Geometric verification.
4. Camera pose estimation.
5. Sparse 3D reconstruction.

AeroSplat includes support for loading COLMAP binary and text reconstruction formats.

### Step 4: 3D Scene Generation

The reconstructed scene becomes the foundation for 3D visualization and downstream analysis.

The system supports a point-cloud fallback and an adapter for launching the official 3D Gaussian Splatting implementation.

### Step 5: AI-Based Damage and Hazard Analysis

Optional segmentation models identify potential damage indicators and environmental hazards in the available imagery.

### Step 6: Geometric Damage Assessment

Geometric processing estimates measurable indicators such as wall inclination, crack dimensions, and debris characteristics.

### Step 7: Preliminary Triage

The assessment outputs are used to generate preliminary risk categories.

### Step 8: Responder Route Planning

The route planner evaluates candidate paths using risk-aware A* search.

### Step 9: Dashboard and Reporting

The FastAPI application exposes the current analysis state, incident report, reconstructed scene data, and recommended route.

---

## 3D Gaussian Splatting Integration

AeroSplat integrates the official **GraphDeco-Inria 3D Gaussian Splatting** implementation rather than claiming to independently implement its complete CUDA rasterizer and optimization pipeline.

Reference implementation:

[GraphDeco-Inria — Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

### Why 3D Gaussian Splatting?

3D Gaussian Splatting provides a method for representing and rendering reconstructed scenes using optimized 3D Gaussian primitives.

Within AeroSplat, this integration is intended to support:

- Detailed 3D scene visualization.
- Interactive exploration of reconstructed environments.
- Visual inspection of disaster-affected structures.
- Integration of 3D scene outputs into the dashboard.

### Integration Approach

AeroSplat uses an adapter architecture:

- `scripts/setup_3dgs.py` clones the official implementation into `third_party/gaussian-splatting`.
- `aerosplat_system/gs_adapter.py` provides the interface for launching the training pipeline.
- The official implementation handles the actual Gaussian Splatting optimization and rendering components.

### Setup

```powershell
python scripts/setup_3dgs.py
```

**Important:** The official implementation has its own CUDA, PyTorch, compiler, and dependency requirements. Successful installation of AeroSplat's Python packages does not automatically guarantee that the 3DGS CUDA extensions will compile.

---

## AI-Based Hazard Detection

AeroSplat supports optional YOLO-based segmentation for identifying potential disaster-related objects and hazards.

The segmentation module is designed to support categories such as:

- Buildings and structural regions.
- Cracks.
- Debris and rubble.
- Fire.
- Smoke.
- Other configured hazards.

### Model Configuration

The default model location is:

```text
models/disaster_seg.pt
```

Alternatively, configure the model path through:

```text
AEROSPLAT_YOLO_MODEL
```

### Processing Workflow

```text
Drone Image
     |
     v
YOLO Segmentation Model
     |
     v
Segmentation Masks
     |
     v
Hazard Regions
     |
     v
Damage and Risk Analysis
```

The segmentation model must be compatible with the inference code and the required class definitions.

**Note:** The model weights and training dataset are separate from the AeroSplat integration. Detection performance depends on the selected model, training data, image quality, and validation.

---

## Geometry-Based Damage Assessment

AeroSplat includes geometry-oriented assessment modules for estimating physical damage indicators.

The purpose is to move beyond simple visual classification and derive measurable information from available scene geometry and imagery.

### 1. Wall Tilt Estimation

Wall orientation and inclination can be analyzed using available geometric information.

Potential outputs include:

- Estimated wall orientation.
- Inclination or tilt measurements.
- Associated confidence values.

### 2. Crack Measurement

Crack-related analysis is intended to estimate:

- Crack dimensions.
- Approximate width and length.
- Detected crack regions.
- Measurement confidence.

Reliable metric measurements require suitable image resolution, calibration, and scale information.

### 3. Debris Measurement

The system supports geometry-based debris assessment, including:

- Debris regions.
- Approximate dimensions.
- Geometric measurements.
- Confidence information.

### Confidence Values

Measurements should be interpreted alongside their confidence values.

Confidence is intended to communicate the reliability of an estimate based on available input and processing. It must not be interpreted as proof that the physical measurement is accurate.

---

## Preliminary Damage Triage

AeroSplat uses an ATC-20-informed preliminary triage approach.

The system generates three broad categories:

| Category | Interpretation |
|---|---|
| GREEN | Preliminary screening indicates comparatively lower observed concern. |
| YELLOW | Preliminary screening indicates potential damage or restricted-use concerns requiring further assessment. |
| RED | Preliminary screening indicates potentially serious damage or hazards requiring urgent professional evaluation. |

### Triage Workflow

```text
Damage Indicators
       |
       v
Geometric Measurements
       |
       v
Hazard Detection
       |
       v
Risk Evaluation
       |
       v
Preliminary Triage
       |
       v
GREEN / YELLOW / RED
```

The triage output is intended to support prioritization and situational awareness.

**Safety qualification:** These categories are inspired by ATC-20 terminology. They are not official ATC-20 inspections, engineering certifications, or declarations of structural safety. A GREEN result does not establish that a building is safe to enter.

---

## Risk-Aware Route Planning

AeroSplat integrates A* pathfinding to generate candidate ground-responder routes.

Unlike ordinary shortest-path planning, the route planner is designed to account for risk information associated with the available road or navigation graph.

### A* Algorithm

A* evaluates nodes using:

\[
f(n)=g(n)+h(n)
\]

Where:

- \(g(n)\) is the accumulated cost from the starting node.
- \(h(n)\) is the heuristic estimate of the remaining cost.
- \(f(n)\) is the estimated total path cost.

### Risk-Aware Cost Function

The route planner can incorporate risk penalties into path costs.

A general conceptual formulation is:

\[
C(e)=D(e)+\lambda R(e)
\]

Where:

- \(D(e)\) represents the distance or movement cost of an edge.
- \(R(e)\) represents its associated risk penalty.
- \(\lambda\) controls the influence of risk on route selection.

Edges or areas designated as hard no-go regions should be excluded rather than treated as ordinary high-cost alternatives.

### Routing Workflow

1. Receive a road or navigation graph.
2. Identify the start and destination nodes.
3. Associate available risk information with graph elements.
4. Exclude hard no-go areas.
5. Evaluate candidate routes using A*.
6. Return the recommended candidate route.
7. Expose the result through the API.

**Important:** A* produces a graph-based candidate route. It does not independently verify physical ground accessibility, structural stability, or real-world traversability.

---

## Dynamic Route Replanning

AeroSplat includes a D* Lite-ready route replanning API design.

D* Lite is a search algorithm designed to support efficient replanning when the environment changes.

Potential future use cases include:

- Newly detected debris.
- Newly identified hazards.
- Blocked roads.
- Updated risk information.
- Changes in the navigation graph.

The current D* Lite-ready design should not be confused with a claim that a complete live D* Lite replanning system is already operational.

---

## Synthetic Demo Mode

AeroSplat includes a synthetic demonstration mode so that the application can be explored without waiting for drone imagery or a complete COLMAP reconstruction.

This makes it possible to demonstrate the application workflow during early development.

The synthetic mode is useful for:

- Testing dashboard functionality.
- Demonstrating API responses.
- Exploring example incident reports.
- Demonstrating preliminary risk categories.
- Testing route-planning workflows.
- Validating frontend and backend integration.

Synthetic outputs are demonstration data and must not be interpreted as real disaster assessments.

---

## Project Structure

The following illustrates the main architectural components and expected integration locations.

```text
AeroSplat/
│
├── aerosplat_system/
│   ├── server.py
│   ├── gs_adapter.py
│   └── ...
│
├── scripts/
│   └── setup_3dgs.py
│
├── third_party/
│   └── gaussian-splatting/
│
├── models/
│   └── disaster_seg.pt
│
├── requirements.txt
├── .gitignore
└── README.md
```

Additional modules, data directories, and frontend assets may be present depending on the project version.

### Important Components

| Component | Responsibility |
|---|---|
| `server.py` | FastAPI application and API endpoints. |
| `gs_adapter.py` | Adapter for launching the 3DGS implementation. |
| `setup_3dgs.py` | Clones the official GraphDeco-Inria implementation. |
| `disaster_seg.pt` | Optional YOLO segmentation model weights. |
| `requirements.txt` | Python dependency definitions. |

---

## Installation and Setup

### Prerequisites

Recommended development environment:

- Windows 10/11.
- Python 3.10 or 3.11.
- Git.
- COLMAP.
- NVIDIA GPU for GPU-accelerated 3DGS, with compatible CUDA and compiler dependencies.
- PowerShell.

### Step 1: Clone the Repository

```powershell
git clone https://github.com/imsuraj018/AeroSplat.git
cd AeroSplat
```

### Step 2: Create a Virtual Environment

Python 3.10 or 3.11 is recommended.

```powershell
py -3.10 -m venv myenv
```

Activate the environment:

```powershell
.\myenv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use an appropriate execution policy for your environment or activate through Command Prompt instead.

### Step 3: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 5: Install COLMAP

Install COLMAP and ensure that `colmap.exe` is accessible through the system PATH.

Verify:

```powershell
colmap -h
```

If COLMAP is not on PATH, configure its executable location using:

```text
COLMAP_EXE
```

### Step 6: Configure 3D Gaussian Splatting

For the official GraphDeco-Inria integration:

```powershell
python scripts/setup_3dgs.py
```

This prepares the third-party implementation under:

```text
third_party/gaussian-splatting
```

### Step 7: Configure YOLO Segmentation (Optional)

Place compatible segmentation weights at:

```text
models/disaster_seg.pt
```

Or configure:

```text
AEROSPLAT_YOLO_MODEL
```

The YOLO module is optional. The application can use other available analysis paths without a configured segmentation model.

---

## Running AeroSplat

Start the FastAPI application from the project root:

```powershell
uvicorn aerosplat_system.server:app --host 0.0.0.0 --port 8002
```

Open the application in your browser:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

### API Documentation

FastAPI provides interactive API documentation at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

These documentation routes are available when enabled by the application configuration.

---

## API Documentation

AeroSplat exposes REST endpoints for incident state, reports, visualization, uploads, reconstruction synchronization, route planning, and 3DGS integration status.

### 1. Get Current Analysis State

```http
GET /api/state
```

Returns the current analysis state.

### 2. Get Incident Report

```http
GET /api/report
```

Returns the structured incident report.

### 3. Get Splat Points

```http
GET /api/splat-points
```

Provides point-cloud data as a fallback for browser-based visualization.

### 4. Get Recommended Route

```http
GET /api/route
```

Returns the currently recommended candidate responder route.

### 5. Upload Drone Images or Video

```http
POST /api/upload-images
```

Uploads drone imagery or video for the analysis workflow.

### 6. Synchronize Workspace

```http
POST /api/sync
```

Rebuilds the analysis state using the current workspace.

### 7. Submit a Custom Route Graph

```http
POST /api/route
```

Accepts a custom road graph and start/goal nodes for route planning.

### 8. Check 3DGS Integration Status

```http
GET /api/models/3dgs-status
```

Returns the status of the 3D Gaussian Splatting adapter.

---

## Configuration

AeroSplat supports configuration through environment variables.

| Variable | Purpose |
|---|---|
| `COLMAP_EXE` | Path to the COLMAP executable when it is not available through PATH. |
| `AEROSPLAT_YOLO_MODEL` | Path to the optional YOLO segmentation model. |

Example PowerShell configuration:

```powershell
$env:COLMAP_EXE = "C:\Path\To\COLMAP\colmap.exe"

$env:AEROSPLAT_YOLO_MODEL = "models\disaster_seg.pt"
```

Use actual installed paths when configuring these variables.

---

## Expected Outputs

Depending on the available input data and enabled modules, AeroSplat is designed to provide:

### 1. 3D Digital Twin

A reconstructed representation of the affected environment for visual exploration.

### 2. Hazard Segmentation

Potential hazard regions identified by the configured segmentation model.

### 3. Geometric Damage Measurements

Estimated wall tilt, crack dimensions, and debris measurements, with associated confidence information.

### 4. Preliminary Incident Report

Structured incident information and preliminary triage classifications.

### 5. Risk Information

Risk-related information used by the route-planning module.

### 6. Candidate Responder Route

A graph-based route generated using risk-aware A* search.

### 7. Dashboard and API Responses

Access to incident state, reports, scene visualization data, route information, and model integration status.

---

## Limitations and Safety Considerations

AeroSplat is an AI-assisted research and decision-support prototype.

The following limitations must be considered:

1. **Reconstruction quality:** COLMAP and 3DGS results depend on image overlap, camera calibration, lighting, scene texture, and input quality.

2. **Segmentation accuracy:** Hazard detection depends on the model architecture, training data, class definitions, and inference conditions.

3. **Measurement reliability:** Geometric measurements require appropriate calibration, scale information, and suitable reconstruction quality.

4. **Confidence interpretation:** Confidence values do not guarantee physical measurement accuracy.

5. **Triage limitations:** ATC-20-informed categories are preliminary screening outputs, not certified inspection results.

6. **Route limitations:** A* evaluates the supplied graph and risk information. It does not independently confirm ground conditions or physical accessibility.

7. **Dynamic environment:** A route may become invalid when conditions change after analysis.

8. **Human verification:** Qualified structural engineers and emergency-response professionals must review findings before decisions are made.

### Critical Safety Disclaimer

AeroSplat must not be used as a standalone system for declaring buildings safe or unsafe, authorizing entry, or directing emergency personnel into hazardous environments.

Its GREEN/YELLOW/RED outputs are preliminary AI-assisted screening indicators, not official ATC-20 placards or engineering certifications.

All operational decisions must be made by qualified personnel using appropriate field inspections and validated procedures.

---



## References

1. **GraphDeco-Inria — 3D Gaussian Splatting:**  
   https://github.com/graphdeco-inria/gaussian-splatting

2. **COLMAP — Structure-from-Motion and Multi-View Stereo:**  
   https://github.com/colmap/colmap

3. **FastAPI Documentation:**  
   https://fastapi.tiangolo.com/

4. **PyTorch Documentation:**  
   https://pytorch.org/docs/

5. **Ultralytics Documentation:**  
   https://docs.ultralytics.com/

---

## Author

**Suraj Katkar**

GitHub: https://github.com/imsuraj018

Project Repository: https://github.com/imsuraj018/AeroSplat

---

**AeroSplat — From Drone Imagery to Disaster Intelligence.**



## Local Installation and Execution

Follow these steps to download, configure, and run AeroSplat locally on a Windows machine.

### 1. Prerequisites

Install the following tools before running the project:

- Python 3.10 or 3.11
- Git
- Node.js and npm (for the frontend dashboard)
- COLMAP (required for the photogrammetry reconstruction pipeline)
- NVIDIA GPU with compatible CUDA drivers (recommended for GPU acceleration)

Verify the installations:

```powershell
python --version
git --version
node --version
npm --version
```

### 2. Clone the Repository

Open PowerShell and execute:

```powershell
git clone https://github.com/imsuraj018/AeroSplat.git
cd AeroSplat_Upgraded
```

If you already have the project downloaded, navigate directly to its root directory:

```powershell
cd "C:\Users\Admin\Downloads\AeroSplat_Upgraded"
```

### 3. Create and Activate the Python Environment

Create a virtual environment:

```powershell
py -3.10 -m venv myenv
```

Activate it:

```powershell
.\myenv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\myenv\Scripts\Activate.ps1
```

Verify Python:

```powershell
python --version
```

### 4. Install Python Dependencies

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

### 5. Configure Environment Variables

If the project requires API keys or environment-specific configuration, create a `.env` file in the appropriate application directory.

Example:

```env
# Add only the environment variables required by your configuration.
# Never commit real API keys or passwords.
```

### 6. Start the AeroSplat Backend

From the project root directory, with the virtual environment activated:

```powershell
uvicorn aerosplat_system.server:app --host 0.0.0.0 --port 8000 --reload
```

The backend should be accessible at:

- API: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs

Keep this terminal running.

### 7. Start the Frontend Dashboard

Open a second PowerShell terminal and navigate to the dashboard directory:

```powershell
cd "C:\Users\Admin\Downloads\AeroSplat_Upgraded\Dashboard"
```

Install frontend dependencies:

```powershell
npm install
```

Start the development server:

```powershell
npm run dev
```

Open the local URL displayed in the terminal (commonly `http://localhost:5173` for Vite).

If the dashboard uses a different frontend framework or start script, follow the scripts defined in `Dashboard/package.json`.

### 8. Optional: Start the Laya Service

If the Laya advisory/reassessment service is included and configured in your project, activate its dedicated environment and start it using its actual application entry point.

For example, if its environment is named `laya_env`:

```powershell
.\laya_env\Scripts\Activate.ps1
```

Start the service using the entry point configured in that component. Do not run this step if the service is not required for your current workflow.

### 9. Verify the Application

After starting the backend and frontend, verify:

1. The backend starts without Python import or dependency errors.
2. The API documentation opens at `http://127.0.0.1:8002/docs`.
3. The frontend dashboard loads in the browser.
4. The dashboard can communicate with the backend.
5. Image upload, reconstruction, 3D visualization, damage analysis, and route-planning functions work with the required input data and configured dependencies.

### 10. Stopping the Application

Press `Ctrl + C` in each running terminal to stop its development server.

To deactivate the Python virtual environment:

```powershell
deactivate
```

### Troubleshooting

| Problem | Suggested solution |
|---|---|
| `python` not recognized | Verify Python installation and PATH configuration. |
| `uvicorn` not recognized | Activate the virtual environment and run `pip install -r requirements.txt`. |
| Module not found | Verify the working directory and install the missing dependency. |
| Frontend dependency errors | Run `npm install` inside the dashboard directory. |
| Frontend cannot connect to backend | Verify the backend URL and frontend API configuration. |
| COLMAP not recognized | Install COLMAP and configure its executable path. |
| CUDA unavailable | Verify NVIDIA drivers, CUDA compatibility, and the installed PyTorch build. |

**Note:** The commands above describe the expected development workflow. Exact frontend scripts, optional service entry points, and configuration variables must match the files included in the repository.
