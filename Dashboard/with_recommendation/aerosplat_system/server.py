import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
WORKSPACE = Path(os.getenv("AEROSPLAT_WORKSPACE", r"C:\aerosplat_workspace"))
LEGACY_IMAGE_DIR = WORKSPACE / "images"
SCENES_DIR = WORKSPACE / "scenes"
EVIDENCE_DIR = WORKSPACE / "evidence"
JOBS_DIR = WORKSPACE / "jobs"
STATE_FILE = WORKSPACE / "aerosplat_state_v5.json"
COLMAP_EXE = Path(os.getenv("COLMAP_EXE", r"C:\bin\colmap.exe"))
GAUSSIAN_DIR = Path(
    os.getenv(
        "GAUSSIAN_SPLATTING_DIR",
        r"C:\Users\Admin\Downloads\AeroSplat_Upgraded\external\gaussian-splatting",
    )
)
TRAIN_PY = GAUSSIAN_DIR / "train.py"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
DEFAULT_ITERATIONS = int(os.getenv("AEROSPLAT_ITERATIONS", "7000"))
DEFAULT_RESOLUTION = int(os.getenv("AEROSPLAT_RESOLUTION", "2"))
BUILD_MAX_JOBS = int(os.getenv("AEROSPLAT_MAX_JOBS", "2"))
MAX_UPLOAD_MB = int(os.getenv("AEROSPLAT_MAX_UPLOAD_MB", "100"))

for directory in (
    WORKSPACE,
    LEGACY_IMAGE_DIR,
    SCENES_DIR,
    EVIDENCE_DIR,
    JOBS_DIR,
    STATIC_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AeroSplat Command Center", version="5.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_lock = threading.RLock()
_thread: Optional[threading.Thread] = None


def safe_name(name: str) -> str:
    return Path(name or "").name


def image_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        [
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in {x.lower() for x in VALID_EXTS}
        ],
        key=lambda p: p.name.lower(),
    )


def file_hash(files: List[Path]) -> Optional[str]:
    if not files:
        return None
    hasher = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.name.lower()):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        hasher.update(path.name.encode("utf-8", errors="ignore"))
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(stat.st_mtime_ns).encode())
    return hasher.hexdigest()[:16]


def read_json(path: Path, default: Any):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def default_analysis() -> Dict[str, Any]:
    return {
        "status": "NOT_RUN",
        "quality": {
            "images_analyzed": 0,
            "mean_sharpness": None,
            "blurred_images": 0,
            "mean_brightness": None,
            "dark_frame_ratio": None,
        },
        "crack_visual_indicator": {
            "status": "BASELINE",
            "candidate_images": 0,
            "mean_candidate_ratio": None,
        },
        "geometry_indicator": {
            "status": "PENDING",
            "dominant_plane_tilt_deg": None,
            "points_sampled": 0,
        },
        "rubble_indicator": {
            "status": "NOT_COMPUTED",
            "volume_m3": None,
        },
        "notes": [],
    }


def default_route() -> Dict[str, Any]:
    return {
        "status": "NOT_COMPUTED",
        "mode": "approach",
        "entry_authorized": False,
        "plan_frame": None,
        "start": None,
        "target": None,
        "hazards": [],
        "path": [],
        "relative_distance": None,
        "exposure_cost": None,
        "blocked_zones": 0,
        "instructions": [],
        "reasons": [],
        "updated_at": None,
        "disclaimer": "Planning aid only. The route is not a declaration of structural safety or entry authorization.",
    }


def default_site_profile() -> Dict[str, Any]:
    return {
        "disaster_type": "Unknown",
        "location": "",
        "description": "",
        "created_at": None,
    }


def scene_profile(scene_id: str) -> Dict[str, Any]:
    if scene_id == "legacy":
        return {
            "disaster_type": "Existing workspace",
            "location": "",
            "description": "Primary AeroSplat workspace.",
            "created_at": None,
        }
    metadata = read_json(scene_root(scene_id) / "scene.json", {})
    profile = default_site_profile()
    profile.update({
        "disaster_type": metadata.get("disaster_type", "Unknown"),
        "location": metadata.get("location", ""),
        "description": metadata.get("description", ""),
        "created_at": metadata.get("created_at"),
    })
    return profile


def route_plan_path(scene_id: str) -> Path:
    return (WORKSPACE / "route_plan.json") if scene_id == "legacy" else scene_root(scene_id) / "route_plan.json"


def load_route_plan(scene_id: str) -> Dict[str, Any]:
    saved = read_json(route_plan_path(scene_id), None)
    route = default_route()
    if isinstance(saved, dict):
        route.update(saved)
    return route


def save_route_plan(scene_id: str, route: Dict[str, Any]) -> None:
    save_json(route_plan_path(scene_id), route)


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return default


def normalize_point(raw: Any) -> Optional[List[float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    return [clamp01(raw[0]), clamp01(raw[1])]


def hazard_defaults(kind: str) -> Dict[str, Any]:
    key = re.sub(r"[^a-z0-9_ -]", "", str(kind or "debris").lower()).strip().replace(" ", "_")
    defaults = {
        "debris": {"label": "Debris / rubble", "weight": 3.0, "hard": False, "radius": 0.06},
        "unstable_wall": {"label": "Suspected unstable wall", "weight": 9.0, "hard": True, "radius": 0.10},
        "falling_debris": {"label": "Falling-object zone", "weight": 8.0, "hard": True, "radius": 0.08},
        "fire": {"label": "Fire / heat zone", "weight": 10.0, "hard": True, "radius": 0.12},
        "gas": {"label": "Gas / vapor concern", "weight": 10.0, "hard": True, "radius": 0.10},
        "utility": {"label": "Exposed utility concern", "weight": 9.0, "hard": True, "radius": 0.08},
        "void": {"label": "Void / shaft / drop", "weight": 10.0, "hard": True, "radius": 0.08},
        "water": {"label": "Standing water / flow", "weight": 4.0, "hard": False, "radius": 0.08},
        "contamination": {"label": "Contamination concern", "weight": 7.0, "hard": False, "radius": 0.10},
        "restricted": {"label": "No-go / restricted area", "weight": 10.0, "hard": True, "radius": 0.10},
    }
    return dict(defaults.get(key, defaults["debris"]))


def route_instructions(analysis: Dict[str, Any], triage: Dict[str, Any], hazards: List[Dict[str, Any]], mode: str, authorized: bool) -> List[str]:
    instructions = [
        "Treat the route as a planning aid; it is not a structural-clearance decision.",
        "Use site-appropriate PPE, maintain communications, and keep a continuously usable egress path.",
        "Reassess conditions at the point of entry; secondary collapse, falling debris, utilities, fire, and airborne contaminants can change the risk picture.",
    ]
    label = str(triage.get("label") or "PENDING").upper()
    if mode == "controlled_entry":
        if not authorized:
            instructions.insert(0, "Controlled-entry planning is disabled until qualified site authority explicitly authorizes entry.")
        elif label == "RED":
            instructions.insert(0, "Do not use this plan to enter a preliminarily high-concern structure; obtain qualified structural / US&R direction before entry.")
        elif label == "YELLOW":
            instructions.insert(0, "Use only as a restricted-entry planning aid after qualified site authority confirms the entry point and controls.")
        else:
            instructions.insert(0, "Even with authorization, validate the route on scene before committing personnel or equipment.")
    else:
        instructions.insert(0, "This route is an approach-to-inspection corridor; it does not authorize entry into the collapse zone.")

    geometry = analysis.get("geometry_indicator", {})
    tilt = geometry.get("dominant_plane_tilt_deg")
    if tilt is not None and float(tilt) > 4:
        instructions.append("A provisional inclined-surface indicator was detected; avoid routing immediately alongside suspected leaning surfaces until verified.")
    crack = analysis.get("crack_visual_indicator", {})
    if float(crack.get("mean_candidate_ratio") or 0.0) > 0.018:
        instructions.append("Prioritize visual inspection of crack-like regions from a stand-off position before approaching affected surfaces.")
    if float(analysis.get("quality", {}).get("dark_frame_ratio") or 0.0) > 0.15:
        instructions.append("Some imagery is poorly illuminated; reacquire or add lighting before relying on obscured areas for route decisions.")
    for hazard in hazards:
        kind = str(hazard.get("kind") or "debris")
        if kind in {"gas", "fire", "utility", "void"}:
            instructions.append(hazard.get("label") or hazard_defaults(kind)["label"] + ": keep clear unless the relevant specialist confirms controls.")
    return list(dict.fromkeys(instructions))


def plan_route(payload: Dict[str, Any], analysis: Dict[str, Any], triage: Dict[str, Any], frame_name: Optional[str]) -> Dict[str, Any]:
    mode = str(payload.get("mode") or "approach").strip().lower()
    if mode not in {"approach", "controlled_entry"}:
        raise HTTPException(400, "mode must be approach or controlled_entry")

    authorized = bool(payload.get("entry_authorized"))
    start = normalize_point(payload.get("start"))
    target = normalize_point(payload.get("target"))
    if start is None or target is None:
        raise HTTPException(400, "start and target must be normalized [x,y] points")

    if mode == "controlled_entry" and not authorized:
        return {
            **default_route(),
            "status": "ENTRY_AUTHORIZATION_REQUIRED",
            "mode": mode,
            "entry_authorized": authorized,
            "plan_frame": frame_name,
            "start": start,
            "target": target,
            "instructions": route_instructions(analysis, triage, [], mode, authorized),
            "reasons": ["Controlled-entry authorization has not been confirmed."],
            "updated_at": time.time(),
        }

    if mode == "controlled_entry" and str(triage.get("label") or "PENDING").upper() == "RED":
        return {
            **default_route(),
            "status": "ENTRY_ROUTE_DISABLED",
            "mode": mode,
            "entry_authorized": authorized,
            "plan_frame": frame_name,
            "start": start,
            "target": target,
            "instructions": route_instructions(analysis, triage, [], mode, authorized),
            "reasons": ["Preliminary screening produced a RED concern state; the system will not generate an entry path from this evidence alone."],
            "updated_at": time.time(),
        }

    hazards_raw = payload.get("hazards") or []
    if not isinstance(hazards_raw, list):
        raise HTTPException(400, "hazards must be a list")

    hazards: List[Dict[str, Any]] = []
    for raw in hazards_raw[:80]:
        if not isinstance(raw, dict):
            continue
        point = normalize_point(raw.get("point") or [raw.get("x"), raw.get("y")])
        if point is None:
            continue
        kind = str(raw.get("kind") or "debris")
        defaults = hazard_defaults(kind)
        hazard = {
            "point": point,
            "kind": kind,
            "label": str(raw.get("label") or defaults["label"]),
            "weight": clamp_float(raw.get("weight"), 1.0, 10.0, defaults["weight"]),
            "hard": bool(raw.get("hard", defaults["hard"])),
            "radius": clamp_float(raw.get("radius"), 0.015, 0.25, defaults["radius"]),
        }
        hazards.append(hazard)

    width = int(clamp_float(payload.get("grid_width"), 50, 180, 120))
    height = int(clamp_float(payload.get("grid_height"), 35, 120, 80))

    def to_cell(pt: List[float]) -> Tuple[int, int]:
        return (min(width - 1, max(0, int(round(pt[0] * (width - 1))))), min(height - 1, max(0, int(round(pt[1] * (height - 1))))))

    sx, sy = to_cell(start)
    tx, ty = to_cell(target)

    blocked = np.zeros((height, width), dtype=bool)
    risk = np.zeros((height, width), dtype=np.float64)
    cell_x = np.arange(width, dtype=np.float64)[None, :] / max(width - 1, 1)
    cell_y = np.arange(height, dtype=np.float64)[:, None] / max(height - 1, 1)

    for hz in hazards:
        hx, hy = hz["point"]
        dx = (cell_x - hx) * max(width / max(height, 1), 1.0)
        dy = cell_y - hy
        dist = np.sqrt(dx * dx + dy * dy)
        radius = max(float(hz["radius"]), 1e-4)
        influence = np.clip(1.0 - dist / radius, 0.0, 1.0)
        risk += float(hz["weight"]) * influence
        if hz["hard"]:
            blocked |= dist <= radius

    blocked[sy, sx] = False
    if blocked[ty, tx]:
        return {
            **default_route(),
            "status": "TARGET_IN_BLOCKED_ZONE",
            "mode": mode,
            "entry_authorized": authorized,
            "plan_frame": frame_name,
            "start": start,
            "target": target,
            "hazards": hazards,
            "blocked_zones": sum(1 for h in hazards if h["hard"]),
            "instructions": route_instructions(analysis, triage, hazards, mode, authorized),
            "reasons": ["The selected target lies inside a marked hard hazard zone."],
            "updated_at": time.time(),
        }

    import heapq
    open_heap = [(0.0, sy, sx)]
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_cost = {(sx, sy): 0.0}
    moves = [
        (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
        (0, -1, 1.0), (0, 1, 1.0),
        (1, -1, math.sqrt(2)), (1, 0, 1.0), (1, 1, math.sqrt(2)),
    ]
    found = False
    while open_heap:
        current_f, cy, cx = heapq.heappop(open_heap)
        current_g = g_cost.get((cx, cy))
        if current_g is None:
            continue
        if (cx, cy) == (tx, ty):
            found = True
            break
        for dx, dy, step in moves:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height or blocked[ny, nx]:
                continue
            exposure = risk[ny, nx]
            tentative = current_g + step * (1.0 + exposure / 7.0)
            if tentative < g_cost.get((nx, ny), float("inf")):
                g_cost[(nx, ny)] = tentative
                came_from[(nx, ny)] = (cx, cy)
                heuristic = math.hypot(tx - nx, ty - ny)
                heapq.heappush(open_heap, (tentative + heuristic, ny, nx))

    if not found:
        return {
            **default_route(),
            "status": "NO_PATH",
            "mode": mode,
            "entry_authorized": authorized,
            "plan_frame": frame_name,
            "start": start,
            "target": target,
            "hazards": hazards,
            "blocked_zones": sum(1 for h in hazards if h["hard"]),
            "instructions": route_instructions(analysis, triage, hazards, mode, authorized),
            "reasons": ["No traversable path exists under the currently marked hard hazard constraints."],
            "updated_at": time.time(),
        }

    path_cells = [(tx, ty)]
    current = (tx, ty)
    while current != (sx, sy):
        current = came_from[current]
        path_cells.append(current)
    path_cells.reverse()

    path = [[round(x / max(width - 1, 1), 5), round(y / max(height - 1, 1), 5)] for x, y in path_cells]
    # Preserve corners so hard hazard avoidance remains visible in the UI.
    simplified = [path[0]]
    for i in range(1, len(path) - 1):
        ax, ay = path_cells[i - 1]
        bx, by = path_cells[i]
        cx, cy = path_cells[i + 1]
        d1 = (bx - ax, by - ay)
        d2 = (cx - bx, cy - by)
        if d1 != d2:
            simplified.append(path[i])
    simplified.append(path[-1])

    relative_distance = 0.0
    exposure_cost = 0.0
    for (ax, ay), (bx, by) in zip(path_cells, path_cells[1:]):
        step = math.hypot(bx - ax, by - ay)
        relative_distance += step
        exposure_cost += step * (1.0 + risk[by, bx] / 7.0)

    return {
        "status": "CONTROLLED_ENTRY_ROUTE" if mode == "controlled_entry" else "APPROACH_ROUTE",
        "mode": mode,
        "entry_authorized": authorized,
        "plan_frame": frame_name,
        "start": start,
        "target": target,
        "hazards": hazards,
        "path": simplified,
        "relative_distance": round(relative_distance, 2),
        "exposure_cost": round(exposure_cost, 2),
        "blocked_zones": sum(1 for h in hazards if h["hard"]),
        "instructions": route_instructions(analysis, triage, hazards, mode, authorized),
        "reasons": [
            "Path generated by weighted A* over the selected 2D planning frame.",
            "Hard hazards are treated as no-go cells and soft hazards contribute exposure cost.",
        ],
        "updated_at": time.time(),
        "disclaimer": "Relative path and exposure cost are image-plane planning metrics, not measured physical distances or structural safety guarantees.",
    }


def default_state() -> Dict[str, Any]:
    return {
        "active_scene_id": "legacy",
        "active_scene_name": "Existing Workspace",
        "busy": False,
        "stage": "idle",
        "status": "Ready",
        "progress": 0,
        "job_id": None,
        "started_at": None,
        "finished_at": None,
        "updated_at": time.time(),
        "images_count": 0,
        "reconstruction_images_count": 0,
        "evidence_images_count": 0,
        "registered_images": 0,
        "points3d": 0,
        "observations": 0,
        "mean_track_length": 0.0,
        "mean_observations_per_image": 0.0,
        "mean_reprojection_error_px": 0.0,
        "camera_model": "—",
        "model_ready": False,
        "model_status": "NONE",
        "model_iteration": None,
        "model_size_bytes": 0,
        "model_source": None,
        "model_job_dir": None,
        "dataset_hash": None,
        "model_dataset_hash": None,
        "new_images_count": 0,
        "last_error": None,
        "analysis": default_analysis(),
        "triage": {
            "label": "PENDING",
            "score": None,
            "confidence": None,
            "severity": "Analysis not run",
            "directives": [],
            "reasons": [],
        },
        "recommendations": [],
        "route": default_route(),
        "site_profile": default_site_profile(),
        "images": [],
        "evidence": [],
        "log": [],
    }


STATE = default_state()
loaded = read_json(STATE_FILE, None)
if isinstance(loaded, dict):
    STATE.update(loaded)
STATE.setdefault("route", default_route())
STATE.setdefault("site_profile", default_site_profile())


def save_state() -> None:
    with _lock:
        STATE["updated_at"] = time.time()
        save_json(STATE_FILE, STATE)


def log(message: str) -> None:
    with _lock:
        STATE.setdefault("log", []).append({"time": time.time(), "message": message})
        STATE["log"] = STATE["log"][-300:]
    save_state()


def set_state(**changes: Any) -> None:
    with _lock:
        STATE.update(changes)
    save_state()


def scene_root(scene_id: str) -> Path:
    return SCENES_DIR / scene_id


def scene_image_dir(scene_id: str) -> Path:
    return LEGACY_IMAGE_DIR if scene_id == "legacy" else scene_root(scene_id) / "images"


def scene_evidence_dir(scene_id: str) -> Path:
    return EVIDENCE_DIR / "legacy" if scene_id == "legacy" else scene_root(scene_id) / "evidence"


def scene_model_root(scene_id: str) -> Path:
    return WORKSPACE / "gaussian_output" if scene_id == "legacy" else scene_root(scene_id) / "gaussian_output"


def scene_dense_root(scene_id: str) -> Path:
    return WORKSPACE / "dense" if scene_id == "legacy" else scene_root(scene_id) / "dense"


def ensure_scene(scene_id: str) -> None:
    scene_image_dir(scene_id).mkdir(parents=True, exist_ok=True)
    scene_evidence_dir(scene_id).mkdir(parents=True, exist_ok=True)
    if scene_id != "legacy":
        scene_root(scene_id).mkdir(parents=True, exist_ok=True)


def scene_meta(scene_id: str) -> Dict[str, str]:
    if scene_id == "legacy":
        return {
            "id": "legacy",
            "name": "Existing Workspace",
            "description": "Primary AeroSplat workspace.",
            "disaster_type": "Existing workspace",
            "location": "",
        }
    metadata = read_json(scene_root(scene_id) / "scene.json", {})
    return {
        "id": scene_id,
        "name": metadata.get("name", scene_id),
        "description": metadata.get("description", ""),
        "disaster_type": metadata.get("disaster_type", "Unknown"),
        "location": metadata.get("location", ""),
    }


def list_scenes() -> List[Dict[str, str]]:
    ensure_scene("legacy")
    result = [scene_meta("legacy")]
    for path in sorted(SCENES_DIR.iterdir(), key=lambda p: p.name.lower()):
        if path.is_dir():
            result.append(scene_meta(path.name))
    return result


def active_scene_id() -> str:
    scene_id = STATE.get("active_scene_id") or "legacy"
    ensure_scene(scene_id)
    return scene_id


def active_images() -> List[Path]:
    return image_files(scene_image_dir(active_scene_id()))


def active_evidence() -> List[Path]:
    return image_files(scene_evidence_dir(active_scene_id()))


def latest_model(root: Path) -> Optional[Dict[str, Any]]:
    """Return the highest valid GraphDeco PLY for this scene only.

    We deliberately do not search JOBS_DIR here. A completed job is copied
    into the scene's gaussian_output by the pipeline, and a global job search
    can accidentally make one disaster site display another site's/test model.
    """
    candidates: List[Tuple[int, Path, Path]] = []

    point_root = root / "point_cloud"
    if point_root.exists():
        for iteration_dir in point_root.glob("iteration_*"):
            if not iteration_dir.is_dir():
                continue
            try:
                iteration = int(iteration_dir.name.rsplit("_", 1)[-1])
            except ValueError:
                continue

            ply = iteration_dir / "point_cloud.ply"
            if not ply.exists():
                continue

            try:
                size = ply.stat().st_size
            except FileNotFoundError:
                continue

            if size < 1024:
                continue

            candidates.append((iteration, ply, root))

    if candidates:
        iteration, ply, base = max(candidates, key=lambda item: item[0])
        return {
            "iteration": iteration,
            "path": ply,
            "root": base,
            "size": ply.stat().st_size,
        }

    # Legacy-only fallback for the older 100-iteration test model.
    # This is used only when the authoritative gaussian_output has no model.
    if root == scene_model_root("legacy"):
        test_root = WORKSPACE / "gaussian_test"
        test_point_root = test_root / "point_cloud"
        test_candidates: List[Tuple[int, Path, Path]] = []

        if test_point_root.exists():
            for iteration_dir in test_point_root.glob("iteration_*"):
                if not iteration_dir.is_dir():
                    continue
                try:
                    iteration = int(iteration_dir.name.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                ply = iteration_dir / "point_cloud.ply"
                if not ply.exists():
                    continue
                try:
                    size = ply.stat().st_size
                except FileNotFoundError:
                    continue
                if size < 1024:
                    continue
                test_candidates.append((iteration, ply, test_root))

        if test_candidates:
            iteration, ply, base = max(test_candidates, key=lambda item: item[0])
            return {
                "iteration": iteration,
                "path": ply,
                "root": base,
                "size": ply.stat().st_size,
            }

    return None


def parse_model_stats(text: str) -> Dict[str, Any]:
    patterns = {
        "registered_images": r"Registered images:\s+(\d+)",
        "points3d": r"Points:\s+(\d+)",
        "observations": r"Observations:\s+(\d+)",
        "mean_track_length": r"Mean track length:\s+([\d.]+)",
        "mean_observations_per_image": r"Mean observations per image:\s+([\d.]+)",
        "mean_reprojection_error_px": r"Mean reprojection error:\s+([\d.]+)px",
    }
    result: Dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            continue
        result[key] = (
            int(match.group(1))
            if key in {"registered_images", "points3d", "observations"}
            else float(match.group(1))
        )
    return result


def run_command(
    command: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    stage: Optional[str] = None,
) -> str:
    pretty = " ".join(f'"{item}"' if " " in item else item for item in command)
    log("$ " + pretty)

    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Executable not found: {command[0]}") from exc

    lines: List[str] = []
    assert process.stdout is not None

    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        lines.append(line)

        if stage == "3dgs":
            match = re.search(r"(\d+)%", line)
            if match and "Training progress" in line:
                percent = int(match.group(1))
                set_state(
                    stage="3dgs",
                    status=f"Training 3DGS — {percent}%",
                    progress=60 + int(percent * 0.40),
                )

        elif stage == "colmap_mapper" and "Keeping successful reconstruction" in line:
            set_state(status="Sparse reconstruction accepted", progress=45)

    code = process.wait()
    output = "\n".join(lines)

    if code != 0:
        tail = "\n".join(lines[-100:])
        raise RuntimeError(
            f"Command failed with exit code {code}:\n{tail}"
        )

    return output


def read_camera_model(model_dir: Path) -> str:
    text_dir = model_dir.parent / f"{model_dir.name}_txt"
    text_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_command([
            str(COLMAP_EXE),
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(text_dir),
            "--output_type",
            "TXT",
        ])
        cameras_file = text_dir / "cameras.txt"
        if cameras_file.exists():
            for line in cameras_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]
    except Exception as exc:
        log(f"Camera model read warning: {exc}")
    return "—"


def find_sparse_model(root: Path) -> Path:
    direct = root
    nested = root / "0"
    if all((direct / filename).exists() for filename in ("cameras.bin", "images.bin", "points3D.bin")):
        return direct
    if all((nested / filename).exists() for filename in ("cameras.bin", "images.bin", "points3D.bin")):
        return nested
    raise RuntimeError(f"No valid COLMAP sparse model under {root}")


def prepare_graphdeco_source(dense: Path) -> Path:
    # GraphDeco expects dense/sparse/0 while current COLMAP 4.x undistorter
    # may produce dense/sparse directly.
    sparse = dense / "sparse"
    sparse0 = sparse / "0"
    sparse0.mkdir(parents=True, exist_ok=True)

    for source in sparse.iterdir():
        if source.is_file() and source.suffix.lower() == ".bin":
            target = sparse0 / source.name
            if not target.exists() or source.stat().st_mtime_ns > target.stat().st_mtime_ns:
                shutil.copy2(source, target)

    required = ("cameras.bin", "images.bin", "points3D.bin")
    missing = [name for name in required if not (sparse0 / name).exists()]
    if missing:
        raise RuntimeError("GraphDeco source is missing: " + ", ".join(missing))

    return dense


def scene_snapshot(job_dir: Path, scene_id: str) -> Tuple[Path, List[Path]]:
    raw_images = job_dir / "images"
    raw_images.mkdir(parents=True, exist_ok=True)
    files = image_files(scene_image_dir(scene_id))

    for source in files:
        target = raw_images / source.name
        shutil.copy2(source, target)

    return raw_images, [raw_images / file.name for file in files]


def parse_points3d_bin(path: Path, max_points: int = 120_000) -> np.ndarray:
    points = []
    try:
        with path.open("rb") as stream:
            first = stream.read(8)
            if len(first) != 8:
                return np.empty((0, 3), dtype=np.float64)
            total = struct.unpack("<Q", first)[0]
            keep = min(total, max_points)

            for index in range(total):
                record_header = stream.read(51)
                if len(record_header) != 51:
                    break

                # POINT3D_ID (8), XYZ (24), RGB (3), error (8), track length (8)
                _, x, y, z, _, _, _, track_length = struct.unpack(
                    "<QdddBBBdQ", record_header
                )

                # Each track element is IMAGE_ID + POINT2D_IDX = 16 bytes.
                stream.seek(track_length * 16, os.SEEK_CUR)

                if index < keep and math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                    points.append((x, y, z))
    except Exception as exc:
        log(f"points3D parser warning: {exc}")
        return np.empty((0, 3), dtype=np.float64)

    return np.asarray(points, dtype=np.float64)


def quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def read_colmap_images_bin(path: Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            return result
        image_count = struct.unpack("<Q", raw)[0]

        for _ in range(image_count):
            raw = stream.read(8 + 8 * 4 + 8 * 3 + 4)
            if len(raw) != 52:
                break

            image_id = struct.unpack_from("<Q", raw, 0)[0]
            qw, qx, qy, qz = struct.unpack_from("<dddd", raw, 8)
            tx, ty, tz = struct.unpack_from("<ddd", raw, 40)
            camera_id = struct.unpack_from("<I", raw, 48)[0]

            name_bytes = bytearray()
            while True:
                byte = stream.read(1)
                if not byte:
                    break
                if byte == b"\x00":
                    break
                name_bytes.extend(byte)
            name = name_bytes.decode("utf-8", errors="replace")

            raw = stream.read(8)
            if len(raw) != 8:
                break
            point_count = struct.unpack("<Q", raw)[0]
            stream.seek(point_count * 24, os.SEEK_CUR)

            rotation = quaternion_to_rotation(qw, qx, qy, qz)
            translation = np.array([tx, ty, tz], dtype=np.float64)
            position = -(rotation.T @ translation)
            forward = rotation.T @ np.array([0.0, 0.0, 1.0])
            forward_norm = np.linalg.norm(forward)
            if forward_norm > 0:
                forward = forward / forward_norm
            target = position + forward

            result[name] = {
                "image_id": int(image_id),
                "camera_id": int(camera_id),
                "position": position.tolist(),
                "target": target.tolist(),
            }

    return result


def parse_binary_ply_header(stream) -> Dict[str, Any]:
    header_lines: List[str] = []
    while True:
        line = stream.readline()
        if not line:
            raise RuntimeError("PLY header is incomplete.")
        decoded = line.decode("ascii", errors="ignore").strip()
        header_lines.append(decoded)
        if decoded == "end_header":
            break

    if not header_lines or header_lines[0] != "ply":
        raise RuntimeError("Not a PLY file.")
    if not any(line == "format binary_little_endian 1.0" for line in header_lines):
        raise RuntimeError("Only binary little-endian PLY is supported for server-side bounds parsing.")

    vertex_count = None
    properties: List[Tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
            continue
        if parts[:2] == ["element", "face"]:
            in_vertex = False
        if in_vertex and len(parts) == 3 and parts[0] == "property":
            properties.append((parts[1], parts[2]))

    if vertex_count is None:
        raise RuntimeError("PLY has no vertex element.")

    return {"vertex_count": vertex_count, "properties": properties}


def ply_bounds(path: Path) -> Dict[str, Any]:
    scalar_formats = {
        "char": ("b", 1), "uchar": ("B", 1), "short": ("h", 2), "ushort": ("H", 2),
        "int": ("i", 4), "uint": ("I", 4), "float": ("f", 4), "double": ("d", 8),
    }

    with path.open("rb") as stream:
        header = parse_binary_ply_header(stream)
        properties = header["properties"]
        property_offsets: Dict[str, Tuple[str, int]] = {}
        stride = 0
        for data_type, name in properties:
            fmt, size = scalar_formats.get(data_type, (None, None))
            if fmt is None:
                raise RuntimeError(f"Unsupported PLY property type: {data_type}")
            property_offsets[name] = (fmt, stride)
            stride += size

        if not all(key in property_offsets for key in ("x", "y", "z")):
            raise RuntimeError("PLY is missing x/y/z properties.")

        xyz_offsets = [property_offsets[key][1] for key in ("x", "y", "z")]
        xyz_formats = [property_offsets[key][0] for key in ("x", "y", "z")]
        vertex_count = int(header["vertex_count"])

        min_xyz = np.array([math.inf, math.inf, math.inf], dtype=np.float64)
        max_xyz = np.array([-math.inf, -math.inf, -math.inf], dtype=np.float64)

        chunk_vertices = 50_000
        unpackers = [struct.Struct("<" + fmt) for fmt in xyz_formats]

        for chunk_start in range(0, vertex_count, chunk_vertices):
            count = min(chunk_vertices, vertex_count - chunk_start)
            data = stream.read(count * stride)
            if len(data) != count * stride:
                raise RuntimeError("PLY vertex data ended unexpectedly.")

            xs = np.empty(count, dtype=np.float64)
            ys = np.empty(count, dtype=np.float64)
            zs = np.empty(count, dtype=np.float64)

            for idx in range(count):
                base = idx * stride
                xs[idx] = unpackers[0].unpack_from(data, base + xyz_offsets[0])[0]
                ys[idx] = unpackers[1].unpack_from(data, base + xyz_offsets[1])[0]
                zs[idx] = unpackers[2].unpack_from(data, base + xyz_offsets[2])[0]

            chunk = np.column_stack([xs, ys, zs])
            chunk = chunk[np.isfinite(chunk).all(axis=1)]
            if len(chunk):
                min_xyz = np.minimum(min_xyz, chunk.min(axis=0))
                max_xyz = np.maximum(max_xyz, chunk.max(axis=0))

    if not np.isfinite(min_xyz).all() or not np.isfinite(max_xyz).all():
        raise RuntimeError("PLY has no finite vertices.")

    center = ((min_xyz + max_xyz) / 2.0).tolist()
    extent = (max_xyz - min_xyz).tolist()
    radius = max(float(np.linalg.norm(max_xyz - min_xyz) / 2.0), 0.5)

    return {
        "min": min_xyz.tolist(),
        "max": max_xyz.tolist(),
        "center": center,
        "extent": extent,
        "radius": radius,
    }


def analyze_images(files: List[Path]) -> Dict[str, Any]:
    sharpness: List[float] = []
    brightness: List[float] = []
    crack_ratios: List[float] = []
    blurred = 0
    dark_frames = 0
    candidates = 0

    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        height, width = image.shape[:2]
        scale = min(1.0, 1200 / max(width, height))
        if scale < 1:
            image = cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        variance = float(cv2.Laplacian(image, cv2.CV_64F).var())
        mean_brightness = float(np.mean(image))
        sharpness.append(variance)
        brightness.append(mean_brightness)

        if variance < 80:
            blurred += 1
        if mean_brightness < 35:
            dark_frames += 1

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
        edges = cv2.Canny(clahe, 50, 140)
        dark = cv2.GaussianBlur(clahe, (0, 0), 2.0) < np.percentile(clahe, 22)
        ratio = float(np.logical_and(edges > 0, dark).mean())
        crack_ratios.append(ratio)
        if ratio > 0.012:
            candidates += 1

    analyzed = len(sharpness)
    return {
        "quality": {
            "images_analyzed": analyzed,
            "mean_sharpness": round(float(np.mean(sharpness)), 3) if sharpness else None,
            "blurred_images": blurred,
            "mean_brightness": round(float(np.mean(brightness)), 3) if brightness else None,
            "dark_frame_ratio": round(dark_frames / analyzed, 4) if analyzed else None,
        },
        "crack_visual_indicator": {
            "status": "VISUAL BASELINE",
            "candidate_images": candidates,
            "mean_candidate_ratio": round(float(np.mean(crack_ratios)), 5) if crack_ratios else None,
        },
    }


def analyze_geometry(sparse_model: Path) -> Dict[str, Any]:
    points = parse_points3d_bin(sparse_model / "points3D.bin")
    if len(points) < 50:
        return {
            "status": "INSUFFICIENT_POINTS",
            "dominant_plane_tilt_deg": None,
            "points_sampled": int(len(points)),
        }

    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > 25_000:
        indices = np.linspace(0, len(points) - 1, 25_000).astype(int)
        points = points[indices]

    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    _, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, np.argmin(np.linalg.eigvalsh(covariance))]
    z = min(1.0, max(0.0, abs(float(normal[2]))))
    angle = math.degrees(math.acos(z))

    return {
        "status": "PROVISIONAL SCENE GEOMETRY",
        "dominant_plane_tilt_deg": round(angle, 2),
        "points_sampled": int(len(points)),
        "note": "Dominant-plane inclination only; not a certified wall-tilt measurement.",
    }


def compute_analysis(scene_id: str) -> Dict[str, Any]:
    files = image_files(scene_image_dir(scene_id))
    analysis = default_analysis()
    analysis["status"] = "COMPLETE"
    analysis.update(analyze_images(files))

    sparse_candidates = [
        scene_dense_root(scene_id) / "sparse",
        scene_dense_root(scene_id),
        WORKSPACE / "sparse" / "0" if scene_id == "legacy" else scene_root(scene_id) / "sparse" / "0",
    ]

    sparse_model = next(
        (candidate for candidate in sparse_candidates if (candidate / "points3D.bin").exists()),
        None,
    )

    if sparse_model:
        analysis["geometry_indicator"] = analyze_geometry(sparse_model)
    else:
        analysis["geometry_indicator"] = {
            "status": "NO_SPARSE_MODEL",
            "dominant_plane_tilt_deg": None,
            "points_sampled": 0,
        }

    analysis["notes"] = [
        "Crack indicator is a visual baseline for prioritizing images; it is not crack metrology.",
        "Dominant-plane inclination is a geometry indicator; engineering wall tilt requires identified wall surfaces and calibrated coordinates.",
        "Rubble volume remains unavailable until rubble segmentation and metric 3D reconstruction are connected.",
    ]
    return analysis


def make_triage(analysis: Dict[str, Any]) -> Dict[str, Any]:
    quality = analysis.get("quality", {})
    crack = analysis.get("crack_visual_indicator", {})
    geometry = analysis.get("geometry_indicator", {})

    score = 0.0
    reasons: List[str] = []
    directives: List[str] = []

    analyzed = int(quality.get("images_analyzed") or 0)
    blurred = int(quality.get("blurred_images") or 0)
    dark_ratio = float(quality.get("dark_frame_ratio") or 0.0)
    candidate_ratio = float(crack.get("mean_candidate_ratio") or 0.0)
    tilt = geometry.get("dominant_plane_tilt_deg")

    if analyzed and blurred > max(3, analyzed * 0.25):
        score += 0.15
        reasons.append("High fraction of low-sharpness imagery")
        directives.append("Acquire additional sharp close-range imagery.")

    if dark_ratio > 0.15:
        score += 0.10
        reasons.append("A meaningful fraction of frames are underexposed")
        directives.append("Review or reacquire underexposed frames before structural inference.")

    if candidate_ratio > 0.018:
        score += 0.30
        reasons.append("Elevated crack-like visual texture")
        directives.append("Prioritize frames flagged for crack-like regions.")

    if tilt is not None:
        if tilt > 8:
            score += 0.25
            reasons.append("Dominant scene-plane inclination is elevated")
            directives.append("Validate suspected inclined surfaces with calibrated geometry.")
        elif tilt > 4:
            score += 0.10
            reasons.append("Dominant scene-plane inclination is non-negligible")
            directives.append("Review dominant surfaces against known vertical references.")

    score = min(1.0, score)
    label = "RED" if score >= 0.70 else "YELLOW" if score >= 0.35 else "GREEN"
    directives = directives or ["No high-priority baseline indicator triggered; continue inspection."]

    return {
        "label": label,
        "score": round(score, 2),
        "confidence": None,
        "severity": "Preliminary evidence-based screening only",
        "directives": directives,
        "reasons": reasons,
        "disclaimer": "This screening does not certify structural safety or replace qualified inspection.",
    }


def sync_existing() -> None:
    scene_id = active_scene_id()
    images = image_files(scene_image_dir(scene_id))
    evidence = image_files(scene_evidence_dir(scene_id))
    dataset = file_hash(images)
    model = latest_model(scene_model_root(scene_id))

    with _lock:
        previous_model_hash = STATE.get("model_dataset_hash")
        previous_model_iteration = STATE.get("model_iteration")
        previous_model_source = STATE.get("model_source")

    if model:
        # Preserve a known dataset hash when possible; otherwise consider the
        # current dataset the model baseline at startup.
        if (
            previous_model_hash
            and previous_model_iteration == model["iteration"]
            and previous_model_source
        ):
            model_hash = previous_model_hash
        else:
            model_hash = dataset

        status = "CURRENT" if model_hash == dataset else "STALE"
        set_state(
            model_ready=True,
            model_status=status,
            model_iteration=model["iteration"],
            model_size_bytes=model["size"],
            model_source=str(model["root"]),
            model_job_dir=str(model["root"]),
            model_dataset_hash=model_hash,
            dataset_hash=dataset,
            images_count=len(images),
            reconstruction_images_count=len(images),
            evidence_images_count=len(evidence),
        )
    else:
        set_state(
            model_ready=False,
            model_status="NONE",
            model_iteration=None,
            model_size_bytes=0,
            model_source=None,
            model_job_dir=None,
            model_dataset_hash=None,
            dataset_hash=dataset,
            images_count=len(images),
            reconstruction_images_count=len(images),
            evidence_images_count=len(evidence),
        )

    set_state(
        site_profile=scene_profile(scene_id),
        route=load_route_plan(scene_id),
    )

    try:
        analysis = compute_analysis(scene_id) if images else default_analysis()
        triage = make_triage(analysis)
        set_state(
            analysis=analysis,
            triage=triage,
            recommendations=triage.get("directives", []),
        )
    except Exception as exc:
        log(f"Analysis startup warning: {exc}")


def pipeline(job_id: str, scene_id: str, iterations: int, source_hash: Optional[str]) -> None:
    job = JOBS_DIR / job_id
    raw_images, source_files = scene_snapshot(job, scene_id)
    database = job / "database.db"
    sparse = job / "sparse"
    dense = job / "dense"
    output = job / "gaussian_model"

    try:
        if len(source_files) < 3:
            raise RuntimeError("At least 3 overlapping reconstruction images are required.")
        if not COLMAP_EXE.exists():
            raise RuntimeError(f"COLMAP executable not found: {COLMAP_EXE}")
        if not TRAIN_PY.exists():
            raise RuntimeError(f"GraphDeco train.py not found: {TRAIN_PY}")

        set_state(stage="colmap_features", status=f"Extracting features from {len(source_files)} images…", progress=2)
        run_command([
            str(COLMAP_EXE), "feature_extractor",
            "--database_path", str(database),
            "--image_path", str(raw_images),
            "--ImageReader.camera_model", "SIMPLE_RADIAL",
            "--ImageReader.single_camera", "1",
            "--FeatureExtraction.use_gpu", "0",
            "--FeatureExtraction.max_image_size", "2400",
            "--FeatureExtraction.num_threads", "8",
        ], stage="colmap_features")

        set_state(stage="colmap_match", status="Matching overlapping views…", progress=15)
        run_command([
            str(COLMAP_EXE), "sequential_matcher",
            "--database_path", str(database),
            "--SequentialMatching.overlap", "10",
            "--FeatureMatching.use_gpu", "0",
        ], stage="colmap_match")

        sparse.mkdir(parents=True, exist_ok=True)
        set_state(stage="colmap_mapper", status="Building sparse reconstruction…", progress=25)
        run_command([
            str(COLMAP_EXE), "mapper",
            "--database_path", str(database),
            "--image_path", str(raw_images),
            "--output_path", str(sparse),
        ], stage="colmap_mapper")

        model = find_sparse_model(sparse)
        analyzer_output = run_command([
            str(COLMAP_EXE), "model_analyzer", "--path", str(model)
        ])
        stats = parse_model_stats(analyzer_output)
        camera_model = read_camera_model(model)
        if stats.get("registered_images", 0) < 3:
            raise RuntimeError("COLMAP registered fewer than 3 images.")

        set_state(
            **stats,
            camera_model=camera_model,
            stage="undistort",
            status="Undistorting for GraphDeco…",
            progress=48,
        )

        run_command([
            str(COLMAP_EXE), "image_undistorter",
            "--image_path", str(raw_images),
            "--input_path", str(model),
            "--output_path", str(dense),
            "--output_type", "COLMAP",
        ])

        graph_source = prepare_graphdeco_source(dense)

        env = os.environ.copy()
        cuda_home = env.get(
            "AEROSPLAT_CUDA_HOME",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        )
        env["CUDA_HOME"] = cuda_home
        env["PATH"] = str(Path(cuda_home) / "bin") + os.pathsep + env.get("PATH", "")
        env["TORCH_CUDA_ARCH_LIST"] = env.get("TORCH_CUDA_ARCH_LIST", "8.6")
        env["MAX_JOBS"] = str(BUILD_MAX_JOBS)
        env["DISTUTILS_USE_SDK"] = "1"

        set_state(stage="3dgs", status="Training 3DGS — 0%", progress=60)
        run_command([
            sys.executable,
            str(TRAIN_PY),
            "-s", str(graph_source),
            "-m", str(output),
            "--iterations", str(iterations),
            "--save_iterations", str(iterations),
            "--disable_viewer",
            "-r", str(DEFAULT_RESOLUTION),
            "--data_device", "cpu",
        ], cwd=GAUSSIAN_DIR, env=env, stage="3dgs")

        trained_ply = output / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
        if not trained_ply.exists():
            raise RuntimeError("3DGS completed but point_cloud.ply was not created.")

        target = scene_model_root(scene_id) / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trained_ply, target)

        current_images = image_files(scene_image_dir(scene_id))
        current_hash = file_hash(current_images)
        analysis = compute_analysis(scene_id)
        triage = make_triage(analysis)

        set_state(
            stage="ready",
            status="Reconstruction and baseline analysis complete",
            progress=100,
            busy=False,
            finished_at=time.time(),
            model_ready=True,
            model_status="CURRENT" if current_hash == source_hash else "STALE",
            model_iteration=iterations,
            model_size_bytes=target.stat().st_size,
            model_source="AeroSplat reconstruction job",
            model_job_dir=str(target.parent.parent.parent),
            model_dataset_hash=source_hash,
            dataset_hash=current_hash,
            reconstruction_images_count=len(current_images),
            images_count=len(current_images),
            evidence_images_count=len(active_evidence()),
            new_images_count=max(0, len(current_images) - len(source_files)),
            analysis=analysis,
            triage=triage,
            recommendations=triage.get("directives", []),
            last_error=None,
        )
        log(f"Reconstruction complete: {target}")
    except Exception as exc:
        set_state(
            busy=False,
            stage="error",
            status="Pipeline failed",
            last_error=str(exc),
            finished_at=time.time(),
            model_status="STALE" if STATE.get("model_ready") else "NONE",
        )
        log("ERROR: " + str(exc))


def start_pipeline(iterations: int) -> str:
    global _thread
    with _lock:
        if STATE.get("busy"):
            raise RuntimeError("A reconstruction job is already running.")

    scene_id = active_scene_id()
    files = image_files(scene_image_dir(scene_id))
    if len(files) < 3:
        raise RuntimeError("Add at least 3 overlapping reconstruction images first.")

    job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    dataset = file_hash(files)
    current_model = STATE.get("model_dataset_hash")

    set_state(
        job_id=job_id,
        busy=True,
        stage="queued",
        status="Reconstruction queued",
        progress=0,
        started_at=time.time(),
        finished_at=None,
        last_error=None,
        model_status="BUILDING" if STATE.get("model_ready") else "BUILDING",
        dataset_hash=dataset,
        model_dataset_hash=current_model,
        new_images_count=max(0, len(files) - int(STATE.get("registered_images") or 0)),
    )

    _thread = threading.Thread(
        target=pipeline,
        args=(job_id, scene_id, int(iterations), dataset),
        daemon=True,
        name=f"aerosplat-{job_id}",
    )
    _thread.start()
    return job_id


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "api": "ok",
        "python": sys.executable,
        "colmap": str(COLMAP_EXE),
        "colmap_exists": COLMAP_EXE.exists(),
        "graphdeco": str(TRAIN_PY),
        "graphdeco_exists": TRAIN_PY.exists(),
        "workspace": str(WORKSPACE),
        "active_scene": active_scene_id(),
    }


@app.get("/api/scenes")
def api_scenes() -> Dict[str, Any]:
    return {
        "active_scene_id": active_scene_id(),
        "scenes": list_scenes(),
    }


@app.post("/api/scenes")
def create_scene(name: str = Query(..., min_length=1), description: str = Query(""), disaster_type: str = Query("Unknown"), location: str = Query("")):
    if STATE.get("busy"):
        raise HTTPException(409, "Current reconstruction is running.")

    scene_id = "incident_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    ensure_scene(scene_id)
    save_json(
        scene_root(scene_id) / "scene.json",
        {
            "id": scene_id,
            "name": name.strip(),
            "description": description.strip(),
            "disaster_type": disaster_type.strip() or "Unknown",
            "location": location.strip(),
            "created_at": time.time(),
        },
    )
    set_state(
        active_scene_id=scene_id,
        active_scene_name=name.strip(),
        model_ready=False,
        model_status="NONE",
        model_dataset_hash=None,
        model_job_dir=None,
        model_iteration=None,
        new_images_count=0,
        analysis=default_analysis(),
        triage=default_state()["triage"],
        recommendations=[],
        route=default_route(),
        site_profile=scene_profile(scene_id),
    )
    sync_existing()
    return {"scene": scene_meta(scene_id)}


@app.post("/api/scenes/select")
def select_scene(scene_id: str = Query(...)):
    if STATE.get("busy"):
        raise HTTPException(409, "Current reconstruction is running.")
    if scene_id != "legacy" and not scene_root(scene_id).exists():
        raise HTTPException(404, "Scene not found")

    ensure_scene(scene_id)
    set_state(
        active_scene_id=scene_id,
        active_scene_name=scene_meta(scene_id)["name"],
        new_images_count=0,
        model_ready=False,
        model_status="NONE",
        model_dataset_hash=None,
        model_job_dir=None,
        model_iteration=None,
        route=default_route(),
        site_profile=scene_profile(scene_id),
    )
    sync_existing()
    return {"scene": scene_meta(scene_id)}


@app.post("/api/upload-images")
async def upload_images(
    files: List[UploadFile] = File(...),
    handling: str = Query("append"),
):
    if STATE.get("busy"):
        raise HTTPException(409, "Wait for the current reconstruction to finish.")

    handling = handling.lower().strip()
    if handling not in {"append", "evidence"}:
        raise HTTPException(400, "handling must be append or evidence")

    destination = (
        scene_image_dir(active_scene_id())
        if handling == "append"
        else scene_evidence_dir(active_scene_id())
    )
    destination.mkdir(parents=True, exist_ok=True)

    added: List[str] = []
    rejected: List[Dict[str, str]] = []

    for upload in files:
        original_name = safe_name(upload.filename or "")
        extension = Path(original_name).suffix.lower()
        if not original_name or extension not in {x.lower() for x in VALID_EXTS}:
            rejected.append({"name": original_name, "reason": "Unsupported image format"})
            await upload.close()
            continue

        target = destination / original_name
        if target.exists():
            target = destination / (
                f"{Path(original_name).stem}_{uuid.uuid4().hex[:5]}{extension}"
            )

        total = 0
        try:
            with target.open("wb") as output:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_MB * 1024 * 1024:
                        target.unlink(missing_ok=True)
                        rejected.append({"name": original_name, "reason": "File too large"})
                        break
                    output.write(chunk)
            if target.exists():
                added.append(target.name)
        finally:
            await upload.close()

    images = active_images()
    evidence = active_evidence()
    dataset = file_hash(images)

    if handling == "append" and added:
        set_state(
            images_count=len(images),
            reconstruction_images_count=len(images),
            evidence_images_count=len(evidence),
            dataset_hash=dataset,
            new_images_count=len(added),
            model_status="STALE" if STATE.get("model_ready") else "NONE",
        )
    else:
        set_state(
            images_count=len(images),
            reconstruction_images_count=len(images),
            evidence_images_count=len(evidence),
            dataset_hash=dataset,
        )

    try:
        analysis = compute_analysis(active_scene_id())
        triage = make_triage(analysis)
        set_state(analysis=analysis, triage=triage, recommendations=triage.get("directives", []))
    except Exception as exc:
        log("Analysis upload warning: " + str(exc))

    log(f"Added {len(added)} image(s) as {handling}.")
    return {
        "added": added,
        "rejected": rejected,
        "handling": handling,
        "reconstruction_images_count": len(images),
        "evidence_images_count": len(evidence),
        "model_status": STATE.get("model_status"),
    }


@app.post("/api/analyze")
def analyze():
    analysis = compute_analysis(active_scene_id())
    triage = make_triage(analysis)
    set_state(
        analysis=analysis,
        triage=triage,
        recommendations=triage.get("directives", []),
    )
    return {
        "analysis": analysis,
        "triage": triage,
        "recommendations": triage.get("directives", []),
    }


@app.post("/api/run")
def run_pipeline(iterations: int = Query(DEFAULT_ITERATIONS, ge=100, le=30000)):
    try:
        return {
            "job_id": start_pipeline(iterations),
            "iterations": iterations,
        }
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/state")
def api_state():
    scene_id = active_scene_id()
    images = active_images()
    evidence = active_evidence()
    dataset = file_hash(images)

    with _lock:
        snapshot = json.loads(json.dumps(STATE))

    snapshot.update({
        "active_scene_id": scene_id,
        "active_scene_name": scene_meta(scene_id)["name"],
        "images_count": len(images),
        "reconstruction_images_count": len(images),
        "evidence_images_count": len(evidence),
        "dataset_hash": dataset,
    })

    snapshot["model_stale"] = bool(
        snapshot.get("model_ready")
        and snapshot.get("model_dataset_hash")
        and dataset != snapshot.get("model_dataset_hash")
    )

    if snapshot["model_stale"] and not snapshot.get("busy"):
        snapshot["model_status"] = "STALE"
    elif snapshot.get("model_ready") and not snapshot.get("busy"):
        snapshot["model_status"] = "CURRENT"

    snapshot["images"] = [
        {
            "name": path.name,
            "url": f"/api/scene-image/{scene_id}/{path.name}",
            "kind": "reconstruction",
        }
        for path in images
    ]
    snapshot["evidence"] = [
        {
            "name": path.name,
            "url": f"/api/evidence-image/{scene_id}/{path.name}",
            "kind": "evidence",
        }
        for path in evidence
    ]
    snapshot["model_url"] = "/api/model.ply" if snapshot.get("model_ready") else None
    snapshot["site_profile"] = scene_profile(scene_id)
    snapshot["route"] = load_route_plan(scene_id)

    return JSONResponse(snapshot, headers={"Cache-Control": "no-cache, no-store"})


@app.get("/api/model/info")
def model_info():
    with _lock:
        model_root = Path(STATE.get("model_job_dir") or "")
        iteration = STATE.get("model_iteration")
        ready = bool(STATE.get("model_ready"))

    if not ready or not model_root or iteration is None:
        raise HTTPException(404, "No trained Gaussian model available")

    candidates = [
        model_root / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
        model_root.parent.parent / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
    ]

    for path in candidates:
        if path.exists():
            try:
                with path.open("rb") as stream:
                    header = parse_binary_ply_header(stream)
                return {
                    "path": str(path),
                    "exists": True,
                    "size_bytes": path.stat().st_size,
                    "iteration": int(iteration),
                    "vertex_count": int(header["vertex_count"]),
                    "properties": header["properties"],
                    "bounds": ply_bounds(path),
                }
            except Exception as exc:
                raise HTTPException(500, f"Unable to inspect PLY: {exc}") from exc

    raise HTTPException(404, "Trained PLY not found")


@app.get("/api/model/bounds")
def model_bounds():
    with _lock:
        model_root = Path(STATE.get("model_job_dir") or "")
        iteration = STATE.get("model_iteration")

    if not model_root or iteration is None:
        raise HTTPException(404, "No trained Gaussian model available")

    candidates = [
        model_root / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
        model_root.parent.parent / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
    ]

    for ply in candidates:
        if ply.exists():
            try:
                return {
                    "iteration": int(iteration),
                    **ply_bounds(ply),
                }
            except Exception as exc:
                raise HTTPException(500, f"Unable to read PLY bounds: {exc}") from exc

    raise HTTPException(404, "Trained PLY not found")


@app.get("/api/camera/{filename:path}")
def camera_pose(filename: str):
    scene_id = active_scene_id()
    dense_sparse = scene_dense_root(scene_id) / "sparse"

    candidates = [dense_sparse, dense_sparse / "0"]
    model = next(
        (candidate for candidate in candidates if (candidate / "images.bin").exists()),
        None,
    )

    if model is None:
        # Existing legacy scene may still have the original sparse model.
        legacy = WORKSPACE / "sparse" / "0"
        if scene_id == "legacy" and (legacy / "images.bin").exists():
            model = legacy

    if model is None:
        raise HTTPException(404, "COLMAP camera poses are unavailable")

    try:
        poses = read_colmap_images_bin(model / "images.bin")
    except Exception as exc:
        raise HTTPException(500, f"Unable to read COLMAP camera poses: {exc}") from exc

    pose = poses.get(safe_name(filename))
    if pose is None:
        raise HTTPException(404, f"No COLMAP pose found for {filename}")

    return pose


@app.get("/api/scene-image/{sid}/{filename}")
def scene_image(sid: str, filename: str):
    if sid != "legacy" and not scene_root(sid).exists():
        raise HTTPException(404, "Scene not found")
    path = scene_image_dir(sid) / safe_name(filename)
    if not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(str(path))


@app.get("/api/evidence-image/{sid}/{filename}")
def evidence_image(sid: str, filename: str):
    path = scene_evidence_dir(sid) / safe_name(filename)
    if not path.exists():
        raise HTTPException(404, "Evidence image not found")
    return FileResponse(str(path))


@app.get("/api/model.ply")
def model_ply():
    with _lock:
        model_root = Path(STATE.get("model_job_dir") or "")
        iteration = STATE.get("model_iteration")
        ready = bool(STATE.get("model_ready"))

    if not ready or not model_root or iteration is None:
        raise HTTPException(404, "No trained Gaussian model available")

    candidates = [
        model_root / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
        model_root.parent.parent / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
    ]

    for path in candidates:
        if path.exists():
            return FileResponse(
                str(path),
                media_type="application/octet-stream",
                filename="aerosplat_model.ply",
            )

    raise HTTPException(404, "Trained PLY not found")


@app.post("/api/route/plan")
def api_route_plan(payload: Dict[str, Any] = Body(...)):
    scene_id = active_scene_id()
    analysis = STATE.get("analysis") or default_analysis()
    triage = STATE.get("triage") or default_state()["triage"]
    frame_name = safe_name(str(payload.get("frame_name") or "")) or None
    route = plan_route(payload, analysis, triage, frame_name)
    save_route_plan(scene_id, route)
    set_state(route=route)
    return {"route": route}


@app.get("/api/route")
def get_route():
    return load_route_plan(active_scene_id())


@app.delete("/api/route")
def clear_route():
    scene_id = active_scene_id()
    route = default_route()
    save_route_plan(scene_id, route)
    set_state(route=route)
    return {"route": route}


@app.get("/api/report")
def report():
    with _lock:
        return {
            "incident": STATE.get("active_scene_name"),
            "site_profile": scene_profile(active_scene_id()),
            "generated_at": time.time(),
            "reconstruction": {
                key: STATE.get(key)
                for key in [
                    "images_count",
                    "registered_images",
                    "points3d",
                    "observations",
                    "mean_track_length",
                    "mean_observations_per_image",
                    "mean_reprojection_error_px",
                    "camera_model",
                ]
            },
            "model": {
                key: STATE.get(key)
                for key in [
                    "model_ready",
                    "model_status",
                    "model_iteration",
                    "model_size_bytes",
                ]
            },
            "analysis": STATE.get("analysis"),
            "triage": STATE.get("triage"),
            "recommendations": STATE.get("recommendations"),
            "route": STATE.get("route"),
        }


sync_existing()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("AEROSPLAT_PORT", "8002")), reload=False)
