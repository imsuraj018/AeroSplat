import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

WORKSPACE = Path(os.getenv("AEROSPLAT_WORKSPACE", r"C:\aerosplat_workspace"))
LEGACY_IMAGE_DIR = WORKSPACE / "images"
SCENES_DIR = WORKSPACE / "scenes"
EVIDENCE_DIR = WORKSPACE / "evidence"
JOBS_DIR = WORKSPACE / "jobs"
STATE_FILE = WORKSPACE / "aerosplat_state.json"
COLMAP_EXE = Path(os.getenv("COLMAP_EXE", r"C:\bin\colmap.exe"))
GAUSSIAN_DIR = Path(os.getenv(
    "GAUSSIAN_SPLATTING_DIR",
    r"C:\Users\Admin\Downloads\AeroSplat_Upgraded\external\gaussian-splatting",
))
TRAIN_PY = GAUSSIAN_DIR / "train.py"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_MB = int(os.getenv("AEROSPLAT_MAX_UPLOAD_MB", "100"))
DEFAULT_ITERATIONS = int(os.getenv("AEROSPLAT_ITERATIONS", "7000"))
DEFAULT_RESOLUTION = int(os.getenv("AEROSPLAT_RESOLUTION", "2"))
BUILD_MAX_JOBS = int(os.getenv("AEROSPLAT_MAX_JOBS", "2"))

for directory in (WORKSPACE, LEGACY_IMAGE_DIR, SCENES_DIR, EVIDENCE_DIR, JOBS_DIR, STATIC_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AeroSplat Command Center", version="3.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_lock = threading.RLock()
_pipeline_thread: Optional[threading.Thread] = None


def _safe_name(name: str) -> str:
    return Path(name or "").name


def _hash_files(files: List[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(files, key=lambda x: x.name.lower()):
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        h.update(p.name.encode("utf-8", errors="ignore"))
        h.update(str(st.st_size).encode())
        h.update(str(st.st_mtime_ns).encode())
    return h.hexdigest()[:16]


def _image_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS],
        key=lambda p: p.name.lower(),
    )


def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _default_state() -> Dict[str, Any]:
    return {
        "active_scene_id": "legacy",
        "active_scene_name": "Existing Workspace",
        "job_id": None,
        "stage": "idle",
        "status": "Ready",
        "progress": 0,
        "busy": False,
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
        "damage": {
            "status": "Awaiting validated damage model",
            "crack_length_m": None,
            "crack_width_mm": None,
            "wall_tilt_deg": None,
            "rubble_volume_m3": None,
        },
        "triage": {
            "label": "PENDING",
            "severity": "Awaiting damage assessment",
            "confidence": None,
            "directives": ["Run reconstruction before making response decisions."],
        },
        "route": {
            "status": "Not computed",
            "distance_m": None,
            "eta_min": None,
            "blocked_zones": 0,
        },
        "log": [],
    }


STATE = _default_state()
loaded = _read_json(STATE_FILE, None)
if isinstance(loaded, dict):
    STATE.update(loaded)


def _save_state() -> None:
    with _lock:
        STATE["updated_at"] = time.time()
        _save_json(STATE_FILE, STATE)


def _log(message: str) -> None:
    with _lock:
        STATE.setdefault("log", []).append({"time": time.time(), "message": message})
        STATE["log"] = STATE["log"][-200:]
    _save_state()


def _set(**kwargs: Any) -> None:
    with _lock:
        STATE.update(kwargs)
    _save_state()


def _scene_root(scene_id: str) -> Path:
    return SCENES_DIR / scene_id


def _scene_image_dir(scene_id: str) -> Path:
    if scene_id == "legacy":
        return LEGACY_IMAGE_DIR
    return _scene_root(scene_id) / "images"


def _scene_evidence_dir(scene_id: str) -> Path:
    if scene_id == "legacy":
        return EVIDENCE_DIR / "legacy"
    return _scene_root(scene_id) / "evidence"


def _scene_model_root(scene_id: str) -> Path:
    if scene_id == "legacy":
        return WORKSPACE / "gaussian_output"
    return _scene_root(scene_id) / "gaussian_output"


def _ensure_scene_dirs(scene_id: str) -> None:
    _scene_image_dir(scene_id).mkdir(parents=True, exist_ok=True)
    _scene_evidence_dir(scene_id).mkdir(parents=True, exist_ok=True)
    if scene_id != "legacy":
        _scene_root(scene_id).mkdir(parents=True, exist_ok=True)


def _scene_meta(scene_id: str) -> Dict[str, Any]:
    if scene_id == "legacy":
        return {"id": "legacy", "name": "Existing Workspace", "description": "Existing AeroSplat workspace dataset."}
    data = _read_json(_scene_root(scene_id) / "scene.json", {})
    return {
        "id": scene_id,
        "name": data.get("name", scene_id),
        "description": data.get("description", ""),
    }


def _list_scenes() -> List[Dict[str, Any]]:
    items = [_scene_meta("legacy")]
    for p in sorted(SCENES_DIR.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_dir():
            continue
        items.append(_scene_meta(p.name))
    return items


def _active_scene_id() -> str:
    sid = STATE.get("active_scene_id") or "legacy"
    _ensure_scene_dirs(sid)
    return sid


def _active_image_dir() -> Path:
    return _scene_image_dir(_active_scene_id())


def _active_evidence_dir() -> Path:
    return _scene_evidence_dir(_active_scene_id())


def _find_latest_model(root: Path) -> Optional[Dict[str, Any]]:
    candidates = []
    roots = [root]
    if root == _scene_model_root("legacy"):
        roots.extend([WORKSPACE / "gaussian_test"])
    for job_root in JOBS_DIR.glob("*"):
        if job_root.is_dir():
            roots.append(job_root / "gaussian_model")
            roots.append(job_root)
    for base in roots:
        pc_root = base / "point_cloud"
        if not pc_root.exists():
            pc_root = base / "gaussian_model" / "point_cloud"
        if not pc_root.exists():
            continue
        for itdir in pc_root.glob("iteration_*"):
            if not itdir.is_dir():
                continue
            try:
                iteration = int(itdir.name.split("_")[-1])
            except ValueError:
                continue
            pc = itdir / "point_cloud.ply"
            if pc.exists():
                candidates.append((pc.stat().st_mtime, iteration, pc, base))
    if not candidates:
        return None
    _, iteration, pc, base = max(candidates, key=lambda x: x[0])
    return {"iteration": iteration, "path": pc, "root": base, "size": pc.stat().st_size}


def _parse_model_stats(text: str) -> Dict[str, Any]:
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
        m = re.search(pattern, text)
        if not m:
            continue
        value = m.group(1)
        result[key] = int(value) if key in {"registered_images", "points3d", "observations"} else float(value)
    return result


def _find_colmap_model(root: Path) -> Path:
    if all((root / f).exists() for f in ("cameras.bin", "images.bin", "points3D.bin")):
        return root
    if (root / "0").exists() and all((root / "0" / f).exists() for f in ("cameras.bin", "images.bin", "points3D.bin")):
        return root / "0"
    dirs = sorted([p for p in root.glob("*") if p.is_dir() and p.name.isdigit()], key=lambda x: x.name)
    for p in dirs:
        if all((p / f).exists() for f in ("cameras.bin", "images.bin", "points3D.bin")):
            return p
    raise RuntimeError(f"No valid COLMAP sparse model found under {root}")


def _read_camera_model(model_dir: Path) -> str:
    txt_dir = model_dir.parent / f"{model_dir.name}_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run([
            str(COLMAP_EXE), "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(txt_dir),
            "--output_type", "TXT",
        ])
        p = txt_dir / "cameras.txt"
        if p.exists():
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) > 1:
                        return parts[1]
    except Exception:
        pass
    return "—"


def _prepare_graphdeco_layout(dense_dir: Path) -> Path:
    sparse = dense_dir / "sparse"
    if not sparse.exists():
        raise RuntimeError("COLMAP undistortion did not create dense\\sparse")
    sparse0 = sparse / "0"
    sparse0.mkdir(parents=True, exist_ok=True)
    for item in sparse.iterdir():
        if item.is_file() and item.suffix.lower() == ".bin":
            shutil.copy2(item, sparse0 / item.name)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (sparse0 / name).exists():
            raise RuntimeError(f"Missing GraphDeco sparse file: {sparse0 / name}")
    return dense_dir


def _run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None, stage: Optional[str] = None) -> str:
    pretty = " ".join(f'"{x}"' if " " in x else x for x in cmd)
    _log("$ " + pretty)
    try:
        proc = subprocess.Popen(
            cmd,
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
        raise RuntimeError(f"Executable not found: {cmd[0]}") from exc

    output: List[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line:
            continue
        output.append(line)
        if stage == "3dgs":
            m = re.search(r"Training progress:\s*(\d+)%", line)
            if m:
                p = int(m.group(1))
                _set(stage="3dgs", status=f"Training 3DGS — {p}%", progress=60 + int(p * 0.4))
        elif stage == "colmap_features" and "Processed file" in line:
            _set(status=line, progress=5)
        elif stage == "colmap_match" and "Generating sequential image pairs" in line:
            _set(status="Matching overlapping views…", progress=18)
        elif stage == "colmap_mapper" and "Retriangulation" in line:
            _set(status="Refining sparse reconstruction…", progress=38)
        elif stage == "colmap_mapper" and "Keeping successful reconstruction" in line:
            _set(status="Sparse reconstruction accepted", progress=45)
    code = proc.wait()
    text = "\n".join(output)
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}:\n" + "\n".join(output[-50:]))
    return text


def _sync_existing_model() -> None:
    sid = _active_scene_id()
    image_dir = _scene_image_dir(sid)
    files = _image_files(image_dir)
    model = _find_latest_model(_scene_model_root(sid))
    current_hash = _hash_files(files) if files else None
    evidence_count = len(_image_files(_scene_evidence_dir(sid)))

    if not model:
        _set(
            model_ready=False,
            model_status="NONE",
            model_dataset_hash=None,
            images_count=len(files),
            reconstruction_images_count=len(files),
            evidence_images_count=evidence_count,
            dataset_hash=current_hash,
        )
        return

    # For an existing baseline model, the current dataset hash is used only if
    # we have not previously recorded a model hash. This avoids marking the
    # known-good existing model stale on first application launch.
    model_hash = STATE.get("model_dataset_hash") or current_hash
    _set(
        model_ready=True,
        model_status="CURRENT" if model_hash == current_hash else "STALE",
        model_iteration=model["iteration"],
        model_size_bytes=model["size"],
        model_source="existing workspace model",
        model_job_dir=str(model["root"]),
        model_dataset_hash=model_hash,
        dataset_hash=current_hash,
        images_count=len(files),
        reconstruction_images_count=len(files),
        evidence_images_count=evidence_count,
    )


_sync_existing_model()


def _copy_scene_images(source_dir: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for src in _image_files(source_dir):
        shutil.copy2(src, destination / src.name)


def _new_scene_id() -> str:
    return time.strftime("incident_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def _dataset_for_job(job: Path, scene_id: str):
    """Snapshot the active reconstruction images into an isolated job directory.

    The active dataset is copied at the moment the reconstruction starts, so the
    dashboard can continue to serve the previous model while COLMAP/3DGS works
    against a stable snapshot.
    """
    raw = job / "raw" / "images"
    source_dir = _scene_image_dir(scene_id)
    source_files = _image_files(source_dir)
    raw.mkdir(parents=True, exist_ok=True)
    for src in source_files:
        shutil.copy2(src, raw / src.name)
    copied = _image_files(raw)
    return raw, copied


def _pipeline(job_id: str, scene_id: str, iterations: int) -> None:
    job = JOBS_DIR / job_id
    raw, source_files = _dataset_for_job(job, scene_id)
    db = job / "database.db"
    sparse = job / "sparse"
    dense = job / "dense"
    model_out = job / "gaussian_model"
    try:
        if len(source_files) < 3:
            raise RuntimeError("At least 3 overlapping reconstruction images are required.")
        _set(stage="colmap_features", status=f"Extracting features from {len(source_files)} images…", progress=2)
        _run([
            str(COLMAP_EXE), "feature_extractor",
            "--database_path", str(db),
            "--image_path", str(raw),
            "--ImageReader.camera_model", "SIMPLE_RADIAL",
            "--ImageReader.single_camera", "1",
            "--FeatureExtraction.use_gpu", "0",
            "--FeatureExtraction.max_image_size", "2400",
            "--FeatureExtraction.num_threads", "8",
        ], stage="colmap_features")

        _set(stage="colmap_match", status="Matching overlapping views…", progress=12)
        _run([
            str(COLMAP_EXE), "sequential_matcher",
            "--database_path", str(db),
            "--SequentialMatching.overlap", "10",
            "--FeatureMatching.use_gpu", "0",
        ], stage="colmap_match")

        sparse.mkdir(parents=True, exist_ok=True)
        _set(stage="colmap_mapper", status="Building sparse reconstruction…", progress=25)
        _run([
            str(COLMAP_EXE), "mapper",
            "--database_path", str(db),
            "--image_path", str(raw),
            "--output_path", str(sparse),
        ], stage="colmap_mapper")

        model_dir = _find_colmap_model(sparse)
        analyzer = _run([str(COLMAP_EXE), "model_analyzer", "--path", str(model_dir)])
        stats = _parse_model_stats(analyzer)
        camera = _read_camera_model(model_dir)
        if stats.get("registered_images", 0) < 3:
            raise RuntimeError("COLMAP produced fewer than 3 registered images. Capture more overlapping views.")
        _set(**stats, camera_model=camera, stage="undistort", status="Undistorting for GraphDeco…", progress=48)
        if camera not in {"SIMPLE_RADIAL", "PINHOLE", "SIMPLE_PINHOLE", "—"}:
            raise RuntimeError(f"Unsupported COLMAP camera model: {camera}")

        _run([
            str(COLMAP_EXE), "image_undistorter",
            "--image_path", str(raw),
            "--input_path", str(model_dir),
            "--output_path", str(dense),
            "--output_type", "COLMAP",
        ], stage="undistort")
        graph_source = _prepare_graphdeco_layout(dense)

        env = os.environ.copy()
        env["CUDA_HOME"] = env.get("AEROSPLAT_CUDA_HOME", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")
        env["TORCH_CUDA_ARCH_LIST"] = env.get("TORCH_CUDA_ARCH_LIST", "8.6")
        env["MAX_JOBS"] = str(BUILD_MAX_JOBS)
        env["DISTUTILS_USE_SDK"] = "1"
        env["PATH"] = str(Path(env["CUDA_HOME"]) / "bin") + os.pathsep + env.get("PATH", "")

        _set(stage="3dgs", status=f"Training 3DGS — 0/{iterations}", progress=60)
        _run([
            sys.executable, str(TRAIN_PY),
            "-s", str(graph_source),
            "-m", str(model_out),
            "--iterations", str(iterations),
            "--save_iterations", str(iterations),
            "--disable_viewer",
            "-r", str(DEFAULT_RESOLUTION),
            "--data_device", "cpu",
        ], cwd=GAUSSIAN_DIR, env=env, stage="3dgs")

        pc = model_out / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
        if not pc.exists():
            raise RuntimeError(f"3DGS finished but {pc} was not created.")
        ds_hash = _hash_files(source_files)
        target_model_root = _scene_model_root(scene_id)
        target_model_root.mkdir(parents=True, exist_ok=True)
        target_pc = target_model_root / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
        target_pc.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pc, target_pc)
        # Keep job folder for logs/reproducibility; the scene's gaussian_output
        # contains the active model displayed by the UI.

        _set(
            stage="ready",
            status="Reconstruction complete",
            progress=100,
            busy=False,
            finished_at=time.time(),
            model_ready=True,
            model_status="CURRENT",
            model_iteration=iterations,
            model_size_bytes=target_pc.stat().st_size,
            model_source="AeroSplat reconstruction job",
            model_job_dir=str(target_model_root),
            model_dataset_hash=ds_hash,
            dataset_hash=ds_hash,
            reconstruction_images_count=len(source_files),
            images_count=len(source_files),
            evidence_images_count=len(_image_files(_scene_evidence_dir(scene_id))),
            new_images_count=0,
            last_error=None,
            damage={
                "status": "Awaiting validated damage model",
                "crack_length_m": None,
                "crack_width_mm": None,
                "wall_tilt_deg": None,
                "rubble_volume_m3": None,
            },
            triage={
                "label": "PENDING",
                "severity": "Reconstruction complete; damage model not configured",
                "confidence": None,
                "directives": [
                    "Use the 3D twin as spatial evidence.",
                    "Connect validated damage segmentation before safety decisions.",
                ],
            },
            route={
                "status": "Awaiting road/hazard graph",
                "distance_m": None,
                "eta_min": None,
                "blocked_zones": 0,
            },
        )
        _log(f"3DGS model ready for scene {scene_id}: {target_pc}")
    except Exception as exc:
        _set(busy=False, stage="error", status="Pipeline failed", last_error=str(exc), finished_at=time.time())
        _log(f"ERROR: {exc}")


def start_pipeline(iterations: int) -> str:
    global _pipeline_thread
    with _lock:
        if STATE.get("busy"):
            raise RuntimeError("A reconstruction job is already running.")
        scene_id = _active_scene_id()
    files = _image_files(_scene_image_dir(scene_id))
    if len(files) < 3:
        raise RuntimeError("Add at least 3 overlapping reconstruction images first.")
    job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    ds_hash = _hash_files(files)
    _set(
        job_id=job_id,
        busy=True,
        stage="queued",
        status="Reconstruction queued",
        progress=0,
        started_at=time.time(),
        finished_at=None,
        dataset_hash=ds_hash,
        new_images_count=0,
        last_error=None,
        model_status="BUILDING" if STATE.get("model_ready") else "BUILDING",
        # Crucially, retain model_ready/model_job_dir/model_iteration so the old
        # digital twin remains visible while the new one is being built.
    )
    _pipeline_thread = threading.Thread(
        target=_pipeline,
        args=(job_id, scene_id, iterations),
        daemon=True,
        name=f"aerosplat-{job_id}",
    )
    _pipeline_thread.start()
    return job_id


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "api": "ok",
        "colmap": str(COLMAP_EXE),
        "colmap_exists": COLMAP_EXE.exists(),
        "graphdeco": str(TRAIN_PY),
        "graphdeco_exists": TRAIN_PY.exists(),
        "workspace": str(WORKSPACE),
        "active_scene": _active_scene_id(),
    }


@app.get("/api/scenes")
def list_scenes() -> Dict[str, Any]:
    return {"active_scene_id": _active_scene_id(), "scenes": _list_scenes()}


@app.post("/api/scenes")
def create_scene(name: str = Query(..., min_length=1), description: str = Query("")) -> Dict[str, Any]:
    if STATE.get("busy"):
        raise HTTPException(409, "Wait until the current reconstruction finishes.")
    scene_id = _new_scene_id()
    root = _scene_root(scene_id)
    root.mkdir(parents=True, exist_ok=True)
    _ensure_scene_dirs(scene_id)
    _save_json(root / "scene.json", {"id": scene_id, "name": name.strip(), "description": description.strip()})
    with _lock:
        STATE["active_scene_id"] = scene_id
        STATE["active_scene_name"] = name.strip()
        STATE["model_ready"] = False
        STATE["model_status"] = "NONE"
        STATE["model_job_dir"] = None
        STATE["model_iteration"] = None
        STATE["model_size_bytes"] = 0
        STATE["model_dataset_hash"] = None
        STATE["new_images_count"] = 0
    _sync_existing_model()
    _log(f"Created and activated new scene: {scene_id}")
    return {"scene": _scene_meta(scene_id)}


@app.post("/api/scenes/select")
def select_scene(scene_id: str = Query(...)) -> Dict[str, Any]:
    if STATE.get("busy"):
        raise HTTPException(409, "Wait until the current reconstruction finishes.")
    if scene_id != "legacy" and not _scene_root(scene_id).exists():
        raise HTTPException(404, "Scene not found.")
    _ensure_scene_dirs(scene_id)
    meta = _scene_meta(scene_id)
    with _lock:
        STATE["active_scene_id"] = scene_id
        STATE["active_scene_name"] = meta["name"]
        STATE["new_images_count"] = 0
    _sync_existing_model()
    _log(f"Selected scene: {scene_id}")
    return {"scene": meta}


@app.post("/api/upload-images")
async def upload_images(
    files: List[UploadFile] = File(...),
    handling: str = Query("append"),
) -> Dict[str, Any]:
    if STATE.get("busy"):
        raise HTTPException(409, "Wait until the current reconstruction finishes before changing the dataset.")
    handling = handling.lower().strip()
    if handling not in {"append", "evidence"}:
        raise HTTPException(400, "handling must be 'append' or 'evidence'. Use /api/scenes for a new scene.")

    destination = _active_image_dir() if handling == "append" else _active_evidence_dir()
    destination.mkdir(parents=True, exist_ok=True)
    added, rejected = [], []

    for upload in files:
        name = _safe_name(upload.filename or "")
        ext = Path(name).suffix.lower()
        if not name or ext not in VALID_EXTS:
            rejected.append({"name": name, "reason": "Unsupported image format"})
            continue
        target = destination / name
        # Avoid overwriting a previously uploaded image silently.
        if target.exists():
            stem, suffix = target.stem, target.suffix
            target = destination / f"{stem}_{uuid.uuid4().hex[:5]}{suffix}"
            name = target.name
        total = 0
        try:
            with target.open("wb") as out:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_MB * 1024 * 1024:
                        target.unlink(missing_ok=True)
                        rejected.append({"name": name, "reason": f"File exceeds {MAX_UPLOAD_MB} MB"})
                        break
                    out.write(chunk)
            if target.exists():
                added.append(target.name)
        finally:
            await upload.close()

    reconstruction_files = _image_files(_active_image_dir())
    evidence_files = _image_files(_active_evidence_dir())
    ds_hash = _hash_files(reconstruction_files) if reconstruction_files else None
    with _lock:
        STATE["images_count"] = len(reconstruction_files)
        STATE["reconstruction_images_count"] = len(reconstruction_files)
        STATE["evidence_images_count"] = len(evidence_files)
        STATE["dataset_hash"] = ds_hash
        if handling == "append" and added:
            STATE["new_images_count"] = len(added)
            if STATE.get("model_ready"):
                STATE["model_status"] = "STALE"
    _save_state()
    _log(f"Added {len(added)} image(s) as {handling}; reconstruction dataset now has {len(reconstruction_files)} image(s).")
    return {
        "added": added,
        "rejected": rejected,
        "handling": handling,
        "reconstruction_images_count": len(reconstruction_files),
        "evidence_images_count": len(evidence_files),
        "model_status": STATE.get("model_status"),
    }


@app.delete("/api/images/{filename}")
def delete_image(filename: str, kind: str = Query("reconstruction")) -> Dict[str, Any]:
    if STATE.get("busy"):
        raise HTTPException(409, "Wait for the current reconstruction to finish.")
    safe = _safe_name(filename)
    directory = _active_image_dir() if kind == "reconstruction" else _active_evidence_dir()
    target = directory / safe
    if not target.exists() or target.suffix.lower() not in VALID_EXTS:
        raise HTTPException(404, "Image not found.")
    target.unlink()
    reconstruction_files = _image_files(_active_image_dir())
    evidence_files = _image_files(_active_evidence_dir())
    ds_hash = _hash_files(reconstruction_files) if reconstruction_files else None
    _set(
        images_count=len(reconstruction_files),
        reconstruction_images_count=len(reconstruction_files),
        evidence_images_count=len(evidence_files),
        dataset_hash=ds_hash,
    )
    if kind == "reconstruction" and STATE.get("model_ready"):
        _set(model_status="STALE", new_images_count=0)
    _log(f"Removed {kind} image: {safe}")
    return {"deleted": safe, "kind": kind, "images_count": len(reconstruction_files), "evidence_images_count": len(evidence_files)}


@app.post("/api/run")
def run_pipeline(iterations: int = Query(DEFAULT_ITERATIONS, ge=100, le=30000)) -> Dict[str, Any]:
    try:
        job_id = start_pipeline(int(iterations))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"job_id": job_id, "iterations": int(iterations)}


@app.get("/api/state")
def api_state() -> JSONResponse:
    sid = _active_scene_id()
    reconstruction_files = _image_files(_scene_image_dir(sid))
    evidence_files = _image_files(_scene_evidence_dir(sid))
    current_hash = _hash_files(reconstruction_files) if reconstruction_files else None
    with _lock:
        state = json.loads(json.dumps(STATE))
        state["active_scene_id"] = sid
        state["active_scene_name"] = _scene_meta(sid)["name"]
        state["images_count"] = len(reconstruction_files)
        state["reconstruction_images_count"] = len(reconstruction_files)
        state["evidence_images_count"] = len(evidence_files)
        state["dataset_hash"] = current_hash
        state["model_stale"] = bool(
            state.get("model_ready") and state.get("model_dataset_hash") and current_hash != state.get("model_dataset_hash")
        )
        if state["model_stale"]:
            state["model_status"] = "STALE"
        elif state.get("model_ready") and not state.get("busy"):
            state["model_status"] = "CURRENT"
        state["images"] = [
            {"name": p.name, "url": f"/api/scene-image/{sid}/{p.name}", "kind": "reconstruction"}
            for p in reconstruction_files
        ]
        state["evidence"] = [
            {"name": p.name, "url": f"/api/evidence-image/{sid}/{p.name}", "kind": "evidence"}
            for p in evidence_files
        ]
        state["model_url"] = "/api/model.ply" if state.get("model_ready") else None
    return JSONResponse(state, headers={"Cache-Control": "no-cache, no-store"})


@app.get("/api/scene-image/{scene_id}/{filename}")
def scene_image(scene_id: str, filename: str) -> FileResponse:
    if scene_id != "legacy" and not _scene_root(scene_id).exists():
        raise HTTPException(404, "Scene not found.")
    target = _scene_image_dir(scene_id) / _safe_name(filename)
    if not target.exists() or target.suffix.lower() not in VALID_EXTS:
        raise HTTPException(404, "Image not found.")
    return FileResponse(str(target))


@app.get("/api/evidence-image/{scene_id}/{filename}")
def evidence_image(scene_id: str, filename: str) -> FileResponse:
    if scene_id != "legacy" and not _scene_root(scene_id).exists():
        raise HTTPException(404, "Scene not found.")
    target = _scene_evidence_dir(scene_id) / _safe_name(filename)
    if not target.exists() or target.suffix.lower() not in VALID_EXTS:
        raise HTTPException(404, "Evidence image not found.")
    return FileResponse(str(target))


@app.get("/api/model.ply")
def model_ply() -> FileResponse:
    with _lock:
        if not STATE.get("model_ready"):
            raise HTTPException(404, "No trained Gaussian model available.")
        root = Path(STATE.get("model_job_dir", ""))
        iteration = STATE.get("model_iteration")
        candidates = [
            root / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
            root / "gaussian_model" / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
        ]
    for pc in candidates:
        if pc.exists():
            return FileResponse(str(pc), media_type="application/octet-stream", filename="aerosplat_model.ply")
    raise HTTPException(404, "Trained PLY not found.")


@app.get("/api/report")
def report() -> Dict[str, Any]:
    with _lock:
        return {
            "incident": STATE.get("active_scene_name"),
            "generated_at": time.time(),
            "reconstruction": {
                "images": STATE["images_count"],
                "registered_images": STATE["registered_images"],
                "points3d": STATE["points3d"],
                "observations": STATE["observations"],
                "mean_track_length": STATE["mean_track_length"],
                "mean_observations_per_image": STATE["mean_observations_per_image"],
                "mean_reprojection_error_px": STATE["mean_reprojection_error_px"],
                "camera_model": STATE["camera_model"],
            },
            "model": {
                "ready": STATE["model_ready"],
                "status": STATE["model_status"],
                "iteration": STATE["model_iteration"],
                "size_bytes": STATE["model_size_bytes"],
            },
            "damage": STATE["damage"],
            "triage": STATE["triage"],
            "route": STATE["route"],
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
