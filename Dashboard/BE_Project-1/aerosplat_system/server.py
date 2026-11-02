import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

WORKSPACE = Path(os.getenv("AEROSPLAT_WORKSPACE", r"C:\aerosplat_workspace"))
IMAGE_DIR = WORKSPACE / "images"
JOBS_DIR = WORKSPACE / "jobs"
STATE_FILE = WORKSPACE / "aerosplat_state.json"
COLMAP_EXE = os.getenv("COLMAP_EXE", r"C:\bin\colmap.exe")
GAUSSIAN_DIR = Path(os.getenv(
    "GAUSSIAN_SPLATTING_DIR",
    r"C:\Users\Admin\Downloads\AeroSplat_Upgraded\external\gaussian-splatting",
))
TRAIN_PY = GAUSSIAN_DIR / "train.py"
RENDER_PY = GAUSSIAN_DIR / "render.py"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_MB = int(os.getenv("AEROSPLAT_MAX_UPLOAD_MB", "100"))
DEFAULT_ITERATIONS = int(os.getenv("AEROSPLAT_ITERATIONS", "7000"))
DEFAULT_RESOLUTION = int(os.getenv("AEROSPLAT_RESOLUTION", "2"))
MAX_JOBS = int(os.getenv("AEROSPLAT_MAX_JOBS", "2"))

app = FastAPI(title="AeroSplat Command Center", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGE_DIR), html=False), name="images")

_lock = threading.RLock()
_pipeline_thread: Optional[threading.Thread] = None


def _default_state() -> Dict[str, Any]:
    return {
        "job_id": None,
        "stage": "idle",
        "status": "Ready",
        "progress": 0,
        "busy": False,
        "started_at": None,
        "finished_at": None,
        "updated_at": time.time(),
        "images_count": 0,
        "registered_images": 0,
        "points3d": 0,
        "observations": 0,
        "mean_track_length": 0.0,
        "mean_observations_per_image": 0.0,
        "mean_reprojection_error_px": 0.0,
        "camera_model": "—",
        "model_ready": False,
        "model_iteration": None,
        "model_size_bytes": 0,
        "model_source": None,
        "model_job_dir": None,
        "dataset_hash": None,
        "model_dataset_hash": None,
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


def _load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            base = _default_state()
            base.update(state)
            return base
        except Exception:
            pass
    return _default_state()


STATE = _load_state()


def _image_files(directory: Path = IMAGE_DIR) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS],
        key=lambda p: p.name.lower(),
    )


def _dataset_hash(files: List[Path]) -> str:
    import hashlib
    h = hashlib.sha256()
    for p in files:
        st = p.stat()
        h.update(p.name.encode("utf-8", errors="ignore"))
        h.update(str(st.st_size).encode())
        h.update(str(st.st_mtime_ns).encode())
    return h.hexdigest()[:16]


def _save_state() -> None:
    with _lock:
        STATE["updated_at"] = time.time()
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(STATE, indent=2), encoding="utf-8")
        tmp.replace(STATE_FILE)


def _log(message: str) -> None:
    with _lock:
        STATE.setdefault("log", []).append({"time": time.time(), "message": message})
        STATE["log"] = STATE["log"][-200:]
    _save_state()


def _set(**kwargs: Any) -> None:
    with _lock:
        STATE.update(kwargs)
    _save_state()


def _run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None, stage: Optional[str] = None) -> str:
    command_text = " ".join(f'"{x}"' if " " in x else x for x in cmd)
    _log("$ " + command_text)
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

    lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        lines.append(line)
        pct = re.search(r"Training progress:\s*(\d+)%", line)
        if pct and stage == "3dgs":
            p = int(pct.group(1))
            _set(stage="3dgs", status=f"Training 3DGS — {p}%", progress=60 + int(p * 0.4))
        if "Processed file" in line and stage == "colmap_features":
            _set(stage="colmap_features", status=line, progress=5)
        if "Generating sequential image pairs" in line and stage == "colmap_match":
            _set(stage="colmap_match", status="Matching overlapping views…", progress=18)
        if "Retriangulation" in line and stage == "colmap_mapper":
            _set(stage="colmap_mapper", status="Refining sparse reconstruction…", progress=38)
        if "Keeping successful reconstruction" in line and stage == "colmap_mapper":
            _set(stage="colmap_mapper", status="Sparse reconstruction accepted", progress=45)
    code = proc.wait()
    output = "\n".join(lines)
    if code != 0:
        raise RuntimeError("Command failed with exit code %s:\n%s" % (code, "\n".join(lines[-40:])))
    return output


def _parse_model_stats(text: str) -> Dict[str, Any]:
    patterns = {
        "registered_images": r"Registered images:\s+(\d+)",
        "points3d": r"Points:\s+(\d+)",
        "observations": r"Observations:\s+(\d+)",
        "mean_track_length": r"Mean track length:\s+([\d.]+)",
        "mean_observations_per_image": r"Mean observations per image:\s+([\d.]+)",
        "mean_reprojection_error_px": r"Mean reprojection error:\s+([\d.]+)px",
    }
    out: Dict[str, Any] = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if not m:
            continue
        v = m.group(1)
        if key in {"registered_images", "points3d", "observations"}:
            out[key] = int(v)
        else:
            out[key] = float(v)
    return out


def _find_colmap_model(root: Path) -> Path:
    if all((root / f).exists() for f in ("cameras.bin", "images.bin", "points3D.bin")):
        return root
    if (root / "0").exists() and all((root / "0" / f).exists() for f in ("cameras.bin", "images.bin", "points3D.bin")):
        return root / "0"
    dirs = sorted([p for p in root.glob("*") if p.is_dir() and p.name.isdigit()])
    for p in dirs:
        if all((p / f).exists() for f in ("cameras.bin", "images.bin", "points3D.bin")):
            return p
    raise RuntimeError(f"No valid COLMAP sparse model found under {root}")


def _read_camera_model(model_dir: Path) -> str:
    txt_dir = model_dir.parent / f"{model_dir.name}_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run([
            COLMAP_EXE, "model_converter",
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


def _find_latest_model() -> Optional[Dict[str, Any]]:
    candidates = []
    roots = [WORKSPACE / "gaussian_output", WORKSPACE / "gaussian_test"]
    roots.extend([p for p in JOBS_DIR.glob("*") if p.is_dir()])
    for root in roots:
        pc_root = root / "point_cloud"
        if not pc_root.exists():
            pc_root = root / "gaussian_model" / "point_cloud"
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
                candidates.append((pc.stat().st_mtime, iteration, pc, root))
    if not candidates:
        return None
    _, iteration, pc, root = max(candidates, key=lambda x: x[0])
    return {"iteration": iteration, "path": pc, "root": root, "size": pc.stat().st_size}


def _sync_existing_model() -> None:
    model = _find_latest_model()
    if not model:
        return
    files = _image_files()
    current_hash = _dataset_hash(files) if files else None
    with _lock:
        if STATE.get("model_job_dir") and STATE.get("model_ready"):
            return
        STATE.update({
            "model_ready": True,
            "model_iteration": model["iteration"],
            "model_size_bytes": model["size"],
            "model_source": "existing workspace model",
            "model_job_dir": str(model["root"]),
            "model_dataset_hash": current_hash,
        })
        if STATE.get("images_count", 0) == 0:
            STATE["images_count"] = len(files)
    _save_state()


_sync_existing_model()


def _pipeline(job_id: str, iterations: int) -> None:
    job = JOBS_DIR / job_id
    raw = job / "images"
    db = job / "database.db"
    sparse = job / "sparse"
    dense = job / "dense"
    model_out = job / "gaussian_model"
    try:
        imgs = _image_files(raw)
        if len(imgs) < 3:
            raise RuntimeError("At least 3 overlapping images are required.")
        _set(stage="colmap_features", status=f"Extracting features from {len(imgs)} images…", progress=2)
        _run([
            COLMAP_EXE, "feature_extractor",
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
            COLMAP_EXE, "sequential_matcher",
            "--database_path", str(db),
            "--SequentialMatching.overlap", "10",
            "--FeatureMatching.use_gpu", "0",
        ], stage="colmap_match")

        sparse.mkdir(parents=True, exist_ok=True)
        _set(stage="colmap_mapper", status="Building sparse reconstruction…", progress=25)
        _run([
            COLMAP_EXE, "mapper",
            "--database_path", str(db),
            "--image_path", str(raw),
            "--output_path", str(sparse),
        ], stage="colmap_mapper")
        model_dir = _find_colmap_model(sparse)

        analyzer = _run([COLMAP_EXE, "model_analyzer", "--path", str(model_dir)])
        stats = _parse_model_stats(analyzer)
        camera = _read_camera_model(model_dir)
        _set(**stats, camera_model=camera, stage="undistort", status="Undistorting for GraphDeco…", progress=48)
        if camera not in {"SIMPLE_RADIAL", "PINHOLE", "SIMPLE_PINHOLE", "—"}:
            raise RuntimeError(f"Unsupported COLMAP camera model: {camera}")

        _run([
            COLMAP_EXE, "image_undistorter",
            "--image_path", str(raw),
            "--input_path", str(model_dir),
            "--output_path", str(dense),
            "--output_type", "COLMAP",
        ], stage="undistort")
        graph_source = _prepare_graphdeco_layout(dense)

        env = os.environ.copy()
        env["CUDA_HOME"] = env.get("AEROSPLAT_CUDA_HOME", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")
        env["TORCH_CUDA_ARCH_LIST"] = env.get("TORCH_CUDA_ARCH_LIST", "8.6")
        env["MAX_JOBS"] = str(MAX_JOBS)
        env["DISTUTILS_USE_SDK"] = "1"
        cuda_bin = str(Path(env["CUDA_HOME"]) / "bin")
        env["PATH"] = cuda_bin + os.pathsep + env.get("PATH", "")

        _set(stage="3dgs", status=f"Training 3DGS — 0/{iterations}", progress=60)
        _run([
            "python", str(TRAIN_PY),
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
        ds_hash = _dataset_hash(imgs)
        _set(
            stage="ready",
            status="Reconstruction complete",
            progress=100,
            busy=False,
            finished_at=time.time(),
            model_ready=True,
            model_iteration=iterations,
            model_size_bytes=pc.stat().st_size,
            model_source="background reconstruction job",
            model_job_dir=str(job),
            model_dataset_hash=ds_hash,
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
                    "Use the 3D twin as reconstruction evidence, not as engineering certification.",
                    "Connect validated damage segmentation to enable measured triage.",
                ],
            },
            route={"status": "Awaiting road/hazard graph", "distance_m": None, "eta_min": None, "blocked_zones": 0},
        )
        _log(f"3DGS model ready: {pc}")
    except Exception as exc:
        _set(busy=False, stage="error", status="Pipeline failed", last_error=str(exc), finished_at=time.time())
        _log(f"ERROR: {exc}")


def start_pipeline(iterations: int) -> str:
    global _pipeline_thread
    with _lock:
        if STATE.get("busy"):
            raise RuntimeError("A pipeline job is already running.")
    files = _image_files()
    job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    job = JOBS_DIR / job_id
    raw = job / "images"
    raw.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, raw / src.name)
    ds_hash = _dataset_hash(files) if files else None
    _set(
        job_id=job_id,
        busy=True,
        stage="queued",
        status="Job queued",
        progress=0,
        started_at=time.time(),
        finished_at=None,
        model_job_dir=str(job),
        model_ready=False,
        dataset_hash=ds_hash,
        last_error=None,
    )
    _pipeline_thread = threading.Thread(target=_pipeline, args=(job_id, iterations), daemon=True, name=f"aerosplat-{job_id}")
    _pipeline_thread.start()
    return job_id


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/state")
def api_state() -> JSONResponse:
    files = _image_files()
    with _lock:
        state = json.loads(json.dumps(STATE))
        state["images"] = [{"name": p.name, "url": f"/images/{p.name}"} for p in files]
        state["images_count"] = len(files)
        current_hash = _dataset_hash(files) if files else None
        state["dataset_hash"] = current_hash
        state["model_stale"] = bool(state.get("model_ready") and state.get("model_dataset_hash") and current_hash != state.get("model_dataset_hash"))
        if state.get("model_ready"):
            root = Path(state.get("model_job_dir")) if state.get("model_job_dir") else None
            pc = None
            if root:
                candidates = [root / "point_cloud" / f"iteration_{state.get('model_iteration')}" / "point_cloud.ply", root / "gaussian_model" / "point_cloud" / f"iteration_{state.get('model_iteration')}" / "point_cloud.ply"]
                for c in candidates:
                    if c.exists():
                        pc = c
                        break
            state["model_url"] = "/api/model.ply" if pc else None
        else:
            state["model_url"] = None
    return JSONResponse(state, headers={"Cache-Control": "no-cache, no-store"})


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "api": "ok",
        "colmap": str(COLMAP_EXE),
        "colmap_exists": COLMAP_EXE.exists(),
        "graphdeco": str(TRAIN_PY),
        "graphdeco_exists": TRAIN_PY.exists(),
        "workspace": str(WORKSPACE),
        "images": len(_image_files()),
    }


@app.post("/api/upload-images")
async def upload_images(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    if STATE.get("busy"):
        raise HTTPException(409, "Wait until the current reconstruction finishes before changing the dataset.")
    added, rejected = [], []
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = Path(f.filename or "").name
        ext = Path(name).suffix.lower()
        if ext not in VALID_EXTS:
            rejected.append({"name": name, "reason": "Unsupported image format"})
            continue
        target = IMAGE_DIR / name
        total = 0
        with target.open("wb") as out:
            while True:
                chunk = await f.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_MB * 1024 * 1024:
                    target.unlink(missing_ok=True)
                    rejected.append({"name": name, "reason": f"File exceeds {MAX_UPLOAD_MB} MB"})
                    break
                out.write(chunk)
        if target.exists():
            added.append(name)
    files_now = _image_files()
    _set(images_count=len(files_now), dataset_hash=_dataset_hash(files_now) if files_now else None)
    _log(f"Added {len(added)} image(s); dataset now contains {len(files_now)} image(s).")
    return {"added": added, "rejected": rejected, "images_count": len(files_now)}


@app.delete("/api/images/{filename}")
def delete_image(filename: str) -> Dict[str, Any]:
    if STATE.get("busy"):
        raise HTTPException(409, "Wait for the current reconstruction to finish.")
    safe = Path(filename).name
    target = IMAGE_DIR / safe
    if not target.exists() or target.suffix.lower() not in VALID_EXTS:
        raise HTTPException(404, "Image not found.")
    target.unlink()
    files_now = _image_files()
    _set(images_count=len(files_now), dataset_hash=_dataset_hash(files_now) if files_now else None)
    _log(f"Removed image: {safe}")
    return {"deleted": safe, "images_count": len(files_now)}


@app.post("/api/run")
def run_pipeline(iterations: int = DEFAULT_ITERATIONS) -> Dict[str, Any]:
    iterations = max(100, min(int(iterations), 30000))
    try:
        job_id = start_pipeline(iterations)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"job_id": job_id, "iterations": iterations}


@app.get("/api/log")
def api_log() -> List[Dict[str, Any]]:
    with _lock:
        return list(STATE.get("log", []))[-100:]


@app.get("/api/model.ply")
def model_ply() -> FileResponse:
    with _lock:
        if not STATE.get("model_ready"):
            raise HTTPException(404, "No trained Gaussian model available.")
        root = Path(STATE.get("model_job_dir", ""))
        it = STATE.get("model_iteration")
        candidates = [
            root / "point_cloud" / f"iteration_{it}" / "point_cloud.ply",
            root / "gaussian_model" / "point_cloud" / f"iteration_{it}" / "point_cloud.ply",
        ]
    for pc in candidates:
        if pc.exists():
            return FileResponse(str(pc), media_type="application/octet-stream", filename="aerosplat_model.ply", headers={"Cache-Control":"no-store", "Access-Control-Allow-Origin":"*"})
    raise HTTPException(404, "Trained PLY not found.")


@app.get("/api/report")
def report() -> Dict[str, Any]:
    with _lock:
        return {
            "incident": "AeroSplat post-disaster assessment",
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
                "iteration": STATE["model_iteration"],
                "size_bytes": STATE["model_size_bytes"],
                "stale": bool(STATE.get("model_dataset_hash") and STATE.get("dataset_hash") != STATE.get("model_dataset_hash")),
            },
            "damage": STATE["damage"],
            "triage": STATE["triage"],
            "route": STATE["route"],
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
