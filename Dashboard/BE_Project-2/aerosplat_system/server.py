import hashlib, json, math, os, re, shutil, subprocess, threading, time, uuid, struct
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
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
STATE_FILE = WORKSPACE / "aerosplat_state_v4.json"
COLMAP_EXE = Path(os.getenv("COLMAP_EXE", r"C:\bin\colmap.exe"))
GAUSSIAN_DIR = Path(os.getenv("GAUSSIAN_SPLATTING_DIR", r"C:\Users\Admin\Downloads\AeroSplat_Upgraded\external\gaussian-splatting"))
TRAIN_PY = GAUSSIAN_DIR / "train.py"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_ITERATIONS = int(os.getenv("AEROSPLAT_ITERATIONS", "7000"))
DEFAULT_RESOLUTION = int(os.getenv("AEROSPLAT_RESOLUTION", "2"))
BUILD_MAX_JOBS = int(os.getenv("AEROSPLAT_MAX_JOBS", "2"))
MAX_UPLOAD_MB = int(os.getenv("AEROSPLAT_MAX_UPLOAD_MB", "100"))

for d in (WORKSPACE, LEGACY_IMAGE_DIR, SCENES_DIR, EVIDENCE_DIR, JOBS_DIR, STATIC_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AeroSplat Command Center", version="4.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
_lock = threading.RLock()
_thread: Optional[threading.Thread] = None


def safe_name(name: str) -> str:
    return Path(name or "").name


def image_files(directory: Path) -> List[Path]:
    if not directory.exists(): return []
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS], key=lambda p: p.name.lower())


def file_hash(files: List[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(files, key=lambda x: x.name.lower()):
        try: st = p.stat()
        except FileNotFoundError: continue
        h.update(p.name.encode()); h.update(str(st.st_size).encode()); h.update(str(st.st_mtime_ns).encode())
    return h.hexdigest()[:16]


def read_json(path: Path, default: Any):
    try: return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception: return default


def save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def default_state() -> Dict[str, Any]:
    return {
        "active_scene_id": "legacy", "active_scene_name": "Existing Workspace",
        "busy": False, "stage": "idle", "status": "Ready", "progress": 0,
        "job_id": None, "started_at": None, "finished_at": None, "updated_at": time.time(),
        "images_count": 0, "reconstruction_images_count": 0, "evidence_images_count": 0,
        "registered_images": 0, "points3d": 0, "observations": 0, "mean_track_length": 0.0,
        "mean_observations_per_image": 0.0, "mean_reprojection_error_px": 0.0, "camera_model": "—",
        "model_ready": False, "model_status": "NONE", "model_iteration": None, "model_size_bytes": 0,
        "model_source": None, "model_job_dir": None, "dataset_hash": None, "model_dataset_hash": None,
        "new_images_count": 0, "last_error": None,
        "analysis": {
            "status": "NOT_RUN", "quality": {"images_analyzed": 0, "mean_sharpness": None, "blurred_images": 0},
            "crack_visual_indicator": {"status": "BASELINE", "candidate_images": 0, "mean_candidate_ratio": None},
            "geometry_indicator": {"status": "PENDING", "dominant_plane_tilt_deg": None, "points_sampled": 0},
            "rubble_indicator": {"status": "NOT_COMPUTED", "volume_m3": None},
            "notes": []
        },
        "triage": {"label": "PENDING", "score": None, "confidence": None, "severity": "Analysis not run", "directives": []},
        "recommendations": [],
        "route": {"status": "NOT_COMPUTED", "distance_m": None, "eta_min": None, "blocked_zones": 0},
        "images": [], "evidence": [], "log": []
    }

STATE = default_state()
loaded = read_json(STATE_FILE, None)
if isinstance(loaded, dict): STATE.update(loaded)


def save_state():
    with _lock:
        STATE["updated_at"] = time.time(); save_json(STATE_FILE, STATE)


def log(msg: str):
    with _lock:
        STATE.setdefault("log", []).append({"time": time.time(), "message": msg}); STATE["log"] = STATE["log"][-200:]
    save_state()


def set_state(**kw):
    with _lock: STATE.update(kw)
    save_state()


def scene_root(sid): return SCENES_DIR / sid

def scene_image_dir(sid): return LEGACY_IMAGE_DIR if sid == "legacy" else scene_root(sid) / "images"
def scene_evidence_dir(sid): return EVIDENCE_DIR / "legacy" if sid == "legacy" else scene_root(sid) / "evidence"
def scene_model_root(sid): return WORKSPACE / "gaussian_output" if sid == "legacy" else scene_root(sid) / "gaussian_output"

def ensure_scene(sid):
    scene_image_dir(sid).mkdir(parents=True, exist_ok=True); scene_evidence_dir(sid).mkdir(parents=True, exist_ok=True)
    if sid != "legacy": scene_root(sid).mkdir(parents=True, exist_ok=True)

def scene_meta(sid):
    if sid == "legacy": return {"id":"legacy","name":"Existing Workspace","description":"Primary AeroSplat workspace."}
    d = read_json(scene_root(sid)/"scene.json", {})
    return {"id":sid,"name":d.get("name",sid),"description":d.get("description","")}

def scenes():
    ensure_scene("legacy"); out=[scene_meta("legacy")]
    for p in sorted(SCENES_DIR.iterdir(), key=lambda x:x.name.lower()):
        if p.is_dir(): out.append(scene_meta(p.name))
    return out

def active_sid():
    sid=STATE.get("active_scene_id") or "legacy"; ensure_scene(sid); return sid

def active_images(): return image_files(scene_image_dir(active_sid()))

def active_evidence(): return image_files(scene_evidence_dir(active_sid()))

def dataset_hash(files): return file_hash(files) if files else None


def latest_model(root: Path):
    candidates=[]
    bases=[root]
    if root==scene_model_root("legacy"): bases.append(WORKSPACE/"gaussian_test")
    for job in JOBS_DIR.glob("*"):
        if job.is_dir(): bases += [job/"gaussian_model", job]
    for base in bases:
        pcroot=base/"point_cloud"
        if not pcroot.exists(): continue
        for it in pcroot.glob("iteration_*"):
            if not it.is_dir(): continue
            try: n=int(it.name.split("_")[-1])
            except: continue
            pc=it/"point_cloud.ply"
            if pc.exists(): candidates.append((pc.stat().st_mtime,n,pc,base))
    if not candidates: return None
    _,n,pc,base=max(candidates,key=lambda x:x[0]); return {"iteration":n,"path":pc,"root":base,"size":pc.stat().st_size}


def parse_model_stats(text):
    pats={"registered_images":r"Registered images:\s+(\d+)","points3d":r"Points:\s+(\d+)","observations":r"Observations:\s+(\d+)","mean_track_length":r"Mean track length:\s+([\d.]+)","mean_observations_per_image":r"Mean observations per image:\s+([\d.]+)","mean_reprojection_error_px":r"Mean reprojection error:\s+([\d.]+)px"}
    out={}
    for k,p in pats.items():
        m=re.search(p,text)
        if m: out[k]=int(m.group(1)) if k in {"registered_images","points3d","observations"} else float(m.group(1))
    return out


def read_camera_model(model_dir: Path):
    out=model_dir.parent / f"{model_dir.name}_txt"; out.mkdir(parents=True,exist_ok=True)
    try:
        run([str(COLMAP_EXE),"model_converter","--input_path",str(model_dir),"--output_path",str(out),"--output_type","TXT"])
        p=out/"cameras.txt"
        for line in p.read_text(encoding="utf-8",errors="ignore").splitlines():
            if line and not line.startswith("#"):
                parts=line.split(); return parts[1]
    except Exception: pass
    return "—"


def find_model(root: Path):
    if all((root/f).exists() for f in ("cameras.bin","images.bin","points3D.bin")): return root
    if (root/"0").exists() and all((root/"0"/f).exists() for f in ("cameras.bin","images.bin","points3D.bin")): return root/"0"
    raise RuntimeError(f"No valid COLMAP sparse model under {root}")


def run(cmd, cwd=None, env=None, stage=None):
    pretty=" ".join(f'"{x}"' if " " in x else x for x in cmd); log("$ "+pretty)
    try:
        p=subprocess.Popen(cmd,cwd=str(cwd) if cwd else None,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1)
    except FileNotFoundError as e: raise RuntimeError(f"Executable not found: {cmd[0]}") from e
    lines=[]; assert p.stdout
    for raw in p.stdout:
        line=raw.rstrip();
        if not line: continue
        lines.append(line)
        if stage=="3dgs":
            m=re.search(r"(\d+)%",line)
            if m and "Training progress" in line: set_state(stage="3dgs",status=f"Training 3DGS — {m.group(1)}%",progress=60+int(int(m.group(1))*0.4))
        elif stage=="colmap_mapper" and "Keeping successful reconstruction" in line: set_state(status="Sparse reconstruction accepted",progress=45)
    code=p.wait(); txt="\n".join(lines)
    if code!=0: raise RuntimeError(f"Command failed with exit code {code}:\n"+"\n".join(lines[-80:]))
    return txt


def prepare_graph(dense):
    sparse=dense/"sparse"; sparse0=sparse/"0"; sparse0.mkdir(parents=True,exist_ok=True)
    for x in sparse.iterdir():
        if x.is_file() and x.suffix==".bin": shutil.copy2(x,sparse0/x.name)
    for f in ("cameras.bin","images.bin","points3D.bin"):
        if not (sparse0/f).exists(): raise RuntimeError("Missing "+f)
    return dense


def scene_snapshot(job, sid):
    raw=job/"images"; raw.mkdir(parents=True,exist_ok=True); files=active_images()
    for src in files: shutil.copy2(src,raw/src.name)
    return raw, [raw/f.name for f in files]


def parse_points3d_bin(path: Path, max_points=120000):
    pts=[]
    try:
        with path.open("rb") as f:
            n=struct.unpack("<Q",f.read(8))[0]
            for i in range(min(n,max_points)):
                pid=struct.unpack("<Q",f.read(8))[0]
                xyz=struct.unpack("<ddd",f.read(24)); rgb=f.read(3); err=struct.unpack("<d",f.read(8))[0]
                track_len=struct.unpack("<Q",f.read(8))[0]; f.seek(track_len*8*2,1)
                pts.append(xyz)
    except Exception: return np.empty((0,3),dtype=np.float64)
    return np.asarray(pts,dtype=np.float64)


def analyze_images(files: List[Path]):
    sharp=[]; crack_ratios=[]; blurred=0; candidates=0
    for p in files:
        im=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        h,w=im.shape[:2]
        scale=min(1.0,1200/max(w,h))
        if scale<1: im=cv2.resize(im,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_AREA)
        s=float(cv2.Laplacian(im,cv2.CV_64F).var()); sharp.append(s)
        if s<80: blurred+=1
        clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(im)
        edges=cv2.Canny(clahe,50,140)
        # Conservative crack-like indicator: thin edge pixels within dark/local-contrast regions.
        dark=cv2.GaussianBlur(clahe,(0,0),2.0) < np.percentile(clahe,22)
        ratio=float(np.logical_and(edges>0,dark).mean())
        crack_ratios.append(ratio)
        if ratio>0.012: candidates+=1
    return {
        "quality":{"images_analyzed":len(sharp),"mean_sharpness":float(np.mean(sharp)) if sharp else None,"blurred_images":blurred},
        "crack_visual_indicator":{"status":"VISUAL BASELINE","candidate_images":candidates,"mean_candidate_ratio":float(np.mean(crack_ratios)) if crack_ratios else None}
    }


def analyze_geometry(sparse_model: Path):
    pts=parse_points3d_bin(sparse_model/"points3D.bin")
    if len(pts)<50: return {"status":"INSUFFICIENT_POINTS","dominant_plane_tilt_deg":None,"points_sampled":len(pts)}
    # Robust-ish subsample and PCA. This is a scene-plane indicator, not engineering wall tilt.
    pts=pts[np.isfinite(pts).all(axis=1)]
    if len(pts)>25000:
        idx=np.linspace(0,len(pts)-1,25000).astype(int); pts=pts[idx]
    center=pts.mean(axis=0); cov=np.cov((pts-center).T); vals,vecs=np.linalg.eigh(cov); normal=vecs[:,np.argmin(vals)]
    # Assume world Z is vertical; angle from horizontal plane normal is indicative of dominant plane inclination.
    z=abs(float(normal[2])); z=min(1.0,max(0.0,z)); angle=math.degrees(math.acos(z))
    return {"status":"PROVISIONAL SCENE GEOMETRY","dominant_plane_tilt_deg":round(angle,2),"points_sampled":len(pts),"note":"Dominant-plane inclination only; not a certified wall-tilt measurement."}


def compute_analysis(sid):
    files=image_files(scene_image_dir(sid)); out={"status":"COMPLETE","quality":{},"crack_visual_indicator":{},"geometry_indicator":{},"rubble_indicator":{"status":"NOT_COMPUTED","volume_m3":None},"notes":[]}
    out.update(analyze_images(files))
    # Prefer active dense/sparse reconstruction for the scene.
    dense=WORKSPACE/"dense" if sid=="legacy" else scene_root(sid)/"dense"
    model=None
    for candidate in [dense/"sparse", dense, WORKSPACE/"sparse/0"]:
        if candidate.exists() and (candidate/"points3D.bin").exists(): model=candidate; break
    if model:
        out["geometry_indicator"]=analyze_geometry(model)
    else:
        out["geometry_indicator"]={"status":"NO_SPARSE_MODEL","dominant_plane_tilt_deg":None,"points_sampled":0}
    out["notes"] += [
        "Crack indicator is a computer-vision baseline for prioritizing images; it is not crack metrology.",
        "Dominant-plane inclination is a geometry indicator; engineering wall tilt requires identified wall surfaces and calibrated coordinates.",
        "Rubble volume remains unavailable until rubble segmentation and metric 3D reconstruction are connected."
    ]
    return out


def make_triage(analysis):
    q=analysis.get("quality",{}); c=analysis.get("crack_visual_indicator",{}); g=analysis.get("geometry_indicator",{})
    score=0.0; reasons=[]; directives=[]
    if q.get("blurred_images",0)>max(3,(q.get("images_analyzed",0)*0.25)):
        score+=0.15; reasons.append("High fraction of low-sharpness imagery"); directives.append("Acquire additional sharp close-range imagery.")
    if (c.get("mean_candidate_ratio") or 0)>0.018:
        score+=0.30; reasons.append("Elevated crack-like visual texture"); directives.append("Prioritize frames flagged for crack-like regions.")
    tilt=g.get("dominant_plane_tilt_deg")
    if tilt is not None and tilt>8:
        score+=0.25; reasons.append("Dominant scene-plane inclination is elevated"); directives.append("Validate suspected inclined surfaces with calibrated geometry.")
    elif tilt is not None and tilt>4:
        score+=0.10; reasons.append("Dominant scene-plane inclination is non-negligible")
    score=min(1.0,score)
    label="RED" if score>=0.7 else "YELLOW" if score>=0.35 else "GREEN"
    directives = directives or ["No high-risk baseline indicator triggered; continue inspection."]
    return {"label":label,"score":round(score,2),"confidence":None,"severity":"Preliminary, evidence-based screening only","directives":directives,"reasons":reasons}


def _sync_existing():
    sid=active_sid(); files=image_files(scene_image_dir(sid)); ev=image_files(scene_evidence_dir(sid)); h=file_hash(files) if files else None
    model=latest_model(scene_model_root(sid))
    with _lock: prev_hash=STATE.get("model_dataset_hash")
    if model:
        mh=prev_hash or h
        set_state(model_ready=True,model_status="CURRENT" if mh==h else "STALE",model_iteration=model["iteration"],model_size_bytes=model["size"],model_source="existing trained model",model_job_dir=str(model["root"]),model_dataset_hash=mh,dataset_hash=h,images_count=len(files),reconstruction_images_count=len(files),evidence_images_count=len(ev))
    else:
        set_state(model_ready=False,model_status="NONE",model_iteration=None,model_size_bytes=0,model_job_dir=None,model_dataset_hash=None,dataset_hash=h,images_count=len(files),reconstruction_images_count=len(files),evidence_images_count=len(ev))
    try:
        if files and STATE.get("analysis",{}).get("status")!="COMPLETE":
            a=compute_analysis(sid); t=make_triage(a); set_state(analysis=a,triage=t,recommendations=t.get("directives",[]))
    except Exception as e: log("Analysis startup warning: "+str(e))


def pipeline(job_id,sid,iterations):
    job=JOBS_DIR/job_id; raw,source_files=scene_snapshot(job,sid); db=job/"database.db"; sparse=job/"sparse"; dense=job/"dense"; out=job/"gaussian_model"
    try:
        if len(source_files)<3: raise RuntimeError("At least 3 overlapping reconstruction images are required.")
        set_state(stage="colmap_features",status=f"Extracting features from {len(source_files)} images…",progress=2)
        run([str(COLMAP_EXE),"feature_extractor","--database_path",str(db),"--image_path",str(raw),"--ImageReader.camera_model","SIMPLE_RADIAL","--ImageReader.single_camera","1","--FeatureExtraction.use_gpu","0","--FeatureExtraction.max_image_size","2400","--FeatureExtraction.num_threads","8"],stage="colmap_features")
        set_state(stage="colmap_match",status="Matching overlapping views…",progress=15)
        run([str(COLMAP_EXE),"sequential_matcher","--database_path",str(db),"--SequentialMatching.overlap","10","--FeatureMatching.use_gpu","0"],stage="colmap_match")
        sparse.mkdir(parents=True,exist_ok=True); set_state(stage="colmap_mapper",status="Building sparse reconstruction…",progress=25)
        run([str(COLMAP_EXE),"mapper","--database_path",str(db),"--image_path",str(raw),"--output_path",str(sparse)],stage="colmap_mapper")
        model=find_model(sparse); analyzer=run([str(COLMAP_EXE),"model_analyzer","--path",str(model)]); stats=parse_model_stats(analyzer); cam=read_camera_model(model)
        if stats.get("registered_images",0)<3: raise RuntimeError("COLMAP registered fewer than 3 images.")
        set_state(**stats,camera_model=cam,stage="undistort",status="Undistorting for GraphDeco…",progress=48)
        run([str(COLMAP_EXE),"image_undistorter","--image_path",str(raw),"--input_path",str(model),"--output_path",str(dense),"--output_type","COLMAP"])
        source=prepare_graph(dense)
        env=os.environ.copy(); env["CUDA_HOME"]=env.get("AEROSPLAT_CUDA_HOME",r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"); env["PATH"]=str(Path(env["CUDA_HOME"])/"bin")+os.pathsep+env.get("PATH",""); env["TORCH_CUDA_ARCH_LIST"]=env.get("TORCH_CUDA_ARCH_LIST","8.6"); env["MAX_JOBS"]=str(BUILD_MAX_JOBS); env["DISTUTILS_USE_SDK"]="1"
        set_state(stage="3dgs",status=f"Training 3DGS — 0%",progress=60)
        run(["python",str(TRAIN_PY),"-s",str(source),"-m",str(out),"--iterations",str(iterations),"--save_iterations",str(iterations),"--disable_viewer","-r",str(DEFAULT_RESOLUTION),"--data_device","cpu"],cwd=GAUSSIAN_DIR,env=env,stage="3dgs")
        pc=out/"point_cloud"/f"iteration_{iterations}"/"point_cloud.ply"
        if not pc.exists(): raise RuntimeError("3DGS completed but point_cloud.ply was not created.")
        target=scene_model_root(sid)/"point_cloud"/f"iteration_{iterations}"/"point_cloud.ply"; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(pc,target)
        files=image_files(scene_image_dir(sid)); h=file_hash(files); a=compute_analysis(sid); t=make_triage(a)
        set_state(stage="ready",status="Reconstruction and baseline analysis complete",progress=100,busy=False,finished_at=time.time(),model_ready=True,model_status="CURRENT",model_iteration=iterations,model_size_bytes=target.stat().st_size,model_source="AeroSplat reconstruction job",model_job_dir=str(target.parent.parent.parent),model_dataset_hash=h,dataset_hash=h,reconstruction_images_count=len(files),images_count=len(files),evidence_images_count=len(active_evidence()),new_images_count=0,analysis=a,triage=t,recommendations=t.get("directives",[]),last_error=None)
        log(f"Reconstruction complete: {target}")
    except Exception as e:
        set_state(busy=False,stage="error",status="Pipeline failed",last_error=str(e),finished_at=time.time())
        log("ERROR: "+str(e))


def start_pipeline(iterations):
    global _thread
    with _lock:
        if STATE.get("busy"): raise RuntimeError("A reconstruction job is already running.")
    files=active_images()
    if len(files)<3: raise RuntimeError("Add at least 3 overlapping reconstruction images first.")
    jid=time.strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6]
    set_state(job_id=jid,busy=True,stage="queued",status="Reconstruction queued",progress=0,started_at=time.time(),finished_at=None,last_error=None,model_status="BUILDING",dataset_hash=file_hash(files),new_images_count=max(0,len(files)-(STATE.get("registered_images") or 0)))
    _thread=threading.Thread(target=pipeline,args=(jid,active_sid(),int(iterations)),daemon=True); _thread.start(); return jid


@app.get("/",response_class=HTMLResponse)
def index(): return (STATIC_DIR/"index.html").read_text(encoding="utf-8")

@app.get("/api/health")
def health(): return {"api":"ok","colmap":str(COLMAP_EXE),"colmap_exists":COLMAP_EXE.exists(),"graphdeco":str(TRAIN_PY),"graphdeco_exists":TRAIN_PY.exists(),"workspace":str(WORKSPACE),"active_scene":active_sid()}

@app.get("/api/scenes")
def api_scenes(): return {"active_scene_id":active_sid(),"scenes":scenes()}

@app.post("/api/scenes")
def create_scene(name:str=Query(...,min_length=1),description:str=Query("")):
    if STATE.get("busy"): raise HTTPException(409,"Current reconstruction is running.")
    sid="incident_"+time.strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6]; ensure_scene(sid); save_json(scene_root(sid)/"scene.json",{"id":sid,"name":name.strip(),"description":description.strip()})
    set_state(active_scene_id=sid,active_scene_name=name.strip(),model_ready=False,model_status="NONE",model_dataset_hash=None,model_job_dir=None,model_iteration=None,new_images_count=0,analysis=default_state()["analysis"],triage=default_state()["triage"]); _sync_existing(); return {"scene":scene_meta(sid)}

@app.post("/api/scenes/select")
def select_scene(scene_id:str=Query(...)):
    if STATE.get("busy"): raise HTTPException(409,"Current reconstruction is running.")
    if scene_id!="legacy" and not scene_root(scene_id).exists(): raise HTTPException(404,"Scene not found")
    ensure_scene(scene_id); set_state(active_scene_id=scene_id,active_scene_name=scene_meta(scene_id)["name"],new_images_count=0); _sync_existing(); return {"scene":scene_meta(scene_id)}

@app.post("/api/upload-images")
async def upload_images(files:List[UploadFile]=File(...),handling:str=Query("append")):
    if STATE.get("busy"): raise HTTPException(409,"Wait for current reconstruction.")
    handling=handling.lower().strip();
    if handling not in {"append","evidence"}: raise HTTPException(400,"handling must be append or evidence")
    dest=scene_image_dir(active_sid()) if handling=="append" else scene_evidence_dir(active_sid()); dest.mkdir(parents=True,exist_ok=True); added=[]; rejected=[]
    for up in files:
        name=safe_name(up.filename or ""); ext=Path(name).suffix.lower()
        if not name or ext not in VALID_EXTS: rejected.append({"name":name,"reason":"Unsupported image format"}); continue
        target=dest/name
        if target.exists(): target=dest/(Path(name).stem+"_"+uuid.uuid4().hex[:5]+ext)
        total=0
        try:
            with target.open("wb") as out:
                while True:
                    c=await up.read(1024*1024)
                    if not c: break
                    total+=len(c)
                    if total>MAX_UPLOAD_MB*1024*1024: target.unlink(missing_ok=True); rejected.append({"name":name,"reason":"File too large"}); break
                    out.write(c)
            if target.exists(): added.append(target.name)
        finally: await up.close()
    files2=active_images(); ev=active_evidence(); h=file_hash(files2) if files2 else None
    if handling=="append" and added:
        set_state(images_count=len(files2),reconstruction_images_count=len(files2),evidence_images_count=len(ev),dataset_hash=h,new_images_count=len(added),model_status="STALE" if STATE.get("model_ready") else STATE.get("model_status","NONE"))
    else: set_state(images_count=len(files2),reconstruction_images_count=len(files2),evidence_images_count=len(ev),dataset_hash=h)
    try:
        a=compute_analysis(active_sid()); t=make_triage(a); set_state(analysis=a,triage=t,recommendations=t.get("directives",[]))
    except Exception as e: log("Analysis warning: "+str(e))
    log(f"Added {len(added)} image(s) as {handling}.")
    return {"added":added,"rejected":rejected,"handling":handling,"reconstruction_images_count":len(files2),"evidence_images_count":len(ev),"model_status":STATE.get("model_status")}

@app.post("/api/analyze")
def analyze():
    a=compute_analysis(active_sid()); t=make_triage(a); set_state(analysis=a,triage=t,recommendations=t.get("directives",[])); return {"analysis":a,"triage":t,"recommendations":t.get("directives",[])}

@app.post("/api/run")
def run_pipeline(iterations:int=Query(DEFAULT_ITERATIONS,ge=100,le=30000)):
    try: return {"job_id":start_pipeline(iterations),"iterations":iterations}
    except RuntimeError as e: raise HTTPException(409,str(e))

@app.get("/api/state")
def state():
    sid=active_sid(); imgs=active_images(); ev=active_evidence(); h=file_hash(imgs) if imgs else None
    with _lock: s=json.loads(json.dumps(STATE))
    s.update({"active_scene_id":sid,"active_scene_name":scene_meta(sid)["name"],"images_count":len(imgs),"reconstruction_images_count":len(imgs),"evidence_images_count":len(ev),"dataset_hash":h})
    s["model_stale"]=bool(s.get("model_ready") and s.get("model_dataset_hash") and h!=s.get("model_dataset_hash"))
    if s["model_stale"] and not s.get("busy"): s["model_status"]="STALE"
    elif s.get("model_ready") and not s.get("busy"): s["model_status"]="CURRENT"
    s["images"]=[{"name":p.name,"url":f"/api/scene-image/{sid}/{p.name}","kind":"reconstruction"} for p in imgs]
    s["evidence"]=[{"name":p.name,"url":f"/api/evidence-image/{sid}/{p.name}","kind":"evidence"} for p in ev]
    s["model_url"]="/api/model.ply" if s.get("model_ready") else None
    return JSONResponse(s,headers={"Cache-Control":"no-cache, no-store"})

@app.get("/api/scene-image/{sid}/{filename}")
def scene_image(sid,filename):
    if sid!="legacy" and not scene_root(sid).exists(): raise HTTPException(404,"Scene not found")
    p=scene_image_dir(sid)/safe_name(filename)
    if not p.exists(): raise HTTPException(404,"Image not found")
    return FileResponse(str(p))

@app.get("/api/evidence-image/{sid}/{filename}")
def evidence_image(sid,filename):
    p=scene_evidence_dir(sid)/safe_name(filename)
    if not p.exists(): raise HTTPException(404,"Evidence image not found")
    return FileResponse(str(p))

@app.get("/api/model.ply")
def model_ply():
    with _lock:
        if not STATE.get("model_ready"): raise HTTPException(404,"No trained Gaussian model available")
        root=Path(STATE.get("model_job_dir", "")); it=STATE.get("model_iteration")
        candidates=[root/"point_cloud"/f"iteration_{it}"/"point_cloud.ply", root.parent.parent/"point_cloud"/f"iteration_{it}"/"point_cloud.ply"]
    for p in candidates:
        if p.exists(): return FileResponse(str(p),media_type="application/octet-stream",filename="aerosplat_model.ply")
    raise HTTPException(404,"Trained PLY not found")

@app.get("/api/report")
def report():
    with _lock: return {"incident":STATE.get("active_scene_name"),"generated_at":time.time(),"reconstruction":{k:STATE.get(k) for k in ["images_count","registered_images","points3d","observations","mean_track_length","mean_observations_per_image","mean_reprojection_error_px","camera_model"]},"model":{k:STATE.get(k) for k in ["model_ready","model_status","model_iteration","model_size_bytes"]},"analysis":STATE.get("analysis"),"triage":STATE.get("triage"),"recommendations":STATE.get("recommendations"),"route":STATE.get("route")}

_sync_existing()

if __name__=="__main__":
    import uvicorn; uvicorn.run("server:app",host="0.0.0.0",port=8002,reload=False)
