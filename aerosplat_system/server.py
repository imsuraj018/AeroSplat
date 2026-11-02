from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .colmap_loader import ColmapDataset
from .config import CONFIG
from .damage_quantifier import StructuralDamageQuantifier
from .gaussian_model import save_initial_splat_ply
from .gs_adapter import GaussianSplatAdapter
from .hazard_detector import HazardDetector
from .ingest_drone_data import DroneDataIngestor
from .risk_engine import RiskEngine
from .route_planner import RiskAwareAStar, demo_route
from .triage_engine import ATCTriageEngine

app = FastAPI(title="AeroSplat — Post-Disaster Digital Twin & Route Intelligence", version="2.0.0")
CONFIG.ensure_dirs()
app.mount("/images", StaticFiles(directory=CONFIG.images_dir), name="images")

STATE = {
    "status": "Ready",
    "is_busy": False,
    "error": None,
    "points3d": [],
    "point_colors": [],
    "image_filenames": [],
    "metrics": {},
    "hazards": [],
    "risk": {},
    "triage": {},
    "route": demo_route(),
    "gs": GaussianSplatAdapter().status(),
    "last_updated": 0.0,
}
LOCK = threading.Lock()


class RouteEdge(BaseModel):
    a: str
    b: str
    distance_m: float = Field(gt=0)
    risk: float = Field(default=0, ge=0, le=1)
    obstruction: float = Field(default=0, ge=0, le=1)
    flood: float = Field(default=0, ge=0, le=1)
    fire: float = Field(default=0, ge=0, le=1)
    blocked: bool = False


class RouteRequest(BaseModel):
    coordinates: dict[str, list[float]]
    edges: list[RouteEdge]
    start: str
    goal: str
    weights: dict[str, float] = {}


def _image_names() -> list[str]:
    return sorted([p.name for p in Path(CONFIG.images_dir).iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}])


def sync_pipeline(train_3dgs: bool = False, allow_synthetic: bool = True) -> None:
    with LOCK:
        STATE["is_busy"] = True; STATE["status"] = "Running COLMAP/3D analysis..."; STATE["error"] = None
    try:
        ds = ColmapDataset(CONFIG.colmap_dir, CONFIG.images_dir)
        if len(ds.points3d) == 0:
            if not allow_synthetic:
                raise RuntimeError("No COLMAP reconstruction found. Upload at least 3 overlapping drone images.")
            ds.synthesize_disaster_geometry()
        save_initial_splat_ply(CONFIG.scene_ply, ds.points3d, ds.point_colors, radius=max(CONFIG.default_metric_scale_m_per_sfm_unit*0.5,0.002))
        gs = GaussianSplatAdapter().run() if train_3dgs else {"status":"initialization_only", **GaussianSplatAdapter().status()}
        q = StructuralDamageQuantifier(CONFIG.default_metric_scale_m_per_sfm_unit)
        metrics = {
            "crack": q.analyze_crack(ds.points3d[ds.crack_indices]),
            "wall_tilt": q.analyze_wall_tilt(ds.points3d[ds.wall_indices]),
            "debris": q.analyze_debris_volume(ds.points3d[ds.debris_indices]),
        }
        names = _image_names()
        full_paths = [os.path.join(CONFIG.images_dir,n) for n in names]
        detector = HazardDetector()
        hazards = detector.predict(full_paths[:30]) if detector.available else detector.heuristic_hazards(full_paths[:30])
        risk = RiskEngine().evaluate(metrics, hazards).as_dict()
        triage = ATCTriageEngine.evaluate(metrics, risk)
        with LOCK:
            idx = np.arange(len(ds.points3d))
            if len(idx) > CONFIG.max_points_for_browser:
                rng = np.random.default_rng(42); idx = rng.choice(idx, CONFIG.max_points_for_browser, replace=False)
            STATE.update({
                "status":"Ready","is_busy":False,"points3d":ds.points3d[idx].tolist(),"point_colors":ds.point_colors[idx].tolist(),
                "image_filenames":names,"metrics":metrics,"hazards":hazards,"risk":risk,"triage":triage,"gs":gs,"last_updated":time.time()
            })
    except Exception as exc:
        with LOCK:
            STATE["is_busy"] = False; STATE["status"] = "Error"; STATE["error"] = str(exc); STATE["last_updated"] = time.time()
        raise


@app.on_event("startup")
def startup() -> None:
    # Do not run expensive COLMAP/ML automatically. Load a synthetic scene only if workspace is empty.
    try:
        sync_pipeline(train_3dgs=False, allow_synthetic=True)
    except Exception as exc:
        STATE["error"] = str(exc)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/api/state")
def get_state():
    with LOCK:
        return JSONResponse({k:v for k,v in STATE.items() if k not in {"points3d","point_colors"}}, headers={"Cache-Control":"no-store"})


@app.get("/api/splat-points")
def splat_points():
    with LOCK:
        return JSONResponse({"points":STATE["points3d"],"colors":STATE["point_colors"],"timestamp":STATE["last_updated"]})


@app.get("/api/report")
def report():
    with LOCK:
        return JSONResponse({"metrics":STATE["metrics"],"hazards":STATE["hazards"],"risk":STATE["risk"],"triage":STATE["triage"],"route":STATE["route"],"gs":STATE["gs"]})


@app.get("/api/route")
def route():
    return JSONResponse(STATE["route"])


@app.get("/api/models/3dgs-status")
def gs_status():
    return JSONResponse(GaussianSplatAdapter().status())


@app.get("/api/scene.ply")
def scene_ply():
    if os.path.exists(CONFIG.scene_ply):
        return FileResponse(CONFIG.scene_ply, media_type="application/octet-stream", filename="scene_model.ply")
    raise HTTPException(404,"No scene model available")


@app.post("/api/sync")
def sync():
    sync_pipeline(train_3dgs=False, allow_synthetic=False)
    return {"status":"synced","images_count":len(STATE["image_filenames"])}


@app.post("/api/train-3dgs")
def train_3dgs():
    sync_pipeline(train_3dgs=True, allow_synthetic=False)
    return {"status":"completed","gs":STATE["gs"]}


@app.post("/api/upload-images")
async def upload_images(files: List[UploadFile] = File(...)):
    os.makedirs(CONFIG.images_dir, exist_ok=True)
    saved = []
    for f in files:
        name = os.path.basename(f.filename or "upload.jpg")
        target = Path(CONFIG.images_dir)/name
        with open(target,"wb") as out:
            shutil.copyfileobj(f.file,out)
        saved.append(name)
    videos = [x for x in saved if Path(x).suffix.lower() in {".mp4",".mov",".avi",".mkv"}]
    ing = DroneDataIngestor()
    if videos:
        ing.extract_frames_from_video(str(Path(CONFIG.images_dir)/videos[0]))
        ing.run_colmap(is_video=True)
    else:
        ing.run_colmap(is_video=False)
    sync_pipeline(train_3dgs=False, allow_synthetic=False)
    return {"status":"success","saved":saved,"images_count":len(STATE["image_filenames"])}


@app.post("/api/route")
def custom_route(req: RouteRequest):
    try:
        coords = {k:(float(v[0]),float(v[1])) for k,v in req.coordinates.items() if len(v)>=2}
        planner = RiskAwareAStar(coords)
        for e in req.edges:
            planner.add_edge(e.a,e.b,e.distance_m,risk=e.risk,obstruction=e.obstruction,flood=e.flood,fire=e.fire,blocked=e.blocked)
        result = planner.solve(req.start,req.goal,req.weights)
        STATE["route"] = result
        return result
    except Exception as exc:
        raise HTTPException(400,str(exc))


HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AeroSplat 2.0</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
*{box-sizing:border-box}body{margin:0;background:#071018;color:#d9e2ea;font:14px system-ui,Segoe UI,sans-serif}header{height:56px;display:flex;align-items:center;justify-content:space-between;padding:0 18px;background:#0c1620;border-bottom:1px solid #22313d}h1{font-size:18px;margin:0;color:#7dd3fc}.pill{padding:5px 10px;border:1px solid #2d4251;border-radius:99px}.grid{display:grid;grid-template-columns:minmax(0,1fr) 360px;height:calc(100vh - 56px)}.left{display:grid;grid-template-rows:1fr 150px;min-width:0}.views{display:grid;grid-template-columns:1.25fr 1fr;min-height:0}.panel{background:#09131b;border-right:1px solid #22313d;border-bottom:1px solid #22313d;position:relative}.title{position:absolute;left:12px;top:10px;padding:5px 8px;background:#0b1721cc;border:1px solid #2b3d49;border-radius:6px;z-index:2}.film{display:flex;gap:9px;padding:12px;overflow:auto;background:#0c1620}.thumb{width:130px;height:110px;object-fit:cover;border-radius:7px;border:2px solid #233643}.thumb:hover{border-color:#7dd3fc}.side{padding:16px;overflow:auto;background:#0a141d}.card{border:1px solid #22313d;border-radius:10px;background:#0d1a24;padding:13px;margin-bottom:12px}.tag{font-size:22px;font-weight:800;padding:12px;border-radius:9px;text-align:center}.green{background:#0d2d1a;color:#65d58b;border:1px solid #1e7a47}.yellow{background:#2b250d;color:#f1cf65;border:1px solid #7a681e}.red{background:#321212;color:#ff7c7c;border:1px solid #7a2e2e}.kv{display:grid;grid-template-columns:1fr 1fr;gap:8px}.kv div{background:#0a141d;padding:9px;border-radius:7px}.v{font-size:18px;font-weight:700;margin-top:3px}.muted{color:#91a4b1;font-size:12px}button{background:#123044;color:#d9e2ea;border:1px solid #315267;border-radius:7px;padding:8px 10px;cursor:pointer}button:hover{background:#173d55}.route{line-height:1.8}.danger{color:#ff8a8a}.safe{color:#67e8a2}
</style></head><body><header><h1>AeroSplat 2.0 — Post-Disaster Digital Twin</h1><div class="pill" id="status">Ready</div></header>
<div class="grid"><main class="left"><section class="views"><div class="panel"><div class="title">3D Spatial Reconstruction</div><div id="scene" style="width:100%;height:100%"></div></div><div class="panel" style="display:flex;align-items:center;justify-content:center"><div class="title">High-Resolution Inspection</div><img id="mainimg" style="max-width:94%;max-height:88%;object-fit:contain;border-radius:8px;border:1px solid #2b3d49"/></div></section><section class="film" id="film"></section></main><aside class="side"><div class="card"><div id="placard" class="tag">LOADING</div><div class="muted" style="margin-top:8px">AI-assisted preliminary screening — not an engineering certificate.</div></div><div class="card"><b>Structural Metrics</b><div class="kv" style="margin-top:10px"><div><span class="muted">Wall tilt</span><div class="v" id="tilt">—</div></div><div><span class="muted">Crack width</span><div class="v" id="crack">—</div></div><div><span class="muted">Crack length</span><div class="v" id="length">—</div></div><div><span class="muted">Debris</span><div class="v" id="debris">—</div></div></div></div><div class="card"><b>Risk Engine</b><div style="margin-top:9px">Score: <span id="risk">—</span></div><div class="muted" id="factors" style="margin-top:8px"></div></div><div class="card"><b>Hazards</b><div id="hazards" class="muted" style="margin-top:8px">—</div></div><div class="card"><b>Recommended Responder Route</b><div class="route" id="route" style="margin-top:8px">—</div></div><div class="card"><button onclick="sync()">Re-analyze</button> <button onclick="train()">Run 3DGS</button></div></aside></div>
<script>
let scene,camera,renderer,controls,pts;
function init3d(){scene=new THREE.Scene();scene.background=new THREE.Color(0x050a0f);camera=new THREE.PerspectiveCamera(60,1,.01,2000);camera.position.set(12,10,16);renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(devicePixelRatio);document.getElementById('scene').appendChild(renderer.domElement);controls=new THREE.OrbitControls(camera,renderer.domElement);window.addEventListener('resize',resize);resize();animate()}
function resize(){let el=document.getElementById('scene');camera.aspect=el.clientWidth/el.clientHeight;camera.updateProjectionMatrix();renderer.setSize(el.clientWidth,el.clientHeight)}
function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera)}
function drawCloud(data){if(pts)scene.remove(pts);let g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(data.points.flat(),3));g.setAttribute('color',new THREE.Float32BufferAttribute(data.colors.flat(),3));let m=new THREE.PointsMaterial({size:.055,vertexColors:true,sizeAttenuation:true});pts=new THREE.Points(g,m);scene.add(pts);if(data.points.length){let c=data.points[0];}}
async function api(u,o){return fetch(u,o).then(r=>r.json())}
async function refresh(){let s=await api('/api/state');document.getElementById('status').textContent=s.status+(s.error?' — '+s.error:'');let t=s.triage||{};let tag=t.safety_tag||'—';let p=document.getElementById('placard');p.textContent=tag==='GREEN'?'GREEN — PRELIMINARY LOW RISK':tag==='YELLOW'?'YELLOW — RESTRICTED / PRELIMINARY':tag==='RED'?'RED — UNSAFE / PRELIMINARY':tag;p.className='tag '+(tag==='GREEN'?'green':tag==='YELLOW'?'yellow':tag==='RED'?'red':'');let m=s.metrics||{};document.getElementById('tilt').textContent=(m.wall_tilt?.tilt_degrees??0)+'°';document.getElementById('crack').textContent=(m.crack?.width_mm??0)+' mm';document.getElementById('length').textContent=(m.crack?.length_m??0)+' m';document.getElementById('debris').textContent=(m.debris?.volume_m3??0)+' m³';let r=s.risk||{};document.getElementById('risk').textContent=((r.score??0)*100).toFixed(1)+'% ('+(r.level||'—')+')';document.getElementById('factors').textContent=(r.factors||[]).join(' • ')||'No elevated factor';document.getElementById('hazards').innerHTML=(s.hazards||[]).map(h=>`${h.class} (${(h.confidence*100).toFixed(0)}%)`).join('<br>')||'No model hazards detected';let rr=s.route||{};document.getElementById('route').innerHTML=rr.found?`${rr.nodes.join(' → ')}<br><span class="muted">Distance ${rr.distance_m} m | Risk cost ${rr.risk_cost}</span>`:rr.reason||'No feasible route';let cloud=await api('/api/splat-points');drawCloud(cloud);let imgs=s.images||[];document.getElementById('film').innerHTML='';imgs.forEach((n,i)=>{let im=document.createElement('img');im.className='thumb';im.src='/images/'+encodeURIComponent(n);im.onclick=()=>document.getElementById('mainimg').src=im.src;document.getElementById('film').appendChild(im);if(i===0)document.getElementById('mainimg').src=im.src})}
async function sync(){await api('/api/sync',{method:'POST'});refresh()}async function train(){await api('/api/train-3dgs',{method:'POST'});refresh()}init3d();refresh();setInterval(refresh,5000);
</script></body></html>'''
