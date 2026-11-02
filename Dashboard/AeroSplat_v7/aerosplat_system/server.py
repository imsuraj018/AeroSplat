import hashlib, json, os, re, shutil, subprocess, sys, threading, time, uuid, struct
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / 'static'
WORKSPACE = Path(os.getenv('AEROSPLAT_WORKSPACE', r'C:\aerosplat_workspace'))
LEGACY_IMAGE_DIR = WORKSPACE / 'images'
SCENES_DIR = WORKSPACE / 'scenes'
EVIDENCE_DIR = WORKSPACE / 'evidence'
JOBS_DIR = WORKSPACE / 'jobs'
STATE_FILE = WORKSPACE / 'aerosplat_state.json'
COLMAP_EXE = Path(os.getenv('COLMAP_EXE', r'C:\bin\colmap.exe'))
GAUSSIAN_DIR = Path(os.getenv('GAUSSIAN_SPLATTING_DIR', r'C:\Users\Admin\Downloads\AeroSplat_Upgraded\external\gaussian-splatting'))
TRAIN_PY = GAUSSIAN_DIR / 'train.py'
VALID_EXTS = {'.jpg','.jpeg','.png','.bmp','.webp','.JPG','.JPEG','.PNG'}
DEFAULT_ITERATIONS = int(os.getenv('AEROSPLAT_ITERATIONS','7000'))
DEFAULT_RESOLUTION = int(os.getenv('AEROSPLAT_RESOLUTION','2'))
BUILD_MAX_JOBS = int(os.getenv('AEROSPLAT_MAX_JOBS','2'))
MAX_UPLOAD_MB = int(os.getenv('AEROSPLAT_MAX_UPLOAD_MB','100'))

for d in (WORKSPACE, LEGACY_IMAGE_DIR, SCENES_DIR, EVIDENCE_DIR, JOBS_DIR, STATIC_DIR): d.mkdir(parents=True, exist_ok=True)
app = FastAPI(title='AeroSplat Command Center', version='5.0.0')
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')
_lock=threading.RLock(); _pipeline_thread=None

def safe_name(name:str)->str: return Path(name or '').name

def image_files(d:Path)->List[Path]:
    if not d.exists(): return []
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in {e.lower() for e in VALID_EXTS}], key=lambda p:p.name.lower())

def hash_files(files):
    h=hashlib.sha256()
    for p in sorted(files,key=lambda x:x.name.lower()):
        try: s=p.stat()
        except FileNotFoundError: continue
        h.update(p.name.encode()); h.update(str(s.st_size).encode()); h.update(str(s.st_mtime_ns).encode())
    return h.hexdigest()[:16]

def read_json(p, default):
    try: return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default
    except Exception: return default

def save_json(p,data):
    p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(data,indent=2),encoding='utf-8'); tmp.replace(p)

def default_state():
    return {'active_scene_id':'legacy','active_scene_name':'Existing Workspace','job_id':None,'stage':'idle','status':'Ready','progress':0,'busy':False,'started_at':None,'finished_at':None,'updated_at':time.time(),'images_count':0,'reconstruction_images_count':0,'evidence_images_count':0,'registered_images':0,'points3d':0,'observations':0,'mean_track_length':0.0,'mean_observations_per_image':0.0,'mean_reprojection_error_px':0.0,'camera_model':'—','model_ready':False,'model_status':'NONE','model_iteration':None,'model_size_bytes':0,'model_source':None,'model_job_dir':None,'dataset_hash':None,'model_dataset_hash':None,'new_images_count':0,'last_error':None,'damage':{},'triage':{'label':'PENDING','score':None,'severity':'No assessment yet','directives':[]},'route':{'status':'Not computed','distance_m':None,'eta_min':None,'blocked_zones':0},'log':[]}
STATE=default_state(); STATE.update(read_json(STATE_FILE,{}))

def save_state():
    with _lock: STATE['updated_at']=time.time(); save_json(STATE_FILE,STATE)

def log(msg):
    with _lock: STATE.setdefault('log',[]).append({'time':time.time(),'message':msg}); STATE['log']=STATE['log'][-200:]
    save_state()

def set_state(**kw):
    with _lock: STATE.update(kw)
    save_state()

def scene_root(sid): return SCENES_DIR/sid
def scene_image_dir(sid): return LEGACY_IMAGE_DIR if sid=='legacy' else scene_root(sid)/'images'
def scene_evidence_dir(sid): return EVIDENCE_DIR/'legacy' if sid=='legacy' else scene_root(sid)/'evidence'
def scene_model_root(sid): return WORKSPACE/'gaussian_output' if sid=='legacy' else scene_root(sid)/'gaussian_output'

def ensure_scene(sid):
    scene_image_dir(sid).mkdir(parents=True,exist_ok=True); scene_evidence_dir(sid).mkdir(parents=True,exist_ok=True)
    if sid!='legacy': scene_root(sid).mkdir(parents=True,exist_ok=True)

def scene_meta(sid):
    if sid=='legacy': return {'id':'legacy','name':'Existing Workspace','description':'Existing AeroSplat workspace dataset.'}
    d=read_json(scene_root(sid)/'scene.json',{}); return {'id':sid,'name':d.get('name',sid),'description':d.get('description','')}

def list_scenes():
    out=[scene_meta('legacy')]
    for p in sorted(SCENES_DIR.iterdir(),key=lambda x:x.name.lower()):
        if p.is_dir(): out.append(scene_meta(p.name))
    return out

def active_sid(): sid=STATE.get('active_scene_id') or 'legacy'; ensure_scene(sid); return sid

def find_latest_model(root:Path):
    c=[]; roots=[root]
    if root==scene_model_root('legacy'): roots += [WORKSPACE/'gaussian_test']
    for j in JOBS_DIR.glob('*'):
        if j.is_dir(): roots += [j/'gaussian_model',j]
    for base in roots:
        pcroot=base/'point_cloud'
        if not pcroot.exists(): continue
        for it in pcroot.glob('iteration_*'):
            try: n=int(it.name.split('_')[-1])
            except: continue
            p=it/'point_cloud.ply'
            if p.exists(): c.append((p.stat().st_mtime,n,p,base))
    if not c:return None
    _,n,p,b=max(c,key=lambda x:x[0]); return {'iteration':n,'path':p,'root':b,'size':p.stat().st_size}

def parse_stats(txt):
    pats={'registered_images':r'Registered images:\s+(\d+)','points3d':r'Points:\s+(\d+)','observations':r'Observations:\s+(\d+)','mean_track_length':r'Mean track length:\s+([\d.]+)','mean_observations_per_image':r'Mean observations per image:\s+([\d.]+)','mean_reprojection_error_px':r'Mean reprojection error:\s+([\d.]+)px'}
    out={}
    for k,p in pats.items():
        m=re.search(p,txt)
        if m: out[k]=int(m.group(1)) if k in ('registered_images','points3d','observations') else float(m.group(1))
    return out

def run(cmd,cwd=None,env=None,stage=None):
    log('$ '+' '.join('"'+x+'"' if ' ' in x else x for x in cmd)); out=[]
    try: proc=subprocess.Popen(cmd,cwd=str(cwd) if cwd else None,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
    except FileNotFoundError as e: raise RuntimeError(f'Executable not found: {cmd[0]}') from e
    assert proc.stdout
    for raw in proc.stdout:
        line=raw.rstrip();
        if not line: continue
        out.append(line)
        if stage=='3dgs':
            m=re.search(r'Training progress:\s*(\d+)%',line)
            if m: set_state(stage='3dgs',status=f'Training 3DGS — {m.group(1)}%',progress=60+int(int(m.group(1))*.4))
        elif stage=='colmap_features' and 'Processed file' in line: set_state(status=line,progress=8)
        elif stage=='colmap_match' and 'Processing image' in line: set_state(status=line,progress=20)
        elif stage=='colmap_mapper' and 'Keeping successful reconstruction' in line: set_state(status='Sparse reconstruction accepted',progress=45)
    code=proc.wait(); text='\n'.join(out)
    if code!=0: raise RuntimeError(f'Command failed with exit code {code}:\n'+'\n'.join(out[-80:]))
    return text

def find_colmap_model(root:Path)->Path:
    for p in [root,root/'0']+[x for x in root.glob('*') if x.is_dir() and x.name.isdigit()]:
        if all((p/f).exists() for f in ('cameras.bin','images.bin','points3D.bin')): return p
    raise RuntimeError(f'No valid COLMAP model under {root}')

def read_camera_model(model:Path):
    txt=model.parent/(model.name+'_txt'); txt.mkdir(parents=True,exist_ok=True)
    try:
        run([str(COLMAP_EXE),'model_converter','--input_path',str(model),'--output_path',str(txt),'--output_type','TXT'])
        p=txt/'cameras.txt'
        for line in p.read_text(errors='ignore').splitlines():
            if line and not line.startswith('#'):
                parts=line.split(); return parts[1]
    except Exception: pass
    return '—'

def prepare_graphdeco(dense):
    s=dense/'sparse'; s0=s/'0'; s0.mkdir(parents=True,exist_ok=True)
    for p in s.glob('*.bin'): shutil.copy2(p,s0/p.name)
    for n in ('cameras.bin','images.bin','points3D.bin'):
        if not (s0/n).exists(): raise RuntimeError(f'Missing {n}')
    return dense

def snapshot_images(job,sid):
    raw=job/'raw'/'images'; raw.mkdir(parents=True,exist_ok=True)
    src=scene_image_dir(sid); files=image_files(src)
    for p in files: shutil.copy2(p,raw/p.name)
    return raw,image_files(raw)

def _sync_model():
    sid=active_sid()
    files=image_files(scene_image_dir(sid))
    ev=image_files(scene_evidence_dir(sid))
    model=find_latest_model(scene_model_root(sid))
    dh=hash_files(files) if files else None
    if not model:
        set_state(model_ready=False,model_status='NONE',model_dataset_hash=None,dataset_hash=dh,images_count=len(files),reconstruction_images_count=len(files),evidence_images_count=len(ev))
        return
    # Keep the existing trained model active, but synchronize reconstruction telemetry from the actual COLMAP model.
    stats={}
    cam='—'
    dense=WORKSPACE/'dense' if sid=='legacy' else scene_root(sid)/'dense'
    candidates=[]
    if dense.exists():
        try: candidates.append(find_colmap_model(dense/'sparse'))
        except Exception: pass
    try:
        model_root=model['root']
        if model_root.exists():
            # A job model may contain a copied sparse reconstruction.
            for cand in [model_root/'dense'/'sparse', model_root/'sparse']:
                if cand.exists():
                    try: candidates.append(find_colmap_model(cand))
                    except Exception: pass
    except Exception: pass
    if candidates:
        try:
            txt=run([str(COLMAP_EXE),'model_analyzer','--path',str(candidates[0])])
            stats=parse_stats(txt)
            cam=read_camera_model(candidates[0])
        except Exception:
            pass
    # Model-dataset hash is stored only for reconstructions created through AeroSplat; legacy models get current hash once.
    mh=STATE.get('model_dataset_hash') or dh
    st={
        'model_ready':True,'model_status':'CURRENT' if mh==dh else 'STALE',
        'model_dataset_hash':mh,'dataset_hash':dh,
        'model_iteration':model['iteration'],'model_size_bytes':model['size'],
        'model_source':'existing workspace model','model_job_dir':str(model['root']),
        'images_count':len(files),'reconstruction_images_count':len(files),'evidence_images_count':len(ev),
    }
    st.update(stats)
    if cam!='—': st['camera_model']=cam
    set_state(**st)
    # Populate image/geometry screening immediately so the dashboard is not empty on first load.
    try:
        dmg,tri=build_damage(files, sparse_points=stats.get('points3d',0), model_path=model['path'])
        set_state(damage=dmg, triage=tri)
    except Exception as exc:
        log('Initial screening unavailable: '+str(exc))


def dataset_analysis(files, model_points=None):
    # Data-quality / visual screening only. These are not engineering measurements.
    sharp=[]; brightness=[]; dark_ratio=[]
    try:
        from PIL import Image, ImageStat, ImageFilter
        for p in files[:128]:
            im=Image.open(p).convert('L')
            small=im.resize((480,360))
            edge=small.filter(ImageFilter.FIND_EDGES)
            sharp.append(float(ImageStat.Stat(edge).var[0]))
            mean=float(ImageStat.Stat(small).mean[0])
            brightness.append(mean)
            dark_ratio.append(1.0 if mean<65 else 0.0)
    except Exception:
        pass
    return {
        'status':'Automated visual screening only — not an engineering measurement',
        'images_analyzed':len(files),
        'mean_edge_variance':round(sum(sharp)/len(sharp),2) if sharp else None,
        'mean_brightness':round(sum(brightness)/len(brightness),1) if brightness else None,
        'dark_frame_ratio':round(sum(dark_ratio)/len(dark_ratio),3) if dark_ratio else None,
        'candidate_damage_frames':None,
        'candidate_damage_ratio':None,
        'crack_length_m':None,
        'crack_width_mm':None,
        'wall_tilt_deg':None,
        'rubble_volume_m3':None,
        'measurement_status':'Awaiting validated segmentation + metric calibration'
    }

def pca_screening(points):
    if not points: return None
    try:
        import numpy as np
        a=np.asarray(points,dtype=np.float64); c=a-a.mean(0); _,_,vh=np.linalg.svd(c,full_matrices=False); n=vh[-1]
        # Coordinate-frame inclination is a scene-geometry indicator, not automatically a wall-tilt measurement.
        z=np.array([0.,0.,1.]); ang=np.degrees(np.arccos(np.clip(abs(float(np.dot(n,z))),-1,1)))
        return float(round(ang,2))
    except Exception:
        return None

def build_damage(files, sparse_points=0, model_path=None):
    d=dataset_analysis(files)
    # Compute a lightweight image-quality warning from actual source frames.
    score=0.0; reasons=[]
    if d['dark_frame_ratio'] is not None and d['dark_frame_ratio']>0.20:
        score+=0.15; reasons.append('A noticeable fraction of frames are dark; inspect exposure before damage classification.')
    if d['mean_edge_variance'] is not None and d['mean_edge_variance']<35:
        score+=0.10; reasons.append('Image edge detail is relatively low; closer imagery may improve segmentation.')
    # Use a sampled Gaussian geometry orientation as an *orientation indicator* only.
    if model_path:
        try:
            pts=ply_sample(model_path,8000)
            xyz=[p[:3] for p in pts]
            tilt=pca_screening(xyz)
        except Exception:
            tilt=None
    else:
        tilt=None
    d['scene_surface_inclination_deg']=tilt
    d['structural_metrics_available']=False
    label='GREEN' if score<.30 else 'YELLOW' if score<.60 else 'RED'
    severity='Low data-quality concern' if label=='GREEN' else 'Moderate data-quality concern' if label=='YELLOW' else 'High data-quality concern'
    directives=[
        'No structural safety verdict is issued by this screening layer.',
        'Run the validated damage-segmentation and metric-calibration modules before making access decisions.'
    ]+reasons
    if tilt is not None:
        directives.append(f'Scene geometry includes a dominant surface inclination indicator of {tilt:.2f}°; this is not a wall-tilt measurement.')
    tri={'label':label,'score':round(score,2),'severity':severity,'confidence':None,'directives':directives}
    return d,tri

# Initialize from the existing workspace only after analysis functions are defined.
_sync_model()

# model points endpoint: parse binary little-endian PLY x/y/z + optional color, sample deterministically.
def ply_sample(path:Path,max_points:int=25000):
    SH_C0=0.28209479177387814
    with path.open('rb') as f:
        header=[]; fmt=None; n=0; decl=[]
        while True:
            raw=f.readline()
            if not raw: raise RuntimeError('Invalid PLY: missing end_header')
            line=raw.decode('ascii','ignore').strip(); header.append(line)
            if line.startswith('format '): fmt=line.split()[1]
            elif line.startswith('element vertex '): n=int(line.split()[-1])
            elif line.startswith('property ') and not line.startswith('property list'):
                parts=line.split(); decl.append((parts[2],parts[1]))
            if line=='end_header': break
        if fmt!='binary_little_endian': raise RuntimeError(f'Unsupported PLY format: {fmt}')
        size_map={'float':4,'float32':4,'double':8,'uchar':1,'uint8':1,'short':2,'ushort':2,'int':4,'uint':4,'int32':4,'uint32':4}
        fmt_map={'float':'f','float32':'f','double':'d','uchar':'B','uint8':'B','short':'h','ushort':'H','int':'i','uint':'I','int32':'i','uint32':'I'}
        stride=0; offsets={}
        for name,t in decl:
            if t not in size_map: raise RuntimeError(f'Unsupported PLY property type: {t}')
            offsets[name]=(stride,t); stride+=size_map[t]
        step=max(1,n//max_points) if n else 1
        out=[]
        for i in range(n):
            rec=f.read(stride)
            if len(rec)<stride: break
            if i%step: continue
            def val(name):
                item=offsets.get(name)
                if not item:return None
                off,t=item
                return struct.unpack_from('<'+fmt_map[t],rec,off)[0]
            x,y,z=val('x'),val('y'),val('z')
            if x is None or y is None or z is None: continue
            # GraphDeco/INRIA PLYs store view-independent DC color in f_dc_0..2.
            dc=[val('f_dc_0'),val('f_dc_1'),val('f_dc_2')]
            if all(v is not None for v in dc):
                rgb=[int(max(0,min(255,round((0.5+SH_C0*v)*255)))) for v in dc]
            else:
                rgb=[val('red'),val('green'),val('blue')]
                if any(v is None for v in rgb): rgb=[170,170,170]
                rgb=[int(max(0,min(255,v))) for v in rgb]
            op=val('opacity')
            alpha=1.0
            if op is not None:
                try: alpha=1/(1+__import__('math').exp(-max(-20,min(20,op))))
                except Exception: alpha=1.0
            out.append([float(x),float(y),float(z),rgb[0],rgb[1],rgb[2],round(alpha,3)])
        return out

def active_model_path():
    sid=active_sid(); m=find_latest_model(scene_model_root(sid)); return m['path'] if m else None


def pipeline(job_id,sid,iterations):
    job=JOBS_DIR/job_id; raw,files=snapshot_images(job,sid); db=job/'database.db'; sparse=job/'sparse'; dense=job/'dense'; out=job/'gaussian_model'
    try:
        if len(files)<3: raise RuntimeError('At least 3 overlapping reconstruction images are required.')
        set_state(stage='colmap_features',status=f'Extracting features from {len(files)} images…',progress=2)
        run([str(COLMAP_EXE),'feature_extractor','--database_path',str(db),'--image_path',str(raw),'--ImageReader.camera_model','SIMPLE_RADIAL','--ImageReader.single_camera','1','--FeatureExtraction.use_gpu','0','--FeatureExtraction.max_image_size','2400','--FeatureExtraction.num_threads','8'],stage='colmap_features')
        set_state(stage='colmap_match',status='Matching overlapping views…',progress=12)
        run([str(COLMAP_EXE),'sequential_matcher','--database_path',str(db),'--SequentialMatching.overlap','10','--FeatureMatching.use_gpu','0'],stage='colmap_match')
        sparse.mkdir(parents=True,exist_ok=True); set_state(stage='colmap_mapper',status='Building sparse reconstruction…',progress=25)
        run([str(COLMAP_EXE),'mapper','--database_path',str(db),'--image_path',str(raw),'--output_path',str(sparse)],stage='colmap_mapper')
        model=find_colmap_model(sparse); stats=parse_stats(run([str(COLMAP_EXE),'model_analyzer','--path',str(model)])); cam=read_camera_model(model)
        if stats.get('registered_images',0)<3: raise RuntimeError('COLMAP registered fewer than 3 images.')
        set_state(**stats,camera_model=cam,stage='undistort',status='Undistorting for GraphDeco…',progress=48)
        run([str(COLMAP_EXE),'image_undistorter','--image_path',str(raw),'--input_path',str(model),'--output_path',str(dense),'--output_type','COLMAP'])
        src=prepare_graphdeco(dense); env=os.environ.copy(); env['CUDA_HOME']=env.get('AEROSPLAT_CUDA_HOME',r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4'); env['TORCH_CUDA_ARCH_LIST']='8.6'; env['MAX_JOBS']=str(BUILD_MAX_JOBS); env['DISTUTILS_USE_SDK']='1'; env['PATH']=str(Path(env['CUDA_HOME'])/'bin')+os.pathsep+env.get('PATH','')
        set_state(stage='3dgs',status=f'Training 3DGS — 0/{iterations}',progress=60)
        run([sys.executable,str(TRAIN_PY),'-s',str(src),'-m',str(out),'--iterations',str(iterations),'--save_iterations',str(iterations),'--disable_viewer','-r',str(DEFAULT_RESOLUTION),'--data_device','cpu'],cwd=GAUSSIAN_DIR,env=env,stage='3dgs')
        pc=out/'point_cloud'/f'iteration_{iterations}'/'point_cloud.ply'
        if not pc.exists(): raise RuntimeError('3DGS finished but point_cloud.ply was not created.')
        target=scene_model_root(sid)/'point_cloud'/f'iteration_{iterations}'/'point_cloud.ply'; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(pc,target)
        ds=hash_files(files); dmg,tri=build_damage(files)
        set_state(stage='ready',status='Reconstruction complete',progress=100,busy=False,finished_at=time.time(),model_ready=True,model_status='CURRENT',model_iteration=iterations,model_size_bytes=target.stat().st_size,model_source='AeroSplat reconstruction job',model_job_dir=str(scene_model_root(sid)),model_dataset_hash=ds,dataset_hash=ds,reconstruction_images_count=len(files),images_count=len(files),evidence_images_count=len(image_files(scene_evidence_dir(sid))),new_images_count=0,damage=dmg,triage=tri,route={'status':'Awaiting road/hazard graph','distance_m':None,'eta_min':None,'blocked_zones':0},last_error=None)
        log(f'Reconstruction ready: {target}')
    except Exception as e:
        set_state(busy=False,stage='error',status='Pipeline failed',last_error=str(e),finished_at=time.time(),model_status='STALE' if STATE.get('model_ready') else 'NONE')
        log('ERROR: '+str(e))

def start_pipeline(iterations):
    global _pipeline_thread
    with _lock:
        if STATE.get('busy'): raise RuntimeError('A reconstruction job is already running.')
        sid=active_sid()
    files=image_files(scene_image_dir(sid))
    if len(files)<3: raise RuntimeError('Add at least 3 overlapping reconstruction images first.')
    jid=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    set_state(job_id=jid,busy=True,stage='queued',status='Reconstruction queued',progress=0,started_at=time.time(),finished_at=None,new_images_count=0,last_error=None,model_status='BUILDING')
    _pipeline_thread=threading.Thread(target=pipeline,args=(jid,sid,iterations),daemon=True); _pipeline_thread.start(); return jid

@app.get('/',response_class=HTMLResponse)
def index(): return (STATIC_DIR/'index.html').read_text(encoding='utf-8')
@app.get('/api/health')
def health(): return {'api':'ok','colmap':str(COLMAP_EXE),'colmap_exists':COLMAP_EXE.exists(),'graphdeco':str(TRAIN_PY),'graphdeco_exists':TRAIN_PY.exists(),'workspace':str(WORKSPACE),'active_scene':active_sid()}
@app.get('/api/scenes')
def scenes(): return {'active_scene_id':active_sid(),'scenes':list_scenes()}
@app.post('/api/scenes')
def create_scene(name:str=Query(...,min_length=1),description:str=Query('')):
    if STATE.get('busy'): raise HTTPException(409,'Wait until reconstruction finishes.')
    sid=time.strftime('incident_%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:5]; ensure_scene(sid); save_json(scene_root(sid)/'scene.json',{'id':sid,'name':name.strip(),'description':description.strip()})
    set_state(active_scene_id=sid,active_scene_name=name.strip(),model_ready=False,model_status='NONE',model_job_dir=None,model_iteration=None,model_dataset_hash=None,new_images_count=0); _sync_model(); log('Created scene '+sid); return {'scene':scene_meta(sid)}
@app.post('/api/scenes/select')
def select_scene(scene_id:str=Query(...)):
    if STATE.get('busy'): raise HTTPException(409,'Wait until reconstruction finishes.')
    if scene_id!='legacy' and not scene_root(scene_id).exists(): raise HTTPException(404,'Scene not found.')
    ensure_scene(scene_id); set_state(active_scene_id=scene_id,active_scene_name=scene_meta(scene_id)['name'],new_images_count=0); _sync_model(); return {'scene':scene_meta(scene_id)}
@app.post('/api/upload-images')
async def upload_images(files:List[UploadFile]=File(...),handling:str=Query('append')):
    if STATE.get('busy'): raise HTTPException(409,'Wait until current reconstruction finishes.')
    handling=handling.lower().strip()
    if handling not in ('append','evidence'): raise HTTPException(400,'handling must be append or evidence')
    dest=scene_image_dir(active_sid()) if handling=='append' else scene_evidence_dir(active_sid()); dest.mkdir(parents=True,exist_ok=True); added=[]; rejected=[]
    for up in files:
        name=safe_name(up.filename or ''); ext=Path(name).suffix.lower()
        if not name or ext not in {e.lower() for e in VALID_EXTS}: rejected.append({'name':name,'reason':'Unsupported image format'}); continue
        target=dest/name
        if target.exists(): target=dest/f'{target.stem}_{uuid.uuid4().hex[:5]}{target.suffix}'; name=target.name
        total=0
        try:
            with target.open('wb') as f:
                while True:
                    chunk=await up.read(1024*1024)
                    if not chunk: break
                    total+=len(chunk)
                    if total>MAX_UPLOAD_MB*1024*1024: target.unlink(missing_ok=True); rejected.append({'name':name,'reason':f'Over {MAX_UPLOAD_MB} MB'}); break
                    f.write(chunk)
            if target.exists(): added.append(target.name)
        finally: await up.close()
    imgs=image_files(scene_image_dir(active_sid())); ev=image_files(scene_evidence_dir(active_sid())); dh=hash_files(imgs) if imgs else None
    set_state(images_count=len(imgs),reconstruction_images_count=len(imgs),evidence_images_count=len(ev),dataset_hash=dh,new_images_count=len(added) if handling=='append' else STATE.get('new_images_count',0),model_status='STALE' if handling=='append' and added and STATE.get('model_ready') else STATE.get('model_status'))
    log(f'Added {len(added)} image(s) as {handling}.'); return {'added':added,'rejected':rejected,'handling':handling,'reconstruction_images_count':len(imgs),'evidence_images_count':len(ev),'model_status':STATE.get('model_status')}
@app.post('/api/run')
def run_api(iterations:int=Query(DEFAULT_ITERATIONS,ge=100,le=30000)):
    try: jid=start_pipeline(int(iterations)); return {'job_id':jid,'iterations':int(iterations)}
    except RuntimeError as e: raise HTTPException(409,str(e))
@app.get('/api/state')
def state_api():
    sid=active_sid(); imgs=image_files(scene_image_dir(sid)); ev=image_files(scene_evidence_dir(sid)); dh=hash_files(imgs) if imgs else None
    with _lock: st=json.loads(json.dumps(STATE))
    st.update(active_scene_id=sid,active_scene_name=scene_meta(sid)['name'],images_count=len(imgs),reconstruction_images_count=len(imgs),evidence_images_count=len(ev),dataset_hash=dh)
    st['model_stale']=bool(st.get('model_ready') and st.get('model_dataset_hash') and dh!=st.get('model_dataset_hash'))
    if st['model_stale'] and not st.get('busy'): st['model_status']='STALE'
    elif st.get('model_ready') and not st.get('busy') and not st.get('model_status'): st['model_status']='CURRENT'
    st['images']=[{'name':p.name,'url':f'/api/scene-image/{sid}/{p.name}','kind':'reconstruction'} for p in imgs]
    st['evidence']=[{'name':p.name,'url':f'/api/evidence-image/{sid}/{p.name}','kind':'evidence'} for p in ev]
    st['model_url']='/api/model.ply' if st.get('model_ready') else None
    st['viewer_points_url']='/api/model/points' if st.get('model_ready') else None
    return JSONResponse(st,headers={'Cache-Control':'no-store'})
@app.get('/api/scene-image/{sid}/{filename}')
def scene_image(sid,filename):
    p=scene_image_dir(sid)/safe_name(filename)
    if not p.exists() or p.suffix.lower() not in {e.lower() for e in VALID_EXTS}: raise HTTPException(404,'Image not found')
    return FileResponse(str(p))
@app.get('/api/evidence-image/{sid}/{filename}')
def evidence_image(sid,filename):
    p=scene_evidence_dir(sid)/safe_name(filename)
    if not p.exists(): raise HTTPException(404,'Evidence image not found')
    return FileResponse(str(p))
@app.get('/api/model.ply')
def model_ply():
    p=active_model_path()
    if not p: raise HTTPException(404,'No trained Gaussian model available')
    return FileResponse(str(p),media_type='application/octet-stream',filename='aerosplat_model.ply')
@app.get('/api/model/points')
def model_points(limit:int=Query(25000,ge=1000,le=100000)):
    p=active_model_path()
    if not p: raise HTTPException(404,'No model')
    try: pts=ply_sample(p,limit)
    except Exception as e: raise HTTPException(500,str(e))
    return {'count':len(pts),'points':pts}
@app.post('/api/analyze')
def analyze():
    sid=active_sid(); files=image_files(scene_image_dir(sid)); dmg,tri=build_damage(files); set_state(damage=dmg,triage=tri,status='Screening analysis complete'); return {'damage':dmg,'triage':tri}
@app.get('/api/report')
def report():
    with _lock: return json.loads(json.dumps({'incident':STATE.get('active_scene_name'),'generated_at':time.time(),'reconstruction':{k:STATE.get(k) for k in ['images_count','registered_images','points3d','observations','mean_track_length','mean_observations_per_image','mean_reprojection_error_px','camera_model']},'model':{k:STATE.get(k) for k in ['model_ready','model_status','model_iteration','model_size_bytes']},'damage':STATE.get('damage'),'triage':STATE.get('triage'),'route':STATE.get('route')}))

if __name__=='__main__':
    import uvicorn; uvicorn.run('server:app',host='0.0.0.0',port=8000,reload=False)
