"""
Demo 4 - PTZ 自主巡視
根據各區域的 surprise 歷史自動決定攝影機巡視方向

啟動: python web_app.py → http://localhost:8772
"""
import sys, os, asyncio, time, base64, json, threading, hashlib
from collections import deque
from pathlib import Path
from datetime import datetime

for p in ["/home/rai/code/le-wm", os.path.expanduser("~/code/2026/le-wm-local")]:
    if os.path.exists(p): sys.path.insert(0, p); break

import cv2, numpy as np, h5py, torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import stable_pretraining as spt
from module import MLP, Embedder, ARPredictor
from jepa import JEPA

BASE_DIR = Path(__file__).parent.parent
DEMO2_CKPT = BASE_DIR.parent / "demo2-voe-anomaly" / "checkpoints"
DATA_DIR = BASE_DIR / "data"; DATA_DIR.mkdir(exist_ok=True)
CKPT_DIR = BASE_DIR / "checkpoints"; CKPT_DIR.mkdir(exist_ok=True)

class AppState:
    device = None; model = None
    regions = {}  # {name: {rtsp, last_observed, surprise_ema, alert_count, latent_history}}
    active_region = None
    auto_patrol = False
    recording = False; training = False
    record_progress = {}; train_progress = {}
state = AppState()
state.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def build_model(device):
    encoder = spt.backbone.utils.vit_hf("tiny",patch_size=16,image_size=64,pretrained=False,use_mask_token=False)
    hd=encoder.config.hidden_size;ed=192
    predictor=ARPredictor(num_frames=4,input_dim=ed,hidden_dim=hd,output_dim=hd,depth=2,heads=4,mlp_dim=hd*4,dropout=0.0)
    ae=Embedder(input_dim=2,emb_dim=ed)
    pj=MLP(input_dim=hd,output_dim=ed,hidden_dim=2048,norm_fn=torch.nn.BatchNorm1d)
    pp=MLP(input_dim=hd,output_dim=ed,hidden_dim=2048,norm_fn=torch.nn.BatchNorm1d)
    return JEPA(encoder=encoder,predictor=predictor,action_encoder=ae,projector=pj,pred_proj=pp).to(device)

def load_model(path):
    m=build_model(state.device)
    s=torch.load(path,map_location=state.device,weights_only=False)
    m.load_state_dict(s["model"] if isinstance(s,dict) and "model" in s else s)
    m.eval();state.model=m;return True

# Recording/Training reuse
sys.path.insert(0, str(BASE_DIR.parent / "demo2-voe-anomaly" / "src"))

app = FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])


@app.get("/about")
async def about_page():
    about_file = Path(__file__).parent.parent / "about.html"
    if about_file.exists():
        return HTMLResponse(about_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>About page not found</h1>")

@app.get("/")
async def index(): return HTMLResponse(FRONTEND)

@app.get("/api/status")
async def api_status():
    ckpts=[]
    for cb in [CKPT_DIR,DEMO2_CKPT]:
        if not cb.exists():continue
        for d in sorted(cb.iterdir()):
            if d.is_dir() and (d/"best_model.pt").exists():
                ckpts.append({"name":d.name,"path":str(d/"best_model.pt")})
    regions_info = {}
    for name, r in state.regions.items():
        regions_info[name] = {
            "rtsp": r["rtsp"],
            "last_observed": r.get("last_observed", 0),
            "surprise_ema": round(r.get("surprise_ema", 0), 6),
            "alert_count": r.get("alert_count", 0),
            "info_gain": round((time.time() - r.get("last_observed", time.time())) * r.get("surprise_ema", 0.001), 3),
        }
    return {"device":state.device,"model_loaded":state.model is not None,
            "checkpoints":ckpts,"regions":regions_info,
            "active_region":state.active_region,"auto_patrol":state.auto_patrol,
            "recording":state.recording,"training":state.training}

@app.post("/api/load_model")
async def api_load(body:dict):return {"loaded":load_model(body["path"])}

@app.post("/api/regions")
async def set_regions(body:dict):
    for name, cfg in body.get("regions",{}).items():
        if name not in state.regions:
            state.regions[name] = {"rtsp":cfg["rtsp"],"last_observed":0,"surprise_ema":0.001,"alert_count":0}
        else:
            state.regions[name]["rtsp"] = cfg["rtsp"]
    return {"regions":list(state.regions.keys())}

@app.post("/api/patrol")
async def toggle_patrol(body:dict):
    state.auto_patrol = body.get("enabled", False)
    return {"auto_patrol": state.auto_patrol}

@app.post("/api/record")
async def api_record(body:dict):
    if state.recording: return JSONResponse({"error":"busy"},400)
    from record_camera import record_rtsp
    def worker():
        state.recording=True;state.record_progress={"status":"recording"}
        record_rtsp(body.get("rtsp",""),body.get("duration",300),body.get("fps",5),64,
                    str(DATA_DIR/f"ptz_{datetime.now().strftime('%H%M%S')}.h5"))
        state.record_progress["status"]="done";state.recording=False
    threading.Thread(target=worker,daemon=True).start()
    return {"status":"started"}

@app.post("/api/train")
async def api_train(body:dict):
    if state.training: return JSONResponse({"error":"busy"},400)
    from train_voe import train as tv
    def worker():
        state.training=True;state.train_progress={"status":"training"}
        tv(body["data_path"],str(CKPT_DIR/body.get("name","ptz_model")),body.get("epochs",30),body.get("batch_size",32),body.get("lr",3e-4),state.device)
        state.train_progress["status"]="done";state.training=False
    threading.Thread(target=worker,daemon=True).start()
    return {"status":"started"}

@app.websocket("/ws/patrol")
async def ws_patrol(websocket:WebSocket):
    await websocket.accept()
    try:
        data=await websocket.receive_json()
        if data.get("model_path") and state.model is None:
            load_model(data["model_path"])
        if state.model is None:
            await websocket.send_json({"error":"No model"});return
        if not state.regions:
            await websocket.send_json({"error":"No regions configured"});return

        await websocket.send_json({"status":"started"})
        region_bufs = {n: deque(maxlen=5) for n in state.regions}
        region_surp = {n: deque(maxlen=100) for n in state.regions}

        while True:
            # Decide which region to observe
            if state.auto_patrol:
                best_region = max(state.regions.keys(),
                    key=lambda n: (time.time()-state.regions[n].get("last_observed",0)) * state.regions[n].get("surprise_ema",0.001))
            else:
                best_region = state.active_region or list(state.regions.keys())[0]

            state.active_region = best_region
            region = state.regions[best_region]

            # Capture frame
            cap = cv2.VideoCapture(region["rtsp"], cv2.CAP_FFMPEG)
            if not cap.isOpened():
                await websocket.send_json({"region":best_region,"error":"Cannot connect"})
                await asyncio.sleep(2);continue

            ret,frame=cap.read();cap.release()
            if not ret:
                await asyncio.sleep(1);continue

            region["last_observed"]=time.time()

            # LeWM surprise
            resized=cv2.resize(frame,(64,64))
            rgb=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
            tensor=torch.from_numpy(rgb).float().permute(2,0,1)/255.0
            region_bufs[best_region].append(tensor)
            buf=region_bufs[best_region]

            surprise=0
            if len(buf)>=5:
                frames=list(buf)
                ctx=torch.stack(frames[:4]).unsqueeze(0).to(state.device)
                nxt=torch.stack(frames[1:5]).unsqueeze(0).to(state.device)
                act=torch.zeros(1,4,2,device=state.device)
                with torch.no_grad():
                    ic=state.model.encode({"pixels":ctx,"action":act})
                    pred=state.model.predict(ic["emb"],ic["act_emb"])
                    inxt=state.model.encode({"pixels":nxt,"action":act})
                    surprise=(pred-inxt["emb"]).pow(2).mean().item()

            region_surp[best_region].append(surprise)
            region["surprise_ema"]=0.9*region.get("surprise_ema",0)+0.1*surprise

            thresh=None
            if len(region_surp[best_region])>20:
                vals=list(region_surp[best_region])
                thresh=np.mean(vals)+2*np.std(vals)
            is_anom=thresh is not None and surprise>thresh
            if is_anom:region["alert_count"]=region.get("alert_count",0)+1

            # Draw
            display=cv2.resize(frame,(480,270))
            cv2.putText(display,f"Region: {best_region}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
            if is_anom:
                cv2.rectangle(display,(0,0),(479,269),(0,0,255),4)
                cv2.putText(display,"ANOMALY",(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            cv2.putText(display,f"S:{surprise:.5f}",(10,265),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)

            _,b=cv2.imencode('.jpg',display,[cv2.IMWRITE_JPEG_QUALITY,70])

            # Region status
            region_status={}
            for n,r in state.regions.items():
                elapsed=time.time()-r.get("last_observed",time.time())
                ig=(elapsed)*r.get("surprise_ema",0.001)
                region_status[n]={
                    "surprise_ema":round(r.get("surprise_ema",0),6),
                    "last_seen_ago":round(elapsed,1),
                    "info_gain":round(ig,3),
                    "alerts":r.get("alert_count",0),
                    "active":n==best_region,
                }

            await websocket.send_json({
                "frame":base64.b64encode(b).decode(),
                "region":best_region,
                "surprise":round(surprise,6),
                "is_anomaly":is_anom,
                "regions":region_status,
                "auto_patrol":state.auto_patrol,
                "surprise_history":list(region_surp[best_region])[-100:],
            })

            try:
                cmd=await asyncio.wait_for(websocket.receive_json(),timeout=0.5)
                if cmd.get("action")=="stop":break
                if "switch_region" in cmd:state.active_region=cmd["switch_region"]
                if "auto_patrol" in cmd:state.auto_patrol=cmd["auto_patrol"]
            except:pass

            await asyncio.sleep(0.2)

    except WebSocketDisconnect:pass

FRONTEND="""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Demo 4 - Smart PTZ</title>
<style>
:root{--bg:#0f172a;--card:#1e293b;--accent:#3b82f6;--green:#22c55e;--red:#ef4444;--orange:#f59e0b;--text:#e2e8f0;--dim:#64748b;--border:#334155}
*{margin:0;padding:0;box-sizing:border-box}body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
.header{padding:1rem 2rem;display:flex;align-items:center;gap:1rem;border-bottom:1px solid var(--border)}.header h1{font-size:1.3rem}
.tag{background:var(--orange);padding:2px 10px;border-radius:999px;font-size:.75rem;font-weight:600;color:#000}
.tabs{display:flex;gap:0;margin-left:auto}.tab{padding:6px 16px;cursor:pointer;border:1px solid var(--border);font-size:.85rem;color:var(--dim);background:transparent}
.tab:first-child{border-radius:6px 0 0 6px}.tab:last-child{border-radius:0 6px 6px 0}.tab.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.page{display:none;padding:1.5rem 2rem;max-width:1100px;margin:0 auto}.page.active{display:block}
.card{background:var(--card);border-radius:10px;padding:1.5rem;margin-bottom:1rem}.card h2{font-size:1.1rem;margin-bottom:1rem}
.form-row{display:flex;gap:1rem;margin-bottom:.8rem;align-items:flex-end;flex-wrap:wrap}
.form-group{flex:1;min-width:120px}.form-group label{display:block;font-size:.8rem;color:var(--dim);margin-bottom:4px}
.form-group input,.form-group select{width:100%;padding:8px 12px;border:1px solid var(--border);border-radius:6px;background:var(--bg);color:var(--text);font-size:.9rem}
.btn{padding:8px 20px;border:none;border-radius:6px;font-size:.9rem;font-weight:600;cursor:pointer}
.btn-green{background:var(--green);color:#000}.btn-red{background:var(--red);color:#fff}.btn-blue{background:var(--accent);color:#fff}.btn-orange{background:var(--orange);color:#000}
.main{display:grid;grid-template-columns:1fr 300px;height:calc(100vh - 56px)}@media(max-width:900px){.main{grid-template-columns:1fr}}
.video-area{padding:1rem;display:flex;flex-direction:column;gap:.5rem}
.video-box{flex:1;background:#000;border-radius:8px;overflow:hidden;display:flex;align-items:center;justify-content:center}.video-box img{max-width:100%;max-height:100%}
.chart-box{height:80px;background:var(--card);border-radius:8px;padding:.5rem}.chart-box canvas{width:100%;height:100%}
.sidebar{background:var(--card);padding:1rem;overflow-y:auto;border-left:1px solid var(--border)}
.sidebar h3{font-size:.85rem;color:var(--dim);text-transform:uppercase;margin:1rem 0 .5rem}
.region-card{background:var(--bg);border-radius:8px;padding:.8rem;margin-bottom:.5rem;cursor:pointer;border:2px solid transparent;transition:border .2s}
.region-card.active{border-color:var(--accent)}.region-card.alert{border-color:var(--red);animation:pulse .5s infinite alternate}
.region-name{font-weight:600;font-size:.9rem}.region-meta{font-size:.75rem;color:var(--dim);margin-top:2px}
.region-bar{height:4px;border-radius:2px;margin-top:4px;background:var(--border)}.region-fill{height:100%;border-radius:2px}
.toggle{display:flex;align-items:center;gap:.5rem;margin:.5rem 0}.toggle input{width:40px;height:20px}
@keyframes pulse{from{opacity:1}to{opacity:.6}}
.lang-btn{position:fixed;top:12px;right:16px;z-index:999;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.3);color:#fff;padding:4px 14px;border-radius:999px;cursor:pointer;font-size:.8rem;font-weight:600;backdrop-filter:blur(8px)}.lang-btn:hover{background:rgba(255,255,255,.25)}.about-link{position:fixed;top:12px;right:70px;z-index:999;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);color:#94a3b8;padding:4px 14px;border-radius:999px;cursor:pointer;font-size:.8rem;text-decoration:none}.about-link:hover{color:#fff;background:rgba(255,255,255,.2)}.lang-zh{display:block}.lang-en{display:none}body.en .lang-zh{display:none}body.en .lang-en{display:block}</style>
</head>
<body>
<div class="header"><h1>Demo 4: Smart PTZ Patrol</h1><span class="tag">AUTO</span>
<div class="tabs">
<button class="tab" onclick="showPage('setup',this)">Setup</button>
<button class="tab active" onclick="showPage('patrol',this)">Patrol</button>
</div></div>

<div class="page" id="page-setup">
<div class="card"><h2>Configure Regions</h2>
<p style="color:var(--dim);font-size:.85rem;margin-bottom:1rem">Define camera regions. Each region can be a different RTSP URL or preset position.<br>For a single PTZ camera, use the same URL for all regions (switching will be simulated).</p>
<div id="regionInputs">
<div class="form-row"><div class="form-group"><label>Region A</label><input id="rA" value="rtsp://admin:Ms!23456@116.59.11.189:554/sub" /></div></div>
<div class="form-row"><div class="form-group"><label>Region B</label><input id="rB" value="rtsp://admin:Ms!23456@116.59.11.189:554/sub" /></div></div>
<div class="form-row"><div class="form-group"><label>Region C</label><input id="rC" value="rtsp://admin:Ms!23456@116.59.11.189:554/sub" /></div></div>
</div>
<button class="btn btn-blue" onclick="saveRegions()">Save Regions</button>
</div>
<div class="card"><h2>Model</h2>
<div class="form-row"><div class="form-group"><label>Checkpoint</label><select id="sMdl"></select></div>
<button class="btn btn-blue" onclick="loadMdl()">Load</button></div>
<div id="mdlStatus" style="font-size:.85rem;color:var(--dim)"></div>
</div>
</div>

<div class="page active" id="page-patrol">
<div class="main">
<div class="video-area">
<div class="video-box"><span id="ph" style="color:var(--dim)">Configure regions in Setup, then start patrol</span><img id="vFrame" style="display:none" /></div>
<div class="chart-box"><canvas id="sChart"></canvas></div>
</div>
<div class="sidebar">
<button class="btn btn-green" id="bStart" onclick="startPatrol()" style="width:100%">Start Patrol</button>
<button class="btn btn-red" id="bStop" onclick="stopPatrol()" style="display:none;width:100%">Stop</button>
<div class="toggle" style="margin-top:.5rem"><label style="font-size:.85rem"><input type="checkbox" id="autoChk" onchange="toggleAuto()" checked /> Auto Patrol</label></div>
<h3>Regions</h3>
<div id="regionList"><span style="color:var(--dim)">No regions</span></div>
<h3>Active Region</h3>
<div style="font-size:1.2rem;font-weight:700" id="activeRegion">-</div>
</div>
</div>
</div>

<script>
let ws=null;
function showPage(n,btn){document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));document.getElementById('page-'+n).classList.add('active');document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));btn.classList.add('active');refresh();}
async function refresh(){try{const r=await fetch('/api/status');const d=await r.json();document.getElementById('sMdl').innerHTML=d.checkpoints.map(x=>'<option value="'+x.path+'">'+x.name+'</option>').join('');document.getElementById('mdlStatus').textContent=d.model_loaded?'Model loaded':'No model';}catch(e){}}
async function saveRegions(){
  const regions={};
  ['A','B','C'].forEach(n=>{const v=document.getElementById('r'+n).value;if(v)regions['Region '+n]={rtsp:v};});
  await fetch('/api/regions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({regions})});
  alert('Regions saved!');
}
async function loadMdl(){const r=await fetch('/api/load_model',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:document.getElementById('sMdl').value})});const d=await r.json();document.getElementById('mdlStatus').textContent=d.loaded?'Loaded!':'Failed';}
function toggleAuto(){if(ws&&ws.readyState===1)ws.send(JSON.stringify({auto_patrol:document.getElementById('autoChk').checked}));}
function startPatrol(){
  const p=location.protocol==='https:'?'wss:':'ws:';ws=new WebSocket(p+'//'+location.host+'/ws/patrol');
  ws.onopen=()=>{ws.send(JSON.stringify({model_path:document.getElementById('sMdl').value}));};
  ws.onmessage=(e)=>{const d=JSON.parse(e.data);
  if(d.error){alert(d.error);return;}if(d.status==='started'){document.getElementById('ph').style.display='none';document.getElementById('vFrame').style.display='block';return;}
  if(d.frame)document.getElementById('vFrame').src='data:image/jpeg;base64,'+d.frame;
  document.getElementById('activeRegion').textContent=d.region||'-';
  // Regions
  if(d.regions){const rl=document.getElementById('regionList');rl.innerHTML=Object.entries(d.regions).map(([n,r])=>{
    const cls=r.active?'active':'';const acls=r.alerts>0?'alert':'';
    const ig=r.info_gain;const maxIg=5;const pct=Math.min(ig/maxIg*100,100);
    const color=r.active?'var(--accent)':(ig>3?'var(--red)':'var(--green)');
    return '<div class="region-card '+cls+' '+acls+'" onclick="switchRegion(\\''+n+'\\')">'+
      '<div class="region-name">'+n+'</div>'+
      '<div class="region-meta">Surprise: '+r.surprise_ema+' | Last: '+r.last_seen_ago+'s ago | Alerts: '+r.alerts+'</div>'+
      '<div class="region-bar"><div class="region-fill" style="width:'+pct+'%;background:'+color+'"></div></div></div>';
  }).join('');}
  if(d.surprise_history)drawChart(d.surprise_history);};
  document.getElementById('bStart').style.display='none';document.getElementById('bStop').style.display='block';
}
function stopPatrol(){if(ws){ws.send(JSON.stringify({action:'stop'}));ws.close();}document.getElementById('bStart').style.display='block';document.getElementById('bStop').style.display='none';}
function switchRegion(name){if(ws&&ws.readyState===1)ws.send(JSON.stringify({switch_region:name,auto_patrol:false}));document.getElementById('autoChk').checked=false;}
function drawChart(h){const c=document.getElementById('sChart');const ctx=c.getContext('2d');const W=c.width=c.offsetWidth*2;const H=c.height=c.offsetHeight*2;ctx.clearRect(0,0,W,H);const mx=Math.max(...h)*1.3||1;const p=8;ctx.beginPath();ctx.lineWidth=2;ctx.strokeStyle='#3b82f6';h.forEach((v,i)=>{const x=p+(i/(h.length-1||1))*(W-2*p);const y=p+(H-2*p)-(v/mx)*(H-2*p);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});ctx.stroke();}
refresh();setInterval(refresh,3000);
function toggleLang(){document.body.classList.toggle('en');var b=document.getElementById('langToggle');b.textContent=document.body.classList.contains('en')?'中文':'EN';localStorage.setItem('lewm_lang',document.body.classList.contains('en')?'en':'zh')}if(localStorage.getItem('lewm_lang')==='en'){document.body.classList.add('en');document.addEventListener('DOMContentLoaded',function(){var b=document.getElementById('langToggle');if(b)b.textContent='中文';})}
</script>
</body>
</html>
"""

if __name__=="__main__":
    print(f"\\n{'='*50}\\n  Demo 4: Smart PTZ Patrol\\n  http://localhost:8772\\n  Device: {state.device}\\n{'='*50}\\n")
    uvicorn.run(app,host="0.0.0.0",port=8772)
