"""
Demo 3 - YOLO + LeWM 雙重視野
左側 YOLO 物件偵測，右側 LeWM 物理理解，展示互補效果

啟動: python web_app.py → http://localhost:8771
"""
import sys, os, asyncio, time, base64, json, threading
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
from module import MLP, Embedder, ARPredictor, SIGReg
from jepa import JEPA

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CKPT_DIR = BASE_DIR / "checkpoints"

# Also check demo2 checkpoints
DEMO2_CKPT = BASE_DIR.parent / "demo2-voe-anomaly" / "checkpoints"

class AppState:
    device = None; model = None; recording = False; training = False
    record_progress = {}; train_progress = {}
state = AppState()
state.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def build_model(device):
    encoder = spt.backbone.utils.vit_hf("tiny", patch_size=16, image_size=64, pretrained=False, use_mask_token=False)
    hd = encoder.config.hidden_size; ed = 192
    predictor = ARPredictor(num_frames=4, input_dim=ed, hidden_dim=hd, output_dim=hd, depth=2, heads=4, mlp_dim=hd*4, dropout=0.0)
    action_encoder = Embedder(input_dim=2, emb_dim=ed)
    projector = MLP(input_dim=hd, output_dim=ed, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hd, output_dim=ed, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    return JEPA(encoder=encoder, predictor=predictor, action_encoder=action_encoder, projector=projector, pred_proj=pred_proj).to(device)

def load_model(path):
    model = build_model(state.device)
    saved = torch.load(path, map_location=state.device, weights_only=False)
    if isinstance(saved, dict) and "model" in saved:
        model.load_state_dict(saved["model"])
    else:
        model.load_state_dict(saved)
    model.eval()
    state.model = model
    return True

# Recording + Training reuse Demo 2's logic
sys.path.insert(0, str(BASE_DIR.parent / "demo2-voe-anomaly" / "src"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def index(): return HTMLResponse(FRONTEND)

@app.get("/api/status")
async def api_status():
    # Gather checkpoints from demo2 and demo3
    checkpoints = []
    for ckpt_base in [CKPT_DIR, DEMO2_CKPT]:
        if not ckpt_base.exists(): continue
        for d in sorted(ckpt_base.iterdir()):
            if d.is_dir() and (d / "best_model.pt").exists():
                checkpoints.append({"name": d.name, "path": str(d / "best_model.pt")})
    datasets = []
    for dd in [DATA_DIR, BASE_DIR.parent / "demo2-voe-anomaly" / "data"]:
        if not dd.exists(): continue
        for f in sorted(dd.glob("*.h5")):
            with h5py.File(f, "r") as h:
                datasets.append({"name": f.stem, "file": str(f), "episodes": int(h.attrs.get("num_episodes", 0))})
    return {"device": state.device, "model_loaded": state.model is not None,
            "checkpoints": checkpoints, "datasets": datasets,
            "recording": state.recording, "training": state.training,
            "record_progress": state.record_progress, "train_progress": state.train_progress}

@app.post("/api/load_model")
async def api_load(body: dict):
    ok = load_model(body["path"])
    return {"loaded": ok}

@app.post("/api/preview")
async def preview(body: dict):
    cap = cv2.VideoCapture(body.get("rtsp",""), cv2.CAP_FFMPEG)
    if not cap.isOpened(): return JSONResponse({"error":"fail"},400)
    ret, frame = cap.read(); cap.release()
    if not ret: return JSONResponse({"error":"no frame"},400)
    d = cv2.resize(frame,(640,360))
    _,buf = cv2.imencode('.jpg',d,[cv2.IMWRITE_JPEG_QUALITY,80])
    return {"frame": base64.b64encode(buf).decode()}

@app.post("/api/record")
async def api_record(body: dict):
    if state.recording: return JSONResponse({"error":"busy"},400)
    from record_camera import record_rtsp
    def worker():
        state.recording = True
        state.record_progress = {"status":"recording"}
        record_rtsp(body.get("rtsp",""), body.get("duration",300), body.get("fps",5), 64,
                    str(DATA_DIR / f"{body.get('name','dual_'+datetime.now().strftime('%H%M%S'))}.h5"))
        state.record_progress["status"] = "done"
        state.recording = False
    threading.Thread(target=worker, daemon=True).start()
    return {"status":"started"}

@app.post("/api/train")
async def api_train(body: dict):
    if state.training: return JSONResponse({"error":"busy"},400)
    from train_voe import train as train_voe
    def worker():
        state.training = True
        state.train_progress = {"status":"training"}
        train_voe(body["data_path"], str(CKPT_DIR / body.get("name","dual_model")),
                  body.get("epochs",30), body.get("batch_size",32), body.get("lr",3e-4), state.device)
        state.train_progress["status"] = "done"
        state.training = False
    threading.Thread(target=worker, daemon=True).start()
    return {"status":"started"}

@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        rtsp = data.get("rtsp","")
        if data.get("model_path") and state.model is None:
            load_model(data["model_path"])
        if state.model is None:
            await websocket.send_json({"error":"No model loaded"}); return

        cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            await websocket.send_json({"error":"Cannot connect"}); return

        try:
            from ultralytics import YOLO
            yolo = YOLO("yolo11n.pt")
        except:
            yolo = None

        await websocket.send_json({"status":"connected"})
        buf = deque(maxlen=5); surp_hist = deque(maxlen=300); skip = 0

        while True:
            ret, frame = cap.read()
            if not ret: break
            skip += 1
            if skip % 5 != 0: continue

            oh, ow = frame.shape[:2]
            # === YOLO side ===
            yolo_results = []
            yolo_frame = cv2.resize(frame, (320, 180))
            if yolo:
                results = yolo(frame, verbose=False, conf=0.4)
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        name = r.names[cls]
                        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        yolo_results.append({"name":name,"conf":round(conf,2),
                                            "x1":x1/ow,"y1":y1/oh,"x2":x2/ow,"y2":y2/oh})
                        # Draw on yolo_frame
                        sx1,sy1,sx2,sy2 = int(x1/ow*320),int(y1/oh*180),int(x2/ow*320),int(y2/oh*180)
                        cv2.rectangle(yolo_frame,(sx1,sy1),(sx2,sy2),(0,255,0),1)
                        cv2.putText(yolo_frame,f"{name} {conf:.0%}",(sx1,sy1-4),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,255,0),1)

            # === LeWM side ===
            resized = cv2.resize(frame,(64,64))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).float().permute(2,0,1)/255.0
            buf.append(tensor)

            surprise = 0; lewm_frame = cv2.resize(frame,(320,180))
            if len(buf) >= 5:
                frames = list(buf)
                ctx = torch.stack(frames[:4]).unsqueeze(0).to(state.device)
                nxt = torch.stack(frames[1:5]).unsqueeze(0).to(state.device)
                act = torch.zeros(1,4,2,device=state.device)
                with torch.no_grad():
                    ic = state.model.encode({"pixels":ctx,"action":act})
                    pred = state.model.predict(ic["emb"],ic["act_emb"])
                    inxt = state.model.encode({"pixels":nxt,"action":act})
                    surprise = (pred-inxt["emb"]).pow(2).mean().item()

            surp_hist.append(surprise)
            thresh = None
            if len(surp_hist) > 30:
                vals = list(surp_hist)
                thresh = np.mean(vals) + 2*np.std(vals)
            is_anom = thresh is not None and surprise > thresh

            # Draw on lewm_frame
            if is_anom:
                cv2.rectangle(lewm_frame,(0,0),(319,179),(0,0,255),3)
                cv2.putText(lewm_frame,"ANOMALY",(5,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            bar_h = int(min(surprise*5000,170))
            cv2.rectangle(lewm_frame,(300,180-bar_h),(315,180),(0,0,255) if is_anom else (0,200,0),-1)
            cv2.putText(lewm_frame,f"S:{surprise:.5f}",(5,175),cv2.FONT_HERSHEY_SIMPLEX,0.35,(200,200,200),1)

            # Combine side by side
            combined = np.zeros((200,660,3),dtype=np.uint8)
            combined[10:190,10:330] = yolo_frame
            combined[10:190,330:650] = lewm_frame
            cv2.putText(combined,"YOLO Detection",(80,9),cv2.FONT_HERSHEY_SIMPLEX,0.35,(150,150,150),1)
            cv2.putText(combined,"LeWM Physics",(420,9),cv2.FONT_HERSHEY_SIMPLEX,0.35,(150,150,150),1)
            cv2.line(combined,(330,0),(330,200),(80,80,80),1)

            _,b = cv2.imencode('.jpg',combined,[cv2.IMWRITE_JPEG_QUALITY,75])

            await websocket.send_json({
                "frame": base64.b64encode(b).decode(),
                "yolo_objects": yolo_results,
                "surprise": round(surprise,6),
                "is_anomaly": is_anom,
                "threshold": round(thresh,6) if thresh else None,
                "surprise_history": list(surp_hist)[-200:],
                "yolo_count": len(yolo_results),
                "persons": sum(1 for y in yolo_results if y["name"]=="person"),
            })

            try:
                cmd = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                if cmd.get("action")=="stop": break
            except: pass

        cap.release()
    except WebSocketDisconnect: pass

FRONTEND = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Demo 3 - Dual Vision</title>
<style>
:root{--bg:#0f172a;--card:#1e293b;--accent:#3b82f6;--green:#22c55e;--red:#ef4444;--text:#e2e8f0;--dim:#64748b;--border:#334155}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
.header{padding:1rem 2rem;display:flex;align-items:center;gap:1rem;border-bottom:1px solid var(--border)}
.header h1{font-size:1.3rem}
.tag{background:#8b5cf6;padding:2px 10px;border-radius:999px;font-size:.75rem;font-weight:600;color:#fff}
.tabs{display:flex;gap:0;margin-left:auto}
.tab{padding:6px 16px;cursor:pointer;border:1px solid var(--border);font-size:.85rem;color:var(--dim);background:transparent}
.tab:first-child{border-radius:6px 0 0 6px}.tab:last-child{border-radius:0 6px 6px 0}
.tab.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.page{display:none;padding:1.5rem 2rem;max-width:1100px;margin:0 auto}.page.active{display:block}
.card{background:var(--card);border-radius:10px;padding:1.5rem;margin-bottom:1rem}
.card h2{font-size:1.1rem;margin-bottom:1rem}
.form-row{display:flex;gap:1rem;margin-bottom:.8rem;align-items:flex-end;flex-wrap:wrap}
.form-group{flex:1;min-width:120px}
.form-group label{display:block;font-size:.8rem;color:var(--dim);margin-bottom:4px}
.form-group input,.form-group select{width:100%;padding:8px 12px;border:1px solid var(--border);border-radius:6px;background:var(--bg);color:var(--text);font-size:.9rem}
.btn{padding:8px 20px;border:none;border-radius:6px;font-size:.9rem;font-weight:600;cursor:pointer}
.btn-green{background:var(--green);color:#000}.btn-red{background:var(--red);color:#fff}.btn-blue{background:var(--accent);color:#fff}
.video-box{background:#000;border-radius:8px;overflow:hidden;text-align:center;margin:.5rem 0}
.video-box img{max-width:100%}
.chart-box{height:100px;background:var(--card);border-radius:8px;padding:.5rem}.chart-box canvas{width:100%;height:100%}
.stat-row{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--border);font-size:.85rem}
.v{font-weight:600;font-family:monospace}.v.green{color:var(--green)}.v.red{color:var(--red)}
.obj-list{max-height:150px;overflow-y:auto;font-size:.8rem}
.obj-item{padding:3px 0;border-bottom:1px solid var(--border);display:flex;justify-content:space-between}
</style>
</head>
<body>
<div class="header">
  <h1>Demo 3: Dual Vision</h1><span class="tag">YOLO + LeWM</span>
  <div class="tabs">
    <button class="tab" onclick="showPage('setup',this)">Setup</button>
    <button class="tab active" onclick="showPage('detect',this)">Detect</button>
  </div>
</div>

<div class="page" id="page-setup">
  <div class="card"><h2>Setup</h2>
    <p style="color:var(--dim);font-size:.85rem;margin-bottom:1rem">This demo uses existing models from Demo 2. Record & train there first, then load model here.</p>
    <div id="mdlList">Loading...</div>
  </div>
</div>

<div class="page active" id="page-detect">
  <div class="card">
    <div class="form-row">
      <div class="form-group" style="flex:3"><label>RTSP</label><input id="dRtsp" value="rtsp://admin:Ms!23456@116.59.11.189:554/sub" /></div>
      <div class="form-group"><label>Model</label><select id="dModel"></select></div>
      <button class="btn btn-green" id="bDet" onclick="startDet()">Start</button>
      <button class="btn btn-red" id="bStop" onclick="stopDet()" style="display:none">Stop</button>
    </div>
  </div>

  <div class="video-box"><span id="ph" style="color:var(--dim);padding:2rem;display:inline-block">Click Start to begin dual detection</span><img id="vFrame" style="display:none" /></div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:.5rem">
    <div class="card">
      <h2 style="font-size:.95rem">YOLO Objects</h2>
      <div class="stat-row"><span>Total objects</span><span class="v" id="sYolo">-</span></div>
      <div class="stat-row"><span>Persons</span><span class="v" id="sPersons">-</span></div>
      <div class="obj-list" id="objList"></div>
    </div>
    <div class="card">
      <h2 style="font-size:.95rem">LeWM Physics</h2>
      <div class="stat-row"><span>Surprise</span><span class="v" id="sSurp">-</span></div>
      <div class="stat-row"><span>Threshold</span><span class="v" id="sThresh">-</span></div>
      <div class="stat-row"><span>Anomaly</span><span class="v" id="sAnom">-</span></div>
    </div>
  </div>
  <div class="chart-box"><canvas id="sChart"></canvas></div>
</div>

<script>
let ws=null;
function showPage(n,btn){document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));document.getElementById('page-'+n).classList.add('active');document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));btn.classList.add('active');refresh();}
async function refresh(){try{const r=await fetch('/api/status');const d=await r.json();const s=document.getElementById('dModel');s.innerHTML=d.checkpoints.map(x=>'<option value="'+x.path+'">'+x.name+'</option>').join('');document.getElementById('mdlList').innerHTML=d.checkpoints.length?d.checkpoints.map(x=>'<div class="obj-item"><span>'+x.name+'</span></div>').join(''):'<span style="color:var(--dim)">No models. Train in Demo 2 first.</span>';}catch(e){}}
function startDet(){
  const p=location.protocol==='https:'?'wss:':'ws:';ws=new WebSocket(p+'//'+location.host+'/ws/detect');
  ws.onopen=()=>{ws.send(JSON.stringify({rtsp:document.getElementById('dRtsp').value,model_path:document.getElementById('dModel').value}));};
  ws.onmessage=(e)=>{const d=JSON.parse(e.data);
  if(d.error){alert(d.error);return;}if(d.status==='connected'){document.getElementById('ph').style.display='none';document.getElementById('vFrame').style.display='block';return;}
  if(d.frame)document.getElementById('vFrame').src='data:image/jpeg;base64,'+d.frame;
  document.getElementById('sYolo').textContent=d.yolo_count||0;
  document.getElementById('sPersons').textContent=d.persons||0;
  document.getElementById('sSurp').textContent=(d.surprise||0).toFixed(5);
  document.getElementById('sThresh').textContent=d.threshold?(d.threshold).toFixed(5):'-';
  const a=document.getElementById('sAnom');a.textContent=d.is_anomaly?'YES':'No';a.className='v '+(d.is_anomaly?'red':'green');
  document.getElementById('objList').innerHTML=(d.yolo_objects||[]).map(o=>'<div class="obj-item"><span>'+o.name+'</span><span>'+Math.round(o.conf*100)+'%</span></div>').join('');
  if(d.surprise_history)drawChart(d.surprise_history);};
  document.getElementById('bDet').style.display='none';document.getElementById('bStop').style.display='inline-block';
}
function stopDet(){if(ws){ws.send(JSON.stringify({action:'stop'}));ws.close();}document.getElementById('bDet').style.display='inline-block';document.getElementById('bStop').style.display='none';}
function drawChart(h){const c=document.getElementById('sChart');const ctx=c.getContext('2d');const W=c.width=c.offsetWidth*2;const H=c.height=c.offsetHeight*2;ctx.clearRect(0,0,W,H);const mx=Math.max(...h)*1.3||1;const p=10;ctx.beginPath();ctx.lineWidth=2;ctx.strokeStyle='#3b82f6';h.forEach((v,i)=>{const x=p+(i/(h.length-1||1))*(W-2*p);const y=p+(H-2*p)-(v/mx)*(H-2*p);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});ctx.stroke();}
refresh();setInterval(refresh,3000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"\\n{'='*50}\\n  Demo 3: Dual Vision\\n  http://localhost:8771\\n  Device: {state.device}\\n{'='*50}\\n")
    uvicorn.run(app, host="0.0.0.0", port=8771)
