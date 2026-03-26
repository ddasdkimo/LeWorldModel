"""
Demo 1 - 軌跡預測預警：Web 介面
三階段一站式：錄製 → 訓練 → 即時偵測

啟動: python web_app.py → http://localhost:8770
"""
import sys
import os
import asyncio
import time
import base64
import json
import threading
from collections import deque
from pathlib import Path
from datetime import datetime

for p in ["/home/rai/code/le-wm", os.path.expanduser("~/code/2026/le-wm-local")]:
    if os.path.exists(p):
        sys.path.insert(0, p)
        break

import cv2
import numpy as np
import h5py
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import stable_pretraining as spt
from module import MLP, Embedder, ARPredictor, SIGReg
from jepa import JEPA

# Import local modules
sys.path.insert(0, str(Path(__file__).parent))
from web_record import record_with_yolo
from train_trajectory import build_model, train, PositionProbe, TrajectoryDataset

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CKPT_DIR = BASE_DIR / "checkpoints"
DATA_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)


# ============================================================
# State
# ============================================================

class AppState:
    device = None
    recording = False
    training = False
    detecting = False
    record_progress = {}
    train_progress = {}
    danger_zones = []  # list of {"x1","y1","x2","y2"} normalized 0-1
    model = None
    probe = None

state = AppState()
state.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Default danger zone (bottom-right area)
state.danger_zones = [{"x1": 0.6, "y1": 0.7, "x2": 1.0, "y2": 1.0, "label": "Danger Zone"}]


def load_checkpoint(ckpt_dir):
    ckpt = Path(ckpt_dir) / "best_model.pt"
    if not ckpt.exists():
        return False
    model = build_model(state.device)
    saved = torch.load(ckpt, map_location=state.device, weights_only=False)
    if isinstance(saved, dict) and "model" in saved:
        model.load_state_dict(saved["model"])
        probe = PositionProbe(192).to(state.device)
        if "probe" in saved:
            probe.load_state_dict(saved["probe"])
        state.probe = probe
    else:
        model.load_state_dict(saved)
        state.probe = None
    model.eval()
    state.model = model
    return True


# ============================================================
# App
# ============================================================

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])



@app.get("/about")
async def about_page():
    about_file = Path(__file__).parent.parent / "about.html"
    if about_file.exists():
        return HTMLResponse(about_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>About page not found</h1>")

@app.get("/")
async def index():
    return HTMLResponse(FRONTEND)


@app.get("/api/status")
async def api_status():
    datasets = []
    for f in sorted(DATA_DIR.glob("*.h5")):
        with h5py.File(f, "r") as h:
            datasets.append({
                "name": f.stem, "file": str(f),
                "episodes": int(h.attrs.get("num_episodes", 0)),
                "has_yolo": bool(h.attrs.get("has_yolo", False)),
                "size_mb": round(f.stat().st_size / 1e6, 1),
            })
    checkpoints = []
    for d in sorted(CKPT_DIR.iterdir()):
        if d.is_dir() and (d / "best_model.pt").exists():
            hist_file = d / "train_history.json"
            info = {"name": d.name, "path": str(d), "epochs": 0, "pred_loss": None}
            if hist_file.exists():
                with open(hist_file) as f:
                    h = json.load(f)
                    if h:
                        info["epochs"] = len(h)
                        info["pred_loss"] = h[-1].get("pred")
            checkpoints.append(info)
    return {
        "device": state.device,
        "recording": state.recording,
        "training": state.training,
        "model_loaded": state.model is not None,
        "has_probe": state.probe is not None,
        "datasets": datasets,
        "checkpoints": checkpoints,
        "danger_zones": state.danger_zones,
        "record_progress": state.record_progress,
        "train_progress": state.train_progress,
    }


@app.post("/api/preview")
async def preview(body: dict):
    cap = cv2.VideoCapture(body.get("rtsp", ""), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return JSONResponse({"error": "Cannot connect"}, 400)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return JSONResponse({"error": "No frame"}, 400)
    display = cv2.resize(frame, (640, 360))
    _, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return {"frame": base64.b64encode(buf).decode()}


@app.post("/api/record")
async def api_record(body: dict):
    if state.recording:
        return JSONResponse({"error": "Already recording"}, 400)
    def worker():
        state.recording = True
        state.record_progress = {"status": "recording"}
        result = record_with_yolo(
            rtsp_url=body.get("rtsp", ""),
            duration=body.get("duration", 300),
            fps=body.get("fps", 5),
            output_path=str(DATA_DIR / f"{body.get('name', 'traj_' + datetime.now().strftime('%H%M%S'))}.h5"),
            progress_callback=lambda p: state.record_progress.update(p),
        )
        state.record_progress["status"] = "done" if result else "error"
        if result:
            state.record_progress.update(result)
        state.recording = False
    threading.Thread(target=worker, daemon=True).start()
    return {"status": "started"}


@app.post("/api/record/stop")
async def stop_record():
    state.recording = False
    return {"status": "stopping"}


@app.post("/api/train")
async def api_train(body: dict):
    if state.training:
        return JSONResponse({"error": "Already training"}, 400)
    def worker():
        state.training = True
        state.train_progress = {"status": "training"}
        train(
            data_path=body.get("data_path", ""),
            output_dir=str(CKPT_DIR / body.get("name", f"traj_{datetime.now().strftime('%H%M%S')}")),
            epochs=body.get("epochs", 30),
            batch_size=body.get("batch_size", 32),
            lr=body.get("lr", 3e-4),
            device=state.device,
            progress_callback=lambda p: state.train_progress.update(p),
        )
        state.train_progress["status"] = "done"
        state.training = False
    threading.Thread(target=worker, daemon=True).start()
    return {"status": "started"}


@app.post("/api/train/stop")
async def stop_train():
    state.training = False
    return {"status": "stopping"}


@app.post("/api/load_model")
async def api_load(body: dict):
    ok = load_checkpoint(body.get("path", ""))
    return {"loaded": ok, "has_probe": state.probe is not None}


@app.post("/api/danger_zones")
async def set_zones(body: dict):
    state.danger_zones = body.get("zones", [])
    return {"zones": state.danger_zones}


@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        rtsp = data.get("rtsp", "")
        ckpt_path = data.get("checkpoint", "")

        if ckpt_path and (state.model is None):
            load_checkpoint(ckpt_path)

        if state.model is None:
            await websocket.send_json({"error": "No model loaded"})
            return

        cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            await websocket.send_json({"error": "Cannot connect"})
            return

        await websocket.send_json({"status": "connected"})

        # YOLO for person detection
        try:
            from ultralytics import YOLO
            yolo = YOLO("yolo11n.pt")
        except:
            yolo = None

        frame_buffer = deque(maxlen=5)
        surprise_history = deque(maxlen=300)
        trajectory_history = deque(maxlen=60)  # 12 seconds at 5fps
        skip = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            skip += 1
            if skip % 5 != 0:
                continue

            orig_h, orig_w = frame.shape[:2]

            # YOLO detect persons
            persons = []
            if yolo:
                results = yolo(frame, classes=[0], verbose=False, conf=0.5)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx, cy = (x1+x2)/2/orig_w, (y1+y2)/2/orig_h
                        persons.append({"x1": x1/orig_w, "y1": y1/orig_h,
                                       "x2": x2/orig_w, "y2": y2/orig_h,
                                       "cx": cx, "cy": cy,
                                       "conf": float(box.conf[0])})

            # Track first person trajectory
            if persons:
                trajectory_history.append({"x": persons[0]["cx"], "y": persons[0]["cy"], "t": time.time()})

            # LeWM surprise
            resized = cv2.resize(frame, (64, 64))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
            frame_buffer.append(tensor)

            surprise = 0
            predicted_positions = []
            if len(frame_buffer) >= 5:
                frames = list(frame_buffer)
                ctx = torch.stack(frames[:4]).unsqueeze(0).to(state.device)
                nxt = torch.stack(frames[1:5]).unsqueeze(0).to(state.device)
                actions = torch.zeros(1, 4, 2, device=state.device)

                with torch.no_grad():
                    info_ctx = state.model.encode({"pixels": ctx, "action": actions})
                    pred = state.model.predict(info_ctx["emb"], info_ctx["act_emb"])
                    info_nxt = state.model.encode({"pixels": nxt, "action": actions})
                    surprise = (pred - info_nxt["emb"]).pow(2).mean().item()

                    # Position prediction from predicted embeddings
                    if state.probe is not None:
                        pred_pos = state.probe(pred.reshape(-1, 192)).reshape(-1, 2)
                        predicted_positions = pred_pos.cpu().numpy().tolist()

            surprise_history.append(surprise)

            # Check danger zone collision
            in_danger = False
            time_to_danger = None
            for zone in state.danger_zones:
                for p in persons:
                    if (p["cx"] >= zone["x1"] and p["cx"] <= zone["x2"] and
                        p["cy"] >= zone["y1"] and p["cy"] <= zone["y2"]):
                        in_danger = True

                # Predict future collision from trajectory
                if len(trajectory_history) >= 5 and not in_danger:
                    pts = list(trajectory_history)[-5:]
                    dx = (pts[-1]["x"] - pts[0]["x"]) / 4
                    dy = (pts[-1]["y"] - pts[0]["y"]) / 4
                    for step in range(1, 15):  # predict 3 seconds ahead
                        fx = pts[-1]["x"] + dx * step
                        fy = pts[-1]["y"] + dy * step
                        if (fx >= zone["x1"] and fx <= zone["x2"] and
                            fy >= zone["y1"] and fy <= zone["y2"]):
                            time_to_danger = round(step * 0.2, 1)
                            break

            # Draw overlay
            display = cv2.resize(frame, (640, 360))
            dh, dw = 360, 640

            # Draw danger zones
            for zone in state.danger_zones:
                zx1, zy1 = int(zone["x1"]*dw), int(zone["y1"]*dh)
                zx2, zy2 = int(zone["x2"]*dw), int(zone["y2"]*dh)
                overlay = display.copy()
                cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 0, 200), -1)
                display = cv2.addWeighted(overlay, 0.2, display, 0.8, 0)
                cv2.rectangle(display, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
                cv2.putText(display, zone.get("label", "DANGER"), (zx1+4, zy1+16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw persons
            for p in persons:
                px1, py1 = int(p["x1"]*dw), int(p["y1"]*dh)
                px2, py2 = int(p["x2"]*dw), int(p["y2"]*dh)
                color = (0, 0, 255) if in_danger else (0, 255, 0)
                cv2.rectangle(display, (px1, py1), (px2, py2), color, 2)

            # Draw trajectory
            traj = list(trajectory_history)
            for i in range(1, len(traj)):
                pt1 = (int(traj[i-1]["x"]*dw), int(traj[i-1]["y"]*dh))
                pt2 = (int(traj[i]["x"]*dw), int(traj[i]["y"]*dh))
                cv2.line(display, pt1, pt2, (255, 200, 0), 2)

            # Draw predicted trajectory (dashed)
            if traj and len(traj) >= 2:
                dx = (traj[-1]["x"] - traj[-2]["x"])
                dy = (traj[-1]["y"] - traj[-2]["y"])
                for step in range(1, 10):
                    fx = traj[-1]["x"] + dx * step
                    fy = traj[-1]["y"] + dy * step
                    pt = (int(fx * dw), int(fy * dh))
                    cv2.circle(display, pt, 3, (0, 200, 255), -1)

            # Warning text
            if in_danger:
                cv2.putText(display, "IN DANGER ZONE!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            elif time_to_danger:
                cv2.putText(display, f"WARNING: {time_to_danger}s to danger", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                cv2.putText(display, "Safe", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

            _, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf).decode()

            msg = {
                "frame": b64,
                "surprise": round(surprise, 6),
                "persons": len(persons),
                "in_danger": in_danger,
                "time_to_danger": time_to_danger,
                "trajectory": [{"x": t["x"], "y": t["y"]} for t in traj],
                "surprise_history": list(surprise_history)[-200:],
            }
            await websocket.send_json(msg)

            try:
                cmd = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                if cmd.get("action") == "stop":
                    break
            except:
                pass

        cap.release()
    except WebSocketDisconnect:
        pass


# ============================================================
# Frontend
# ============================================================

FRONTEND = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Demo 1 - Trajectory Prediction</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --accent:#3b82f6; --green:#22c55e; --red:#ef4444; --orange:#f59e0b; --text:#e2e8f0; --dim:#64748b; --border:#334155; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
  .header { padding:1rem 2rem; display:flex; align-items:center; gap:1rem; border-bottom:1px solid var(--border); }
  .header h1 { font-size:1.3rem; }
  .tag { background:var(--orange); padding:2px 10px; border-radius:999px; font-size:0.75rem; font-weight:600; color:#000; }
  .tabs { display:flex; gap:0; margin-left:auto; }
  .tab { padding:6px 16px; cursor:pointer; border:1px solid var(--border); font-size:0.85rem; color:var(--dim); background:transparent; }
  .tab:first-child { border-radius:6px 0 0 6px; }
  .tab:last-child { border-radius:0 6px 6px 0; }
  .tab.active { background:var(--accent); color:#fff; border-color:var(--accent); }
  .page { display:none; padding:1.5rem 2rem; max-width:1100px; margin:0 auto; }
  .page.active { display:block; }
  .card { background:var(--card); border-radius:10px; padding:1.5rem; margin-bottom:1rem; }
  .card h2 { font-size:1.1rem; margin-bottom:1rem; }
  .form-row { display:flex; gap:1rem; margin-bottom:0.8rem; align-items:flex-end; flex-wrap:wrap; }
  .form-group { flex:1; min-width:120px; }
  .form-group label { display:block; font-size:0.8rem; color:var(--dim); margin-bottom:4px; }
  .form-group input,.form-group select { width:100%; padding:8px 12px; border:1px solid var(--border); border-radius:6px; background:var(--bg); color:var(--text); font-size:0.9rem; }
  .btn { padding:8px 20px; border:none; border-radius:6px; font-size:0.9rem; font-weight:600; cursor:pointer; }
  .btn-green { background:var(--green); color:#000; }
  .btn-red { background:var(--red); color:#fff; }
  .btn-blue { background:var(--accent); color:#fff; }
  .btn-orange { background:var(--orange); color:#000; }
  .btn:disabled { opacity:0.4; }
  .progress-bar { height:8px; background:var(--bg); border-radius:4px; overflow:hidden; margin:0.5rem 0; }
  .progress-fill { height:100%; border-radius:4px; transition:width 0.3s; }
  .main-detect { display:grid; grid-template-columns:1fr 320px; height:calc(100vh - 56px); }
  @media(max-width:900px) { .main-detect { grid-template-columns:1fr; } }
  .video-area { padding:1rem; display:flex; flex-direction:column; gap:0.5rem; }
  .video-box { flex:1; background:#000; border-radius:8px; overflow:hidden; display:flex; align-items:center; justify-content:center; position:relative; }
  .video-box img { max-width:100%; max-height:100%; }
  .chart-box { height:100px; background:var(--card); border-radius:8px; padding:0.5rem; }
  .chart-box canvas { width:100%; height:100%; }
  .sidebar { background:var(--card); padding:1.5rem; overflow-y:auto; border-left:1px solid var(--border); }
  .sidebar h3 { font-size:0.9rem; color:var(--dim); text-transform:uppercase; margin:1rem 0 0.5rem; }
  .stat-row { display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid var(--border); font-size:0.85rem; }
  .stat-row .v { font-weight:600; font-family:monospace; }
  .v.green { color:var(--green); } .v.red { color:var(--red); } .v.orange { color:var(--orange); }
  .warning-banner { background:var(--red); color:#fff; padding:8px 16px; border-radius:6px; text-align:center; font-weight:700; animation:pulse 0.5s infinite alternate; margin-bottom:0.5rem; }
  .caution-banner { background:var(--orange); color:#000; padding:8px 16px; border-radius:6px; text-align:center; font-weight:700; margin-bottom:0.5rem; }
  @keyframes pulse { from{opacity:1} to{opacity:0.6} }
  .list-item { padding:8px; border-bottom:1px solid var(--border); font-size:0.85rem; display:flex; justify-content:space-between; align-items:center; }
  .log { background:#000; color:#86efac; padding:0.8rem; border-radius:6px; font-family:monospace; font-size:0.8rem; max-height:180px; overflow-y:auto; margin-top:0.5rem; white-space:pre-wrap; }
  .preview-img { max-width:100%; border-radius:8px; margin-top:0.5rem; }
.lang-btn{position:fixed;top:12px;right:16px;z-index:999;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.3);color:#fff;padding:4px 14px;border-radius:999px;cursor:pointer;font-size:.8rem;font-weight:600;backdrop-filter:blur(8px)}.lang-btn:hover{background:rgba(255,255,255,.25)}.about-link{position:fixed;top:12px;right:70px;z-index:999;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);color:#94a3b8;padding:4px 14px;border-radius:999px;cursor:pointer;font-size:.8rem;text-decoration:none}.about-link:hover{color:#fff;background:rgba(255,255,255,.2)}.lang-zh{display:block}.lang-en{display:none}body.en .lang-zh{display:none}body.en .lang-en{display:block}</style>
</head>
<body>

<div class="header">
  <h1>Demo 1: Trajectory Prediction</h1>
  <span class="tag">PREDICTIVE</span>
  <div class="tabs">
    <button class="tab active" onclick="showPage('record',this)">1. Record</button>
    <button class="tab" onclick="showPage('train',this)">2. Train</button>
    <button class="tab" onclick="showPage('detect',this)">3. Detect</button>
  </div>
</div>

<!-- RECORD -->
<div class="page active" id="page-record">
  <div class="card">
    <h2>Record Training Data (with YOLO person detection)</h2>
    <div class="form-row">
      <div class="form-group" style="flex:3"><label>RTSP URL</label>
        <input id="rRtsp" value="rtsp://admin:Ms!23456@116.59.11.189:554/sub" /></div>
      <button class="btn btn-blue" onclick="doPreview()">Preview</button>
    </div>
    <img id="prevImg" class="preview-img" style="display:none" />
    <div class="form-row">
      <div class="form-group"><label>Duration (s)</label><input id="rDur" type="number" value="300" /></div>
      <div class="form-group"><label>FPS</label><input id="rFps" type="number" value="5" /></div>
      <div class="form-group"><label>Name</label><input id="rName" placeholder="auto" /></div>
    </div>
    <button class="btn btn-green" id="bRec" onclick="startRec()">Start Recording</button>
    <button class="btn btn-red" id="bRecS" onclick="stopRec()" style="display:none">Stop</button>
    <div id="recProg" style="display:none;margin-top:0.5rem">
      <div class="progress-bar"><div class="progress-fill" id="recBar" style="width:0%;background:var(--green)"></div></div>
      <div id="recTxt" style="font-size:0.8rem;color:var(--dim)"></div>
    </div>
  </div>
  <div class="card"><h2>Datasets</h2><div id="dsList">Loading...</div></div>
</div>

<!-- TRAIN -->
<div class="page" id="page-train">
  <div class="card">
    <h2>Train Model (LeWM + Position Probe)</h2>
    <div class="form-row">
      <div class="form-group" style="flex:2"><label>Dataset</label><select id="tData"></select></div>
      <div class="form-group"><label>Name</label><input id="tName" placeholder="auto" /></div>
    </div>
    <div class="form-row">
      <div class="form-group"><label>Epochs</label><input id="tEp" type="number" value="30" /></div>
      <div class="form-group"><label>Batch</label><input id="tBs" type="number" value="32" /></div>
      <div class="form-group"><label>LR</label><input id="tLr" type="number" value="0.0003" step="0.0001" /></div>
    </div>
    <button class="btn btn-green" id="bTr" onclick="startTr()">Start Training</button>
    <button class="btn btn-red" id="bTrS" onclick="stopTr()" style="display:none">Stop</button>
  </div>
  <div class="card" id="trCard" style="display:none">
    <h2>Progress</h2>
    <div class="progress-bar"><div class="progress-fill" id="trBar" style="width:0%;background:var(--accent)"></div></div>
    <div style="display:flex;gap:1rem;margin:0.5rem 0;flex-wrap:wrap">
      <span style="font-size:0.85rem">Epoch: <b id="trEp">-</b></span>
      <span style="font-size:0.85rem">Pred Loss: <b id="trPl">-</b></span>
      <span style="font-size:0.85rem">Best: <b id="trBl">-</b></span>
      <span style="font-size:0.85rem">Time: <b id="trTm">-</b></span>
    </div>
    <div class="log" id="trLog"></div>
  </div>
  <div class="card"><h2>Models</h2><div id="mdlList">Loading...</div></div>
</div>

<!-- DETECT -->
<div class="page" id="page-detect">
  <div class="main-detect">
    <div class="video-area">
      <div class="video-box">
        <span id="ph" style="color:var(--dim)">Load model & start detection</span>
        <img id="vFrame" style="display:none" />
      </div>
      <div class="chart-box"><canvas id="sChart"></canvas></div>
    </div>
    <div class="sidebar">
      <div class="form-group"><label>RTSP</label>
        <input id="dRtsp" value="rtsp://admin:Ms!23456@116.59.11.189:554/sub" /></div>
      <div class="form-group" style="margin-top:0.5rem"><label>Model</label><select id="dModel"></select></div>
      <div style="margin-top:0.5rem">
        <button class="btn btn-green" id="bDet" onclick="startDet()">Start Detection</button>
        <button class="btn btn-red" id="bDetS" onclick="stopDet()" style="display:none">Stop</button>
      </div>
      <div id="bannerArea" style="margin-top:0.5rem"></div>
      <h3>Status</h3>
      <div class="stat-row"><span>Persons</span><span class="v" id="sPersons">-</span></div>
      <div class="stat-row"><span>Surprise</span><span class="v" id="sSurp">-</span></div>
      <div class="stat-row"><span>In Danger</span><span class="v" id="sDanger">-</span></div>
      <div class="stat-row"><span>Time to Danger</span><span class="v" id="sTtd">-</span></div>
      <h3>Danger Zone (normalized 0-1)</h3>
      <div class="form-row">
        <div class="form-group"><label>x1</label><input id="zx1" value="0.6" /></div>
        <div class="form-group"><label>y1</label><input id="zy1" value="0.7" /></div>
        <div class="form-group"><label>x2</label><input id="zx2" value="1.0" /></div>
        <div class="form-group"><label>y2</label><input id="zy2" value="1.0" /></div>
      </div>
      <button class="btn btn-orange" onclick="setZone()" style="width:100%">Set Danger Zone</button>
    </div>
  </div>
</div>

<script>
let ws=null;
function showPage(n,btn){document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));document.getElementById('page-'+n).classList.add('active');document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));btn.classList.add('active');refresh();}
async function refresh(){
  try{const r=await fetch('/api/status');const d=await r.json();
  // datasets
  const ds=document.getElementById('dsList');const sel=document.getElementById('tData');const dsel=document.getElementById('dModel');
  ds.innerHTML=d.datasets.length?d.datasets.map(x=>'<div class="list-item"><span>'+x.name+' ('+x.episodes+' ep, '+x.size_mb+'MB)'+(x.has_yolo?' [YOLO]':'')+'</span></div>').join(''):'<span style="color:var(--dim)">No datasets</span>';
  sel.innerHTML=d.datasets.map(x=>'<option value="'+x.file+'">'+x.name+'</option>').join('');
  dsel.innerHTML=d.checkpoints.map(x=>'<option value="'+x.path+'">'+x.name+' ('+x.epochs+'ep)</option>').join('');
  // models
  document.getElementById('mdlList').innerHTML=d.checkpoints.length?d.checkpoints.map(x=>'<div class="list-item"><span>'+x.name+'</span><span style="color:var(--dim)">'+x.epochs+'ep | pred='+((x.pred_loss||0).toFixed(4))+'</span></div>').join(''):'<span style="color:var(--dim)">No models</span>';
  // record progress
  if(d.recording){document.getElementById('recProg').style.display='block';document.getElementById('recBar').style.width=(d.record_progress.percent||0)+'%';document.getElementById('recTxt').textContent=(d.record_progress.frames||0)+' frames | '+(d.record_progress.elapsed||0)+'s';document.getElementById('bRec').style.display='none';document.getElementById('bRecS').style.display='inline-block';}
  // train progress
  const tp=d.train_progress;if(tp&&tp.epoch){document.getElementById('trCard').style.display='block';document.getElementById('trBar').style.width=(tp.percent||0)+'%';document.getElementById('trEp').textContent=tp.epoch+'/'+tp.total_epochs;document.getElementById('trPl').textContent=(tp.pred||0).toFixed(4);document.getElementById('trBl').textContent=(tp.best_loss||0).toFixed(4);document.getElementById('trTm').textContent=(tp.epoch_time||0)+'s';if(tp.history){document.getElementById('trLog').textContent=tp.history.map(h=>'Ep '+String(h.epoch).padStart(3)+' | pred='+h.pred.toFixed(4)+' | '+h.time+'s').join('\\n');}}
  }catch(e){}
}
async function doPreview(){const r=await fetch('/api/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rtsp:document.getElementById('rRtsp').value})});const d=await r.json();if(d.frame){const i=document.getElementById('prevImg');i.src='data:image/jpeg;base64,'+d.frame;i.style.display='block';}}
async function startRec(){await fetch('/api/record',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rtsp:document.getElementById('rRtsp').value,duration:+document.getElementById('rDur').value,fps:+document.getElementById('rFps').value,name:document.getElementById('rName').value||undefined})});document.getElementById('bRec').style.display='none';document.getElementById('bRecS').style.display='inline-block';document.getElementById('recProg').style.display='block';}
async function stopRec(){await fetch('/api/record/stop',{method:'POST'});document.getElementById('bRec').style.display='inline-block';document.getElementById('bRecS').style.display='none';}
async function startTr(){await fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({data_path:document.getElementById('tData').value,name:document.getElementById('tName').value||undefined,epochs:+document.getElementById('tEp').value,batch_size:+document.getElementById('tBs').value,lr:+document.getElementById('tLr').value})});document.getElementById('bTr').style.display='none';document.getElementById('bTrS').style.display='inline-block';document.getElementById('trCard').style.display='block';}
async function stopTr(){await fetch('/api/train/stop',{method:'POST'});document.getElementById('bTr').style.display='inline-block';document.getElementById('bTrS').style.display='none';}
async function setZone(){await fetch('/api/danger_zones',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({zones:[{x1:+document.getElementById('zx1').value,y1:+document.getElementById('zy1').value,x2:+document.getElementById('zx2').value,y2:+document.getElementById('zy2').value,label:'Danger Zone'}]})});}
function startDet(){
  const proto=location.protocol==='https:'?'wss:':'ws:';
  ws=new WebSocket(proto+'//'+location.host+'/ws/detect');
  ws.onopen=()=>{ws.send(JSON.stringify({rtsp:document.getElementById('dRtsp').value,checkpoint:document.getElementById('dModel').value}));};
  ws.onmessage=(e)=>{const d=JSON.parse(e.data);if(d.error){alert(d.error);return;}if(d.status==='connected'){document.getElementById('ph').style.display='none';document.getElementById('vFrame').style.display='block';return;}
  if(d.frame)document.getElementById('vFrame').src='data:image/jpeg;base64,'+d.frame;
  document.getElementById('sPersons').textContent=d.persons||0;
  document.getElementById('sSurp').textContent=(d.surprise||0).toFixed(5);
  const da=document.getElementById('sDanger');da.textContent=d.in_danger?'YES':'No';da.className='v '+(d.in_danger?'red':'green');
  const ttd=document.getElementById('sTtd');ttd.textContent=d.time_to_danger?d.time_to_danger+'s':'-';ttd.className='v '+(d.time_to_danger?'orange':'green');
  const ba=document.getElementById('bannerArea');
  if(d.in_danger)ba.innerHTML='<div class="warning-banner">IN DANGER ZONE!</div>';
  else if(d.time_to_danger)ba.innerHTML='<div class="caution-banner">WARNING: '+d.time_to_danger+'s to danger</div>';
  else ba.innerHTML='';
  if(d.surprise_history)drawChart(d.surprise_history);};
  document.getElementById('bDet').style.display='none';document.getElementById('bDetS').style.display='inline-block';
}
function stopDet(){if(ws){ws.send(JSON.stringify({action:'stop'}));ws.close();}document.getElementById('bDet').style.display='inline-block';document.getElementById('bDetS').style.display='none';}
function drawChart(h){const c=document.getElementById('sChart');const ctx=c.getContext('2d');const W=c.width=c.offsetWidth*2;const H=c.height=c.offsetHeight*2;ctx.clearRect(0,0,W,H);const mx=Math.max(...h)*1.3||1;const p=10;ctx.beginPath();ctx.lineWidth=2;ctx.strokeStyle='#3b82f6';h.forEach((v,i)=>{const x=p+(i/(h.length-1||1))*(W-2*p);const y=p+(H-2*p)-(v/mx)*(H-2*p);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});ctx.stroke();}
setInterval(refresh,2000);refresh();
function toggleLang(){document.body.classList.toggle('en');var b=document.getElementById('langToggle');b.textContent=document.body.classList.contains('en')?'中文':'EN';localStorage.setItem('lewm_lang',document.body.classList.contains('en')?'en':'zh')}if(localStorage.getItem('lewm_lang')==='en'){document.body.classList.add('en');document.addEventListener('DOMContentLoaded',function(){var b=document.getElementById('langToggle');if(b)b.textContent='中文';})}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  Demo 1: Trajectory Prediction")
    print(f"  http://localhost:8770")
    print(f"  Device: {state.device}")
    print(f"{'='*50}\n")
    uvicorn.run(app, host="0.0.0.0", port=8770)
