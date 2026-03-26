"""
Demo 2 - VoE 訓練管理 Web 介面
一站式完成：錄製 → 訓練 → 評估 → 部署

用法:
  python web_train.py
  # 開啟 http://localhost:8766
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

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CKPT_DIR = BASE_DIR / "checkpoints"
DATA_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

# ============================================================
# Global State
# ============================================================

class AppState:
    device = None
    recording = False
    training = False
    train_progress = {}
    record_progress = {}
    train_thread = None
    record_thread = None

state = AppState()

if torch.cuda.is_available():
    state.device = "cuda"
elif torch.backends.mps.is_available():
    state.device = "mps"
else:
    state.device = "cpu"

# ============================================================
# Model Builder
# ============================================================

def build_model(device="cuda"):
    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=16, image_size=64,
        pretrained=False, use_mask_token=False,
    )
    hd = encoder.config.hidden_size
    ed = 192
    predictor = ARPredictor(num_frames=4, input_dim=ed, hidden_dim=hd,
                            output_dim=hd, depth=2, heads=4, mlp_dim=hd*4, dropout=0.0)
    action_encoder = Embedder(input_dim=2, emb_dim=ed)
    projector = MLP(input_dim=hd, output_dim=ed, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hd, output_dim=ed, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    model = JEPA(encoder=encoder, predictor=predictor, action_encoder=action_encoder,
                 projector=projector, pred_proj=pred_proj).to(device)
    return model

# ============================================================
# Recording
# ============================================================

def record_worker(rtsp_url, duration, fps, img_size, output_name):
    state.recording = True
    state.record_progress = {"status": "connecting", "frames": 0, "elapsed": 0, "total": duration}

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        state.record_progress = {"status": "error", "message": "Cannot connect to camera"}
        state.recording = False
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    skip = max(1, int(src_fps / fps))
    state.record_progress["status"] = "recording"

    all_frames = []
    frame_count = 0
    start = time.time()
    ep_len = 30

    while time.time() - start < duration and state.recording:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        frame_count += 1
        if frame_count % skip != 0:
            continue

        resized = cv2.resize(frame, (img_size, img_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        all_frames.append(rgb.transpose(2, 0, 1))

        elapsed = time.time() - start
        state.record_progress = {
            "status": "recording",
            "frames": len(all_frames),
            "elapsed": round(elapsed, 1),
            "total": duration,
            "percent": round(elapsed / duration * 100, 1),
        }

    cap.release()

    if not all_frames:
        state.record_progress = {"status": "error", "message": "No frames captured"}
        state.recording = False
        return

    # Save HDF5
    state.record_progress["status"] = "saving"
    total = len(all_frames)
    num_ep = total // ep_len
    usable = num_ep * ep_len
    pixels = np.array(all_frames[:usable], dtype=np.uint8)
    actions = np.zeros((usable, 2), dtype=np.float32)
    for i in range(1, usable):
        diff = pixels[i].astype(np.float32) - pixels[i-1].astype(np.float32)
        actions[i, 0] = diff.mean()
        actions[i, 1] = np.abs(diff).mean()
    episode_ends = np.arange(ep_len, usable + 1, ep_len, dtype=np.int64)

    out_path = DATA_DIR / f"{output_name}.h5"
    with h5py.File(out_path, "w") as f:
        f.create_dataset("pixels", data=pixels, compression="gzip", compression_opts=1)
        f.create_dataset("action", data=actions)
        f.create_dataset("episode_ends", data=episode_ends)
        f.attrs["num_episodes"] = num_ep
        f.attrs["num_steps_per_episode"] = ep_len
        f.attrs["img_size"] = img_size
        f.attrs["source_url"] = rtsp_url
        f.attrs["recorded_at"] = datetime.now().isoformat()

    state.record_progress = {
        "status": "done",
        "frames": usable,
        "episodes": num_ep,
        "file": str(out_path),
        "size_mb": round(out_path.stat().st_size / 1e6, 1),
    }
    state.recording = False


# ============================================================
# Training
# ============================================================

class SyntheticH5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, window_size=8, augment=True):
        self.window_size = window_size
        self.augment = augment
        with h5py.File(h5_path, "r") as f:
            self.pixels = f["pixels"][:]
            self.actions = f["action"][:]
            self.episode_ends = f["episode_ends"][:]
        self.valid_starts = []
        ep_start = 0
        for ep_end in self.episode_ends:
            for i in range(ep_start, ep_end - window_size):
                self.valid_starts.append(i)
            ep_start = ep_end

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.window_size
        pixels = self.pixels[start:end].astype(np.float32) / 255.0
        actions = self.actions[start:end]
        if self.augment:
            pixels = pixels + np.random.uniform(-0.3, 0.3)
            pixels = pixels * np.random.uniform(0.7, 1.3)
            for c in range(3):
                pixels[:, c] += np.random.uniform(-0.2, 0.2)
            pixels = np.clip(pixels, 0, 1)
        return {"pixels": torch.from_numpy(pixels.copy()), "action": torch.from_numpy(actions)}


def train_worker(data_path, output_name, epochs, batch_size, lr):
    state.training = True
    state.train_progress = {"status": "loading", "epoch": 0, "total_epochs": epochs}

    try:
        dataset = SyntheticH5Dataset(data_path, window_size=8)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=(state.device == "cuda"), drop_last=True)

        model = build_model(state.device)
        sigreg = SIGReg(knots=17, num_proj=256).to(state.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        num_params = sum(p.numel() for p in model.parameters())
        state.train_progress["params"] = f"{num_params/1e6:.1f}M"
        state.train_progress["windows"] = len(dataset)
        state.train_progress["status"] = "training"

        out_dir = CKPT_DIR / output_name
        out_dir.mkdir(exist_ok=True)
        best_loss = float("inf")
        history = []

        for epoch in range(1, epochs + 1):
            if not state.training:
                break

            model.train()
            ep_loss = {"total": 0, "pred": 0, "sigreg": 0}
            t0 = time.time()

            for batch in loader:
                pixels = batch["pixels"].to(state.device)
                actions = torch.nan_to_num(batch["action"].to(state.device), 0.0)
                info = model.encode({"pixels": pixels, "action": actions})
                emb = info["emb"]
                act_emb = info["act_emb"]
                pred_emb = model.predict(emb[:, :4], act_emb[:, :4])
                tgt_emb = emb[:, 4:]
                pred_loss = (pred_emb - tgt_emb).pow(2).mean()
                sigreg_loss = sigreg(emb.transpose(0, 1))
                loss = pred_loss + sigreg_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss["total"] += loss.item()
                ep_loss["pred"] += pred_loss.item()
                ep_loss["sigreg"] += sigreg_loss.item()

            scheduler.step()
            n = len(loader)
            avg = {k: round(v / n, 5) for k, v in ep_loss.items()}
            elapsed = round(time.time() - t0, 1)
            history.append({"epoch": epoch, **avg, "time": elapsed})

            if avg["total"] < best_loss:
                best_loss = avg["total"]
                torch.save(model.state_dict(), out_dir / "best_model.pt")

            state.train_progress = {
                "status": "training",
                "epoch": epoch,
                "total_epochs": epochs,
                "percent": round(epoch / epochs * 100, 1),
                "pred_loss": avg["pred"],
                "sigreg_loss": avg["sigreg"],
                "total_loss": avg["total"],
                "epoch_time": elapsed,
                "best_loss": round(best_loss, 5),
                "history": history,
                "params": f"{num_params/1e6:.1f}M",
                "windows": len(dataset),
            }

        torch.save(model.state_dict(), out_dir / "final_model.pt")
        with open(out_dir / "train_history.json", "w") as f:
            json.dump(history, f, indent=2)

        state.train_progress["status"] = "done"
        state.train_progress["checkpoint"] = str(out_dir)

    except Exception as e:
        state.train_progress = {"status": "error", "message": str(e)}

    state.training = False


# ============================================================
# FastAPI
# ============================================================

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def index():
    return HTMLResponse(FRONTEND_HTML)


@app.get("/api/status")
async def api_status():
    datasets = []
    for f in sorted(DATA_DIR.glob("*.h5")):
        with h5py.File(f, "r") as h:
            datasets.append({
                "name": f.stem,
                "file": str(f),
                "episodes": int(h.attrs.get("num_episodes", 0)),
                "size_mb": round(f.stat().st_size / 1e6, 1),
                "recorded_at": h.attrs.get("recorded_at", ""),
            })
    checkpoints = []
    for d in sorted(CKPT_DIR.iterdir()):
        if d.is_dir() and (d / "best_model.pt").exists():
            hist_file = d / "train_history.json"
            last_loss = None
            epochs_done = 0
            if hist_file.exists():
                with open(hist_file) as f:
                    hist = json.load(f)
                    if hist:
                        last_loss = hist[-1].get("pred")
                        epochs_done = len(hist)
            checkpoints.append({
                "name": d.name,
                "path": str(d),
                "epochs": epochs_done,
                "pred_loss": last_loss,
            })
    return {
        "device": state.device,
        "recording": state.recording,
        "training": state.training,
        "datasets": datasets,
        "checkpoints": checkpoints,
        "record_progress": state.record_progress,
        "train_progress": state.train_progress,
    }


@app.post("/api/record")
async def api_record(body: dict):
    if state.recording:
        return JSONResponse({"error": "Already recording"}, 400)
    rtsp = body.get("rtsp", "")
    duration = body.get("duration", 300)
    fps = body.get("fps", 5)
    name = body.get("name", f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    state.record_thread = threading.Thread(
        target=record_worker, args=(rtsp, duration, fps, 64, name), daemon=True)
    state.record_thread.start()
    return {"status": "started", "name": name}


@app.post("/api/record/stop")
async def api_record_stop():
    state.recording = False
    return {"status": "stopping"}


@app.post("/api/train")
async def api_train(body: dict):
    if state.training:
        return JSONResponse({"error": "Already training"}, 400)
    data_path = body.get("data_path", "")
    name = body.get("name", f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    epochs = body.get("epochs", 30)
    batch_size = body.get("batch_size", 32)
    lr = body.get("lr", 3e-4)
    state.train_thread = threading.Thread(
        target=train_worker, args=(data_path, name, epochs, batch_size, lr), daemon=True)
    state.train_thread.start()
    return {"status": "started", "name": name}


@app.post("/api/train/stop")
async def api_train_stop():
    state.training = False
    return {"status": "stopping"}


@app.post("/api/preview")
async def api_preview(body: dict):
    rtsp = body.get("rtsp", "")
    cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return JSONResponse({"error": "Cannot connect"}, 400)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return JSONResponse({"error": "No frame"}, 400)
    display = cv2.resize(frame, (640, 360))
    _, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return {"frame": base64.b64encode(buf).decode('utf-8')}


# ============================================================
# Frontend
# ============================================================

FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LeWM VoE - Training Manager</title>
<style>
  :root { --bg: #0f172a; --card: #1e293b; --accent: #3b82f6; --green: #22c55e; --red: #ef4444; --orange: #f59e0b; --text: #e2e8f0; --dim: #64748b; --border: #334155; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

  .header { padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem; border-bottom: 1px solid var(--border); }
  .header h1 { font-size: 1.3rem; }
  .tabs { display: flex; gap: 0; margin-left: auto; }
  .tab { padding: 6px 16px; cursor: pointer; border: 1px solid var(--border); font-size: 0.85rem; color: var(--dim); background: transparent; }
  .tab:first-child { border-radius: 6px 0 0 6px; }
  .tab:last-child { border-radius: 0 6px 6px 0; }
  .tab.active { background: var(--accent); color: #fff; border-color: var(--accent); }

  .page { display: none; padding: 1.5rem 2rem; max-width: 1000px; margin: 0 auto; }
  .page.active { display: block; }

  .card { background: var(--card); border-radius: 10px; padding: 1.5rem; margin-bottom: 1rem; }
  .card h2 { font-size: 1.1rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
  .card h2 .icon { font-size: 1.3rem; }

  .form-row { display: flex; gap: 1rem; margin-bottom: 0.8rem; align-items: flex-end; }
  .form-group { flex: 1; }
  .form-group label { display: block; font-size: 0.8rem; color: var(--dim); margin-bottom: 4px; }
  .form-group input, .form-group select { width: 100%; padding: 8px 12px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg); color: var(--text); font-size: 0.9rem; }

  .btn { padding: 8px 20px; border: none; border-radius: 6px; font-size: 0.9rem; font-weight: 600; cursor: pointer; transition: all 0.15s; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover { background: #2563eb; }
  .btn-green { background: var(--green); color: #000; }
  .btn-green:hover { background: #16a34a; }
  .btn-red { background: var(--red); color: #fff; }
  .btn-red:hover { background: #dc2626; }
  .btn-orange { background: var(--orange); color: #000; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .progress-bar { height: 8px; background: var(--bg); border-radius: 4px; overflow: hidden; margin: 0.5rem 0; }
  .progress-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }

  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.8rem; margin: 1rem 0; }
  .stat-box { background: var(--bg); border-radius: 8px; padding: 0.8rem; text-align: center; }
  .stat-box .num { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
  .stat-box .label { font-size: 0.75rem; color: var(--dim); }

  .list-item { display: flex; align-items: center; justify-content: space-between; padding: 10px; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
  .list-item:last-child { border-bottom: none; }
  .list-item .name { font-weight: 600; }
  .list-item .meta { color: var(--dim); font-size: 0.8rem; }
  .badge { padding: 2px 8px; border-radius: 999px; font-size: 0.7rem; font-weight: 600; }
  .badge-green { background: #166534; color: #86efac; }
  .badge-blue { background: #1e40af; color: #93c5fd; }

  .preview-img { max-width: 100%; border-radius: 8px; margin-top: 0.5rem; }

  .chart-box { height: 150px; background: var(--bg); border-radius: 8px; position: relative; margin: 1rem 0; }
  .chart-box canvas { width: 100%; height: 100%; }

  .log { background: #000; color: #86efac; padding: 1rem; border-radius: 8px; font-family: monospace; font-size: 0.8rem; max-height: 200px; overflow-y: auto; margin-top: 0.5rem; white-space: pre-wrap; }
</style>
</head>
<body>

<div class="header">
  <h1>LeWM Training Manager</h1>
  <div class="tabs">
    <button class="tab active" onclick="showPage('record')">1. Record</button>
    <button class="tab" onclick="showPage('train')">2. Train</button>
    <button class="tab" onclick="showPage('models')">3. Models</button>
  </div>
</div>

<!-- ===== RECORD PAGE ===== -->
<div class="page active" id="page-record">
  <div class="card">
    <h2><span class="icon">📹</span> Record Training Data</h2>
    <p style="color:var(--dim);font-size:0.85rem;margin-bottom:1rem">Record camera footage and convert to HDF5 training data</p>

    <div class="form-row">
      <div class="form-group" style="flex:3">
        <label>RTSP URL</label>
        <input id="recRtsp" value="rtsp://admin:Ms!23456@116.59.11.189:554/sub" />
      </div>
      <button class="btn btn-primary" onclick="previewCamera()">Preview</button>
    </div>

    <img id="previewImg" class="preview-img" style="display:none" />

    <div class="form-row">
      <div class="form-group">
        <label>Duration (sec)</label>
        <input id="recDuration" type="number" value="300" />
      </div>
      <div class="form-group">
        <label>FPS</label>
        <input id="recFps" type="number" value="5" />
      </div>
      <div class="form-group">
        <label>Dataset Name</label>
        <input id="recName" value="" placeholder="auto" />
      </div>
    </div>

    <div style="margin-top:1rem">
      <button class="btn btn-green" id="btnRecord" onclick="startRecord()">Start Recording</button>
      <button class="btn btn-red" id="btnRecStop" onclick="stopRecord()" style="display:none">Stop</button>
    </div>

    <div id="recProgress" style="display:none;margin-top:1rem">
      <div class="progress-bar"><div class="progress-fill" id="recBar" style="width:0%;background:var(--green)"></div></div>
      <div id="recStatus" style="font-size:0.85rem;color:var(--dim)"></div>
    </div>
  </div>

  <div class="card">
    <h2><span class="icon">📁</span> Available Datasets</h2>
    <div id="datasetList"><span style="color:var(--dim)">Loading...</span></div>
  </div>
</div>

<!-- ===== TRAIN PAGE ===== -->
<div class="page" id="page-train">
  <div class="card">
    <h2><span class="icon">🧠</span> Train Model</h2>

    <div class="form-row">
      <div class="form-group" style="flex:2">
        <label>Dataset</label>
        <select id="trainData"></select>
      </div>
      <div class="form-group">
        <label>Model Name</label>
        <input id="trainName" value="" placeholder="auto" />
      </div>
    </div>

    <div class="form-row">
      <div class="form-group">
        <label>Epochs</label>
        <input id="trainEpochs" type="number" value="30" />
      </div>
      <div class="form-group">
        <label>Batch Size</label>
        <input id="trainBatch" type="number" value="32" />
      </div>
      <div class="form-group">
        <label>Learning Rate</label>
        <input id="trainLr" type="number" value="0.0003" step="0.0001" />
      </div>
    </div>

    <div style="margin-top:1rem">
      <button class="btn btn-green" id="btnTrain" onclick="startTrain()">Start Training</button>
      <button class="btn btn-red" id="btnTrainStop" onclick="stopTrain()" style="display:none">Stop</button>
    </div>
  </div>

  <div class="card" id="trainProgressCard" style="display:none">
    <h2><span class="icon">📊</span> Training Progress</h2>
    <div class="progress-bar"><div class="progress-fill" id="trainBar" style="width:0%;background:var(--accent)"></div></div>
    <div class="stat-grid">
      <div class="stat-box"><div class="num" id="tEpoch">-</div><div class="label">Epoch</div></div>
      <div class="stat-box"><div class="num" id="tPred">-</div><div class="label">Pred Loss</div></div>
      <div class="stat-box"><div class="num" id="tBest">-</div><div class="label">Best Loss</div></div>
      <div class="stat-box"><div class="num" id="tTime">-</div><div class="label">Epoch Time</div></div>
    </div>
    <div class="chart-box"><canvas id="trainChart"></canvas></div>
    <div class="log" id="trainLog"></div>
  </div>
</div>

<!-- ===== MODELS PAGE ===== -->
<div class="page" id="page-models">
  <div class="card">
    <h2><span class="icon">📦</span> Trained Models</h2>
    <div id="modelList"><span style="color:var(--dim)">Loading...</span></div>
  </div>
  <div class="card">
    <h2><span class="icon">🚀</span> Deploy</h2>
    <p style="color:var(--dim);font-size:0.85rem">Select a model above, then start the VoE detection demo:</p>
    <div style="margin-top:0.5rem">
      <code style="background:var(--bg);padding:8px 12px;border-radius:6px;display:block;font-size:0.85rem">
        python web_demo.py  →  http://localhost:8765
      </code>
    </div>
  </div>
</div>

<script>
function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  event.target.classList.add('active');
  refreshStatus();
}

async function refreshStatus() {
  try {
    const r = await fetch('/api/status');
    const data = await r.json();
    renderDatasets(data.datasets);
    renderModels(data.checkpoints);
    renderTrainSelect(data.datasets);
    updateRecordUI(data);
    updateTrainUI(data);
  } catch(e) {}
}

function renderDatasets(datasets) {
  const el = document.getElementById('datasetList');
  if (!datasets.length) { el.innerHTML = '<span style="color:var(--dim)">No datasets yet. Record one above.</span>'; return; }
  el.innerHTML = datasets.map(d => `
    <div class="list-item">
      <div><div class="name">${d.name}</div><div class="meta">${d.episodes} episodes · ${d.size_mb} MB · ${d.recorded_at ? d.recorded_at.slice(0,16) : ''}</div></div>
      <span class="badge badge-green">${d.episodes} ep</span>
    </div>
  `).join('');
}

function renderModels(models) {
  const el = document.getElementById('modelList');
  if (!models.length) { el.innerHTML = '<span style="color:var(--dim)">No models yet. Train one first.</span>'; return; }
  el.innerHTML = models.map(m => `
    <div class="list-item">
      <div><div class="name">${m.name}</div><div class="meta">${m.epochs} epochs · pred_loss: ${m.pred_loss !== null ? m.pred_loss.toFixed(4) : '-'}</div></div>
      <span class="badge badge-blue">${m.epochs} ep</span>
    </div>
  `).join('');
}

function renderTrainSelect(datasets) {
  const sel = document.getElementById('trainData');
  const cur = sel.value;
  sel.innerHTML = datasets.map(d => `<option value="${d.file}">${d.name} (${d.episodes} ep)</option>`).join('');
  if (cur) sel.value = cur;
}

function updateRecordUI(data) {
  if (data.recording && data.record_progress.status === 'recording') {
    document.getElementById('recProgress').style.display = 'block';
    document.getElementById('recBar').style.width = (data.record_progress.percent || 0) + '%';
    document.getElementById('recStatus').textContent = `${data.record_progress.frames} frames · ${data.record_progress.elapsed}s / ${data.record_progress.total}s`;
    document.getElementById('btnRecord').style.display = 'none';
    document.getElementById('btnRecStop').style.display = 'inline-block';
  } else if (data.record_progress.status === 'done') {
    document.getElementById('recProgress').style.display = 'block';
    document.getElementById('recBar').style.width = '100%';
    document.getElementById('recStatus').textContent = `Done! ${data.record_progress.episodes} episodes · ${data.record_progress.size_mb} MB`;
    document.getElementById('btnRecord').style.display = 'inline-block';
    document.getElementById('btnRecStop').style.display = 'none';
  }
}

function updateTrainUI(data) {
  if (!data.training && data.train_progress.status !== 'training') return;
  document.getElementById('trainProgressCard').style.display = 'block';

  const tp = data.train_progress;
  document.getElementById('trainBar').style.width = (tp.percent || 0) + '%';
  document.getElementById('tEpoch').textContent = `${tp.epoch || 0}/${tp.total_epochs || 0}`;
  document.getElementById('tPred').textContent = tp.pred_loss !== undefined ? tp.pred_loss.toFixed(4) : '-';
  document.getElementById('tBest').textContent = tp.best_loss !== undefined ? tp.best_loss.toFixed(4) : '-';
  document.getElementById('tTime').textContent = tp.epoch_time ? tp.epoch_time + 's' : '-';

  if (tp.status === 'done') {
    document.getElementById('trainBar').style.background = 'var(--green)';
    document.getElementById('btnTrain').style.display = 'inline-block';
    document.getElementById('btnTrainStop').style.display = 'none';
  }

  if (tp.history) drawTrainChart(tp.history);

  // Log
  if (tp.history) {
    const log = document.getElementById('trainLog');
    log.textContent = tp.history.map(h =>
      `Epoch ${String(h.epoch).padStart(3)} | pred=${h.pred.toFixed(4)} sigreg=${h.sigreg.toFixed(4)} | ${h.time}s`
    ).join('\\n');
    log.scrollTop = log.scrollHeight;
  }
}

function drawTrainChart(history) {
  const canvas = document.getElementById('trainChart');
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.offsetWidth * 2;
  const H = canvas.height = canvas.offsetHeight * 2;
  ctx.clearRect(0, 0, W, H);

  const preds = history.map(h => h.pred);
  const maxVal = Math.max(...preds) * 1.2 || 1;
  const pad = 20;

  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + ((H - 2*pad) / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W-pad, y); ctx.stroke();
  }

  ctx.beginPath();
  ctx.lineWidth = 3;
  ctx.strokeStyle = var_accent = '#3b82f6';
  preds.forEach((v, i) => {
    const x = pad + (i / (preds.length - 1 || 1)) * (W - 2*pad);
    const y = pad + (H - 2*pad) - (v / maxVal) * (H - 2*pad);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = '#94a3b8';
  ctx.font = '18px sans-serif';
  ctx.fillText('pred_loss', pad + 5, pad + 15);
  ctx.fillText(preds[preds.length-1]?.toFixed(4) || '', W - pad - 70, pad + 15);
}

async function previewCamera() {
  const rtsp = document.getElementById('recRtsp').value;
  try {
    const r = await fetch('/api/preview', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({rtsp}) });
    const data = await r.json();
    if (data.frame) {
      const img = document.getElementById('previewImg');
      img.src = 'data:image/jpeg;base64,' + data.frame;
      img.style.display = 'block';
    } else {
      alert(data.error || 'Preview failed');
    }
  } catch(e) { alert('Connection failed'); }
}

async function startRecord() {
  const body = {
    rtsp: document.getElementById('recRtsp').value,
    duration: parseInt(document.getElementById('recDuration').value),
    fps: parseInt(document.getElementById('recFps').value),
    name: document.getElementById('recName').value || undefined,
  };
  await fetch('/api/record', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
  document.getElementById('btnRecord').style.display = 'none';
  document.getElementById('btnRecStop').style.display = 'inline-block';
  document.getElementById('recProgress').style.display = 'block';
}

async function stopRecord() {
  await fetch('/api/record/stop', { method: 'POST' });
  document.getElementById('btnRecord').style.display = 'inline-block';
  document.getElementById('btnRecStop').style.display = 'none';
}

async function startTrain() {
  const body = {
    data_path: document.getElementById('trainData').value,
    name: document.getElementById('trainName').value || undefined,
    epochs: parseInt(document.getElementById('trainEpochs').value),
    batch_size: parseInt(document.getElementById('trainBatch').value),
    lr: parseFloat(document.getElementById('trainLr').value),
  };
  await fetch('/api/train', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
  document.getElementById('btnTrain').style.display = 'none';
  document.getElementById('btnTrainStop').style.display = 'inline-block';
  document.getElementById('trainProgressCard').style.display = 'block';
}

async function stopTrain() {
  await fetch('/api/train/stop', { method: 'POST' });
  document.getElementById('btnTrain').style.display = 'inline-block';
  document.getElementById('btnTrainStop').style.display = 'none';
}

// Auto-refresh
setInterval(refreshStatus, 2000);
refreshStatus();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  LeWM Training Manager - http://localhost:8766")
    print(f"  Device: {state.device}")
    print(f"  Data:   {DATA_DIR}")
    print(f"  Models: {CKPT_DIR}")
    print(f"{'='*50}\n")
    uvicorn.run(app, host="0.0.0.0", port=8766)
