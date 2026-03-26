"""
Demo 2 - VoE 異常偵測 Web 介面
FastAPI 後端 + WebSocket 即時串流 + Surprise 波形圖

用法:
  python web_demo.py
  # 開啟 http://localhost:8765
"""
import sys
import os
import asyncio
import time
import base64
import json
from collections import deque
from pathlib import Path

# le-wm path
for p in ["/home/rai/code/le-wm", os.path.expanduser("~/code/2026/le-wm-local")]:
    if os.path.exists(p):
        sys.path.insert(0, p)
        break

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import stable_pretraining as spt
from module import MLP, Embedder, ARPredictor
from jepa import JEPA

# ============================================================
# Model
# ============================================================

def load_model(checkpoint, device):
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
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


# ============================================================
# Surprise Engine
# ============================================================

class SurpriseEngine:
    def __init__(self, model, device, history_size=4, img_size=64):
        self.model = model
        self.device = device
        self.history_size = history_size
        self.img_size = img_size
        self.frame_buffer = deque(maxlen=history_size + 1)
        self.surprise_history = deque(maxlen=600)
        self.frame_count = 0

    def reset(self):
        self.frame_buffer.clear()
        self.surprise_history.clear()
        self.frame_count = 0

    def preprocess(self, frame):
        resized = cv2.resize(frame, (self.img_size, self.img_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0

    def compute(self, frame):
        self.frame_count += 1
        tensor = self.preprocess(frame)
        self.frame_buffer.append(tensor)

        if len(self.frame_buffer) < self.history_size + 1:
            return None

        frames = list(self.frame_buffer)
        ctx = torch.stack(frames[:self.history_size]).unsqueeze(0).to(self.device)
        nxt = torch.stack(frames[1:self.history_size + 1]).unsqueeze(0).to(self.device)
        actions = torch.zeros(1, self.history_size, 2, device=self.device)

        with torch.no_grad():
            info_ctx = self.model.encode({"pixels": ctx, "action": actions})
            pred = self.model.predict(info_ctx["emb"], info_ctx["act_emb"])
            info_nxt = self.model.encode({"pixels": nxt, "action": actions})
            surprise = (pred - info_nxt["emb"]).pow(2).mean().item()

        self.surprise_history.append(surprise)
        return surprise

    def get_threshold(self):
        if len(self.surprise_history) < 30:
            return None
        vals = list(self.surprise_history)
        return np.mean(vals) + 2 * np.std(vals)

    def get_stats(self):
        if not self.surprise_history:
            return {}
        vals = list(self.surprise_history)
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }


# ============================================================
# App
# ============================================================

app = FastAPI()

# Global state
MODEL = None
ENGINE = None
DEVICE = None

SCRIPT_DIR = Path(__file__).parent.parent


def init_model():
    global MODEL, ENGINE, DEVICE
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    ckpt = SCRIPT_DIR / "checkpoints" / "camera_v2" / "best_model.pt"
    if not ckpt.exists():
        ckpt = SCRIPT_DIR / "checkpoints" / "demo2_v3" / "best_model.pt"
    if not ckpt.exists():
        print("No checkpoint found!")
        return False

    print(f"Loading model from {ckpt} on {DEVICE}")
    MODEL = load_model(str(ckpt), DEVICE)
    ENGINE = SurpriseEngine(MODEL, DEVICE)
    print("Model loaded!")
    return True


# ============================================================
# Frontend
# ============================================================


@app.get("/about")
async def about_page():
    about_file = Path(__file__).parent.parent / "about.html"
    if about_file.exists():
        return HTMLResponse(about_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>About page not found</h1>")

@app.get("/")
async def index():
    return HTMLResponse(FRONTEND_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        # Wait for start command
        data = await websocket.receive_json()
        rtsp_url = data.get("rtsp", "")
        if not rtsp_url:
            await websocket.send_json({"error": "No RTSP URL"})
            return

        await websocket.send_json({"status": "connecting", "rtsp": rtsp_url})

        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            await websocket.send_json({"error": f"Cannot connect to {rtsp_url}"})
            return

        await websocket.send_json({"status": "connected"})
        ENGINE.reset()

        frame_skip = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                await websocket.send_json({"error": "Stream lost"})
                break

            frame_skip += 1
            if frame_skip % 5 != 0:  # ~5 fps
                continue

            # Compute surprise
            t0 = time.perf_counter()
            surprise = ENGINE.compute(frame)
            infer_ms = (time.perf_counter() - t0) * 1000

            threshold = ENGINE.get_threshold()
            is_anomaly = surprise is not None and threshold is not None and surprise > threshold

            # Draw overlay on frame
            display = cv2.resize(frame, (640, 360))
            if is_anomaly:
                cv2.rectangle(display, (0, 0), (639, 359), (0, 0, 255), 4)
                cv2.putText(display, "ANOMALY", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                cv2.putText(display, "Normal", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)

            if surprise is not None:
                cv2.putText(display, f"Surprise: {surprise:.5f}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(display, f"Infer: {infer_ms:.1f}ms", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Encode frame as JPEG
            _, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf).decode('utf-8')

            # Send data
            msg = {
                "frame": b64,
                "surprise": surprise,
                "threshold": threshold,
                "is_anomaly": is_anomaly,
                "infer_ms": round(infer_ms, 1),
                "frame_count": ENGINE.frame_count,
                "history": list(ENGINE.surprise_history)[-200:],
                "stats": ENGINE.get_stats(),
            }
            await websocket.send_json(msg)

            # Check for stop command (non-blocking)
            try:
                cmd = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                if cmd.get("action") == "stop":
                    break
            except (asyncio.TimeoutError, Exception):
                pass

        cap.release()
        await websocket.send_json({"status": "stopped"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass


# ============================================================
# Frontend HTML
# ============================================================

FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LeWM VoE Demo</title>
<style>
  :root { --bg: #0f172a; --card: #1e293b; --accent: #3b82f6; --green: #22c55e; --red: #ef4444; --text: #e2e8f0; --dim: #64748b; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
  .header { padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem; border-bottom: 1px solid #334155; }
  .header h1 { font-size: 1.3rem; }
  .header .tag { background: var(--accent); padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; }
  .main { display: grid; grid-template-columns: 1fr 340px; height: calc(100vh - 56px); }
  @media (max-width: 900px) { .main { grid-template-columns: 1fr; } }

  .video-section { padding: 1rem; display: flex; flex-direction: column; gap: 0.5rem; }
  .video-container { position: relative; background: #000; border-radius: 8px; overflow: hidden; flex: 1; display: flex; align-items: center; justify-content: center; }
  .video-container img { max-width: 100%; max-height: 100%; }
  .video-container .placeholder { color: var(--dim); font-size: 1.1rem; }

  .chart-container { height: 120px; background: var(--card); border-radius: 8px; padding: 0.5rem; position: relative; }
  .chart-container canvas { width: 100%; height: 100%; }
  .chart-label { position: absolute; top: 4px; left: 8px; font-size: 0.7rem; color: var(--dim); }

  .sidebar { background: var(--card); padding: 1.5rem; overflow-y: auto; border-left: 1px solid #334155; }
  .sidebar h2 { font-size: 1rem; margin-bottom: 1rem; color: var(--dim); text-transform: uppercase; letter-spacing: 1px; }

  .input-group { margin-bottom: 1rem; }
  .input-group label { display: block; font-size: 0.8rem; color: var(--dim); margin-bottom: 4px; }
  .input-group input, .input-group select { width: 100%; padding: 8px 12px; border: 1px solid #475569; border-radius: 6px; background: var(--bg); color: var(--text); font-size: 0.9rem; }
  .input-group select { cursor: pointer; }

  .btn { width: 100%; padding: 10px; border: none; border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: all 0.2s; }
  .btn-start { background: var(--green); color: #000; }
  .btn-start:hover { background: #16a34a; }
  .btn-stop { background: var(--red); color: #fff; }
  .btn-stop:hover { background: #dc2626; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }

  .stats { margin-top: 1.5rem; }
  .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #334155; font-size: 0.85rem; }
  .stat-row .val { font-weight: 600; font-family: 'SF Mono', monospace; }
  .stat-row .val.green { color: var(--green); }
  .stat-row .val.red { color: var(--red); }

  .status { margin: 1rem 0; padding: 8px 12px; border-radius: 6px; font-size: 0.85rem; text-align: center; }
  .status.idle { background: #334155; }
  .status.connected { background: #166534; }
  .status.error { background: #991b1b; }
  .status.anomaly { background: var(--red); animation: pulse 0.5s infinite alternate; }
  @keyframes pulse { from { opacity: 1; } to { opacity: 0.6; } }

  .presets { margin-top: 1rem; }
  .preset-btn { display: block; width: 100%; padding: 6px 10px; margin: 4px 0; background: var(--bg); border: 1px solid #475569; border-radius: 6px; color: var(--text); font-size: 0.8rem; cursor: pointer; text-align: left; }
  .preset-btn:hover { border-color: var(--accent); }
.lang-btn{position:fixed;top:12px;right:16px;z-index:999;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.3);color:#fff;padding:4px 14px;border-radius:999px;cursor:pointer;font-size:.8rem;font-weight:600;backdrop-filter:blur(8px)}.lang-btn:hover{background:rgba(255,255,255,.25)}.about-link{position:fixed;top:12px;right:70px;z-index:999;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);color:#94a3b8;padding:4px 14px;border-radius:999px;cursor:pointer;font-size:.8rem;text-decoration:none}.about-link:hover{color:#fff;background:rgba(255,255,255,.2)}.lang-zh{display:block}.lang-en{display:none}body.en .lang-zh{display:none}body.en .lang-en{display:block}</style>
</head>
<body>

<div class="header">
  <h1>LeWM VoE Demo</h1>
  <span class="tag">LIVE</span>
  <span style="color:var(--dim);font-size:0.85rem">Violation-of-Expectation Anomaly Detection</span>
</div>

<div class="main">
  <div class="video-section">
    <div class="video-container" id="videoContainer">
      <span class="placeholder" id="placeholder">Select RTSP source and click Start</span>
      <img id="videoFrame" style="display:none" />
    </div>
    <div class="chart-container">
      <span class="chart-label">Surprise Timeline</span>
      <canvas id="surpriseChart"></canvas>
    </div>
  </div>

  <div class="sidebar">
    <h2>Control</h2>

    <div class="input-group">
      <label>RTSP URL</label>
      <input type="text" id="rtspInput" placeholder="rtsp://admin:password@ip:554/sub" />
    </div>

    <div class="presets">
      <label style="font-size:0.8rem;color:var(--dim)">Presets</label>
      <button class="preset-btn" onclick="setPreset('rtsp://admin:Ms!23456@116.59.11.189:554/sub')">
        Milesight (sub stream)
      </button>
      <button class="preset-btn" onclick="setPreset('rtsp://admin:Ms!23456@116.59.11.189:554/main')">
        Milesight (main stream)
      </button>
    </div>

    <div style="margin-top:1rem">
      <button class="btn btn-start" id="btnStart" onclick="startDetection()">Start Detection</button>
      <button class="btn btn-stop" id="btnStop" onclick="stopDetection()" style="display:none">Stop</button>
    </div>

    <div class="status idle" id="statusBar">Idle</div>

    <div class="stats">
      <h2>Statistics</h2>
      <div class="stat-row"><span>Frames</span><span class="val" id="statFrames">-</span></div>
      <div class="stat-row"><span>Current Surprise</span><span class="val" id="statSurprise">-</span></div>
      <div class="stat-row"><span>Threshold</span><span class="val" id="statThreshold">-</span></div>
      <div class="stat-row"><span>Mean</span><span class="val" id="statMean">-</span></div>
      <div class="stat-row"><span>Std</span><span class="val" id="statStd">-</span></div>
      <div class="stat-row"><span>Max</span><span class="val" id="statMax">-</span></div>
      <div class="stat-row"><span>Inference</span><span class="val green" id="statInfer">-</span></div>
      <div class="stat-row"><span>Status</span><span class="val" id="statStatus">-</span></div>
    </div>
  </div>
</div>

<script>
let ws = null;
let running = false;

function setPreset(url) {
  document.getElementById('rtspInput').value = url;
}

function setStatus(text, cls) {
  const bar = document.getElementById('statusBar');
  bar.textContent = text;
  bar.className = 'status ' + cls;
}

function startDetection() {
  const rtsp = document.getElementById('rtspInput').value.trim();
  if (!rtsp) { alert('Please enter RTSP URL'); return; }

  running = true;
  document.getElementById('btnStart').style.display = 'none';
  document.getElementById('btnStop').style.display = 'block';
  setStatus('Connecting...', 'idle');

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws');

  ws.onopen = () => {
    ws.send(JSON.stringify({ rtsp: rtsp }));
  };

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);

    if (data.error) {
      setStatus('Error: ' + data.error, 'error');
      stopDetection();
      return;
    }

    if (data.status === 'connecting') {
      setStatus('Connecting to camera...', 'idle');
      return;
    }
    if (data.status === 'connected') {
      setStatus('Connected - Calibrating...', 'connected');
      document.getElementById('placeholder').style.display = 'none';
      document.getElementById('videoFrame').style.display = 'block';
      return;
    }
    if (data.status === 'stopped') {
      setStatus('Stopped', 'idle');
      return;
    }

    // Frame data
    if (data.frame) {
      document.getElementById('videoFrame').src = 'data:image/jpeg;base64,' + data.frame;
    }

    // Stats
    document.getElementById('statFrames').textContent = data.frame_count || '-';
    document.getElementById('statInfer').textContent = data.infer_ms ? data.infer_ms + ' ms' : '-';

    if (data.surprise !== null && data.surprise !== undefined) {
      const el = document.getElementById('statSurprise');
      el.textContent = data.surprise.toFixed(5);
      el.className = 'val ' + (data.is_anomaly ? 'red' : 'green');
    }

    if (data.threshold !== null && data.threshold !== undefined) {
      document.getElementById('statThreshold').textContent = data.threshold.toFixed(5);
    }

    if (data.stats) {
      document.getElementById('statMean').textContent = data.stats.mean ? data.stats.mean.toFixed(5) : '-';
      document.getElementById('statStd').textContent = data.stats.std ? data.stats.std.toFixed(5) : '-';
      document.getElementById('statMax').textContent = data.stats.max ? data.stats.max.toFixed(5) : '-';
    }

    if (data.is_anomaly) {
      setStatus('ANOMALY DETECTED', 'anomaly');
      document.getElementById('statStatus').textContent = 'ANOMALY';
      document.getElementById('statStatus').className = 'val red';
    } else if (data.surprise !== null) {
      setStatus('Monitoring', 'connected');
      document.getElementById('statStatus').textContent = 'Normal';
      document.getElementById('statStatus').className = 'val green';
    }

    // Draw chart
    if (data.history && data.history.length > 1) {
      drawChart(data.history, data.threshold);
    }
  };

  ws.onclose = () => {
    if (running) setStatus('Disconnected', 'error');
  };
}

function stopDetection() {
  running = false;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: 'stop' }));
    ws.close();
  }
  document.getElementById('btnStart').style.display = 'block';
  document.getElementById('btnStop').style.display = 'none';
  setStatus('Stopped', 'idle');
}

function drawChart(history, threshold) {
  const canvas = document.getElementById('surpriseChart');
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.offsetWidth * 2;
  const H = canvas.height = canvas.offsetHeight * 2;
  ctx.scale(1, 1);

  ctx.clearRect(0, 0, W, H);

  const maxVal = Math.max(...history) * 1.3 || 1;
  const pad = 10;
  const chartW = W - pad * 2;
  const chartH = H - pad * 2;

  // Grid
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + (chartH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(W - pad, y);
    ctx.stroke();
  }

  // Threshold line
  if (threshold !== null && threshold !== undefined) {
    const thY = pad + chartH - (threshold / maxVal) * chartH;
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(pad, thY);
    ctx.lineTo(W - pad, thY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#f59e0b';
    ctx.font = '18px sans-serif';
    ctx.fillText('threshold', W - pad - 80, thY - 5);
  }

  // Surprise line
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < history.length; i++) {
    const x = pad + (i / (history.length - 1)) * chartW;
    const y = pad + chartH - (history[i] / maxVal) * chartH;

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = '#3b82f6';
  ctx.stroke();

  // Anomaly highlights
  if (threshold !== null && threshold !== undefined) {
    for (let i = 0; i < history.length; i++) {
      if (history[i] > threshold) {
        const x = pad + (i / (history.length - 1)) * chartW;
        const y = pad + chartH - (history[i] / maxVal) * chartH;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#ef4444';
        ctx.fill();
      }
    }
  }
}

// Set default
document.getElementById('rtspInput').value = 'rtsp://admin:Ms!23456@116.59.11.189:554/sub';
function toggleLang(){document.body.classList.toggle('en');var b=document.getElementById('langToggle');b.textContent=document.body.classList.contains('en')?'中文':'EN';localStorage.setItem('lewm_lang',document.body.classList.contains('en')?'en':'zh')}if(localStorage.getItem('lewm_lang')==='en'){document.body.classList.add('en');document.addEventListener('DOMContentLoaded',function(){var b=document.getElementById('langToggle');if(b)b.textContent='中文';})}
</script>
</body>
</html>
"""


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if not init_model():
        print("Failed to load model. Place checkpoint in checkpoints/camera_v2/ or checkpoints/demo2_v3/")
        sys.exit(1)

    print("\n" + "=" * 50)
    print(f"  LeWM VoE Demo - http://localhost:8765")
    print(f"  Device: {DEVICE}")
    print("=" * 50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8765)
