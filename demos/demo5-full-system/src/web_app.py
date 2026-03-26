"""
Demo 5 - 完整系統整合 Dashboard
整合所有 Demo 的入口，提供統一啟動頁面

啟動: python web_app.py → http://localhost:8780
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()


@app.get("/about")
async def about_page():
    about_file = Path(__file__).parent.parent / "about.html"
    if about_file.exists():
        return HTMLResponse(about_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>About page not found</h1>")

@app.get("/")
async def index():
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>LeWM Industrial Safety - Full System</title>
<style>
:root{--bg:#0f172a;--card:#1e293b;--accent:#3b82f6;--green:#22c55e;--red:#ef4444;--orange:#f59e0b;--purple:#8b5cf6;--text:#e2e8f0;--dim:#64748b;--border:#334155}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-height:100vh}
.hero{text-align:center;padding:3rem 2rem 2rem}
.hero h1{font-size:2.2rem;margin-bottom:.5rem}
.hero p{color:var(--dim);font-size:1rem;max-width:600px;margin:0 auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1.5rem;max-width:1100px;margin:2rem auto;padding:0 2rem}
.demo-card{background:var(--card);border-radius:16px;padding:2rem;transition:transform .2s,box-shadow .2s;cursor:pointer;text-decoration:none;color:var(--text);border:2px solid transparent;position:relative;overflow:hidden}
.demo-card:hover{transform:translateY(-4px);box-shadow:0 8px 30px rgba(0,0,0,.3);border-color:var(--accent)}
.demo-card .icon{font-size:2.5rem;margin-bottom:1rem}
.demo-card h2{font-size:1.2rem;margin-bottom:.5rem}
.demo-card p{color:var(--dim);font-size:.85rem;line-height:1.5}
.demo-card .port{position:absolute;top:1rem;right:1rem;background:var(--bg);padding:2px 10px;border-radius:999px;font-size:.75rem;color:var(--dim);font-family:monospace}
.demo-card .status{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.demo-card .status.on{background:var(--green)}.demo-card .status.off{background:var(--red)}
.demo-card .tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.7rem;font-weight:600;margin-top:.5rem}
.footer{text-align:center;padding:2rem;color:var(--dim);font-size:.85rem}
.footer a{color:var(--accent)}
.section-title{max-width:1100px;margin:2rem auto 0;padding:0 2rem;font-size:.9rem;color:var(--dim);text-transform:uppercase;letter-spacing:2px}
.lang-btn{position:fixed;top:12px;right:16px;z-index:999;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.3);color:#fff;padding:4px 14px;border-radius:999px;cursor:pointer;font-size:.8rem;font-weight:600;backdrop-filter:blur(8px)}.lang-btn:hover{background:rgba(255,255,255,.25)}.about-link{position:fixed;top:12px;right:70px;z-index:999;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);color:#94a3b8;padding:4px 14px;border-radius:999px;cursor:pointer;font-size:.8rem;text-decoration:none}.about-link:hover{color:#fff;background:rgba(255,255,255,.2)}.lang-zh{display:block}.lang-en{display:none}body.en .lang-zh{display:none}body.en .lang-en{display:block}</style>
</head>
<body>
<div class="hero">
  <h1>LeWM Industrial Safety</h1>
  <p>JEPA World Model for Predictive Safety Monitoring — 5 Demo Modules</p>
</div>

<div class="section-title">Detection & Monitoring</div>
<div class="grid">
  <a class="demo-card" href="http://localhost:8765" target="_blank" style="border-color:var(--green)">
    <span class="port">:8765</span>
    <div class="icon">🔍</div>
    <h2>Demo 2: VoE Anomaly Detection</h2>
    <p>Violation-of-Expectation: detect any physics anomaly in real-time. Surprise waveform + dynamic threshold.</p>
    <div class="tag" style="background:#166534;color:#86efac">CORE</div>
  </a>

  <a class="demo-card" href="http://localhost:8770" target="_blank">
    <span class="port">:8770</span>
    <div class="icon">🎯</div>
    <h2>Demo 1: Trajectory Prediction</h2>
    <p>Predict worker trajectory, warn before entering danger zones. YOLO person detection + LeWM prediction.</p>
    <div class="tag" style="background:#92400e;color:#fcd34d">PREDICTIVE</div>
  </a>

  <a class="demo-card" href="http://localhost:8771" target="_blank">
    <span class="port">:8771</span>
    <div class="icon">👁️</div>
    <h2>Demo 3: Dual Vision</h2>
    <p>YOLO object detection + LeWM physics understanding side-by-side. See what each system catches.</p>
    <div class="tag" style="background:#581c87;color:#d8b4fe">COMPARISON</div>
  </a>

  <a class="demo-card" href="http://localhost:8772" target="_blank">
    <span class="port">:8772</span>
    <div class="icon">📡</div>
    <h2>Demo 4: Smart PTZ Patrol</h2>
    <p>Camera auto-rotates to highest-risk region based on info gain = time × surprise.</p>
    <div class="tag" style="background:#1e40af;color:#93c5fd">AUTONOMOUS</div>
  </a>
</div>

<div class="section-title">Training & Management</div>
<div class="grid">
  <a class="demo-card" href="http://localhost:8766" target="_blank">
    <span class="port">:8766</span>
    <div class="icon">🧠</div>
    <h2>Training Manager</h2>
    <p>Record camera footage → Train LeWM model → Manage checkpoints. One-stop training pipeline.</p>
    <div class="tag" style="background:var(--border);color:var(--text)">TRAINING</div>
  </a>

  <a class="demo-card" href="/report" target="_blank">
    <div class="icon">📊</div>
    <h2>Project Report</h2>
    <p>Full technical report with architecture, training results, VoE evaluation, and inference benchmarks.</p>
    <div class="tag" style="background:var(--border);color:var(--text)">DOCS</div>
  </a>
</div>

<div class="section-title">Quick Start</div>
<div class="grid">
  <div class="demo-card" style="cursor:default">
    <div class="icon">⚡</div>
    <h2>Start All Services</h2>
    <pre style="background:var(--bg);padding:1rem;border-radius:8px;font-size:.8rem;margin-top:.5rem;overflow-x:auto;color:#86efac">cd demos/demo2-voe-anomaly && python3 src/web_demo.py &   # :8765
cd demos/demo2-voe-anomaly && python3 src/web_train.py &  # :8766
cd demos/demo1-trajectory-prediction && python3 src/web_app.py &  # :8770
cd demos/demo3-dual-vision && python3 src/web_app.py &    # :8771
cd demos/demo4-smart-ptz && python3 src/web_app.py &      # :8772
cd demos/demo5-full-system && python3 src/web_app.py &    # :8780</pre>
  </div>
</div>

<div class="footer">
  <p>LeWorldModel Industrial Safety Demo — 2026-03-26</p>
  <p><a href="https://github.com/ddasdkimo/LeWorldModel">GitHub: ddasdkimo/LeWorldModel</a></p>
</div>
<script>function toggleLang(){document.body.classList.toggle('en');var b=document.getElementById('langToggle');b.textContent=document.body.classList.contains('en')?'中文':'EN';localStorage.setItem('lewm_lang',document.body.classList.contains('en')?'en':'zh')}if(localStorage.getItem('lewm_lang')==='en'){document.body.classList.add('en');document.addEventListener('DOMContentLoaded',function(){var b=document.getElementById('langToggle');if(b)b.textContent='中文';})}</script></body>
</html>
""")

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  Demo 5: Full System Dashboard")
    print(f"  http://localhost:8780")
    print(f"{'='*50}\n")
    uvicorn.run(app, host="0.0.0.0", port=8780)
