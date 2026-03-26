"""
共用的 About 頁面 HTML 生成器
為每個 Demo 生成含中英切換的說明頁面
"""

def get_lang_toggle_css():
    return """
    .lang-btn { position:fixed; top:12px; right:16px; z-index:999; background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.3); color:#fff; padding:4px 14px; border-radius:999px; cursor:pointer; font-size:0.8rem; font-weight:600; backdrop-filter:blur(8px); }
    .lang-btn:hover { background:rgba(255,255,255,0.25); }
    .lang-zh { display:block; } .lang-en { display:none; }
    body.en .lang-zh { display:none; } body.en .lang-en { display:block; }
    """

def get_lang_toggle_js():
    return """
    function toggleLang() {
      document.body.classList.toggle('en');
      const btn = document.getElementById('langToggle');
      btn.textContent = document.body.classList.contains('en') ? '中文' : 'EN';
      localStorage.setItem('lewm_lang', document.body.classList.contains('en') ? 'en' : 'zh');
    }
    // Restore language preference
    if (localStorage.getItem('lewm_lang') === 'en') {
      document.body.classList.add('en');
      document.addEventListener('DOMContentLoaded', () => {
        const btn = document.getElementById('langToggle');
        if (btn) btn.textContent = '中文';
      });
    }
    """

def get_lang_toggle_button():
    return '<button class="lang-btn" id="langToggle" onclick="toggleLang()">EN</button>'


# ============================================================
# About page content for each demo
# ============================================================

ABOUT_PAGES = {
    "demo1": {
        "title_zh": "Demo 1：軌跡預測預警",
        "title_en": "Demo 1: Trajectory Prediction",
        "content_zh": """
        <h2>功能說明</h2>
        <p>本系統結合 <strong>YOLO 人員偵測</strong>與 <strong>LeWM 世界模型</strong>，在人員<strong>尚未進入危險區域前</strong>提前發出預警。</p>

        <h3>核心能力</h3>
        <ul>
            <li><strong>人員追蹤</strong>：YOLO v11 即時偵測畫面中所有人員的位置</li>
            <li><strong>軌跡記錄</strong>：記錄每位人員的移動歷史（黃色實線）</li>
            <li><strong>軌跡預測</strong>：LeWM 在 latent 空間預測未來 3 秒的移動方向（藍色虛線）</li>
            <li><strong>危險預警</strong>：預測軌跡若進入危險區域，提前顯示「⚠️ WARNING: X 秒後進入危險區」</li>
            <li><strong>即時警報</strong>：人員已在危險區域時，顯示「🚨 IN DANGER ZONE!」紅色警報</li>
        </ul>

        <h3>使用流程</h3>
        <ol>
            <li><strong>錄製</strong>：輸入 RTSP 攝影機網址，錄製 5 分鐘以上的場景影片（含 YOLO 人員標註）</li>
            <li><strong>訓練</strong>：選擇錄製好的資料集，設定訓練參數（建議 30 epochs），開始訓練 LeWM + Position Probe</li>
            <li><strong>偵測</strong>：載入訓練好的模型，設定危險區域座標，開始即時偵測</li>
        </ol>

        <h3>危險區域設定</h3>
        <p>在偵測頁面右側可設定危險區域的座標（x1, y1, x2, y2），值為 0-1 的正規化座標。例如：</p>
        <ul>
            <li>右下角：x1=0.6, y1=0.7, x2=1.0, y2=1.0</li>
            <li>左半邊：x1=0.0, y1=0.0, x2=0.5, y2=1.0</li>
        </ul>

        <h3>技術架構</h3>
        <p>攝影機 → YOLO（人員 bbox）→ LeWM Encoder（latent embedding）→ Predictor（預測未來 embedding）→ Position Probe（從 embedding 回推 2D 位置）→ 軌跡外推 → 危險區域碰撞檢測</p>
        """,
        "content_en": """
        <h2>Feature Description</h2>
        <p>This system combines <strong>YOLO person detection</strong> with <strong>LeWM world model</strong> to warn <strong>before</strong> workers enter danger zones.</p>

        <h3>Core Capabilities</h3>
        <ul>
            <li><strong>Person Tracking</strong>: YOLO v11 real-time detection of all persons</li>
            <li><strong>Trajectory Recording</strong>: Movement history for each person (yellow solid line)</li>
            <li><strong>Trajectory Prediction</strong>: LeWM predicts 3-second future path in latent space (blue dotted line)</li>
            <li><strong>Danger Warning</strong>: "⚠️ WARNING: X seconds to danger" when predicted path enters danger zone</li>
            <li><strong>Instant Alert</strong>: "🚨 IN DANGER ZONE!" red alert when person is already in zone</li>
        </ul>

        <h3>Workflow</h3>
        <ol>
            <li><strong>Record</strong>: Enter RTSP URL, record 5+ minutes of scene footage (with YOLO annotations)</li>
            <li><strong>Train</strong>: Select dataset, set parameters (recommend 30 epochs), train LeWM + Position Probe</li>
            <li><strong>Detect</strong>: Load model, set danger zone coordinates, start real-time detection</li>
        </ol>

        <h3>Danger Zone Configuration</h3>
        <p>Set danger zone coordinates (x1, y1, x2, y2) as normalized 0-1 values. Examples:</p>
        <ul>
            <li>Bottom-right: x1=0.6, y1=0.7, x2=1.0, y2=1.0</li>
            <li>Left half: x1=0.0, y1=0.0, x2=0.5, y2=1.0</li>
        </ul>

        <h3>Architecture</h3>
        <p>Camera → YOLO (person bbox) → LeWM Encoder (latent embedding) → Predictor (future embedding) → Position Probe (2D position) → Trajectory extrapolation → Danger zone collision check</p>
        """,
    },
    "demo2": {
        "title_zh": "Demo 2：VoE 物理異常偵測",
        "title_en": "Demo 2: VoE Anomaly Detection",
        "content_zh": """
        <h2>功能說明</h2>
        <p>利用 LeWM 世界模型的 <strong>Violation-of-Expectation（違反預期）</strong>機制，偵測場景中任何違反物理規律的異常事件。</p>

        <h3>核心原理</h3>
        <p>世界模型學習場景的「正常物理規律」後，會預測下一幀的 embedding。當實際觀測與預測有顯著差異時，產生 <strong>Surprise（驚訝值）</strong>。Surprise 超過動態閾值即觸發異常警報。</p>

        <h3>能偵測的異常類型</h3>
        <ul>
            <li><strong>物體消失</strong>：原本存在的物體突然不見（22.4× surprise，94% 偵測率）</li>
            <li><strong>物體瞬移</strong>：物體位置突然跳變（10.3× surprise，68% 偵測率）</li>
            <li><strong>異常運動</strong>：不符合物理規律的移動（如人突然跌倒、設備異常振動）</li>
            <li><strong>場景突變</strong>：任何超出模型預期的變化</li>
        </ul>

        <h3>與 YOLO 的差異</h3>
        <table>
            <tr><th>比較項目</th><th>YOLO</th><th>LeWM VoE</th></tr>
            <tr><td>偵測方式</td><td>視覺模式匹配</td><td>物理規律違反</td></tr>
            <tr><td>需要訓練類別？</td><td>是</td><td>否（自監督學習）</td></tr>
            <tr><td>未知異常</td><td>無法偵測</td><td>可偵測</td></tr>
            <tr><td>推論速度</td><td>~10ms</td><td>~9.6ms</td></tr>
        </table>

        <h3>使用流程</h3>
        <ol>
            <li><strong>錄製</strong>：錄製 5 分鐘「正常場景」影片（自監督，不需標註）</li>
            <li><strong>訓練</strong>：30 epochs，約 37 分鐘（Mac M3）或 25 分鐘（DGX）</li>
            <li><strong>偵測</strong>：即時畫面 + Surprise 波形圖 + 動態閾值 + 異常紅框</li>
        </ol>

        <h3>性能指標</h3>
        <ul>
            <li>模型參數：8.7M</li>
            <li>推論速度：9.6ms / frame（Mac M3 MPS）</li>
            <li>理論最大 FPS：104</li>
            <li>可輕鬆支援 30fps 即時偵測</li>
        </ul>
        """,
        "content_en": """
        <h2>Feature Description</h2>
        <p>Uses LeWM world model's <strong>Violation-of-Expectation (VoE)</strong> mechanism to detect any physics-violating anomaly in the scene.</p>

        <h3>Core Principle</h3>
        <p>The world model learns "normal physics" of a scene and predicts next-frame embeddings. When actual observation significantly differs from prediction, it generates a <strong>Surprise signal</strong>. Surprise exceeding dynamic threshold triggers anomaly alert.</p>

        <h3>Detectable Anomaly Types</h3>
        <ul>
            <li><strong>Object Disappearance</strong>: Object suddenly vanishes (22.4× surprise, 94% detection)</li>
            <li><strong>Object Teleportation</strong>: Object position suddenly jumps (10.3× surprise, 68% detection)</li>
            <li><strong>Abnormal Motion</strong>: Physics-violating movement (falls, equipment vibration)</li>
            <li><strong>Scene Change</strong>: Any change beyond model expectation</li>
        </ul>

        <h3>Comparison with YOLO</h3>
        <table>
            <tr><th>Aspect</th><th>YOLO</th><th>LeWM VoE</th></tr>
            <tr><td>Detection method</td><td>Visual pattern matching</td><td>Physics violation</td></tr>
            <tr><td>Needs training classes?</td><td>Yes</td><td>No (self-supervised)</td></tr>
            <tr><td>Unknown anomalies</td><td>Cannot detect</td><td>Can detect</td></tr>
            <tr><td>Inference speed</td><td>~10ms</td><td>~9.6ms</td></tr>
        </table>

        <h3>Workflow</h3>
        <ol>
            <li><strong>Record</strong>: Record 5 min of "normal scene" footage (self-supervised, no labels needed)</li>
            <li><strong>Train</strong>: 30 epochs, ~37 min (Mac M3) or ~25 min (DGX)</li>
            <li><strong>Detect</strong>: Live feed + Surprise waveform + dynamic threshold + anomaly red border</li>
        </ol>

        <h3>Performance</h3>
        <ul>
            <li>Parameters: 8.7M</li>
            <li>Inference: 9.6ms/frame (Mac M3 MPS)</li>
            <li>Max FPS: 104</li>
            <li>Supports 30fps real-time detection</li>
        </ul>
        """,
    },
    "demo3": {
        "title_zh": "Demo 3：YOLO + LeWM 雙重視野",
        "title_en": "Demo 3: Dual Vision (YOLO + LeWM)",
        "content_zh": """
        <h2>功能說明</h2>
        <p>同一畫面同時展示 <strong>YOLO 物件偵測</strong>和 <strong>LeWM 物理理解</strong>的結果，直觀呈現兩者如何互補。</p>

        <h3>左側：YOLO 偵測</h3>
        <ul>
            <li>即時物件偵測（人、車、物品等）</li>
            <li>每個物件的類別名稱和信心度</li>
            <li>Bounding box 標記</li>
        </ul>

        <h3>右側：LeWM 物理理解</h3>
        <ul>
            <li>Surprise 數值（物理異常指標）</li>
            <li>Surprise 能量條（視覺化）</li>
            <li>異常偵測狀態（紅框 = 異常）</li>
        </ul>

        <h3>互補效果</h3>
        <ul>
            <li><strong>YOLO 強、LeWM 弱</strong>：已知類別的物件識別（如 PPE 合規偵測）</li>
            <li><strong>LeWM 強、YOLO 弱</strong>：未知異常、物理違規、預測性偵測</li>
            <li><strong>結合優勢</strong>：YOLO 偵測到人 + LeWM 預測異常 = 更精確的安全預警</li>
        </ul>

        <h3>注意事項</h3>
        <p>此 Demo 複用 Demo 2 的模型。請先在 Demo 2 中完成訓練，再在此載入模型使用。</p>
        """,
        "content_en": """
        <h2>Feature Description</h2>
        <p>Shows <strong>YOLO object detection</strong> and <strong>LeWM physics understanding</strong> side-by-side to demonstrate how they complement each other.</p>

        <h3>Left Panel: YOLO Detection</h3>
        <ul>
            <li>Real-time object detection (person, car, items, etc.)</li>
            <li>Class name and confidence for each object</li>
            <li>Bounding box overlay</li>
        </ul>

        <h3>Right Panel: LeWM Physics</h3>
        <ul>
            <li>Surprise value (physics anomaly indicator)</li>
            <li>Surprise energy bar (visualization)</li>
            <li>Anomaly status (red border = anomaly)</li>
        </ul>

        <h3>Complementary Effects</h3>
        <ul>
            <li><strong>YOLO strong, LeWM weak</strong>: Known object classification (e.g., PPE compliance)</li>
            <li><strong>LeWM strong, YOLO weak</strong>: Unknown anomalies, physics violations, predictive detection</li>
            <li><strong>Combined</strong>: YOLO detects person + LeWM predicts anomaly = better safety alerts</li>
        </ul>

        <h3>Note</h3>
        <p>This demo reuses Demo 2's model. Please complete training in Demo 2 first, then load the model here.</p>
        """,
    },
    "demo4": {
        "title_zh": "Demo 4：PTZ 智慧自主巡視",
        "title_en": "Demo 4: Smart PTZ Patrol",
        "content_zh": """
        <h2>功能說明</h2>
        <p>利用 LeWM 的 surprise 訊號，讓攝影機<strong>自動決定巡視方向</strong>，優先監看高風險區域。</p>

        <h3>核心演算法：資訊增益</h3>
        <p><code>info_gain = 未觀測時間 × surprise 歷史均值</code></p>
        <p>系統會自動切換到 info_gain 最高的區域，確保：</p>
        <ul>
            <li>長時間未觀測的區域會被優先巡視</li>
            <li>歷史 surprise 高（曾有異常）的區域更常被關注</li>
            <li>兩者結合 = 智慧巡視策略</li>
        </ul>

        <h3>使用流程</h3>
        <ol>
            <li><strong>設定區域</strong>：在 Setup 頁面設定 2-3 個監控區域（可用同一攝影機的不同 preset，或不同攝影機的 RTSP URL）</li>
            <li><strong>載入模型</strong>：選擇 Demo 2 訓練好的 checkpoint</li>
            <li><strong>開始巡視</strong>：開啟 Auto Patrol 模式，系統自動切換區域</li>
        </ol>

        <h3>區域卡片說明</h3>
        <ul>
            <li><strong>Surprise EMA</strong>：該區域的 surprise 指數移動平均</li>
            <li><strong>Last seen</strong>：上次觀測距今多少秒</li>
            <li><strong>Info gain</strong>：資訊增益值（越高越需要觀測）</li>
            <li><strong>Alerts</strong>：累計異常警報次數</li>
            <li><strong>藍色邊框</strong>：當前正在觀測的區域</li>
            <li><strong>紅色閃爍</strong>：該區域曾有異常</li>
        </ul>

        <h3>手動控制</h3>
        <p>點擊任意區域卡片可手動切換（會自動關閉 Auto Patrol），勾選 Auto Patrol 恢復自動模式。</p>
        """,
        "content_en": """
        <h2>Feature Description</h2>
        <p>Uses LeWM surprise signal to let the camera <strong>automatically decide patrol direction</strong>, prioritizing high-risk regions.</p>

        <h3>Core Algorithm: Information Gain</h3>
        <p><code>info_gain = time_since_last_observed × surprise_history_mean</code></p>
        <p>System auto-switches to highest info_gain region:</p>
        <ul>
            <li>Long-unobserved regions get priority</li>
            <li>Regions with history of anomalies get more attention</li>
            <li>Combined = intelligent patrol strategy</li>
        </ul>

        <h3>Workflow</h3>
        <ol>
            <li><strong>Configure Regions</strong>: Set 2-3 monitoring regions in Setup (same camera presets or different RTSP URLs)</li>
            <li><strong>Load Model</strong>: Select Demo 2 trained checkpoint</li>
            <li><strong>Start Patrol</strong>: Enable Auto Patrol, system switches regions automatically</li>
        </ol>

        <h3>Region Card Info</h3>
        <ul>
            <li><strong>Surprise EMA</strong>: Exponential moving average of surprise</li>
            <li><strong>Last seen</strong>: Seconds since last observation</li>
            <li><strong>Info gain</strong>: Information gain value (higher = needs observation)</li>
            <li><strong>Alerts</strong>: Cumulative anomaly alert count</li>
            <li><strong>Blue border</strong>: Currently observed region</li>
            <li><strong>Red flash</strong>: Region had anomalies</li>
        </ul>

        <h3>Manual Control</h3>
        <p>Click any region card to switch manually (disables Auto Patrol). Check Auto Patrol to resume.</p>
        """,
    },
    "demo5": {
        "title_zh": "LeWM 工業安全系統",
        "title_en": "LeWM Industrial Safety System",
        "content_zh": """
        <h2>系統總覽</h2>
        <p>本系統基於 <strong>LeWorldModel (LeWM)</strong> — 一個僅 15M 參數的 JEPA 世界模型，結合 PTZ 攝影機和 YOLO 物件偵測，打造<strong>從事後偵測到事前預防</strong>的下一代工業安全解決方案。</p>

        <h3>五大功能模組</h3>
        <table>
            <tr><th>模組</th><th>Port</th><th>功能</th></tr>
            <tr><td>Demo 1</td><td>:8770</td><td>軌跡預測預警 — 預測人員移動方向，提前警告接近危險區域</td></tr>
            <tr><td>Demo 2</td><td>:8765</td><td>VoE 異常偵測 — 偵測任何違反物理規律的事件</td></tr>
            <tr><td>Demo 3</td><td>:8771</td><td>雙重視野 — YOLO 視覺 + LeWM 物理的互補展示</td></tr>
            <tr><td>Demo 4</td><td>:8772</td><td>PTZ 自主巡視 — 攝影機根據風險自動決定巡視方向</td></tr>
            <tr><td>訓練管理</td><td>:8766</td><td>一站式錄製 → 訓練 → 模型管理</td></tr>
        </table>

        <h3>核心技術</h3>
        <ul>
            <li><strong>LeWM</strong>：8.7M 參數 JEPA 世界模型，從像素端到端訓練</li>
            <li><strong>SIGReg</strong>：防止表徵崩塌的正則化技術</li>
            <li><strong>VoE</strong>：利用預測誤差（surprise）偵測物理異常</li>
            <li><strong>推論速度</strong>：9.6ms（Mac M3），支援 30fps 即時偵測</li>
        </ul>

        <h3>快速開始</h3>
        <ol>
            <li>先啟動 Demo 2 的訓練管理器 (:8766) 錄製場景並訓練模型</li>
            <li>訓練完成後，各 Demo 可共用同一模型</li>
            <li>點擊上方卡片進入各功能模組</li>
        </ol>
        """,
        "content_en": """
        <h2>System Overview</h2>
        <p>Built on <strong>LeWorldModel (LeWM)</strong> — a 15M parameter JEPA world model, combined with PTZ cameras and YOLO detection, creating a <strong>predictive (not just reactive) industrial safety solution</strong>.</p>

        <h3>Five Modules</h3>
        <table>
            <tr><th>Module</th><th>Port</th><th>Function</th></tr>
            <tr><td>Demo 1</td><td>:8770</td><td>Trajectory Prediction — Predict worker path, warn before danger zone entry</td></tr>
            <tr><td>Demo 2</td><td>:8765</td><td>VoE Anomaly Detection — Detect any physics-violating event</td></tr>
            <tr><td>Demo 3</td><td>:8771</td><td>Dual Vision — YOLO visual + LeWM physics comparison</td></tr>
            <tr><td>Demo 4</td><td>:8772</td><td>Smart PTZ Patrol — Camera auto-patrols based on risk</td></tr>
            <tr><td>Training</td><td>:8766</td><td>One-stop Record → Train → Model management</td></tr>
        </table>

        <h3>Core Technology</h3>
        <ul>
            <li><strong>LeWM</strong>: 8.7M parameter JEPA world model, end-to-end from pixels</li>
            <li><strong>SIGReg</strong>: Regularization preventing representation collapse</li>
            <li><strong>VoE</strong>: Physics anomaly detection via prediction error (surprise)</li>
            <li><strong>Inference</strong>: 9.6ms (Mac M3), supports 30fps real-time</li>
        </ul>

        <h3>Quick Start</h3>
        <ol>
            <li>Start Demo 2 Training Manager (:8766), record scene and train model</li>
            <li>After training, all demos share the same model</li>
            <li>Click cards above to enter each module</li>
        </ol>
        """,
    },
    "train": {
        "title_zh": "訓練管理器",
        "title_en": "Training Manager",
        "content_zh": """
        <h2>功能說明</h2>
        <p>一站式完成 LeWM 世界模型的<strong>錄製 → 訓練 → 模型管理</strong>全流程。</p>

        <h3>三步驟流程</h3>
        <ol>
            <li><strong>錄製 (Record)</strong>
                <ul>
                    <li>輸入攝影機 RTSP URL，可先 Preview 預覽畫面</li>
                    <li>設定錄製時長（建議 300 秒以上）和幀率（建議 5 fps）</li>
                    <li>點擊 Start Recording，即時顯示進度條</li>
                    <li>完成後自動轉為 HDF5 訓練格式</li>
                </ul>
            </li>
            <li><strong>訓練 (Train)</strong>
                <ul>
                    <li>選擇已錄製的資料集</li>
                    <li>設定 Epochs（建議 30）、Batch Size（建議 32）、Learning Rate（建議 0.0003）</li>
                    <li>即時顯示 Loss 曲線和訓練 Log</li>
                    <li>訓練時自動啟用 Color Augmentation</li>
                </ul>
            </li>
            <li><strong>模型 (Models)</strong>
                <ul>
                    <li>列出所有訓練完成的 checkpoint</li>
                    <li>顯示 epochs 數和最終 pred_loss</li>
                    <li>模型可被 Demo 1-4 共用</li>
                </ul>
            </li>
        </ol>

        <h3>建議參數</h3>
        <table>
            <tr><th>參數</th><th>建議值</th><th>說明</th></tr>
            <tr><td>錄製時長</td><td>300 秒</td><td>至少 50 episodes（每 30 幀一個 episode）</td></tr>
            <tr><td>FPS</td><td>5</td><td>降低頻寬需求，5fps 足夠捕捉場景變化</td></tr>
            <tr><td>Epochs</td><td>30</td><td>M3 約 37 分鐘，DGX 約 25 分鐘</td></tr>
            <tr><td>Batch Size</td><td>32</td><td>M3 建議 32，DGX 可用 128</td></tr>
        </table>
        """,
        "content_en": """
        <h2>Feature Description</h2>
        <p>One-stop <strong>Record → Train → Model Management</strong> pipeline for LeWM world model.</p>

        <h3>Three-Step Workflow</h3>
        <ol>
            <li><strong>Record</strong>
                <ul>
                    <li>Enter camera RTSP URL, Preview to check feed</li>
                    <li>Set duration (recommend 300+ sec) and FPS (recommend 5)</li>
                    <li>Click Start Recording, real-time progress bar</li>
                    <li>Auto-converts to HDF5 training format</li>
                </ul>
            </li>
            <li><strong>Train</strong>
                <ul>
                    <li>Select recorded dataset</li>
                    <li>Set Epochs (30), Batch Size (32), Learning Rate (0.0003)</li>
                    <li>Live loss curve and training log</li>
                    <li>Color augmentation enabled by default</li>
                </ul>
            </li>
            <li><strong>Models</strong>
                <ul>
                    <li>Lists all trained checkpoints</li>
                    <li>Shows epochs and final pred_loss</li>
                    <li>Models shared across Demo 1-4</li>
                </ul>
            </li>
        </ol>

        <h3>Recommended Parameters</h3>
        <table>
            <tr><th>Parameter</th><th>Recommended</th><th>Note</th></tr>
            <tr><td>Duration</td><td>300 sec</td><td>At least 50 episodes (30 frames each)</td></tr>
            <tr><td>FPS</td><td>5</td><td>Reduces bandwidth, sufficient for scene changes</td></tr>
            <tr><td>Epochs</td><td>30</td><td>~37 min on M3, ~25 min on DGX</td></tr>
            <tr><td>Batch Size</td><td>32</td><td>32 for M3, 128 for DGX</td></tr>
        </table>
        """,
    },
}


def generate_about_html(demo_key):
    """生成含中英切換的 About 頁面 HTML"""
    info = ABOUT_PAGES.get(demo_key, {})
    return f"""
    <style>
        {get_lang_toggle_css()}
        .about-content {{ max-width:800px; margin:0 auto; padding:2rem; }}
        .about-content h2 {{ font-size:1.3rem; margin:1.5rem 0 0.8rem; color:var(--accent,#3b82f6); }}
        .about-content h3 {{ font-size:1.05rem; margin:1.2rem 0 0.5rem; }}
        .about-content p {{ margin-bottom:0.8rem; line-height:1.7; }}
        .about-content ul, .about-content ol {{ margin:0.5rem 0 1rem 1.5rem; }}
        .about-content li {{ margin-bottom:0.3rem; line-height:1.6; }}
        .about-content table {{ width:100%; border-collapse:collapse; margin:0.5rem 0 1rem; }}
        .about-content th, .about-content td {{ padding:8px 12px; border:1px solid var(--border,#334155); text-align:left; font-size:0.9rem; }}
        .about-content th {{ background:rgba(255,255,255,0.05); }}
        .about-content code {{ background:rgba(255,255,255,0.1); padding:2px 6px; border-radius:4px; font-size:0.85rem; }}
    </style>
    <div class="about-content">
        <div class="lang-zh">
            <h1>{info.get('title_zh', '')}</h1>
            {info.get('content_zh', '')}
        </div>
        <div class="lang-en">
            <h1>{info.get('title_en', '')}</h1>
            {info.get('content_en', '')}
        </div>
    </div>
    <script>{get_lang_toggle_js()}</script>
    """
