# Demo 1: 軌跡預測預警

**主力展示項目** — 在人員尚未進入危險區域前，提前 2-5 秒發出預警。

## 展示效果

```
┌─────────────────────────────────┐
│  [攝影機畫面]                     │
│                                   │
│   🟢 工人 A（安全）               │
│        ↓ 移動方向                 │
│   🟡 預測軌跡（虛線）  ⚠️ 2.3秒後 │
│        ↓                         │
│   ═══════════════                │
│   🔴 危險區域（機台運作中）        │
│                                   │
│  ┌──────────────────┐            │
│  │ YOLO: 工人偵測 ✅  │            │
│  │ LeWM: 預測碰撞 ⚠️ │            │
│  │ 預估碰撞: 2.3 秒   │            │
│  └──────────────────┘            │
└─────────────────────────────────┘
```

## 技術架構

```
攝影機影像串流
    │
    ├── YOLO v11 ──→ 人員 bbox + 中心點座標序列
    │
    ├── LeWM Encoder ──→ z_t (192-dim latent)
    │       │
    │       ▼
    │   LeWM Predictor ──→ ẑ_{t+1} ... ẑ_{t+H} (predicted embeddings)
    │       │
    │       ▼
    │   Safety Classifier (MLP) ──→ 安全 / 警告 / 危險
    │       │
    │       ▼
    │   Position Probe (Linear) ──→ 預測位置 (x, y) → 畫虛線軌跡
    │
    └── Fusion Layer ──→ 警報決策 + UI 渲染
```

## 實作步驟

### Phase 1: 資料收集 (Week 1-2)
- [ ] 拍攝 50-100 段室內場景「人員走動」影片（含多種路徑）
- [ ] 定義危險區域（多邊形座標）
- [ ] 格式轉換為 LeWM 所需的 HDF5 格式（pixels + pseudo-actions）

### Phase 2: 模型訓練 (Week 2-3)
- [ ] 訓練 LeWM 學習場景物理動態（自監督，不需標註）
- [ ] 訓練 Safety Classifier on predicted embeddings
- [ ] 訓練 Position Probe（線性 probe 預測 2D 位置）
- [ ] 評估 prediction horizon: 最遠能準確預測幾步？

### Phase 3: 即時推論管線 (Week 3-5)
- [ ] 整合 YOLO 人員偵測 + LeWM 軌跡預測
- [ ] 實作即時 UI（畫面疊加預測軌跡虛線 + 警報框）
- [ ] 延遲測試：目標 < 50ms end-to-end

### Phase 4: 展示調校 (Week 5-6)
- [ ] 調校警報閾值（precision/recall 平衡）
- [ ] 邊緣案例處理（多人交叉、遮擋、光線變化）
- [ ] 錄製展示影片

## 關鍵技術問題

### Action 定義
LeWM 需要 action 作為 predictor 的條件輸入。在監控場景中，「action」可定義為：
- **選項 A**: YOLO bbox 的幀間位移 (dx, dy) → 作為 pseudo-action
- **選項 B**: 光流 (optical flow) 的區域平均
- **選項 C**: 無 action（設為零向量），僅靠 latent dynamics 預測

建議先試選項 A（最簡單），效果不好再換 B。

### Prediction Horizon
- LeWM 在模擬環境中 horizon 為 25 步
- 監控場景中 25 幀 ≈ ~1 秒（@25fps），可能不夠
- 需要實測：降低 fps（如 5fps）+ 25 步 = 5 秒預測窗口

## 所需資源

| 項目 | 規格 | 用途 |
|------|------|------|
| 攝影機 | webcam 或 IP cam | 影像擷取 |
| GPU（訓練） | RTX 3060+ 或 L40S | 訓練 LeWM + classifier |
| GPU（推論） | RTX 3060 / Jetson Orin Nano | 即時推論 |
| 場地 | 可定義「危險區域」的室內空間 | 拍攝 + 展示 |

## 成功指標

- [ ] 預警提前時間 >= 2 秒
- [ ] 預測命中率 >= 80%
- [ ] 誤報率 < 10%
- [ ] 端到端延遲 < 50ms
- [ ] 可 live demo（非預錄）
