# 系統架構設計

## 整體架構

```
PTZ 攝影機 ──→ 影像串流 (RTSP/ONVIF)
                  │
          ┌───────┴───────┐
          ▼               ▼
     YOLO v11         LeWM Encoder
    (即時偵測)        (latent 壓縮)
     │ bbox/cls          │ z_t (192-dim)
     │                   │
     │              LeWM Predictor
     │              (latent rollout)
     │                   │ ẑ_{t+1:t+H}
     │                   │
     ▼                   ▼
  ┌──────────────────────────┐
  │     Safety Fusion Layer   │
  │  • YOLO 偵測 + LeWM 預測  │
  │  • Surprise > θ → 異常    │
  │  • Predicted risk → 預警   │
  │  • PTZ action optimizer   │
  └──────────────────────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
 警報系統    PTZ 控制指令
```

## 軟體堆疊

| 層 | 技術 |
|---|---|
| 偵測 | YOLOv11 (ultralytics) |
| 追蹤 | ByteTrack / BoT-SORT |
| 世界模型 | LeWM (le-wm) |
| PTZ 控制 | python-onvif-zeep |
| 影像串流 | OpenCV + RTSP / DeepStream |
| 後端 | FastAPI |
| 前端 | Streamlit (原型) → React (產品) |
| 推論加速 | TensorRT / ONNX Runtime |

## 資料流

```
Camera Frame (H×W×3, ~30fps)
    │
    ├─→ YOLO: frame → [bbox, class, conf] (~10ms)
    │
    ├─→ LeWM Encoder: frame → z_t (192-dim) (~5ms)
    │       │
    │       ├─→ Surprise: MSE(ẑ_t, z_t) → scalar
    │       │
    │       └─→ Predictor: (z_t, action) → ẑ_{t+1:t+H}
    │               │
    │               └─→ Safety Classifier: ẑ → risk_level
    │                       │
    │                       └─→ Position Probe: ẑ → (x,y) predicted
    │
    └─→ Fusion: (yolo_results, surprise, risk, trajectory) → decision
            │
            ├─→ Alert (if risk > threshold)
            └─→ PTZ Command (if info_gain warrants rotation)
```

## 多區域狀態管理（PTZ 場景）

```python
class RegionState:
    region_id: str
    preset_position: (pan, tilt, zoom)
    latent_history: deque[Tensor]  # 最近 H 步的 z_t
    last_observed: float           # 上次觀測時間戳
    surprise_ema: float            # surprise 指數移動平均
    alert_count: int               # 累計警報次數

class MultiRegionManager:
    regions: dict[str, RegionState]
    current_region: str

    def compute_info_gain(self, region) -> float:
        elapsed = now() - region.last_observed
        return elapsed * region.surprise_ema

    def decide_next_region(self) -> str:
        return max(self.regions, key=self.compute_info_gain)
```
