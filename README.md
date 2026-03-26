# LeWorldModel Industrial Safety Demo

基於 [LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm) 的工業安全智慧監控系統，結合 PTZ 攝影機 + YOLO 物件偵測 + JEPA 世界模型，實現**從事後偵測到事前預防**的下一代工安解決方案。

## 核心理念

| 傳統方案 (YOLO only) | 本專案 (YOLO + LeWM) |
|---|---|
| 偵測「正在發生」的違規 | 預測「即將發生」的危險 |
| 純視覺模式匹配 | 物理感知 + 時序推理 |
| 只認識訓練過的類別 | 能偵測未知類型的物理異常 |
| PTZ 固定巡邏 | PTZ 自主判斷該看哪裡 |

## 技術基礎

- **LeWorldModel**: ~15M 參數的 JEPA 世界模型，單 GPU 數小時可訓練，規劃速度 <1 秒
- **SIGReg**: Sketched Isotropic Gaussian Regularization，防止表徵崩塌
- **VoE (Violation of Expectation)**: 利用 surprise 訊號偵測物理異常

## Demo 項目

| Demo | 名稱 | 難度 | 開發時間 | 狀態 |
|------|------|------|----------|------|
| [Demo 1](demos/demo1-trajectory-prediction/) | 軌跡預測預警 | ⭐⭐⭐ | 4-6 週 | 📋 規劃中 |
| [Demo 2](demos/demo2-voe-anomaly/) | VoE 物理異常偵測 | ⭐⭐ | 3-5 週 | 📋 規劃中 |
| [Demo 3](demos/demo3-dual-vision/) | YOLO + LeWM 雙重視野 | ⭐⭐⭐ | 2-3 週* | 📋 規劃中 |
| [Demo 4](demos/demo4-smart-ptz/) | PTZ 自主巡視 | ⭐⭐⭐⭐ | 6-8 週 | 📋 規劃中 |
| [Demo 5](demos/demo5-full-system/) | 完整系統整合 | ⭐⭐⭐⭐⭐ | 4-6 週* | 📋 規劃中 |

\* 基於前置 Demo 完成後的增量開發時間

## 建議開發順序

```
Demo 2 (VoE) → Demo 1 (軌跡預測) → Demo 3 (雙重視野) → Demo 4 (PTZ) → Demo 5 (整合)
最快驗證        主力展示            技術論文素材        系統整合         展會/客戶版
```

## 硬體需求

### 最低配置 (~$630 USD)
- 攝影機: webcam 或 IP cam
- GPU: NVIDIA RTX 3060 或 Jetson Orin Nano ($199)

### 完整配置 (~$1,030 USD)
- PTZ 攝影機: Hikvision DS-2DE4A425IWG-E 或同級 ONVIF 相容機型
- 邊緣 GPU: NVIDIA Jetson Orin Nano
- 顯示器 + PoE 交換器

## 專案結構

```
LeWorldModel/
├── README.md
├── docs/                          # 研究文件與技術文檔
│   ├── research-report.md         # LeWM 深度研究報告
│   ├── feasibility-analysis.md    # 可行性分析
│   └── architecture.md            # 系統架構設計
├── demos/
│   ├── demo1-trajectory-prediction/   # Demo 1: 軌跡預測預警
│   ├── demo2-voe-anomaly/             # Demo 2: VoE 異常偵測
│   ├── demo3-dual-vision/             # Demo 3: 雙重視野
│   ├── demo4-smart-ptz/               # Demo 4: PTZ 自主巡視
│   └── demo5-full-system/             # Demo 5: 完整系統
├── shared/                        # 跨 Demo 共用模組
│   ├── models/                    # LeWM / YOLO 模型封裝
│   ├── utils/                     # 共用工具
│   └── data/                      # 共用資料處理
└── scripts/                       # 環境設定、資料下載等腳本
```

## 相關資源

- [LeWorldModel 論文](https://arxiv.org/abs/2603.19312)
- [LeWM GitHub](https://github.com/lucas-maes/le-wm)
- [LeWM 專案首頁](https://le-wm.github.io/)
- [LeJEPA / SIGReg 論文](https://arxiv.org/abs/2511.08544)

## License

MIT
