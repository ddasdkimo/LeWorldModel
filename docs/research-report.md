# LeWorldModel 深度研究報告

## 論文資訊

- **標題**: LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels
- **作者**: Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero
- **機構**: Mila & Université de Montréal, NYU, Samsung SAIL, Brown University
- **arXiv**: [2603.19312](https://arxiv.org/abs/2603.19312)
- **提交日期**: 2026-03-13 (v1), 2026-03-24 (v2)
- **GitHub**: [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) (MIT License, 828+ stars)
- **專案首頁**: [le-wm.github.io](https://le-wm.github.io/)

## 核心貢獻

LeWM 是第一個能從像素端到端穩定訓練的 JEPA 世界模型，不依賴 EMA / stop-gradient / 凍結預訓練 encoder / 多項複雜 loss。

| 指標 | 數值 |
|------|------|
| 參數量 | ~15M (ViT-Tiny ~5M + Transformer predictor ~10M) |
| latent 維度 | 192-dim (單一 CLS token / frame) |
| 可調超參數 | 1 個 (λ, SIGReg 權重) |
| 訓練需求 | 單 GPU (L40S), 數小時 |
| 規劃速度 | <1 秒 (vs DINO-WM ~47 秒 = 48× 加速) |
| Push-T 成功率 | 96% (比 PLDM 高 18%) |

## 架構

```
像素 o_t → ViT Encoder (CLS token) → Projector MLP → z_t
動作 a_t → Action Encoder (Embedder) → act_emb
(z_t, act_emb) → Transformer Predictor (AdaLN-zero) → Pred-proj MLP → ẑ_{t+1}

Loss = MSE(ẑ_{t+1}, z_{t+1}) + λ × SIGReg(z_{0:T})
```

## SIGReg

- 全稱: Sketched Isotropic Gaussian Regularization
- 來源: [LeJEPA 論文](https://arxiv.org/abs/2511.08544)
- 原理: 用 Cramér-Wold 定理 + Epps-Pulley 統計量，強制 embedding 分佈趨近各向同性高斯
- 實作: 隨機 1024 個投影方向，17 個 knots

## 推論流程 (Goal-conditioned MPC + CEM)

```
(1) Encode(goal_image) → goal_emb
(2) CEM 產生 300 組 action sequences (30 次迭代)
(3) 每組 rollout → predicted_emb_{t+1:t+H}
(4) cost = MSE(pred_last, goal_emb)
(5) 更新 CEM 分佈 → 輸出最佳 action
```

## 評測任務

| 任務 | Episodes | Horizon | 類型 |
|------|----------|---------|------|
| PushT | 33k | 25 | 2D 推物 |
| TwoRoom | 10k | 100 | 2D 導航 |
| OGBench-Cube | 313 | 25 | 3D 方塊操控 |
| Reacher | 2000 | 25 | 機械臂到達 |

## 已知限制

1. 長視野誤差累積（horizon 25-100 步）
2. 旋轉/姿態資訊較難編碼
3. 需要 action 標註（無法從自然影片學習）
4. 低多樣性環境可能正則化效果變差
5. 僅在模擬環境驗證（未有 sim-to-real）

## 事實查證結果 (2026-03-26)

所有 9 項核心宣稱均經查證確認，無事實錯誤。
詳見完整查證報告。
