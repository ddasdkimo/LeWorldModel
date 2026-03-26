# Demo 4: PTZ 自主巡視

**系統整合展示** — PTZ 攝影機根據 LeWM 的場景理解，自主決定巡視策略。

## 展示效果

```
時間軸展示：

T=0s   PTZ 監視 A 區 → LeWM 偵測到 B 區歷史 surprise 偏高
T=1s   PTZ 自動轉向 B 區 → 拉近 zoom 確認狀況
T=5s   B 區恢復正常 → LeWM 評估各區不確定性
T=6s   C 區長時間未觀測（資訊增益最高）→ PTZ 轉向 C 區

對比傳統方式：
A → B → C → D → A → B → ... （固定巡邏，可能錯過事件）
```

## 核心演算法

```python
# 每個區域維護獨立的 LeWM 狀態
regions = {
    "A": {"last_observed": t-2, "avg_surprise": 0.03, "latent_history": [...]},
    "B": {"last_observed": t-5, "avg_surprise": 0.15, "latent_history": [...]},
    "C": {"last_observed": t-12, "avg_surprise": 0.08, "latent_history": [...]},
}

# 資訊增益 = 未觀測時間 × 歷史風險
for region in regions:
    region.info_gain = (t - region.last_observed) * region.avg_surprise

# PTZ 轉向資訊增益最高的區域
target = argmax(regions, key=lambda r: r.info_gain)
ptz.move_to(target.preset_position)
```

## PTZ 控制整合

### 硬體需求
- 支援 ONVIF 協議的 IP PTZ 攝影機
- 預先定義每個監控區域的 PTZ preset 位置（pan, tilt, zoom）

### 軟體介面
```
python-onvif-zeep  →  ONVIF PTZ 控制
    │
    ├── AbsoluteMove(pan, tilt, zoom)    # 轉到指定角度
    ├── GetPresets()                       # 取得預設位置
    ├── GotoPreset(preset_token)          # 跳到預設位置
    └── GetStatus()                        # 取得當前角度
```

## 實作步驟

### Phase 1: PTZ 控制基礎 (Week 1-2)
- [ ] 選購 ONVIF 相容 PTZ 攝影機
- [ ] 用 python-onvif-zeep 連線測試
- [ ] 定義 3-5 個監控區域的 preset 位置
- [ ] 實作 PTZ 平滑轉動（避免突然跳轉造成 LeWM 狀態斷裂）

### Phase 2: 多區域 LeWM 狀態管理 (Week 2-4)
- [ ] 為每個區域維護獨立的 latent history
- [ ] PTZ 轉到某區域時更新該區 LeWM 狀態
- [ ] 實作跨區域的 surprise 累積與衰減
- [ ] 處理 PTZ 轉動中的過渡幀（丟棄或特殊處理）

### Phase 3: 自主巡視策略 (Week 4-6)
- [ ] 實作 info_gain 計算
- [ ] 實作 PTZ 決策邏輯（每 N 秒評估一次轉向）
- [ ] 加入「事件鎖定」模式：高 surprise 時停留不轉
- [ ] 加入「最低巡視頻率」：確保每個區域至少每 M 秒觀測一次

### Phase 4: 展示與評估 (Week 6-8)
- [ ] 對比實驗：固定巡邏 vs 智慧巡視 的覆蓋率/偵測率
- [ ] 模擬「在 PTZ 轉向其他區域時發生事件」的場景
- [ ] 錄製展示影片

## 關鍵技術挑戰

### 1. PTZ 轉動時的表徵斷裂
- PTZ 轉向時影像突變，LeWM 的 latent 連續性被打斷
- 解法：每區域獨立 latent history，轉回時從上次狀態續接

### 2. 轉動延遲
- 機械式 PTZ 轉動需 0.5-2 秒到位
- 決策頻率不能太高（每次轉動都有成本）
- 建議：至少在每個區域停留 5-10 秒再評估轉向

### 3. Zoom 控制策略
- 遠景（廣角）：覆蓋大範圍，用於巡視
- 近景（拉近）：確認細節，用於事件鎖定
- LeWM surprise 高 → 自動拉近確認

## 前置依賴

- Demo 2 的 VoE / surprise 計算管線
- ONVIF PTZ 攝影機硬體

## 成功指標

- [ ] 覆蓋率：每個區域的最大未觀測時間 < 30 秒
- [ ] 事件偵測率：比固定巡邏提升 >= 20%
- [ ] PTZ 轉動決策延遲 < 1 秒
- [ ] 無無效轉動（轉過去什麼都沒有又立刻轉走）
