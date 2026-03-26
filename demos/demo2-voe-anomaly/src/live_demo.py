"""
Demo 2 - VoE 即時展示腳本
從攝影機（或合成影片）即時計算 surprise 並視覺化。

用法:
  # 用合成移動方塊展示
  python live_demo.py --source synthetic

  # 用 webcam
  python live_demo.py --source webcam

  # 用 RTSP 串流
  python live_demo.py --source rtsp://192.168.1.100/stream

  # 用預錄影片
  python live_demo.py --source video.mp4
"""
import sys
sys.path.insert(0, "/home/rai/code/le-wm")

import time
import numpy as np
import torch
import cv2
from collections import deque
from pathlib import Path

import stable_pretraining as spt
from module import MLP, Embedder, ARPredictor
from jepa import JEPA


# ============================================================
# Model
# ============================================================

def load_model(checkpoint_path: str, device="cuda"):
    """載入訓練好的 JEPA 模型"""
    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=16, image_size=64,
        pretrained=False, use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = 192

    predictor = ARPredictor(
        num_frames=4, input_dim=embed_dim, hidden_dim=hidden_dim,
        output_dim=hidden_dim, depth=2, heads=4,
        mlp_dim=hidden_dim * 4, dropout=0.0,
    )
    action_encoder = Embedder(input_dim=2, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)

    model = JEPA(
        encoder=encoder, predictor=predictor,
        action_encoder=action_encoder,
        projector=projector, pred_proj=pred_proj,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"模型已載入: {checkpoint_path}")
    return model


# ============================================================
# Surprise Engine
# ============================================================

class SurpriseEngine:
    """即時 surprise 計算引擎"""

    def __init__(self, model, history_size=4, img_size=64, device="cuda"):
        self.model = model
        self.history_size = history_size
        self.img_size = img_size
        self.device = device

        self.frame_buffer = deque(maxlen=history_size + 1)
        self.surprise_history = deque(maxlen=300)  # 最近 300 個 surprise 值
        self.prev_emb = None
        self.prev_pred = None

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """BGR frame → (C, H, W) float tensor"""
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        return tensor

    def compute_surprise(self) -> float:
        """用 buffer 中的幀計算 surprise"""
        if len(self.frame_buffer) < self.history_size + 1:
            return 0.0

        frames = list(self.frame_buffer)
        # context frames: [0:history_size], next frame: [history_size]
        ctx_frames = torch.stack(frames[:self.history_size]).unsqueeze(0).to(self.device)
        next_frame = torch.stack(frames[1:self.history_size + 1]).unsqueeze(0).to(self.device)

        # 用零向量作為 action（監控場景無明確 action）
        actions = torch.zeros(1, self.history_size, 2, device=self.device)

        with torch.no_grad():
            # Encode context
            info_ctx = self.model.encode({"pixels": ctx_frames, "action": actions})
            # Predict next embedding
            pred_emb = self.model.predict(info_ctx["emb"], info_ctx["act_emb"])
            # Encode actual next
            info_next = self.model.encode({"pixels": next_frame, "action": actions})
            # Surprise
            surprise = (pred_emb - info_next["emb"]).pow(2).mean().item()

        self.surprise_history.append(surprise)
        return surprise

    def add_frame(self, frame: np.ndarray) -> float:
        """加入新幀並回傳 surprise"""
        tensor = self.preprocess_frame(frame)
        self.frame_buffer.append(tensor)
        return self.compute_surprise()

    def get_threshold(self) -> float:
        """動態閾值: mean + 2*std"""
        if len(self.surprise_history) < 30:
            return float("inf")
        vals = list(self.surprise_history)
        return np.mean(vals) + 2 * np.std(vals)


# ============================================================
# Visualization
# ============================================================

def draw_overlay(frame, surprise, threshold, surprise_history, fps):
    """在畫面上繪製 surprise 資訊"""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 狀態判定
    is_anomaly = surprise > threshold and threshold < float("inf")
    status_text = "ANOMALY!" if is_anomaly else "Normal"
    status_color = (0, 0, 255) if is_anomaly else (0, 200, 0)

    # 邊框
    if is_anomaly:
        cv2.rectangle(overlay, (0, 0), (w-1, h-1), (0, 0, 255), 4)

    # 狀態文字
    cv2.putText(overlay, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

    # Surprise 數值
    cv2.putText(overlay, f"Surprise: {surprise:.3f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(overlay, f"Threshold: {threshold:.3f}" if threshold < float("inf") else "Threshold: calibrating...",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Surprise 曲線（底部）
    bar_h = 80
    bar_y = h - bar_h - 10
    cv2.rectangle(overlay, (10, bar_y), (w - 10, h - 10), (40, 40, 40), -1)

    if len(surprise_history) > 1:
        vals = list(surprise_history)
        max_val = max(vals) * 1.2 if max(vals) > 0 else 1.0
        points = []
        bar_w = w - 20
        for j, v in enumerate(vals):
            x = 10 + int(j / len(vals) * bar_w)
            y = h - 10 - int(v / max_val * bar_h)
            y = max(bar_y, min(h - 10, y))
            points.append((x, y))

        for j in range(1, len(points)):
            color = (0, 0, 255) if vals[j] > threshold else (0, 200, 0)
            cv2.line(overlay, points[j-1], points[j], color, 2)

        # 閾值線
        if threshold < float("inf"):
            th_y = h - 10 - int(threshold / max_val * bar_h)
            th_y = max(bar_y, min(h - 10, th_y))
            cv2.line(overlay, (10, th_y), (w - 10, th_y), (0, 200, 255), 1)

    return overlay


# ============================================================
# Synthetic Source (for demo without camera)
# ============================================================

class SyntheticSource:
    """產生合成移動方塊影片，可注入異常事件"""

    def __init__(self, img_size=320, block_size=40):
        self.img_size = img_size
        self.block_size = block_size
        self.x = img_size // 2
        self.y = img_size // 2
        self.vx = 3.0
        self.vy = 2.0
        self.color = (0, 100, 255)  # Orange
        self.frame_count = 0
        self.anomaly_mode = None
        self.anomaly_until = 0

    def trigger_anomaly(self, mode="teleport"):
        """觸發異常事件"""
        self.anomaly_mode = mode
        self.anomaly_until = self.frame_count + 10

    def read(self):
        """模擬 cv2.VideoCapture.read()"""
        frame = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 50

        # 更新位置
        self.x += self.vx
        self.y += self.vy
        bs = self.block_size // 2
        if self.x <= bs or self.x >= self.img_size - bs:
            self.vx = -self.vx
        if self.y <= bs or self.y >= self.img_size - bs:
            self.vy = -self.vy
        self.x = np.clip(self.x, bs, self.img_size - bs)
        self.y = np.clip(self.y, bs, self.img_size - bs)

        # 異常處理
        if self.anomaly_mode and self.frame_count < self.anomaly_until:
            if self.anomaly_mode == "teleport":
                draw_x = np.random.randint(bs, self.img_size - bs)
                draw_y = np.random.randint(bs, self.img_size - bs)
            elif self.anomaly_mode == "disappear":
                draw_x, draw_y = -100, -100  # 畫面外
            else:
                draw_x, draw_y = int(self.x), int(self.y)
        else:
            draw_x, draw_y = int(self.x), int(self.y)
            self.anomaly_mode = None

        # 畫方塊
        x1, y1 = draw_x - bs, draw_y - bs
        x2, y2 = draw_x + bs, draw_y + bs
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, -1)

        # 畫軌跡提示
        cv2.putText(frame, f"Frame: {self.frame_count}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        if self.anomaly_mode:
            cv2.putText(frame, f"INJECTED: {self.anomaly_mode}", (5, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        self.frame_count += 1
        return True, frame

    def release(self):
        pass


# ============================================================
# Main Loop
# ============================================================

def run_demo(source="synthetic", checkpoint="checkpoints/demo2_v2/best_model.pt",
             device="cuda", display_size=640):
    """主展示迴圈"""

    # 載入模型
    model = load_model(checkpoint, device)
    engine = SurpriseEngine(model, device=device)

    # 開啟影像源
    if source == "synthetic":
        cap = SyntheticSource()
        print("使用合成影像源（按 't' 觸發瞬移, 'd' 觸發消失, 'r' 重置）")
    elif source == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    print("按 'q' 離開, 's' 截圖")

    frame_times = deque(maxlen=30)

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # 計算 surprise
        surprise = engine.add_frame(frame)
        threshold = engine.get_threshold()

        # FPS
        frame_times.append(time.time() - t0)
        fps = len(frame_times) / sum(frame_times) if frame_times else 0

        # 繪製
        display = cv2.resize(frame, (display_size, display_size))
        display = draw_overlay(display, surprise, threshold,
                               engine.surprise_history, fps)

        cv2.imshow("LeWM VoE Demo", display)

        # 鍵盤控制
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"screenshot_{int(time.time())}.png", display)
            print("截圖已儲存")
        elif key == ord('t') and isinstance(cap, SyntheticSource):
            cap.trigger_anomaly("teleport")
            print("注入異常：瞬移")
        elif key == ord('d') and isinstance(cap, SyntheticSource):
            cap.trigger_anomaly("disappear")
            print("注入異常：消失")
        elif key == ord('r') and isinstance(cap, SyntheticSource):
            cap.anomaly_mode = None
            print("重置為正常")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="synthetic")
    parser.add_argument("--checkpoint", default="checkpoints/demo2_v2/best_model.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_demo(source=args.source, checkpoint=args.checkpoint, device=args.device)
