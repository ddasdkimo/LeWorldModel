"""
Demo 2 - 真實攝影機 VoE 測試（無 GUI）
從 RTSP 即時拉流，用訓練好的模型計算 surprise，輸出到終端。
"""
import sys
import os

# 自動偵測 le-wm 路徑
for p in ["/home/rai/code/le-wm", os.path.expanduser("~/code/2026/le-wm-local")]:
    if os.path.exists(p):
        sys.path.insert(0, p)
        break

import time
import numpy as np
import cv2
import torch
from collections import deque

import stable_pretraining as spt
from module import MLP, Embedder, ARPredictor
from jepa import JEPA


def load_model(checkpoint, device):
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
    model = JEPA(encoder=encoder, predictor=predictor,
                 action_encoder=action_encoder,
                 projector=projector, pred_proj=pred_proj).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def run_test(rtsp_url, checkpoint, device="mps", duration=60, img_size=64):
    print(f"Device: {device}")
    print(f"載入模型: {checkpoint}")
    model = load_model(checkpoint, device)

    print(f"連線: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("無法連線！")
        return

    history_size = 4
    frame_buffer = deque(maxlen=history_size + 1)
    surprise_values = []
    frame_count = 0
    start = time.time()

    print(f"即時 VoE 測試 {duration} 秒...\n")
    print(f"{'Frame':>6} | {'Surprise':>10} | {'Status':>10} | {'Bar'}")
    print("-" * 55)

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % 5 != 0:  # ~5 fps
            continue

        # Preprocess
        resized = cv2.resize(frame, (img_size, img_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        frame_buffer.append(tensor)

        if len(frame_buffer) < history_size + 1:
            continue

        # Compute surprise
        frames = list(frame_buffer)
        ctx = torch.stack(frames[:history_size]).unsqueeze(0).to(device)
        nxt = torch.stack(frames[1:history_size + 1]).unsqueeze(0).to(device)
        actions = torch.zeros(1, history_size, 2, device=device)

        with torch.no_grad():
            info_ctx = model.encode({"pixels": ctx, "action": actions})
            pred = model.predict(info_ctx["emb"], info_ctx["act_emb"])
            info_nxt = model.encode({"pixels": nxt, "action": actions})
            surprise = (pred - info_nxt["emb"]).pow(2).mean().item()

        surprise_values.append(surprise)

        # Dynamic threshold
        if len(surprise_values) > 20:
            mean_s = np.mean(surprise_values[-100:])
            std_s = np.std(surprise_values[-100:])
            threshold = mean_s + 2 * std_s
            is_anomaly = surprise > threshold
        else:
            threshold = float("inf")
            is_anomaly = False

        # Display
        bar_len = min(50, int(surprise * 500))
        bar = "█" * bar_len
        status = "⚠️ ANOMALY" if is_anomaly else "  normal"
        print(f"{len(surprise_values):6d} | {surprise:10.4f} | {status:>10} | {bar}")

    cap.release()

    # Summary
    vals = np.array(surprise_values)
    print(f"\n{'='*55}")
    print(f"測試完成：{len(vals)} 幀 / {duration} 秒")
    print(f"  mean surprise: {vals.mean():.4f}")
    print(f"  std surprise:  {vals.std():.4f}")
    print(f"  min:           {vals.min():.4f}")
    print(f"  max:           {vals.max():.4f}")
    print(f"  threshold:     {vals.mean() + 2 * vals.std():.4f}")

    anomaly_count = np.sum(vals > vals.mean() + 2 * vals.std())
    print(f"  anomalies:     {anomaly_count} ({anomaly_count/len(vals)*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp", default="rtsp://admin:Ms!23456@116.59.11.189:554/sub")
    parser.add_argument("--checkpoint", default="checkpoints/camera_v2/best_model.pt")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    run_test(args.rtsp, args.checkpoint, args.device, args.duration)
