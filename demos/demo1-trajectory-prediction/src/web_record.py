"""
Demo 1 - 軌跡預測：資料錄製模組
複用 Demo 2 的錄製邏輯，加入 YOLO 人員偵測標註。

錄製時同步跑 YOLO 偵測人員位置，存入 HDF5：
- pixels: (N, C, H, W) 影像
- action: (N, 2) pseudo-action（幀間差異）
- person_positions: (N, max_persons, 4) bbox [x1, y1, x2, y2]
- person_count: (N,) 每幀人數
"""
import sys
import os
import time
import numpy as np
import h5py
import cv2
from pathlib import Path
from datetime import datetime

# Optional YOLO
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("Warning: ultralytics not installed, recording without YOLO annotations")


def record_with_yolo(
    rtsp_url: str,
    duration: int = 300,
    fps: int = 5,
    img_size: int = 64,
    output_path: str = "data/trajectory_train.h5",
    episode_length: int = 30,
    max_persons: int = 10,
    progress_callback=None,
):
    """錄製影片並用 YOLO 標註人員位置"""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    skip = max(1, int(src_fps / fps))

    yolo = YOLO("yolo11n.pt") if HAS_YOLO else None

    all_frames = []
    all_positions = []  # (N, max_persons, 4)
    all_counts = []
    frame_count = 0
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1
        if frame_count % skip != 0:
            continue

        # Original frame for YOLO (before resize)
        orig_h, orig_w = frame.shape[:2]

        # YOLO detection on original frame
        positions = np.zeros((max_persons, 4), dtype=np.float32)
        count = 0
        if yolo is not None:
            results = yolo(frame, classes=[0], verbose=False)  # class 0 = person
            for r in results:
                for box in r.boxes:
                    if count >= max_persons:
                        break
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # Normalize to 0-1
                    positions[count] = [x1/orig_w, y1/orig_h, x2/orig_w, y2/orig_h]
                    count += 1

        # Resize for LeWM
        resized = cv2.resize(frame, (img_size, img_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        all_frames.append(rgb.transpose(2, 0, 1))
        all_positions.append(positions)
        all_counts.append(count)

        if progress_callback:
            elapsed = time.time() - start
            progress_callback({
                "frames": len(all_frames),
                "elapsed": round(elapsed, 1),
                "total": duration,
                "percent": round(elapsed / duration * 100, 1),
                "persons_detected": count,
            })

    cap.release()

    if not all_frames:
        return None

    # Save HDF5
    total = len(all_frames)
    num_ep = total // episode_length
    usable = num_ep * episode_length

    pixels = np.array(all_frames[:usable], dtype=np.uint8)
    positions = np.array(all_positions[:usable], dtype=np.float32)
    counts = np.array(all_counts[:usable], dtype=np.int32)

    # Pseudo-actions
    actions = np.zeros((usable, 2), dtype=np.float32)
    for i in range(1, usable):
        diff = pixels[i].astype(np.float32) - pixels[i-1].astype(np.float32)
        actions[i, 0] = diff.mean()
        actions[i, 1] = np.abs(diff).mean()

    episode_ends = np.arange(episode_length, usable + 1, episode_length, dtype=np.int64)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out, "w") as f:
        f.create_dataset("pixels", data=pixels, compression="gzip", compression_opts=1)
        f.create_dataset("action", data=actions)
        f.create_dataset("person_positions", data=positions)
        f.create_dataset("person_count", data=counts)
        f.create_dataset("episode_ends", data=episode_ends)
        f.attrs["num_episodes"] = num_ep
        f.attrs["num_steps_per_episode"] = episode_length
        f.attrs["img_size"] = img_size
        f.attrs["max_persons"] = max_persons
        f.attrs["source_url"] = rtsp_url
        f.attrs["recorded_at"] = datetime.now().isoformat()
        f.attrs["has_yolo"] = HAS_YOLO

    return {
        "file": str(out),
        "episodes": num_ep,
        "frames": usable,
        "size_mb": round(out.stat().st_size / 1e6, 1),
    }
