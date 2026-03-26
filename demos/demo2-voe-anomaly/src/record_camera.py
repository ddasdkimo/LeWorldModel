"""
Demo 2 - 真實攝影機錄製腳本
從 RTSP 串流錄製影片，並轉為 HDF5 訓練資料。

用法:
  # 錄製 5 分鐘（預設）
  python record_camera.py --rtsp "rtsp://admin:Ms!23456@116.59.11.189:554/main"

  # 錄製 10 分鐘，每秒取 5 幀
  python record_camera.py --rtsp "..." --duration 600 --fps 5

  # 錄完後自動轉為 HDF5
  python record_camera.py --rtsp "..." --output data/office_train.h5
"""
import cv2
import numpy as np
import h5py
import time
import argparse
from pathlib import Path


def record_rtsp(rtsp_url: str, duration: int = 300, target_fps: int = 5,
                img_size: int = 64, output_h5: str = "data/camera_train.h5",
                episode_length: int = 30):
    """
    從 RTSP 錄製並直接轉為 HDF5 訓練資料。

    Args:
        rtsp_url: RTSP 串流 URL
        duration: 錄製秒數
        target_fps: 目標幀率（從原始串流抽幀）
        img_size: 輸出影像大小
        output_h5: HDF5 輸出路徑
        episode_length: 每個 episode 的幀數
    """
    print(f"連線: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        # 嘗試 TCP transport
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        if not cap.isOpened():
            print("無法連線到攝影機！")
            return None

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(src_fps / target_fps))
    print(f"原始 FPS: {src_fps}, 抽幀間隔: {frame_interval}, 目標 FPS: {target_fps}")
    print(f"錄製 {duration} 秒...")

    all_frames = []
    frame_count = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print("串流中斷，嘗試重連...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            continue

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        # Resize + RGB
        frame_resized = cv2.resize(frame, (img_size, img_size))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        # (H, W, C) -> (C, H, W)
        frame_chw = frame_rgb.transpose(2, 0, 1)
        all_frames.append(frame_chw)

        elapsed = time.time() - start_time
        if len(all_frames) % 100 == 0:
            print(f"  {elapsed:.0f}s / {duration}s | {len(all_frames)} 幀已擷取")

    cap.release()
    total_frames = len(all_frames)
    print(f"\n錄製完成：{total_frames} 幀 ({total_frames / target_fps:.0f}s)")

    if total_frames < episode_length:
        print("幀數不足，請增加錄製時間！")
        return None

    # 切成 episodes
    num_episodes = total_frames // episode_length
    usable_frames = num_episodes * episode_length
    pixels = np.array(all_frames[:usable_frames], dtype=np.uint8)

    # Pseudo-actions: 用連續幀的 pixel 差異近似（僅 x, y 方向平均位移）
    # 更好的做法是用光流，但這裡先用簡單近似
    actions = np.zeros((usable_frames, 2), dtype=np.float32)
    for i in range(1, usable_frames):
        diff = pixels[i].astype(np.float32) - pixels[i-1].astype(np.float32)
        # 簡單的 x/y 位移估計（用差異的質心偏移）
        actions[i, 0] = diff.mean()  # 整體亮度變化作為 proxy
        actions[i, 1] = np.abs(diff).mean()  # 變動量

    episode_ends = np.arange(episode_length, usable_frames + 1, episode_length, dtype=np.int64)

    # 寫 HDF5
    output_path = Path(output_h5)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("pixels", data=pixels, compression="gzip", compression_opts=1)
        f.create_dataset("action", data=actions)
        f.create_dataset("episode_ends", data=episode_ends)
        f.attrs["num_episodes"] = num_episodes
        f.attrs["num_steps_per_episode"] = episode_length
        f.attrs["img_size"] = img_size
        f.attrs["source_fps"] = target_fps
        f.attrs["source_url"] = rtsp_url
        f.attrs["description"] = "Real camera footage for VoE training"

    file_size_mb = output_path.stat().st_size / 1e6
    print(f"\n已寫入: {output_path}")
    print(f"  {num_episodes} episodes × {episode_length} steps")
    print(f"  pixels: {pixels.shape}")
    print(f"  檔案大小: {file_size_mb:.1f} MB")

    return output_path


def preview_rtsp(rtsp_url: str):
    """預覽 RTSP 串流"""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("無法連線！")
        return

    print("預覽中... 按 'q' 離開, 's' 截圖")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = cv2.resize(frame, (640, 360))
        cv2.imshow("Camera Preview", display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"preview_{int(time.time())}.jpg", frame)
            print("截圖已儲存")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp", default="rtsp://admin:Ms!23456@116.59.11.189:554/main")
    parser.add_argument("--duration", type=int, default=300, help="錄製秒數")
    parser.add_argument("--fps", type=int, default=5, help="目標幀率")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--output", default="data/camera_train.h5")
    parser.add_argument("--episode-length", type=int, default=30)
    parser.add_argument("--preview", action="store_true", help="僅預覽不錄製")
    args = parser.parse_args()

    if args.preview:
        preview_rtsp(args.rtsp)
    else:
        record_rtsp(
            rtsp_url=args.rtsp,
            duration=args.duration,
            target_fps=args.fps,
            img_size=args.img_size,
            output_h5=args.output,
            episode_length=args.episode_length,
        )
