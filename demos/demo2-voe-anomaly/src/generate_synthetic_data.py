"""
Demo 2 - 合成資料集生成器
產生「移動方塊」的合成影片資料，用於訓練 LeWM 並驗證 VoE surprise。

場景：
- 正常：一個方塊在畫面中勻速移動（可預測的物理行為）
- 異常：方塊突然瞬移/消失/改變方向（違反物理預期）

輸出：HDF5 格式，與 LeWM 訓練格式相容。
"""
import numpy as np
import h5py
from pathlib import Path


def generate_moving_block_episode(
    num_steps: int = 30,
    img_size: int = 64,
    block_size: int = 8,
    speed: float = 2.0,
    seed: int = None,
):
    """
    產生一段「移動方塊」的影片序列。
    回傳 pixels (T, C, H, W) 和 actions (T, 2)。
    """
    rng = np.random.RandomState(seed)

    # 初始位置與速度
    x = rng.uniform(block_size, img_size - block_size)
    y = rng.uniform(block_size, img_size - block_size)
    angle = rng.uniform(0, 2 * np.pi)
    vx = speed * np.cos(angle)
    vy = speed * np.sin(angle)

    # 方塊顏色（固定）
    color = rng.uniform(0.5, 1.0, size=3)

    pixels = np.zeros((num_steps, 3, img_size, img_size), dtype=np.float32)
    actions = np.zeros((num_steps, 2), dtype=np.float32)

    for t in range(num_steps):
        # 畫背景（灰色）
        frame = np.ones((3, img_size, img_size), dtype=np.float32) * 0.2

        # 畫方塊
        x1 = max(0, int(x - block_size // 2))
        y1 = max(0, int(y - block_size // 2))
        x2 = min(img_size, int(x + block_size // 2))
        y2 = min(img_size, int(y + block_size // 2))
        for c in range(3):
            frame[c, y1:y2, x1:x2] = color[c]

        pixels[t] = frame
        actions[t] = [vx, vy]

        # 更新位置（彈性碰撞邊界）
        x += vx
        y += vy
        if x <= block_size // 2 or x >= img_size - block_size // 2:
            vx = -vx
            x = np.clip(x, block_size // 2, img_size - block_size // 2)
        if y <= block_size // 2 or y >= img_size - block_size // 2:
            vy = -vy
            y = np.clip(y, block_size // 2, img_size - block_size // 2)

    return pixels, actions


def generate_dataset(
    output_path: str,
    num_episodes: int = 500,
    num_steps: int = 30,
    img_size: int = 64,
):
    """產生完整的 HDF5 訓練資料集"""
    print(f"產生 {num_episodes} 個 episodes，每個 {num_steps} 步...")

    all_pixels = []
    all_actions = []
    episode_ends = []

    total_steps = 0
    for ep in range(num_episodes):
        pixels, actions = generate_moving_block_episode(
            num_steps=num_steps,
            img_size=img_size,
            seed=ep,
        )
        all_pixels.append(pixels)
        all_actions.append(actions)
        total_steps += num_steps
        episode_ends.append(total_steps)

        if (ep + 1) % 100 == 0:
            print(f"  已產生 {ep + 1}/{num_episodes} episodes")

    # 合併為大陣列
    all_pixels = np.concatenate(all_pixels, axis=0)   # (N, C, H, W)
    all_actions = np.concatenate(all_actions, axis=0)  # (N, 2)
    episode_ends = np.array(episode_ends, dtype=np.int64)

    # 轉為 uint8 像素 (0-255) 以節省空間
    all_pixels_uint8 = (all_pixels * 255).clip(0, 255).astype(np.uint8)

    print(f"\n資料形狀:")
    print(f"  pixels: {all_pixels_uint8.shape} ({all_pixels_uint8.dtype})")
    print(f"  actions: {all_actions.shape} ({all_actions.dtype})")
    print(f"  episode_ends: {episode_ends.shape}")

    # 寫入 HDF5
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("pixels", data=all_pixels_uint8, compression="gzip")
        f.create_dataset("action", data=all_actions)
        f.create_dataset("episode_ends", data=episode_ends)
        f.attrs["num_episodes"] = num_episodes
        f.attrs["num_steps_per_episode"] = num_steps
        f.attrs["img_size"] = img_size
        f.attrs["description"] = "Synthetic moving block dataset for VoE testing"

    file_size_mb = output_path.stat().st_size / 1e6
    print(f"\n已寫入: {output_path} ({file_size_mb:.1f} MB)")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic_train.h5")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--img-size", type=int, default=64)
    args = parser.parse_args()

    generate_dataset(
        output_path=args.output,
        num_episodes=args.episodes,
        num_steps=args.steps,
        img_size=args.img_size,
    )
