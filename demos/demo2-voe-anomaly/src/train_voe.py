"""
Demo 2 - VoE 世界模型訓練腳本
用合成資料訓練 LeWM，然後測試 surprise 對正常/異常事件的區分能力。

不依賴 Hydra/Lightning 的完整流程，直接用 PyTorch 訓練迴圈，
方便理解和修改。
"""
import sys
sys.path.insert(0, "/home/rai/code/le-wm")

import time
import json
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import stable_pretraining as spt
from module import SIGReg, MLP, Embedder, ARPredictor
from jepa import JEPA


# ============================================================
# Dataset
# ============================================================

class SyntheticH5Dataset(Dataset):
    """從 HDF5 載入合成資料，產生固定長度的 window"""

    def __init__(self, h5_path: str, window_size: int = 8, augment: bool = True):
        self.h5_path = h5_path
        self.window_size = window_size
        self.augment = augment

        with h5py.File(h5_path, "r") as f:
            self.pixels = f["pixels"][:]       # (N, C, H, W) uint8
            self.actions = f["action"][:]       # (N, 2) float32
            self.episode_ends = f["episode_ends"][:]

        # 建立有效的 window 起始索引（不跨 episode）
        self.valid_starts = []
        ep_start = 0
        for ep_end in self.episode_ends:
            for i in range(ep_start, ep_end - window_size):
                self.valid_starts.append(i)
            ep_start = ep_end

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.window_size

        pixels = self.pixels[start:end].astype(np.float32) / 255.0  # (T, C, H, W)
        actions = self.actions[start:end]  # (T, 2)

        # Color augmentation: 對整個 window 施加相同的 jitter（保持時序一致性）
        if self.augment:
            # brightness
            pixels = pixels + np.random.uniform(-0.3, 0.3)
            # contrast
            pixels = pixels * np.random.uniform(0.7, 1.3)
            # hue-like shift per channel
            for c in range(3):
                pixels[:, c] += np.random.uniform(-0.2, 0.2)
            pixels = np.clip(pixels, 0, 1)

        return {
            "pixels": torch.from_numpy(pixels.copy()),
            "action": torch.from_numpy(actions),
        }


# ============================================================
# Model Builder
# ============================================================

def build_model(img_size=64, patch_size=16, embed_dim=192, action_dim=2,
                frameskip=1, history_size=4, device="cuda"):
    """建構 JEPA 模型"""
    encoder = spt.backbone.utils.vit_hf(
        "tiny",
        patch_size=patch_size,
        image_size=img_size,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size

    predictor = ARPredictor(
        num_frames=history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        depth=2,
        heads=4,
        mlp_dim=hidden_dim * 4,
        dropout=0.0,
    )

    effective_act_dim = frameskip * action_dim
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    projector = MLP(
        input_dim=hidden_dim, output_dim=embed_dim,
        hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d,
    )
    pred_proj = MLP(
        input_dim=hidden_dim, output_dim=embed_dim,
        hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d,
    )

    model = JEPA(
        encoder=encoder, predictor=predictor,
        action_encoder=action_encoder,
        projector=projector, pred_proj=pred_proj,
    ).to(device)

    return model


# ============================================================
# Training
# ============================================================

def train(
    data_path: str,
    output_dir: str = "checkpoints",
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    sigreg_weight: float = 1.0,
    history_size: int = 4,
    num_preds: int = 4,
    device: str = "cuda",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    window_size = history_size + num_preds
    dataset = SyntheticH5Dataset(data_path, window_size=window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    print(f"資料集: {len(dataset)} windows, batch_size={batch_size}")
    print(f"每 epoch {len(loader)} batches")

    # Model
    model = build_model(device=device, history_size=history_size)
    sigreg = SIGReg(knots=17, num_proj=256).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數: {num_params:,} ({num_params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    history = []
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = {"total": 0, "pred": 0, "sigreg": 0}
        t0 = time.time()

        for batch in loader:
            pixels = batch["pixels"].to(device)   # (B, T, C, H, W)
            actions = batch["action"].to(device)   # (B, T, 2)

            # NaN 處理
            actions = torch.nan_to_num(actions, 0.0)

            # Encode
            info = {"pixels": pixels, "action": actions}
            output = model.encode(info)
            emb = output["emb"]          # (B, T, D)
            act_emb = output["act_emb"]  # (B, T, A)

            # Context / Target split
            ctx_emb = emb[:, :history_size]
            ctx_act = act_emb[:, :history_size]
            tgt_emb = emb[:, num_preds:]

            # Predict
            pred_emb = model.predict(ctx_emb, ctx_act)

            # Loss
            pred_loss = (pred_emb - tgt_emb).pow(2).mean()
            sigreg_loss = sigreg(emb.transpose(0, 1))
            loss = pred_loss + sigreg_weight * sigreg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses["total"] += loss.item()
            epoch_losses["pred"] += pred_loss.item()
            epoch_losses["sigreg"] += sigreg_loss.item()

        scheduler.step()
        n = len(loader)
        avg = {k: v / n for k, v in epoch_losses.items()}
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"loss={avg['total']:.4f} pred={avg['pred']:.4f} sigreg={avg['sigreg']:.4f} | "
              f"{elapsed:.1f}s")

        history.append({"epoch": epoch, **avg, "time": elapsed})

        # Save best
        if avg["total"] < best_loss:
            best_loss = avg["total"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    # Save final
    torch.save(model.state_dict(), output_dir / "final_model.pt")

    with open(output_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n訓練完成！最佳 loss: {best_loss:.4f}")
    print(f"模型存於: {output_dir}")
    return model


# ============================================================
# VoE Surprise 測試
# ============================================================

def test_voe(model, device="cuda", img_size=64, num_tests=100):
    """
    測試訓練後模型的 VoE 能力：
    - 正常：連續幀（方塊勻速移動）
    - 異常A：方塊瞬移
    - 異常B：方塊消失
    """
    from generate_synthetic_data import generate_moving_block_episode

    model.eval()
    results = {"normal": [], "teleport": [], "disappear": []}

    history_size = 4  # 必須與訓練時的 history_size 一致

    for i in range(num_tests):
        # 產生正常 episode
        pixels, actions = generate_moving_block_episode(num_steps=10, img_size=img_size, seed=10000 + i)
        T = history_size
        pixels_t = torch.from_numpy(pixels[:T]).unsqueeze(0).float().to(device)   # (1, T, C, H, W)
        pixels_t1 = torch.from_numpy(pixels[1:T+1]).unsqueeze(0).float().to(device)
        acts = torch.from_numpy(actions[:T]).unsqueeze(0).float().to(device)

        with torch.no_grad():
            info_t = model.encode({"pixels": pixels_t, "action": acts})
            info_t1 = model.encode({"pixels": pixels_t1, "action": acts})
            pred_z = model.predict(info_t["emb"], info_t["act_emb"])
            normal_surprise = (pred_z - info_t1["emb"]).pow(2).mean().item()
            results["normal"].append(normal_surprise)

        # 異常A：瞬移（最後一幀的方塊位置隨機改變）
        anomaly_pixels = pixels.copy()
        anomaly_pixels[T] = np.ones_like(anomaly_pixels[T]) * 0.2  # 清空
        rx, ry = np.random.randint(10, img_size - 10, size=2)
        anomaly_pixels[T, :, ry-4:ry+4, rx-4:rx+4] = 0.8
        anomaly_t1 = torch.from_numpy(anomaly_pixels[1:T+1]).unsqueeze(0).float().to(device)

        with torch.no_grad():
            info_anom = model.encode({"pixels": anomaly_t1, "action": acts})
            teleport_surprise = (pred_z - info_anom["emb"]).pow(2).mean().item()
            results["teleport"].append(teleport_surprise)

        # 異常B：消失（最後一幀方塊消失）
        disappear_pixels = pixels.copy()
        disappear_pixels[T] = np.ones_like(disappear_pixels[T]) * 0.2
        disappear_t1 = torch.from_numpy(disappear_pixels[1:T+1]).unsqueeze(0).float().to(device)

        with torch.no_grad():
            info_dis = model.encode({"pixels": disappear_t1, "action": acts})
            disappear_surprise = (pred_z - info_dis["emb"]).pow(2).mean().item()
            results["disappear"].append(disappear_surprise)

    # 統計
    print("\n=== VoE Surprise 測試結果 ===")
    for key in results:
        vals = results[key]
        print(f"  {key:12s}: mean={np.mean(vals):.4f} ± {np.std(vals):.4f}")

    normal_mean = np.mean(results["normal"])
    teleport_mean = np.mean(results["teleport"])
    disappear_mean = np.mean(results["disappear"])

    print(f"\n  瞬移/正常 比率: {teleport_mean / max(normal_mean, 1e-8):.2f}x")
    print(f"  消失/正常 比率: {disappear_mean / max(normal_mean, 1e-8):.2f}x")

    # 判定
    if teleport_mean > normal_mean * 1.5 and disappear_mean > normal_mean * 1.5:
        print(f"\n🎯 VoE 測試通過！模型能區分正常與異常事件。")
        return True
    else:
        print(f"\n⚠️ VoE 區分能力不足，可能需要更多訓練或調參。")
        return False


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_train.h5")
    parser.add_argument("--output", default="checkpoints/demo2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.test_only:
        assert args.checkpoint, "需要指定 --checkpoint"
        model = build_model(device=device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        test_voe(model, device=device)
    else:
        model = train(
            data_path=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        test_voe(model, device=device)
