"""
Demo 1 - 軌跡預測：訓練模組
訓練 LeWM + Position Probe + Safety Classifier

1. LeWM: 學場景物理（同 Demo 2）
2. Position Probe: 從 embedding 預測人員位置
3. Safety Classifier: 預測的 embedding 是否進入危險區域
"""
import sys
import os
import time
import json
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

for p in ["/home/rai/code/le-wm", os.path.expanduser("~/code/2026/le-wm-local")]:
    if os.path.exists(p):
        sys.path.insert(0, p)
        break

import stable_pretraining as spt
from module import MLP, Embedder, ARPredictor, SIGReg
from jepa import JEPA


# ============================================================
# Dataset
# ============================================================

class TrajectoryDataset(Dataset):
    def __init__(self, h5_path, window_size=8, augment=True):
        self.window_size = window_size
        self.augment = augment
        with h5py.File(h5_path, "r") as f:
            self.pixels = f["pixels"][:]
            self.actions = f["action"][:]
            self.positions = f["person_positions"][:] if "person_positions" in f else None
            self.counts = f["person_count"][:] if "person_count" in f else None
            self.episode_ends = f["episode_ends"][:]

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
        pixels = self.pixels[start:end].astype(np.float32) / 255.0
        actions = self.actions[start:end]

        if self.augment:
            pixels = pixels + np.random.uniform(-0.3, 0.3)
            pixels = pixels * np.random.uniform(0.7, 1.3)
            for c in range(3):
                pixels[:, c] += np.random.uniform(-0.2, 0.2)
            pixels = np.clip(pixels, 0, 1)

        result = {
            "pixels": torch.from_numpy(pixels.copy()),
            "action": torch.from_numpy(actions),
        }

        if self.positions is not None:
            # 取第一個人的中心點作為位置標籤
            pos = self.positions[start:end, 0, :]  # (T, 4) - first person bbox
            centers = np.zeros((self.window_size, 2), dtype=np.float32)
            for t in range(self.window_size):
                if self.counts[start + t] > 0:
                    centers[t, 0] = (pos[t, 0] + pos[t, 2]) / 2  # center x
                    centers[t, 1] = (pos[t, 1] + pos[t, 3]) / 2  # center y
            result["position"] = torch.from_numpy(centers)

        return result


# ============================================================
# Position Probe
# ============================================================

class PositionProbe(nn.Module):
    """從 embedding 預測人員 2D 位置"""
    def __init__(self, embed_dim=192, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # (x, y) normalized 0-1
            nn.Sigmoid(),
        )

    def forward(self, emb):
        return self.net(emb)


# ============================================================
# Model Builder
# ============================================================

def build_model(device="mps"):
    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=16, image_size=64,
        pretrained=False, use_mask_token=False,
    )
    hd = encoder.config.hidden_size
    ed = 192
    predictor = ARPredictor(num_frames=4, input_dim=ed, hidden_dim=hd,
                            output_dim=hd, depth=2, heads=4, mlp_dim=hd*4, dropout=0.0)
    action_encoder = Embedder(input_dim=2, emb_dim=ed)
    projector = MLP(input_dim=hd, output_dim=ed, hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hd, output_dim=ed, hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    model = JEPA(encoder=encoder, predictor=predictor, action_encoder=action_encoder,
                 projector=projector, pred_proj=pred_proj).to(device)
    return model


# ============================================================
# Training
# ============================================================

def train(data_path, output_dir, epochs=30, batch_size=32, lr=3e-4, device="mps",
          progress_callback=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = TrajectoryDataset(data_path, window_size=8)
    has_positions = dataset.positions is not None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=(device == "cuda"), drop_last=True)

    model = build_model(device)
    sigreg = SIGReg(knots=17, num_proj=256).to(device)
    probe = PositionProbe(192).to(device) if has_positions else None

    params = list(model.parameters())
    if probe:
        params += list(probe.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        if probe:
            probe.train()
        ep_loss = {"total": 0, "pred": 0, "sigreg": 0, "position": 0}
        t0 = time.time()

        for batch in loader:
            pixels = batch["pixels"].to(device)
            actions = torch.nan_to_num(batch["action"].to(device), 0.0)

            info = model.encode({"pixels": pixels, "action": actions})
            emb = info["emb"]
            act_emb = info["act_emb"]

            pred_emb = model.predict(emb[:, :4], act_emb[:, :4])
            tgt_emb = emb[:, 4:]

            pred_loss = (pred_emb - tgt_emb).pow(2).mean()
            sigreg_loss = sigreg(emb.transpose(0, 1))
            loss = pred_loss + sigreg_loss

            # Position probe loss
            pos_loss = torch.tensor(0.0, device=device)
            if probe and "position" in batch:
                positions = batch["position"].to(device)  # (B, T, 2)
                # Probe from embeddings
                B, T, D = emb.shape
                pred_pos = probe(emb.reshape(B * T, D)).reshape(B, T, 2)
                # Only compute loss where person exists (non-zero position)
                mask = (positions.sum(dim=-1) > 0).float()
                if mask.sum() > 0:
                    pos_loss = ((pred_pos - positions).pow(2) * mask.unsqueeze(-1)).sum() / (mask.sum() * 2)
                    loss = loss + 0.1 * pos_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            ep_loss["total"] += loss.item()
            ep_loss["pred"] += pred_loss.item()
            ep_loss["sigreg"] += sigreg_loss.item()
            ep_loss["position"] += pos_loss.item()

        scheduler.step()
        n = len(loader)
        avg = {k: round(v / n, 5) for k, v in ep_loss.items()}
        elapsed = round(time.time() - t0, 1)
        history.append({"epoch": epoch, **avg, "time": elapsed})

        if avg["total"] < best_loss:
            best_loss = avg["total"]
            state = {"model": model.state_dict()}
            if probe:
                state["probe"] = probe.state_dict()
            torch.save(state, out / "best_model.pt")

        if progress_callback:
            progress_callback({
                "epoch": epoch, "total_epochs": epochs,
                "percent": round(epoch / epochs * 100, 1),
                **avg, "best_loss": round(best_loss, 5),
                "epoch_time": elapsed, "history": history,
            })

    state = {"model": model.state_dict()}
    if probe:
        state["probe"] = probe.state_dict()
    torch.save(state, out / "final_model.pt")
    with open(out / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {"checkpoint": str(out), "best_loss": best_loss, "epochs": epochs}
