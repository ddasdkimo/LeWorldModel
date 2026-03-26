"""
Demo 2 - VoE Surprise 機制驗證
用合成資料測試 LeWM 的 encoder + predictor 是否能產生有意義的 surprise 訊號。
"""
import sys
sys.path.insert(0, "/home/rai/code/le-wm")

import torch
import torch.nn.functional as F
import stable_pretraining as spt
from module import SIGReg, MLP, Embedder, ARPredictor
from jepa import JEPA


def build_jepa(device="cuda"):
    """依照 train.py 的方式建構 JEPA 模型"""
    img_size = 64
    patch_size = 16
    history_size = 4
    embed_dim = 192
    action_dim = 2
    frameskip = 1

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
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=512,
        norm_fn=torch.nn.BatchNorm1d,
    )

    pred_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=512,
        norm_fn=torch.nn.BatchNorm1d,
    )

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    ).to(device)

    return model


def compute_surprise(model, frames_t, frames_t1, actions):
    """
    計算 surprise = MSE(predicted_embedding, actual_embedding)
    frames_t:  (B, T, C, H, W) 當前幀序列
    frames_t1: (B, T, C, H, W) 下一幀序列
    actions:   (B, T, action_dim)
    """
    model.eval()
    with torch.no_grad():
        # Encode 當前幀
        info_t = {"pixels": frames_t, "action": actions}
        info_t = model.encode(info_t)
        z_t = info_t["emb"]          # (B, T, D)
        act_emb = info_t["act_emb"]  # (B, T, A)

        # Predict 下一步 embedding
        pred_z = model.predict(z_t, act_emb)  # (B, T, D)

        # Encode 實際下一幀
        info_t1 = {"pixels": frames_t1, "action": actions}
        info_t1 = model.encode(info_t1)
        z_t1 = info_t1["emb"]  # (B, T, D)

        # Surprise = MSE per sample per timestep
        surprise = (pred_z - z_t1).pow(2).mean(dim=-1)  # (B, T)

    return surprise


def test_pipeline():
    """測試整個 surprise pipeline"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. 建立模型
    print("\n[1/4] 建立 JEPA 模型...")
    model = build_jepa(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  參數量: {num_params:,} ({num_params/1e6:.1f}M)")

    # 2. 產生合成資料
    print("\n[2/4] 產生合成測試資料...")
    B, T, C, H, W = 4, 4, 3, 64, 64
    action_dim = 2

    # 正常：連續幀（微小差異）
    base = torch.randn(B, 1, C, H, W, device=device)
    normal_t = base.expand(B, T, C, H, W) + torch.randn(B, T, C, H, W, device=device) * 0.01
    normal_t1 = base.expand(B, T, C, H, W) + torch.randn(B, T, C, H, W, device=device) * 0.01

    # 異常：下一幀完全不同
    anomaly_t = base.expand(B, T, C, H, W) + torch.randn(B, T, C, H, W, device=device) * 0.01
    anomaly_t1 = torch.randn(B, T, C, H, W, device=device)

    actions = torch.randn(B, T, action_dim, device=device) * 0.1

    # 3. 計算 surprise
    print("\n[3/4] 計算 surprise...")
    normal_surprise = compute_surprise(model, normal_t, normal_t1, actions)
    anomaly_surprise = compute_surprise(model, anomaly_t, anomaly_t1, actions)

    # 4. 結果
    print(f"\n[4/4] 結果（未訓練模型，驗證 pipeline）")
    print(f"  正常場景 surprise: mean={normal_surprise.mean():.4f}, std={normal_surprise.std():.4f}")
    print(f"  異常場景 surprise: mean={anomaly_surprise.mean():.4f}, std={anomaly_surprise.std():.4f}")
    ratio = anomaly_surprise.mean() / max(normal_surprise.mean(), 1e-8)
    print(f"  差異倍率: {ratio:.2f}x")

    print(f"\n=== Pipeline 驗證 ===")
    checks = [
        ("模型建立", True),
        ("Encoder 前向推論", True),
        ("Predictor 前向推論", True),
        ("Surprise 計算", normal_surprise.shape == (B, T)),
        ("Surprise > 0", normal_surprise.mean() > 0),
    ]
    all_pass = True
    for name, ok in checks:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")
        all_pass = all_pass and ok

    if all_pass:
        print(f"\n🎯 Pipeline 完整驗證通過！")
        print(f"   注意：未訓練模型的正常/異常差異不一定明顯。")
        print(f"   訓練後 surprise 應能明確區分正常與異常事件。")
    else:
        print(f"\n❌ 有檢查未通過，需排查。")

    return all_pass


if __name__ == "__main__":
    success = test_pipeline()
    exit(0 if success else 1)
