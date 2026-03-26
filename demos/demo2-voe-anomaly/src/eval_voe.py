"""
Demo 2 - 改進版 VoE 評估
逐幀計算 surprise，而非取 window 平均。
這樣能準確捕捉「哪一幀」發生了異常。
"""
import sys
sys.path.insert(0, "/home/rai/code/le-wm")

import numpy as np
import torch
import stable_pretraining as spt
from module import MLP, Embedder, ARPredictor
from jepa import JEPA
from generate_synthetic_data import generate_moving_block_episode


def build_model(device="cuda"):
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
    return model


def per_frame_surprise(model, frames, actions, device="cuda"):
    """
    逐幀計算 surprise：
    用 [t-3, t-2, t-1, t] 預測 t+1，與實際 t+1 比較。
    回傳每幀的 surprise 值。
    """
    T = frames.shape[0]
    hs = 4  # history_size
    surprises = []

    model.eval()
    with torch.no_grad():
        for t in range(hs, T - 1):
            ctx = torch.from_numpy(frames[t-hs:t]).unsqueeze(0).float().to(device)
            nxt = torch.from_numpy(frames[t:t+1]).unsqueeze(0).float().to(device)
            act = torch.from_numpy(actions[t-hs:t]).unsqueeze(0).float().to(device)

            info_ctx = model.encode({"pixels": ctx, "action": act})
            pred = model.predict(info_ctx["emb"], info_ctx["act_emb"])

            info_nxt = model.encode({"pixels": nxt, "action": act[:, :1]})

            # 只比較最後一步的預測 vs 實際
            s = (pred[:, -1] - info_nxt["emb"][:, 0]).pow(2).mean().item()
            surprises.append(s)

    return surprises


def evaluate(checkpoint, device="cuda", num_tests=50):
    model = build_model(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"已載入: {checkpoint}")

    normal_surprises = []
    teleport_peak = []
    disappear_peak = []
    color_change_peak = []

    for i in range(num_tests):
        pixels, actions = generate_moving_block_episode(
            num_steps=20, img_size=64, seed=20000 + i)

        # --- 正常 ---
        s_normal = per_frame_surprise(model, pixels, actions, device)
        normal_surprises.extend(s_normal)

        # --- 異常A: 瞬移（第 10 幀方塊隨機位置）---
        anom_tp = pixels.copy()
        anom_tp[10] = np.ones_like(anom_tp[10]) * 0.2
        rx, ry = np.random.randint(10, 54, size=2)
        anom_tp[10, :, ry-4:ry+4, rx-4:rx+4] = 0.8
        s_tp = per_frame_surprise(model, anom_tp, actions, device)
        # 擾動在 frame 10，對應的 surprise 在 index 10-hs = 6
        if len(s_tp) > 6:
            teleport_peak.append(s_tp[6])

        # --- 異常B: 消失（第 10 幀方塊消失）---
        anom_dis = pixels.copy()
        anom_dis[10] = np.ones_like(anom_dis[10]) * 0.2
        s_dis = per_frame_surprise(model, anom_dis, actions, device)
        if len(s_dis) > 6:
            disappear_peak.append(s_dis[6])

        # --- 對照: 顏色變化（第 10 幀顏色改但位置不變）---
        anom_clr = pixels.copy()
        anom_clr[10] = 1.0 - anom_clr[10]  # 反轉顏色
        s_clr = per_frame_surprise(model, anom_clr, actions, device)
        if len(s_clr) > 6:
            color_change_peak.append(s_clr[6])

        if (i + 1) % 10 == 0:
            print(f"  進度: {i+1}/{num_tests}")

    # 統計
    normal_mean = np.mean(normal_surprises)
    normal_std = np.std(normal_surprises)
    threshold = normal_mean + 2 * normal_std

    print(f"\n{'='*50}")
    print(f"VoE 逐幀 Surprise 評估 ({num_tests} episodes)")
    print(f"{'='*50}")
    print(f"\n正常基線:")
    print(f"  mean = {normal_mean:.4f} ± {normal_std:.4f}")
    print(f"  動態閾值 (mean + 2σ) = {threshold:.4f}")

    for name, vals in [("瞬移", teleport_peak), ("消失", disappear_peak), ("顏色變化", color_change_peak)]:
        if vals:
            m = np.mean(vals)
            ratio = m / normal_mean
            detect_rate = np.mean([v > threshold for v in vals]) * 100
            print(f"\n{name}:")
            print(f"  peak surprise = {m:.4f} (比正常 {ratio:.2f}x)")
            print(f"  偵測率 (>{threshold:.3f}) = {detect_rate:.0f}%")

    # 判定
    tp_ratio = np.mean(teleport_peak) / normal_mean if teleport_peak else 0
    dis_ratio = np.mean(disappear_peak) / normal_mean if disappear_peak else 0
    clr_ratio = np.mean(color_change_peak) / normal_mean if color_change_peak else 0

    print(f"\n{'='*50}")
    print(f"結論:")
    if dis_ratio > 1.5:
        print(f"  ✅ 消失偵測: {dis_ratio:.2f}x（有效）")
    else:
        print(f"  ⚠️ 消失偵測: {dis_ratio:.2f}x（不足）")

    if tp_ratio > 1.5:
        print(f"  ✅ 瞬移偵測: {tp_ratio:.2f}x（有效）")
    else:
        print(f"  ⚠️ 瞬移偵測: {tp_ratio:.2f}x（不足）")

    if clr_ratio < 1.3:
        print(f"  ✅ 顏色變化不觸發: {clr_ratio:.2f}x（正確，物理理解）")
    else:
        print(f"  ⚠️ 顏色變化誤觸發: {clr_ratio:.2f}x（過敏）")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/demo2_v2/best_model.pt")
    parser.add_argument("--num-tests", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate(args.checkpoint, device, args.num_tests)
