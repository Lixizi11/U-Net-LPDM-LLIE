import os
import cv2
import numpy as np
import torch
import lpips
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# =========================
# 配置路径
# =========================
pred_dirs = {
    "lpdm": "phi300_s30",
    "cnn": "lol_dataset\our485\cnn_enhanced"
}
gt_dir = "lol_dataset/our485/high"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LPIPS 初始化（官方）
# =========================
lpips_fn = lpips.LPIPS(net='alex').to(device)

# =========================
# 指标函数
# =========================
def calc_metrics(img_pred, img_gt):
    # 转 float
    img_pred = img_pred.astype(np.float32) / 255.0
    img_gt   = img_gt.astype(np.float32) / 255.0

    # PSNR
    psnr = peak_signal_noise_ratio(img_gt, img_pred, data_range=1.0)

    # SSIM（多通道）
    ssim = structural_similarity(img_gt, img_pred, channel_axis=2, data_range=1.0)

    # MAE
    mae = np.mean(np.abs(img_gt - img_pred))

    # NSR（Noise-to-Signal Ratio）
    noise = img_gt - img_pred
    nsr = np.sum(noise ** 2) / np.sum(img_gt ** 2 + 1e-8)

    # LPIPS（官方输入 [-1,1]）
    pred_t = torch.from_numpy(img_pred).permute(2,0,1).unsqueeze(0).to(device)*2-1
    gt_t   = torch.from_numpy(img_gt).permute(2,0,1).unsqueeze(0).to(device)*2-1

    with torch.no_grad():
        lp = lpips_fn(pred_t, gt_t).item()

    return psnr, ssim, nsr, mae, lp

# =========================
# 主评测函数
# =========================
def evaluate(pred_dir, gt_dir):
    files = sorted(os.listdir(gt_dir))

    psnr_list, ssim_list, nsr_list, mae_list, lpips_list = [], [], [], [], []

    for f in tqdm(files):
        gt_path = os.path.join(gt_dir, f)
        pred_path = os.path.join(pred_dir, f)

        if not os.path.exists(pred_path):
            continue

        gt = cv2.imread(gt_path)
        pred = cv2.imread(pred_path)

        # BGR → RGB
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

        # 尺寸对齐（保险）
        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        psnr, ssim, nsr, mae, lp = calc_metrics(pred, gt)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        nsr_list.append(nsr)
        mae_list.append(mae)
        lpips_list.append(lp)

    return {
        "PSNR": np.mean(psnr_list),
        "SSIM": np.mean(ssim_list),
        "NSR":  np.mean(nsr_list),
        "MAE":  np.mean(mae_list),
        "LPIPS": np.mean(lpips_list)
    }

# =========================
# 执行评测
# =========================
for name, pred_dir in pred_dirs.items():
    print(f"\n==== Evaluating {name} ====")
    results = evaluate(pred_dir, gt_dir)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")