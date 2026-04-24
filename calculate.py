import os
import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# ====================== 路径配置 ======================
GT_PATH = r"lol_dataset\eval15\high"
TEST_FOLDERS = [
    r"lol_dataset\eval15\cnn_enhanced",
    r"lpdm\test\denoised\eval\lpdm_lol\phi300_s30",
    r"lol_dataset\eval15\low"
]
FOLDER_NAMES = ["CNN增强", "LPDM增强", "低光原图"]

# ====================== 初始化 LPIPS ======================
loss_fn = lpips.LPIPS(net='alex')  # 轻量、速度快
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = loss_fn.to(device)

# ====================== 指标计算 ======================
def calculate_metrics(gt_img, test_img):
    gt = gt_img.astype(np.float64)
    test = test_img.astype(np.float64)

    # 1. PSNR
    psnr_val = psnr(gt, test, data_range=255)

    # 2. SSIM
    ssim_val = ssim(gt, test, channel_axis=2, data_range=255)

    # 3. NSR
    noise = test - gt
    nsr_val = np.mean(noise ** 2) / (np.mean(gt ** 2) + 1e-8)

    # 4. MAE
    mae_val = np.mean(np.abs(gt - test))

    return psnr_val, ssim_val, nsr_val, mae_val

# ====================== LPIPS 专用计算 ======================
def calculate_lpips(img1, img2):
    # 转为 [-1,1] 格式
    img1 = (img1 / 255.0) * 2 - 1
    img2 = (img2 / 255.0) * 2 - 1

    # HWC -> CHW
    img1 = img1.transpose(2, 0, 1)
    img2 = img2.transpose(2, 0, 1)

    # 转 tensor
    t1 = torch.from_numpy(img1).float().unsqueeze(0).to(device)
    t2 = torch.from_numpy(img2).float().unsqueeze(0).to(device)

    with torch.no_grad():
        lpips_val = loss_fn(t1, t2).item()
    return lpips_val

# ====================== 评估文件夹 ======================
def evaluate_folder(gt_dir, test_dir, test_name):
    print(f"\n===== 正在评估：{test_name} =====")
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('png', 'jpg', 'jpeg'))])
    test_files = set(os.listdir(test_dir))

    total_psnr = 0.0
    total_ssim = 0.0
    total_nsr = 0.0
    total_mae = 0.0
    total_lpips = 0.0
    count = 0

    for filename in tqdm(gt_files, desc=test_name):
        if filename not in test_files:
            continue

        gt_path = os.path.join(gt_dir, filename)
        test_path = os.path.join(test_dir, filename)

        gt_img = cv2.imread(gt_path)
        test_img = cv2.imread(test_path)

        if gt_img is None or test_img is None:
            continue

        # 自动统一尺寸到 GT 大小
        h, w = gt_img.shape[:2]
        test_img = cv2.resize(test_img, (w, h), interpolation=cv2.INTER_CUBIC)

        # 计算基础指标
        p, s, n, m = calculate_metrics(gt_img, test_img)
        # 计算 LPIPS
        lp = calculate_lpips(gt_img, test_img)

        total_psnr += p
        total_ssim += s
        total_nsr += n
        total_mae += m
        total_lpips += lp
        count += 1

    if count == 0:
        print(f"{test_name} 无有效图片")
        return

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_nsr = total_nsr / count
    avg_mae = total_mae / count
    avg_lpips = total_lpips / count

    print(f"[{test_name}] 结果")
    print(f"PSNR:   {avg_psnr:.4f} dB")
    print(f"SSIM:   {avg_ssim:.4f}")
    print(f"NSR:    {avg_nsr:.6f}")
    print(f"MAE:    {avg_mae:.4f}")
    print(f"LPIPS:  {avg_lpips:.4f}")
    print(f"计算图片数: {count}")

# ====================== 主程序 ======================
if __name__ == "__main__":
    print("===== 统一尺寸 | PSNR / SSIM / NSR / MAE / LPIPS 全指标评估 =====")
    for i, folder in enumerate(TEST_FOLDERS):
        evaluate_folder(GT_PATH, folder, FOLDER_NAMES[i])
    print("\n✅ 全部计算完成！")