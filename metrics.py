# metrics.py
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ===============================
# tensor → numpy
# ===============================
def tensor_to_numpy(img):
    img = img.squeeze(0).detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0, 1)

# ===============================
# PSNR
# ===============================
def calc_psnr(pred, target):
    pred = tensor_to_numpy(pred)
    target = tensor_to_numpy(target)
    return peak_signal_noise_ratio(target, pred, data_range=1.0)

# ===============================
# SSIM
# ===============================
def calc_ssim(pred, target):
    pred = tensor_to_numpy(pred)
    target = tensor_to_numpy(target)

    return structural_similarity(
        target,
        pred,
        channel_axis=2,
        data_range=1.0
    )

# ===============================
# NSR（新增）
# ===============================
def calc_nsr(pred, target):
    pred = tensor_to_numpy(pred)
    target = tensor_to_numpy(target)

    noise = pred - target

    noise_power = (noise ** 2).mean()
    signal_power = (target ** 2).mean()

    return noise_power / (signal_power + 1e-8)