# psnr 계산 코드입니다.
import math
import torch
import torch.fft

def get_psnr(pred, target, max_pixel = 1.0):
    mse = torch.mean((pred - target)**2)
    if mse == 0:
        return 100
    return 20 * math.log10(max_pixel / math.sqrt(mse))