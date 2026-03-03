# HQS PnP의 Deblur 성능 측정을 위한 코드입니다.
# PSNR 측정 및 시각화 함수 있습니다.
import argparse
import torch
import numpy as np
import os
import random  
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import DIV2KBlur
from models import NAFNet, HQS_PnP 
from pathlib import Path
from utils import get_psnr

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='HQS-PnP Validation')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--sigma', type=float, default=5)
    parser.add_argument('--valid-dir', type=str, default='./DIV2K/DIV2K_valid_HR')
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--save-limit', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=4)
    parser.add_argument('--kernel-idx', type=int, default=0)
    parser.add_argument('--rho-min', type=float, default=-3.0)
    parser.add_argument('--rho-max', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42) 
    return parser.parse_args()

def save_comparison(save_dir, idx, gt, blur, restored, psnr_restored):
    psnr_noisy = get_psnr(blur, gt)

    def to_np(tensor):
        return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    gt_np = to_np(gt)
    blur_np = to_np(blur)
    restored_np = to_np(restored)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    axes[0].imshow(gt_np)
    axes[0].set_title("Ground Truth", fontsize=15)
    axes[0].axis('off')

    axes[1].imshow(blur_np)
    axes[1].set_title(f"Blurry | PSNR: {psnr_noisy:.2f} dB", fontsize=15)
    axes[1].axis('off')

    axes[2].imshow(restored_np)
    axes[2].set_title(f"Restored | PSNR: {psnr_restored:.2f} dB", fontsize=15)
    axes[2].axis('off')

    save_path = os.path.join(save_dir, f"result_{idx}_PSNR_{psnr_restored:.2f}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()

    seed_everything(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = DIV2KBlur(args.valid_dir, sigma=args.sigma, kernel_idx=args.kernel_idx)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = NAFNet(img_channel=4, width=32).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    
    # models.py에 HQS_PnP 있습니다.
    hqs_pnp = HQS_PnP(model, iterations=args.iterations, rho_range=(args.rho_min, args.rho_max))

    val_save_dir = os.path.join(
        "./deblur_validate",
        f"sigma_{int(args.sigma)}",
        f"iter_{int(args.iterations)}",
        f"kernelIdx_{int(args.kernel_idx)}"
    )
    os.makedirs(val_save_dir, exist_ok=True)

    psnr_values = []
    psnr_noisy_values = [] 

    print(f"HQS-PnP 검증 시작 (Sigma: {args.sigma}, Kernel idx: {args.kernel_idx}, Seed: {args.seed})")

    with torch.no_grad():
        for i, (x_t, y_t, m_t, kernel_t) in enumerate(dataloader):
            if i >= args.num_samples: break
            
            x_t, y_t, m_t, kernel_t = x_t.to(device), y_t.to(device), m_t.to(device), kernel_t.to(device)

            output = hqs_pnp.solve(y_t, kernel_t.squeeze(0), m_t)
            
            psnr = get_psnr(output, x_t)
            psnr_values.append(psnr)
            
            psnr_noisy = get_psnr(y_t, x_t)
            psnr_noisy_values.append(psnr_noisy)
            
            if i < args.save_limit:
                save_comparison(val_save_dir, i, x_t, y_t, output, psnr)
            
            print(f"({i+1}/{args.num_samples}) Noisy: {psnr_noisy:.2f} dB | Restored: {psnr:.2f} dB", end='\r')

    print(f"\n{'='*40}")
    print(f" [최종 결과 요약]")
    print(f"{'='*40}")
    print(f" 1. Noisy (Blurry) Image")
    print(f"    - 평균 PSNR: {np.mean(psnr_noisy_values):.2f} dB")
    print(f"    - 표준 편차: {np.std(psnr_noisy_values):.2f} dB")
    print(f"{'-'*40}")
    print(f" 2. Restored (HQS-PnP) Image")
    print(f"    - 평균 PSNR: {np.mean(psnr_values):.2f} dB")
    print(f"    - 표준 편차: {np.std(psnr_values):.2f} dB")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()