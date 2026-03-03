# NAFNet에서 Denoising만 성능 측정하기 위한 코드입니다.
"""
사용법:
    valid.py --model-path ./experiments/weight_folder_name/best_model.pth --sigma 50
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
from pathlib import Path

import matplotlib.pyplot as plt
from models import NAFNet 
from utils import get_psnr

def parse_args():
    parser = argparse.ArgumentParser(description='NAFNet Validation')
    parser.add_argument('--model-path', type=str, required=True, help='학습된 NAFNet .pth 경로')
    parser.add_argument('--sigma', type=float, required=True, help='테스트 노이즈 강도 (0-255)')
    parser.add_argument('--valid-dir', type=str, default='./DIV2K/DIV2K_valid_HR')
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--save-limit', type=int, default=100)
    return parser.parse_args()

def get_fixed_sample(img_path, patch_size, sigma, seed_idx):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    h, w, _ = img.shape
    
    # 중앙 크롭
    th, tw = patch_size, patch_size
    top = (h - th) // 2
    left = (w - tw) // 2
    x = img[top:top+th, left:left+tw, :]
    
    # 고정 노이즈 추가
    np.random.seed(seed_idx)
    noise = (sigma / 255.0) * np.random.randn(th, tw, 3).astype(np.float32)
    y = x + noise
    
    # 노이즈 맵 생성 (NAFNet의 4번째 채널 입력용)
    m = np.ones((th, tw, 1), dtype=np.float32) * (sigma / 255.0)
    
    def to_tensor(array):
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    y_tensor = to_tensor(y)
    m_tensor = to_tensor(m)
    input_tensor = torch.cat([y_tensor, m_tensor], dim=1)

    return to_tensor(x), input_tensor, y_tensor

def save_comparison(save_dir, filename, gt, noisy, restored, psnr):
    def to_np(tensor):
        return tensor[0].permute(1, 2, 0).cpu().numpy() # .clip(0, 1)

    gt_np = to_np(gt)
    noisy_np = to_np(noisy)
    restored_np = to_np(restored)
    
    h, w = gt_np.shape[:2]
    separator = np.ones((h, 5, 3), dtype=np.float32)
    
    # 이미지 합치기
    combined = np.hstack([gt_np, separator, noisy_np, separator, restored_np])
    
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.imshow(combined)
    ax.set_title(f"{filename} | PSNR: {psnr:.2f} dB", fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # 저장
    save_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_PSNR_{psnr:.2f}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # NAFNet 모델 로드
    print(f"\n[1] NAFNet 로드 중: {args.model_path}")
    model = NAFNet(img_channel=4, width=32).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 저장 폴더
    val_save_dir = os.path.join(os.path.dirname(args.model_path), f"nafnet_val_sigma_{int(args.sigma)}")
    os.makedirs(val_save_dir, exist_ok=True)

    img_list = sorted([f for f in os.listdir(args.valid_dir) if f.lower().endswith(('.png', '.jpg'))])
    num_test = min(len(img_list), args.num_samples)

    # 검증 루프
    psnr_values = []
    print(f"[2] NAFNet 단독 검증 시작 (Sigma: {args.sigma})")

    with torch.no_grad():
        for i in range(num_test):
            img_name = img_list[i]
            gt, net_input, noisy_only = get_fixed_sample(os.path.join(args.valid_dir, img_name), 
                                                           args.patch_size, args.sigma, i)
            
            gt, net_input = gt.to(device), net_input.to(device)
            
            output = model(net_input)
            
            psnr = get_psnr(output, gt)
            psnr_values.append(psnr)
            
            if i < args.save_limit:
                save_comparison(val_save_dir, img_name, gt, noisy_only, output, psnr)
            
            print(f"    ({i+1}/{num_test}) {img_name} | PSNR: {psnr:.2f} dB", end='\r')

    mean_psnr = np.mean(psnr_values)
    std_psnr = np.std(psnr_values)

    print(f"\n\n========================================")
    print(f" 최종 결과 요약")
    print(f"========================================")
    print(f" 테스트 샘플 수 : {len(psnr_values)}")
    print(f" 평균 PSNR      : {mean_psnr:.2f} dB")
    print(f" 표준편차 (Std) : {std_psnr:.2f} dB") 
    print(f" 결과 (Mean±Std): {mean_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"========================================")
    print(f" 결과 이미지 저장: {val_save_dir}")

if __name__ == "__main__":
    main()