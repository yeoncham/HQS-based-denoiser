"""
compares.py - GT, Noisy, Pretrained, Random-Init NAFNet 결과 비교 (Patch Version)
"""
# patch 단위 결과 비교용 코드입니다.
import argparse
import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from models import NAFNet
from utils import get_psnr

def parse_args():
    parser = argparse.ArgumentParser(description='Compare Pretrained vs Random-Init NAFNet')
    parser.add_argument('--pretrained-path', type=str, required=True, help='Pretrained 기반 학습 모델 경로')
    parser.add_argument('--random-path', type=str, required=True, help='Random Init 기반 학습 모델 경로')
    parser.add_argument('--sigma', type=float, default=50.0, help='테스트 노이즈 강도')
    parser.add_argument('--valid-dir', type=str, default='./DIV2K/DIV2K_valid_HR')
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--num-samples', type=int, default=100, help='시각화할 샘플 개수')
    parser.add_argument('--output-dir', type=str, default='./comparison_results')
    parser.add_argument('--no-clamp', action='store_true', help='사용 시 PSNR 계산에서 0~1 Clamping을 해제합니다. (분석용)')
    
    return parser.parse_args()

def load_model(path, device):
    model = NAFNet(img_channel=4, width=32).to(device)
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_data(img_path, patch_size, sigma, seed_idx):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    h, w, _ = img.shape
    
    if h < patch_size or w < patch_size:
        patch_size = min(h, w)
        
    top = (h - patch_size) // 2
    left = (w - patch_size) // 2
    gt = img[top:top+patch_size, left:left+patch_size, :]
    
    np.random.seed(seed_idx)
    noise = (sigma / 255.0) * np.random.randn(patch_size, patch_size, 3).astype(np.float32)
    noisy = gt + noise
    
    m = np.ones((patch_size, patch_size, 1), dtype=np.float32) * (sigma / 255.0)
    
    to_tensor = lambda x: torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    
    gt_t = to_tensor(gt)
    noisy_t = to_tensor(noisy)
    input_t = torch.cat([noisy_t, to_tensor(m)], dim=1)
    
    return gt_t, noisy_t, input_t

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mode_str = "no_clamp" if args.no_clamp else "clamped"
    save_dir = os.path.join(args.output_dir, f"sigma_{int(args.sigma)}_{mode_str}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading Pretrained model: {args.pretrained_path}")
    model_pre = load_model(args.pretrained_path, device)
    
    print(f"Loading Random-Init model: {args.random_path}")
    model_ran = load_model(args.random_path, device)

    img_list = sorted([f for f in os.listdir(args.valid_dir) if f.lower().endswith(('.png', '.jpg'))])
    if not img_list:
        print(f"[Error] 이미지를 찾을 수 없습니다: {args.valid_dir}")
        return

    num_test = min(len(img_list), args.num_samples)

    print(f"\nComparing {num_test} samples at Sigma {args.sigma}...")
    print(f"PSNR Mode: {'[ANALYSIS] Raw Output (No Clamp)' if args.no_clamp else '[STANDARD] 0~1 Clamping'}")
    print(f"Visualization: {'Simple Min-Max Scaling (No Clipping)' if args.no_clamp else 'Standard Clipping (0-1)'}")

    psnr_pre_list = []
    psnr_ran_list = []
    psnr_noisy_list = []

    with torch.no_grad():
        for i in range(num_test):
            gt, noisy, net_input = get_data(os.path.join(args.valid_dir, img_list[i]), 
                                            args.patch_size, args.sigma, i)
            
            net_input = net_input.to(device)
            gt = gt.to(device)
            noisy = noisy.to(device)
            
            # 모델 추론 (Raw Output)
            out_pre_raw = model_pre(net_input)
            out_ran_raw = model_ran(net_input)
            
            # PSNR 계산용 데이터 준비 (옵션에 따라 Clamp 적용)
            if args.no_clamp:
                eval_pre = out_pre_raw
                eval_ran = out_ran_raw
                eval_noisy = noisy
                vis_prefix = "[Raw]"
            else:
                eval_pre = out_pre_raw.clamp(0, 1)
                eval_ran = out_ran_raw.clamp(0, 1)
                eval_noisy = noisy.clamp(0, 1)
                vis_prefix = ""

            # PSNR 계산
            psnr_pre = get_psnr(eval_pre, gt)
            psnr_ran = get_psnr(eval_ran, gt)
            psnr_noisy = get_psnr(eval_noisy, gt)

            psnr_pre_list.append(psnr_pre)
            psnr_ran_list.append(psnr_ran)
            psnr_noisy_list.append(psnr_noisy)

            # 시각화
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            def show(ax, tensor, title, psnr_val=None):
                img_np = tensor[0].permute(1, 2, 0).cpu().numpy()
                
                if args.no_clamp:
                    v_max = img_np.max()
                    v_min = img_np.min()
                    
                    img_vis = (img_np - v_min) / (v_max - v_min)
                    
                    ax.imshow(img_vis)
                    
                    title += f"\n[Raw View]\nRange: {v_min:.2f} ~ {v_max:.2f}"
                else:
                    ax.imshow(np.clip(img_np, 0, 1))
                
                #if psnr_val is not None:
                #    title += f"\n({vis_prefix} PSNR: {psnr_val:.2f} dB)"
                    
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.axis('off')

            show(axes[0], gt, "\nGround Truth")
            show(axes[1], noisy, f"\nNoisy Input (PSNR: {psnr_noisy:.2f} dB)", psnr_noisy) 
            show(axes[2], out_pre_raw, f"\nFine-Tuned (PSNR: {psnr_pre:.2f} dB)", psnr_pre) 
            show(axes[3], out_ran_raw, f"\nRandom Init (PSNR: {psnr_ran:.2f} dB)", psnr_ran) 

            plt.tight_layout()
            save_name = f"compare_{os.path.splitext(img_list[i])[0]}_s{int(args.sigma)}.png"
            plt.savefig(os.path.join(save_dir, save_name), dpi=150)
            plt.close()
            
            print(f"[{i+1}/{num_test}] Saved: {save_name} | Pre: {psnr_pre:.2f} | Ran: {psnr_ran:.2f}")

    mean_pre = np.mean(psnr_pre_list)
    std_pre = np.std(psnr_pre_list)
    
    mean_ran = np.mean(psnr_ran_list)
    std_ran = np.std(psnr_ran_list)
    
    mean_noisy = np.mean(psnr_noisy_list)
    std_noisy = np.std(psnr_noisy_list)
    
    print("\n" + "="*50)
    print(f"  [Final Results (Mean ± Std)] (Samples: {num_test})")
    print(f"  Mode: {'NO CLAMP (Raw Analysis)' if args.no_clamp else 'CLAMPED (Standard Evaluation)'}")
    print(f"  * Noisy Input : {mean_noisy:.2f} ± {std_noisy:.2f} dB")
    print(f"  * Random Init : {mean_ran:.2f} ± {std_ran:.2f} dB")
    print(f"  * Pretrained  : {mean_pre:.2f} ± {std_pre:.2f} dB")
    print(f"  ------------------------------------------")
    
    diff = mean_pre - mean_ran
    winner = "Pretrained" if diff > 0 else "Random Init"
    print(f"  >> Winner: {winner} (Lead by {abs(diff):.2f} dB)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()