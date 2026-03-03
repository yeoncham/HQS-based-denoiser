# 패치가 아닌 Original Resolution 비교용 코드입니다.
import argparse
import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from models import NAFNet  
from utils import get_psnr 

def parse_args():
    parser = argparse.ArgumentParser(description='Compare Pretrained vs Random-Init NAFNet (Full Resolution)')
    
    parser.add_argument('--pretrained-path', type=str, required=True, help='Pretrained 모델 경로 (.pth)')
    parser.add_argument('--random-path', type=str, required=True, help='Random Init 모델 경로 (.pth)')
    
    parser.add_argument('--sigma', type=float, default=50.0, help='테스트할 노이즈 레벨 (Sigma)')
    parser.add_argument('--valid-dir', type=str, default='./DIV2K/DIV2K_valid_HR', help='검증 이미지 폴더')
    parser.add_argument('--num-samples', type=int, default=100, help='시각화 및 평가할 이미지 개수')
    parser.add_argument('--output-dir', type=str, default='./comparison_results_full')
    parser.add_argument('--no-clamp', action='store_true', help='사용 시 0~1 Clamping을 해제합니다. (분석용)')
    
    return parser.parse_args()

def load_model(path, device):
    model = NAFNet(img_channel=4, width=32, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]).to(device)
    
    checkpoint = torch.load(path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_data_full(img_path, sigma, seed_idx):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    h, w, _ = img.shape
    
    np.random.seed(seed_idx)
    noise = (sigma / 255.0) * np.random.randn(h, w, 3).astype(np.float32)
    noisy = img + noise
    
    m = np.ones((h, w, 1), dtype=np.float32) * (sigma / 255.0)
    
    to_tensor = lambda x: torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    
    gt_t = to_tensor(img)
    noisy_t = to_tensor(noisy)
    m_t = to_tensor(m)
    
    input_t = torch.cat([noisy_t, m_t], dim=1)
    
    return gt_t, noisy_t, input_t

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 저장 폴더 이름에 clamp 여부 표시
    mode_str = "no_clamp" if args.no_clamp else "clamped"
    save_dir = os.path.join(args.output_dir, f"sigma_{int(args.sigma)}_{mode_str}")
    os.makedirs(save_dir, exist_ok=True)

    print("="*50)
    print(f"Start Comparing Models (Sigma={args.sigma})")
    print(f"Mode: {'[ANALYSIS] No Clamp' if args.no_clamp else '[STANDARD] 0~1 Clamping'}")
    print(f"1. Pretrained: {args.pretrained_path}")
    print(f"2. Random Init: {args.random_path}")
    print("="*50)

    try:
        model_pre = load_model(args.pretrained_path, device)
        model_ran = load_model(args.random_path, device)
    except Exception as e:
        print(f"\n[Error] 모델 로드 중 오류 발생:\n{e}")
        return

    img_list = sorted([f for f in os.listdir(args.valid_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if len(img_list) == 0:
        print(f"[Error] {args.valid_dir} 경로에 이미지가 없습니다.")
        return
        
    num_test = min(len(img_list), args.num_samples)
    print(f"\nProcessing {num_test} images...")

    psnr_pre_list = []
    psnr_ran_list = []
    psnr_noisy_list = []

    with torch.no_grad():
        for i in range(num_test):
            img_name = img_list[i]
            gt, noisy, net_input = get_data_full(
                os.path.join(args.valid_dir, img_name), 
                args.sigma, 
                i 
            )
            
            gt = gt.to(device)
            net_input = net_input.to(device)
            noisy_vis = noisy.to(device)
            
            # 모델 추론 (Raw Output)
            out_pre_raw = model_pre(net_input)
            out_ran_raw = model_ran(net_input)
            
            if args.no_clamp:
                # Clamp 끔
                eval_pre = out_pre_raw
                eval_ran = out_ran_raw
                eval_noisy = noisy_vis
                vis_prefix = "[Raw]"
            else:
                # Clamp 켬 
                eval_pre = out_pre_raw.clamp(0, 1)
                eval_ran = out_ran_raw.clamp(0, 1)
                eval_noisy = noisy_vis.clamp(0, 1)
                vis_prefix = ""

            # PSNR 계산
            psnr_pre = get_psnr(eval_pre, gt)
            psnr_ran = get_psnr(eval_ran, gt)
            psnr_noisy = get_psnr(eval_noisy, gt)

            psnr_pre_list.append(psnr_pre)
            psnr_ran_list.append(psnr_ran)
            psnr_noisy_list.append(psnr_noisy)
            
            # 시각화 및 저장
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            def show(ax, tensor, title):
                img_np = tensor[0].permute(1, 2, 0).cpu().numpy()
                
                if args.no_clamp:
                    v_max = img_np.max()
                    v_min = img_np.min()
                    
                    img_vis = (img_np - v_min) / (v_max - v_min)
                    
                    ax.imshow(img_vis)
                    
                    title += f"\n[Raw View] Range: {v_min:.2f} ~ {v_max:.2f}"
                else:
                    ax.imshow(np.clip(img_np, 0, 1))
                
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.axis('off')   
            show(axes[0], gt, "\nGround Truth")
            show(axes[1], eval_noisy, f"\n{vis_prefix} Noisy Input (PSNR: {psnr_noisy:.2f} dB)")
            show(axes[2], eval_pre, f"\n{vis_prefix} Fine-Tuned (PSNR: {psnr_pre:.2f} dB)")
            show(axes[3], eval_ran, f"\n{vis_prefix} Random Init (PSNR: {psnr_ran:.2f} dB)")

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"compare_{os.path.splitext(img_name)[0]}.png")
            plt.savefig(save_path, dpi=100)
            plt.close()
            
            torch.cuda.empty_cache()
            
            print(f"[{i+1}/{num_test}] {img_name} | Pre: {psnr_pre:.2f}dB | Ran: {psnr_ran:.2f}dB")

    mean_pre = np.mean(psnr_pre_list)
    std_pre = np.std(psnr_pre_list)
    
    mean_ran = np.mean(psnr_ran_list)
    std_ran = np.std(psnr_ran_list)
    
    mean_noisy = np.mean(psnr_noisy_list)
    std_noisy = np.std(psnr_noisy_list)
    
    print("\n" + "="*50)
    print(f"  [Final Results (Mean ± Std) - Sigma {args.sigma}]")
    print(f"  Mode: {'NO CLAMP (Raw Analysis)' if args.no_clamp else 'CLAMPED (Standard Evaluation)'}")
    print(f"  * Noisy Input : {mean_noisy:.2f} ± {std_noisy:.2f} dB")
    print(f"  * Random Init : {mean_ran:.2f} ± {std_ran:.2f} dB")
    print(f"  * Pretrained  : {mean_pre:.2f} ± {std_pre:.2f} dB")
    print("-" * 50)
    
    diff = mean_pre - mean_ran
    winner = "Pretrained" if diff > 0 else "Random Init"
    print(f"  >> Winner: {winner} (Lead by {abs(diff):.2f} dB)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()