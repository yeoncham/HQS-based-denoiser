# 15개 모델들 비교하는 코드입니다.
"""
compare_models.py - 여러 모델 가중치의 성능 비교

사용법:
    python compare_models.py
    python compare_models.py --num-samples 10
    python compare_models.py --sigma 50
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import cv2

from models import HQS_Unrolled
from utils import get_psnr
from dataset import DIV2KDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Compare multiple model weights')
    parser.add_argument('--valid-dir', type=str, default='./DIV2K_data/DIV2K_valid_HR', help='검증 데이터 경로')
    parser.add_argument('--num-samples', type=int, default=100, help='테스트할 샘플 수')
    parser.add_argument('--patch-size', type=int, default=64, help='패치 크기')
    parser.add_argument('--sigma', type=float, default=50, help='노이즈 레벨 (0-255)')
    parser.add_argument('--output-dir', type=str, default='./comparison_results', help='결과 저장 경로')
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    """체크포인트에서 모델 로드 및 설정 추출"""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 설정 추출
    if isinstance(checkpoint, dict) and 'args' in checkpoint: # isinstance(checkpoint, dict): checkpoint가 dict인지 확인
        args = checkpoint['args']
        iterations = args.get('iterations', 8) # 좌측은 args가 가지고 있는 값, 없으면 우측의 디폴트 값 사용
        use_lora = args.get('use_lora', False)
        lora_rank = args.get('lora_rank', 8)
        lora_alpha = args.get('lora_alpha', 8.0)
    else:
        # 기본값
        iterations = 8
        use_lora = False
        lora_rank = 8
        lora_alpha = 8.0
    
    # 모델 생성
    model = HQS_Unrolled(iterations=iterations).to(device)
    
    # LoRA 적용
    if use_lora:
        from lora import apply_lora_to_nafnet
        apply_lora_to_nafnet(
            model.nafnet,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=['encoders', 'decoders', 'middle_blks']
        )
        model = model.to(device)
    
    # 가중치 로드
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    return model, iterations, use_lora, lora_rank


def create_fixed_noisy_images(dataset, num_samples, sigma, device):
    """고정된 노이즈 이미지 생성"""
    samples = []
    
    for i in range(num_samples):
        # 원본 이미지 로드
        img = cv2.cvtColor(cv2.imread(os.path.join(dataset.root, dataset.files[i])), 
                          cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # 중앙 패치 추출
        h, w, _ = img.shape
        ps = 64
        top = (h - ps) // 2
        left = (w - ps) // 2
        x = img[top:top+ps, left:left+ps, :]
        
        # 고정된 노이즈 추가 (seed 고정)
        np.random.seed(i)  # 각 이미지마다 다르지만 재현 가능
        noise = (sigma / 255.0) * np.random.randn(ps, ps, 3).astype(np.float32)
        y = x + noise
        
        # Kernel과 noise map
        h_kernel = np.zeros((ps, ps), dtype=np.float32)
        h_kernel[0, 0] = 1
        m = np.ones((ps, ps, 1), dtype=np.float32) * (sigma / 255.0)
        
        # Tensor 변환
        x_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
        y_t = torch.from_numpy(y).permute(2, 0, 1).unsqueeze(0).to(device)
        h_t = torch.from_numpy(h_kernel).unsqueeze(0).unsqueeze(0).to(device)
        m_t = torch.from_numpy(m).permute(2, 0, 1).unsqueeze(0).to(device)
        
        samples.append({
            'ground_truth': x_t,
            'noisy': y_t,
            'kernel': h_t,
            'noise_map': m_t,
            'filename': dataset.files[i]
        })
    
    return samples


def test_model(model, samples):
    """모델 테스트"""
    psnr_list = []
    outputs = []
    
    with torch.no_grad():
        for sample in samples:
            y = sample['noisy']
            h = sample['kernel']
            m = sample['noise_map']
            x = sample['ground_truth']
            
            # 추론
            output = model(y, h, m)
            
            # PSNR 계산
            psnr = get_psnr(output, x)
            psnr_list.append(psnr)
            outputs.append(output)
    
    avg_psnr = np.mean(psnr_list)
    std_psnr = np.std(psnr_list)
    
    return avg_psnr, std_psnr, psnr_list, outputs


def visualize_comparison(samples, all_results, output_dir):    
    num_samples = len(samples)
    model_names = list(all_results.keys())
    num_models = len(model_names)
    
    # 한 figure당 보여줄 최대 열 수
    COLS_PER_ROW = 5
    
    for sample_idx in range(num_samples):
        sample = samples[sample_idx]
        
        # 총 보여줄 아이템 수 = GT + Noisy + 모델 수
        total_items = 2 + num_models
        
        # 필요한 행 수 계산 (올림)
        rows_needed = (total_items + COLS_PER_ROW - 1) // COLS_PER_ROW
        rows = min(rows_needed, 3)  # 최대 3줄로 제한 (원하시면 이 줄 제거 가능)
        
        # figure 크기: 열 5개 기준, 행 수에 따라 조정
        fig, axes = plt.subplots(
            nrows=rows,
            ncols=COLS_PER_ROW,
            figsize=(5 * COLS_PER_ROW, 5 * rows),   # 가로 5개 × 세로 rows
            squeeze=False  # axes가 2D 배열로 나오게
        )
        
        # 모든 axes 평평하게 만들기
        axes_flat = axes.ravel()
        
        # 0: Ground Truth
        gt = sample['ground_truth'][0].cpu().permute(1, 2, 0).numpy()
        axes_flat[0].imshow(gt)
        axes_flat[0].set_title('Ground Truth', fontsize=11, fontweight='bold')
        axes_flat[0].axis('off')
        
        # 1: Noisy
        noisy = sample['noisy'][0].cpu().permute(1, 2, 0).numpy()
        noisy_psnr = get_psnr(sample['noisy'], sample['ground_truth'])
        axes_flat[1].imshow(noisy)
        axes_flat[1].set_title(f'Noisy\nPSNR: {noisy_psnr:.2f}', fontsize=11, fontweight='bold')
        axes_flat[1].axis('off')
        
        # 2번부터: 모델들
        for i, model_name in enumerate(model_names):
            pos = i + 2
            if pos >= len(axes_flat):
                break  # 공간 부족 시 중단
                
            output = all_results[model_name]['outputs'][sample_idx][0].cpu().permute(1, 2, 0).numpy()
            psnr = all_results[model_name]['psnr_list'][sample_idx]
            
            axes_flat[pos].imshow(output)
            axes_flat[pos].set_title(f'{model_name}\nPSNR: {psnr:.2f}', fontsize=10, fontweight='bold')
            axes_flat[pos].axis('off')
        
        # 사용하지 않는 axes 숨기기
        for j in range(total_items, len(axes_flat)):
            axes_flat[j].axis('off')
        
        plt.suptitle(f"Sample {sample_idx} - {sample['filename']}", fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # suptitle 공간 확보
        
        save_path = os.path.join(output_dir, f'comparison_sample_{sample_idx}_5x{rows}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: comparison_sample_{sample_idx}_5x{rows}.png")


def plot_psnr_comparison(all_results, output_dir):
    """PSNR 비교 바 그래프"""
    
    model_names = list(all_results.keys())
    avg_psnrs = [result['avg_psnr'] for result in all_results.values()]
    std_psnrs = [result['std_psnr'] for result in all_results.values()]
    
    # 색상 설정
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    
    fig, ax = plt.subplots(figsize=(max(12, len(model_names) * 1.5), 8))
    
    x = np.arange(len(model_names))
    bars = ax.bar(x, avg_psnrs, yerr=std_psnrs, capsize=5, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 값 표시
    for i, (bar, psnr, std) in enumerate(zip(bars, avg_psnrs, std_psnrs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.3,
               f'{psnr:.2f} ± {std:.2f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average PSNR (dB)', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psnr_comparison.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: psnr_comparison.png")


def load_pretrained_nafnet(weights_path, device):
    """Pretrained NAFNet 로드 (baseline)"""
    from models import NAFNet
    
    model = NAFNet(img_channel=4, width=32).to(device)
    
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint['params'] if 'params' in checkpoint else checkpoint
        
        # 채널 확장: 3 -> 4
        w = state_dict['intro.weight']
        new_w = torch.zeros((w.size(0), 4, 3, 3), device=w.device)
        new_w[:, :3] = w
        new_w[:, 3:4] = w.mean(dim=1, keepdim=True)
        state_dict['intro.weight'] = new_w
        
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    return model


def test_nafnet_baseline(model, samples):
    """NAFNet baseline 테스트 (HQS 없이)"""
    psnr_list = []
    outputs = []
    
    with torch.no_grad():
        for sample in samples:
            y = sample['noisy']
            m = sample['noise_map']
            x = sample['ground_truth']
            
            # NAFNet만 사용 (HQS iteration 없음)
            input_cat = torch.cat([y, m], dim=1)
            output = model(input_cat)
            
            # PSNR 계산
            psnr = get_psnr(output, x)
            psnr_list.append(psnr)
            outputs.append(output)
    
    avg_psnr = np.mean(psnr_list)
    std_psnr = np.std(psnr_list)
    
    return avg_psnr, std_psnr, psnr_list, outputs


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 결과 디렉토리 생성
    current_output_dir = os.path.join(args.output_dir, f'sigma_{args.sigma}')
    os.makedirs(current_output_dir, exist_ok=True)
    
    # 모델 경로들
    model_paths = [
        './experiments/fnaf_iter4/best_model.pth', 
        './experiments/fnaf_iter8/best_model.pth',  
        './experiments/fnaf_batch12_iter10/best_model.pth',
        
        './experiments/full_tuning_batch1_iter100_epoch10/best_model.pth',
        './experiments/full_tuning_batch8_iter6_epoch50/best_model.pth',
        './experiments/full_tuning_batch8_iter8_epoch50/best_model.pth',
        './experiments/full_tuning_batch8_iter10_epoch50/best_model.pth',
        './experiments/full_tuning_batch8_iter12_epoch50/best_model.pth',

        './experiments/lofnaf_batch12_iter8_rank8_epoch20/best_model.pth', 
        './experiments/lofnaf_batch12_iter10_rank8_epoch20/best_model.pth',  
        './experiments/lofnaf_batch12_iter8_rank16_alpha8_epoch20/best_model.pth', 
        './experiments/lofnaf_batch12_iter8_rank16_alpha16_epoch20/best_model.pth',  
        './experiments/lofnat_batch8_iter12_rank8_epoch20/best_model.pth', 
        './experiments/lofnaf_batch12_iter10_rank8_epoch50/best_model.pth',
    ]
    

    # 존재하는 모델만 필터링
    existing_models = []
    for path in model_paths:
        if os.path.exists(path):
            existing_models.append(path)
        else:
            print(f"⚠ Warning: {path} not found, skipping...")
    
    if len(existing_models) == 0:
        print("✗ Error: No model checkpoints found!")
        return
    
    print(f"Found {len(existing_models)} models to compare\n")
    
    # 고정된 테스트 샘플 생성
    print("="*60)
    print("Creating fixed test samples...")
    print("="*60)
    print(f"  Noise level (σ): {args.sigma}/255")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Patch size: {args.patch_size}x{args.patch_size}")
    
    dataset = DIV2KDataset(args.valid_dir, patch_size=args.patch_size)
    samples = create_fixed_noisy_images(dataset, args.num_samples, args.sigma, device)
    
    # noisy image average psnr
    avg_noisy_psnr = np.mean([get_psnr(s['noisy'], s['ground_truth']) for s in samples])
    print(f"  Average Noisy PSNR (Total {len(samples)} samples): {avg_noisy_psnr:.2f} dB")
    print()
    
    # 각 모델 테스트
    print("="*60)
    print("Testing Models...")
    print("="*60)
    
    all_results = {}
    
    # 1. Pretrained NAFNet Baseline 테스트
    pretrained_path = './weights/NAFNet-SIDD-width32.pth'
    if os.path.exists(pretrained_path):
        print(f"\n Testing: Pretrained NAFNet (Baseline)")
        print(f"   Path: {pretrained_path}")
        print(f"   Mode: Single-pass (No HQS)")
        
        try:
            nafnet_baseline = load_pretrained_nafnet(pretrained_path, device)
            avg_psnr, std_psnr, psnr_list, outputs = test_nafnet_baseline(nafnet_baseline, samples)
            
            print(f"   ✓ Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
            
            all_results['NAFNet-Baseline'] = {
                'avg_psnr': avg_psnr,
                'std_psnr': std_psnr,
                'psnr_list': psnr_list,
                'outputs': outputs,
                'iterations': 1,
                'use_lora': False,
            }
        except Exception as e:
            print(f"   ✗ Error loading baseline: {e}")
    else:
        print(f"\n⚠ Warning: Pretrained NAFNet not found at {pretrained_path}")
        print("   Skipping baseline comparison...")
    
    # 2. HQS 모델들 테스트
    for model_path in existing_models:
        model_name = Path(model_path).parent.name
        print(f"\n📊 Testing: {model_name}")
        print(f"   Path: {model_path}")
        
        try:
            # 모델 로드
            model, iterations, use_lora, lora_rank = load_model_from_checkpoint(model_path, device)
            print(f"   Iterations: {iterations}")
            if use_lora:
                print(f"   LoRA: Enabled (rank={lora_rank})")
            else:
                print(f"   LoRA: Disabled")
            
            # 테스트
            avg_psnr, std_psnr, psnr_list, outputs = test_model(model, samples)
            
            print(f"   ✓ Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
            
            all_results[model_name] = {
                'avg_psnr': avg_psnr,
                'std_psnr': std_psnr,
                'psnr_list': psnr_list,
                'outputs': outputs,
                'iterations': iterations,
                'use_lora': use_lora,
                'lora_rank': lora_rank

            }
            
        except Exception as e:
            print(f"   ✗ Error loading model: {e}")
            continue
    
    if len(all_results) == 0:
        print("\n✗ No models could be tested successfully!")
        return
    
    # 결과 시각화
    print("\n" + "="*60)
    print("Generating Visualizations...")
    print("="*60)
    
    visualize_comparison(samples, all_results, current_output_dir)
    plot_psnr_comparison(all_results, current_output_dir)
    
    # 최종 리포트
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    # PSNR 순으로 정렬
    sorted_results = sorted(all_results.items(), 
                           key=lambda x: x[1]['avg_psnr'], 
                           reverse=True)
    
    print(f"\n{'Rank':<6} {'Model':<50} {'Avg PSNR':<15} {'Std':<10} {'Iter':<6} {'LoRA'}")
    print("-" * 100)
    
    for rank, (model_name, result) in enumerate(sorted_results, 1):
        lora_str = f"Rank {result.get('lora_rank', 'N/A')}" if result['use_lora'] else "No"
        print(f"{rank:<6} {model_name:<50} {result['avg_psnr']:<15.2f} "
              f"{result['std_psnr']:<10.2f} {result['iterations']:<6} {lora_str}")
    
    best_model = sorted_results[0][0]
    best_psnr = sorted_results[0][1]['avg_psnr']
    
    print("\n" + "="*60)
    print(f"🏆 Best Model: {best_model}")
    print(f"   PSNR: {best_psnr:.2f} dB")
    print(f"   Results saved to: {current_output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()