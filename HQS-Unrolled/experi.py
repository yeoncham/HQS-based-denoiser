# actual noise와 map noise mismatch 비교 코드입니다.
# experiment_sigma_mismatch랑 다른 점은 blur를 적용하지 않고
# 단지 노이즈만 더했을 때의 경우를 보고 싶어서 만든 코드입니다.
import torch
import numpy as np
import cv2
import os
import argparse 
import matplotlib.pyplot as plt
from models import HQS_Unrolled
from utils import get_psnr

def get_args():
    parser = argparse.ArgumentParser(description='Deblur Sigma Mismatch Experiment')

    parser.add_argument('--kernel_idx', type=int, required=True, 
                        help='Index of kernel (0-11)')

    parser.add_argument('--img_path', type=str,
                        default='./DIV2K_data/DIV2K_valid_HR/0862.png')

    parser.add_argument('--weights', type=str,
                        default='./experiments/full_tuning_batch8_iter12_epoch50/best_model.pth')

    parser.add_argument('--kernel_path', type=str, default='kernels_12.npy')

    parser.add_argument('--output_dir', type=str, default='./result_sigma_mismatch')

    parser.add_argument('--real_sigma', type=float, default=5.0,
                        help='Actual noise added to image')

    parser.add_argument('--map_sigma', type=float, default=5.0,
                        help='Sigma injected to model (noise map)')

    return parser.parse_args()

def main():
    args = get_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    real_sigma_val = args.real_sigma / 255.0
    map_sigma_val  = args.map_sigma / 255.0
    patch_size = 512

    # 모델 로드
    print(f"Loading model from {args.weights}...")
    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    model = HQS_Unrolled(iterations=12).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    temp_kernels = np.load(args.kernel_path)
    ref_h, ref_w = temp_kernels[0].shape 
    
    # 모든 값을 0으로 채우고, 정중앙에만 1을 넣습니다. (항등원)
    k_np = np.zeros((ref_h, ref_w), dtype=np.float32)
    center_h, center_w = ref_h // 2, ref_w // 2
    k_np[center_h, center_w] = 1.0

    print("-" * 50)
    print("!!! PURE DENOISING MODE ACTIVATED !!!")
    print(f"Kernel is forced to DELTA function (Shape: {k_np.shape})")
    print(f"kernel_idx {args.kernel_idx} is IGNORED.")
    print("-" * 50)
    
    print(f"Real sigma: {args.real_sigma}")
    print(f"Map sigma: {args.map_sigma}")

    # 이미지 로드
    full_img = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    full_img = full_img.astype(np.float32) / 255.0
    h, w, _ = full_img.shape

    top, left = (h - patch_size) // 2, (w - patch_size) // 2
    gt = full_img[top:top+patch_size, left:left+patch_size, :]

    # Blur 과정 생략 
    blurred = gt.copy() 
    
    # 실제 노이즈 추가
    np.random.seed(42)
    noise = real_sigma_val * np.random.randn(*blurred.shape).astype(np.float32)
    y = blurred + noise
    y = np.clip(y, 0, 1)

    # padding
    kh, kw = k_np.shape
    pad_h, pad_w = kh, kw

    y_padded = np.pad(y, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
    target_h, target_w = y_padded.shape[:2]

    # kernel FFT preprocess
    h_padded = np.zeros((target_h, target_w), dtype=np.float32)
    h_padded[:kh, :kw] = k_np
    h_roll = np.roll(h_padded, (-(kh//2), -(kw//2)), axis=(0, 1))

    # tensor 변환
    y_t = torch.from_numpy(y_padded).permute(2, 0, 1).unsqueeze(0).to(device)
    h_t = torch.from_numpy(h_roll).unsqueeze(0).unsqueeze(0).to(device)

    # 모델에는 map sigma 주입
    m_t = torch.full((1, 1, target_h, target_w), map_sigma_val).to(device)

    # inference
    print("Running inference...")
    with torch.no_grad():
        restored_t = model(y_t, h_t, m_t)

        restored_t = restored_t[..., pad_h:-pad_h, pad_w:-pad_w]
        restored = restored_t[0].cpu().permute(1, 2, 0).numpy()
        restored = np.clip(restored, 0, 1)

    # 저장 
    filename = f'PureDenoise_real{int(args.real_sigma)}_map{int(args.map_sigma)}.png'
    save_path = os.path.join(args.output_dir, filename)

    # 결과 시각화 (
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].imshow(gt); axes[0].axis('off') #axes[0].set_title('GT')
    axes[1].imshow(y); axes[1].axis('off') # axes[1].set_title('Noisy Input')
    axes[2].imshow(restored); axes[2].axis('off') # axes[2].set_title('Restored')
    '''
    # 차영상 (어디가 뭉개졌는지 확인용)
    diff = np.abs(gt - restored)
    diff = np.clip(diff * 10, 0, 1) # 10배 증폭
    axes[3].imshow(diff, cmap='inferno'); axes[3].axis('off'); axes[3].set_title('Diff Map (x10)')
 '''
    gt_t = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0).float()
    y_torch = torch.from_numpy(y).permute(2,0,1).unsqueeze(0).float()
    restored_torch = torch.from_numpy(restored).permute(2,0,1).unsqueeze(0).float()

    noisy_psnr = get_psnr(gt_t, y_torch)
    restored_psnr = get_psnr(gt_t, restored_torch)
    print(f"noisy_psnr:{noisy_psnr:.2f}, psnr:{restored_psnr:.2f}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved -> {save_path}")

if __name__ == "__main__":
    main()