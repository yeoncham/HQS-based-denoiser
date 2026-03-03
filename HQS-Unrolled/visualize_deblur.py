# blur kernel 씌우고 복원하는 코드입니다.
import torch
import numpy as np
import cv2
import os
import argparse 
import matplotlib.pyplot as plt
from models import HQS_Unrolled

def get_args():
    parser = argparse.ArgumentParser(description='Deblur Visualization')

    parser.add_argument('--kernel_idx', type=int, required=True, 
                        help='Index of the kernel to use (0-11)')

    parser.add_argument('--img_path', type=str, default='./DIV2K_data/DIV2K_valid_HR/0862.png',
                        help='Path to the test image')
    parser.add_argument('--weights', type=str, default='./experiments/full_tuning_batch8_iter12_epoch50/best_model.pth',
                        help='Path to the model weights')
    parser.add_argument('--kernel_path', type=str, default='kernels_12.npy',
                        help='Path to the kernel numpy file')
    parser.add_argument('--output_dir', type=str, default='./result_single',
                        help='Directory to save the result')
    parser.add_argument('--sigma', type=float, default=5.0,
                        help='Noise level (sigma)')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    sigma_val = args.sigma / 255.0
    patch_size = 512
    # 모델 로드
    print(f"Loading model from {args.weights}...")
    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model = HQS_Unrolled(iterations=12).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    # 커널 로드 및 선택
    kernels = np.load(args.kernel_path)
    if args.kernel_idx < 0 or args.kernel_idx >= len(kernels):
        raise ValueError(f"Kernel index must be between 0 and {len(kernels)-1}")
    
    k_np = kernels[args.kernel_idx] # (kh, kw)
    print(f"Selected Kernel Index: {args.kernel_idx}, Shape: {k_np.shape}")
    # 이미지 로드 및 전처리
    full_img = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    full_img = full_img.astype(np.float32) / 255.0
    h, w, _ = full_img.shape
    # 중앙 Crop
    top, left = (h - patch_size) // 2, (w - patch_size) // 2
    gt = full_img[top:top+patch_size, left:left+patch_size, :]
    # 블러 생성 (Convolution)
    # cv2.filter2D는 Correlation이므로, Convolution을 위해 커널을 뒤집어줌(Flip)
    k_flipped = np.flip(k_np, axis=(0, 1))
    blurred = cv2.filter2D(gt, -1, k_flipped)
    # 노이즈 추가
    np.random.seed(42)
    noise = sigma_val * np.random.randn(*blurred.shape).astype(np.float32)
    y = blurred + noise
    y = np.clip(y, 0, 1)
    # 커널 크기만큼 패딩
    kh, kw = k_np.shape
    pad_h, pad_w = kh, kw
    
    y_padded = np.pad(y, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
    target_h, target_w = y_padded.shape[:2]

    # 커널 FFT 전처리 (Shift/Roll)
    h_padded = np.zeros((target_h, target_w), dtype=np.float32)
    h_padded[:kh, :kw] = k_np # 여기서는 원본 커널 사용
    
    # 중심을 (0,0)으로 이동
    h_roll = np.roll(h_padded, (-(kh//2), -(kw//2)), axis=(0, 1))

    # 텐서 변환
    y_t = torch.from_numpy(y_padded).permute(2, 0, 1).unsqueeze(0).to(device)
    h_t = torch.from_numpy(h_roll).unsqueeze(0).unsqueeze(0).to(device)
    m_t = torch.full((1, 1, target_h, target_w), sigma_val).to(device)

    # 추론
    print("Running Inference...")
    with torch.no_grad():
        restored_t = model(y_t, h_t, m_t)
        
        # 결과 Crop (패딩 제거)
        restored_t = restored_t[..., pad_h:-pad_h, pad_w:-pad_w]
        restored = restored_t[0].cpu().permute(1, 2, 0).numpy()
        restored = np.clip(restored, 0, 1)

    # 결과 저장
    save_path = os.path.join(args.output_dir, f'result_kernel_{args.kernel_idx}.png')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gt)
    #axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    axes[1].imshow(y)
    #axes[1].set_title(f"Blurred (Kernel {args.kernel_idx})")
    axes[1].axis('off')
    
    axes[2].imshow(restored)
    #axes[2].set_title("HQS-NAFNet Restored")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Result saved to {save_path}")

if __name__ == "__main__":
    main()