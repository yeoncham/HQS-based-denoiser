# actual noise와 map noise mismatch 비교 코드입니다.
# blur 커널 적용했을 때의 경우를 보고 싶어서 만들어놨습니다.
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

    # 커널 로드
    kernels = np.load(args.kernel_path)
    if args.kernel_idx < 0 or args.kernel_idx >= len(kernels):
        raise ValueError("Invalid kernel index")

    k_np = kernels[args.kernel_idx]
    print(f"Kernel index: {args.kernel_idx}")
    print(f"Real sigma: {args.real_sigma}")
    print(f"Map sigma: {args.map_sigma}")

    # 이미지 로드
    full_img = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    full_img = full_img.astype(np.float32) / 255.0
    h, w, _ = full_img.shape

    top, left = (h - patch_size) // 2, (w - patch_size) // 2
    gt = full_img[top:top+patch_size, left:left+patch_size, :]

    # blur 생성
    k_flipped = np.flip(k_np, axis=(0, 1))
    blurred = cv2.filter2D(gt, -1, k_flipped)

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
    filename = f'k{args.kernel_idx}_real{int(args.real_sigma)}_map{int(args.map_sigma)}.png'
    save_path = os.path.join(args.output_dir, filename)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(gt); axes[0].axis('off')
    axes[1].imshow(y); axes[1].axis('off')
    axes[2].imshow(restored); axes[2].axis('off')
 
    gt_t = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0).float()
    y_torch = torch.from_numpy(y).permute(2,0,1).unsqueeze(0).float()
    restored_torch = torch.from_numpy(restored).permute(2,0,1).unsqueeze(0).float()

    noisy_psnr = get_psnr(gt_t, y_torch)
    restored_psnr = get_psnr(gt_t, restored_torch)
    print(f"noisy_psnr:{noisy_psnr}, psnr:{restored_psnr}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()