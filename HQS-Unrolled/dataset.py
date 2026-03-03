import torch
from torch.utils.data import Dataset
import cv2
import os
import random
import numpy as np

class DIV2KDataset(Dataset):
    def __init__(self, root, patch_size=64, sigma_range=(5, 70)):
        self.root = root
        self.files = sorted([f for f in os.listdir(root) if f.endswith(('.png', '.jpg'))])
        self.patch_size = patch_size
        self.sigma_min = sigma_range[0] / 255.0
        self.sigma_max = sigma_range[1] / 255.0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.root, self.files[idx])), cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0 # [0, 255] -> [0, 1]로 스케일링
        
        h_orig, w_orig, _ = img.shape # 원본 이미지의 H, W
        ps = self.patch_size
        top, left = random.randint(0, h_orig - ps), random.randint(0, w_orig - ps)
        x = img[top:top+ps, left:left+ps, :]

        sigma = random.uniform(self.sigma_min, self.sigma_max)
        y = x + (sigma * np.random.randn(ps, ps, 3)).astype(np.float32)
        m = np.ones((ps, ps, 1), dtype=np.float32) * sigma
        
        x_t = torch.from_numpy(x).permute(2, 0, 1)
        y_t = torch.from_numpy(y).permute(2, 0, 1)
        m_t = torch.from_numpy(m).permute(2, 0, 1)

        return x_t, y_t, m_t

# 이 부분은 짜놓긴 했는데 안 쓴 코드입니다.
# visualize_deblur.py에 실제로 blur도 입히고
# 복원도 했습니다.
class DIV2KUnrolled(Dataset):
    def __init__(self, root, kernel_path='kernels_12.npy', patch_size=64, sigma_range=(5, 70)):
        self.root = root
        self.files = sorted([f for f in os.listdir(root) if f.endswith(('.png', '.jpg'))])
        self.kernels = np.load(kernel_path)
        self.patch_size = patch_size
        self.sigma_min = sigma_range[0] / 255.0
        self.sigma_max = sigma_range[1] / 255.0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.root, self.files[idx])), cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        h, w, _ = img.shape
        ps = self.patch_size
        
        # Random Crop (GT)
        if h < ps or w < ps:
            img = cv2.resize(img, (max(h, ps), max(w, ps)))
            h, w, _ = img.shape
            
        top, left = random.randint(0, h - ps), random.randint(0, w - ps)
        gt = img[top:top+ps, left:left+ps, :]

        # 랜덤 커널 선택 및 전처리 (Flip)
        k_idx = random.randint(0, len(self.kernels) - 1)
        k_np = self.kernels[k_idx].copy()
        k_np = k_np / np.sum(k_np) 
        
        k_flipped = np.flip(k_np, axis=(0, 1))
        # Blur 적용
        blurred = cv2.filter2D(gt, -1, k_flipped)

        # Noise 추가
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        noise = sigma * np.random.randn(ps, ps, 3).astype(np.float32)
        y = blurred + noise
        
        # Sigma Map
        m = np.full((ps, ps, 1), sigma, dtype=np.float32)
        
        # Tensor 변환
        x_t = torch.from_numpy(gt).permute(2, 0, 1).float()
        y_t = torch.from_numpy(y).permute(2, 0, 1).float()
        m_t = torch.from_numpy(m).permute(2, 0, 1).float()
        k_t = torch.from_numpy(k_np).unsqueeze(0).float() # (1, kh, kw)
        return x_t, y_t, m_t, k_t