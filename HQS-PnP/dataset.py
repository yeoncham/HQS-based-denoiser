import torch
from torch.utils.data import Dataset
import cv2
import os
import random
import numpy as np

# 패치 단위로 가우시안 노이즈만 추가한 데이터를 return
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
        img = img.astype(np.float32) / 255.0 
        
        h_orig, w_orig, _ = img.shape 
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

# 전체 이미지 해상도에 대해서 blur를 적용하고 가우시안 노이즈도 추가한 데이터를 return
class DIV2KBlur(Dataset):
    def __init__(self, root, sigma=5, kernel_idx=0):
        self.root = root
        self.files = sorted([f for f in os.listdir(root) if f.endswith(('.png', '.jpg'))])
        self.kernels = np.load("kernels_12.npy")
        self.sigma = sigma
        self.kernel_idx = kernel_idx

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        x = img 
        h, w, _ = x.shape

        # 커널 로드 및 정규화
        curr_kernel = self.kernels[self.kernel_idx]
        curr_kernel = curr_kernel / np.sum(curr_kernel)
        kernel_flipped = np.flip(curr_kernel, axis=(0, 1))

        # 블러 적용
        y_blur = np.zeros_like(x)
        for c in range(3):
            y_blur[:, :, c] = cv2.filter2D(x[:, :, c], -1, kernel_flipped) # convolution

        # 노이즈 추가
        noise = (self.sigma / 255.0) * np.random.randn(h, w, 3).astype(np.float32)
        y = y_blur + noise
        y = np.clip(y, 0, 1)

        # Noise map
        m = np.full((h, w, 1), self.sigma / 255.0, dtype=np.float32)

        # Tensor 변환
        x_t = torch.from_numpy(x).permute(2, 0, 1)
        y_t = torch.from_numpy(y).permute(2, 0, 1)
        m_t = torch.from_numpy(m).permute(2, 0, 1)
        kernel_t = torch.from_numpy(curr_kernel.copy()).float() # flip 전의 Kernel return
        return x_t, y_t, m_t, kernel_t