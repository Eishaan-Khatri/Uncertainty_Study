import torch
from torch.utils.data import Dataset
import numpy as np

class NoisyECGDataset(Dataset):
    def __init__(self, clean_data_path, noise_level=0.1, mask_prob=0.0):
        self.clean_data = torch.load(clean_data_path)
        self.noise_level = noise_level
        self.mask_prob = mask_prob
        
    def __len__(self):
        return len(self.clean_data)
    
    def __getitem__(self, idx):
        clean = self.clean_data[idx]
        
        # 1. Add Gaussian Noise
        noise = torch.randn_like(clean) * self.noise_level
        noisy = clean + noise
        
        # 2. Add Random Masking
        if self.mask_prob > 0:
            if torch.rand(1).item() < 0.5:
                mask_len = int(clean.shape[-1] * 0.1)
                start = torch.randint(0, clean.shape[-1] - mask_len, (1,)).item()
                noisy[:, start : start + mask_len] = 0.0
        
        noisy = torch.clamp(noisy, 0.0, 1.0)
        return noisy, clean
