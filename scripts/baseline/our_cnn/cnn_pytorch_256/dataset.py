import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class ASCADDataset(Dataset):
    def __init__(self, h5_path, set_type='profiling'):
        self.set_type = set_type
        
        with h5py.File(h5_path, 'r') as f:
            if set_type == 'profiling':
                self.traces = f['Profiling_traces']['traces'][:]
                self.labels = f['Profiling_traces']['labels'][:]
            else:
                self.traces = f['Attack_traces']['traces'][:]
                self.labels = f['Attack_traces']['labels'][:]
        
        # 转换为float32并保持原始范围（不做标准化）
        self.traces = self.traces.astype(np.float32)
        
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        x = self.traces[idx]
        y = self.labels[idx]
        
        # PyTorch Conv1d期望 (batch, channels, length)
        # 所以需要 (1, 700)
        x = torch.from_numpy(x).float().unsqueeze(0)  # (1, 700)
        y = torch.tensor(y, dtype=torch.long)
        
        return x, y