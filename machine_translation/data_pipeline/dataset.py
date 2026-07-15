import numpy as np

import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, dims):
        self.src_memmap = np.memmap(
            src_file,
            mode="r",
            dtype=np.uint16,
            shape=dims
        )
        self.tgt_memmap = np.memmap(
            tgt_file,
            mode="r",
            dtype=np.uint16,
            shape=dims
        )

    def __len__(self):
        return self.src_memmap.shape[0]

    def __getitem__(self, idx):
        return self.src_memmap[idx], self.tgt_memmap[idx]


class TranslationDatasetFast(Dataset):
    def __init__(self, src_file, tgt_file, dims):
        self.rows, self.size = dims
        self.src_file = src_file
        self.tgt_file = tgt_file

    def __len__(self):
        return self.rows

    def __getitem__(self, idx):
        element_size = 2    # uint16
        
        offset = idx * self.size * element_size
        
        # High-performance, native PyTorch file streaming (bypasses numpy)
        x = torch.from_file(self.src_file, shared=True, size=self.size, dtype=np.uint16, offset=offset)
        y = torch.from_file(self.tgt_file, shared=True, size=self.size, dtype=np.uint16, offset=offset)
        
        return x, y
