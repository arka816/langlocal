import numpy as np

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
