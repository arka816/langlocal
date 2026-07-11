import numpy as np
import torch

from batch import Batch

class DynamicTrimmingCollator:
    def __init__(self, bos_id=0, eos_id=1, pad_id=2):
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    def __call__(self, batch):
        src_batch = np.array([item[0] for item in batch])
        tgt_batch = np.array([item[1] for item in batch])

        src_padding_mask = src_batch == self.pad_id
        tgt_padding_mask = tgt_batch == self.pad_id

        src_all_padding = np.all(src_padding_mask, axis=0)
        tgt_all_padding = np.all(tgt_padding_mask, axis=0)

        src_maxlen = np.argmax(src_all_padding) if np.any(src_all_padding) else src_batch.shape[1]
        tgt_maxlen = np.argmax(tgt_all_padding) if np.any(tgt_all_padding) else tgt_batch.shape[1]

        src_batch_trimmed = src_batch[:, :src_maxlen]
        tgt_batch_trimmed = tgt_batch[:, :tgt_maxlen]

        tgt_batch_final = np.full((len(batch),  tgt_maxlen + 2), self.pad_id, dtype=tgt_batch.dtype)
        tgt_batch_final[:, 0] = self.bos_id
        tgt_batch_final[:, 1:-1] = tgt_batch_trimmed
        tgt_batch_final[:, -1] = self.eos_id

        src_tensors = torch.from_numpy(src_batch_trimmed)
        tgt_tensors = torch.from_numpy(tgt_batch_final)

        batch = Batch(src=src_tensors, tgt=tgt_tensors, pad=self.pad_id)

        return batch
