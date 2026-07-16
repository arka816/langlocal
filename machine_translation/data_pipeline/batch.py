from machine_translation.transformer.utils import subsequent_mask


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):   # 2 = <blank>
        self.src = src                              # Dimension (B x S)
        self.src_mask = (src != pad).unsqueeze(-2)  # Dimension (B x 1 x S)

        if tgt is not None:                         # looks like [<bos>, el, come, manzanas, <eos>]
            self.tgt = tgt[:, :-1]                  # looks like [<bos>, el, come, manzanas]
            self.tgt_y = tgt[:, 1:]                 # looks like [el, come, manzanas, <eos>]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words"
        tgt_mask = (tgt != pad).unsqueeze(-2)                                       # Dimension (B x 1 x S)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)  # Dimension (B x S x S)
        return tgt_mask

    def to_device(self, device):
        if device:
            self.src = self.src.to(device, non_blocking=True)
            self.tgt = self.tgt.to(device, non_blocking=True)
            self.tgt_y = self.tgt_y.to(device, non_blocking=True)
            self.src_mask = self.src_mask.to(device, non_blocking=True)
            self.tgt_mask = self.tgt_mask.to(device, non_blocking=True)
            self.ntokens = self.ntokens.to(device, non_blocking=True)


def test_batch_masks_and_targets():
    bos, eos, pad = 0, 1, 2

    # Create a small toy batch (B x S)
    src = torch.tensor([
        [5, 6, 7, pad, pad, pad],
        [8, 9, 10, 11, pad, pad],
    ], dtype=torch.long)

    # Targets include BOS at start, EOS somewhere, then PADs
    tgt = torch.tensor([
        [bos, 11, 12, 13, eos, pad],
        [bos, 21, 22, eos, pad, pad],
    ], dtype=torch.long)

    batch = Batch(src, tgt, pad=pad)

    # Targets: shifted inputs
    assert torch.equal(batch.tgt, tgt[:, :-1]), "batch.tgt mismatch"
    assert torch.equal(batch.tgt_y, tgt[:, 1:]), "batch.tgt_y mismatch"

    # ntokens should count non-pad tokens in tgt_y
    expected_ntokens = int((batch.tgt_y != pad).sum().item())
    ntokens_val = int(batch.ntokens.item()) if hasattr(batch.ntokens, "item") else int(batch.ntokens)
    assert ntokens_val == expected_ntokens, f"ntokens {ntokens_val} != {expected_ntokens}"

    # src_mask boolean check
    expected_src_mask = (src != pad).unsqueeze(-2)
    assert torch.equal(batch.src_mask, expected_src_mask), "src_mask mismatch"

    # tgt_mask shape and causal property (B x L x L)
    L = batch.tgt.size(1)
    assert batch.tgt_mask.shape == (src.size(0), L, L), "tgt_mask shape wrong"

    # causal: positions where j > i must be masked (False)
    tgt_mask_bool = batch.tgt_mask.bool()
    for b_idx in range(src.size(0)):
        for i in range(L):
            for j in range(L):
                if j > i:
                    assert not tgt_mask_bool[b_idx, i, j].item(), f"future not masked at b{b_idx}, {i},{j}"

    print("Batch masks and targets look correct.")


if __name__ == "__main__":
    test_batch_masks_and_targets()
