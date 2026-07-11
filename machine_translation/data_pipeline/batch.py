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
            self.src = self.src.to(device)
            self.tgt = self.tgt.to(device)
            self.tgt_y = self.tgt_y.to(device)
            self.src_mask = self.src_mask.to(device)
            self.tgt_mask = self.tgt_mask.to(device)
            self.ntokens = self.ntokens.to(device)
