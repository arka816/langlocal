from copy import deepcopy

import torch
import torch.nn as nn

from machine_translation.transformer.embeddings import Embeddings, PositionalEncoding
from machine_translation.transformer.utils import subsequent_mask, clones
from machine_translation.transformer.attention import MultiHeadedAttention
from machine_translation.transformer.layers import LayerNorm, SublayerConnection, PositionFeedForward, Generator


class EncoderLayer(nn.Module):
    "Each encoder layer is made up of self-attention and feed forward."

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayers[1](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N such encoder layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Each decoder layer is made up of self attention, cross attention and feed forward."

    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.sublayers = clones(SublayerConnection(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.cross_attn(x, memory, memory, src_mask))
        return self.sublayers[2](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Pass the input (and memory and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, heads=8, dropout=0.1, share_embedding=True,
):
    "Helper: Construct a model from hyperparamters."

    c = deepcopy

    attn = MultiHeadedAttention(heads, d_model)
    ff = PositionFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))

    encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
    encoder = Encoder(encoder_layer, N)

    decoder_layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
    decoder = Decoder(decoder_layer, N)

    generator = Generator(d_model, tgt_vocab)

    model = EncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        generator=generator,
    )
    
    if share_embedding:
        assert src_vocab == tgt_vocab, "Source and target vocabulary size should be same if share_embedding is set to True."

        # tie source and target embedding
        model.src_embed[0].lut.weight = model.tgt_embed[0].lut.weight

        # tie generator weights to the target embeddings as well
        model.generator.proj.weight = model.tgt_embed[0].lut.weight


    print("Number of parameters: ", model.num_params)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Ensure all parameters are FP32
    model = model.float()

    return model


def test_inference():
    "Smoke test to check if forward pass is wired correctly."
    test_model = make_model(10, 10, N=2, d_model=16, d_ff=64, heads=2)
    test_model.eval()

    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    src_mask = torch.ones(1, 1, src.size(1), dtype=torch.bool)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory=memory,
            src_mask=src_mask,
            tgt=ys,
            tgt_mask=subsequent_mask(ys.size(1)).type_as(src.data),
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print(ys)


def run_tests():
    for _ in range(10):
        test_inference()


if __name__ == "__main__":
    run_tests()

