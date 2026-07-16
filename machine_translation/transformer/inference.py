from os.path import join
from pathlib import Path

from tokenizers import Tokenizer
import yaml

import torch

from machine_translation.transformer.transformer import make_model
from machine_translation.transformer.utils import load_checkpoint, subsequent_mask


class MachineTranslator:
    def __init__(self, bos_id, eos_id, tokenizer_path, weights_path, config_path):
        self.bos_id = bos_id
        self.eos_id = eos_id

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.model_configs = {
            "src_vocab":    config['data']['VOCAB_SIZE'],
            "tgt_vocab":    config['data']['VOCAB_SIZE'],
            "N":            config['model']['NUM_LAYERS'],
            "d_model":      config['model']['EMBEDDING_DIM'], 
            "d_ff":         config['model'].get("FFN_DIM", 4 * config['model']['EMBEDDING_DIM']), 
            "heads":        config['model']['HEADS'],
            "dropout":      config['model']['DROPOUT'], 
            "share_embedding":  config['model']['SHARE_EMBEDDINGS']
        }

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.model = make_model(**self.model_configs)

        checkpoint = load_checkpoint(weights_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        del checkpoint
        

    def _tokenize(self, text):
        return self.tokenizer.encode(text).ids

    def _detokenize(self, tokens):
        return self.tokenizer.decode(tokens)

    def run(self, text):
        src = torch.LongTensor([self._tokenize(text)])
        src_mask = torch.ones(1, 1, src.size(1), dtype=torch.bool)

        memory = self.model.encode(src, src_mask)
        ys = torch.full(size=(1, 1), fill_value=self.bos_id, dtype=torch.long).type_as(src)

        while True:
            tgt_mask = subsequent_mask(ys.size(1)).type_as(src_mask)

            # print(src_mask)
            # print(tgt_mask)
            # print(ys)

            out = self.model.decode(
                memory=memory,
                src_mask=src_mask,
                tgt=ys,
                tgt_mask=tgt_mask,
            )
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            print(torch.topk(prob, dim=1, k=5))
            next_word = next_word.item() # Use .item() instead of .data[0] (cleaner PyTorch syntax)
    
            # Append the predicted token
            ys = torch.cat(
                [ys, torch.full((1, 1), next_word, dtype=torch.long).type_as(src)], dim=1
            )
            
            # FIX 3: Stop generating early if the model signals it is finished
            if next_word == self.eos_id:
                break

        y = ys[0]
        print(y.tolist())
        return self._detokenize(y.tolist())


def make_translator(bos_id, eos_id):
    current_dir = Path(__file__).resolve().parent
    cache_dir = join(current_dir.parent, "cache")

    tokenizer_path = join(cache_dir, "bpe.json")
    weights_path = join(cache_dir, "en-es-machine-translator.pt")
    config_path = join(current_dir, "config.yaml")

    translator = MachineTranslator(bos_id, eos_id, tokenizer_path, weights_path, config_path)

    return translator
