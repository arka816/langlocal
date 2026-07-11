from collections import Counter


# class BPETokenizer:
#     """Naive implementation of BPE Tokenizer"""

#     _NUM_RESERVED_TOKENS = 3

#     def __init__(self, corpus: str, vocab_size:int):
#         if len(corpus) < vocab_size:
#             raise ValueError("Corpus size must be larger than target vocabulary size.")

#         self.tokens = self._split(corpus)
#         self.merges = {}
#         self.vocab_size = vocab_size

#     def _split(self, corpus: str) -> List:
#         return list(corpus.encode("utf-8"))
        
#     def _count(self, tokens):
#         counter = Counter()

#         for pair in zip(tokens, tokens[1:]):
#             counter[pair] += 1

#         return counter

#     def _merge(self, tokens, pair):
#         a, b = pair

#         i, j = 0, 1

#         while j < len(tokens):
#             if tokens[i] == a and tokens[j] == b:
#                 tokens[i] = a + b
#                 del tokens[j]
#             i += 1
#             j += 1

#         return tokens

#     def _token_to_idx(self, token):
#         return self._NUM_RESERVED_TOKENS + self.tokens.index(token)

#     def _idx_to_token(self, idx):
#         return self.tokens[idx - self._NUM_RESERVED_TOKENS]


#     def train(self) -> None:
#         index = 0

#         while len(self.tokens) > self.vocab_size:
#             # count
#             counter = self._count(self.tokens)

#             # pick most frequent pair
#             pair = counter.most_common()[0][0]

#             self.merges[pair] = index
#             index += 1

#             self.tokens = self._merge(self.tokens, pair)


#     def encode(self, text):
#         tokens = self._split(text)

#         if len(tokens) < 2:
#             return tokens

#         while True:
#             counter = self._count(tokens)

#             pair = min(counter, key=lambda p: self.merges.get(p, float('inf')))
#             if pair not in self.merges:
#                 break

#             tokens = self._merge(tokens, pair)

#         return [self._token_to_idx(token) for token in tokens]

#     def decode(self, idxs):
#         return [self._idx_to_token(idx) for idx in idxs]

class BPETokenizer:
    def __init__(self, target_vocab_size):
        self.target_vocab_size = target_vocab_size
        self.merges = {}  # Map of (int, int) -> int
        self.vocab = {}   # Map of int -> bytes

    def _count(self, ids):
        """Count frequencies of all adjacent pairs."""
        return Counter(zip(ids, ids[1:]))

    def _merge(self, ids, pair, idx):
        """Replace occurrences of `pair` with the new token `idx`."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text):
        # 1. Initialize base vocabulary with 256 raw UTF-8 bytes
        tokens = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        # Guard against a target smaller than our base vocabulary
        if self.target_vocab_size <= len(self.vocab):
            print(f"Target size {self.target_vocab_size} is already met by base bytes.")
            return

        current_token_id = 256

        # 2. Keep loop running until your exact vocabulary size objective is reached
        while len(self.vocab) < self.target_vocab_size:
            stats = self._count(tokens)
            
            # Stop if no remaining adjacent pairs exist to merge
            if not stats:
                print(f"Training stopped early at vocab size {len(self.vocab)}. No more pairs.")
                break
                
            # Grab the single most frequent adjacent token pair
            best_pair = max(stats, key=stats.get)
            
            # Register the new ID rule
            self.merges[best_pair] = current_token_id
            self.vocab[current_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Apply the new merge to compress the sequence
            tokens = self._merge(tokens, best_pair, current_token_id)
            current_token_id += 1

        print(f"Training completed. Final vocabulary size: {len(self.vocab)}")

    def encode(self, text):
        """Convert string to token IDs using your learned vocab map."""
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self._count(tokens)
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break 
            tokens = self._merge(tokens, pair, self.merges[pair])
        return tokens

    def decode(self, ids):
        """Convert token IDs back to a readable string."""
        byte_segments = [self.vocab.get(idx, b'') for idx in ids]
        return b"".join(byte_segments).decode("utf-8", errors="replace")

