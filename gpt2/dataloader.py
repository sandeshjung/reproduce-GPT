import tiktoken
import torch
import numpy as np

def load_tokens(filename):
    """Function to load the tokens from the shard file."""
    tokens = np.load(filename)
    tokens = tokens.astype(np.int32)
    tok_tensor = torch.tensor(tokens, dtype=torch.long)
    return tok_tensor

class DataLoader:

    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in ["train", "val"], "split must be either 'train' or 'val'."

        # load the tokens from the shard files.
        data_root = "../data/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split: {split}"

        if self.process_rank == 0:
            print(f"loading {len(shards)} shards for split: {split}")
        self.reset()
        # print(f"1 epoch has {len(self.tokens) // (B * T)} batches.")

    def reset(self):
        self.cur_shard = 0
        self.tokens = load_tokens(self.shards[self.cur_shard])
        # self.cur_pos = 0
        self.cur_pos = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        tokens_tensors = self.tokens[self.cur_pos : self.cur_pos + B * T + 1]
        x = (tokens_tensors[:-1]).view(B, T)  # (batch, T)
        y = (tokens_tensors[1:]).view(B, T)
        self.cur_pos += B * T * self.num_processes # move the pointer.

        if self.cur_pos + (B * T * self.num_processes + 1) > len(self.tokens):
            self.cur_shard = (self.cur_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.cur_shard])
            self.cur_pos = self.B * self.T * self.num_processes

        return x, y