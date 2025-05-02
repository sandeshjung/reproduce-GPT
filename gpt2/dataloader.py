import tiktoken
import torch


class DataLoader:

    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("../data/input.txt", "r") as f:
            text = f.read()
        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"total tokens: {len(self.tokens)}")
        print(f"1 epoch has {len(self.tokens) // (B * T)} batches.")

        self.cur_pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        tokens_tensors = self.tokens[self.cur_pos : self.cur_pos + B * T + 1]
        x = (tokens_tensors[:-1]).view(B, T)  # (batch, T)
        y = (tokens_tensors[1:]).view(B, T)
        self.cur_pos += B * T  # move the pointer.

        if self.cur_pos + (B * T + 1) >= len(self.tokens):
            self.cur_pos = 0

        return x, y