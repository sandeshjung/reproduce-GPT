import tiktoken
import torch


class DataLoader:

    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open("../data/input.txt", "r") as f:
            text = f.read()
        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens)

        if self.process_rank == 0:
            print(f"total tokens: {len(self.tokens)}")
        # print(f"1 epoch has {len(self.tokens) // (B * T)} batches.")

        # self.cur_pos = 0
        self.cur_pos = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        tokens_tensors = self.tokens[self.cur_pos : self.cur_pos + B * T + 1]
        x = (tokens_tensors[:-1]).view(B, T)  # (batch, T)
        y = (tokens_tensors[1:]).view(B, T)
        self.cur_pos += B * T * self.num_processes # move the pointer.

        if self.cur_pos + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.cur_pos = self.B * self.T * self.num_processes

        return x, y