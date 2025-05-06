import torch
import torch.nn.functional as F
import time
from model import GPT, GPTConfig
from dataloader import DataLoader
import math

# Code taken from: https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L354C5-L364C46
def cosine_lr_schedule(max_lr, min_lr, warmup_steps, max_steps, cur_step):
    """Cosine learning rate schedule with warm-up
    """

    # 1) linear warmup for warmup_iters steps
    if cur_step < warmup_steps:
        return max_lr * (cur_step + 1) / warmup_steps

    # 2) if cur_step > lr_decay_iters, return min learning rate
    if cur_step > max_steps:
        return min_lr

    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (cur_step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0

    return min_lr + coeff * (max_lr - min_lr)

"""
Cosine decay for lr down to 10% of its value
after 260B tokens, training was continued at 10% of the original lr
^^^ this isn't implemented in this code. 
"""

def train():
    # initialize the model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT(GPTConfig(vocab_size=50304)) # increasing vocab size to nice number
    model.to(device)
    # This will cost initial compilation time but increase the training speed
    # This will see the entire code which allows the compiler to know which code
    # executes next
    # Didn't have enough gpu to test while sequence length was 1024
    # model = torch.compile(model)

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    train_loader = DataLoader(B=1, T=1024)
    torch.set_float32_matmul_precision('high')

    # initialize the hyperparameters.
    max_lr = 6e-4
    min_lr = max_lr * 0.1  # 10% of max_lr.
    warmup_steps = 10
    max_steps = 50

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    for i in range(50):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # gpu didn't support bfloat so prompted to default float32
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
        loss.backward()
        # prevent model from getting sudden spike in gradient magnitude
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the lr for this iter
        lr = cosine_lr_schedule(max_lr, min_lr, warmup_steps, max_steps, i)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        # print(f"step: {i}, loss: {loss.item()}")
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)*1000
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
        print(f"step {i:4d}, loss: {loss.item()}, lr: {lr:.4e}, norm: {norm:.4f} dt: {dt:.2f}ms, tok/sec: {tokens_per_sec}")

    # import sys; sys.exit(0)
 
 
if __name__ == "__main__":
    train()