import torch
import torch.nn.functional as F
import time
from model import GPT, GPTConfig
from dataloader import DataLoader
import math
import os

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setting up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 # check if DDP is enabled
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "DDP requires CUDA."
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # set device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    # added after video, pytorch can be serious about it's device vs. device_type distinction
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # torch.manual_seed(1337)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(1337)

    """"
    --------------------------------------------------
    MAKE CHANGES WHILE TRAINING
    --------------------------------------------------
    """

    # initialize the hyperparameters.
    max_lr = 6e-4
    min_lr = max_lr * 0.1  # 10% of max_lr.
    warmup_steps = 22888 # 375000000 / 2^14 = 22888
    max_steps = 610351 # 10e9 / 2^14 = 610351

    # total_batch_size = 524288 # 2^19, ~0.5M tokens (batch size) <---- in the paper
    total_batch_size = 16384 # 2^14
    B = 1 
    T = 1024
    
    assert(
        total_batch_size % (B * T * ddp_world_size) == 0
        ), "total_batch_size must be divisible by (B * T * ddp_world_size)."
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(
            f"total_batch_size: {total_batch_size} => gradient_accum_steps: {grad_accum_steps}"
            )
    
    train_loader = DataLoader(
        B=B, 
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="train"
        )

    model = GPT(GPTConfig(vocab_size=50304)) # increasing vocab size to nice number
    model.to(device)
    # This will cost initial compilation time but increase the training speed
    # This will see the entire code which allows the compiler to know which code
    # executes next
    # Didn't have enough gpu to test while sequence length was 1024
    # model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model # underlying model
    torch.set_float32_matmul_precision('high')

    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

    # The device.type attribute will be the string "cuda" or "cpu", which is what the method is expecting. 
    # This change will allow the code to correctly detect when fused AdamW should be used.
    optimizer = raw_model.configure_optimizer(weight_decay=0.1, lr=6e-4, device=device_type, process_rank=ddp_rank)
    
    for i in range(50):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # gpu didn't support bfloat so prompted to default float32
            # with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                # in ddp, we need to sync gradients only at the last micro-step
                # default behavior of loss.backward() is at every micro-step
                model.require_backward_grad_sync = (
                    micro_step == grad_accum_steps - 1
                    )
            loss.backward()
            if ddp:
                # reduce the gradients across all processes
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        # prevent model from getting sudden spike in gradient magnitude
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the lr for this iter
        lr = cosine_lr_schedule(max_lr, min_lr, warmup_steps, max_steps, i)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        # print(f"step: {i}, loss: {loss.item()}")
        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)*1000
        tokens_per_sec = (
            train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
            ) / (t1 - t0) # no. of tokens processed per second
        if master_process:
            print(
            f"step: {i:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )

    # destroy process group if DDP is enabled
    if ddp:
        destroy_process_group()

    # import sys; sys.exit(0)
 
 
if __name__ == "__main__":
    train()



# instructions to run the code.
# simple run with one GPU: python train_gpt.py
# run with DDP: torchrun --standalone --nproc_per_node=4 train.py