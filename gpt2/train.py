import torch
import torch.nn.functional as F
import time
from model import GPT, GPTConfig
from dataloader import DataLoader

def train():
    # initialize the model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT(GPTConfig())
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # gpu didn't support bfloat so prompted to default float32
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        # print(f"step: {i}, loss: {loss.item()}")
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)*1000
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
        print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec}")

    # import sys; sys.exit(0)
 
 
if __name__ == "__main__":
    train()