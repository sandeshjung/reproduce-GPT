import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from dataloader import DataLoader

def train():
    # initialize the model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT(GPTConfig())
    model.to(device)

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    train_loader = DataLoader(B=4, T=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"step: {i}, loss: {loss.item()}")
 
 
if __name__ == "__main__":
    train()