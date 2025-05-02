import tiktoken
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

def generate_text(model, prompt, max_length=30, num_return_sequences=5):
    enc = tiktoken.get_encoding('gpt2')
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
    # with open('../data/input.txt', 'r') as f:
    #     text = f.read()
    # text = text[:1000]
    # tokens = enc.encode(text)
    # B, T = 4, 32
    # buf = torch.tensor(tokens[:B*T + 1])
    # x = buf[:-1].view(B,T)
    # y = buf[1:].view(B,T)
    

    # Move the tensor to the CPU (or GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = tokens.to(device)
    model.to(device)
    # logits, loss = model(x,y)
    # print(loss)
    # import sys; sys.exit(0)

    # code from Andrej Karpathy's nanoGPT:
    # generate! right now x is (B, T) where B = 5, T = 8
    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            logits, _ = model(x) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)

if __name__ == "__main__":
    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig())
    model.eval()

    # set the seed to 42
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    prompt = "Hello, I'm a language model,"
    generate_text(model, prompt, max_length=30, num_return_sequences=5)