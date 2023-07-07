import torch
import torch.nn as nn
from torch.nn import functional as F

cuda = torch.cuda.is_available()
if cuda:
    device = torch.device('cuda')
    print("using CUDA")
else:
    device = torch.device('cpu')

# define a Transformer in PyTorch and use it to predict a single value in a given sequence

model = nn.Transformer(d_model=1, nhead=1)
    
# a toy univariate time series dataset

data = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).unsqueeze(dim=-1)

# training code

def train():
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Training...")
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data, data)
        loss = F.mse_loss(out, data)
        loss.backward()
        optimizer.step()
        print(loss.item())

train()

print(model(data, data))