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

data = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).unsqueeze(dim=-1).to(device)
print(data)

# training code

def train():
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Training...")
    for epoch in range(10000):
        optimizer.zero_grad()
        out = model(data, data)
        loss = F.mse_loss(out, data)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print("Epoch", epoch, " - " , loss.item())

train()

print(model(data, data))