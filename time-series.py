# https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
# https://github.com/KasperGroesLudvigsen/influenza_transformer
# https://github.com/oliverguhr/transformer-time-series-prediction
# https://github.com/AIStream-Peelout/flow-forecast


import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# a toy univariate time series dataset

data = torch.Tensor([[0., 0.1],
                    [0.5, 0.2],
                    [0., 0.3],
                    [0.25, 0.4],
                    [0.5, 0.5],
                    [0.75, 0.6],
                    [0., 0.7],
                    [0.25, 0.8],
                    [0.5, 0.9],
                    [0.75, 1.0]]).to(device)

data = torch.stack([data, data, data, data, data, data, data, data]) # 8 batches -> ENCODER
# mask future
for i in range(len(data)):
    for j in range(len(data[i])):
        if j > i+1:
            data[i][j][1] = -np.inf
#print(data)
#exit()

input_seq = torch.stack([data[0][i:i+2] for i in range(len(data[0])-2)]) # 8 batches, 2 timesteps -> DECODER

output = torch.Tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).unsqueeze(dim=-1).unsqueeze(dim=-1).to(device)

# define a Transformer in PyTorch and use it to predict a single value in a given time-series
# data entries are pairs: [time, value]

pos_enc = "regular" # "regular" or "embedding" or "linLayer"

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.pos_embed = nn.Embedding(input_dim, d_model)
        self.pos_linLayer = nn.Linear(1, d_model)
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead)
        self.fc = nn.Linear(d_model, output_dim)

    def generatePE(self, x: torch.Tensor):
        if pos_enc == "embedding":
            return self.pos_embed(x)
        elif pos_enc == "linLayer":
            return self.pos_linLayer(x)
        elif pos_enc == "regular":
            PE = torch.zeros((len(x), self.d_model))
            pos = x[:,0]
            PE = torch.sin(pos * 2 * np.pi)
            #PE = PE.repeat((2, self.d_model))
            print(x, pos, PE)
            return PE

    def forward(self, src, tgt):
        src_posenc = self.generatePE(src)
        src_tgtenc = self.generatePE(tgt)        
        src = self.embedding(src) + src_posenc
        tgt = self.embedding(tgt) + src_tgtenc
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output
    
model = TransformerModel(2, 2, d_model=8, nhead=2).to(device)
    
# training code

def train():
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    print("Training...")
    for epoch in range(8000):
        for i in range(len(input_seq)):
            optimizer.zero_grad()
            out = model(data[i], input_seq[i])
            loss = F.mse_loss(out[-1], output[i][0])
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print("Epoch", epoch, " - " , loss.item())

train()

print("Last value is the usable prediction:", model(data[0], input_seq[0]))
print(model(data[3], input_seq[3]))
print(model(data[6], input_seq[6]))