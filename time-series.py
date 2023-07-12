# https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
# https://github.com/KasperGroesLudvigsen/influenza_transformer
# https://github.com/oliverguhr/transformer-time-series-prediction
# https://github.com/AIStream-Peelout/flow-forecast

pos_enc = "regular" # "regular" or "embedding" or "linLayer"
train = True
load_model = 'ckpt.pt'

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
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

data = torch.stack([data, data, data, data, data, data, data, data, data]) # 9 batches -> ENCODER

input_seq = torch.stack([data[i][i] for i in range(len(data[0])-1)]).unsqueeze(dim=1) # 9 batches, 2 timesteps -> DECODER
# print(input_seq)
# exit()

output = torch.stack([data[i][i+1] for i in range(len(data[0])-1)]).unsqueeze(dim=1)
# print(output)
# exit()

# define a Transformer in PyTorch and use it to predict a single value in a given time-series
# data entries are pairs: [time, value]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.pos_embed = nn.Embedding(input_dim, d_model)
        self.pos_linLayer = nn.Linear(1, d_model)
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead)
        self.fc = nn.Linear(d_model, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def generatePE(self, x: torch.Tensor):
        if pos_enc == "embedding":
            return self.pos_embed(x)
        elif pos_enc == "linLayer":
            return self.pos_linLayer(x)
        elif pos_enc == "regular":
            PE = torch.zeros((len(x), self.d_model))
            pos = x[:,0]
            PE = torch.sin(pos * 2 * np.pi).unsqueeze(-1)
            PE = PE.repeat((1, self.d_model))
            # print(PE)
            return PE

    def forward(self, src, tgt):
        src_posenc = self.generatePE(src)
        src_tgtenc = self.generatePE(tgt)
        src = self.embedding(src) + src_posenc
        tgt = self.embedding(tgt) + src_tgtenc
        # print(src)
        src_mask = self.transformer.generate_square_subsequent_mask(src.size(0)).to(device)
        # print(src_mask)
        output = self.transformer(src, tgt, src_mask=src_mask)
        # print(output)
        output = self.fc(output)
        return self.sigmoid(output)
    
model = TransformerModel(2, 2, d_model=8, nhead=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
# training code

def train():
    model.train()
    print("Training...")
    for epoch in range(80):
        for i in range(len(input_seq)):
            optimizer.zero_grad()
            out = model(data[i], input_seq[i])
            # print((out, output[i]))
            loss = F.mse_loss(out, output[i])
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print("Epoch", epoch, " - " , loss.item())

if load_model and not os.path.isfile(load_model):
    print("No model found at", load_model, "-- starting from scratch...")
    load_model = None

if load_model:
    print("Resuming training from", load_model, "...")
    checkpoint = torch.load(load_model, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    
if train:
    train()

print(model(data[0], input_seq[0]))
print(model(data[3], input_seq[3]))
print(model(data[6], input_seq[6]))


checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }
torch.save(checkpoint, 'ckpt.pt')