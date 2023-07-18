

pos_enc = "regular" # "regular" or "linLayer"
training = True
load_model = 'drums.pt'

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import numpy as np

from utils import EarlyStopper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# a toy drums series dataset: bar_pos, K, S, H, tau

data = torch.Tensor([[0., 0.8, 0, 0.8, 0],
                    [0.5, 0, 1, 0.9, 0.007],
                    [0., 0.6, 0, 0.8, 0.002],
                    [0.25, 0, 0, 0.4, -0.01],
                    [0.5, 0, 1, 0.7, 0.002],
                    [0.75, 0, 0, 0.45, -0.005],
                    [0., 0.7, 0, 0.9, 0.001],
                    [0.25, 0.6, 0, 0.8, -0.002],
                    [0.5, 0.2, 0.9, 0.8, 0.005],
                    [0.75, 0.5, 0, 0.6, 0.002]]).to(device)

data = torch.stack([data, data, data, data, data, data, data, data]) # 8 batches -> ENCODER

input_seq = torch.stack([data[i][i:i+2] for i in range(len(data[0])-2)]) # 8 batches, 2 timesteps -> DECODER
# print(input_seq)
# exit()

output = torch.stack([data[i][i+1:i+3] for i in range(len(data[0])-2)])
# print(output)
# exit()

# define a Transformer in PyTorch and use it to predict a single value in a given time-series
# data entries are pairs: [time, value]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.pos_linLayer = nn.Linear(input_dim, d_model)
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead)
        self.fc = nn.Linear(d_model, output_dim)

    def generatePE(self, x: torch.Tensor):
        if pos_enc == "linLayer":
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
        return output
    
model = TransformerModel(5, 5, d_model=64, nhead=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=4e-5)
    
# training code

early_stopper = EarlyStopper(patience=10, min_delta=0.3)

def train():
    global checkpoint
    model.train()
    min_loss = np.inf
    print("Training...")
    for epoch in range(5000):
        total_loss = 0
        for i in range(len(input_seq)):
            optimizer.zero_grad()
            out = model(data[i], input_seq[i])
            # print((out, output[i]))
            loss = F.mse_loss(out, output[i])
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
        if total_loss < min_loss:
            min_loss = total_loss
            print("New min loss:", min_loss)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
            torch.save(checkpoint, load_model)
        if early_stopper.early_stop(total_loss):            
            break
        if epoch % 10 == 0:
            print("Epoch", epoch, " - " , total_loss)

load = True
if load_model and not os.path.isfile(load_model):
    print("No model found at", load_model, "-- starting from scratch...")
    load = False

if load:
    print("Resuming from", load_model, "...")
    checkpoint = torch.load(load_model, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

if training:
    train()
    model.load_state_dict(checkpoint['model'])

print(model(data[0], input_seq[0]), "should be 0., 0.6, 0, 0.8, 0.002")
print(model(data[3], input_seq[3]), "should be 0.75, 0, 0, 0.45, -0.005")
print(model(data[6], input_seq[6]), "should be 0.5, 0.2, 0.9, 0.8, 0.005")

print("OUT OF SAMPLE TESTS:")
print("0., 0.6, 0.9, 0, -0.01 ->", model(data[-1], torch.Tensor([[0., 0.6, 0.9, 0, -0.01]]).to(device)))
print("0.75, 0, 0, 0.6, 0.01 ->", model(data[-1], torch.Tensor([[0.75, 0, 0, 0.6, 0.01]]).to(device)))
print("0.5, 0, 0.9, 0.85, -0.001 ->", model(data[-1], torch.Tensor([[0.5, 0, 0.9, 0.85, -0.001]]).to(device)))