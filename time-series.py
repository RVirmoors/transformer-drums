# https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
# https://github.com/KasperGroesLudvigsen/influenza_transformer
# https://github.com/oliverguhr/transformer-time-series-prediction
# https://github.com/AIStream-Peelout/flow-forecast


import torch
import torch.nn as nn
from torch.nn import functional as F

import sys
sys.path.append('transformer')
from transformer.tst import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# define a Transformer in PyTorch and use it to predict a single value in a given sequence

pos_enc = "index" # "index" or "linear"

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        if pos_enc == "index":
            self.pos_enc = nn.Embedding(10, d_model)
        else:
            self.pos_enc = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):            
        pos_src = torch.arange(0, len(src), dtype=torch.long).to(device)
        pos_tgt = torch.round((tgt - 0.1)*10).long().squeeze()
        # tgt = torch.randn_like(tgt)
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        if pos_enc == "index":
            src = src + self.pos_enc(pos_src)
            tgt = tgt + self.pos_enc(pos_tgt)
        else:
            src = src + self.pos_enc(src)
            tgt = tgt + self.pos_enc(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output
    
# model = TransformerModel(1, 1, d_model=8, nhead=2).to(device)
model = Transformer(1, 8, 1, 4, 4, 4, 2).to(device)
    
# a toy univariate time series dataset

data = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).unsqueeze(dim=-1).to(device)
data = torch.stack([data, data, data, data, data, data, data]) # 7 batches

input_seq = torch.Tensor([[0.1, 0.2, 0.3], 
                          [0.2, 0.3, 0.4],
                          [0.3, 0.4, 0.5],
                          [0.4, 0.5, 0.6],
                          [0.5, 0.6, 0.7],
                          [0.6, 0.7, 0.8],
                          [0.7, 0.8, 0.9]]).unsqueeze(dim=-1).to(device)

output = torch.Tensor([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).unsqueeze(dim=-1).unsqueeze(dim=-1).to(device)

#src_mask = model.generate_square_subsequent_mask(input_seq.size(0)).to(device)
#tgt_mask = model.generate_square_subsequent_mask(output_seq.size(0)).to(device)

# training code

def train():
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    print("Training...")
    for epoch in range(2000):
        for i in range(len(input_seq)):
            optimizer.zero_grad()
            out = model(input_seq[i])
            loss = F.mse_loss(out[-1], output[i][0])
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print("Epoch", epoch, " - " , loss.item())

train()

print("Last value is the usable prediction:", model(input_seq[0]))
print(model(input_seq[5]))