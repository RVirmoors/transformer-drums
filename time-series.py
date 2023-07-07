# https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
# https://github.com/KasperGroesLudvigsen/influenza_transformer
# https://github.com/oliverguhr/transformer-time-series-prediction
# https://github.com/AIStream-Peelout/flow-forecast


import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# define a Transformer in PyTorch and use it to predict a single value in a given sequence

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead):
        super(TransformerModel, self).__init__()
        self.pos_enc = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead)
        self.fc = nn.Linear(d_model, output_dim)
# define a Transformer in PyTorch and use it to predict a single value in a given sequence

    def forward(self, src, tgt):
        src = self.pos_enc(src)
        tgt = self.pos_enc(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output

model = TransformerModel(1, 1, d_model=8, nhead=2)
    
# a toy univariate time series dataset

data = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).unsqueeze(dim=-1).to(device)
# print(data)
input_seq = data[:-1] 
output_seq = data[1:]

#src_mask = model.generate_square_subsequent_mask(input_seq.size(0)).to(device)
#tgt_mask = model.generate_square_subsequent_mask(output_seq.size(0)).to(device)

# training code

def train():
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    print("Training...")
    for epoch in range(20000):
        optimizer.zero_grad()
        out = model(data, data)
        loss = F.mse_loss(out, data)
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0:
            print("Epoch", epoch, " - " , loss.item())

train()

print(model(data, data))