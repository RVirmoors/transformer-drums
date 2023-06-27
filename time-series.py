import torch
import torch.nn as nn

# define a Transformer in PyTorch and use it to predict a single value in a given sequence

class Transformer(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, n_layer, dropout=0.5):
        super(Transformer, self).__init__()
        self.encoder = Encoder(n_feature, n_hidden, n_layer, dropout)
        self.decoder = Decoder(n_hidden, n_output, n_layer, dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# a toy univariate time series dataset

data = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(data)