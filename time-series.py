# define a Transformer in PyTorch and use it to predict a single value in a given sequence

import torch.nn as nn
class Transformer(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, n_layer, dropout=0.5):
        super(Transformer, self).__init__()
        self.encoder = Encoder(n_feature, n_hidden, n_layer, dropout)
        self.decoder = Decoder(n_hidden, n_output, n_layer, dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
