import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# define a Transformer module for univariate time series prediction
class UnivariateTransformer(nn.Module):
    def __init__(self, d_model):
        super(UnivariateTransformer, self).__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, src):
        output = self.linear(src)
        return output

# create the model with appropriate input and output dimensions
model = UnivariateTransformer(d_model=1).to(device)

# a toy univariate time series dataset
data = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).unsqueeze(dim=-1).to(device)
print(data)

# training code
def train():
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Training...")
    for epoch in range(7000):
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data)
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0:
            print("Epoch", epoch, " - ", loss.item())

train()

print(model(data))
