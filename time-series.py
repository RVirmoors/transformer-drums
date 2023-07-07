import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define a simple Transformer-based model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, src):
        output = self.transformer(src, src)
        output = self.fc(output)
        return output


# Define the input and output dimensions
input_dim = 1
output_dim = 1

# Create the model instance
model = TransformerModel(input_dim, output_dim, d_model=16, nhead=2).to(device)

# Toy univariate time series dataset
data = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).unsqueeze(dim=-1).to(device)

# Training code
def train():
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Training...")
    for epoch in range(1000):
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print("Epoch", epoch, "- Loss:", loss.item())

train()

# Test the trained model
with torch.no_grad():
    model.eval()
    output = model(data)
    print("Output:")
    print(output)
