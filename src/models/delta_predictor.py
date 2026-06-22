import torch
import torch.nn as nn

class DeltaPredictor(nn.Module):
    def __init__(self, size, hidden_dims=[1024, 512, 256]):
        super().__init__()
        self.size = size
        input_dim = (size * size) * 2
        
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Linear(in_d, h_d))
            layers.append(nn.ReLU())
            in_d = h_d
        layers.append(nn.Linear(in_d, size * size))
        self.net = nn.Sequential(*layers)
        
    def forward(self, A, B):
        batch_size = A.size(0)
        A_flat = A.view(batch_size, -1)
        B_flat = B.view(batch_size, -1)
        x = torch.cat([A_flat, B_flat], dim=1)
        delta_pred = self.net(x)
        return delta_pred.view(batch_size, self.size, self.size)
