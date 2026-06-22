import torch
import torch.nn as nn

class LatentMultiplier(nn.Module):
    def __init__(self, size, latent_dim, hidden_dims=[1024, 512, 256]):
        super().__init__()
        self.size = size
        # Takes Z_A (encoded A) and raw matrix B
        input_dim = latent_dim + (size * size)
        
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Linear(in_d, h_d))
            layers.append(nn.ReLU())
            in_d = h_d
        layers.append(nn.Linear(in_d, size * size))
        self.net = nn.Sequential(*layers)
        
    def forward(self, z_A, B):
        batch_size = B.size(0)
        B_flat = B.view(batch_size, -1)
        x = torch.cat([z_A, B_flat], dim=1)
        c_pred = self.net(x)
        return c_pred.view(batch_size, self.size, self.size)
