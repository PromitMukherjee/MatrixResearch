import torch
import torch.nn as nn

class LatentOperator(nn.Module):
    """
    Experiment 7: Latent Operator.
    Determines whether matrix multiplication can occur entirely in latent space.
    Input: Encoded(A), Encoded(B)
    Output: Encoded(C)
    """
    def __init__(self, latent_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Takes Z_A and Z_B
        input_dim = latent_dim * 2
        
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Linear(in_d, h_d))
            layers.append(nn.ReLU())
            in_d = h_d
        layers.append(nn.Linear(in_d, latent_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, z_A, z_B):
        # Concatenate encoded representations
        x = torch.cat([z_A, z_B], dim=1)
        # Predict Z_C
        z_C_pred = self.net(x)
        return z_C_pred
