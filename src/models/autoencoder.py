import torch
import torch.nn as nn

class MatrixAutoencoder(nn.Module):
    def __init__(self, size, latent_dim, hidden_dims=[512, 256]):
        super().__init__()
        self.size = size
        input_dim = size * size
        
        # Encoder
        enc_layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            enc_layers.append(nn.Linear(in_d, h_d))
            enc_layers.append(nn.ReLU())
            in_d = h_d
        enc_layers.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        
        # Decoder
        dec_layers = []
        in_d = latent_dim
        for h_d in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_d, h_d))
            dec_layers.append(nn.ReLU())
            in_d = h_d
        dec_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*dec_layers)
        
    def encode(self, x):
        batch_size = x.size(0)
        return self.encoder(x.view(batch_size, -1))

    def decode(self, z):
        batch_size = z.size(0)
        return self.decoder(z).view(batch_size, self.size, self.size)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
