import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.matrix_dataset import DynamicMatrixDataset
from src.models.latent_multiplier import LatentMultiplier
from src.models.autoencoder import MatrixAutoencoder
from src.training.trainer import BaseTrainer
from src.utils.config_loader import load_experiment_config
from src.utils.seed import set_seed

class LatentMultiplierTrainer(BaseTrainer):
    def __init__(self, model, autoencoder, optimizer, criterion, device, save_dir, log_dir):
        super().__init__(model, optimizer, criterion, device, save_dir, log_dir)
        self.autoencoder = autoencoder.to(device)
        self.autoencoder.eval() # freeze autoencoder
        
    def training_step(self, batch):
        A = batch['A'].to(self.device)
        B = batch['B'].to(self.device)
        C_true = batch['C'].to(self.device)
        
        with torch.no_grad():
            z_A = self.autoencoder.encode(A)
            
        C_pred = self.model(z_A, B)
        return self.criterion(C_pred, C_true)
        
    def validation_step(self, batch):
        A = batch['A'].to(self.device)
        B = batch['B'].to(self.device)
        C_true = batch['C'].to(self.device)
        
        with torch.no_grad():
            z_A = self.autoencoder.encode(A)
            C_pred = self.model(z_A, B)
        return self.criterion(C_pred, C_true)

def run_latent_training(size=32, latent_dim=128, ae_checkpoint=None):
    config = load_experiment_config("configs/default.yaml", "configs/latent_multiplication.yaml")
    set_seed(42)
    device = torch.device(config.get('device', 'cpu'))
    
    # Needs pre-trained autoencoder
    autoencoder = MatrixAutoencoder(size=size, latent_dim=latent_dim, hidden_dims=[512, 256])
    if ae_checkpoint and os.path.exists(ae_checkpoint):
        autoencoder.load_state_dict(torch.load(ae_checkpoint, map_location=device))
    else:
        print("WARNING: Using untrained autoencoder for LatentMultiplier!")
        
    train_dataset = DynamicMatrixDataset(config['data']['train_samples'], size=size, matrix_type="random")
    val_dataset = DynamicMatrixDataset(config['data']['val_samples'], size=size, matrix_type="random")
    
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    
    model = LatentMultiplier(size=size, latent_dim=latent_dim, hidden_dims=config['model']['hidden_dims'])
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    criterion = nn.MSELoss()
    
    save_dir = f"checkpoints/latent_multiplier/size_{size}_latent_{latent_dim}"
    log_dir = f"logs/tensorboard/latent_multiplier/size_{size}_latent_{latent_dim}"
    
    trainer = LatentMultiplierTrainer(model, autoencoder, optimizer, criterion, device, save_dir, log_dir)
    print(f"Starting Latent Multiplier training on {device}...")
    trainer.fit(train_loader, val_loader, epochs=config['training']['epochs'], patience=config['training']['early_stopping_patience'])
    return model
