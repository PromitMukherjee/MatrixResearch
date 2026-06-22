import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.matrix_dataset import DynamicMatrixDataset
from src.models.autoencoder import MatrixAutoencoder
from src.training.trainer import BaseTrainer
from src.utils.config_loader import load_experiment_config
from src.utils.seed import set_seed

class AutoencoderTrainer(BaseTrainer):
    def training_step(self, batch):
        A = batch['A'].to(self.device)
        A_rec = self.model(A)
        return self.criterion(A_rec, A)
        
    def validation_step(self, batch):
        A = batch['A'].to(self.device)
        A_rec = self.model(A)
        return self.criterion(A_rec, A)

def run_autoencoder_training(size=32, latent_dim=128):
    config = load_experiment_config("configs/default.yaml", "configs/autoencoder.yaml")
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_dataset = DynamicMatrixDataset(config['data']['train_samples'], size=size, matrix_type="low_rank")
    val_dataset = DynamicMatrixDataset(config['data']['val_samples'], size=size, matrix_type="low_rank")
    
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    
    model = MatrixAutoencoder(
        size=size, 
        latent_dim=latent_dim, 
        hidden_dims=config['model']['hidden_dims']
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    
    criterion = nn.MSELoss()
    
    save_dir = f"checkpoints/autoencoder/size_{size}_latent_{latent_dim}"
    log_dir = f"logs/tensorboard/autoencoder/size_{size}_latent_{latent_dim}"
    
    trainer = AutoencoderTrainer(model, optimizer, criterion, device, save_dir, log_dir)
    print(f"Starting autoencoder training on {device}...")
    trainer.fit(train_loader, val_loader, epochs=config['training']['epochs'], patience=config['training']['early_stopping_patience'])
    
    return model

if __name__ == "__main__":
    run_autoencoder_training()
