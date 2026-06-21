import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.matrix_dataset import DynamicMatrixDataset
from src.models.delta_predictor import DeltaPredictor
from src.training.trainer import BaseTrainer
from src.utils.config_loader import load_experiment_config
from src.utils.seed import set_seed

class DeltaTrainer(BaseTrainer):
    def training_step(self, batch):
        A = batch['A'].to(self.device)
        B = batch['B'].to(self.device)
        C_true = batch['C'].to(self.device)
        
        delta_true = C_true - A
        delta_pred = self.model(A, B)
        return self.criterion(delta_pred, delta_true)
        
    def validation_step(self, batch):
        A = batch['A'].to(self.device)
        B = batch['B'].to(self.device)
        C_true = batch['C'].to(self.device)
        
        delta_true = C_true - A
        delta_pred = self.model(A, B)
        return self.criterion(delta_pred, delta_true)

def run_delta_training(size=32):
    config = load_experiment_config("configs/default.yaml", "configs/delta_learning.yaml")
    set_seed(42)
    device = torch.device(config.get('device', 'cpu'))
    
    train_dataset = DynamicMatrixDataset(config['data']['train_samples'], size=size, matrix_type="random")
    val_dataset = DynamicMatrixDataset(config['data']['val_samples'], size=size, matrix_type="random")
    
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    
    model = DeltaPredictor(size=size, hidden_dims=config['model']['hidden_dims'])
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    criterion = nn.MSELoss()
    
    save_dir = f"checkpoints/delta_model/size_{size}"
    log_dir = f"logs/tensorboard/delta_model/size_{size}"
    
    trainer = DeltaTrainer(model, optimizer, criterion, device, save_dir, log_dir)
    print(f"Starting Delta Predictor training on {device}...")
    trainer.fit(train_loader, val_loader, epochs=config['training']['epochs'], patience=config['training']['early_stopping_patience'])
    return model

if __name__ == "__main__":
    run_delta_training()
