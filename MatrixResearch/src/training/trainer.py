import torch
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class BaseTrainer:
    def __init__(self, model, optimizer, criterion, device, save_dir, log_dir):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.writer = SummaryWriter(log_dir)
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training", leave=False):
            loss = self.training_step(batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(dataloader)
        
    def val_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation", leave=False):
                loss = self.validation_step(batch)
                total_loss += loss.item()
        return total_loss / len(dataloader)
        
    def training_step(self, batch):
        raise NotImplementedError
        
    def validation_step(self, batch):
        raise NotImplementedError
        
    def fit(self, train_loader, val_loader, epochs, patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.val_epoch(val_loader)
            
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            
            print(f"Epoch {epoch:03d}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pt"))
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
        # Load best model
        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, "best_model.pt")))
        return self.model
