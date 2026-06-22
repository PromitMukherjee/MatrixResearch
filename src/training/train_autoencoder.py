import os
import sys
import math
import json
import datetime
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.matrix_dataset import DynamicMatrixDataset
from src.models.autoencoder import MatrixAutoencoder
from src.training.trainer import BaseTrainer
from src.utils.config_loader import load_experiment_config
from src.utils.seed import set_seed
from src.utils.research_smoke_test import run_research_smoke_test

class AutoencoderTrainer(BaseTrainer):
    def training_step(self, batch):
        A = batch['A'].to(self.device)
        A_rec = self.model(A)
        return self.criterion(A_rec, A)
        
    def validation_step(self, batch):
        A = batch['A'].to(self.device)
        A_rec = self.model(A)
        loss = self.criterion(A_rec, A)
        mae = torch.nn.functional.l1_loss(A_rec, A)
        return loss, mae.item()

    def val_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_mae = 0
        with torch.no_grad():
            for batch in dataloader:
                loss, mae = self.validation_step(batch)
                total_loss += loss.item()
                total_mae += mae
        return total_loss / len(dataloader), total_mae / len(dataloader)

    def fit(self, train_loader, val_loader, epochs, patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.history = {'train_loss': [], 'val_loss': [], 'rmse': [], 'mae': []}
        
        for epoch in range(1, epochs + 1):
            train_loss = super().train_epoch(train_loader)
            val_loss, val_mae = self.val_epoch(val_loader)
            val_rmse = math.sqrt(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['rmse'].append(val_rmse)
            self.history['mae'].append(val_mae)
            
            print(f"\nEpoch {epoch:03d}")
            print(f"Train Loss : {train_loss:.6f}")
            print(f"Val Loss   : {val_loss:.6f}")
            print(f"RMSE       : {val_rmse:.6f}")
            print(f"MAE        : {val_mae:.6f}")
            print("-" * 30)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pt"))
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, "best_model.pt")))
        return self.model

def run_autoencoder_training(size=32, latent_dim=128):
    config = load_experiment_config("configs/default.yaml", "configs/autoencoder.yaml")
    set_seed(42)
    
    # --- Research Smoke Test ---
    train_matrix_type = config['data'].get('matrix_type', 'low_rank')
    smoke_passed = run_research_smoke_test(
        size=size,
        latent_dim=latent_dim,
        matrix_type=train_matrix_type,
        rank=4
    )
    if not smoke_passed:
        print("\nABORTING: Smoke test failed. Fix issues above before launching full training.")
        return None
    
    # --- GPU Diagnostics ---
    print("\n--- GPU Diagnostics ---")
    print(f"CUDA Available : {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA is not available. Training will be slow on CPU.")
    
    val_matrix_type = config['data'].get('matrix_type', 'low_rank')
    
    # --- Experiment Metadata Logging ---
    print("\n" + "=" * 60)
    print("Autoencoder Experiment Configuration")
    print("=" * 60)
    print(f"Train Matrix Type      : {train_matrix_type}")
    print(f"Validation Matrix Type : {val_matrix_type}")
    print(f"Matrix Size            : {size}x{size}")
    print(f"Latent Dimension       : {latent_dim}")
    print(f"Train Samples          : {config['data']['train_samples']}")
    print(f"Validation Samples     : {config['data']['val_samples']}")
    print(f"Batch Size             : {config['data']['batch_size']}")
    print(f"Learning Rate          : {config['training']['learning_rate']}")
    print(f"Device                 : {device}")
    print("=" * 60)

    # --- Dataset Diagnostics ---
    print("\n--- Dataset Diagnostics ---")
    sample_dataset = DynamicMatrixDataset(1, size=size, matrix_type=train_matrix_type)
    sample_A = sample_dataset[0]['A']
    print(f"Mean : {sample_A.mean().item():.6f}")
    print(f"Std  : {sample_A.std().item():.6f}")
    print(f"Min  : {sample_A.min().item():.6f}")
    print(f"Max  : {sample_A.max().item():.6f}")

    if train_matrix_type == "low_rank":
        print(f"Numerical Rank : {torch.linalg.matrix_rank(sample_A).item()}")
        
        # --- Matrix Structure Analysis ---
        U, S, V = torch.linalg.svd(sample_A)
        print(f"Effective Rank : {(S > 1e-5).sum().item()}")
        print(f"Largest Singular Value : {S[0].item():.6f}")
        print(f"Smallest Singular Value: {S[-1].item():.6f}")
        cond = S[0] / S[-1] if S[-1] > 0 else float('inf')
        print(f"Condition Number : {cond:.6f}")

    # --- Compression Statistics ---
    original_values = size * size
    print("\n--- Compression Statistics ---")
    print(f"Original Values : {original_values}")
    print(f"Latent Values : {latent_dim}")
    print(f"Compression Ratio : {original_values / latent_dim:.2f}x")
    print("-" * 30 + "\n")

    train_dataset = DynamicMatrixDataset(config['data']['train_samples'], size=size, matrix_type=train_matrix_type)
    val_dataset = DynamicMatrixDataset(config['data']['val_samples'], size=size, matrix_type=val_matrix_type)
    
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
    
    # --- Save Research Results & Latent Space Analysis ---
    print("\nSaving research results and visualizing latent space...")
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    report_dir = f"reports/autoencoder_results/{timestamp}_{size}_{latent_dim}"
    os.makedirs(report_dir, exist_ok=True)
    
    # History CSV
    history_df = pd.DataFrame(trainer.history)
    history_df.to_csv(os.path.join(report_dir, "training_history.csv"), index=False)
    
    # Final Metrics JSON
    final_metrics = {
        "final_train_loss": trainer.history['train_loss'][-1],
        "final_val_loss": trainer.history['val_loss'][-1],
        "final_rmse": trainer.history['rmse'][-1],
        "final_mae": trainer.history['mae'][-1],
        "epochs_trained": len(trainer.history['train_loss'])
    }
    with open(os.path.join(report_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    # Pick a sample batch
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        A_sample = sample_batch['A'].to(device)
        latent_vectors = model.encode(A_sample)
        A_rec = model.decode(latent_vectors)
        
    latent_np = latent_vectors.cpu().numpy()
    latent_var = latent_np.var(axis=0)
    
    with open(os.path.join(report_dir, "latent_statistics.txt"), "w") as f:
        f.write(f"Mean activation: {latent_np.mean():.6f}\n")
        f.write(f"Std activation: {latent_np.std():.6f}\n")
        f.write(f"Active dimensions (var > 1e-4): {(latent_var > 1e-4).sum()}/{latent_dim}\n")
        
    # PCA
    if latent_dim >= 2:
        pca = PCA(n_components=2)
        latent_pca = pca.fit_transform(latent_np)
        plt.figure(figsize=(8, 6))
        plt.scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.5)
        plt.title("PCA of Latent Space")
        plt.savefig(os.path.join(report_dir, "latent_pca.png"))
        plt.close()
        
    # Variance distribution
    plt.figure(figsize=(8, 6))
    plt.bar(range(latent_dim), sorted(latent_var, reverse=True))
    plt.title("Latent Dimension Variance Distribution")
    plt.xlabel("Dimension (sorted by variance)")
    plt.ylabel("Variance")
    plt.savefig(os.path.join(report_dir, "latent_variance.png"))
    plt.close()
    
    # Histogram of latent activations
    plt.figure(figsize=(8, 6))
    plt.hist(latent_np.flatten(), bins=50)
    plt.title("Histogram of Latent Activations")
    plt.savefig(os.path.join(report_dir, "latent_histogram.png"))
    plt.close()
    
    # Reconstruction plots
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(A_sample[0].cpu().numpy(), cmap='viridis')
    plt.title("Original Matrix")
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(A_rec[0].cpu().numpy(), cmap='viridis')
    plt.title("Reconstructed Matrix")
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(torch.abs(A_sample[0] - A_rec[0]).cpu().numpy(), cmap='magma')
    plt.title("Absolute Error")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "reconstruction.png"))
    plt.close()
    
    print(f"Research results saved to: {report_dir}")
    return model

if __name__ == "__main__":
    run_autoencoder_training()
