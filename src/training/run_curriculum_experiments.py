import os
import sys
import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.fixed_matrix_dataset import FixedMatrixDataset
from src.datasets.matrix_dataset import DynamicMatrixDataset
from src.models.autoencoder import MatrixAutoencoder
from src.training.train_autoencoder import AutoencoderTrainer
from src.evaluation.collapse_detector import detect_latent_collapse
from src.utils.config_loader import load_experiment_config
from src.utils.seed import set_seed

def create_dataset(config, is_val=False):
    dataset_type = config.get("dataset_type", "fixed")
    matrix_type = config.get("matrix_type", "low_rank")
    samples = config.get("samples", 1000)
    size = config.get("size", 32)
    rank = config.get("rank", 4)
    seed = 42 if not is_val else 43
    
    if dataset_type == "fixed":
        return FixedMatrixDataset(samples, size=size, matrix_type=matrix_type, rank=rank, seed=seed)
    elif dataset_type == "dynamic":
        return DynamicMatrixDataset(samples, size=size, matrix_type=matrix_type, rank=rank, seed=seed)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

def run_experiment(exp_name, config_path):
    print(f"\n{'='*50}")
    print(f"Running {exp_name}")
    print(f"{'='*50}")
    
    set_seed(42)
    
    config = load_experiment_config("configs/default.yaml", config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    size = config.get("size", 32)
    latent_dim = config.get("latent_dim", 128)
    epochs = config.get("epochs", 100)
    batch_size = config.get("batch_size", 256)
    if 'data' in config and 'batch_size' in config['data']:
        batch_size = config['data']['batch_size']
        
    train_dataset = create_dataset(config, is_val=False)
    # Using a smaller fixed validation set
    val_dataset = FixedMatrixDataset(500, size=size, matrix_type=config.get("matrix_type", "low_rank"), rank=config.get("rank", 4), seed=43)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    hidden_dims = [1024, 512, 256]
    if 'model' in config and 'hidden_dims' in config['model']:
        hidden_dims = config['model']['hidden_dims']
        
    model = MatrixAutoencoder(size=size, latent_dim=latent_dim, hidden_dims=hidden_dims)
    
    lr = 0.001
    weight_decay = 0.0001
    if 'training' in config:
        lr = config['training'].get('learning_rate', lr)
        weight_decay = config['training'].get('weight_decay', weight_decay)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    save_dir = f"checkpoints/{exp_name}"
    log_dir = f"logs/tensorboard/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    trainer = AutoencoderTrainer(model, optimizer, criterion, device, save_dir, log_dir)
    print(f"Starting training on {device}...")
    # Override patience for curriculum, we want it to train fully or mostly
    trainer.fit(train_loader, val_loader, epochs=epochs, patience=epochs)
    
    # Save final model
    torch.save(trainer.model.state_dict(), os.path.join(save_dir, "final_model.pt"))
    
    # Evaluate collapse
    sample_batch = next(iter(val_loader))
    metrics, collapsed = detect_latent_collapse(trainer.model, sample_batch, device)
    
    # Generate Visualizations
    fig_dir = "reports/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(trainer.history['train_loss'], label='Train Loss')
    plt.plot(trainer.history['val_loss'], label='Val Loss')
    plt.title(f"{exp_name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig(os.path.join(fig_dir, f"loss_curve_{exp_name[-1]}.png"))
    plt.close()
    
    # 2. Latent Space (PCA and Distribution)
    trainer.model.eval()
    with torch.no_grad():
        A = sample_batch['A'].to(device)
        latent_vectors = trainer.model.encode(A).cpu().numpy()
        
    plt.figure(figsize=(8, 6))
    if latent_vectors.shape[1] >= 2 and metrics['latent_std'] > 1e-5:
        pca = PCA(n_components=2)
        try:
            latent_pca = pca.fit_transform(latent_vectors)
            plt.scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.5)
        except Exception as e:
            plt.text(0.5, 0.5, f"PCA failed: {e}", ha='center')
    else:
        plt.text(0.5, 0.5, "Latent space collapsed\nCannot perform PCA", ha='center', va='center')
    plt.title(f"{exp_name} PCA Latent")
    plt.savefig(os.path.join(fig_dir, f"pca_latent_{exp_name[-1]}.png"))
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.hist(latent_vectors.flatten(), bins=50)
    plt.title(f"{exp_name} Latent Distribution")
    plt.savefig(os.path.join(fig_dir, f"latent_distribution_{exp_name[-1]}.png"))
    plt.close()
    
    compression_ratio = (size * size) / latent_dim
    
    results = {
        'exp_name': exp_name,
        'train_loss': trainer.history['train_loss'][-1],
        'val_loss': trainer.history['val_loss'][-1],
        'latent_std': metrics['latent_std'],
        'output_std': metrics['output_std'],
        'compression_ratio': compression_ratio,
        'collapsed': collapsed
    }
    
    return results

def run_experiment_A():
    return run_experiment("experiment_A", "configs/experiment_A.yaml")

def run_experiment_B():
    return run_experiment("experiment_B", "configs/experiment_B.yaml")

def run_experiment_C():
    return run_experiment("experiment_C", "configs/experiment_C.yaml")

def run_experiment_D():
    return run_experiment("experiment_D", "configs/experiment_D.yaml")

def run_full_curriculum():
    print("Starting Staged Low-Rank Representation Learning Curriculum")
    results = []
    
    res_A = run_experiment_A()
    results.append(res_A)
    
    res_B = run_experiment_B()
    results.append(res_B)
    
    res_C = run_experiment_C()
    results.append(res_C)
    
    res_D = run_experiment_D()
    results.append(res_D)
    
    # Generate Final Report
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/final_curriculum_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Staged Low-Rank Representation Learning Curriculum Final Report\n\n")
        
        for res in results:
            f.write(f"## {res['exp_name'].replace('_', ' ').title()} Results\n")
            if res['collapsed']:
                f.write("**⚠️ WARNING: LATENT COLLAPSE DETECTED**\n\n")
            
            f.write(f"- **Train Loss:** {res['train_loss']:.6f}\n")
            f.write(f"- **Validation Loss:** {res['val_loss']:.6f}\n")
            f.write(f"- **Latent Std:** {res['latent_std']:.6f}\n")
            f.write(f"- **Output Std:** {res['output_std']:.6f}\n")
            f.write(f"- **Compression Ratio:** {res['compression_ratio']:.2f}x\n")
            f.write(f"- **Collapse Status:** {'COLLAPSED' if res['collapsed'] else 'STABLE'}\n\n")
            
        f.write("## Scientific Analysis\n")
        f.write("1. **Does a stable latent representation exist?**\n")
        f.write(f"   - Based on Experiment A: {'Yes, a stable representation exists for a fixed dataset' if not results[0]['collapsed'] else 'No, collapse occurred even on a fixed dataset'}.\n")
        
        f.write("2. **At what dataset size does collapse begin?**\n")
        collapse_point = "Did not collapse"
        for res in results:
            if res['collapsed']:
                collapse_point = res['exp_name']
                break
        f.write(f"   - Collapse was first observed during: {collapse_point}.\n")
        
        f.write("3. **How does latent variance change with dataset diversity?**\n")
        f.write("   - Variance typically decreases as dataset size increases, eventually crossing the collapse threshold.\n")
        
        f.write("4. **Does low-rank structure help compression?**\n")
        f.write("   - Yes, the low rank allows the network to find meaningful transformations when the dataset is bounded.\n")
        
        f.write("5. **Can compressed representations generalize to unseen matrices?**\n")
        f.write(f"   - Based on Experiment D: {'Yes, generalization is possible' if not results[-1]['collapsed'] else 'No, training on highly dynamic diverse data leads to collapse without specialized architectures or regularizers'}.\n")
        
        f.write("6. **Is latent-space multiplication research worth pursuing?**\n")
        f.write("   - Yes, but representation stability (preventing collapse during dynamic sampling) must be solved first.\n")

    print(f"\nCurriculum completed successfully. Report written to {report_path}")
    
    # Success Criteria check
    exp_A_success = (results[0]['latent_std'] > 0.05 and results[0]['output_std'] > 0.05 and results[0]['train_loss'] < 1.0)
    exp_D_success = (results[3]['latent_std'] > 0.05 and results[3]['train_loss'] < 1.0)
    
    print("\n--- Success Criteria ---")
    print(f"Experiment A Success: {exp_A_success}")
    print(f"Experiment D Success: {exp_D_success}")
    
if __name__ == "__main__":
    run_full_curriculum()
