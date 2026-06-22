import torch
import numpy as np

def compute_metrics(true_matrix, pred_matrix):
    """Computes MSE, RMSE, MAE, and Relative Frobenius Error."""
    if isinstance(true_matrix, np.ndarray):
        true_matrix = torch.tensor(true_matrix)
    if isinstance(pred_matrix, np.ndarray):
        pred_matrix = torch.tensor(pred_matrix)
        
    diff = true_matrix - pred_matrix
    mse = torch.mean(diff ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(diff)).item()
    
    true_norm = torch.linalg.matrix_norm(true_matrix, ord='fro')
    diff_norm = torch.linalg.matrix_norm(diff, ord='fro')
    
    # Avoid division by zero
    rel_error = (diff_norm / (true_norm + 1e-8)).mean().item()
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "relative_error": rel_error
    }

def compression_ratio(original_size, latent_dim):
    return (original_size * original_size) / latent_dim
