import torch

def detect_latent_collapse(model, sample_batch, device=None):
    """
    Detects if the latent representation has collapsed.
    Returns a dictionary of metrics and a boolean indicating collapse.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
        
    A = sample_batch['A'].to(device)
    
    with torch.no_grad():
        latent_vectors = model.encode(A)
        output = model.decode(latent_vectors)
        
    latent_np = latent_vectors.cpu().numpy()
    latent_std = latent_np.std().item()
    
    output_np = output.cpu().numpy()
    output_std = output_np.std().item()
    
    latent_var = latent_np.var(axis=0)
    activation_variance = latent_var.mean().item()
    
    # Simple proxy for entropy using histogram
    hist, _ = torch.histogram(latent_vectors.cpu().flatten(), bins=100)
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    latent_entropy = -(prob * torch.log(prob)).sum().item()
    
    collapsed = latent_std < 0.01 or output_std < 0.01
    
    metrics = {
        'latent_std': latent_std,
        'output_std': output_std,
        'latent_entropy': latent_entropy,
        'activation_variance': activation_variance,
        'collapsed': collapsed
    }
    
    return metrics, collapsed
