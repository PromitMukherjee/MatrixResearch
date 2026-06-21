import matplotlib.pyplot as plt
import os

def plot_compression_frontier(latent_dims, accuracies, compression_ratios, save_path=None):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Latent Dimension')
    ax1.set_ylabel('Accuracy (1 - Rel Error)', color=color)
    ax1.plot(latent_dims, accuracies, marker='o', color=color, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log', base=2)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Compression Ratio', color=color)
    ax2.plot(latent_dims, compression_ratios, marker='s', linestyle='--', color=color, label='Compression Ratio')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log', base=10)
    
    plt.title('Compression Frontier: Accuracy vs Compression Ratio')
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()
