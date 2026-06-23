import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.run_curriculum_experiments import run_experiment

def run_capacity_scaling():
    print("Starting Capacity Scaling Study")
    
    experiments = [
        {"name": "experiment_A", "config": "configs/experiment_A.yaml", "latent": 128, "samples": 1000},
        {"name": "experiment_A2", "config": "configs/experiment_A2.yaml", "latent": 256, "samples": 1000},
        {"name": "experiment_A3", "config": "configs/experiment_A3.yaml", "latent": 512, "samples": 1000},
        {"name": "experiment_B", "config": "configs/experiment_B.yaml", "latent": 128, "samples": 5000},
        {"name": "experiment_B2", "config": "configs/experiment_B2.yaml", "latent": 256, "samples": 5000},
        {"name": "experiment_B3", "config": "configs/experiment_B3.yaml", "latent": 512, "samples": 5000},
    ]
    
    results = []
    for exp in experiments:
        res = run_experiment(exp["name"], exp["config"])
        res["latent_dim"] = exp["latent"]
        res["samples"] = exp["samples"]
        results.append(res)
        
    # Generate Plots
    fig_dir = "reports/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    group_1000 = [r for r in results if r["samples"] == 1000]
    group_5000 = [r for r in results if r["samples"] == 5000]
    
    latent_dims_1000 = [r["latent_dim"] for r in group_1000]
    latent_dims_5000 = [r["latent_dim"] for r in group_5000]
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(latent_dims_1000, [r["val_loss"] for r in group_1000], marker='o', label='1000 Samples (A-series)')
    plt.plot(latent_dims_5000, [r["val_loss"] for r in group_5000], marker='s', label='5000 Samples (B-series)')
    plt.title("Capacity vs Validation Loss")
    plt.xlabel("Latent Dimension")
    plt.ylabel("Validation MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "capacity_loss_curves.png"))
    plt.close()
    
    # Latent Std plot
    plt.figure(figsize=(10, 6))
    plt.plot(latent_dims_1000, [r["latent_std"] for r in group_1000], marker='o', label='1000 Samples')
    plt.plot(latent_dims_5000, [r["latent_std"] for r in group_5000], marker='s', label='5000 Samples')
    plt.title("Capacity vs Latent Standard Deviation")
    plt.xlabel("Latent Dimension")
    plt.ylabel("Latent Std")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "capacity_latent_std.png"))
    plt.close()
    
    # Output Std plot
    plt.figure(figsize=(10, 6))
    plt.plot(latent_dims_1000, [r["output_std"] for r in group_1000], marker='o', label='1000 Samples')
    plt.plot(latent_dims_5000, [r["output_std"] for r in group_5000], marker='s', label='5000 Samples')
    plt.title("Capacity vs Output Standard Deviation")
    plt.xlabel("Latent Dimension")
    plt.ylabel("Output Std")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "capacity_output_std.png"))
    plt.close()
    
    # Generate Report
    report_path = "reports/capacity_scaling_report.md"
    with open(report_path, "w") as f:
        f.write("# Capacity Scaling Study Report\n\n")
        f.write("This report analyzes how increasing the latent dimension capacity affects the representation learning of low-rank matrices.\n\n")
        
        f.write("## Experiment Results\n\n")
        f.write("| Experiment | Samples | Latent Dim | Val Loss | Latent Std | Output Std | Collapsed |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for res in results:
            f.write(f"| {res['exp_name']} | {res['samples']} | {res['latent_dim']} | {res['val_loss']:.6f} | {res['latent_std']:.6f} | {res['output_std']:.6f} | {res['collapsed']} |\n")
            
        f.write("\n## Analysis\n")
        f.write("By observing the capacity vs loss and standard deviations plots (saved in `reports/figures/`), we can determine whether larger latent capacities help prevent representation collapse or if they exacerbate the issue by giving the network too many degrees of freedom leading to zero-activations.\n")

    print(f"\nCapacity Scaling Study completed successfully. Report written to {report_path}")

if __name__ == "__main__":
    run_capacity_scaling()
