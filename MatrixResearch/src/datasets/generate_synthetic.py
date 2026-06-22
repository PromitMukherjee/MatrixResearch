import argparse
import torch
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.matrix_dataset import DynamicMatrixDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate static synthetic evaluation datasets.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--size", type=int, default=32, help="Matrix dimension size")
    parser.add_argument("--type", type=str, default="random", choices=["random", "sparse", "low_rank"])
    parser.add_argument("--output", type=str, default="data/synthetic/test_set.pt")
    args = parser.parse_args()
    
    print(f"Generating {args.samples} static {args.size}x{args.size} {args.type} matrices...")
    
    dataset = DynamicMatrixDataset(num_samples=args.samples, size=args.size, matrix_type=args.type)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_A, all_B, all_C = [], [], []
    for batch in tqdm(loader):
        all_A.append(batch['A'])
        all_B.append(batch['B'])
        all_C.append(batch['C'])
        
    A = torch.cat(all_A, dim=0)
    B = torch.cat(all_B, dim=0)
    C = torch.cat(all_C, dim=0)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    torch.save({
        'A': A,
        'B': B,
        'C': C,
        'size': args.size,
        'type': args.type
    }, args.output)
    
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
