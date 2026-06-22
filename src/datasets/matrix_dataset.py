import torch
from torch.utils.data import Dataset

class DynamicMatrixDataset(Dataset):
    """
    Dynamically generates pairs of matrices (A, B) and their product C = A @ B.
    This avoids storing massive amounts of data on disk.
    """
    def __init__(self, num_samples, size=32, matrix_type="random", sparsity=0.8, rank=4, seed=42):
        self.num_samples = num_samples
        self.size = size
        self.matrix_type = matrix_type
        self.sparsity = sparsity
        self.rank = rank
        self.base_seed = seed
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        gen = torch.Generator()
        gen.manual_seed(self.base_seed + idx)
        
        if self.matrix_type == "random":
            A = torch.randn(self.size, self.size, generator=gen)
            B = torch.randn(self.size, self.size, generator=gen)
        elif self.matrix_type == "sparse":
            A = torch.randn(self.size, self.size, generator=gen)
            B = torch.randn(self.size, self.size, generator=gen)
            A[torch.rand(self.size, self.size, generator=gen) < self.sparsity] = 0.0
            B[torch.rand(self.size, self.size, generator=gen) < self.sparsity] = 0.0
        elif self.matrix_type == "low_rank":
            A_u = torch.randn(self.size, self.rank, generator=gen)
            A_v = torch.randn(self.rank, self.size, generator=gen)
            A = A_u @ A_v
            
            if A.std() > 0:
                A = A / A.std()
            
            B_u = torch.randn(self.size, self.rank, generator=gen)
            B_v = torch.randn(self.rank, self.size, generator=gen)
            B = B_u @ B_v
            
            if B.std() > 0:
                B = B / B.std()
        else:
            raise ValueError(f"Unknown matrix type: {self.matrix_type}")
            
        C = A @ B
        
        return {
            'A': A,
            'B': B,
            'C': C
        }
