import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class FixedMatrixDataset(Dataset):
    """
    Fixed dataset that generates matrices once and stores them in memory.
    Useful for testing if representation learning works before attempting generalization.
    """
    def __init__(self, num_samples, size=32, matrix_type="low_rank", rank=4, seed=42):
        self.num_samples = num_samples
        self.size = size
        self.matrix_type = matrix_type
        self.rank = rank
        self.base_seed = seed
        
        self.A_data = []
        self.B_data = []
        self.C_data = []
        
        self._generate_data()
        
    def _generate_data(self):
        print(f"Generating {self.num_samples} fixed samples of type {self.matrix_type}...")
        gen = torch.Generator()
        gen.manual_seed(self.base_seed)
        
        for idx in tqdm(range(self.num_samples), desc="Generating Fixed Matrices", leave=False):
            if self.matrix_type == "random":
                A = torch.randn(self.size, self.size, generator=gen)
                B = torch.randn(self.size, self.size, generator=gen)
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
            
            self.A_data.append(A)
            self.B_data.append(B)
            self.C_data.append(C)
            
        self.A_data = torch.stack(self.A_data)
        self.B_data = torch.stack(self.B_data)
        self.C_data = torch.stack(self.C_data)
            
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return {
            'A': self.A_data[idx],
            'B': self.B_data[idx],
            'C': self.C_data[idx]
        }
