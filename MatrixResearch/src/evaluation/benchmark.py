import torch
import time
from sklearn.decomposition import TruncatedSVD
import numpy as np

def benchmark_standard_gemm(A, B):
    """Measures latency of standard PyTorch matmul."""
    start = time.perf_counter()
    C = A @ B
    end = time.perf_counter()
    return C, (end - start) * 1000  # ms

def benchmark_svd_compression(A, B, n_components):
    """Compresses A using SVD, then multiplies."""
    A_np = A.cpu().numpy() if isinstance(A, torch.Tensor) else A
    B_np = B.cpu().numpy() if isinstance(B, torch.Tensor) else B
    
    start = time.perf_counter()
    svd = TruncatedSVD(n_components=n_components)
    A_reduced = svd.fit_transform(A_np)  # U * Sigma
    V_T = svd.components_                # V^T
    
    # More efficient: (U * Sigma) @ (V^T @ B)
    C_approx = A_reduced @ (V_T @ B_np)
    end = time.perf_counter()
    
    return torch.tensor(C_approx), (end - start) * 1000

def benchmark_int8_quantization(A, B):
    """Simulates INT8 quantization multiplier."""
    # Simple dynamic quantization
    start = time.perf_counter()
    # Find max absolute for scaling
    scale_A = A.abs().max() / 127.0
    scale_B = B.abs().max() / 127.0
    
    A_int8 = torch.round(A / scale_A).to(torch.int8)
    B_int8 = torch.round(B / scale_B).to(torch.int8)
    
    # Matmul in int32
    C_int32 = torch.matmul(A_int8.to(torch.int32), B_int8.to(torch.int32))
    
    # Dequantize
    C_approx = C_int32.to(torch.float32) * (scale_A * scale_B)
    end = time.perf_counter()
    
    return C_approx, (end - start) * 1000
