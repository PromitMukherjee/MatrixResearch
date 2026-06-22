def compute_memory_traffic(matrix_size, latent_dim, batch_size=1):
    """
    Computes theoretical memory traffic (bytes) and FLOPs.
    Assumes float32 (4 bytes per element).
    """
    bytes_per_element = 4
    N = matrix_size
    
    # Standard GEMM (C = A @ B)
    # Read A (N*N), Read B (N*N), Write C (N*N)
    std_memory_bytes = (3 * N * N) * bytes_per_element * batch_size
    # FLOPs: N^2 dot products of size N -> 2 * N^3 operations
    std_flops = 2 * (N ** 3) * batch_size
    std_arithmetic_intensity = std_flops / std_memory_bytes if std_memory_bytes > 0 else 0
    
    # Latent Multiplier (C = F(Z_A, B))
    # Read Z_A (latent_dim), Read B (N*N), Write C (N*N)
    latent_mult_memory_bytes = (latent_dim + 2 * N * N) * bytes_per_element * batch_size
    
    # Latent Operator (Z_C = G(Z_A, Z_B))
    # Read Z_A (latent_dim), Read Z_B (latent_dim), Write Z_C (latent_dim)
    latent_op_memory_bytes = (3 * latent_dim) * bytes_per_element * batch_size
    
    compression_ratio = (N * N) / latent_dim if latent_dim > 0 else 0
    
    return {
        "standard_gemm": {
            "memory_bytes": std_memory_bytes,
            "flops": std_flops,
            "arithmetic_intensity": std_arithmetic_intensity
        },
        "latent_multiplier": {
            "memory_bytes": latent_mult_memory_bytes,
        },
        "latent_operator": {
            "memory_bytes": latent_op_memory_bytes,
        },
        "compression_ratio": compression_ratio
    }
