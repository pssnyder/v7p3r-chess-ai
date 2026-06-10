"""
Quick parameter count calculator for architecture comparison
"""

def count_parameters(input_dim, shared_dims, policy_hidden, value_hidden):
    """Calculate total parameters in V7P3R AI architecture"""
    params = 0
    
    # Shared embedding layers
    prev_dim = input_dim
    for dim in shared_dims:
        # ResidualBlock: linear + bn + projection (if needed)
        params += prev_dim * dim + dim  # linear weights + bias
        params += dim * 2  # batch norm (gamma + beta)
        if prev_dim != dim:
            params += prev_dim * dim + dim  # projection layer
        prev_dim = dim
    
    # Policy head
    # Hidden layer
    params += prev_dim * policy_hidden + policy_hidden
    # Output layer (6 classes)
    params += policy_hidden * 6 + 6
    
    # Value head  
    # Hidden layer
    params += prev_dim * value_hidden + value_hidden
    # Output layer (1 value)
    params += value_hidden * 1 + 1
    
    return params

# v5.1 (current)
v51_params = count_parameters(325, [256, 256, 128, 64], 64, 32)
print("=" * 60)
print("V7P3R AI v5.1 (Current)")
print("=" * 60)
print(f"Architecture: 325 → 256 → 256 → 128 → 64")
print(f"Policy Head: 64 hidden → 6 classes")
print(f"Value Head: 32 hidden → 1 value")
print(f"Total Parameters: {v51_params:,}")
print(f"Memory (float32): ~{v51_params * 4 / 1024 / 1024:.1f} MB")
print()

# v5.2 (proposed wide)
v52_params = count_parameters(325, [512, 512, 256, 128], 128, 64)
print("=" * 60)
print("V7P3R AI v5.2 (Wide Architecture)")
print("=" * 60)
print(f"Architecture: 325 → 512 → 512 → 256 → 128")
print(f"Policy Head: 128 hidden → 6 classes")
print(f"Value Head: 64 hidden → 1 value")
print(f"Total Parameters: {v52_params:,}")
print(f"Memory (float32): ~{v52_params * 4 / 1024 / 1024:.1f} MB")
print()

print("=" * 60)
print("Comparison")
print("=" * 60)
print(f"Parameter increase: {v52_params / v51_params:.2f}x")
print(f"Additional parameters: +{v52_params - v51_params:,}")
print(f"Memory increase: +{(v52_params - v51_params) * 4 / 1024 / 1024:.1f} MB")
print()

print("Expected Impact:")
print("  - Capacity for more complex patterns: ✅ HIGH")
print("  - Training time increase: ~2x (13min → 25min)")
print("  - CPU-friendly: ✅ Still <2GB")
print("  - Target accuracy: 50-53% (+5-8pp)")
print()
