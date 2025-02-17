import torch
import triton
import triton.language as tl
import time
import numpy as np
import matplotlib.pyplot as plt

def fast_dequantize(weight, quant_state):
    """Baseline Unsloth fast_dequantize implementation (placeholder)"""
    return weight.float() * quant_state[:, None]

@triton.jit
def triton_dequantize_nf4_kernel(weight_ptr, quant_state_ptr, output_ptr, N: tl.constexpr):
    """Triton kernel for nf4 dequantization."""
    row_idx = tl.program_id(0)
    absmax = tl.load(quant_state_ptr + row_idx)
    row_start = row_idx * N // 8
    row_end = row_start + (N // 8)

    for i in range(row_start, row_end):
        packed = tl.load(weight_ptr + i)
        unpacked_vals = tl.zeros([8], dtype=tl.int32)  # Corrected initialization

        for j in range(8):
            unpacked_vals[j] = (packed >> (4 * j)) & 0xF  # Store unpacked values

            output_idx = (i - row_start) * 8 + j
            dequantized_val = absmax * (unpacked_vals[j] / 15.0)
            tl.store(output_ptr + row_idx * N + output_idx, dequantized_val)


def triton_dequantize_nf4(weight, quant_state):
    """Wrapper function to run Triton kernel."""
    output = torch.empty_like(weight, dtype=torch.float16)
    grid = (weight.shape[0],)
    triton_dequantize_nf4_kernel[grid](
        weight, quant_state, output, weight.shape[1]
    )
    return output

def benchmark_function(fn, weight, quant_state, runs=100):
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(runs):
        fn(weight, quant_state)
    torch.cuda.synchronize()
    return (time.time() - start_time) / runs

# Generate test tensors
torch.manual_seed(42)
N = 1024  # Feature size
rows = 512  # Number of rows
weight_packed = torch.randint(0, 2**32 - 1, (rows, N // 8), dtype=torch.int32, device="cuda")
quant_state = torch.rand(rows, dtype=torch.float16, device="cuda") * 2

# Benchmark
baseline_time = benchmark_function(fast_dequantize, weight_packed, quant_state, runs=100)
triton_time = benchmark_function(triton_dequantize_nf4, weight_packed, quant_state, runs=100)

# Compare performance
speedup = baseline_time / triton_time

# Display results
print(f"Baseline Execution Time: {baseline_time * 1000:.3f} ms")
print(f"Triton Execution Time: {triton_time * 1000:.3f} ms")
print(f"Speedup: {speedup:.2f}x")

# Plot results
labels = ['Baseline (fast_dequantize)', 'Triton Kernel']
times = [baseline_time * 1000, triton_time * 1000]
plt.bar(labels, times, color=['blue', 'green'])
plt.ylabel('Execution Time (ms)')
plt.title('Performance Comparison')
plt.show()
